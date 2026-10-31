"""CPU-only immutable run policy inputs, isolation, CLI and cloning."""

import copy
from pathlib import Path

import pytest
import yaml

from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.config import Parallel, assert_frozen, resolve
from core.io import hash_json, read_json, sha256
from core.parallel_policy import apply_run_parallel
from tests.conftest import task


@pytest.fixture
def parallel_cfg(cfg, monkeypatch):
    value = copy.deepcopy(cfg)
    manifest = Path(cfg["project_root"]) / "fixture-estimator.json"
    manifest.write_text("{}")
    value["parallel"] = Parallel(enabled=True, mode="parallel_consensus", branch_count=4,
                                 estimator_ref=str(manifest), max_inflight_branches=4,
                                 save_sites=[3], fallback_policy="stop").model_dump()
    value["capture"].update(enabled=True, anchors=["think_close"], layers=[3])
    value["capture"]["export"]["enabled"] = True
    value["serving"].update(management="managed", max_num_seqs=8)
    value["config_hash"] = hash_json({k: v for k, v in value.items() if k != "config_hash"})
    monkeypatch.setattr("runtime.serving.parallel.artifact_contract", lambda *a, **kw: {"sites": [3]})
    Catalog(cfg["project_root"]).register_tasks([task("policy-task")])
    return value


def policy_file(cfg, name="policy", rule="shannon_exp_keep_v1", representative="min_step_tokens", **changes):
    payload = {"schema_version": 1, "mode": "parallel_enthophy", "enabled": True,
               "fallback_policy": "stop", "consensus": {"representative_policy": representative},
               "entropy": {"retention_rule": rule}, **changes}
    path = Path(cfg["project_root"]) / (name + ".yaml")
    path.write_text(yaml.safe_dump(payload))
    return path


def test_run_inputs_are_isolated_and_frozen_without_changing_root(parallel_cfg):
    cfg = parallel_cfg
    root = Path(cfg["project_root"])
    root_hash = sha256(root / "config.yaml")
    original_cfg = copy.deepcopy(cfg)
    paths = [policy_file(cfg, "a"), policy_file(cfg, "b", rule="effective_count_v1", representative="estimator_score")]
    for name, path in zip(("policy-a", "policy-b"), paths):
        run = create_run(cfg, name, operation="rollout", task_ids=["policy-task"], parallel_config=path)
        loaded = load_plan(root, name)
        assert loaded["config"]["parallel"]["mode"] == "parallel_enthophy"
        assert (run / "parallel-config.yaml").read_bytes() == path.read_bytes()
        assert yaml.safe_load((run / "config.yaml").read_text())["parallel"]["entropy"] == (
            loaded["config"]["parallel"]["entropy"])
        assert_frozen(loaded["config"])
    paths[0].write_text("edited externally")
    paths[1].unlink()
    a, b = load_plan(root, "policy-a"), load_plan(root, "policy-b")
    assert a["config"]["parallel"]["consensus"]["representative_policy"] == "min_step_tokens"
    assert b["config"]["parallel"]["consensus"]["representative_policy"] == "estimator_score"
    assert a["config"]["parallel"]["entropy"] != b["config"]["parallel"]["entropy"]
    assert sha256(root / "config.yaml") == root_hash
    assert cfg == original_cfg


def test_clone_can_inherit_or_change_policy_and_switch_to_fixed(parallel_cfg):
    cfg = parallel_cfg
    first = create_run(cfg, "policy-source", operation="rollout", task_ids=["policy-task"],
                       parallel_config=policy_file(cfg))
    protected = {name: sha256(first / name) for name in ("config.yaml", "resolved-config.json", "parallel-config.yaml")}
    inherited = create_run(cfg, "policy-inherited", operation="rollout", from_run=first.name)
    assert (inherited / "parallel-config.yaml").read_bytes() == (first / "parallel-config.yaml").read_bytes()
    assert load_plan(cfg["project_root"], inherited.name)["config"]["parallel_policy"]["inherited_from_run"] == first.name
    changed = create_run(cfg, "policy-changed", operation="rollout", from_run=first.name,
                         parallel_config=policy_file(cfg, "c", rule="baseline_floor_v1", representative="estimator_score"))
    assert load_plan(cfg["project_root"], changed.name)["config"]["parallel"]["entropy"]["retention_rule"] == "baseline_floor_v1"
    fixed_input = policy_file(cfg, "fixed", mode="parallel_consensus")
    raw = yaml.safe_load(fixed_input.read_text())
    raw.pop("entropy")
    fixed_input.write_text(yaml.safe_dump(raw))
    fixed = create_run(cfg, "policy-fixed", operation="rollout", from_run=first.name, parallel_config=fixed_input)
    assert "entropy" not in load_plan(cfg["project_root"], fixed.name)["config"]["parallel"]
    assert all(sha256(first / name) == digest for name, digest in protected.items())


def test_cli_accepts_run_local_policy_without_resolving_it_as_root(parallel_cfg, capsys):
    from core.cli import main

    cfg = parallel_cfg
    source = create_run(cfg, "cli-source", operation="rollout", task_ids=["policy-task"])
    path = policy_file(cfg)
    main(["--config", str(Path(cfg["project_root"]) / "config.yaml"), "run", "create", "cli-policy",
          "--operation", "rollout", "--from-run", source.name, "--parallel-config", str(path)])
    assert load_plan(cfg["project_root"], "cli-policy")["config"]["parallel"]["mode"] == "parallel_enthophy"
    assert "cli-policy" in capsys.readouterr().out


@pytest.mark.parametrize("changes", [
    {"entropy": {"retention_rule": "wrong"}}, {"entropy": None},
    {"mode": "parallel_consensus"}, {"unknown": 3}, {"schema_version": 2},
    {"consensus": {"representative_policy": "random"}},
    {"consensus": {"min_retained_branches": 5}}, {"fallback_policy": "first_valid"},
    {"consensus": {"retention_ratio": float("nan")}},
    {"consensus": {"representative_policy": None}}, {"enabled": False},
])
def test_invalid_run_inputs_fail_before_creating_run(parallel_cfg, changes):
    cfg = parallel_cfg
    path = policy_file(cfg, **changes)
    with pytest.raises(ValueError):
        create_run(cfg, "invalid-policy", operation="rollout", task_ids=["policy-task"], parallel_config=path)
    assert not (Path(cfg["project_root"]) / "runs/invalid-policy").exists()


def test_missing_rule_duplicate_keys_and_override_conflicts_are_rejected(parallel_cfg):
    cfg = parallel_cfg
    path = policy_file(cfg)
    raw = yaml.safe_load(path.read_text())
    raw.pop("entropy")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="explicit entropy"):
        apply_run_parallel(cfg, path)
    path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ValueError, match="Duplicate YAML"):
        apply_run_parallel(cfg, path)
    path = policy_file(cfg)
    with pytest.raises(ValueError, match="Conflicting"):
        apply_run_parallel(cfg, path, overrides=["parallel.mode=parallel_consensus"])
    amended = copy.deepcopy(cfg)
    amended["parallel"]["consensus"]["retention_ratio"] = .75
    configured, _ = apply_run_parallel(amended, path, overrides=["parallel.consensus.retention_ratio=.75"])
    assert configured["parallel"]["consensus"]["retention_ratio"] == .75


def test_legacy_snapshot_does_not_gain_new_fields_and_root_rejects_policy_keys(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("legacy")])
    run = create_run(cfg, "legacy-policy", operation="rollout", task_ids=["legacy"])
    loaded = load_plan(cfg["project_root"], run.name)
    assert "parallel_policy" not in loaded["config"]
    assert not (run / "parallel-config.yaml").exists()
    root_config = Path(cfg["project_root"]) / "config.yaml"
    raw = yaml.safe_load(root_config.read_text())
    raw["parallel"] = {"consensus": {"representative_policy": "min_step_tokens"}}
    root_config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="representative_policy"):
        resolve(root_config)


@pytest.mark.parametrize("which", ["input", "effective"])
def test_run_policy_tampering_is_rejected(parallel_cfg, which):
    cfg = parallel_cfg
    run = create_run(cfg, "tamper-policy", operation="rollout", task_ids=["policy-task"],
                     parallel_config=policy_file(cfg))
    if which == "input":
        (run / "parallel-config.yaml").write_text("tampered")
    else:
        frozen = read_json(run / "resolved-config.json")
        frozen["parallel"]["consensus"]["representative_policy"] = "estimator_score"
        (run / "resolved-config.json").write_text(__import__("json").dumps(frozen))
    with pytest.raises(ValueError, match="modified"):
        load_plan(cfg["project_root"], run.name)


def test_new_run_policy_is_compatible_with_offline_resource_revisions(parallel_cfg):
    from application.execution.capture_budget_revision import revise_concurrency, revise_serving_memory

    cfg = parallel_cfg
    run = create_run(cfg, "revisable-policy", operation="rollout", task_ids=["policy-task"],
                     parallel_config=policy_file(cfg))
    old = load_plan(cfg["project_root"], run.name)["config"]
    sidecar_hash = sha256(run / "parallel-config.yaml")
    revise_serving_memory(cfg["project_root"], run.name, revision_id="memory",
                         gpu_memory_utilization=.35, reason="CPU fixture resource revision")
    revise_concurrency(cfg["project_root"], run.name, revision_id="capacity",
                       reason="CPU fixture resource revision")
    revised = load_plan(cfg["project_root"], run.name)["config"]
    assert revised["parallel_policy"] == old["parallel_policy"]
    assert revised["parallel"]["control_request_seconds"] == 600
    assert revised["parallel"]["decision_ttl_seconds"] == 64800
    assert revised["parallel"]["consensus"]["representative_policy"] == "min_step_tokens"
    assert sha256(run / "parallel-config.yaml") == sidecar_hash


@pytest.mark.parametrize("tamper", [None, "binding", "rule"])
def test_artifact_verifier_binds_new_decisions_to_their_run_policy(parallel_cfg, monkeypatch, tamper):
    import json

    from application.execution import artifact
    from tests.test_parallel_entropy import make_evidence

    cfg = parallel_cfg
    run = create_run(cfg, "artifact-policy", operation="rollout", task_ids=["policy-task"],
                     parallel_config=policy_file(cfg))
    frozen = load_plan(cfg["project_root"], run.name)["config"]
    summary = make_evidence()
    summary["policy_binding"] = {key: frozen["parallel_policy"][key]
                                 for key in ("input_sha256", "effective_sha256")}
    summary["run_config_hash"] = frozen["config_hash"]
    if tamper == "binding":
        summary["policy_binding"]["input_sha256"] = "wrong"
    elif tamper == "rule":
        summary["consensus_config"]["retention_rule"] = "effective_count_v1"
    trajectory = run / "rollout/trajectories/test.json"
    trajectory.write_text(json.dumps({"messages": [{"extra": {"dycon_parallel": summary}}]}))

    class Evidence:
        def __init__(self, *args):
            pass

        def verify(self):
            return True

        def read(self):
            return {"rollouts": {"trajectory": {"run_id": run.name,
                                               "trajectory_path": str(trajectory.relative_to(cfg["project_root"]))}}}

    monkeypatch.setattr(artifact, "RunEvidence", Evidence)
    if tamper:
        with pytest.raises(ValueError):
            artifact.verify(cfg["project_root"], run.name)
    else:
        assert artifact.verify(cfg["project_root"], run.name)["artifacts_checked"] == 1


def test_swe_policy_clone_preserves_unused_legacy_image_builder_metadata(parallel_cfg):
    cfg = copy.deepcopy(parallel_cfg)
    cfg['skillsbench_images'] = {'builder_image': 'legacy-frozen-image'}
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    source = create_run(cfg, 'legacy-builder-source', operation='rollout', task_ids=['policy-task'])
    old = sha256(source / 'resolved-config.json')
    cloned = create_run(cfg, 'legacy-builder-policy', operation='rollout', from_run=source.name,
                        parallel_config=policy_file(cfg, representative='estimator_score', rule='effective_count_v1'))
    loaded = load_plan(cfg['project_root'], cloned.name)['config']
    assert loaded['skillsbench_images'] == cfg['skillsbench_images']
    assert sha256(source / 'resolved-config.json') == old
