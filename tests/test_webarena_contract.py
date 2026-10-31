"""Verified identities, answer isolation and immutable web evidence contracts."""
import copy
import json
from pathlib import Path

import pytest

from application.execution.evidence import RunEvidence
from application.execution.subject import artifact_kinds, verify_subject
from catalog import webarena
from catalog.store import Catalog, Task
from core.backends import capabilities
from core.io import hash_json, read_json, sha256, write_json
from rollout.artifacts import import_rollout


@pytest.fixture
def official(cfg, monkeypatch):
    """Complete synthetic data exercises import without embedding official answers."""
    root = Path(cfg["project_root"])
    source = root / "source/assets/dataset"
    tasks = [{"task_id": index, "revision": 1, "intent_template_id": index // 4,
              "sites": ["reddit"], "start_urls": ["__REDDIT__"], "intent": f"Open article {index}",
              "intent_template": "Open article {{id}}", "instantiation_dict": {"id": index},
              "eval": [{"evaluator": "AgentResponseEvaluator", "results_schema": {"type": "array"},
                        "expected": {"task_type": "NAVIGATE", "status": "SUCCESS",
                                     "retrieved_data": [f"PRIVATE_REFERENCE_{index}"]}}]}
             for index in range(812)]
    write_json(source / "webarena-verified.json", tasks)
    write_json(source / "subsets/webarena-verified-hard.json", {"task_ids": list(range(258))})
    monkeypatch.setattr(webarena, "DATASET_SHA256", sha256(source / "webarena-verified.json"))
    monkeypatch.setattr(webarena, "HARD_SHA256", sha256(source / "subsets/webarena-verified-hard.json"))
    return root / "source"


def imported(cfg, official, ids=(108,)):
    result = webarena.import_tasks(cfg, official, task_ids=ids)
    snapshot = read_json(Path(cfg["project_root"]) / result["snapshot_ref"])
    return result, snapshot, next(iter(snapshot["tasks"].values()))


def test_import_full_hard_and_explicit_subset_preserves_source_identity(cfg, official):
    full = webarena.import_tasks(cfg, official)
    hard = webarena.import_tasks(cfg, official, subset="hard")
    small, snapshot, task = imported(cfg, official)
    assert (full["task_count"], hard["task_count"], small["task_count"]) == (812, 258, 1)
    assert task["instance_id"] == "wav-108" and task["revision"] == webarena.REVISION
    assert task["instance"]["source_task_id"] == 108 and task["instance"]["task_revision"] == 1
    assert "repo" not in task and "base_commit" not in task
    assert snapshot["evaluator"]["task_count"] == 812


def test_reference_answers_are_private_and_not_in_catalog_snapshot_or_profile(cfg, official):
    result, snapshot, _ = imported(cfg, official)
    root = Path(cfg["project_root"])
    for value in (Catalog(root).read(), snapshot, read_json(root / result["profile_ref"])):
        text = json.dumps(value)
        assert "PRIVATE_REFERENCE_" not in text
        assert '"expected"' not in text and '"eval"' not in text
    private = root / result["evaluator"]["path"]
    assert "PRIVATE_REFERENCE_108" in private.read_text()
    assert private.stat().st_mode & 0o777 == 0o600
    assert private.parent.stat().st_mode & 0o777 == 0o700


def test_import_is_idempotent_but_changed_source_and_private_evaluator_are_rejected(cfg, official):
    result, _, _ = imported(cfg, official)
    assert webarena.import_tasks(cfg, official, task_ids=["wav-108"]) == result
    private = Path(cfg["project_root"]) / result["evaluator"]["path"]
    private.write_text("[]")
    with pytest.raises(ValueError, match="Private.*changed"):
        webarena.import_tasks(cfg, official, task_ids=[108])
    dataset = official / "assets/dataset/webarena-verified.json"
    dataset.write_text("[]")
    with pytest.raises(ValueError, match="pinned official"):
        webarena.import_tasks(cfg, official, task_ids=[108])


@pytest.mark.parametrize("ids,subset", [([108, "wav-108"], "full"), ([811], "hard"), ([True], "full")])
def test_import_rejects_duplicate_unknown_or_out_of_cohort_ids(cfg, official, ids, subset):
    with pytest.raises(ValueError):
        webarena.import_tasks(cfg, official, task_ids=ids, subset=subset)


def test_public_payload_cannot_smuggle_evaluation_or_repository_placeholders(cfg, official):
    _, _, task = imported(cfg, official)
    for change in (lambda t: t["instance"].update(expected="private"), lambda t: t.update(repo="fake"),
                   lambda t: t["instance"].update(source_task_id=109)):
        bad = copy.deepcopy(task)
        change(bad)
        with pytest.raises(ValueError):
            Task.model_validate(bad)


def test_catalog_family_selection_never_treats_web_task_as_swe(cfg, official):
    from conftest import task as swe_task
    _, _, web_task = imported(cfg, official)
    root = Path(cfg["project_root"])
    Catalog(root).register_tasks([swe_task()])
    web = Catalog(root).select_tasks(benchmark="webarena_verified")
    assert [t["instance_id"] for t in web["tasks"].values()] == [web_task["instance_id"]]
    swe = Catalog(root).select_tasks(benchmark="swebench")
    assert all(t["benchmark"] != "webarena_verified" for t in swe["tasks"].values())


def test_web_snapshot_selection_does_not_expand_to_other_imports(cfg, official):
    from application.execution.run_plan import _selection_from_tasks
    result, _, _ = imported(cfg, official)
    imported(cfg, official, ids=(109,))
    cfg = copy.deepcopy(cfg)
    cfg["benchmark"] = {"backend": "webarena_verified", "snapshot_ref": result["snapshot_ref"]}
    selected = _selection_from_tasks(Catalog(cfg["project_root"]), cfg=cfg)
    assert [t["instance_id"] for t in selected["tasks"].values()] == ["wav-108"]
    with pytest.raises(ValueError, match="outside.*snapshot"):
        _selection_from_tasks(Catalog(cfg["project_root"]), cfg=cfg, task_ids=["wav-109"])


def test_execution_context_freezes_runtime_profile_and_evaluator_bytes(cfg, official, monkeypatch):
    from benchmark_evaluation import webarena as webarena_eval
    from core.backends import execution_context
    monkeypatch.setattr(webarena_eval, "runtime_reference", lambda cfg: {
        "path": "runtime/evaluator/manifest.json", "sha256": "f" * 64, "revision": webarena.REVISION})
    result, snapshot, task = imported(cfg, official)
    root = Path(cfg["project_root"])
    ref = {"store": "fixture", "manifest": "image.json", "manifest_sha256": "0" * 64,
           "image_id": "sha256:" + "1" * 64}
    pool = {"schema_version": 1, "benchmark": "webarena_verified", "revision": webarena.REVISION,
            "sites": {"reddit": {"image_ref": ref, "port": 80, "base_url": "http://reddit", "health_path": "/"}}}
    write_json(root / "runtime/browser.json", ref)
    write_json(root / "runtime/pool.json", pool)
    cfg = copy.deepcopy(cfg)
    cfg["benchmark"] = {"backend": "webarena_verified", "snapshot_ref": result["snapshot_ref"]}
    cfg["agent"].update(backend="minisweagent", profile_ref=result["profile_ref"])
    cfg["webarena"] = {"browser_image_ref": "runtime/browser.json", "environment_pool_ref": "runtime/pool.json",
                       "max_browser_actions": 30, "evaluator_python": ".venvs/webarena-eval/bin/python",
                       "evaluator_timeout_seconds": 120}
    context = execution_context(cfg, snapshot["tasks"])
    assert context["webarena_runtime"]["environment_pool"] == pool
    assert context["evaluator"]["sha256"] == webarena.DATASET_SHA256
    assert "{{ initial_observation }}" in context["profile"]["instance_template"]
    assert "10000" not in context["profile"]["observation_template"]
    assert "head" not in context["profile"]["format_error_template"]
    write_json(root / "runtime/pool.json", {**pool, "sites": {
        "shopping": {"image_ref": ref, "port": 80, "base_url": "http://shopping", "health_path": "/"}}})
    assert context["webarena_runtime"]["environment_pool"] == pool
    with pytest.raises(ValueError, match="sites missing"):
        execution_context(cfg, snapshot["tasks"])
    write_json(root / "runtime/pool.json", pool)
    (root / result["evaluator"]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="dataset is missing or changed"):
        execution_context(cfg, snapshot["tasks"])


def web_artifacts(cfg, task, *, complete=True, har_flushed=True):
    root = Path(cfg["project_root"])
    target = root / "runs/web/rollout/tasks/wav-108/attempt-a0"
    target.mkdir(parents=True, exist_ok=True)
    raw = target / "trajectory.json"
    write_json(raw, {"instance_id": task["instance_id"], "info": {
        "exit_status": "Submitted" if complete else "LimitsExceeded", "submission": ""}, "messages": [
            {"role": "system", "content": "Browser"}, {"role": "user", "content": "Task and initial AX"},
            {"role": "assistant", "content": "click"}, {"role": "tool", "content": "AX update"},
            {"role": "assistant", "content": "stop"}]})
    write_json(target / "answer.json", {"task_type": "NAVIGATE", "status": "SUCCESS", "retrieved_data": None,
                                       "error_details": None})
    write_json(target / "network.har", {"log": {"version": "1.2", "entries": []}})
    (target / "actions.jsonl").write_text('{"action":"click [1]"}\n{"action":"stop"}\n')
    outputs = []
    names = ["answer.json", "network.har", "actions.jsonl"] if complete else ["actions.jsonl"]
    for name in names:
        path = target / name
        outputs.append({"path": name, "sha256": sha256(path), "size": path.stat().st_size})
    subject = target / "subject.json"
    write_json(subject, {"schema_version": 1, "kind": "web_trace", "outputs": outputs,
        "binding": {"run_id": "web", "attempt_id": "a0", "task_key": hash_json({
            k: task[k] for k in ("benchmark", "revision", "instance_id")})},
        "complete": complete, "har_flushed": har_flushed, "answer_path": "answer.json" if complete else None,
        "har_path": "network.har" if complete else None, "actions_path": "actions.jsonl",
        "task_sha256": hash_json(task)})
    return {"source": str(raw), "subject": str(subject), "subject_sha256": sha256(subject), "task": task,
            "run_id": "web", "attempt_id": "a0", "agent": "minisweagent",
            "model_fingerprint": cfg["model"]["fingerprint"]}


def test_web_import_registers_subject_not_patch_and_binds_offline_evaluation(cfg, official):
    _, _, task = imported(cfg, official)
    spec = web_artifacts(cfg, task)
    rid = import_rollout(cfg["project_root"], spec)
    evidence = RunEvidence(cfg["project_root"], "web")
    row = evidence.read()["rollouts"][rid]
    assert artifact_kinds(row) == ("trajectory", "subject")
    assert row["evaluation_subject"]["re_evaluation"] is True and row["submission_status"] == "submitted"
    assert "patch_path" not in row
    assert verify_subject(cfg["project_root"], row)
    report = Path(cfg["project_root"]) / "runs/web/harness/e0/summary.json"
    write_json(report, {})
    with pytest.raises(ValueError, match="subject mismatch"):
        evidence.record_evaluation("wrong", report, {rid: {"subject_sha256": "0" * 64,
            "status": "completed", "resolved": True}}, {})
    evidence.record_evaluation("e0", report, {rid: {"subject_sha256": row["subject_sha256"],
        "status": "completed", "resolved": False}}, {})
    assert evidence.read()["rollouts"][rid]["resolved"] is False


def test_har_or_answer_tampering_is_detected_after_canonical_import(cfg, official):
    _, _, task = imported(cfg, official)
    rid = import_rollout(cfg["project_root"], web_artifacts(cfg, task))
    row = RunEvidence(cfg["project_root"], "web").read()["rollouts"][rid]
    path = Path(cfg["project_root"]) / row["subject_path"]
    (path.parent / "network.har").write_text('{"log":{"entries":[{}]}}')
    with pytest.raises(ValueError, match="Changed subject output"):
        verify_subject(cfg["project_root"], row)


def test_incomplete_limit_subject_is_preserved_without_inventing_an_answer(cfg, official):
    _, _, task = imported(cfg, official)
    rid = import_rollout(cfg["project_root"], web_artifacts(cfg, task, complete=False, har_flushed=False))
    row = RunEvidence(cfg["project_root"], "web").read()["rollouts"][rid]
    assert row["submission_status"] == "unsubmitted" and row["resolved"] is None
    assert verify_subject(cfg["project_root"], row)
    manifest = read_json(Path(cfg["project_root"]) / row["subject_path"])
    assert not manifest["complete"] and manifest["answer_path"] is None


def test_complete_unflushed_subject_and_web_patch_fallback_are_rejected(cfg, official):
    _, _, task = imported(cfg, official)
    spec = web_artifacts(cfg, task, har_flushed=False)
    with pytest.raises(ValueError, match="flushed HAR"):
        import_rollout(cfg["project_root"], spec)
    spec.pop("subject")
    with pytest.raises(ValueError, match="fallback is forbidden"):
        import_rollout(cfg["project_root"], spec)


def test_web_action_analysis_rejected_and_template_grouping_does_not_leak_instances():
    from difficulty.datasets.identity import task_group, task_key
    cfg = {"benchmark": {"backend": "webarena_verified"}, "agent": {"backend": "minisweagent"}}
    assert capabilities(cfg)["subject"] == "web_trace"
    with pytest.raises(ValueError, match="action analysis"):
        capabilities(cfg, "action")
    a = {"benchmark": "webarena_verified", "revision": webarena.REVISION,
         "instance_id": "wav-108", "intent_template_id": 50}
    b = {**a, "instance_id": "wav-109"}
    assert task_key(a) != task_key(b) and task_group(a) == task_group(b)
    with pytest.raises(ValueError, match="template"):
        task_group({k: v for k, v in a.items() if k != "intent_template_id"})


def test_resolved_web_remaining_labels_bind_exact_subject_and_load_without_patch(cfg, official):
    from types import SimpleNamespace

    from difficulty.datasets.resolved import build_resolved_selection
    from difficulty.datasets.submission import load_selection

    _, _, task = imported(cfg, official)
    root = Path(cfg["project_root"])
    rid = import_rollout(root, web_artifacts(cfg, task))
    evidence = RunEvidence(root, "web")
    row = evidence.read()["rollouts"][rid]
    report = root / "runs/web/harness/e0/summary.json"
    outcome = {"subject_sha256": row["subject_sha256"], "status": "completed", "resolved": True}
    write_json(report, {"outcomes": {rid: outcome}})
    evidence.record_evaluation("e0", report, {rid: outcome}, {})
    trace = root / "runs/web/enrichment/online"
    event = {"sample_id": "web-close", "rollout_id": rid, "trajectory_sha256": row["trajectory_sha256"],
             "request_id": "native", "selected_evaluation_id": "e0", "message_index": 2,
             "labels": {"remaining_steps": 2, "remaining_reasoning_tokens": 3}, "remaining_steps": 2,
             "censored": False, "anchor": "think_close", "alignment_status": "exact", "sha256": "tensor-sha"}
    write_json(trace / "index.json", [event])
    write_json(trace / "requests.json", {"native": {"is_action_response": True}})
    write_json(trace / "receipt.json", {"files": {name: sha256(trace / name)
        for name in ("index.json", "requests.json")}})
    write_json(root / "difficulty/datasets/web-view/index.json", {"schema_version": 3, "dataset_id": "web",
        "source_receipt_sha256": sha256(trace / "receipt.json"), "hidden_root": str(root / "nas"), "samples": [event]})
    result = build_resolved_selection(root, "web", "web-view", "web-resolved")
    assert result["samples"] == 1
    dataset = root / "difficulty/datasets/web-resolved"
    settings = SimpleNamespace(dataset_id="web-resolved", dataset_receipt_sha256=sha256(dataset / "receipt.json"),
                               expected_rollouts=1, expected_tasks=1, expected_samples=1)
    rollouts, samples, _ = load_selection(cfg, settings)
    assert "patch_sha256" not in rollouts[0]
    assert rollouts[0]["evaluation_subject_sha256"] == row["subject_sha256"]
    assert samples[0]["intent_template_id"] == task["instance"]["intent_template_id"]
    assert samples[0]["remaining_steps_to_success"] == 2
    (root / row["subject_path"]).parent.joinpath("answer.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed subject output"):
        load_selection(cfg, settings)


def test_web_split_groups_different_instances_of_one_template_together():
    from difficulty.training.splits import _build_split

    rollouts, samples = [], []
    for index in range(20):
        row = {"benchmark": "webarena_verified", "revision": webarena.REVISION,
               "instance_id": f"wav-{index:03d}", "intent_template_id": index // 4,
               "rollout_id": f"r-{index}", "trajectory_sha256": f"{index:064x}", "source_kind": "online_close"}
        rollouts.append(row)
        samples.append({"sample_id": f"s-{index}", "rollout_id": row["rollout_id"],
                        "task_group": f"webarena_verified@{webarena.REVISION}::template-{index // 4}"})
    split = _build_split(rollouts, samples, "task_random_v1", 42, .2, .2, 1)
    assigned = {sid: role for role, ids in split["sample_ids"].items() for sid in ids}
    for template in range(5):
        assert len({assigned[f"s-{i}"] for i in range(template * 4, template * 4 + 4)}) == 1
    for strategy in ("event_random_v1", "repo_source_quota_v1", "inherit_task_split_v1"):
        with pytest.raises(ValueError, match="intent_template_id"):
            _build_split(rollouts, samples, strategy, 42, .2, .2, 1)
