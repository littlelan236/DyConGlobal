"""End-to-end action artifact contracts, cross-run identity and numeric chart semantics."""

import copy
import json
from pathlib import Path

import pytest

from conftest import task
from action.analyzers.trajectories import classify_trajectories, check_analysis
from action.artifacts import plot_analysis
from action.datasets.trajectories import select_trajectories
from action.evaluation.distributions import matrices, phase_metadata
from application.execution.evidence import RunEvidence
from core.io import artifact_dir, hash_json, read_json, sha256, write_json


def source(cfg, run, rid="same-id", commands=("cat x", "pytest -q"), *, iid=None):
    root = Path(cfg["project_root"])
    definition = task(iid or f"{run}-task")
    key = hash_json({k: definition[k] for k in ("benchmark", "revision", "instance_id")})
    trajectory = root / "runs" / run / "rollout/trajectories" / rid / "raw.traj.json"
    write_json(trajectory, {"messages": [{"role": "assistant", "tool_calls": [{"function": {
        "name": "bash", "arguments": json.dumps({"command": cmd})}}]} for cmd in commands]})
    patch = root / "runs" / run / "rollout/patches" / rid / "patch.diff"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("patch", encoding="utf-8")
    row = {"run_id": run, "task_key": key, "attempt_id": "a0", "model_fingerprint": "fixture-model",
           "trajectory_path": str(trajectory.relative_to(root)), "trajectory_sha256": sha256(trajectory),
           "patch_path": str(patch.relative_to(root)), "patch_sha256": sha256(patch),
           "evaluation_status": "not_run", "resolved": None, "selected_evaluation_id": None,
           "submission_status": "submitted", "termination_reason": "Submitted", "provenance": {}}
    RunEvidence(root, run).register_rollout(rid, row, definition)
    return row


def classify(cfg, aid="analysis", **kwargs):
    source(cfg, "one")
    return Path(classify_trajectories(cfg, aid, run_id="one", **kwargs))


def test_action_paths_ignore_run_ownership(cfg):
    bound = {**cfg, "run_id": "irrelevant"}
    root = Path(cfg["project_root"])
    assert artifact_dir(bound, "action", "unit") == root / "action/outputs/unit"
    assert artifact_dir(bound, "action", "view", "datasets") == root / "action/datasets/view"
    assert artifact_dir(bound, "hidden", "capture") == root / "runs/irrelevant/hidden/outputs/capture"
    assert artifact_dir(bound, "difficulty", "fit", "checkpoints") == root / "difficulty/checkpoints/fit"


def test_cross_run_same_rollout_id_cli_and_frozen_evidence(cfg, capsys, monkeypatch):
    from core.cli import main
    from catalog.store import Catalog
    source(cfg, "one", commands=("cat x", "pytest"))
    source(cfg, "two", commands=("rm /tmp/x", "cat y"))
    root = Path(cfg["project_root"])
    listing = root / "bindings.json"
    write_json(listing, [{"source_run_id": run, "rollout_id": "same-id", "cohort": run} for run in ("one", "two")])
    def no_catalog(*args, **kwargs):
        raise AssertionError("Action must not query the live Catalog")
    monkeypatch.setattr(Catalog, "read", no_catalog)
    main(["--config", cfg["config_source"], "action-classify", "combined", "--source-run", "one",
          "--source-run", "two", "--rollout-list", str(listing)])
    out = root / "action/outputs/combined"
    analysis = read_json(out / "analysis.json")
    assert [s["run_id"] for s in analysis["source_runs"]] == ["one", "two"]
    rows = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    assert len({r["action_id"] for r in rows}) == 4
    assert [r["label"] for r in rows] == ["INSPECT", "RUN", "CLEAN", "INSPECT"]
    before = sha256(out / "receipt.json")
    # A later exact-patch evaluation must not invalidate the old frozen analysis.
    evidence = RunEvidence(root, "one")
    report = root / "runs/one/harness/later.json"
    write_json(report, {"status": "completed"})
    evidence.record_evaluation("later", report.relative_to(root), {"same-id": {
        "patch_sha256": evidence.read()["rollouts"]["same-id"]["patch_sha256"], "status": "completed", "resolved": True}}, {})
    assert check_analysis(cfg, "combined")["sources"] == 2
    main(["--config", cfg["config_source"], "artifact", "verify", "combined", "--scope", "action-analysis"])
    main(["--config", cfg["config_source"], "artifact", "show", "combined", "--scope", "action-analysis"])
    assert sha256(out / "receipt.json") == before
    assert not (root / "enrichment").exists() and not (root / "hidden").exists()
    assert set(read_json(out / "summary.json")["by_source_run"]) == {"one", "two"}


def test_all_eligible_multi_run_and_explicit_selection_errors(cfg):
    source(cfg, "one")
    source(cfg, "two", commands=("ls",))
    assert len(select_trajectories(cfg, ["one", "two"])["sources"]) == 2
    with pytest.raises(ValueError, match="exactly one"):
        select_trajectories(cfg, ["one", "two"], ["same-id"])
    with pytest.raises(ValueError, match="explicit source run"):
        select_trajectories({**cfg, "run_id": "one"})
    with pytest.raises(ValueError, match="duplicate"):
        select_trajectories(cfg, ["one", "one"])
    listing = Path(cfg["project_root"]) / "bindings.json"
    for values in [[], ["same-id"], [{"source_run_id": "three", "rollout_id": "same-id"}],
                   [{"source_run_id": "one", "rollout_id": "same-id"}] * 2]:
        write_json(listing, values)
        with pytest.raises(ValueError):
            select_trajectories(cfg, ["one", "two"], rollout_list=listing)


def test_duplicate_trajectory_aliases_across_runs_and_conflicting_tasks(cfg):
    source(cfg, "one", iid="same-task")
    source(cfg, "two", iid="same-task")
    out = Path(classify_trajectories(cfg, "dedup", run_id=["one", "two"]))
    selection = read_json(out / "selection.json")
    assert len(selection["sources"]) == len(selection["duplicate_trajectory_aliases"]) == 1
    assert check_analysis(cfg, "dedup")["steps"] == 2
    source(cfg, "three", iid="different-task")
    with pytest.raises(ValueError, match="different tasks"):
        classify_trajectories(cfg, "wrong", run_id=["one", "three"])


def test_run_config_freezes_classifier_cwd_and_no_command_execution(cfg):
    root = Path(cfg["project_root"])
    marker = root / "must-not-be-created"
    source(cfg, "one", commands=(f"touch {marker}",))
    # A legacy producer fixture still needs a valid immutable config identity.
    frozen = copy.deepcopy(cfg)
    frozen["environment"]["cwd"] = "/producer-cwd"
    frozen["execution"]["limits_retry"]["enabled"] = False
    frozen.pop("config_hash")
    frozen["config_hash"] = hash_json(frozen)
    frozen.update(selection={}, selection_sha256=hash_json({}))
    write_json(root / "runs/one/resolved-config.json", frozen)
    out = Path(classify_trajectories({**cfg, "run_id": "unrelated"}, "cwd", run_id="one"))
    assert read_json(out / "selection.json")["sources"][0]["classifier_cwd"] == "/producer-cwd"
    assert check_analysis(cfg, "cwd")["verified"]
    assert not marker.exists() and not (root / "runs/unrelated/action").exists()


@pytest.mark.parametrize("filename", ["analysis.json", "selection.json", "source-evidence/one.json", "labels.jsonl",
                                    "summary.json", "plots/progress-share.png", "code/action/analyzers/taxonomy.py"])
def test_artifact_tampering_is_detected(cfg, filename):
    out = classify(cfg)
    with (out / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        check_analysis(cfg, "analysis")


def test_changed_source_bytes_duplicate_binding_and_path_escape(cfg):
    out = classify(cfg)
    row = RunEvidence(cfg["project_root"], "one").read()["rollouts"]["same-id"]
    path = Path(cfg["project_root"]) / row["trajectory_path"]
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        check_analysis(cfg, "analysis")
    with pytest.raises(FileExistsError):
        classify_trajectories(cfg, "analysis", run_id="one")
    with pytest.raises(ValueError, match="escapes"):
        check_analysis(cfg, "../analysis")


def test_heatmap_semantics_micro_macro_empty_bins_and_boundaries():
    def rows(run, labels):
        return [{"source_run_id": run, "rollout_id": "same-id", "step_id": i, "label": label,
                 **phase_metadata(i, len(labels))} for i, label in enumerate(labels)]
    sources = [{"source_run_id": run, "rollout_id": "same-id"} for run in ["long", "short", "empty"]]
    labels = rows("long", ["INSPECT"] * 11) + rows("short", ["RUN", "CLEAN"])
    result = matrices(labels, sources)
    phase, transition = result["phase"], result["transition"]
    inspect = phase["labels"].index("INSPECT")
    assert phase["micro"][inspect][0] == pytest.approx(2 / 3)
    assert phase["macro"][inspect][0] == .5
    assert phase["action_denominators"] == [3, 2, 2, 2, 4]
    assert transition["pairs"] == 11  # 10 long pairs + 1 short pair; no cross-run edge.
    assert transition["zero_step_trajectories"] == 1
    for line, denominator in zip(transition["probabilities"], transition["row_denominators"]):
        assert sum(line) == pytest.approx(1) if denominator else all(v is None for v in line)
    assert [phase_metadata(i, 6)["phase_bin"] for i in range(6)] == [0, 1, 2, 3, 4, 4]
    single = matrices(rows("short", ["RUN"]), sources[1:])
    assert single["transition"]["pairs"] == 0 and single["transition"]["single_step_trajectories"] == 1
    assert all(line[1] is None for line in single["phase"]["micro"])


def test_zero_and_single_action_analyses(cfg):
    source(cfg, "one", commands=())
    out = Path(classify_trajectories(cfg, "zero", run_id="one"))
    assert check_analysis(cfg, "zero")["steps"] == 0
    assert read_json(out / "transition.json")["zero_step_trajectories"] == 1
    source(cfg, "two", commands=("cat x",))
    out = Path(classify_trajectories(cfg, "single", run_id="two"))
    assert read_json(out / "transition.json")["pairs"] == 0
    assert check_analysis(cfg, "single")["steps"] == 1


def test_plotting_is_local_immutable_and_detects_child_tampering(cfg, monkeypatch):
    from PIL import Image
    import action.datasets.trajectories as trajectories
    out = classify(cfg)
    original = sha256(out / "receipt.json")
    def no_sources(*args, **kwargs):
        raise AssertionError("Plotting must not read source trajectories")
    with monkeypatch.context() as patch:
        patch.setattr(trajectories, "read_actions", no_sources)
        assert plot_analysis(cfg, "analysis")["reused"]
        view = plot_analysis(cfg, "analysis", plot_id="macro", aggregation="macro", formats=("png", "svg"))
    assert sha256(out / "receipt.json") == original
    with Image.open(out / "plots/progress-share.png") as image:
        assert image.size == (1800, 1100)
    assert check_analysis(cfg, "analysis")["plot_views"] == ["macro"]
    with pytest.raises(FileExistsError):
        plot_analysis(cfg, "analysis", plot_id="macro")
    with pytest.raises(ValueError, match="plot-id"):
        plot_analysis(cfg, "analysis", aggregation="macro")
    target = Path(view["path"]) / "transition-probability.png"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError):
        check_analysis(cfg, "analysis")


def test_semantic_check_rejects_resigned_summary(cfg):
    out = classify(cfg)
    summary = read_json(out / "summary.json")
    summary["category_counts"]["RUN"] += 1
    write_json(out / "summary.json", summary)
    receipt = read_json(out / "receipt.json")
    receipt["files"]["summary.json"] = sha256(out / "summary.json")
    write_json(out / "receipt.json", receipt)
    with pytest.raises(ValueError):
        check_analysis(cfg, "analysis")


def test_legacy_run_scoped_artifact_is_explicit_read_only(cfg):
    from action.analyzers.taxonomy import VERSION
    from action.evaluation.summary import legacy_summarize
    out = classify(cfg)
    selection = read_json(out / "selection.json")
    rows = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    legacy = Path(cfg["project_root"]) / "runs/legacy/action/outputs/old"
    for name, data in [("selection.json", selection), ("summary.json", legacy_summarize(rows, selection["sources"])),
                       ("examples.json", {}), ("resolved-config.json", cfg)]:
        write_json(legacy / name, data)
    (legacy / "labels.jsonl").write_bytes((out / "labels.jsonl").read_bytes())
    write_json(legacy / "receipt.json", {"schema_version": 1, "analysis_id": "old", "taxonomy_version": VERSION,
               "selection_sha256": hash_json(selection), "files": {p.name: sha256(p) for p in legacy.iterdir()}})
    before = sha256(legacy / "receipt.json")
    assert check_analysis(cfg, "runs/legacy/action/outputs/old")["legacy"]
    assert sha256(legacy / "receipt.json") == before
    with pytest.raises(ValueError, match="schema-2"):
        plot_analysis(cfg, "runs/legacy/action/outputs/old")


def test_plot_failure_does_not_publish_partial_analysis(cfg, monkeypatch):
    import action.evaluation.visualize as plots
    source(cfg, "one")
    def fail(*args, **kwargs):
        raise RuntimeError("fixture drawing failure")
    monkeypatch.setattr(plots, "render_plots", fail)
    with pytest.raises(RuntimeError, match="drawing failure"):
        classify_trajectories(cfg, "incomplete", run_id="one")
    assert not artifact_dir(cfg, "action", "incomplete").exists()
