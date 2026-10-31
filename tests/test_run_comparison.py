"""Synthetic isolated runs exercise cohort identity, retries and winner-only costs."""

import json
from pathlib import Path

import pytest

from application.execution.comparison import compare_runs, format_markdown
from application.execution.evidence import RunEvidence
from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.io import hash_json, read_json, sha256, write_json
from rollout.artifacts import import_rollout
from tests.conftest import task as fixture_task


@pytest.fixture(autouse=True)
def comparison_backend(cfg):
    # These isolated coding-task fixtures must not inherit the live root backend.
    cfg["benchmark"] = {"backend": "swebench", "snapshot_ref": None}
    cfg["agent"].update(backend="minisweagent", profile_ref=None)
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})


def task(iid):
    return {**fixture_task(iid), "benchmark": "swebench"}


def make_runs(cfg, *names, count=1, ids=("a", "b", "c")):
    Catalog(cfg["project_root"]).register_tasks([task(i) for i in ids])
    return [create_run(cfg, name, operation="rollout", task_ids=list(ids), repeat_count=count) for name in names]


def add_attempt(cfg, run, iid, status="Submitted", *, attempt=0, final=True,
                resolved=None, prompt=10, completion=3, missing=False, repetition=0):
    root = Path(cfg["project_root"])
    loaded = load_plan(root, run.name)
    item = next(i for i in loaded["work_items"]["items"]
                if i["instance_id"] == iid and i["repetition_index"] == repetition)
    aid = f"rep-{repetition:03d}-attempt-{attempt:03d}"
    work = run / "rollout/tasks" / iid / "repetitions" / item["repetition_id"]
    out = work / "attempts" / aid
    context = {**item, "attempt_id": aid}
    response = {"id": f"{run.name}-{iid}-{aid}", "usage": {} if missing else {
        "prompt_tokens": prompt, "completion_tokens": completion}}
    # Deliberately enormous all-branch usage must never leak into comparison.
    raw = {"instance_id": iid, "info": {"exit_status": status, "submission": "patch",
                                       "model_stats": {"api_calls": 1}},
           "messages": [{"role": "assistant", "content": "selected",
                         "extra": {"response": response,
                                   "dycon_parallel": {"selected_branch_id": 3, "branch_count": 32,
                                                      "total_usage": {"prompt_tokens": 999999999}}}}]}
    write_json(out / "raw.traj.json", raw)
    write_json(out / "attempt-context.json", context)
    write_json(out / "attempt-started.json", context)
    write_json(out / "agent-outcome.json", {**context, "agent_exit_status": status})
    rid = import_rollout(root, {"source": str(out / "raw.traj.json"), "task": task(iid),
                               "run_id": run.name, "attempt_id": aid, "sample": item,
                               "model_fingerprint": "fixture"})
    result = {**context, "rollout_id": rid, "exit_status": status, "agent_exit_status": status,
              "disposition": "accepted"}
    write_json(out / "attempt.json", result)
    if final:
        write_json(work / "result.json", result)
    if resolved is not None:
        report = run / "harness" / f"{iid}-{aid}.json"
        write_json(report, {"resolved": resolved})
        row = RunEvidence(root, run.name).read()["rollouts"][rid]
        RunEvidence(root, run.name).record_evaluation(f"e-{iid}-{aid}", report, {
            rid: {"patch_sha256": row["patch_sha256"], "status": "completed", "resolved": resolved}}, {})
    return out, rid


def test_common_tasks_retry_dedup_selected_costs_and_changes(cfg):
    left, right = make_runs(cfg, "left", "right")
    add_attempt(cfg, left, "a", "LimitsExceeded", final=False)
    add_attempt(cfg, left, "a", attempt=1, resolved=True)
    add_attempt(cfg, right, "a", prompt=20, completion=4, resolved=False)
    add_attempt(cfg, left, "b")
    add_attempt(cfg, right, "b", "ContextWindowExceededError")
    add_attempt(cfg, left, "c", "reasoningLoop")
    add_attempt(cfg, right, "c")
    report = compare_runs(cfg["project_root"], [left.name, right.name], baseline=left.name)
    assert report["common_submitted_task_count"] == 1
    a, b = report["runs"]
    assert a["exits"]["physical_attempt_count"] == 4
    assert a["exits"]["exit_count"] == 2
    assert a["exits"]["exit_rate"] == .5
    assert b["exits"]["by_reason"]["context_limit_exceeded"]["count"] == 1
    assert a["common_submitted"]["steps"] == 1
    assert a["common_submitted"]["prompt_tokens"] == 10
    assert a["common_submitted"]["completion_tokens"] == 3
    assert a["common_submitted"]["total_tokens"] == 13
    assert b["changes_from_baseline"]["common_submitted"]["prompt_tokens"]["relative_change"] == 1
    assert b["changes_from_baseline"]["common_submitted"]["resolved_rate"]["percentage_point_change"] == -100
    assert "-100.00 pp" in format_markdown(report)


def test_mixed_repeat_counts_need_selection_and_invalid_run_selection_rejected(cfg):
    single, = make_runs(cfg, "single")
    repeated, = make_runs(cfg, "repeated", count=2)
    with pytest.raises(ValueError, match="different repeat_count"):
        compare_runs(cfg["project_root"], [single.name, repeated.name])
    with pytest.raises(ValueError, match="distinct"):
        compare_runs(cfg["project_root"], [single.name, single.name])
    with pytest.raises(ValueError, match="Baseline"):
        compare_runs(cfg["project_root"], ["a", "b"], baseline="c")


def test_missing_usage_unknown_evaluation_and_zero_baseline(cfg):
    left, right = make_runs(cfg, "missing-left", "missing-right", ids=("a",))
    add_attempt(cfg, left, "a", missing=True)
    add_attempt(cfg, right, "a", resolved=True)
    report = compare_runs(cfg["project_root"], [left.name, right.name], baseline=left.name)
    a, b = report["runs"]
    assert a["common_submitted"]["prompt_tokens"] is None
    assert a["common_submitted"]["unknown_evaluation_count"] == 1
    assert a["common_submitted"]["resolved_rate"] == 0
    assert not a["token_coverage"]["complete"]
    rate = b["changes_from_baseline"]["common_submitted"]["resolved_rate"]
    assert rate["relative_change"] is None
    assert rate["percentage_point_change"] == 100
    assert b["changes_from_baseline"]["exits"]["exit_count"]["relative_change"] is None


def test_empty_cohort_has_no_division_by_zero(cfg):
    left, right = make_runs(cfg, "empty-left", "empty-right", ids=("a",))
    add_attempt(cfg, left, "a", "LimitsExceeded")
    add_attempt(cfg, right, "a")
    report = compare_runs(cfg["project_root"], [left.name, right.name], baseline=left.name)
    assert report["common_submitted_task_count"] == 0
    for row in report["runs"]:
        assert row["common_submitted"]["resolved_rate"] is None
    assert "N/A" in format_markdown(report)


def test_equal_repeat_counts_pair_ordinals_and_keep_sample_denominator(cfg):
    left, right = make_runs(cfg, "paired-left", "paired-right", count=2, ids=("a", "b"))
    add_attempt(cfg, left, "a", "LimitsExceeded", final=False)
    add_attempt(cfg, left, "a", attempt=1, resolved=True, prompt=10)
    add_attempt(cfg, left, "a", repetition=1, resolved=False, prompt=30)
    add_attempt(cfg, right, "a", resolved=True, prompt=20)
    add_attempt(cfg, right, "a", repetition=1, resolved=True, prompt=40)
    add_attempt(cfg, left, "b")
    add_attempt(cfg, left, "b", "reasoning_loop", repetition=1)
    add_attempt(cfg, right, "b", "ContextWindowExceededError")
    add_attempt(cfg, right, "b", repetition=1)
    report = compare_runs(cfg["project_root"], [left.name, right.name], baseline=left.name)
    assert report["comparison_mode"] == "all_repetitions"
    assert report["common_submitted_task_count"] == 1
    assert report["common_submitted_sample_count"] == 2
    assert report["resolved_rate_denominator"] == 2
    assert [s["paired_repetition_index"] for s in report["common_submitted_samples"]] == [0, 1]
    a, b = report["runs"]
    assert a["selected_repetition_ids"] == ["rep-000", "rep-001"]
    assert a["exits"]["physical_attempt_count"] == 5
    assert a["exits"]["exit_count"] == 2
    assert b["exits"]["physical_attempt_count"] == 4
    assert a["submitted_task_count"] == 2
    assert a["submitted_sample_count"] == 3
    assert a["common_submitted"]["resolved_rate"] == .5
    assert b["common_submitted"]["resolved_rate"] == 1
    assert a["common_submitted"]["steps"] == 2
    assert a["common_submitted"]["prompt_tokens"] == 40
    assert b["common_submitted"]["prompt_tokens"] == 60
    assert b["changes_from_baseline"]["common_submitted"]["resolved_rate"]["percentage_point_change"] == 50
    assert [s["repetition_id"] for s in a["samples"]] == ["rep-000", "rep-001"]
    assert "task/repetition pairs: 2" in format_markdown(report)


def test_different_repeat_counts_select_distinct_ordinals_and_filter_exits(cfg):
    left, = make_runs(cfg, "select-left", count=2, ids=("a",))
    right, = make_runs(cfg, "select-right", count=3, ids=("a",))
    add_attempt(cfg, left, "a", "reasoning_loop")  # Not selected.
    add_attempt(cfg, left, "a", "LimitsExceeded", repetition=1, final=False)
    add_attempt(cfg, left, "a", repetition=1, attempt=1, resolved=True, prompt=50)
    add_attempt(cfg, right, "a", "ContextWindowExceededError")  # Not selected.
    add_attempt(cfg, right, "a", "reasoning_loop", repetition=1)  # Not selected.
    add_attempt(cfg, right, "a", repetition=2, resolved=False, prompt=25)
    report = compare_runs(cfg["project_root"], [left.name, right.name], baseline=left.name,
                          repetitions={left.name: 1, right.name: "rep-002"})
    assert report["comparison_mode"] == "selected_repetition"
    assert report["common_submitted_sample_count"] == 1
    assert report["common_submitted_task_count"] == 1
    assert report["common_submitted_samples"][0]["repetition_by_run"] == {
        left.name: "rep-001", right.name: "rep-002"}
    a, b = report["runs"]
    assert a["exits"]["physical_attempt_count"] == 2
    assert a["exits"]["exit_count"] == 1
    assert a["exits"]["exit_rate"] == .5
    assert b["exits"]["physical_attempt_count"] == 1
    assert b["exits"]["exit_count"] == 0
    assert a["common_submitted"]["prompt_tokens"] == 50
    assert b["common_submitted"]["prompt_tokens"] == 25
    assert b["changes_from_baseline"]["common_submitted"]["resolved_rate"]["percentage_point_change"] == -100
    assert a["samples"][0]["repetition_id"] == "rep-001"
    assert b["samples"][0]["repetition_id"] == "rep-002"


def test_selected_repetition_also_supported_for_equal_counts(cfg):
    left, right = make_runs(cfg, "equal-left", "equal-right", count=2, ids=("a",))
    add_attempt(cfg, left, "a")
    add_attempt(cfg, right, "a", repetition=1)
    all_report = compare_runs(cfg["project_root"], [left.name, right.name])
    assert all_report["common_submitted_sample_count"] == 0
    report = compare_runs(cfg["project_root"], [left.name, right.name],
                          repetitions={left.name: 0, right.name: 1})
    assert report["common_submitted_sample_count"] == 1


def test_single_repetition_run_defaults_to_its_only_repetition(cfg):
    left, = make_runs(cfg, "only-left", ids=("a",))
    right, = make_runs(cfg, "many-right", count=3, ids=("a",))
    add_attempt(cfg, left, "a")
    add_attempt(cfg, right, "a", repetition=2)
    report = compare_runs(cfg["project_root"], [left.name, right.name], repetitions={right.name: 2})
    assert report["common_submitted_sample_count"] == 1
    assert report["runs"][0]["selected_repetition_ids"] == ["rep-000"]


@pytest.mark.parametrize("selectors,match", [
    ({"absent": 0}, "unrequested"),
    ({"validate-left": 0}, "every multi-repetition"),
    ({"validate-left": -1, "validate-right": 0}, "Invalid repetition"),
    ({"validate-left": 2, "validate-right": 0}, "Invalid repetition"),
    ({"validate-left": "wrong", "validate-right": 0}, "Invalid repetition"),
    ({"validate-left": True, "validate-right": 0}, "Invalid repetition"),
])
def test_repetition_selectors_are_validated(cfg, selectors, match):
    left, right = make_runs(cfg, "validate-left", "validate-right", count=2, ids=("a",))
    with pytest.raises(ValueError, match=match):
        compare_runs(cfg["project_root"], [left.name, right.name], repetitions=selectors)


def test_cli_repetition_selection_and_invalid_selectors(cfg, capsys):
    from core.cli import main
    left, = make_runs(cfg, "repeat-cli-left", count=2, ids=("a",))
    right, = make_runs(cfg, "repeat-cli-right", count=3, ids=("a",))
    add_attempt(cfg, left, "a", repetition=1)
    add_attempt(cfg, right, "a", repetition=2)
    args = ["--config", cfg["config_source"], "run", "compare", left.name, right.name,
            "--repetition", f"{left.name}=1", "--repetition", f"{right.name}=rep-002"]
    main(args)
    report = json.loads(capsys.readouterr().out)
    assert report["common_submitted_sample_count"] == 1
    for extra in (["--repetition", f"{left.name}=0"], ["--repetition", "bad-selector"]):
        with pytest.raises(SystemExit) as exc:
            main([*args, *extra])
        assert exc.value.code == 2


def test_agent_fact_overrides_capture_failure_and_no_double_count(cfg):
    left, right = make_runs(cfg, "capture-left", "capture-right", ids=("a",))
    out, _ = add_attempt(cfg, left, "a", "contextLimitExceeded")
    result = read_json(out / "attempt.json")
    result.update(exit_status="CaptureError", agent_exit_status="contextLimitExceeded")
    write_json(out / "attempt.json", result)
    add_attempt(cfg, right, "a", "reasoning_loop")
    report = compare_runs(cfg["project_root"], [left.name, right.name])
    assert report["runs"][0]["exits"]["physical_attempt_count"] == 1
    assert report["runs"][0]["exits"]["exit_count"] == 1


def test_started_and_reserved_attempts_and_unfinalized_submission(cfg):
    left, right = make_runs(cfg, "live-left", "live-right", ids=("a",))
    out, _ = add_attempt(cfg, left, "a", final=False)
    item = load_plan(cfg["project_root"], left.name)["work_items"]["items"][0]
    started = out.parent / "rep-000-attempt-001"
    write_json(started / "attempt-context.json", {**item, "attempt_id": started.name})
    write_json(started / "attempt-started.json", {})
    reserved = out.parent / "rep-000-attempt-002"
    write_json(reserved / "attempt-context.json", {**item, "attempt_id": reserved.name})
    add_attempt(cfg, right, "a")
    report = compare_runs(cfg["project_root"], [left.name, right.name])
    assert report["runs"][0]["exits"]["physical_attempt_count"] == 2
    assert report["runs"][0]["exits"]["unknown_outcome_count"] == 1
    assert report["common_submitted_task_count"] == 0


def test_tampered_final_trajectory_rejected(cfg):
    left, right = make_runs(cfg, "hash-left", "hash-right", ids=("a",))
    _, rid = add_attempt(cfg, left, "a")
    add_attempt(cfg, right, "a")
    row = RunEvidence(cfg["project_root"], left.name).read()["rollouts"][rid]
    path = Path(cfg["project_root"]) / row["trajectory_path"]
    path.write_text('{}')
    with pytest.raises(ValueError, match="trajectory changed"):
        compare_runs(cfg["project_root"], [left.name, right.name])


def test_cli_uses_snapshots_with_broken_root_and_refuses_overwrite(cfg, capsys):
    from core.cli import main
    left, right = make_runs(cfg, "cli-left", "cli-right", ids=("a",))
    add_attempt(cfg, left, "a")
    add_attempt(cfg, right, "a")
    Path(cfg["config_source"]).write_text("invalid: [")
    output = Path(cfg["project_root"]) / "comparison.json"
    args = ["--config", cfg["config_source"], "run", "compare", left.name, right.name,
            "--baseline", left.name, "--output", str(output)]
    main(args)
    assert json.loads(capsys.readouterr().out) == read_json(output)
    digest = sha256(output)
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert sha256(output) == digest


def test_metadata_only_reader_never_fetches_native_ledger(tmp_path, monkeypatch):
    import hidden.artifacts
    from enrichment.reader import TrajectoryReader

    def forbidden(*args, **kwargs):
        raise AssertionError("Report must not fetch token arrays from NAS")
    monkeypatch.setattr(hidden.artifacts, "read_request", forbidden)
    path = tmp_path / "compact.json"
    write_json(path, {"messages": [{"role": "assistant", "extra": {
        "response": {"ledger_ref": "native.json", "ledger_root": "/unavailable",
                     "usage": {"prompt_tokens": 9, "completion_tokens": 4}}}}]})
    assert TrajectoryReader(path, hydrate_responses=False).usage()["total_tokens"] == 13


def test_migrated_active_run_retains_physical_paths(cfg):
    left, right = make_runs(cfg, "migrated-left", "migrated-right", ids=("a",))
    for run in (left, right):
        out, _ = add_attempt(cfg, run, "a")
        work = out.parent.parent
        task_root = work.parent.parent
        (work / "attempts").rename(task_root / "attempts")
        (work / "result.json").rename(task_root / "result.json")
        write_json(run / "migration/source.json", {"source_schema": 1, "target_schema": 2})
    report = compare_runs(cfg["project_root"], [left.name, right.name])
    assert report["common_submitted_task_count"] == 1
    assert report["runs"][0]["exits"]["physical_attempt_count"] == 1
    # A new unfinished retry must not resurrect the migrated final cursor.
    add_attempt(cfg, left, "a", attempt=1, final=False)
    report = compare_runs(cfg["project_root"], [left.name, right.name])
    assert report["common_submitted_task_count"] == 0
    assert report["runs"][0]["exits"]["physical_attempt_count"] == 2


def test_identical_instance_id_with_other_revision_is_not_common(cfg):
    left, = make_runs(cfg, "revision-left", ids=("a",))
    add_attempt(cfg, left, "a")
    # Rebuild a separate valid run from a distinct benchmark revision.
    alternate = {**task("a"), "revision": "v2"}
    with Catalog(cfg["project_root"]).transaction() as catalog:
        catalog["tasks"] = {}  # The first run retains its immutable v1 selection.
    Catalog(cfg["project_root"]).register_tasks([alternate])
    task_file = Path(cfg["project_root"]) / "v2-tasks.json"
    write_json(task_file, [alternate])
    other = create_run(cfg, "other-revision", operation="rollout", tasks_file=task_file)
    item = load_plan(cfg["project_root"], other.name)["work_items"]["items"][0]
    raw = other / "rollout/tasks/a/repetitions/rep-000/attempts/rep-000-attempt-000/raw.traj.json"
    write_json(raw, {"instance_id": "a", "info": {"exit_status": "Submitted", "submission": "other patch"}})
    rid = import_rollout(cfg["project_root"], {
        "source": str(raw), "task": alternate, "run_id": other.name,
        "attempt_id": "rep-000-attempt-000", "sample": item, "model_fingerprint": "fixture"})
    write_json(raw.parent.parent.parent / "result.json", {
        **item, "attempt_id": "rep-000-attempt-000", "rollout_id": rid, "exit_status": "Submitted"})
    report = compare_runs(cfg["project_root"], [left.name, other.name])
    assert report["runs"][1]["submitted_task_count"] == 1
    assert report["common_submitted_task_count"] == 0
