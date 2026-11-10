"""Active status counts samples once and keeps execution scopes separate."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from application.execution import status
from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.io import read_json, write_json
from tests.conftest import task


@pytest.fixture
def loaded(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("active-task")])
    run = create_run(cfg, "active-summary", operation="rollout", task_ids=["active-task"], repeat_count=3)
    return load_plan(cfg["project_root"], run.name)


def attempt(loaded, index, number, *, execution="002", result=None, started=True):
    item = loaded["work_items"]["items"][index]
    path = (loaded["run"] / "rollout/tasks" / item["instance_id"] / "repetitions" /
            item["repetition_id"] / "attempts" / f"{item['repetition_id']}-attempt-{number:03d}")
    write_json(path / "attempt-context.json", {**item, "attempt_id": path.name})
    if started:
        write_json(path / "attempt-started.json", {"execution_attempt_id": execution})
    if result:
        write_json(path / "attempt.json", {**item, "attempt_id": path.name, **result})
    return path


def summary(loaded, samples=None, latest="002"):
    rows = status._active_sample_rows(loaded, {"samples": samples or {}}, latest)
    return status._active_sample_summary(rows, latest)


def test_retry_replaces_old_failure_and_reserved_attempt_is_not_counted(loaded):
    attempt(loaded, 0, 0, execution="001", result={"exit_status": "OpenHandsError", "disposition": "failed"})
    attempt(loaded, 0, 1, result={"exit_status": "Submitted", "disposition": "accepted"})
    attempt(loaded, 0, 2, started=False)
    result = summary(loaded)
    assert result["exit_status_counts"] == {"Pending": 2, "Submitted": 1}
    assert result["latest_attempt"]["sample_count"] == 3
    assert result["latest_attempt"]["failure_counts"] == {}
    assert result["latest_attempt"]["exit_status_counts"] == {"Pending": 2, "Submitted": 1}


def test_running_retry_uses_new_attempt_execution_not_stale_progress_assignment(loaded):
    attempt(loaded, 0, 0, execution="001", result={"exit_status": "LimitsExceeded", "disposition": "failed"})
    current = attempt(loaded, 0, 1)
    item = loaded["work_items"]["items"][0]
    progress = {item["sample_key"]: {**item, "attempt_id": current.name,
                "exit_status": "Running", "disposition": "pending",
                "serving_assignment": {"attempt_id": "rep-000-attempt-000", "execution_attempt_id": "001"}}}
    result = summary(loaded, progress)
    assert result["exit_status_counts"] == {"Pending": 2, "Running": 1}
    assert result["latest_attempt"]["exit_status_counts"] == {"Pending": 2, "Running": 1}
    assert result["latest_attempt"]["pending_count"] == 3
    assert result["latest_attempt"]["failure_counts"] == {}


def test_resume_pending_does_not_hide_latest_finished_attempt(loaded):
    attempt(loaded, 0, 0, execution="001", result={"exit_status": "VerifierError", "disposition": "failed"})
    result = summary(loaded)
    assert result["exit_status_counts"] == {"Pending": 2, "VerifierError": 1}
    assert result["latest_attempt"]["sample_count"] == 2
    assert result["latest_attempt"]["exit_status_counts"] == {"Pending": 2}


@pytest.mark.parametrize("cancelled, expected", [(True, "Cancelled"), (False, "Interrupted")])
def test_exited_attempt_without_final_receipt_is_not_reported_running(loaded, cancelled, expected):
    attempt(loaded, 0, 0, execution="001")
    execution = loaded["run"] / "execution/attempts/001"
    write_json(execution / "exit.json", {"exit_code": -15})
    if cancelled:
        write_json(execution / "termination.json", {"request_id": "user-stop"})
    assert summary(loaded)["exit_status_counts"] == {expected: 1, "Pending": 2}


def test_attempt_numbers_are_numeric_and_unstarted_replan_history_is_excluded(loaded):
    attempt(loaded, 0, 999, result={"exit_status": "OpenHandsError", "disposition": "failed"})
    attempt(loaded, 0, 1000, result={"exit_status": "Submitted", "disposition": "accepted"})
    item = loaded["work_items"]["items"][0]
    assert summary(loaded)["exit_status_counts"] == {"Pending": 2, "Submitted": 1}
    write_json(loaded["run"] / "rollout/replan-attempt-floors.json",
               {item["instance_id"] + "/" + item["repetition_id"]: 1000})
    assert summary(loaded)["exit_status_counts"] == {"Pending": 3}


def test_incomplete_attempt_without_execution_identity_is_unknown(loaded):
    attempt(loaded, 0, 0, execution=None)
    result = summary(loaded)
    assert result["exit_status_counts"] == {"Pending": 2, "unknown": 1}
    assert result["latest_attempt"]["sample_count"] == 2


def test_active_status_cli_shape_counts_and_read_only_behavior(loaded, cfg, monkeypatch, capsys):
    import json
    import subprocess
    from core.cli import main

    run = loaded["run"]
    attempt(loaded, 0, 0, execution="001", result={"exit_status": "Submitted", "disposition": "accepted"})
    attempt(loaded, 1, 0)
    write_json(run / "status.json", {"stage": "rollout", "updated_at": "status-time"})
    write_json(run / "progress.json", {"updated_at": "progress-time", "samples": {}})
    write_json(run / "execution/latest.json", {"attempt": "002"})
    write_json(Path(cfg["project_root"]) / "state/run-resume-queue/queued/status.json",
               {"stage": "queued", "request_id": "queued"})
    write_json(Path(cfg["project_root"]) / "state/run-resume-queue/queued/request.json",
               {"run_id": "waiting-run", "after_run": run.name})
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="application.execution.supervisor " + run.name))

    main(["--config", cfg["config_source"], "run", "active-status"])
    output = json.loads(capsys.readouterr().out)
    assert output["count"] == 1
    row = output["runs"][0]
    assert list(row) == ["run_id", "operation", "task_count", "status",
                         "progress_updated_at", "sample_count", "sample_summary"]
    assert row["task_count"] == 1 and row["sample_count"] == 3
    assert row["progress_updated_at"] == "progress-time"
    assert row["sample_summary"]["exit_status_counts"] == {"Pending": 1, "Running": 1, "Submitted": 1}
    assert row["sample_summary"]["latest_attempt"]["execution_attempt_id"] == "002"
    assert row["sample_summary"]["pending_count"] == 2
    assert row["sample_summary"]["latest_attempt"]["pending_count"] == 2
    assert row["sample_summary"]["latest_attempt"]["exit_status_counts"] == {"Pending": 1, "Running": 1}
    assert sum(row["sample_summary"]["exit_status_counts"].values()) == row["sample_count"]
    assert output["queued_count"] == 1 and output["queued_runs"][0]["run_id"] == "waiting-run"
    assert {p: p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


def test_pending_is_not_a_failure():
    result = status._sample_summary({"s": {"exit_status": "Pending", "disposition": "pending",
                                          "serving_assignment": {"execution_attempt_id": "002"}}}, "002")
    assert result["failure_counts"] == {}
    assert result["latest_attempt"]["failure_counts"] == {}


def test_resumed_queue_overrides_old_cancelled_attempt_and_counts_running_as_pending(loaded):
    old = attempt(loaded, 0, 0, execution="001")
    write_json(loaded["run"] / "execution/attempts/001/exit.json", {"exit_code": -15})
    write_json(loaded["run"] / "execution/attempts/001/termination.json", {"request_id": "stop"})
    attempt(loaded, 1, 0)
    item = loaded["work_items"]["items"][0]
    cursor = {item["sample_key"]: {**item, "exit_status": "Pending", "disposition": "pending"}}
    before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    result = summary(loaded, cursor)
    assert result["pending_count"] == 3
    assert result["exit_status_counts"] == {"Pending": 2, "Running": 1}
    assert result["latest_attempt"]["sample_count"] == 3
    assert result["latest_attempt"]["pending_count"] == 3
    assert result["latest_attempt"]["exit_status_counts"] == {"Pending": 2, "Running": 1}
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == before


def test_authorized_failure_is_pending_until_new_attempt_starts_then_keeps_new_result(loaded, cfg):
    from application.execution.failed_retry import request_failed_retry

    item = loaded["work_items"]["items"][0]
    old = attempt(loaded, 0, 0, execution="001", result={"exit_status": "OpenHandsError", "disposition": "failed"})
    write_json(old.parent.parent / "result.json", read_json(old / "attempt.json"))
    failure_cursor = {item["sample_key"]: read_json(old / "attempt.json")}
    before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert summary(loaded, failure_cursor)["exit_status_counts"] == {"OpenHandsError": 1, "Pending": 2}
    request_failed_retry(cfg["project_root"], loaded["run"].name, request_id="retry-failure",
                         exit_statuses=["OpenHandsError"], sample_keys=[item["sample_key"]], reason="Retry failed sample")
    # No progress reset or fresh physical attempt is needed for the queue to be visible.
    queued = summary(loaded, failure_cursor)
    assert queued["exit_status_counts"] == {"Pending": 3}
    assert queued["latest_attempt"]["pending_count"] == 3
    assert queued["latest_attempt"]["failure_counts"] == {}
    attempt(loaded, 0, 1)
    running = summary(loaded, failure_cursor)
    assert running["exit_status_counts"] == {"Pending": 2, "Running": 1}
    attempt(loaded, 0, 1, result={"exit_status": "VerifierError", "disposition": "failed"})
    failed_again = summary(loaded, failure_cursor)
    assert failed_again["exit_status_counts"] == {"Pending": 2, "VerifierError": 1}
    assert failed_again["latest_attempt"]["failure_counts"] == {"VerifierError": 1}
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == before


def test_progress_only_worker_failure_authorization_does_not_hide_new_attempt(cfg):
    from tests.test_limits_retry import configure, prepare
    from tests.test_progress_only_operator_retry import broken_attempt, authorize

    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item, work, old, context, failure = broken_attempt(cfg, run, loaded)
    authorize(cfg, run, item)
    queued = summary(loaded, {item["sample_key"]: failure})
    assert queued["exit_status_counts"] == {"Pending": 1}
    assert queued["latest_attempt"]["pending_count"] == 1
    fresh = work / "attempts" / (item["repetition_id"] + "-attempt-001")
    write_json(fresh / "attempt-started.json", {"started_at": "2099-01-01T00:00:00+00:00", "execution_attempt_id": "002"})
    assert summary(loaded, {item["sample_key"]: failure})["exit_status_counts"] == {"Running": 1}


def test_new_result_from_current_execution_wins_over_stale_pending_cursor(loaded):
    item = loaded["work_items"]["items"][0]
    attempt(loaded, 0, 0, result={"exit_status": "Submitted", "disposition": "accepted"})
    result = summary(loaded, {item["sample_key"]: {**item, "exit_status": "Pending", "disposition": "pending"}})
    assert result["exit_status_counts"] == {"Pending": 2, "Submitted": 1}
    assert result["pending_count"] == 2


def test_run_creation_persists_status_summary(loaded):
    document = read_json(loaded["run"] / "status-summary.json")
    assert document["schema_version"] == 1
    assert document["run_id"] == loaded["run"].name
    assert document["status"]["stage"] == "planned"
    assert document["sample_count"] == 3
    assert document["sample_summary"]["exit_status_counts"] == {"Pending": 3}
    assert document["summary_updated_at"]


def test_persisted_summary_uses_active_status_projection(loaded):
    attempt(loaded, 0, 0, execution="001",
            result={"exit_status": "Submitted", "disposition": "accepted"})
    write_json(loaded["run"] / "status.json", {"stage": "rollout", "updated_at": "status-time"})
    write_json(loaded["run"] / "execution/latest.json", {"attempt": "001"})
    document = status.refresh_run_status_summary(loaded["run"].parent.parent, loaded["run"].name)
    progress = read_json(loaded["run"] / "progress.json")
    expected = status._run_summary_projection(
        loaded, read_json(loaded["run"] / "status.json"), progress)
    assert document["sample_summary"] == expected["sample_summary"]
    assert document["sample_count"] == expected["sample_count"]
    assert document["status"] == expected["status"]
    assert document["sample_summary"]["exit_status_counts"] == {"Pending": 2, "Submitted": 1}


def test_rollout_progress_refreshes_summary_and_terminal_stage(loaded, cfg):
    from rollout.progress import Progress

    item = loaded["work_items"]["items"][0]
    physical = attempt(loaded, 0, 0, execution="001")
    write_json(loaded["run"] / "execution/latest.json", {"attempt": "001"})
    tracker = Progress(loaded["run"], loaded["config"], loaded["work_items"]["items"])
    context = {**item, "attempt_id": physical.name,
               "serving_assignment": {"attempt_id": physical.name,
                                          "execution_attempt_id": "001"}}
    tracker.started(context, None)
    document = read_json(loaded["run"] / "status-summary.json")
    assert document["sample_summary"]["exit_status_counts"] == {"Pending": 2, "Running": 1}

    tracker._last_status_summary_write = 0.0
    tracker.result(item["sample_key"], {
        **context, "exit_status": "Submitted", "disposition": "accepted"})
    document = read_json(loaded["run"] / "status-summary.json")
    assert document["sample_summary"]["exit_status_counts"] == {"Pending": 2, "Submitted": 1}

    status.update_run_status(loaded["run"], "complete")
    document = read_json(loaded["run"] / "status-summary.json")
    assert document["status"]["stage"] == "complete"


def test_run_status_matches_active_cli_despite_stale_progress(loaded, cfg, monkeypatch, capsys):
    import json
    import subprocess
    from core.cli import main

    run = loaded["run"]
    old = attempt(loaded, 0, 0, execution="001",
                  result={"exit_status": "OpenHandsError", "disposition": "failed"})
    current = attempt(loaded, 0, 1,
                      result={"exit_status": "Submitted", "disposition": "accepted",
                              "capture_status": "complete"})
    attempt(loaded, 0, 2, started=False)
    attempt(loaded, 1, 0)
    write_json(run / "execution/latest.json", {"attempt": "002"})
    write_json(run / "status.json", {"stage": "rollout"})
    item = loaded["work_items"]["items"][0]
    write_json(run / "progress.json", {
        "updated_at": "source-time", "accepted_samples": 4,
        "failed_samples": 1, "pending_samples": -2, "physical_attempts": 5,
        "samples": {item["sample_key"]: read_json(old / "attempt.json"),
                    "obsolete-sample": {"exit_status": "Submitted", "disposition": "accepted"}},
    })
    write_json(run / "rollout/phases.json", {"updated_at": "source-time"})
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="application.execution.supervisor " + run.name))
    main(["--config", cfg["config_source"], "run", "active-status"])
    active = json.loads(capsys.readouterr().out)["runs"][0]
    main(["--config", cfg["config_source"], "run", "status", run.name])
    detailed = json.loads(capsys.readouterr().out)
    assert detailed["sample_summary"] == active["sample_summary"]
    assert detailed["sample_summary"]["exit_status_counts"] == {"Pending": 1, "Running": 1, "Submitted": 1}
    progress = detailed["progress"]
    assert progress["planned_samples"] == 3
    assert progress["accepted_samples"] == 1
    assert progress["failed_samples"] == 0
    assert progress["pending_samples"] == 2
    assert progress["physical_attempts"] == 5
    assert progress["updated_at"] == "source-time"
    assert progress["samples"][item["sample_key"]]["attempt_id"] == current.name
    assert progress["samples"][item["sample_key"]]["capture_status"] == "complete"
    assert set(progress["samples"]) == {i["sample_key"] for i in loaded["work_items"]["items"]}
    assert {p: p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


def test_run_status_projects_authorized_retry_and_terminal_failures(loaded, cfg):
    from application.execution.failed_retry import request_failed_retry

    run = loaded["run"]
    first = loaded["work_items"]["items"][0]
    old = attempt(loaded, 0, 0, execution="001",
                  result={"exit_status": "OpenHandsError", "disposition": "failed"})
    attempt(loaded, 1, 0, result={"exit_status": "AgentCompleted", "disposition": "failed"})
    write_json(old.parent.parent / "result.json", read_json(old / "attempt.json"))
    write_json(run / "progress.json", {"samples": {first["sample_key"]: read_json(old / "attempt.json")}})
    request_failed_retry(cfg["project_root"], run.name, request_id="status-retry",
                         exit_statuses=["OpenHandsError"], sample_keys=[first["sample_key"]], reason="Retry sample")
    write_json(run / "execution/latest.json", {"attempt": "002"})
    result = status.run_status(cfg["project_root"], run.name)
    assert result["sample_summary"] == summary(loaded, read_json(run / "progress.json")["samples"])
    assert result["sample_summary"]["exit_status_counts"] == {"AgentCompleted": 1, "Pending": 2}
    assert result["progress"]["failed_samples"] == 1
    assert result["progress"]["accepted_samples"] == 0
    assert result["progress"]["pending_samples"] == 2


@pytest.mark.parametrize("cancelled, expected", [(True, "Cancelled"), (False, "Interrupted")])
def test_run_status_classifies_exited_unfinished_attempt_without_progress(loaded, cfg, cancelled, expected):
    run = loaded["run"]
    attempt(loaded, 0, 0, execution="001")
    write_json(run / "execution/latest.json", {"attempt": "002"})
    write_json(run / "execution/attempts/001/exit.json", {"exit_code": -15})
    if cancelled:
        write_json(run / "execution/attempts/001/termination.json", {"request_id": "stop"})
    (run / "progress.json").unlink(missing_ok=True)
    result = status.run_status(cfg["project_root"], run.name)
    assert result["sample_summary"]["exit_status_counts"] == {expected: 1, "Pending": 2}
    assert result["progress"]["planned_samples"] == 3
    assert result["progress"]["pending_samples"] == 2
