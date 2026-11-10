"""Run-local status and cancellation operations."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from core.io import now, read_json, update_status, write_json
from core.locking import file_lock

from .run_plan import list_runs, load_plan, run_path

_FAILURE_DISPOSITIONS = {"failed", "blocked", "retry_pending"}
_NON_FAILURE_EXIT_STATUSES = {
    "Submitted", "AgentCompleted", "Resolved", "Complete", "completed",
    "Running", "Pending", "pending", "queued", "unknown",
}


def _sample_summary(samples, latest_attempt_id):
    rows = list(samples.values()) if isinstance(samples, dict) else []

    def summarize(selected, attempt_id=None, include_attempt_id=False):
        dispositions = [str(sample.get("disposition", "unknown")) for sample in selected]
        failures = {}
        for sample, disposition in zip(selected, dispositions):
            exit_status = str(sample.get("exit_status") or "unknown")
            if disposition in _FAILURE_DISPOSITIONS or exit_status not in _NON_FAILURE_EXIT_STATUSES:
                failures[exit_status] = failures.get(exit_status, 0) + 1
        summary = {
            "execution_attempt_id": attempt_id,
            "sample_count": len(selected),
            "accepted_count": dispositions.count("accepted"),
            "pending_count": dispositions.count("pending"),
            "retry_pending_count": dispositions.count("retry_pending"),
            "failure_counts": dict(sorted(failures.items())),
        }
        if not include_attempt_id:
            summary.pop("execution_attempt_id")
        return summary

    latest_rows = []
    if latest_attempt_id is not None:
        latest_rows = [
            sample for sample in rows
            if str(_sample_execution_attempt(sample))
            == str(latest_attempt_id)
        ]
    return {
        **summarize(rows),
        "latest_attempt": summarize(latest_rows, latest_attempt_id, include_attempt_id=True),
    }


def _sample_execution_attempt(sample):
    return (sample.get("_queued_execution_attempt_id")
            or (sample.get("serving_assignment") or {}).get("execution_attempt_id"))


def _latest_execution_attempt(run):
    path = run / "execution" / "latest.json"
    if not path.exists():
        return None
    latest = read_json(path)
    return latest.get("attempt") if isinstance(latest, dict) else None


def _active_sample_rows(loaded, progress, latest_attempt_id):
    """Project current sample state; an authorized queue supersedes old attempts."""
    import re

    from .failed_retry import retry_index, superseded, worker_failure_authorization

    run = loaded["run"]
    samples = progress.get("samples") or {}
    retries = retry_index(run)

    def pending(item):
        return {**item, "exit_status": "Pending", "disposition": "pending",
                "_queued_execution_attempt_id": latest_attempt_id}

    floors_path = run / "rollout/replan-attempt-floors.json"
    floors = read_json(floors_path) if floors_path.exists() else {}
    rows = {}
    for item in loaded["work_items"]["items"]:
        key = item["sample_key"]
        row = {**item, "exit_status": "Pending", "disposition": "pending",
               **samples.get(key, {})}
        work = run / "rollout/tasks" / item["instance_id"] / "repetitions" / item["repetition_id"]
        attempts = work / "attempts"
        pattern = re.compile(re.escape(item["repetition_id"]) + r"-attempt-(\d+)$")
        floor = floors.get(item["instance_id"] + "/" + item["repetition_id"], -1)
        candidates = []
        if attempts.exists():
            for path in attempts.iterdir():
                match = pattern.fullmatch(path.name)
                if (match and int(match.group(1)) > floor and path.is_dir()
                        and ((path / "attempt.json").exists() or (path / "attempt-started.json").exists())):
                    candidates.append((int(match.group(1)), path))
        if not candidates:
            rows[key] = (pending(item) if row.get("exit_status") == "Pending"
                         or superseded(retries, key, row) else row)
            continue

        _, attempt = max(candidates)
        result_path = attempt / "attempt.json"
        started_path = attempt / "attempt-started.json"
        assignment_path = attempt / "serving-assignment.json"
        started = read_json(started_path) if started_path.exists() else {}
        assignment = read_json(assignment_path) if assignment_path.exists() else {}
        # A live progress row may still carry the preceding attempt's assignment.
        if not assignment and row.get("attempt_id") == attempt.name:
            previous_assignment = row.get("serving_assignment") or {}
            if previous_assignment.get("attempt_id", attempt.name) == attempt.name:
                assignment = previous_assignment
        execution_id = started.get("execution_attempt_id") or assignment.get("execution_attempt_id")
        result = read_json(result_path) if result_path.exists() else None
        if result is not None and superseded(retries, key, result):
            rows[key] = pending(item)
            continue
        authorization = worker_failure_authorization(retries, key)
        if result is None and authorization is not None:
            from datetime import datetime
            started_at = started.get("started_at")
            if (started_at is not None
                    and datetime.fromisoformat(started_at)
                    <= datetime.fromisoformat(authorization["requested_at"])
                    and (execution_id is None or authorization["previous_execution"] is None
                         or int(execution_id) <= int(authorization["previous_execution"]))):
                rows[key] = pending(item)
                continue
        cursor = samples.get(key) or {}
        if (cursor.get("exit_status") == "Pending" and cursor.get("attempt_id") is None
                and (execution_id is None or str(execution_id) != str(latest_attempt_id))):
            # Resume has explicitly put this sample back in its current queue.
            # Old cancelled/failed physical attempts are history, not queue state.
            rows[key] = pending(item)
            continue
        if result_path.exists():
            row = {**item, **result}
            if execution_id is None:
                execution_id = (row.get("serving_assignment") or {}).get("execution_attempt_id")
        elif row.get("attempt_id") != attempt.name or row.get("exit_status") in {"Pending", "Running"}:
            execution = run / "execution/attempts" / str(execution_id)
            termination_path = execution / "termination.json"
            termination = read_json(termination_path) if termination_path.exists() else {}
            if (execution / "exit.json").exists():
                # Match the rollout recovery classification, without recovering or writing.
                cancelled = termination.get("request_id") is not None
                status = "Cancelled" if cancelled else "Interrupted"
                disposition = "interrupted" if cancelled else "infrastructure_retry"
            elif execution_id is not None and str(execution_id) == str(latest_attempt_id):
                status, disposition = "Running", "pending"
            else:
                status, disposition = "unknown", "unknown"
            row = {**item, "exit_status": status, "disposition": disposition}
        row["attempt_id"] = attempt.name
        row["serving_assignment"] = {**assignment, "execution_attempt_id": execution_id}
        rows[key] = row
    return rows


def _active_sample_summary(samples, latest_attempt_id):
    from collections import Counter

    def counts(rows):
        return dict(sorted(Counter(str(row.get("exit_status") or "unknown") for row in rows).items()))

    latest = _sample_summary(samples, latest_attempt_id)["latest_attempt"]
    latest_rows = [row for row in samples.values()
                   if latest_attempt_id is not None
                   and str(_sample_execution_attempt(row))
                   == str(latest_attempt_id)]
    overall = _sample_summary(samples, latest_attempt_id)
    overall.pop("latest_attempt")
    return {**overall, "exit_status_counts": counts(samples.values()),
            "latest_attempt": {**latest, "exit_status_counts": counts(latest_rows)}}


def _sample_status_projection(loaded, progress):
    """Share current sample rows and counts across every run status view."""
    latest_attempt_id = _latest_execution_attempt(loaded["run"])
    samples = _active_sample_rows(loaded, progress, latest_attempt_id)
    return samples, _active_sample_summary(samples, latest_attempt_id)


def run_status(root, run_id):
    loaded = load_plan(root, run_id)
    run = loaded["run"]
    result = {"run_id": run_id, "plan": loaded["plan"], "task_count": len(loaded["tasks"]["tasks"])}
    for name in ("status.json", "progress.json", "artifacts.json"):
        path = run / name
        if path.exists():
            result[name.removesuffix(".json")] = read_json(path)
    stages={}
    for name,path in {"harness":run/"harness/status.json", "online_coverage":run/"enrichment/online/coverage.json",
                      "hidden_publication":run/"hidden/publication.json",
                      "capture_publication":run/"capture-publication.json"}.items():
        if path.exists():
            stages[name] = read_json(path)
    result["stages"]=stages
    if loaded["plan"]["operation"]=="harness" and "harness" in stages:
        result["status"]=stages["harness"]
    attempts = run / "execution" / "attempts"
    result["attempts"] = sorted(p.name for p in attempts.iterdir()) if attempts.exists() else []
    from rollout.retry import count_projection, sampling_notice
    phases = run / "rollout/phases.json"
    if phases.exists():
        result["rollout_phases"] = read_json(phases)
        result.setdefault("progress", {}).update(result["rollout_phases"])
    result["rollout_counts"] = count_projection(run)
    result["sampling_notice"] = sampling_notice(loaded["config"])
    from runtime.containers.dependencies import failure_status
    result['environment_failures'] = failure_status(root, run_id)
    from .supervisor import cancellation_status
    result["cancellation"] = cancellation_status(run)
    workers = run / "serving/worker-pool.json"
    if workers.exists():
        result["serving_workers"] = read_json(workers)
    progress = result.get("progress") or {}
    samples, result["sample_summary"] = _sample_status_projection(loaded, progress)
    summary = result["sample_summary"]
    result["progress"] = {
        **progress,
        "samples": samples,
        "planned_samples": summary["sample_count"],
        "accepted_samples": summary["accepted_count"],
        "failed_samples": sum(row.get("disposition") in {"failed", "blocked"}
                              for row in samples.values()),
        "pending_samples": summary["pending_count"],
    }
    return {
        "run_id": result["run_id"],
        "sample_summary": result["sample_summary"],
        **{key: value for key, value in result.items() if key not in {"run_id", "sample_summary"}},
    }



def running_step_status(root, run_id):
    """Return the latest recorded agent step for each task with a running sample."""
    from datetime import datetime, timezone

    loaded = load_plan(root, run_id)
    run = loaded["run"]
    progress_path = run / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    samples = progress.get("samples") or {}
    if isinstance(samples, list):
        samples = {str(index): sample for index, sample in enumerate(samples)}

    run_root = run.resolve()
    rows_by_task = {}
    for sample in samples.values():
        if not isinstance(sample, dict) or sample.get("exit_status") != "Running":
            continue
        task_id = sample.get("instance_id") or sample.get("task_id") or sample.get("task_key")
        if not task_id:
            continue
        repetition_id = str(sample.get("repetition_id") or "")
        attempt_id = str(sample.get("attempt_id") or "")
        attempt = (run / "rollout" / "tasks" / str(task_id) / "repetitions" /
                   repetition_id / "attempts" / attempt_id)
        try:
            resolved_attempt = attempt.resolve()
            if not resolved_attempt.is_relative_to(run_root):
                continue
        except (OSError, RuntimeError):
            continue

        activity_mtime = 0.0
        started = attempt / "attempt-started.json"
        try:
            if started.exists():
                activity_mtime = started.stat().st_mtime
            elif attempt.exists():
                activity_mtime = attempt.stat().st_mtime
        except OSError:
            pass

        latest_request = None
        latest_request_key = (0.0, -1, "")
        requests = attempt / "requests"
        if requests.exists():
            for request_path in requests.glob("req-*.json"):
                try:
                    mtime = request_path.stat().st_mtime
                    request = read_json(request_path)
                except (OSError, ValueError, TypeError):
                    continue
                activity_mtime = max(activity_mtime, mtime)
                if request.get("role") != "agent_decision" or request.get("step_id") is None:
                    continue
                try:
                    step_value = int(request["step_id"])
                except (TypeError, ValueError):
                    continue
                key = (mtime, step_value, request_path.name)
                if key > latest_request_key:
                    latest_request_key = key
                    latest_request = request

        trajectory_dir = attempt / "benchflow"
        if trajectory_dir.exists():
            for trajectory in trajectory_dir.glob("*/trajectory/acp_trajectory.jsonl"):
                try:
                    activity_mtime = max(activity_mtime, trajectory.stat().st_mtime)
                except OSError:
                    continue

        request = latest_request or {}
        step_id = request.get("step_id")
        try:
            step_count = int(step_id) + 1 if step_id is not None and int(step_id) >= 0 else None
        except (TypeError, ValueError):
            step_count = None
        row = {
            "instance_id": str(task_id),
            "task_key": sample.get("task_key"),
            "sample_key": sample.get("sample_key"),
            "repetition_id": sample.get("repetition_id"),
            "repetition_index": sample.get("repetition_index"),
            "attempt_id": sample.get("attempt_id"),
            "step_id": step_id,
            "step_count": step_count,
            "request_id": request.get("request_id"),
            "request_status": request.get("status"),
            "request_started_at": request.get("started_at"),
            "request_finished_at": request.get("finished_at"),
            "attempt_updated_at": (datetime.fromtimestamp(activity_mtime, timezone.utc).isoformat()
                                    if activity_mtime else None),
        }
        candidate_key = (activity_mtime, latest_request_key, repetition_id, attempt_id)
        previous = rows_by_task.get(str(task_id))
        if previous is None or candidate_key > previous[0]:
            rows_by_task[str(task_id)] = (candidate_key, row)

    rows = [value[1] for value in rows_by_task.values()]
    rows.sort(key=lambda row: row["instance_id"])
    return {
        "run_id": run_id,
        "progress_updated_at": progress.get("updated_at"),
        "running_task_count": len(rows),
        "tasks": rows,
    }

def cancel_run(root, run_id, reason):
    loaded = load_plan(root, run_id)
    run = loaded["run"]
    path = run / "execution/cancel-request.json"
    with file_lock(run / "execution/supervisor.lock"):
        latest = run / "execution/latest.json"
        attempt_id = read_json(latest)["attempt"] if latest.exists() else None
        if attempt_id is not None and (run / "execution/attempts" / attempt_id / "exit.json").exists():
            return {"run_id": run_id, "cancel_requested": False, "already_terminal": True}
        status = run / "status.json"
        if status.exists() and read_json(status).get("stage") in {"complete", "failed", "cancelled", "validated"}:
            # A live supervisor can still be finishing resources after a runner status write.
            if attempt_id is None:
                return {"run_id": run_id, "cancel_requested": False, "already_terminal": True}
        if path.exists() and read_json(path).get('execution_attempt_id') not in (None, attempt_id):
            history = run / 'execution/cancel-history'
            history.mkdir(parents=True, exist_ok=True)
            path.replace(history / (uuid4().hex + '.json'))
        if not path.exists():
            write_json(path, {"run_id": run_id, "reason": reason, "requested_at": now(),
                              "request_id": uuid4().hex, "execution_attempt_id": attempt_id})
        request = read_json(path)
    return {"run_id": run_id, "cancel_requested": True, "path": str(path),
            "request_id": request.get("request_id"), "execution_attempt_id": request.get("execution_attempt_id")}


def reconcile_run(root, run_id):
    loaded = load_plan(root, run_id)
    from .supervisor import reconcile_cancelled
    cancellation = reconcile_cancelled(root, run_id)
    return {
        "run_id": run_id,
        "compatible": True,
        "config_hash": loaded["plan"]["config_hash"],
        "selection_sha256": loaded["plan"]["selection_sha256"],
        "task_count": len(loaded["tasks"]["tasks"]),
        "cancellation": cancellation,
    }
def _run_summary_projection(loaded, status, progress):
    # Shared single-run projection for active-status and persisted snapshots.
    run = loaded["run"]
    samples, summary = _sample_status_projection(loaded, progress)
    return {
        "run_id": run.name,
        "operation": loaded["plan"].get("operation"),
        "task_count": len(loaded["tasks"]["tasks"]),
        "status": status,
        "progress_updated_at": progress.get("updated_at"),
        "sample_count": len(samples),
        "sample_summary": summary,
    }


def refresh_run_status_summary(root, run_id):
    # Atomically refresh a derived view; source evidence remains read-only.
    root = Path(root).resolve()
    run = run_path(root, run_id)
    error_path = run / "status-summary-error.json"
    try:
        with file_lock(run / "status-summary.lock"):
            loaded = load_plan(root, run_id)
            status_path = run / "status.json"
            status = read_json(status_path) if status_path.exists() else {}
            progress_path = run / "progress.json"
            progress = read_json(progress_path) if progress_path.exists() else {}
            document = {
                "schema_version": 1,
                "summary_updated_at": now(),
                **_run_summary_projection(loaded, status, progress),
            }
            write_json(run / "status-summary.json", document)
            error_path.unlink(missing_ok=True)
            return document
    except Exception as exc:
        try:
            write_json(error_path, {"run_id": run.name, "at": now(), "error": repr(exc)})
        except Exception:
            pass
        return None


def update_run_status(run, stage, **values):
    run = Path(run)
    update_status(run / "status.json", stage, **values)
    refresh_run_status_summary(run.parent.parent, run.name)
    return read_json(run / "status.json")


def active_run_statuses(root):
    """Return compact summaries for live and queued runs."""
    import subprocess
    root = Path(root).resolve()
    active = []
    processes = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=False).stdout
    terminal = {"complete", "failed", "cancelled", "validated"}
    for run_id in list_runs(root):
        run = run_path(root, run_id)
        status_path = run / "status.json"
        if not status_path.exists():
            continue
        status = read_json(status_path)
        if status.get("stage") in terminal or "application.execution.supervisor" not in processes or run_id not in processes:
            continue
        loaded = load_plan(root, run_id)
        progress_path = run / "progress.json"
        progress = read_json(progress_path) if progress_path.exists() else {}
        active.append(_run_summary_projection(loaded, status, progress))
    queued = []
    queue_root = root / "state" / "run-resume-queue"
    if queue_root.exists():
        for status_path in sorted(queue_root.glob("*/status.json")):
            try:
                qs = read_json(status_path); stage = qs.get("stage")
                if stage not in {"queued", "waiting_gpu_exit", "launching"}: continue
                rp = status_path.parent / "request.json"; req = read_json(rp) if rp.exists() else {}
                queued.append({"request_id": qs.get("request_id") or req.get("request_id"), "run_id": req.get("run_id"),
                               "after_run": req.get("after_run"), "stage": stage,
                               "queued_at": qs.get("queued_at") or req.get("queued_at"), "updated_at": qs.get("at")})
            except (OSError, ValueError, TypeError):
                continue
    return {"schema_version": 2, "count": len(active), "runs": active, "queued_count": len(queued), "queued_runs": queued}
