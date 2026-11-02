"""Run-local status and cancellation operations."""

from __future__ import annotations

from core.io import now, read_json, write_json
from pathlib import Path
from core.locking import file_lock
from uuid import uuid4

from .run_plan import load_plan, list_runs, run_path


_FAILURE_DISPOSITIONS = {"failed", "blocked", "retry_pending"}
_NON_FAILURE_EXIT_STATUSES = {
    "Submitted", "AgentCompleted", "Resolved", "Complete", "completed",
    "Running", "pending", "queued", "unknown",
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
            if isinstance(sample.get("serving_assignment"), dict)
            and str(sample["serving_assignment"].get("execution_attempt_id"))
            == str(latest_attempt_id)
        ]
    return {
        **summarize(rows),
        "latest_attempt": summarize(latest_rows, latest_attempt_id, include_attempt_id=True),
    }


def _latest_execution_attempt(run):
    path = run / "execution" / "latest.json"
    if not path.exists():
        return None
    latest = read_json(path)
    return latest.get("attempt") if isinstance(latest, dict) else None


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
                      "hidden_publication":run/"hidden/publication.json"}.items():
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
    result["sample_summary"] = _sample_summary(
        (result.get("progress") or {}).get("samples") or {},
        _latest_execution_attempt(run),
    )
    return {
        "run_id": result["run_id"],
        "sample_summary": result["sample_summary"],
        **{key: value for key, value in result.items() if key not in {"run_id", "sample_summary"}},
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
def active_run_statuses(root):
    """Return compact summaries for live and queued runs."""
    import subprocess
    from collections import Counter
    from runtime.containers.dependencies import failure_status
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
        samples = progress.get("samples") or {}
        sample_summary = _sample_summary(samples, _latest_execution_attempt(run))
        exit_counts = Counter(str(x.get("exit_status", "unknown")) for x in samples.values())
        disposition_counts = Counter(str(x.get("disposition", "unknown")) for x in samples.values())
        task_count = len(loaded["tasks"]["tasks"])
        completed = sum(exit_counts.get(x, 0) for x in ("Submitted", "Resolved", "Complete", "completed"))
        failure_statuses = {"failed", "Failure", "ConnectionError", "ParallelSelectionError",
                            "ContextWindowExceededError", "LimitsExceeded", "infra_error", "error"}
        error_samples = sum(1 for sample in samples.values() if sample.get("error"))
        capture_error_samples = sum(1 for sample in samples.values() if sample.get("capture_errors"))
        failure_counts = {key: count for key, count in exit_counts.items() if key in failure_statuses}
        environment = failure_status(root, run_id)
        environment_count = int(environment.get("count", 0)) if isinstance(environment, dict) else 0
        anomaly = {
            "detected": bool(failure_counts or error_samples or capture_error_samples or environment_count),
            "failure_status_counts": dict(sorted(failure_counts.items())),
            "sample_error_count": error_samples,
            "capture_error_sample_count": capture_error_samples,
            "environment_failure_count": environment_count,
        }
        active.append({"run_id": run_id, "operation": loaded["plan"].get("operation"), "task_count": task_count,
                       "status": status, "progress_updated_at": progress.get("updated_at"), "sample_count": len(samples),
                       "sample_summary": sample_summary,
                       "completed_count": completed, "pending_count": max(task_count-completed, 0),
                       "exit_status_counts": dict(sorted(exit_counts.items())),
                       "disposition_counts": dict(sorted(disposition_counts.items())), "anomalies": anomaly})
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
