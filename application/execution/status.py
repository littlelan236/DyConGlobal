"""Run-local status and cancellation operations."""

from __future__ import annotations

from core.io import now, read_json, write_json
from core.locking import file_lock
from uuid import uuid4

from .run_plan import load_plan


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
    return result


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
