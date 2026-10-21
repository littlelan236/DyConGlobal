"""Run-local status and cancellation operations."""

from __future__ import annotations


from core.io import now, read_json, write_json

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
        if path.exists():stages[name]=read_json(path)
    result["stages"]=stages
    if loaded["plan"]["operation"]=="harness" and "harness" in stages:
        result["status"]=stages["harness"]
    attempts = run / "execution" / "attempts"
    result["attempts"] = sorted(p.name for p in attempts.iterdir()) if attempts.exists() else []
    return result


def cancel_run(root, run_id, reason):
    loaded = load_plan(root, run_id)
    path = loaded["run"] / "execution" / "cancel-request.json"
    write_json(path, {"run_id": run_id, "reason": reason, "requested_at": now()})
    return {"run_id": run_id, "cancel_requested": True, "path": str(path)}


def reconcile_run(root, run_id):
    loaded = load_plan(root, run_id)
    return {
        "run_id": run_id,
        "compatible": True,
        "config_hash": loaded["plan"]["config_hash"],
        "selection_sha256": loaded["plan"]["selection_sha256"],
        "task_count": len(loaded["tasks"]["tasks"]),
    }
