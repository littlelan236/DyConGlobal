"""Run-local report projection."""

from __future__ import annotations

from core.io import read_json

from .run_plan import load_plan


def report(root, run_id):
    loaded = load_plan(root, run_id)
    run = loaded["run"]
    result = {"run_id": run_id, "operation": loaded["plan"]["operation"]}
    for name in ("status.json", "progress.json", "harness/summary.json", "artifacts.json"):
        path = run / name
        if path.exists():
            result[name.replace("/", "_").removesuffix(".json")] = read_json(path)
    result["paths"] = {"run": str(run), "tasks": str(run / "tasks.json")}
    from rollout.retry import count_projection, sampling_notice
    phases = run / "rollout/phases.json"
    if phases.exists():
        result["rollout_phases"] = read_json(phases)
        result.setdefault("progress", {}).update(result["rollout_phases"])
    result["rollout_counts"] = count_projection(run)
    result["sampling_notice"] = sampling_notice(loaded["config"])
    from rollout.retry import attempt_usage
    result["all_attempt_usage"] = attempt_usage(run)
    from runtime.containers.dependencies import failure_status
    result['environment_failures'] = failure_status(root, run_id)
    from .supervisor import cancellation_status
    result["cancellation"] = cancellation_status(run)
    workers = run / "serving/worker-pool.json"
    if workers.exists():
        result["serving_workers"] = read_json(workers)
    return result
