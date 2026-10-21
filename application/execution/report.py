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
    return result
