"""Formal Harbor profile, preflight and official control entrypoints."""
from __future__ import annotations

from pathlib import Path

from application.execution.run_plan import load_plan
from benchmark_evaluation.harbor import trial_outcome
from core.io import hash_json, identifier, read_json, sha256, write_json
from runtime.backends.harbor.adapter import worker_request
from runtime.backends.registry import adapter_for


def create_profile(cfg, profile_id, dependency_snapshot, bridge_address="127.0.0.1"):
    from runtime.agents.dependencies import verify_snapshot
    from runtime.agents.openhands import OpenHandsDriverProfile
    dependency_snapshot = Path(dependency_snapshot).resolve()
    verify_snapshot(dependency_snapshot)
    profile = OpenHandsDriverProfile(bridge_listen=bridge_address, bridge_advertise=bridge_address,
        dependency_snapshot=str(dependency_snapshot),
        dependency_snapshot_sha256=sha256(dependency_snapshot / "manifest.json")).model_dump()
    path = Path(cfg["project_root"]) / "runtime/agent-profiles" / identifier(profile_id) / "manifest.json"
    if path.exists() and read_json(path) != profile:
        raise FileExistsError("Harbor profile identity is immutable")
    write_json(path, profile)
    return {"profile_ref": path.relative_to(cfg["project_root"]).as_posix(), "profile_sha256": sha256(path)}


def preflight(root, run_id):
    loaded = load_plan(root, run_id)
    adapter = adapter_for(loaded["config"])
    if adapter.binding.backend_family != "harbor":
        raise ValueError("This preflight requires a frozen Harbor run")
    context = adapter.open_run(loaded["config"], run_id)
    try:
        facts = adapter.preflight(context, loaded["config"]["selection"]["tasks"])
        return {"passed": facts.passed, "blockers": facts.blockers, "receipt": facts.receipt}
    finally:
        adapter.close_run(context)


def controls(root, run_id, control_id, agent, task_ids=()):
    loaded = load_plan(root, run_id)
    cfg = loaded["config"]
    adapter = adapter_for(cfg)
    if adapter.binding.backend_family != "harbor":
        raise ValueError("Official Harbor controls require a frozen Harbor run")
    directory = loaded["run"] / "validation/harbor-controls" / identifier(control_id)
    if directory.exists():
        raise FileExistsError("Control evidence is immutable; use another control ID")
    selected = {key: task for key, task in cfg["selection"]["tasks"].items()
                if not task_ids or task["instance_id"] in task_ids}
    if not selected or set(task_ids) - {t["instance_id"] for t in selected.values()}:
        raise ValueError("Control tasks must belong to the frozen run")
    context = adapter.open_run(cfg, run_id)
    rows = {}
    try:
        context.factory.lock_images([t["instance"] for t in selected.values()])
        for key, task in selected.items():
            output = directory / key
            output.mkdir(parents=True)
            binding = {"run_id": run_id, "task_key": key, "sample_key": "control-" + key,
                       "attempt_id": identifier(control_id)}
            operation = agent if agent in {"oracle", "nop"} else "replay"
            request = worker_request(cfg, run_id, context.factory, task, binding, output, operation)
            if operation == "replay":
                if task["benchmark"] != "swebench_pro_v2":
                    raise ValueError("Patch replay controls apply only to Pro V2")
                evaluator = cfg["execution_context"]["evaluator"]
                request["evaluator"] = {**evaluator, "path": str(Path(root) / evaluator["path"])}
                patch = Path(task["instance"]["task_path"]) / "solution/gold_patch.diff"
                if agent == "empty-replay":
                    patch = output / "empty.patch"
                    patch.write_bytes(b"")
                request["exact_subject"] = {"path": str(patch), "sha256": sha256(patch),
                                             "instance_id": task["instance_id"]}
            response = context.factory.execute_worker(request, seconds=cfg["timeouts"]["container_lifetime_seconds"])
            rows[key] = trial_outcome(response["result"], task=task, attempt_id=control_id)
            rows[key]["model_requests"] = 0
        write_json(directory / "summary.json", {"agent": agent, "tasks": rows,
                   "control_sha256": hash_json(rows), "scope": "official controls; no agent rollout evidence"})
        return {"agent": agent, "tasks": rows, "receipt": str(directory / "summary.json")}
    finally:
        adapter.close_run(context)
