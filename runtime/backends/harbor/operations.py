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


def controls(root, run_id, control_id, agent, task_ids=(), *, reference_ref=None):
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
    reference = None
    reference_tasks = {}
    if reference_ref:
        if agent != 'oracle' or len(selected) != 1:
            raise ValueError('A derived reference control requires one oracle task')
        from runtime.backends.harbor.reference_solutions import reference_task
        for key, official in selected.items():
            reference_tasks[key], reference = reference_task(root, reference_ref, official)
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
            if reference:
                request['task'] = reference_tasks[key]
                request['reference_revision'] = reference
                write_json(output / 'reference-revision.json', reference)
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
            from runtime.backends.harbor.policy import phase_budgets
            budgets = phase_budgets(cfg, task["instance"]["metadata"])
            seconds = sum(budgets.values()) if budgets else cfg["timeouts"]["container_lifetime_seconds"]
            response = context.factory.execute_worker(request, seconds=seconds)
            rows[key] = trial_outcome(response["result"], task=task, attempt_id=control_id,
                                      evidence_dir=output, operation=operation)
            rows[key]["model_requests"] = 0
        write_json(directory / "summary.json", {"agent": agent, "tasks": rows,
                   "control_sha256": hash_json(rows), "scope": "derived reference controls; unchanged official verifier" if reference else "official controls; no agent rollout evidence",
                   **({"reference_revision": reference} if reference else {})})
        return {"agent": agent, "tasks": rows, "receipt": str(directory / "summary.json")}
    finally:
        adapter.close_run(context)


def audit_controls(root, run_id, control_id, audit_id):
    """Reclassify saved evidence into a new receipt; never rerun or edit controls."""
    from collections import Counter
    from core.io import now
    from benchmark_evaluation.harbor_verifier import POLICY

    loaded = load_plan(root, run_id)
    cfg, run = loaded["config"], loaded["run"]
    if cfg["benchmark"]["backend"] != "terminalbench_2_1":
        raise ValueError("This audit requires a frozen Terminal-Bench run")
    source = run / "validation/harbor-controls" / identifier(control_id)
    summary_path = source / "summary.json"
    before = sha256(summary_path)
    original = read_json(summary_path)
    if hash_json(original["tasks"]) != original["control_sha256"]:
        raise ValueError("Control summary hash mismatch")
    target = run / "validation/harbor-controls-audits" / identifier(audit_id)
    if target.exists():
        raise FileExistsError("Audit evidence is immutable; use another audit ID")
    rows, inputs, changes = {}, [], []
    selected = cfg["selection"]["tasks"]
    for key, previous in original["tasks"].items():
        if key not in selected or selected[key]["instance_id"] != previous["task"]:
            raise ValueError("Control task differs from frozen selection")
        path = source / key / "worker-response.json"
        digest = sha256(path)
        response = read_json(path)
        binding = response.get("binding", {})
        if (binding.get("run_id") != run_id or binding.get("task_key") != key
                or binding.get("attempt_id") != control_id):
            raise ValueError("Worker response binding differs from control")
        row = trial_outcome(response["result"], task=selected[key], attempt_id=control_id,
                            evidence_dir=path.parent, operation=original["agent"])
        if row["trial_result_sha256"] != previous["trial_result_sha256"]:
            raise ValueError("Worker result hash differs from original control summary")
        row["model_requests"] = 0
        rows[key] = row
        inputs.append({"path": str(path.relative_to(run)), "sha256": digest})
        if row["resolved"] != previous["resolved"] or row["status"] != previous["status"]:
            changes.append({"task": row["task"], "before_status": previous["status"],
                            "after_status": row["status"], "reason_code": (row["verifier_audit"] or {}).get("reason_code")})
    if sha256(summary_path) != before:
        raise RuntimeError("Source control changed during audit")
    for item in inputs:
        if sha256(run / item["path"]) != item["sha256"]:
            raise RuntimeError("Worker evidence changed during audit")
    for key, row in rows.items():
        for item in (row["verifier_audit"] or {}).get("sources", []):
            if sha256(source / key / item["path"]) != item["sha256"]:
                raise RuntimeError("Verifier evidence changed during audit")
    counts = {"total": len(rows), "passed": sum(r["resolved"] is True for r in rows.values()),
              "valid_zero_or_partial": sum(r["resolved"] is False for r in rows.values()),
              "VerifierError": sum(r["status"] == "infra_error" for r in rows.values()),
              "Review": sum(r["status"] == "unknown" for r in rows.values())}
    reasons = dict(Counter((r["verifier_audit"] or {}).get("reason_code") for r in rows.values()
                           if r["verifier_audit"]))
    report = {"schema_version": 1, "policy": POLICY, "created_at": now(), "run_id": run_id,
              "control_id": control_id, "audit_id": audit_id, "agent": original["agent"],
              "source_summary": {"path": str(summary_path.relative_to(run)), "sha256": before},
              "worker_inputs": inputs, "counts": counts, "reasons": reasons, "changes": changes,
              "tasks": rows, "control_sha256": hash_json(rows),
              "scope": "saved evidence reclassification; no model, container or verifier execution"}
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "summary.json", report)
    return {"run_id": run_id, "control_id": control_id, "counts": counts, "reasons": reasons,
            "changed": len(changes), "receipt": str(target / "summary.json")}
