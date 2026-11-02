"""Run-owned preparation and post-hoc official WebArena Verified evaluation."""
from __future__ import annotations

import concurrent.futures
import copy
from pathlib import Path

from application.execution.evidence import RunEvidence
from application.execution.run_plan import load_plan
from application.execution.subject import verify_subject
from core.io import hash_json, identifier, now, owned_path, read_json, update_status, write_json
from core.locking import exclusive_evaluation


def _verify_selection(root, selection, receipt):
    if hash_json(selection) != receipt["selection_sha256"]:
        raise ValueError("WebArena harness selection changed after prepare")
    if sorted(selection["rollout_ids"]) != sorted(selection["rollouts"]):
        raise ValueError("WebArena harness rollout IDs disagree with selection rows")
    for row in selection["rollouts"].values():
        if row.get("evaluation_subject", {}).get("kind") != "web_trace":
            raise ValueError("WebArena harness requires web_trace subjects")
        if row.get("submission_status") != "submitted":
            raise ValueError("WebArena harness requires an actual agent submission")
        verify_subject(root, row)
        subject = read_json(owned_path(root, row["subject_path"]))
        if not subject.get("complete") or not subject.get("har_flushed"):
            raise ValueError("WebArena harness refuses incomplete browser evidence")
        task = selection["tasks"][row["task_key"]]
        if task["benchmark"] != "webarena_verified":
            raise ValueError("WebArena harness task belongs to another benchmark")


def prepare(root, run_id, rollout_ids=(), source_run=None, incremental=False):
    loaded = load_plan(root, run_id)
    root = Path(root).resolve()
    source_run = source_run or run_id
    target = loaded["run"] / "harness/selection.json"
    if target.exists() and not incremental:
        selection, receipt = read_json(target), read_json(target.with_name("selection-receipt.json"))
        _verify_selection(root, selection, receipt)
        if (selection["source_run_id"] != source_run
                or rollout_ids and sorted(rollout_ids) != sorted(selection["rollout_ids"])):
            raise ValueError("WebArena harness selection is frozen; use a new run")
        return {"run_id": run_id, "rollout_count": len(selection["rollout_ids"]), "selection": str(target)}
    evidence = RunEvidence(root, source_run)
    selected = evidence.select(rollout_ids)
    allowed_tasks = {task["task_key"] for task in loaded["tasks"]["tasks"]}
    rows = {rid: row for rid, row in selected["rollouts"].items()
            if not incremental or row.get("evaluation_status") not in {"completed", "running"}}
    if not rollout_ids:
        # A mixed run may contain failed/unsubmitted canonical attempts. Keep
        # them in the run report denominator, but never send them to evaluator.
        rows = {rid: row for rid, row in rows.items() if row.get("submission_status") == "submitted"}
    if not rows:
        raise ValueError("WebArena harness selection has no eligible submitted trajectories")
    if any(row["task_key"] not in allowed_tasks for row in rows.values()):
        raise ValueError("WebArena harness selection crosses the frozen run tasks")
    selection = {"schema_version": 1, "benchmark": "webarena_verified", "run_id": run_id,
                 "source_run_id": source_run, "evidence_revision": selected["evidence_revision"],
                 "evidence_sha256": selected["evidence_sha256"], "rollout_ids": sorted(rows), "rollouts": rows,
                 "tasks": {row["task_key"]: selected["tasks"][row["task_key"]] for row in rows.values()}}
    receipt = {"selection_sha256": hash_json(selection), "incremental": bool(incremental)}
    _verify_selection(root, selection, receipt)
    write_json(target, selection)
    write_json(target.with_name("selection-receipt.json"), receipt)
    return {"run_id": run_id, "rollout_count": len(rows), "selection": str(target)}


def run(root, run_id, eval_id, gold=False, incremental=False):
    if gold:
        raise ValueError("WebArena Verified has no executable gold trajectory; static fixtures are offline tests")
    loaded = load_plan(root, run_id)
    selection = read_json(loaded["run"] / "harness/selection.json")
    receipt = read_json(loaded["run"] / "harness/selection-receipt.json")
    _verify_selection(root, selection, receipt)
    return evaluate_run({**loaded["config"], "selection": selection}, loaded["run"], identifier(eval_id),
                        incremental=incremental)


@exclusive_evaluation
def evaluate_run(cfg, run, eval_id, incremental=False):
    from benchmark_evaluation.webarena import REVISION, evaluate, runtime_reference

    root, run = Path(cfg["project_root"]).resolve(), Path(run)
    output = run / "harness/evaluations" / eval_id
    if output.exists():
        raise FileExistsError("WebArena evaluation output already exists: " + str(output))
    output.mkdir(parents=True)
    selected = cfg["selection"]
    _verify_selection(root, selected, {"selection_sha256": hash_json(selected)})
    runtime_ref = runtime_reference(cfg)
    write_json(output / "selection.json", selected)
    write_json(output / "resolved-config.json", cfg)
    update_status(output / "status.json", "evaluation")
    outcomes = {}
    try:
        def worker(rid, row):
            result = evaluate(cfg, selected["tasks"][row["task_key"]],
                              owned_path(root, row["subject_path"]), output / "subjects" / identifier(rid))
            return rid, {"subject_sha256": row["subject_sha256"], **result}

        workers = cfg.get("execution", {}).get("harness_workers", 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, rid, row) for rid, row in selected["rollouts"].items()]
            for future in concurrent.futures.as_completed(futures):
                rid, outcome = future.result()
                outcomes[rid] = outcome
                write_json(output / "progress.json", outcomes)
        count = len(outcomes)
        resolved = sum(row["resolved"] is True for row in outcomes.values())
        summary = {"evaluation_id": eval_id, "benchmark": "webarena_verified", "evaluator_revision": REVISION,
                   "gold": False, "incremental": bool(incremental), "requested": count,
                   "resolved": resolved, "resolve_rate": resolved / count if count else None,
                   "graded": sum(row["status"] == "completed" for row in outcomes.values()),
                   "infra_error": sum(row["status"] == "infra_error" for row in outcomes.values()),
                   "outcomes": outcomes, "selection_sha256": hash_json(selected),
                   "evaluator_runtime": runtime_ref, "finished_at": now()}
        write_json(output / "summary.json", summary)
        evidence = RunEvidence(root, run.name)
        with evidence.transaction() as data:
            for rid, row in selected["rollouts"].items():
                if rid not in data["rollouts"]:
                    imported = copy.deepcopy(row)
                    imported.update(run_id=run.name, evaluation_status="not_run", resolved=None,
                                    selected_evaluation_id=None)
                    imported["provenance"] = {**row.get("provenance", {}), "source_run_id":
                                              row.get("provenance", {}).get("source_run_id", row["run_id"])}
                    data["rollouts"][rid] = imported
                    data["tasks"][row["task_key"]] = selected["tasks"][row["task_key"]]
                    data["schema_version"] = 2
        evidence.record_evaluation(eval_id, output / "summary.json", outcomes,
                                   {"harness": "WebArenaVerified.evaluate_task", "revision": REVISION,
                                    "source_run_id": selected["source_run_id"], "evaluator_runtime": runtime_ref,
                                    "config_hash": cfg["config_hash"]})
        stage = "complete" if summary["graded"] == count else "incomplete"
        update_status(output / "status.json", stage)
        # Keep the existing generic harness status/report surface readable.
        write_json(run / "harness/summary.json", summary)
        update_status(run / "harness/status.json", stage, evaluation_id=eval_id)
        return summary
    except BaseException as exc:
        update_status(output / "status.json", "failed", error=str(exc))
        raise
