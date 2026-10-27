"""Harness selection and execution services bound to a run."""

from __future__ import annotations

from pathlib import Path

from application.execution.evidence import RunEvidence
from core.config import load_snapshot
from core.io import hash_json, identifier, read_json, write_json

from .run_plan import load_plan


def prepare(root, run_id, rollout_ids=(), source_run=None):
    loaded = load_plan(root, run_id)
    root = Path(root).resolve()
    source_run = source_run or run_id
    target = loaded["run"] / "harness/selection.json"
    if target.exists():
        from core.io import owned_path, sha256
        frozen = read_json(target)
        proof = read_json(target.with_name("selection-receipt.json"))
        if hash_json(frozen) != proof["selection_sha256"]:
            raise ValueError("Harness selection was modified")
        if frozen.get("source_run_id", run_id) != source_run or (
                rollout_ids and sorted(rollout_ids) != sorted(frozen["rollout_ids"])):
            raise ValueError("Harness selection is already frozen; create a new run")
        for rid, row in frozen["rollouts"].items():
            for kind in ("trajectory", "patch"):
                if sha256(owned_path(root, row[kind+"_path"])) != row[kind+"_sha256"]:
                    raise ValueError(f"Changed {kind}: {rid}")
        return {"run_id": run_id, "rollout_count": len(frozen["rollout_ids"]), "selection": str(target)}
    run_catalog = RunEvidence(root, source_run)
    run_data = run_catalog.read()
    from rollout.retry import enabled
    source_snapshot = root / 'runs' / source_run / 'resolved-config.json'
    final_only = source_snapshot.exists() and enabled(load_snapshot(root, source_run))
    task_keys = {row["task_key"] for row in loaded["tasks"]["tasks"]}
    if rollout_ids:
        requested = list(rollout_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate rollout IDs")
    else:
        requested = run_catalog.select()["rollout_ids"] if final_only else [
            rid for rid, row in run_data["rollouts"].items()
            if not row.get("provenance", {}).get("quarantined")]
    missing = set(requested) - set(run_data["rollouts"])
    if missing:
        raise ValueError(f"Unknown rollout IDs: {sorted(missing)}")
    selected = (run_catalog.select(requested)["rollouts"] if requested else {}) if final_only else {
        rid: run_data["rollouts"][rid] for rid in requested}
    if any(row["task_key"] not in task_keys or row["provenance"].get("quarantined") for row in selected.values()):
        raise ValueError("Harness selection crosses run tasks or includes quarantined trajectories")
    selection = {
        "schema_version": 1,
        "run_id": run_id,
        "source_run_id": source_run,
        "evidence_sha256": hash_json(run_data),
        "rollout_ids": requested,
        "tasks": {key: run_data["tasks"][key] for key in {r["task_key"] for r in selected.values()}},
        "rollouts": selected,
    }
    if not requested:
        raise ValueError("Harness selection is empty")
    target = loaded["run"] / "harness" / "selection.json"
    from core.io import owned_path, sha256
    for rid, row in selected.items():
        for kind in ("trajectory", "patch"):
            if sha256(owned_path(root, row[kind+"_path"])) != row[kind+"_sha256"]:
                raise ValueError(f"Changed {kind}: {rid}")
    if target.exists() and read_json(target) != selection:
        raise ValueError("Harness selection is already frozen; create a new run")
    write_json(target, selection)
    write_json(target.with_name("selection-receipt.json"), {"selection_sha256": hash_json(selection)})
    return {"run_id": run_id, "rollout_count": len(requested), "selection": str(target)}


def run(root, run_id, eval_id, gold=False):
    loaded = load_plan(root, run_id)
    selection = read_json(loaded["run"] / "harness" / "selection.json")
    receipt = read_json(loaded["run"] / "harness/selection-receipt.json")
    if hash_json(selection) != receipt["selection_sha256"]:
        raise ValueError("Harness selection changed after prepare")
    cfg = load_snapshot(root, run_id)
    cfg["selection"] = selection
    from benchmark_evaluation.harness import evaluate

    return evaluate(cfg, loaded["run"], identifier(eval_id), gold)
