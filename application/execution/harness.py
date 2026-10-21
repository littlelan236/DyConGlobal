"""Harness selection and execution services bound to a run."""

from __future__ import annotations

from pathlib import Path

from catalog.store import Catalog
from core.config import load_snapshot
from core.io import hash_json, identifier, read_json, write_json

from .run_plan import load_plan


def prepare(root, run_id, rollout_ids=()):
    loaded = load_plan(root, run_id)
    root = Path(root).resolve()
    catalog = Catalog(root)
    data = catalog.read()
    task_keys = {row["task_key"] for row in loaded["tasks"]["tasks"]}
    if rollout_ids:
        requested = list(rollout_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate rollout IDs")
    else:
        requested = [rid for rid, row in data["rollouts"].items() if row["run_id"] == run_id and not row["provenance"].get("quarantined")]
    missing = set(requested) - set(data["rollouts"])
    if missing:
        raise ValueError(f"Unknown rollout IDs: {sorted(missing)}")
    selected = {rid: data["rollouts"][rid] for rid in requested}
    if any(row["task_key"] not in task_keys or row["provenance"].get("quarantined") for row in selected.values()):
        raise ValueError("Harness selection crosses run tasks or includes quarantined trajectories")
    selection = {
        "schema_version": 1,
        "run_id": run_id,
        "catalog_revision": data["catalog_revision"],
        "catalog_sha256": hash_json(data),
        "rollout_ids": requested,
        "tasks": {key: data["tasks"][key] for key in {r["task_key"] for r in selected.values()}},
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
