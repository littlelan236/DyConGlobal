"""Manifest-driven workflow gates.

No benchmark selection is embedded here. A later rollout can only start after
the user-supplied manifest selection and model/runtime identity are present.
"""

from __future__ import annotations

from pathlib import Path

from catalog.store import Catalog
from core.config import require_model, snapshot
from core.io import hash_json


def prepare_run(cfg, run_id, *, partition=None, purposes=(), rollout_ids=()):
    require_model(cfg)
    catalog = Catalog(cfg["project_root"])
    if rollout_ids:
        # Evaluation cohorts must be frozen by canonical rollout identity; this
        # keeps retries and unrelated catalog rows out of the harness snapshot.
        data = catalog.read()
        requested = list(rollout_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate rollout IDs in evaluation selection")
        missing = set(requested) - set(data["rollouts"])
        if missing:
            raise ValueError(f"Unknown rollout IDs: {sorted(missing)}")
        selected = {rid: data["rollouts"][rid] for rid in requested}
        tasks = {key: data["tasks"][row["task_key"]] for row in selected.values() for key in [row["task_key"]]}
        if partition is not None and any(task["partition"] != partition for task in tasks.values()):
            raise ValueError("Explicit rollout selection crosses the requested partition")
        if purposes and any(not set(purposes).intersection(row["purposes"]) for row in selected.values()):
            raise ValueError("Explicit rollout selection does not satisfy purposes")
        selection = {
            "catalog_revision": data["catalog_revision"],
            "catalog_sha256": hash_json(data),
            "rollout_ids": requested,
            "tasks": tasks,
            "rollouts": selected,
        }
    else:
        selection = catalog.select(
            partition=partition, purposes=purposes, profile=cfg["benchmark"]["profile"]
        )
    if not selection["rollout_ids"]:
        raise ValueError("Manifest selection is empty; provide the baseline rollout list first")
    return snapshot(cfg, run_id, selection)


def assert_runtime_ready(cfg):
    require_model(cfg)
    if not cfg["environment"]["wheelhouse"]:
        raise ValueError("Environment wheelhouse is unset")
    if not Path(cfg["environment"]["wheelhouse"]).is_dir():
        raise FileNotFoundError(cfg["environment"]["wheelhouse"])
    if cfg["steering"]["enabled"] and cfg["steering"]["backend"] != "easysteer":
        raise ValueError("Only EasySteer is supported")
