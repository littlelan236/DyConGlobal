"""Manifest-driven workflow gates.

No benchmark selection is embedded here. A later rollout can only start after
the user-supplied manifest selection and model/runtime identity are present.
"""

from __future__ import annotations

from pathlib import Path

from baseline.catalog import Catalog
from core.config import require_model, snapshot


def prepare_run(cfg, run_id, *, partition=None, purposes=()):
    require_model(cfg)
    selection = Catalog(cfg["project_root"]).select(
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
