"""Manifest-driven workflow gates.

No benchmark selection is embedded here. A later rollout can only start after
the user-supplied manifest selection and model/runtime identity are present.
"""

from __future__ import annotations

from pathlib import Path

from core.config import require_model, snapshot


def prepare_run(cfg, run_id, *, partition=None, rollout_ids=()):
    require_model(cfg)
    from application.execution.evidence import RunEvidence
    source_run = cfg.get("source_run_id")
    if not source_run:
        raise ValueError("Evaluation selection requires an explicit source run")
    selection = RunEvidence(cfg["project_root"], source_run).select(rollout_ids, partition=partition)
    if not selection["rollout_ids"]:
        raise ValueError("Source run trajectory selection is empty")
    base_cfg = {k: v for k, v in cfg.items() if k != "source_run_id"}
    return snapshot(base_cfg, run_id, selection)


def assert_runtime_ready(cfg):
    require_model(cfg)
    if not cfg["environment"]["wheelhouse"]:
        raise ValueError("Environment wheelhouse is unset")
    if not Path(cfg["environment"]["wheelhouse"]).is_dir():
        raise FileNotFoundError(cfg["environment"]["wheelhouse"])
    if cfg["steering"]["enabled"] and cfg["steering"]["backend"] != "easysteer":
        raise ValueError("Only EasySteer is supported")
