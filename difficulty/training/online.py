"""Formal CLI adapter for fitting an online active-pool estimator snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from core.io import now, write_json


def fit_snapshot(cfg, snapshot, output, *, expected_feature_family=None, fit_spec=None):
    snapshot = Path(snapshot).resolve()
    output = Path(output).resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    rows = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("online estimator snapshot is empty")
    required = {"point_id", "hidden", "target", "feature_family", "base_fingerprint",
                "tokenizer_fingerprint", "site", "width", "dtype"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError("online estimator snapshot row lacks feature contract")
    families = {row["feature_family"] for row in rows}
    bases = {row["base_fingerprint"] for row in rows}
    tokenizers = {row["tokenizer_fingerprint"] for row in rows}
    sites = {row["site"] for row in rows}
    widths = {int(row["width"]) for row in rows}
    if len(families) != 1 or len(bases) != 1 or len(tokenizers) != 1 or len(sites) != 1 or len(widths) != 1:
        raise ValueError("online estimator snapshot mixes incompatible feature contracts")
    family, base, tokenizer, site, width = (next(iter(values)) for values in (families, bases, tokenizers, sites, widths))
    if expected_feature_family is not None and family != expected_feature_family:
        raise ValueError("online estimator feature family differs from requested family")
    configured = (cfg.get("model") or {}).get("fingerprint")
    if configured and base != configured:
        raise ValueError("online estimator base fingerprint differs from configured model")
    ids = [str(row["point_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("online estimator snapshot contains duplicate point IDs")
    x = np.asarray([row["hidden"] for row in rows], dtype=np.float64)
    y = np.asarray([row["target"] for row in rows], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("online estimator snapshot contains invalid hidden/target values")
    from difficulty.models.ridge_grid import RidgeGrid, predict_model
    from difficulty.training.train import regression_report
    spec = {"model_kind": "ridge", "alpha": 1.0, "target": "submission_remaining_steps_linear",
            "split_policy": "train_only", "blas_threads": 4, **dict(fit_spec or {})}
    if spec["model_kind"] != "ridge" or spec["split_policy"] != "train_only":
        raise ValueError("online snapshot fitting currently requires the shared Ridge train-only protocol")
    if (y < 0).any():
        raise ValueError("online estimator target must be nonnegative")
    splits = [row.get("split") or "train" for row in rows]
    if set(splits) - {"train", "val", "test"}:
        raise ValueError("unsupported estimator source split")
    train = np.asarray([split == "train" for split in splits])
    if not train.any():
        raise ValueError("active estimator pool has no train samples")
    from threadpoolctl import threadpool_limits
    if type(spec["blas_threads"]) is not int or spec["blas_threads"] < 1:
        raise ValueError("online estimator BLAS thread count must be positive")
    with threadpool_limits(limits=spec["blas_threads"], user_api="blas"):
        grid = RidgeGrid(x[train], y[train])
        coef, intercept = grid.coefficients(float(spec["alpha"]))
    fitted = {"coef": coef[:, 0], "intercept": np.asarray(intercept[0]),
              "feature_mean": grid.scaler.mean, "feature_scale": grid.scaler.scale}
    prediction = predict_model(fitted, x)
    diagnostics = {}
    for split in sorted(set(splits)):
        mask = np.asarray([value == split for value in splits])
        diagnostics[split] = regression_report(y[mask], prediction[mask])
    # Version distributions are observational. Pool admission explicitly permits
    # same-base versions and never selects replacements using these statistics.
    version_diagnostics = {}
    for fingerprint in sorted({row["model_fingerprint"] for row in rows}):
        mask = np.asarray([row["model_fingerprint"] == fingerprint for row in rows])
        subset = x[mask]
        version_diagnostics[fingerprint] = {"count": int(mask.sum()),
            "hidden_norm_mean": float(np.linalg.norm(subset, axis=1).mean()),
            "centroid_distance_from_pool": float(np.linalg.norm(subset.mean(0) - x.mean(0))),
            "prediction": regression_report(y[mask], prediction[mask])}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name("." + output.name + ".tmp.npz")
    np.savez(temp, coef=np.asarray(fitted["coef"], dtype=np.float64),
             intercept=np.asarray(fitted["intercept"], dtype=np.float64),
             feature_mean=fitted["feature_mean"], feature_scale=fitted["feature_scale"],
             target=np.asarray(spec["target"]), alpha=np.asarray(spec["alpha"]),
             feature_family=np.asarray(family), base_fingerprint=np.asarray(base),
             tokenizer_fingerprint=np.asarray(tokenizer), site=np.asarray(site),
             width=np.asarray(width, dtype=np.int64),
             model_fingerprints=np.asarray(sorted({row["model_fingerprint"] for row in rows})))
    temp.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt = output.with_suffix(".receipt.json")
    write_json(receipt, {"schema_version": 1, "state": "complete", "kind": "online_active_pool",
                         "snapshot": str(snapshot), "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                         "checkpoint": str(output), "checkpoint_sha256": digest, "count": len(rows),
                         "point_ids": ids, "feature_family": family, "base_fingerprint": base,
                         "tokenizer_fingerprint": tokenizer, "site": site, "width": width,
                         "split_policy": "train_only", "fit_spec": spec, "train_count": int(train.sum()),
                         "diagnostics": diagnostics, "version_diagnostics": version_diagnostics,
                         "compatibility_policy": "same_base_cross_version_v1", "created_at": now()})
    return {"checkpoint": str(output), "checkpoint_sha256": digest, "receipt": str(receipt),
            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(), "count": len(rows),
            "feature_family": family, "base_fingerprint": base, "site": site, "width": width}
