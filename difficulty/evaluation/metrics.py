"""Evaluation primitives without assuming a final estimator architecture."""

from __future__ import annotations

import math

import numpy as np


def regression_metrics(y_true, y_pred):
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite equal-shaped nonempty arrays required")
    residual = y - p
    ss_res, ss_tot = float((residual**2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "count": int(y.size),
        "mae": float(np.abs(residual).mean()),
        "rmse": math.sqrt(float((residual**2).mean())),
        "r2": None if ss_tot == 0 else 1 - ss_res / ss_tot,
    }
