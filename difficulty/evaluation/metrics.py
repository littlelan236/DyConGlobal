"""Evaluation primitives without assuming a final estimator architecture."""

from __future__ import annotations

import math

import numpy as np


def regression_metrics(y_true, y_pred):
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite equal-shaped nonempty arrays required")
    return paired_regression_metrics(y.ravel(), p.ravel())


def paired_regression_metrics(y_true, y_pred):
    """Finite paired observations, explicit denominators, JSON-safe undefined values."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError('Metric arrays must be matching vectors')
    finite = np.isfinite(y) & np.isfinite(p)
    expected = len(y)
    y, p = y[finite], p[finite]
    n = len(y)
    result = dict(count=n, expected_count=expected, excluded_nonfinite_count=expected-n,
                  r2=None, mae=None, rmse=None, mse=None, sse=None,
                  target_energy=None, target_variance_sum=None,
                  nmse_energy=None, nmse_variance=None,
                  formulas={'nmse_energy': 'SSE / sum(y^2)',
                            'nmse_variance': 'SSE / sum((y-mean(y))^2)',
                            'r2': '1 - nmse_variance'}, reasons={})
    if not n:
        result['reason'] = 'empty' if expected == 0 else 'no_finite_pairs'
        result['reasons'] = {k: result['reason'] for k in ('r2', 'nmse_energy', 'nmse_variance')}
        return result
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        error = y-p
        sse = float(np.sum(error**2))
        energy = float(np.sum(y**2))
        variance = float(np.sum((y-y.mean())**2))
        values = dict(sse=sse, target_energy=energy, target_variance_sum=variance,
                      mae=float(np.mean(np.abs(error))), mse=sse/n, rmse=float(np.sqrt(sse/n)))
        for key, value in values.items():
            result[key] = value if math.isfinite(value) else None
        for key, denominator, reason in (('nmse_energy', energy, 'zero_energy'),
                                          ('nmse_variance', variance, 'zero_variance')):
            value = sse/denominator if denominator > 0 else float('nan')
            if math.isfinite(value):
                result[key] = value
            else:
                result['reasons'][key] = reason if denominator == 0 else 'arithmetic_overflow'
        if n < 2:
            result['reasons']['r2'] = 'insufficient_count'
        elif result['nmse_variance'] is not None:
            result['r2'] = 1-result['nmse_variance']
        else:
            result['reasons']['r2'] = result['reasons']['nmse_variance']
    if result['r2'] is None:
        result['reason'] = result['reasons']['r2']
    return result


def macro_metrics(groups):
    result = {'group_count': len(groups)}
    for key in ('r2', 'nmse_energy', 'nmse_variance', 'mae', 'rmse'):
        values = [g[key] for g in groups if g[key] is not None]
        result[key] = {'value': float(np.mean(values)) if values else None,
                       'defined_groups': len(values), 'undefined_groups': len(groups)-len(values)}
    return result
