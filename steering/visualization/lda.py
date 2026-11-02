"""One binary discriminant dimension, with explicit fit memory admission."""

from __future__ import annotations

import importlib.metadata
import platform

import numpy as np


def estimate_svd_bytes(n, width):
    return int(n * width * 8 * 6 + min(n, width) ** 2 * 8 * 4 + n * 64)


def project(values, transform, *, chunk_size=1024):
    component, center = transform["component"], transform["center"]
    if values.ndim != 2 or component.shape != (values.shape[1], 1) or center.shape != (values.shape[1],):
        raise ValueError("Frozen LDA transform/input shape mismatch")
    result = np.empty((len(values), 1), dtype=np.float64)
    for first in range(0, len(values), chunk_size):
        batch = np.asarray(values[first : first + chunk_size], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("LDA input contains non-finite values")
        result[first : first + len(batch)] = (batch - center) @ component
    if not np.isfinite(result).all():
        raise ValueError("LDA projection contains non-finite values")
    return result


def fit_site(values, labels, *, fit_indices=None, tol=1e-4, memory_budget_bytes=32 * 1024**3):
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (len(values),) or not set(labels.tolist()) <= {0, 1}:
        raise ValueError("Binary LDA input shape/labels are invalid")
    fit_indices = np.arange(len(values)) if fit_indices is None else np.asarray(fit_indices, dtype=np.int64)
    if not len(fit_indices) or len(set(fit_indices.tolist())) != len(fit_indices):
        raise ValueError("LDA fit sample selection is empty or duplicated")
    if fit_indices.min() < 0 or fit_indices.max() >= len(values):
        raise ValueError("LDA fit index is out of bounds")
    training_labels = labels[fit_indices]
    counts = [int((training_labels == y).sum()) for y in (0, 1)]
    if not all(counts):
        return {"status": "unavailable", "reason": "missing_fit_class", "class_counts": counts}
    if len(fit_indices) <= 2:
        return {"status": "unavailable", "reason": "insufficient_fit_samples", "class_counts": counts}
    required = estimate_svd_bytes(len(fit_indices), values.shape[1])
    if memory_budget_bytes <= 0 or required > memory_budget_bytes:
        raise ValueError(
            f"LDA SVD requires an estimated {required} bytes, budget={memory_budget_bytes}; "
            "increase the explicit memory budget; no samples were silently dropped"
        )
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError("LDA tolerance must be finite and positive")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    matrix = np.asarray(values[fit_indices], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("LDA fit data contain non-finite values")
    model = LinearDiscriminantAnalysis(solver="svd", n_components=1, priors=[0.5, 0.5], tol=tol)
    try:
        model.fit(matrix, training_labels)
    except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
        return {
            "status": "unavailable",
            "reason": "degenerate_fit",
            "detail": str(exc),
            "class_counts": counts,
        }
    if model.scalings_.shape[1] < 1 or not np.isfinite(model.scalings_[:, :1]).all():
        return {"status": "unavailable", "reason": "no_discriminant_direction", "class_counts": counts}
    transform = {
        "component": model.scalings_[:, :1].copy(),
        "center": model.xbar_.copy(),
        "classes": np.array([0, 1], dtype=np.int64),
        "priors": np.array([0.5, 0.5]),
        "tol": np.array(tol),
        "n_components": np.array(1, dtype=np.int64),
    }
    fit_coordinates = (matrix - transform["center"]) @ transform["component"]
    means = [float(fit_coordinates[training_labels == y].mean()) for y in (0, 1)]
    if abs(means[1] - means[0]) <= np.finfo(np.float64).eps * max(1, abs(means[0]), abs(means[1])):
        return {"status": "unavailable", "reason": "no_class_mean_contrast", "class_counts": counts}
    if means[1] < means[0]:
        transform["component"] *= -1
        means = [-value for value in means]
    del matrix, model, fit_coordinates
    coordinates = project(values, transform)
    return {
        "status": "available",
        "projection": coordinates,
        "transform": transform,
        "metadata": {
            "solver": "svd",
            "n_components": 1,
            "tol": tol,
            "priors": [0.5, 0.5],
            "preprocessing": "none",
            "orientation": "positive_fit_mean_larger",
            "fit_class_counts": counts,
            "fit_class_means": means,
            "estimated_svd_bytes": required,
            "memory_budget_bytes": memory_budget_bytes,
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
            },
            "python_version": platform.python_version(),
        },
    }


def save_transform(path, transform):
    if any(np.asarray(value).dtype.hasobject for value in transform.values()):
        raise ValueError("LDA transforms cannot contain pickle/object arrays")
    np.savez(path, **transform)


def load_transform(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if int(result["n_components"]) != 1 or result["component"].ndim != 2 or result["component"].shape[1] != 1:
        raise ValueError("Binary LDA transform must contain exactly one component")
    return result
