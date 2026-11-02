"""Unsupervised full-cohort 2-D embeddings; labels never enter either reducer."""

from __future__ import annotations

import importlib.metadata
import platform

import numpy as np

METHODS = {"tsne", "umap"}


def fit_embedding(
    values,
    *,
    method,
    seed=0,
    pca_components=50,
    perplexity=30.0,
    max_iter=1000,
    n_neighbors=15,
    min_dist=0.1,
    threads=4,
    memory_budget_bytes=32 * 1024**3,
):
    if method not in METHODS:
        raise ValueError("Embedding method must be tsne or umap")
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Embedding input must be a nonempty feature matrix")
    n, width = values.shape
    if n < 4:
        return {"status": "unavailable", "reason": "insufficient_embedding_samples"}
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("Embedding seed must be an integer in [0, 2**32)")
    if not isinstance(pca_components, int) or pca_components < 0:
        raise ValueError("PCA components must be a nonnegative integer (0 disables PCA)")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("Embedding threads must be a positive integer")
    if method == "tsne" and (not np.isfinite(perplexity) or not 0 < perplexity < n):
        raise ValueError("t-SNE perplexity must be finite, positive and less than sample count")
    if method == "tsne" and (not isinstance(max_iter, int) or max_iter < 250):
        raise ValueError("t-SNE max_iter must be an integer >= 250")
    if method == "umap" and (not isinstance(n_neighbors, int) or not 2 <= n_neighbors < n):
        raise ValueError("UMAP n_neighbors must be an integer >= 2 and less than sample count")
    if method == "umap" and (not np.isfinite(min_dist) or not 0 <= min_dist <= 1):
        raise ValueError("UMAP min_dist must be finite and in [0, 1] (spread=1)")
    effective_pca = min(pca_components, width, n - 1) if pca_components else 0
    neighbors = min(n - 1, int(3 * perplexity + 1)) if method == "tsne" else n_neighbors
    # Includes full input copies, randomized-PCA workspaces, sparse graph/optimizer
    # workspaces, and a fixed runtime margin. This is admission, not a peak-RSS claim.
    required = int(
        n * width * 4 * 8 + n * max(effective_pca, 2) * 8 * 8 + n * neighbors * 128 + 256 * 1024**2
    )
    if not np.isfinite(memory_budget_bytes) or memory_budget_bytes <= 0 or required > memory_budget_bytes:
        raise ValueError(
            f"{method} requires an estimated {required} bytes, budget={memory_budget_bytes}; "
            "increase the explicit memory budget; no samples were silently dropped"
        )
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    if method == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ValueError(
                "UMAP requires umap-learn in the analysis environment; install locks/steering-analysis.txt"
            ) from exc

    matrix = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding input contains non-finite values")
    if not np.any(matrix != matrix[0]):
        return {"status": "unavailable", "reason": "constant_embedding_features"}
    descriptor = {
        "method": np.array(method),
        "n_components": np.array(2, dtype=np.int64),
        "feature_width": np.array(width, dtype=np.int64),
        "capability": np.array("frozen_coordinates_only_no_out_of_sample_transform"),
    }
    with threadpool_limits(limits=threads):
        if effective_pca:
            print(f"{method}: PCA {width} -> {effective_pca}, all {n} rows", flush=True)
            pca = PCA(n_components=effective_pca, svd_solver="randomized", random_state=seed, whiten=False)
            matrix = pca.fit_transform(matrix)
            descriptor.update(pca_center=pca.mean_, pca_components=pca.components_)
            explained_variance = float(pca.explained_variance_ratio_.sum())
        else:
            explained_variance = None
        if method == "tsne":
            parameters = {
                "n_components": 2,
                "perplexity": perplexity,
                "max_iter": max_iter,
                "learning_rate": "auto",
                "init": "pca" if matrix.shape[1] >= 2 else "random",
                "method": "barnes_hut",
                "angle": 0.5,
                "early_exaggeration": 12.0,
                "metric": "euclidean",
                "random_state": seed,
                "n_jobs": threads,
            }
            model = TSNE(**parameters)
            print(f"tsne: fitting 2-D Barnes-Hut embedding, all {n} rows", flush=True)
            projection = model.fit_transform(matrix)
            diagnostics = {"kl_divergence": float(model.kl_divergence_), "n_iter": int(model.n_iter_)}
            packages = ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        else:
            parameters = {
                "n_components": 2,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "spread": 1.0,
                "metric": "euclidean",
                "init": "spectral",
                "random_state": seed,
                "transform_seed": seed,
                "n_jobs": 1,
                "low_memory": True,
                "n_epochs": 500 if n <= 10000 else 200,
            }
            model = UMAP(**parameters)
            print(f"umap: fitting 2-D embedding, all {n} rows", flush=True)
            projection = model.fit_transform(matrix)
            diagnostics = {"n_epochs": model.n_epochs, "seeded_optimization_threads": 1}
            packages = (
                "numpy",
                "scipy",
                "scikit-learn",
                "umap-learn",
                "numba",
                "llvmlite",
                "pynndescent",
                "threadpoolctl",
            )
    projection = np.asarray(projection, dtype=np.float64)
    if projection.shape != (n, 2) or not np.isfinite(projection).all():
        raise ValueError("Embedding did not produce one finite 2-D coordinate per sample")
    return {
        "status": "available",
        "projection": projection,
        "transform": descriptor,
        "metadata": {
            "method": method,
            "n_components": 2,
            "fit_sample_count": n,
            "labels_used_for_fit": False,
            "parameters": model.get_params(deep=False),
            "preprocessing": {
                "standard_scaler": False,
                "pca_components_requested": pca_components,
                "pca_components_effective": effective_pca,
                "pca_solver": "randomized" if effective_pca else None,
                "pca_whiten": False,
                "pca_explained_variance_ratio": explained_variance,
            },
            "diagnostics": diagnostics,
            "transform_capability": "frozen_coordinates_only_no_out_of_sample_transform",
            "estimated_fit_bytes": required,
            "memory_budget_bytes": memory_budget_bytes,
            "thread_limit": threads,
            "versions": {name: importlib.metadata.version(name) for name in packages},
            "python_version": platform.python_version(),
        },
    }


def load_descriptor(path, method):
    with np.load(path, allow_pickle=False) as archive:
        frozen = {key: archive[key] for key in archive.files}
    if (
        method not in METHODS
        or str(frozen["method"]) != method
        or int(frozen["n_components"]) != 2
        or int(frozen["feature_width"]) < 1
        or str(frozen["capability"]) != "frozen_coordinates_only_no_out_of_sample_transform"
    ):
        raise ValueError("Frozen embedding descriptor/method mismatch")
    if "pca_components" in frozen:
        components, center = frozen["pca_components"], frozen["pca_center"]
        width = int(frozen["feature_width"])
        if (
            components.ndim != 2
            or components.shape[1] != width
            or center.shape != (width,)
            or not np.isfinite(components).all()
            or not np.isfinite(center).all()
        ):
            raise ValueError("Frozen embedding PCA descriptor is invalid")
    return frozen
