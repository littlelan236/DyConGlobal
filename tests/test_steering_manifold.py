import csv
from pathlib import Path

import numpy as np
import pytest
from test_steering_lda import prepared

from core.io import read_json, sha256
from steering.visualization.manifold import fit_embedding, load_descriptor
from steering.visualization.service import check_analysis, redraw, visualize


@pytest.mark.parametrize("method", ["tsne", "umap"])
def test_seeded_full_embedding_and_no_pca(method):
    values = np.random.default_rng(3).normal(size=(36, 8)).astype(np.float32)
    options = {
        "method": method,
        "seed": 7,
        "pca_components": 4,
        "perplexity": 5,
        "n_neighbors": 5,
        "max_iter": 300,
        "threads": 1,
    }
    first = fit_embedding(values, **options)
    second = fit_embedding(values, **options)
    assert first["projection"].shape == (36, 2)
    np.testing.assert_array_equal(first["projection"], second["projection"])
    assert first["metadata"]["labels_used_for_fit"] is False
    assert first["metadata"]["fit_sample_count"] == 36
    assert first["metadata"]["preprocessing"]["pca_components_effective"] == 4
    direct = fit_embedding(values, **{**options, "pca_components": 0})
    assert direct["projection"].shape == (36, 2)
    assert "pca_center" not in direct["transform"]


@pytest.mark.parametrize("method", ["tsne", "umap"])
def test_formal_embedding_artifacts_display_cap_and_offline_redraw(tmp_path, monkeypatch, method):
    fixture, storage, dataset, cache = prepared(tmp_path, single_class=True)
    result = visualize(
        fixture["cfg"],
        "embedding",
        dataset_id="dataset",
        dataset_root=storage,
        cache_root=tmp_path / "features",
        storage_root=tmp_path / "analyses",
        sites=["0", "2"],
        method=method,
        seed=42,
        max_display_points=2,
        embedding_options={"perplexity": 2, "n_neighbors": 2, "max_iter": 300, "threads": 1},
        formats=("png", "svg", "pdf"),
        facet="retained",
    )
    output = Path(result["path"])
    assert result["status"] == "complete"  # Unsupervised fitting supports a single label.
    assert check_analysis(output)["status"] == "verified"
    with (output / "points.csv").open(encoding="utf-8", newline="") as stream:
        points = list(csv.DictReader(stream))
    axis = method.upper() + "1"
    assert len(points) == 12 and axis in points[0] and "LD1" not in points[0]
    assert sum(p["displayed"] == "True" for p in points) == 4
    assert len(list((output / "plots").glob("*"))) == 6
    metadata = read_json(output / "analysis.json")
    assert metadata["method"] == method and metadata["display"]["jitter"] == "none"
    assert "never enter fitting" in (output / "index.html").read_text()
    assert int(load_descriptor(output / "transform/site-0.npz", method)["n_components"]) == 2
    dataset.rename(dataset.with_name("offline-dataset"))
    cache.rename(cache.with_name("offline-cache"))
    fixture["nas"].rename(fixture["nas"].with_name("offline-source"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen redraw attempted reducer fitting")

    monkeypatch.setattr("steering.visualization.service.fit_embedding", forbidden)
    monkeypatch.setattr("steering.visualization.service.fit_site", forbidden)
    new = redraw(
        fixture["cfg"],
        "redrawn",
        source_analysis_id="embedding",
        analysis_root=tmp_path / "analyses",
        storage_root=tmp_path / "redrawn",
        sites=["2"],
        seed=4,
        max_display_points=3,
        formats=("svg",),
    )
    target = Path(new["path"])
    assert check_analysis(target)["available_sites"] == ["2"]
    for relative in ("transform/site-2.npz", "projections/site-2.npy"):
        assert sha256(output / relative) == sha256(target / relative)
    with pytest.raises(ValueError, match="method must match"):
        redraw(
            fixture["cfg"],
            "wrong",
            source_analysis_id="embedding",
            analysis_root=tmp_path / "analyses",
            method="lda",
        )
    np.save(output / "projections/site-0.npy", np.zeros((6, 2)))
    with pytest.raises(ValueError, match="checksum"):
        check_analysis(output)


@pytest.mark.parametrize("method", ["tsne", "umap"])
def test_embedding_admission_and_degenerate_inputs(method):
    values = np.arange(64, dtype=np.float32).reshape(16, 4)
    opts = {"method": method, "perplexity": 3, "n_neighbors": 3}
    with pytest.raises(ValueError, match="budget"):
        fit_embedding(values, **opts, memory_budget_bytes=1)
    with pytest.raises(ValueError, match="non-finite"):
        fit_embedding(values * np.nan, **opts)
    with pytest.raises(ValueError, match="seed"):
        fit_embedding(values, **opts, seed=-1)
    assert fit_embedding(np.ones((16, 4)), **opts)["reason"] == "constant_embedding_features"
    assert fit_embedding(values[:3], **opts)["reason"] == "insufficient_embedding_samples"
    with pytest.raises(ValueError, match="fit-scope all"):
        visualize({}, "bad", dataset_id="absent", sites=["0"], method=method, fit_scope="grouped")


@pytest.mark.parametrize(
    "method,options,match",
    [
        ("tsne", {"perplexity": 16}, "perplexity"),
        ("tsne", {"perplexity": 3, "max_iter": 249}, "max_iter"),
        ("umap", {"n_neighbors": 16}, "n_neighbors"),
        ("umap", {"n_neighbors": 3, "min_dist": float("nan")}, "min_dist"),
        ("umap", {"n_neighbors": 3, "min_dist": 1.1}, "min_dist"),
    ],
)
def test_invalid_algorithm_parameters(method, options, match):
    with pytest.raises(ValueError, match=match):
        fit_embedding(np.arange(64).reshape(16, 4), method=method, **options)
