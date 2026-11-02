import csv
from pathlib import Path

import numpy as np
import pytest
from steering_fixtures import make_source

from core.io import read_json, sha256, write_json
from steering.artifacts import iter_jsonl
from steering.datasets.build import build_dataset
from steering.datasets.cache import cache_features
from steering.visualization.lda import fit_site, load_transform, project, save_transform
from steering.visualization.service import check_analysis, fit_selection, redraw, visualize


def prepared(tmp_path, single_class=False):
    fixture = make_source(tmp_path, single_class=single_class)
    storage = tmp_path / "storage"
    dataset = build_dataset(
        fixture["cfg"], "dataset", source_runs=["run-a"], all_submitted=True, storage_root=storage
    )
    cache = cache_features(
        fixture["cfg"],
        "dataset",
        dataset_root=storage,
        cache_root=tmp_path / "features",
        disk_reserve_bytes=0,
    )
    return fixture, storage, Path(dataset["path"]), Path(cache["path"])


def test_binary_lda_exact_frozen_transform_and_sign(tmp_path):
    rng = np.random.default_rng(42)
    labels = np.array([0] * 20 + [1] * 20)
    values = rng.normal(size=(40, 4)) + labels[:, None] * np.array([3, 2, 1, -1])
    result = fit_site(values, labels)
    assert result["status"] == "available"
    assert result["projection"].shape == (40, 1)
    assert result["projection"][labels == 1].mean() > result["projection"][labels == 0].mean()
    path = tmp_path / "transform.npz"
    save_transform(path, result["transform"])
    frozen = load_transform(path)
    assert all(not value.dtype.hasobject for value in frozen.values())
    np.testing.assert_allclose(project(values, frozen), result["projection"], atol=1e-12)
    assert result["metadata"]["priors"] == [0.5, 0.5]
    assert result["metadata"]["preprocessing"] == "none"


def test_lda_single_class_degenerate_and_memory_budget():
    assert fit_site(np.ones((6, 3)), np.ones(6))["status"] == "unavailable"
    assert fit_site(np.ones((6, 3)), [0, 0, 0, 1, 1, 1])["status"] == "unavailable"
    with pytest.raises(ValueError, match="budget"):
        fit_site(np.ones((6, 3)), [0, 0, 0, 1, 1, 1], memory_budget_bytes=1)
    with pytest.raises(ValueError, match="non-finite"):
        fit_site(np.full((6, 3), np.nan), [0, 0, 0, 1, 1, 1])


def test_all_coordinates_exported_even_with_display_cap_and_redraw_no_refit(tmp_path, monkeypatch):
    fixture, storage, dataset, cache = prepared(tmp_path)
    result = visualize(
        fixture["cfg"],
        "analysis",
        dataset_id="dataset",
        dataset_root=storage,
        cache_root=tmp_path / "features",
        storage_root=tmp_path / "analyses",
        sites=["0", "2"],
        max_display_points=2,
        seed=17,
        formats=("png", "svg", "pdf"),
        facet="retained",
    )
    out = Path(result["path"])
    assert result["status"] == "complete"
    assert check_analysis(out)["status"] == "verified"
    with (out / "points.csv").open(newline="", encoding="utf-8") as stream:
        points = list(csv.DictReader(stream))
    assert len(points) == 12
    assert sum(row["displayed"] == "True" for row in points) == 4
    assert (out / "index.html").is_file() and "not evidence" in (out / "index.html").read_text()
    before = sha256(dataset / "receipt.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("Redraw attempted LDA refit")

    monkeypatch.setattr("steering.visualization.service.fit_site", forbidden)
    # Redrawing must work when the source dataset, cache and NAS have disappeared.
    dataset.rename(dataset.with_name("offline-dataset"))
    cache.rename(cache.with_name("offline-cache"))
    fixture["nas"].rename(fixture["nas"].with_name("offline-source"))
    new = redraw(
        fixture["cfg"],
        "redrawn",
        source_analysis_id="analysis",
        analysis_root=tmp_path / "analyses",
        storage_root=tmp_path / "redraw-output",
        seed=99,
        max_display_points=3,
        formats=("svg",),
    )
    redrawn = Path(new["path"])
    assert check_analysis(redrawn)["status"] == "verified"
    assert sha256(redrawn / "projections/site-0.npy") == sha256(out / "projections/site-0.npy")
    assert sha256(redrawn / "transform/site-0.npz") == sha256(out / "transform/site-0.npz")
    assert sha256(dataset.with_name("offline-dataset") / "receipt.json") == before


def test_tampered_projection_rejected_before_redraw(tmp_path):
    fixture, storage, dataset, cache = prepared(tmp_path)
    result = visualize(
        fixture["cfg"],
        "analysis",
        dataset_id="dataset",
        dataset_root=storage,
        cache_root=tmp_path / "features",
        storage_root=tmp_path / "analyses",
        sites=["0"],
        formats=("svg",),
    )
    out = Path(result["path"])
    np.save(out / "projections/site-0.npy", np.zeros((6, 1)))
    with pytest.raises(ValueError, match="checksum"):
        redraw(
            fixture["cfg"],
            "redrawn",
            source_analysis_id="analysis",
            analysis_root=tmp_path / "analyses",
            storage_root=tmp_path / "analyses",
            formats=("svg",),
        )


def test_single_class_dataset_is_valid_but_lda_unavailable(tmp_path):
    fixture, storage, dataset, cache = prepared(tmp_path, single_class=True)
    result = visualize(
        fixture["cfg"],
        "single",
        dataset_id="dataset",
        dataset_root=storage,
        cache_root=tmp_path / "features",
        storage_root=tmp_path / "analyses",
        sites=["0"],
        formats=("svg",),
    )
    assert result["status"] == "unavailable"
    assert check_analysis(result["path"])["available_sites"] == []
    assert (
        read_json(Path(result["path"]) / "analysis.json")["site_results"]["0"]["reason"]
        == "missing_fit_class"
    )


def test_group_split_keeps_same_task_across_runs_together(tmp_path):
    fixture, storage, dataset, cache = prepared(tmp_path)
    rows = list(iter_jsonl(dataset / "samples.jsonl"))
    extra = [{**row, "source_run_id": "run-b"} for row in rows]
    other = [{**row, "task_key": "other-task"} for row in rows]
    split = tmp_path / "split.json"
    write_json(
        split,
        [
            {"task_key": fixture["task_key"], "split": "train"},
            {"task_key": "other-task", "split": "validation"},
        ],
    )
    indices, selection, roles = fit_selection(rows + extra + other, "grouped", split)
    assert len(indices) == 12 and roles == ["train"] * 12 + ["validation"] * 6
    write_json(
        split,
        [
            {"task_key": fixture["task_key"], "split": "train"},
            {"task_key": fixture["task_key"], "split": "validation"},
        ],
    )
    with pytest.raises(ValueError, match="leaking"):
        fit_selection(rows, "grouped", split)


def test_grouped_analysis_projects_validation_without_refit(tmp_path):
    fixture = make_source(tmp_path, "run-a", task_name="unit__repo-1")
    other = make_source(tmp_path, "run-b", task_name="unit__repo-2")
    storage = tmp_path / "storage"
    build_dataset(
        fixture["cfg"], "dataset", source_runs=["run-a", "run-b"], all_submitted=True, storage_root=storage
    )
    cache_features(
        fixture["cfg"],
        "dataset",
        dataset_root=storage,
        cache_root=tmp_path / "features",
        disk_reserve_bytes=0,
    )
    split = tmp_path / "split.json"
    write_json(
        split,
        [
            {"task_key": fixture["task_key"], "split": "train"},
            {"task_key": other["task_key"], "split": "validation"},
        ],
    )
    result = visualize(
        fixture["cfg"],
        "grouped",
        dataset_id="dataset",
        dataset_root=storage,
        cache_root=tmp_path / "features",
        storage_root=tmp_path / "analyses",
        sites=["0"],
        fit_scope="grouped",
        split_file=split,
        formats=("svg",),
    )
    assert result["status"] == "complete"
    manifest = read_json(Path(result["path"]) / "analysis.json")
    assert manifest["site_results"]["0"]["metadata"]["fit_class_counts"] == [2, 4]
    assert np.load(Path(result["path"]) / "projections/site-0.npy").shape == (12, 1)
