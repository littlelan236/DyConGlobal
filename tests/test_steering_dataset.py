import copy
from pathlib import Path

import numpy as np
import pytest
from steering_fixtures import make_source

from core.io import read_json, sha256, write_json
from hidden.artifacts import read_arrays
from steering.artifacts import iter_jsonl
from steering.datasets.build import build_dataset, check_dataset
from steering.datasets.cache import cache_features, verify_cache
from steering.datasets.paths import directory, locate
from steering.datasets.selection import select_sources


def build(fixture, name="dataset", **kwargs):
    storage = fixture["root"] / "external"
    result = build_dataset(
        fixture["cfg"], name, source_runs=[fixture["run"]], all_submitted=True, storage_root=storage, **kwargs
    )
    return Path(result["path"])


def test_all_candidates_labels_do_not_follow_retention_or_selected(tmp_path):
    fixture = make_source(tmp_path)
    out = build(fixture)
    rows = list(iter_jsonl(out / "samples.jsonl"))
    assert len(rows) == 6
    assert [row["y"] for row in rows] == [1, 0, 1, 1, 0, 1]
    assert rows[3]["retained"] is False and rows[3]["y"] == 1
    assert rows[1]["retained"] is True and rows[1]["y"] == 0
    assert sum(row["selected"] for row in rows) == 1
    assert sum(row["y"] for row in rows) == 4
    assert check_dataset(out)["status"] == "verified"
    assert read_json(out / "dataset.json")["sites"][1]["semantic"] == "final_norm"
    assert read_json(out / "dataset.json")["sites"][1]["id"] == "2"


@pytest.mark.parametrize("representative", ["estimator_score", "min_step_tokens"])
@pytest.mark.parametrize(
    "mode,rule",
    [
        ("parallel_consensus", "effective_count_v1"),
        ("parallel_enthophy", "effective_count_v1"),
        ("parallel_enthophy", "shannon_exp_keep_v1"),
        ("parallel_enthophy", "baseline_floor_v1"),
        ("parallel_enthophy", "effective_count_full_above_v1"),
    ],
)
def test_frozen_fixed_entropy_and_representative_policies(tmp_path, representative, mode, rule):
    fixture = make_source(tmp_path, mode=mode, representative_policy=representative, retention_rule=rule)
    out = build(fixture)
    rows = list(iter_jsonl(out / "samples.jsonl"))
    assert len(rows) == 6 and {row["y"] for row in rows} == {0, 1}
    assert check_dataset(out)["status"] == "verified"


@pytest.mark.parametrize(
    "options,reason,count",
    [
        ({"missing_response": 4}, "missing_tool_calls", 1),
        ({"ambiguous": 3}, "ambiguous_close_events", 1),
        ({"no_closes": True}, "close_generated_hidden_missing", 6),
        ({"startup": True}, "startup_or_smoke_session", 6),
        ({"no_winner": True}, "missing_tool_calls", 6),
    ],
)
def test_missing_and_ambiguous_evidence_is_excluded(tmp_path, options, reason, count):
    out = build(make_source(tmp_path, **options))
    excluded = list(iter_jsonl(out / "excluded.jsonl"))
    assert len(excluded) == count
    assert all(row["label"] == "excluded" and row["y"] is None for row in excluded)
    assert all(row["exclusion_reason"] == reason for row in excluded)
    assert check_dataset(out)["status"] == "verified"


@pytest.mark.parametrize("partial_branch", [0, 3])
def test_partial_branch_preserves_action_audit_and_is_excluded(tmp_path, partial_branch):
    fixture = make_source(tmp_path, partial_branch=partial_branch)
    out = build(fixture)
    samples = list(iter_jsonl(out / "samples.jsonl"))
    excluded = list(iter_jsonl(out / "excluded.jsonl"))
    assert len(samples) == 5 and sum(row["y"] for row in samples) == 3
    assert len(excluded) == 1 and excluded[0]["branch_id"] == partial_branch
    assert excluded[0]["exclusion_reason"] == "incomplete_response"
    assert check_dataset(out)["status"] == "verified"


def test_request_misbinding_is_hard_error_and_partial_output_not_published(tmp_path):
    fixture = make_source(tmp_path, bad_request=3)
    with pytest.raises(ValueError, match="identity mismatch"):
        build(fixture)
    assert not directory(tmp_path / "external", "steering_dataset", "dataset").exists()
    assert not list((tmp_path / "external/steering/datasets").glob("*.partial-*"))


def test_source_tamper_and_later_evaluation_are_distinct(tmp_path):
    fixture = make_source(tmp_path)
    out = build(fixture)
    evidence_path = tmp_path / "runs/run-a/evidence.json"
    evidence = read_json(evidence_path)
    evidence["revision"] += 1
    write_json(evidence_path, evidence)
    assert check_dataset(out)["status"] == "verified"
    decision = fixture["session"] / "parallel/decisions/dec-unit.json"
    decision.write_text(decision.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source checksum"):
        check_dataset(out)


def test_unacknowledged_publication_is_not_consumed(tmp_path):
    fixture = make_source(tmp_path)
    (tmp_path / "runs/run-a/hidden/publication.json").unlink()
    out = build(fixture)
    assert read_json(out / "summary.json")["sample_count"] == 0
    assert read_json(out / "summary.json")["excluded_reasons"]["publication_not_acknowledged"] == 6


def test_multi_run_same_rollout_name_needs_explicit_binding(tmp_path):
    first = make_source(tmp_path, "run-a")
    make_source(tmp_path, "run-b")
    with pytest.raises(ValueError, match="Multiple runs"):
        select_sources(first["cfg"], ["run-a", "run-b"], rollout_ids=["r-unit"])
    bindings = tmp_path / "bindings.json"
    write_json(bindings, [{"source_run_id": run, "rollout_id": "r-unit"} for run in ["run-a", "run-b"]])
    result = build_dataset(
        first["cfg"],
        "multi",
        source_runs=["run-a", "run-b"],
        rollout_list=bindings,
        storage_root=tmp_path / "out",
    )
    rows = list(iter_jsonl(Path(result["path"]) / "samples.jsonl"))
    assert len(rows) == 12 and len({row["uid"] for row in rows}) == 12
    assert {row["source_run_id"] for row in rows} == {"run-a", "run-b"}


def test_explicit_selection_rejects_duplicates_missing_selector_and_sha(tmp_path):
    fixture = make_source(tmp_path)
    with pytest.raises(ValueError, match="selector"):
        select_sources(fixture["cfg"], ["run-a"])
    with pytest.raises(ValueError, match="Duplicate"):
        select_sources(fixture["cfg"], ["run-a"], rollout_ids=["r-unit", "r-unit"])
    file = tmp_path / "bindings.json"
    write_json(file, [{"source_run_id": "run-a", "rollout_id": "r-unit", "trajectory_sha256": "bad"}])
    with pytest.raises(ValueError, match="SHA mismatch"):
        select_sources(fixture["cfg"], ["run-a"], rollout_list=file)


def test_incompatible_tokenizers_rejected(tmp_path):
    fixture = make_source(tmp_path, "run-a")
    make_source(tmp_path, "run-b", tokenizer="different")
    with pytest.raises(ValueError, match="Incompatible"):
        build_dataset(
            fixture["cfg"],
            "multi",
            source_runs=["run-a", "run-b"],
            all_submitted=True,
            storage_root=tmp_path / "out",
        )


def test_paths_external_conflicts_and_invalid_roots(tmp_path):
    fixture = make_source(tmp_path)
    out = build(fixture)
    assert locate(fixture["cfg"], "steering_dataset", "dataset", tmp_path / "external") == out
    with pytest.raises(ValueError):
        directory(tmp_path, "steering_dataset", "../escape")
    with pytest.raises(ValueError):
        directory(Path("/"), "steering_dataset", "id")
    other = tmp_path / "second"
    build_dataset(
        fixture["cfg"], "dataset", source_runs=["run-a"], all_submitted=True, storage_root=other, sites=["0"]
    )
    cfg = copy.deepcopy(fixture["cfg"])
    cfg["paths"]["artifact_scan_roots"] = [str(other)]
    with pytest.raises(ValueError, match="Conflicting"):
        locate(cfg, "steering_dataset", "dataset", tmp_path / "external")


@pytest.mark.parametrize("sharded", [False, True])
def test_cache_matches_source_order_and_dynamic_shapes(tmp_path, sharded):
    fixture = make_source(tmp_path, sharded=sharded)
    out = build(fixture)
    result = cache_features(
        fixture["cfg"],
        "dataset",
        dataset_root=tmp_path / "external",
        cache_root=tmp_path / "cache",
        disk_reserve_bytes=0,
    )
    cache = Path(result["path"])
    rows = list(iter_jsonl(out / "samples.jsonl"))
    for site, key in [("0", "layer_0"), ("2", "layer_2")]:
        matrix = np.load(cache / f"site-{site}.f32.npy")
        assert matrix.shape == (6, 3)
        for i, row in enumerate(rows):
            np.testing.assert_array_equal(matrix[i], read_arrays(row["tensor_root"], row)[key])
    assert verify_cache(cache)["specification"]["sample_count"] == 6
    assert cache_features(
        fixture["cfg"],
        "dataset",
        dataset_root=tmp_path / "external",
        cache_root=tmp_path / "cache",
        disk_reserve_bytes=0,
    )["reused"]
    with (cache / "site-0.f32.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache_features(
            fixture["cfg"],
            "dataset",
            dataset_root=tmp_path / "external",
            cache_root=tmp_path / "cache",
            disk_reserve_bytes=0,
        )


def test_cache_budget_failure_cleans_stage(tmp_path, monkeypatch):
    fixture = make_source(tmp_path)
    build(fixture)

    def fail(*args):
        raise ValueError("Insufficient disk space")

    monkeypatch.setattr("steering.datasets.cache.require_space", fail)
    with pytest.raises(ValueError, match="Insufficient"):
        cache_features(
            fixture["cfg"],
            "dataset",
            dataset_root=tmp_path / "external",
            cache_root=tmp_path / "cache",
            disk_reserve_bytes=0,
        )
    assert not list((tmp_path / "cache/steering/cache").glob("*/cache.json"))


def test_readonly_sources_unchanged_by_build_cache_check(tmp_path):
    fixture = make_source(tmp_path, sharded=True)
    before = {
        p: sha256(p) for p in [*fixture["nas"].rglob("*"), *Path(tmp_path / "runs").rglob("*")] if p.is_file()
    }
    out = build(fixture)
    cache_features(
        fixture["cfg"],
        "dataset",
        dataset_root=tmp_path / "external",
        cache_root=tmp_path / "cache",
        disk_reserve_bytes=0,
    )
    check_dataset(out, with_tensors=True)
    assert all(sha256(path) == digest for path, digest in before.items())
