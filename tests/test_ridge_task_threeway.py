import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from core.io import hash_json, read_json, sha256, write_json
from difficulty.models.ridge_grid import load_model, predict_model
from difficulty.probe_config import RidgeProbe
from difficulty.training.ridge_probe import run_probe
from difficulty.training.split import make_task_threeway_split

# Reuse the existing real receipt -> trajectory -> hidden fixture chain.
from tests.test_ridge_probe import fixture_project  # noqa: F401


def sample_rows():
    rollouts, samples = [], []
    for i in range(20):
        for j in range(2):
            rid = f"r{i}-{j}"
            rollouts.append({"repo": "a", "instance_id": f"t{i}", "base_commit": "abc",
                             "source_kind": "online" if j else "legacy", "rollout_id": rid,
                             "trajectory_sha256": f"sha{i}-{j}"})
            samples.extend({"sample_id": f"{rid}:{k}", "rollout_id": rid,
                            "task_group": f"a::t{i}"} for k in range(3))
    return rollouts, samples


def test_grouped_threeway_is_disjoint_and_retains_all_attempts():
    rollouts, samples = sample_rows()
    split = make_task_threeway_split(rollouts, samples)
    assert split["task_counts"] == {"train": 12, "validation": 4, "test": 4}
    sets = {role: set(ids) for role, ids in split["sample_ids"].items()}
    assert set.union(*sets.values()) == {s["sample_id"] for s in samples}
    assert all(not sets[a] & sets[b] for a, b in itertools.combinations(sets, 2))
    for i in range(20):
        assert len({split["role_by_sample"][f"r{i}-{j}:{k}"] for j in range(2) for k in range(3)}) == 1
    assert split == make_task_threeway_split(list(reversed(rollouts)), samples)
    assert split["tasks"] != make_task_threeway_split(rollouts, samples, seed=43)["tasks"]


def test_task_split_rejects_cross_partition_duplicate_content():
    rollouts, samples = sample_rows()
    split = make_task_threeway_split(rollouts, samples)
    tasks = {t["task_group"]: t["split"] for t in split["tasks"]}
    tr = next(r for r in rollouts if tasks["a::" + r["instance_id"]] == "train")
    te = next(r for r in rollouts if tasks["a::" + r["instance_id"]] == "test")
    te["trajectory_sha256"] = tr["trajectory_sha256"]
    with pytest.raises(ValueError, match="SHA would cross"):
        make_task_threeway_split(rollouts, samples)


@pytest.mark.parametrize("fractions", [(0.8, 0.2), (0.0, 0.2), (0.2, float("nan"))])
def test_task_threeway_invalid_fractions(fractions):
    with pytest.raises(ValueError):
        RidgeProbe(split_strategy="task_random_v1", validation_fraction=fractions[0], test_fraction=fractions[1])


@pytest.mark.parametrize("strategy", ["event_random_v1", "task_random_v1"])
def test_threeway_test_labels_never_affect_selection_or_refit(fixture_project, monkeypatch, strategy):  # noqa: F811
    cfg, settings, dataset = fixture_project
    options = settings.model_dump()
    options.update(split_strategy=strategy, layers=[0], alphas=[1.0, 100.0])
    cfg["difficulty"]["ridge_probe"] = options
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    # Existing tests cover real plot creation; here focus on held-out independence.
    monkeypatch.setattr("difficulty.training.ridge_probe.draw_plots", lambda *args: None)
    first = Path(run_probe(cfg, "threeway-a")["path"])
    attempt = read_json(first / "attempt.json")
    split = read_json(dataset / "splits" / attempt["split_id"] / "split.json")
    test_ids = set(split["sample_ids"]["test"])
    metrics = read_json(first / "metrics.json")
    candidate_selections = read_json(first / "selected.json")["targets"]
    predictions = [json.loads(line) for line in (first / "test-predictions.jsonl").read_text().splitlines()]
    cache = Path(read_json(first / "data-check.json")["cache"])
    samples = [json.loads(line) for line in (dataset / "samples.jsonl").read_text().splitlines()]
    fit_ix = [i for i, s in enumerate(samples) if s["sample_id"] not in test_ids]
    test_ix = [i for i, s in enumerate(samples) if s["sample_id"] in test_ids]
    matrix = np.load(cache / "layer-00.f32.npy")
    for target, value in metrics.items():
        model = load_model(first / value["refit_model"])
        assert np.allclose(model["feature_mean"], matrix[fit_ix].astype(float).mean(0))
        rows = [row for row in predictions if row["target"] == target]
        assert np.allclose(predict_model(model, matrix[test_ix]), [r["prediction_transformed"] for r in rows])
    for sample in samples:
        if sample["sample_id"] in test_ids:
            sample["remaining_reasoning_tokens_to_submission"] += 1000000
    (dataset / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    write_json(dataset / "receipt.json", {"state": "complete", "files": {
        name: sha256(dataset / name) for name in ("rollouts.json", "samples.jsonl")}})
    options["dataset_receipt_sha256"] = sha256(dataset / "receipt.json")
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    second = Path(run_probe(cfg, "threeway-b")["path"])
    later = read_json(second / "selected.json")["targets"]
    for target, before in candidate_selections.items():
        after = later[target]
        assert (before["layer"], before["alpha"], before["validation"]) == (after["layer"], after["alpha"], after["validation"])
        old = load_model(first / metrics[target]["refit_model"])
        new = load_model(second / read_json(second / "metrics.json")[target]["refit_model"])
        assert all(np.array_equal(old[k], new[k]) for k in old)
