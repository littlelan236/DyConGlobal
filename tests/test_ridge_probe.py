import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core.io import hash_json, read_json, sha256, write_json
from difficulty.datasets.submission import arrays_from_bytes, load_selection, public_features
from difficulty.models.estimators import RidgeEstimator
from difficulty.models.ridge_grid import RidgeGrid, load_model, predict_model, save_model
from difficulty.probe_config import RidgeProbe
from difficulty.training.ridge_probe import report, run_probe, select_best
from difficulty.training.split import make_split


@pytest.mark.parametrize("shape", [(40, 9), (9, 40)])
def test_shared_solver_matches_existing_unpenalized_intercept(shape, tmp_path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=shape)
    x[:, -1] = 3  # constant feature and rank deficiency
    y = np.column_stack([5 + x[:, 0] * 2, 7 - x[:, 1]])
    grid = RidgeGrid(x, y)
    xv = rng.normal(size=(7, shape[1])) + 100
    for alpha in (1e-4, 1, 1e8):
        coef, intercept = grid.coefficients(alpha)
        for i in range(2):
            reference = RidgeEstimator(alpha=alpha).fit(x, y[:, i])
            model = {"coef": coef[:, i], "intercept": intercept[i],
                     "feature_mean": grid.scaler.mean, "feature_scale": grid.scaler.scale}
            assert np.allclose(predict_model(model, xv), reference.predict(xv), atol=2e-7, rtol=2e-7)
            assert np.allclose(model["feature_mean"], x.mean(0))
            save_model(tmp_path / "model.npz", model, {"fit_scope": "train"})
            assert np.array_equal(predict_model(load_model(tmp_path / "model.npz"), xv),
                                  predict_model(model, xv))


def test_validation_ranking_ties_negative_and_undefined():
    def trial(layer, alpha, r2):
        return {"layer": layer, "alpha": alpha, "status": "complete", "validation": {"r2": r2}}
    trials = [trial(0, 1, None), trial(3, 1, -2), trial(2, 10, -2-1e-11), trial(1, 10, -2)]
    assert select_best(trials, 1e-10)["layer"] == 1
    assert report(np.ones(4), np.zeros(4))["r2"] is None
    assert report(np.arange(4), np.full(4, 100))["r2"] < 0
    with pytest.raises(ValueError):
        select_best([trial(0, 1, None)], 1e-10)


def tasks():
    return [{"repo": repo, "instance_id": f"{repo}-{i}", "base_commit": "abc",
             "rollout_id": f"r-{repo}-{i}", "trajectory_sha256": f"sha-{repo}-{i}",
             "source_kind": "legacy_close" if i % 2 else "online_close"}
            for repo, count in [("a", 7), ("b", 3), ("single", 1)] for i in range(count)]


def test_split_ignores_old_roles_and_is_task_deterministic():
    rollouts = tasks()
    samples = [{"rollout_id": r["rollout_id"], "sample_id": r["rollout_id"]+f":{i}", "split": "train"}
               for r in rollouts for i in range(3)]
    split = make_split(rollouts, samples, validation_tasks=3)
    again = make_split(list(reversed(rollouts)), samples, validation_tasks=3)
    assert split == again
    assert split["task_counts"] == {"train": 8, "validation": 3}
    assert all(t["split"] == "train" for t in split["tasks"] if t["repo"] == "single")
    for task in split["tasks"]:
        ids = {s["sample_id"] for s in samples if s["rollout_id"] == task["rollout_id"]}
        assert ids <= set(split["sample_ids"][task["split"]])
    with pytest.raises(ValueError, match="exactly one"):
        make_split(rollouts + [rollouts[0]], samples, 3)


def test_native_final_norm_mapping_and_legacy_row_alignment():
    native = {f"layer_{i}": np.full(4096, i, dtype=float) for i in list(range(31))+[32]}
    assert public_features(native, {})[31, 0] == 32
    legacy = {"hidden": np.zeros((2, 32, 4096)), "step": np.array([4, 7])}
    assert public_features(legacy, {"array_row": 1, "agent_step": 7}).shape == (32, 4096)
    with pytest.raises(ValueError, match="row/step"):
        public_features(legacy, {"array_row": 1, "agent_step": 6})
    native["layer_32"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        public_features(native, {})
    with pytest.raises(ValueError, match="SHA"):
        arrays_from_bytes(b"bad", "0"*64)


@pytest.mark.parametrize("overrides", [{"alphas": [0.0]}, {"alphas": [float("nan")]},
                                     {"layers": [32]}, {"layers": [1, 1]}, {"targets": ["success"]},
                                     {"validation_tasks": 238}, {"standardize": False}, {"surprise": True}])
def test_strict_grid_validation(overrides):
    with pytest.raises(ValueError):
        RidgeProbe.model_validate(overrides)


@pytest.fixture
def fixture_project(tmp_path):
    """A real receipt/trajectory/hidden chain with shifted validation features."""
    from difficulty.training.ridge_probe import code_evidence

    source_root = Path(__file__).resolve().parents[1]
    for name in code_evidence(source_root):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / name).read_bytes())
    dataset = tmp_path / "difficulty/datasets/fixture"
    dataset.mkdir(parents=True)
    rollouts, samples, live_rollouts, live_tasks = [], [], {}, {}
    rng = np.random.default_rng(91)
    for i in range(8):
        rid, task_key = f"r{i}", f"t{i}"
        trajectory = f"runs/source/rollout/trajectories/{rid}.json"
        patch = f"runs/source/rollout/patches/{rid}.diff"
        write_json(tmp_path / trajectory, {"info": {"exit_status": "Submitted"},
                   "messages": [{"role": "assistant"} for _ in range(3)]})
        (tmp_path / patch).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / patch).write_text("fixture patch")
        hidden = tmp_path / "hidden" / f"{rid}.npz"
        hidden.parent.mkdir(exist_ok=True)
        x = rng.normal(size=(3, 32, 4096)).astype(np.float32)
        np.savez(hidden, hidden=x, step=np.arange(3))
        row = {"repo": "fixture/repo", "instance_id": f"task{i}", "task_key": task_key,
               "base_commit": "abc", "rollout_id": rid, "trajectory_path": trajectory,
               "patch_path": patch, "patch_sha256": sha256(tmp_path / patch),
               "trajectory_sha256": sha256(tmp_path / trajectory), "model_fingerprint": "model",
               "submission_status": "submitted", "canonical_exit_status": "Submitted",
               "canonical_valid_assistant_steps": 3, "source_kind": "legacy_close",
               "hidden_root": str(hidden.parent), "source_files": {str(hidden): sha256(hidden)}}
        # Unique content is required for no cross-split trajectory SHA leakage.
        trajectory_data = read_json(tmp_path / trajectory)
        trajectory_data["task"] = i
        write_json(tmp_path / trajectory, trajectory_data)
        row["trajectory_sha256"] = sha256(tmp_path / trajectory)
        rollouts.append(row)
        live_rollouts[rid] = {**row, "provenance": {}}
        live_tasks[task_key] = row
        for step in range(3):
            samples.append({"sample_id": f"{rid}:{step}", "rollout_id": rid, "agent_step": step,
                "message_index": step, "anchor": "think_close", "occurrence": 0,
                "endpoint": "submission", "submission_status": "submitted", "both_targets_available": True,
                "repo": row["repo"], "instance_id": row["instance_id"], "source_kind": "legacy_close",
                "task_group": row["repo"]+"::"+row["instance_id"], "split": "train",
                "trajectory_sha256": row["trajectory_sha256"], "remaining_steps_to_submission": 3-step,
                "remaining_reasoning_tokens_to_submission": (2-step)*(i+1),
                "hidden_root": str(hidden.parent), "hidden_path": hidden.name,
                "hidden_sha256": sha256(hidden), "array_name": "hidden", "array_row": step})
    write_json(tmp_path / "catalog/manifest.json", {"catalog_revision": 1, "rollouts": live_rollouts, "tasks": live_tasks})
    write_json(dataset / "rollouts.json", {"rollouts": rollouts})
    (dataset / "samples.jsonl").write_text("".join(json.dumps(s)+"\n" for s in samples))
    write_json(dataset / "receipt.json", {"state": "complete", "files": {
        name: sha256(dataset / name) for name in ("rollouts.json", "samples.jsonl")}})
    settings = RidgeProbe(dataset_id="fixture", dataset_receipt_sha256=sha256(dataset / "receipt.json"),
                          layers=[0, 31], alphas=[1.0, 100.0], validation_tasks=2,
                          expected_tasks=8, expected_samples=24)
    cfg = {"project_root": str(tmp_path), "difficulty": {"ridge_probe": settings.model_dump()}}
    cfg["config_hash"] = hash_json(cfg)
    return cfg, settings, dataset


def test_full_artifact_roundtrip_and_no_validation_fit(fixture_project):
    pytest.importorskip("matplotlib")  # Full fit/report is an analysis-environment contract.
    cfg, settings, _ = fixture_project
    result = run_probe(cfg, "fixture-fit")
    output = Path(result["path"])
    assert result["models"] == 8
    receipt = read_json(output / "receipt.json")
    assert all(sha256(output / name) == digest for name, digest in receipt["files"].items())
    assert len((output / "trials.jsonl").read_text().splitlines()) == 8
    assert len(list((output / "plots").glob("*.png"))) == 6
    _, samples, _ = load_selection(cfg, settings)
    attempt = read_json(output / "attempt.json")
    split_path = Path(cfg["project_root"]) / "difficulty/datasets" / attempt["dataset_id"] / "splits" / attempt["split_id"] / "split.json"
    split = read_json(split_path)
    assert not (output / "split.json").exists()
    train_ids = set(split["sample_ids"]["train"])
    from difficulty.datasets.submission import iter_features

    train = [x for i, x in iter_features(samples) if samples[i]["sample_id"] in train_ids]
    model = load_model(next((output / "models").rglob("*.npz")))
    # The saved scaler belongs to its public layer and only the training samples.
    path = next((output / "models").rglob("*.npz"))
    with np.load(path, allow_pickle=False) as z:
        layer = int(z["layer"])
    assert np.allclose(model["feature_mean"], np.stack(train)[:, layer].mean(0), atol=1e-7)
    with pytest.raises(FileExistsError):
        run_probe(cfg, "fixture-fit")


def test_tampered_selection_and_non_submission_fail(fixture_project):
    cfg, settings, dataset = fixture_project
    samples = (dataset / "samples.jsonl").read_bytes()
    with (dataset / "samples.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="selection file changed"):
        load_selection(cfg, settings)
    (dataset / "samples.jsonl").write_bytes(samples)
    # Live global metadata is irrelevant to a receipt-bound training input.
    path = Path(cfg["project_root"]) / "catalog/manifest.json"
    catalog = read_json(path)
    catalog["rollouts"]["r0"]["submission_status"] = "unsubmitted"
    write_json(path, catalog)
    load_selection(cfg, settings)
    # Invalid submitted status in the actual frozen input still fails.
    frozen = read_json(dataset / "rollouts.json")
    frozen["rollouts"][0]["submission_status"] = "unsubmitted"
    write_json(dataset / "rollouts.json", frozen)
    receipt = read_json(dataset / "receipt.json")
    receipt["files"]["rollouts.json"] = sha256(dataset / "rollouts.json")
    write_json(dataset / "receipt.json", receipt)
    changed = settings.model_copy(update={"dataset_receipt_sha256": sha256(dataset / "receipt.json")})
    with pytest.raises(ValueError, match="submitted canonical"):
        load_selection(cfg, changed)


def test_dry_run_is_not_fit_and_does_not_build_cache(fixture_project):
    cfg, _, _ = fixture_project
    result = run_probe(cfg, "fixture-dry", mode="dry-run")
    output = Path(result["path"])
    receipt = read_json(output / "receipt.json")
    assert receipt["mode"] == "dry-run"
    assert not (Path(cfg["project_root"]) / "state/difficulty-cache").exists()
    assert not (output / "models").exists()
    with pytest.raises(FileExistsError):
        run_probe(cfg, "fixture-dry", mode="fit")


def test_optional_config_hash_compatibility(tmp_path):
    from core.config import resolve

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(root / "config.yaml")
    assert cfg["difficulty"]["ridge_probe"]["validation_tasks"] == 48
    # Optional absence must not add a new key to old resolved snapshots.
    import yaml

    raw = yaml.safe_load((root / "config.yaml").read_text())
    old = copy.deepcopy(raw)
    old["difficulty"].pop("ridge_probe", None)
    # Relocating config must not make an existing external estimator path relative to tmp_path.
    if old.get("parallel", {}).get("estimator_ref"):
        old["parallel"]["estimator_ref"] = str((root / old["parallel"]["estimator_ref"]).resolve())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(old))
    resolved = resolve(path)
    assert "ridge_probe" not in resolved["difficulty"]
    assert resolved["config_hash"] == hash_json({k: v for k, v in resolved.items() if k != "config_hash"})


def test_expanded_selection_load_and_inherited_dryrun(fixture_project):
    cfg, settings, dataset = fixture_project
    root = Path(cfg["project_root"])
    original = read_json(dataset / "rollouts.json")["rollouts"]
    samples = [json.loads(line) for line in (dataset / "samples.jsonl").read_text().splitlines()]
    parent = make_split(original, samples, settings.validation_tasks, settings.split_seed)
    parent_path = root / "difficulty/checkpoints/parent"
    write_json(parent_path / "split.json", parent)
    write_json(parent_path / "receipt.json", {"state": "complete", "files": {
        "split.json": sha256(parent_path / "split.json")}})
    catalog = read_json(root / "catalog/manifest.json")
    for row in original[:2]:
        addition = copy.deepcopy(row)
        rid = row["rollout_id"] + "-additional"
        addition["rollout_id"] = rid
        path = f"runs/source/rollout/trajectories/{rid}.json"
        trajectory = read_json(root / row["trajectory_path"])
        trajectory["independent_attempt"] = rid
        write_json(root / path, trajectory)
        addition["trajectory_path"] = path
        addition["trajectory_sha256"] = sha256(root / path)
        catalog["rollouts"][rid] = {**addition, "provenance": {}}
        for sample in list(samples):
            if sample["rollout_id"] == row["rollout_id"]:
                samples.append({**sample, "sample_id": sample["sample_id"]+"-additional",
                                "rollout_id": rid, "trajectory_sha256": addition["trajectory_sha256"]})
        original.append(addition)
    write_json(root / "catalog/manifest.json", catalog)
    write_json(dataset / "rollouts.json", {"rollouts": original})
    (dataset / "samples.jsonl").write_text("".join(json.dumps(s)+"\n" for s in samples))
    write_json(dataset / "receipt.json", {"state": "complete", "files": {
        name: sha256(dataset / name) for name in ("rollouts.json", "samples.jsonl")}})
    options = settings.model_dump()
    options.update(expected_rollouts=10, expected_samples=30,
                   dataset_receipt_sha256=sha256(dataset / "receipt.json"),
                   split_strategy="inherit_task_split_v1", parent_experiment_id="parent",
                   parent_receipt_sha256=sha256(parent_path / "receipt.json"))
    cfg["difficulty"]["ridge_probe"] = options
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    result = run_probe(cfg, "expanded-dry", mode="dry-run")
    attempt = read_json(Path(result["path"]) / "attempt.json")
    split = read_json(root / "difficulty/datasets" / attempt["dataset_id"] / "splits" / attempt["split_id"] / "split.json")
    assert split["task_counts"] == parent["task_counts"]
    assert len(split["rollouts"]) == 10
    assert sum(split["sample_counts"].values()) == 30
