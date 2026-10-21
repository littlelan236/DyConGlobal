"""Validation-only model selection, immutable publication and API equivalence."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.io import hash_json, read_json, sha256, write_json
from difficulty.models.ridge_grid import load_model, predict_model, save_model
from runtime.serving.parallel import artifact_contract
from runtime.serving.predictors import RIDGE_LOADER, publish_predictor
from runtime.serving.ridge_provider import load

TARGET = "submission_remaining_reasoning_tokens_log1p"


def experiment(cfg):
    directory = Path(cfg["project_root"]) / "difficulty/checkpoints/example"
    directory.mkdir(parents=True)
    write_json(directory / "split.json", {"train": ["a"], "validation": ["b"], "test": ["c"]})
    split_sha = hash_json(read_json(directory / "split.json"))
    trials = []
    for layer, validation, coef in [(3, 0.9, [1.0, -2.0]), (7, 0.8, [-1.0, 2.0])]:
        model = {"coef": np.array(coef), "intercept": np.array(0.5),
                 "feature_mean": np.array([0.25, 1.0]), "feature_scale": np.array([2.0, 0.5])}
        metadata = {"target": TARGET, "layer": layer, "alpha": 10.0, "native_site": layer,
                    "fit_scope": "train", "split_sha256": split_sha, "dataset_receipt_sha256": "abc"}
        path = directory / f"site-{layer}.npz"
        save_model(path, model, metadata)
        trials.append({"target": TARGET, "layer": layer, "alpha": 10.0, "status": "complete",
                       "validation": {"r2": validation, "count": 10}, "model": path.name,
                       "model_sha256": sha256(path)})
        if layer == 3:
            model["coef"] *= 2
            save_model(directory / "refit.npz", model, {**metadata, "fit_scope": "train_validation"})
    (directory / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials))
    write_json(directory / "selected.json", {"selection_metric": "validation_pooled_r2",
               "tie_tolerance": 1e-10, "targets": {TARGET: trials[0]}})
    write_json(directory / "metrics.json", {TARGET: {"selected": trials[0], "refit_model": "refit.npz",
               "test": {"r2": 0.1}}})
    write_json(directory / "experiment.json", {"config": {"model": cfg["model"]},
               "input": {"receipt_sha256": "abc"}, "split_sha256": split_sha})
    return directory


@pytest.mark.parametrize("scope,filename", [("refit", "refit.npz"), ("train", "site-3.npz")])
def test_published_ridge_reuses_prediction_and_selected_fit_scope(cfg, scope, filename):
    directory = experiment(cfg)
    result = publish_predictor(cfg, "online", experiment=directory, fit_scope=scope)
    contract = artifact_contract(result["manifest"], result["manifest_sha256"], cfg["model"]["fingerprint"])
    assert contract["sites"] == [3]
    assert contract["provenance"]["training_receipt_present"] is False
    assert not (directory / "receipt.json").exists()
    features = np.array([[1.0, 3.0], [-2.0, 4.0]], dtype=np.float16)
    predictor = load(Path(result["manifest"]).parent, contract)
    np.testing.assert_array_equal(predictor.predict(features[:, None]),
                                  predict_model(load_model(directory / filename), features))
    with pytest.raises(FileExistsError):
        publish_predictor(cfg, "online", experiment=directory)


def test_selection_rejects_false_champion_and_changed_split(cfg):
    directory = experiment(cfg)
    selected = read_json(directory / "selected.json")
    selected["targets"][TARGET]["layer"] = 7
    write_json(directory / "selected.json", selected)
    with pytest.raises(ValueError, match="validation-only"):
        publish_predictor(cfg, "wrong", experiment=directory)
    assert not (Path(cfg["project_root"]) / "runtime/parallel-artifacts/wrong").exists()
    selected["targets"][TARGET]["layer"] = 3
    write_json(directory / "selected.json", selected)
    write_json(directory / "split.json", {"changed": True})
    with pytest.raises(ValueError, match="split changed"):
        publish_predictor(cfg, "changed-split", experiment=directory)


def test_explicit_checkpoint_checks_target_site_and_payload_hash(cfg):
    directory = experiment(cfg)
    with pytest.raises(ValueError, match="target"):
        publish_predictor(cfg, "wrong-target", checkpoint=directory / "refit.npz", target="remaining_steps")
    with pytest.raises(ValueError, match="site"):
        publish_predictor(cfg, "wrong-site", checkpoint=directory / "refit.npz", sites=[7])
    result = publish_predictor(cfg, "explicit", checkpoint=directory / "refit.npz")
    manifest = Path(result["manifest"])
    (manifest.parent / "model.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        artifact_contract(manifest)


def test_external_provider_and_live_implementation_hash(cfg, tmp_path, monkeypatch):
    source = tmp_path / "test_online_provider.py"
    source.write_text("import numpy as np\nclass Predictor:\n    def predict(self, batch):\n"
                      "        return np.sum(batch, axis=(1, 2))\n"
                      "def load(directory, contract):\n    return Predictor()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    payload = tmp_path / "checkpoint.bin"
    payload.write_bytes(b"trained-external-estimator")
    result = publish_predictor(cfg, "custom", checkpoint=payload, loader="test_online_provider:load",
                               target="custom_remaining", sites=[1, 2], hidden_size=2)
    assert read_json(result["manifest"])["loader"] != RIDGE_LOADER
    source.write_text(source.read_text() + "\n# implementation changed\n")
    with pytest.raises(ValueError, match="implementation changed"):
        artifact_contract(result["manifest"])


def test_predictor_publish_uses_unified_cli(cfg, capsys):
    from core.cli import main

    directory = experiment(cfg)
    main(["--config", cfg["config_source"], "predictor-publish", "cli-provider",
          "--experiment", str(directory), "--target", "reasoning_length"])
    assert json.loads(capsys.readouterr().out)["sites"] == [3]


def test_close_only_capture_does_not_run_newline_mixed_diagnostic():
    from runtime.serving.startup import check_mixed_batches

    result = check_mixed_batches({"capture": {"anchors": ["think_close"]}}, "run", None, {})
    assert result == {"status": "skipped", "reason": "requires_reasoning_newline_capture"}


def test_parallel_plugin_drains_before_score_results():
    source = Path(__file__).parents[1] / "runtime/serving/vllm_plugin.py"
    text = source.read_text()
    assert 'if action == "results":\n                    # A completed generate' in text
    assert "await self.drain_once()" in text
