import json
from pathlib import Path

import numpy as np
import pytest

from core.io import hash_json, sha256, write_json
from difficulty.models.estimators import FFNMLEstimator
from difficulty.models.mlp_checkpoint import (
    ARCHITECTURE_ID,
    FEATURE_SCHEMA,
    MODEL_KIND,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from difficulty.probe_config import MLPProbe


def test_mlp_config_is_strict_and_targets_are_fixed():
    cfg = MLPProbe()
    assert cfg.public_site == 2
    assert set(cfg.targets) == {
        "submission_remaining_steps_log1p",
        "submission_remaining_reasoning_tokens_log1p",
    }
    with pytest.raises(ValueError):
        MLPProbe.model_validate({"targets": ["remaining_steps_log1p"], "surprise": True})
    with pytest.raises(ValueError):
        MLPProbe.model_validate({"device": "cuda:bad"})


def _metadata(model, target="submission_remaining_steps_log1p"):
    return {
        "schema_version": 1, "model_kind": MODEL_KIND, "architecture_id": ARCHITECTURE_ID,
        "feature_schema": FEATURE_SCHEMA, "attempt_id": "attempt", "dataset_id": "dataset",
        "split_id": "split", "dataset_receipt_sha256": "a" * 64,
        "split_receipt_sha256": "b" * 64, "split_sha256": "c" * 64,
        "sample_order_sha256": "d" * 64, "train_sample_ids_sha256": "e" * 64,
        "model_fingerprint": "fingerprint", "input_spec": {
            "public_sites": [2], "width": 3, "feature_transform": "single-site-vector", "dtype": "float32",
        }, "architecture": {"input_width": 3, "hidden_dim": 4, "dropout": 0.0,
                             "layernorm_eps": 1e-5, "activation": "gelu", "dtype": "float32"},
        "target": target, "target_label": "remaining_steps_to_submission", "target_transform": "log1p",
        "endpoint": "submission", "normalization": "layernorm", "fit_scope": "train", "seed": 42,
        "best_epoch": model.best_epoch_, "epochs_trained": model.epochs_trained_,
    }


def test_mlp_validation_history_and_safe_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    y = (x[:, 0] * 0.5 + 2).astype(np.float32)
    history = []
    model = FFNMLEstimator(hidden_dim=4, epochs=5, batch_size=8, patience=2, seed=7)
    model.fit(x[:24], y[:24], validation_data=(x[24:], y[24:]), epoch_callback=history.append)
    assert history and history[-1]["eval_loss"] is not None
    assert model.best_epoch_ == min(history, key=lambda row: (row["eval_loss"], row["epoch"]))["epoch"]
    assert str(next(model._module.parameters()).device) == "cpu"
    metadata = _metadata(model)
    path = tmp_path / "best.npz"
    save_checkpoint(path, model, metadata)
    loaded_metadata, _ = validate_checkpoint(path)
    assert loaded_metadata["input_spec"]["public_sites"] == [2]
    loaded = load_checkpoint(path)
    assert np.allclose(model.predict(x), loaded.predict(x), atol=1e-6)
    path.with_suffix(".json").write_text(path.with_suffix(".json").read_text().replace("attempt", "tampered"))
    with pytest.raises(ValueError, match="sidecar"):
        validate_checkpoint(path)


def test_mlp_cuda_device_is_explicit_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(8)
    x = rng.normal(size=(8, 3)).astype(np.float32)
    y = rng.normal(size=8).astype(np.float32)
    model = FFNMLEstimator(hidden_dim=4, epochs=1, batch_size=4, device="cuda:0")
    model.fit(x, y)
    assert model.device_ == "cuda:0"
    assert np.isfinite(model.predict(x)).all()
