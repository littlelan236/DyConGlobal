"""Real tiny Qwen+PEFT service lifecycle, reference persistence and exact resume."""
import copy
from pathlib import Path

import pytest
import torch

from core.io import hash_json, read_json, sha256, write_json
from RL.artifacts import commit_receipt, verify_receipt
from RL.config import load
from RL.grpo.contract import validate_reference_files, validate_resume
from RL.modeling import adapter_fingerprint
from RL.service import _train_grpo, _source
from tests.lora_helpers import LORA, model_and_targets


def run_tiny(tmp_path, name, *, mode="fresh", source=None, stop=None):
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    config.lora = LORA.model_copy()
    config.run.attempt_id = name
    config.run.start.mode = mode
    config.run.start.checkpoint = str(source) if source else None
    config.validation.stop_after_update = stop
    config.training.learning_rate = .003
    config.training.scheduler = "linear"
    parent = _source(config)
    model, targets = model_and_targets(adapter=parent["adapter"] if parent else None)
    directory = tmp_path / name
    directory.mkdir()
    attempt = {"schema_version": 1, "attempt_id": name, "purpose": "engineering_smoke",
               "start_mode": mode, "parent": {k:v for k,v in parent.items() if k not in {"identity", "adapter"}} if parent else None}
    base = {"fingerprint": "a"*64}
    result = _train_grpo(config, root, directory, base, parent, attempt, model, targets,
                        optimizer_settings=config.training)
    return config, directory, model, result


def assert_tree_close(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            assert_tree_close(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a,b in zip(left,right):
            assert_tree_close(a,b)
    else:
        assert left == right


def test_service_resume_matches_continuous_reference_adapter_optimizer_scheduler_metrics(tmp_path):
    _, full_path, full_model, full = run_tiny(tmp_path, "full")
    _, paused_path, _, paused = run_tiny(tmp_path, "paused", stop=1)
    assert paused["status"] == "paused"
    cp = paused_path / "checkpoints/step-000001"
    config, resumed_path, resumed_model, resumed = run_tiny(tmp_path, "resumed", mode="resume", source=cp)
    assert resumed["status"] == "completed" and resumed["state"] == full["state"]
    assert resumed["state"]["group_cursor"] == 5
    assert resumed["state"]["consumed_branches"] == 15
    assert adapter_fingerprint(resumed_model) == adapter_fingerprint(full_model)
    meta = read_json(cp / "state.json")
    validate_reference_files(resumed_path, meta["contract"])
    assert sha256(full_path / "grpo/reference/adapter/adapter_model.safetensors") == sha256(resumed_path / "grpo/reference/adapter/adapter_model.safetensors")
    full_payload = torch.load(full_path / "checkpoints/step-000002/training-state.pt", weights_only=True)
    resumed_payload = torch.load(resumed_path / "checkpoints/step-000002/training-state.pt", weights_only=True)
    assert_tree_close(full_payload["optimizer"], resumed_payload["optimizer"])
    assert_tree_close(full_payload["scheduler"], resumed_payload["scheduler"])
    from RL.grpo.metrics import read_metric_history
    a = read_metric_history(full_path / "metrics.jsonl")
    b = read_metric_history(resumed_path / "metrics.jsonl")
    for x,y in zip(a,b):
        for k in ("loss", "k3_penalty_mean", "clip_fraction", "ratio_mean", "gradient_norm", "learning_rate", "group_cursor"):
            assert x[k] == pytest.approx(y[k], abs=1e-7)
    with pytest.raises(ValueError, match="already complete"):
        run_tiny(tmp_path, "completed-resume", mode="resume", source=full_path / "checkpoints/step-000002")


def test_warm_start_freezes_new_initial_adapter_and_resets_epoch(tmp_path):
    _, source, old_model, _ = run_tiny(tmp_path, "source")
    old_hash = hash_json(adapter_fingerprint(old_model))
    _, new, _, result = run_tiny(tmp_path, "warm", mode="warm_start", source=source / "adapter")
    assert result["state"]["group_cursor"] == 5
    identity = read_json(new / "attempt.json")["contract"]["grpo"]["reference_identity"]
    assert identity["adapter_hash"] == old_hash
    assert read_json(new / "attempt.json")["initial_state"]["group_cursor"] == 0


@pytest.mark.parametrize("field", ["dataset_receipt_sha256", "group_order_sha256", "advantage_config_sha256", "probability_mode", "eligible_group_ids"])
def test_changed_input_refuses_resume(field, tmp_path):
    config, directory, _, _ = run_tiny(tmp_path, "paused", stop=1)
    metadata = read_json(directory / "checkpoints/step-000001/state.json")
    proof = copy.deepcopy(metadata["contract"]["input"])
    proof[field] = ["different"] if field == "eligible_group_ids" else "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume(config, proof, metadata)


def test_reference_cache_tamper_refuses_resume(tmp_path):
    _, directory, _, _ = run_tiny(tmp_path, "paused", stop=1)
    cp = directory / "checkpoints/step-000001"
    (cp / "grpo/reference-logprobs.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inventory/hash"):
        run_tiny(tmp_path, "bad", mode="resume", source=cp)
