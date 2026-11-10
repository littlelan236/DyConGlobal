import copy
from pathlib import Path

import pytest
import torch

from RL.artifacts import commit_receipt, verify_receipt
from RL.checkpoints import (
    adapter_contract,
    export_adapter,
    restore_checkpoint,
    save_checkpoint,
    verify_adapter,
    verify_checkpoint,
)
from RL.modeling import adapter_fingerprint, inject
from RL.training import optimizer_and_scheduler, train
from RL.validation import causal_ce, probe_logits
from tests.lora_helpers import LORA, batches, model_and_targets, raw_model, settings, stream
from tests.test_lora_training import kwargs


def contract(targets, lora=LORA):
    return adapter_contract({"fingerprint": "a" * 64}, targets, lora)


def test_adapter_round_trip_disable_and_warm_start(tmp_path):
    model, targets = model_and_targets()
    base_logits = probe_logits(model, batches()[0])
    optimizer, scheduler = optimizer_and_scheduler(model, settings(2))
    train(model, stream(4), causal_ce, optimizer, scheduler, **kwargs(model, 2))
    expected_logits = probe_logits(model, batches()[0])
    assert not torch.equal(base_logits, expected_logits)
    owner = tmp_path / "source"
    export_adapter(model, owner / "adapter", contract(targets))
    commit_receipt(owner, "lora-attempt-v1", status="completed")
    receipt_bytes = (owner / "receipt.json").read_bytes()
    verify_adapter(owner / "adapter", contract(targets))
    reloaded, loaded_targets = model_and_targets(adapter=owner / "adapter", expected_targets=targets)
    assert loaded_targets == targets
    torch.testing.assert_close(probe_logits(reloaded, batches()[0]), expected_logits, atol=1e-7, rtol=1e-6)
    with reloaded.disable_adapter():
        torch.testing.assert_close(probe_logits(reloaded, batches()[0]), base_logits, atol=1e-7, rtol=1e-6)
    assert adapter_fingerprint(reloaded) == adapter_fingerprint(model)
    before = adapter_fingerprint(reloaded)
    new_optimizer, new_scheduler = optimizer_and_scheduler(reloaded, settings(1))
    assert not new_optimizer.state and new_scheduler.last_epoch == 0
    state = train(reloaded, stream(2), causal_ce, new_optimizer, new_scheduler, **kwargs(reloaded, 1))
    assert state["global_step"] == 1 and state["consumed_microbatches"] == 2
    assert before != adapter_fingerprint(reloaded)
    assert (owner / "receipt.json").read_bytes() == receipt_bytes
    verify_receipt(owner)


def test_resume_matches_uninterrupted_with_rng_and_complete_schedule(tmp_path):
    lora = LORA.model_copy(update={"dropout": 0.2})
    uninterrupted, _ = model_and_targets(lora)
    opt, sched = optimizer_and_scheduler(uninterrupted, settings(4))
    full_metrics = []
    final = train(uninterrupted, stream(8), causal_ce, opt, sched, **kwargs(uninterrupted, 4),
                  on_update=lambda _, metrics: full_metrics.append(metrics))
    paused, targets = model_and_targets(lora)
    opt, sched = optimizer_and_scheduler(paused, settings(4))
    state = train(paused, stream(4), causal_ce, opt, sched, **kwargs(paused, 2))
    cp = tmp_path / "checkpoints/step-000002"
    identity = contract(targets, lora)
    resume_contract = {"schedule": "4 updates linear", "input": "fixed input v1"}
    save_checkpoint(paused, opt, sched, state, cp, resume_contract, identity)
    verify_checkpoint(cp)
    reloaded, _ = model_and_targets(lora, adapter=cp / "adapter", expected_targets=targets)
    restored_opt, restored_sched = optimizer_and_scheduler(reloaded, settings(4))
    restored_state = restore_checkpoint(cp, restored_opt, restored_sched, resume_contract, identity)
    assert restored_state == state and restored_sched.last_epoch == 2
    resumed_metrics = []
    result = train(reloaded, stream(4, offset=4), causal_ce, restored_opt, restored_sched,
                   **kwargs(reloaded, 4), state=restored_state,
                   on_update=lambda _, metrics: resumed_metrics.append(metrics))
    assert result == final
    assert [m["learning_rate"] for m in resumed_metrics] == [m["learning_rate"] for m in full_metrics[2:]]
    assert [m["loss"] for m in resumed_metrics] == pytest.approx([m["loss"] for m in full_metrics[2:]], abs=1e-7)
    for (name, p), (other_name, q) in zip(uninterrupted.named_parameters(), reloaded.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)


def test_checkpoint_and_adapter_corruption_rejected(tmp_path):
    model, targets = model_and_targets()
    opt, sched = optimizer_and_scheduler(model, settings(1))
    state = train(model, stream(2), causal_ce, opt, sched, **kwargs(model, 1))
    cp = tmp_path / "checkpoints/step-000001"
    identity = contract(targets)
    save_checkpoint(model, opt, sched, state, cp, {"schedule": "one"}, identity)
    with pytest.raises(ValueError, match="contract mismatch"):
        restore_checkpoint(cp, opt, sched, {"schedule": "changed learning rate"}, identity)
    other = copy.deepcopy(identity)
    other["base_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_adapter(cp / "adapter", other)
    with pytest.raises(ValueError, match="warm_start"):
        verify_checkpoint(cp / "adapter")
    with pytest.raises(FileExistsError):
        save_checkpoint(model, opt, sched, state, cp, {}, identity)
    (cp / "training-state.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_checkpoint(cp)
    (cp / "receipt.json").unlink()
    with pytest.raises(ValueError, match="receipt.json missing"):
        verify_checkpoint(cp)


def test_uncommitted_adapter_and_foreign_targets_rejected(tmp_path):
    model, targets = model_and_targets()
    export_adapter(model, tmp_path / "adapter", contract(targets))
    with pytest.raises(ValueError, match="receipt.json missing"):
        verify_adapter(tmp_path / "adapter")
    with pytest.raises(ValueError, match="targets/shapes"):
        model_and_targets(adapter=tmp_path / "adapter", expected_targets=targets[:-1])
    (tmp_path / "adapter/adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="content/hash"):
        verify_adapter(tmp_path / "adapter", verify_owner=False)


def test_checkpoint_requires_cleared_gradients(tmp_path):
    model, targets = model_and_targets()
    opt, sched = optimizer_and_scheduler(model, settings(1))
    value, _ = causal_ce(model, batches()[0])
    value.backward()
    with pytest.raises(ValueError, match="cleared"):
        save_checkpoint(model, opt, sched, {"global_step": 1}, Path(tmp_path) / "step-000001", {}, contract(targets))


def test_large_target_set_retains_exact_paths_across_peft_export_and_reload(tmp_path):
    model, targets = inject(raw_model(layers=4), LORA)
    assert len(targets) > 20
    source = tmp_path / "adapter"
    export_adapter(model, source, contract(targets))
    commit_receipt(tmp_path, "lora-attempt-v1", status="completed")
    verify_adapter(source, contract(targets))
    reloaded, actual = inject(raw_model(layers=4), LORA, adapter=source, expected_targets=targets)
    assert actual == targets
    assert reloaded.peft_config["default"].target_modules == {t["path"] for t in targets}
    torch.testing.assert_close(probe_logits(model, batches()[0]), probe_logits(reloaded, batches()[0]),
                               atol=1e-7, rtol=1e-6)
