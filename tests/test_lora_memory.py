"""Numerical, graph-lifetime and parameter-contract tests for memory backends."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from RL.config import Config, Memory
from RL.memory import (
    LossBackend,
    PositionChunkedMLP,
    chunked_causal_ce,
    configure_checkpointing,
    exact_length_batch,
    install_mlp_backend,
    local_saved_tensors,
    text_backbone,
)
from RL.modeling import enable_checkpointing
from RL.training import optimizer_and_scheduler, train
from RL.validation import causal_ce
from tests.lora_helpers import batches, model_and_targets, settings


@pytest.mark.parametrize("chunk", [1, 3, 128])
def test_chunked_ce_loss_hidden_grad_and_frozen_head(chunk):
    torch.manual_seed(5)
    hidden = torch.randn(2, 7, 12, requires_grad=True)
    other = hidden.detach().clone().requires_grad_(True)
    head = torch.nn.Linear(12, 19).requires_grad_(False)
    original = copy.deepcopy(head.state_dict())
    labels = torch.randint(0, 19, (2, 7))
    labels[0, :3] = -100
    labels[1, -2:] = -100
    expected = F.cross_entropy(head(hidden[:, :-1]).reshape(-1, 19), labels[:, 1:].reshape(-1),
                               reduction="sum", ignore_index=-100)
    actual = chunked_causal_ce(other, labels, head, chunk)
    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(other.grad, hidden.grad, atol=3e-7, rtol=3e-6)
    assert all(p.grad is None for p in head.parameters())
    for key in original:
        torch.testing.assert_close(original[key], head.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_tiny_qwen_full_text_prefix_loss_all_adapter_gradients(checkpointed, exact):
    dense, _ = model_and_targets()
    chunked, _ = model_and_targets()
    if checkpointed:
        enable_checkpointing(dense)
        enable_checkpointing(chunked)
    batch = batches()[1]
    dense.train()
    chunked.train()
    raw = dense.get_base_model()
    with torch.no_grad():
        hidden = text_backbone(dense)(input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
        full = raw(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        torch.testing.assert_close(raw.lm_head(hidden), full)
    expected, count = causal_ce(dense, batch)
    actual, other_count = LossBackend(Memory(loss_chunk_tokens=2, exact_length=exact))(chunked, batch)
    assert other_count == count
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
    expected.backward()
    actual.backward()
    for (name, p), (other_name, q) in zip(dense.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-7, rtol=3e-4)
        else:
            assert q.grad is None


@pytest.mark.parametrize("use_checkpoint", [None, False, True])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("copy_hooks", [False, True])
def test_position_mlp_peft_gradients_checkpoint_autocast_hooks(use_checkpoint, autocast, copy_hooks):
    model, _ = model_and_targets()
    original = copy.deepcopy(text_backbone(model).layers[0].mlp)
    other = copy.deepcopy(original)
    # Nonzero B exercises both adapter factors and input gradients.
    with torch.no_grad():
        for name, p in original.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.03)
    other.load_state_dict(original.state_dict())
    versions = {name: p._version for name, p in other.named_parameters()}
    chunked = PositionChunkedMLP(other, 3)
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    from contextlib import nullcontext
    hooks = (lambda: torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach().clone(), lambda t: t)) if copy_hooks else nullcontext
    with hooks(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        first = original(x)
        second = chunked(y) if use_checkpoint is None else checkpoint(chunked, y, use_reentrant=use_checkpoint)
    torch.testing.assert_close(first, second, atol=0.001 if autocast else 2e-7, rtol=0.02 if autocast else 2e-5)
    first.float().square().sum().backward()
    second.float().square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=0.001 if autocast else 3e-7, rtol=0.04 if autocast else 3e-5)
    for (name, p), (other_name, q) in zip(original.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=0.002 if autocast else 3e-7,
                                       rtol=0.05 if autocast else 3e-5)
            assert p.grad.norm() > 0
        else:
            assert p.grad is q.grad is None
        assert q._version == versions[name]


def test_mlp_parameter_version_guard_dropout_and_higher_order_errors():
    model, _ = model_and_targets()
    backend = PositionChunkedMLP(text_backbone(model).layers[0].mlp, 3)
    hidden = torch.randn(1, 7, 32, requires_grad=True)
    output = backend(hidden)
    with torch.no_grad():
        next(backend.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="version"):
        output.sum().backward()
    output = backend(hidden)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(output.sum(), hidden, create_graph=True)
    head = torch.nn.Linear(32, 64).requires_grad_(False)
    loss = chunked_causal_ce(hidden, torch.ones((1, 7), dtype=torch.long), head, 2)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(loss, hidden, create_graph=True)
    backend.dense.add_module("unsupported_dropout", torch.nn.Dropout(0.1))
    with pytest.raises(ValueError, match="dropout"):
        PositionChunkedMLP(backend.dense, 2)


def test_short_recompute_graphs_override_outer_saved_tensor_hooks():
    head = torch.nn.Linear(12, 19).requires_grad_(False)
    x = torch.randn(1, 7, 12, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t.shape), t.detach().clone())[1], lambda t: t):
        loss = chunked_causal_ce(x, torch.ones((1, 7), dtype=torch.long), head, 2)
        before = len(saved)
        loss.backward()
    assert len(saved) == before
    assert head.weight.shape not in saved
    with local_saved_tensors():
        assert True


def test_combined_optimizer_partial_window_matches_reference():
    dense, _ = model_and_targets()
    chunked, _ = model_and_targets()
    original_names = [(n, id(p)) for n, p in chunked.named_parameters()]
    install_mlp_backend(chunked, 3)
    assert original_names == [(n, id(p)) for n, p in chunked.named_parameters()]
    enable_checkpointing(dense)
    enable_checkpointing(chunked)
    first, first_schedule = optimizer_and_scheduler(dense, settings(2))
    second, second_schedule = optimizer_and_scheduler(chunked, settings(2))
    kwargs = dict(max_updates=2, gradient_accumulation_steps=2, grad_clip=1., vocab_size=64, max_length=32)
    a = train(dense, batches(), causal_ce, first, first_schedule, **kwargs)
    b = train(chunked, batches(), LossBackend(Memory(loss_chunk_tokens=2, mlp_chunk_tokens=3, exact_length=True)),
              second, second_schedule, **kwargs)
    assert a == b and a["consumed_microbatches"] == 3
    for p, q in zip(dense.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=3e-7, rtol=3e-4)


def test_exact_length_rejects_left_padding_holes_batching_and_supervised_padding():
    batch = batches()[1]
    trimmed = exact_length_batch(batch)
    assert trimmed["input_ids"].shape == (1, 7)
    for bad in [dict(batch, attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])),
                dict(batch, attention_mask=torch.tensor([[1, 1, 0, 1, 1, 1, 1, 0]])),
                {key: value.repeat(2, 1) for key, value in batch.items()}]:
        with pytest.raises(ValueError):
            exact_length_batch(bad)


@pytest.mark.parametrize("policy", ["layer", "selective_mlp", "layer_group"])
def test_checkpoint_schedules_preserve_mixed_attention_and_all_adapter_updates(policy):
    from types import SimpleNamespace
    dense, _ = model_and_targets()
    other, _ = model_and_targets()
    enable_checkpointing(dense)
    install_mlp_backend(other, 3)
    config = SimpleNamespace(training=SimpleNamespace(
        checkpoint_policy=policy, checkpoint_group_layers=2, memory=Memory(mlp_chunk_tokens=3)))
    configure_checkpointing(other, config, 2**30)
    expected, _ = causal_ce(dense, batches()[1])
    actual, _ = LossBackend(Memory(loss_chunk_tokens=2, mlp_chunk_tokens=3, exact_length=True))(other, batches()[1])
    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    for (name, p), (other_name, q) in zip(dense.named_parameters(), other.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-7, rtol=3e-4)


def test_memory_schema_and_host_budget_reject_unsafe_configs(tmp_path):
    import json
    from pathlib import Path

    import yaml

    from RL.resources import memory_plan
    root = Path(__file__).parents[1]
    original = yaml.safe_load((root / "RL/experiments/lora-smoke.yaml").read_text())
    original["model"]["path"] = str(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(dict(text_config=dict(
        hidden_size=4096, num_hidden_layers=32, vocab_size=248320))))
    original["validation"]["max_sequence_length"] = 24576
    with pytest.raises(ValueError, match="Long sequences"):
        Config.model_validate(original)
    original["training"]["memory"] = dict(loss_chunk_tokens=512, exact_length=True, attention_policy="flash_only")
    config = Config.model_validate(original)
    plan = memory_plan(config)
    assert plan["temporary_fp32_logits_bytes"] == 512 * 248320 * 4
    assert not plan["capacity_verified"]
    config.training.memory.activation_offload = "checkpoint_boundary"
    config.training.memory.host_budget_gib = 0.1
    config.preflight.check_resources = True
    with pytest.raises(ValueError, match="budget"):
        memory_plan(config)


def test_combined_resume_matches_continuous_and_rejects_strategy_change(tmp_path):
    from RL.checkpoints import adapter_contract, restore_checkpoint, save_checkpoint
    from tests.lora_helpers import LORA, stream

    memory = Memory(loss_chunk_tokens=2, mlp_chunk_tokens=3, exact_length=True)
    contract = {"memory": memory.model_dump(), "checkpoint_policy": "layer", "kernel": "pinned-tiny-v1"}
    kwargs = dict(gradient_accumulation_steps=2, grad_clip=1., vocab_size=64, max_length=32)

    def prepared(adapter=None, targets=None):
        model, actual = model_and_targets(adapter=adapter, expected_targets=targets)
        install_mlp_backend(model, memory.mlp_chunk_tokens)
        enable_checkpointing(model)
        optimizer, scheduler = optimizer_and_scheduler(model, settings(4))
        return model, actual, optimizer, scheduler

    continuous, _, optimizer, scheduler = prepared()
    expected = train(continuous, stream(8), LossBackend(memory), optimizer, scheduler, max_updates=4, **kwargs)
    paused, targets, optimizer, scheduler = prepared()
    state = train(paused, stream(4), LossBackend(memory), optimizer, scheduler, max_updates=2, **kwargs)
    identity = adapter_contract({"fingerprint": "a" * 64}, targets, LORA)
    checkpoint_path = tmp_path / "step-000002"
    save_checkpoint(paused, optimizer, scheduler, state, checkpoint_path, contract, identity)
    resumed, _, optimizer, scheduler = prepared(checkpoint_path / "adapter", targets)
    for changed in [dict(contract, memory={**memory.model_dump(), "loss_chunk_tokens": 4}),
                    dict(contract, kernel="different-kernel"),
                    dict(contract, checkpoint_policy="layer_group")]:
        with pytest.raises(ValueError, match="contract mismatch"):
            restore_checkpoint(checkpoint_path, optimizer, scheduler, changed, identity)
    restored = restore_checkpoint(checkpoint_path, optimizer, scheduler, contract, identity)
    actual = train(resumed, stream(4, offset=4), LossBackend(memory), optimizer, scheduler,
                   max_updates=4, state=restored, **kwargs)
    assert actual == expected
    for p, q in zip(continuous.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)
