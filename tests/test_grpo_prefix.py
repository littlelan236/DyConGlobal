"""Independent full-sequence oracle for hybrid state, gradients and lifetimes.

Predeclared FP32 tolerances: logprob 3e-6/3e-6, loss 3e-7/3e-6,
all LoRA gradients 2e-6/2e-4. Nonzero A and B, clipping and K3 active.
"""
import copy
import gc
from pathlib import Path
import weakref

import pytest
import torch

from RL.config import load
from RL.grpo.backend import GroupObjective
from RL.grpo.groups import GroupBatch
from RL.grpo.logprobs import model_target_logprobs
from RL.grpo.objective import grpo_objective
from RL.grpo.prefix import PrefixContext
from RL.modeling import enable_checkpointing
from RL.training import train_grpo
from tests.lora_helpers import model_and_targets


def group(prompt_length=5, branches=3, name="group"):
    prompt = [2 + i % 50 for i in range(prompt_length)]
    completions = [[3 + (i * 7 + j) % 50 for j in range([1, 2, 5, 67][i % 4])]
                   for i in range(branches)]
    ids = torch.zeros(branches, prompt_length + max(map(len, completions)), dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, completion in enumerate(completions):
        ids[i, :prompt_length + len(completion)] = torch.tensor(prompt + completion)
        attention[i, :prompt_length + len(completion)] = 1
        labels[i, prompt_length:prompt_length + len(completion)] = torch.tensor(completion)
    return GroupBatch(name, [str(i) for i in range(branches)], [prompt[:] for _ in completions],
        completions, ids, attention, labels, labels != -100, list(range(branches)),
        [1.0 if i % 2 else -.7 for i in range(branches)])


def model_pair(checkpointing):
    model, _ = model_and_targets()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.uniform_(-.15, .15)
    model.train()
    other = copy.deepcopy(model)
    if checkpointing:
        enable_checkpointing(model); enable_checkpointing(other)
    return model, other


def branch_loss(values, index, n):
    # Both signs, clipped/unclipped ratios, nonzero K3 derivative.
    reference = values.detach() + torch.linspace(-.3, .25, len(values), device=values.device)
    return grpo_objective(values[None], reference[None], [1. if index % 2 else -.7],
        torch.ones_like(values[None], dtype=torch.bool)).loss / n


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("prompt_length,branches", [(1, 2), (3, 3), (63, 4), (65, 3), (5, 32)])
def test_all_targets_lora_gradients_and_boundary_lifetimes(checkpointing, prompt_length, branches):
    oracle, shared = model_pair(checkpointing)
    row = group(prompt_length, branches)
    prefix = PrefixContext(shared, row.prompt_token_ids[0])
    root_refs = [weakref.ref(t) for t in prefix.roots]
    source = [t.detach().clone() for t in prefix.roots]
    losses = [0., 0.]
    for i, completion in enumerate(row.completion_token_ids):
        full = model_target_logprobs(oracle, row.branch_batch(i, "cpu"), 2)
        actual = prefix.branch_logprobs(shared, completion, 2)
        leaf_refs = [weakref.ref(t) for t in prefix.leaves]
        current_ref = weakref.ref(actual)
        torch.testing.assert_close(actual, full, atol=3e-6, rtol=3e-6)
        a, b = branch_loss(full, i, branches), branch_loss(actual, i, branches)
        torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-6)
        losses[0] += float(a.detach()); losses[1] += float(b.detach())
        a.backward(); b.backward()
        del a, b, full, actual
        prefix.after_branch_backward()
        gc.collect()
        assert current_ref() is None and all(ref() is None for ref in leaf_refs)
        assert all(t is None or t.grad_fn is None for t in prefix.gradients)
        for root, saved in zip(prefix.roots, source, strict=True):
            torch.testing.assert_close(root, saved, atol=0, rtol=0)
        del root
    prefix.finish()
    worst = ("", 0.)
    nonzero = 0
    for (name, p), (_, q) in zip(oracle.named_parameters(), shared.named_parameters(), strict=True):
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None, name
            error = float((p.grad - q.grad).abs().max())
            if error > worst[1]: worst = name, error
            nonzero += int(bool(p.grad.abs().any()))
            torch.testing.assert_close(q.grad, p.grad, atol=2e-6, rtol=2e-4, msg=name)
    assert nonzero == sum(p.requires_grad for p in oracle.parameters())
    print({"P": prompt_length, "N": branches, "layer_checkpoint": checkpointing,
           "worst_parameter": worst[0], "max_gradient_absolute_error": worst[1], "loss": losses})
    assert prefix.metrics["prefix_forward_calls"] == prefix.metrics["prefix_backward_calls"] == 1
    assert prefix.metrics["branch_backward_calls"] == branches
    prefix.close(); gc.collect()
    assert not prefix.roots and not prefix.gradients
    assert all(ref() is None for ref in root_refs)


def test_omitting_prefix_backward_is_detected():
    oracle, shared = model_pair(True)
    row = group(7)
    prefix = PrefixContext(shared, row.prompt_token_ids[0])
    for i, completion in enumerate(row.completion_token_ids):
        full = model_target_logprobs(oracle, row.branch_batch(i, "cpu"), 2)
        actual = prefix.branch_logprobs(shared, completion, 2)
        branch_loss(full, i, 3).backward(); branch_loss(actual, i, 3).backward()
        prefix.after_branch_backward()
    errors = [float((p.grad - (q.grad if q.grad is not None else torch.zeros_like(p))).abs().max())
              for p, q in zip(oracle.parameters(), shared.parameters(), strict=True) if p.requires_grad]
    assert max(errors) > 1e-4
    prefix.close()


def test_group_windows_tail_and_parameter_gradient_accumulation():
    oracle, shared = model_pair(True)
    rows = [group(3, 2, "a"), group(5, 3, "b"), group(7, 4, "c")]
    settings, _ = load(Path(__file__).parents[1] / "RL/experiments/grpo-offline-k3-smoke.yaml", Path(__file__).parents[1])
    with torch.no_grad():
        anchor = {g.group_id: {bid: model_target_logprobs(oracle, g.branch_batch(i, "cpu"), 2).detach()
                  for i, bid in enumerate(g.branch_ids)} for g in rows}
    snapshots = []
    for model, enabled in [(oracle, False), (shared, True)]:
        cfg = settings.grpo.model_copy(deep=True)
        cfg.prefix_reuse.mode = "boundary_grad" if enabled else "disabled"
        objective = GroupObjective(anchor, cfg, 2)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        gradients, metrics = [], []
        state = train_grpo(model, rows, objective, optimizer, scheduler, group_batch_size=2,
            on_gradients=lambda: gradients.append({n: p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}),
            on_window_start=objective.reset, on_update=lambda _, m: metrics.append(m))
        assert state["group_cursor"] == 3 and [m["group_count"] for m in metrics] == [2, 1]
        assert objective.prefix is None
        snapshots.append((gradients, state))
    assert snapshots[0][1] == snapshots[1][1]
    for a,b in zip(snapshots[0][0], snapshots[1][0], strict=True):
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=2e-6, rtol=2e-4, msg=name)


@pytest.mark.parametrize("failure", ["boundary_nan", "parameter_nan", "branch_exception", "prefix_exception"])
def test_failed_window_cleans_context_and_gradients_without_step(failure, monkeypatch):
    model, _ = model_pair(True)
    row = group()
    settings, _ = load(Path(__file__).parents[1] / "RL/experiments/grpo-offline-k3-smoke.yaml", Path(__file__).parents[1])
    settings.grpo.prefix_reuse.mode = "boundary_grad"
    with torch.no_grad():
        anchor = {row.group_id: {bid: model_target_logprobs(model, row.branch_batch(i, "cpu"), 2)
                  for i, bid in enumerate(row.branch_ids)}}
    objective = GroupObjective(anchor, settings.grpo, 2)
    refs = []
    original = PrefixContext.branch_logprobs
    def fault(self, *args):
        refs.append(weakref.ref(self))
        if failure == "branch_exception" and self.metrics["branch_count"] == 1:
            raise RuntimeError("injected branch exception")
        result = original(self, *args)
        if failure == "boundary_nan":
            self.leaves[-1].register_hook(lambda g: g * float("nan"))
        return result
    monkeypatch.setattr(PrefixContext, "branch_logprobs", fault)
    if failure == "parameter_nan":
        next(p for p in model.parameters() if p.requires_grad).register_hook(lambda g: g * float("nan"))
    if failure == "prefix_exception":
        monkeypatch.setattr(PrefixContext, "finish", lambda _: (_ for _ in ()).throw(RuntimeError("injected prefix exception")))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    before = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    with pytest.raises((RuntimeError, ValueError)):
        train_grpo(model, [row], objective, optimizer, scheduler)
    assert objective.prefix is None
    for n,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is None
            torch.testing.assert_close(p, before[n], atol=0, rtol=0)
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_unused_boundaries_and_nonfinite_accumulator():
    model, _ = model_pair(True)
    prefix = PrefixContext(model, [2])
    prefix.branch_logprobs(model, [3], 1).sum().backward()
    assert all(leaf.grad is None for leaf in prefix.leaves[:-1])
    prefix.after_branch_backward()
    assert all(g is None for g in prefix.gradients[:-1])
    prefix.gradients[-1].fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        prefix.finish()
    prefix.close()


@pytest.mark.parametrize("case", ["resume", "warm_start"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_boundary_service_complete_checkpoint_lifecycle(case, dtype, tmp_path, monkeypatch):
    from tests import test_grpo_lifecycle as lifecycle
    original_load = lifecycle.load
    original_model = lifecycle.model_and_targets
    def configured(*args, **kwargs):
        cfg, source = original_load(*args, **kwargs)
        cfg.grpo.prefix_reuse.mode = "boundary_grad"
        cfg.model.dtype = dtype
        return cfg, source
    def checkpointed(*args, **kwargs):
        model, targets = original_model(*args, **kwargs)
        enable_checkpointing(model)
        return model, targets
    monkeypatch.setattr(lifecycle, "load", configured)
    monkeypatch.setattr(lifecycle, "model_and_targets", checkpointed)
    if case == "resume":
        lifecycle.test_service_resume_matches_continuous_reference_adapter_optimizer_scheduler_metrics(tmp_path)
    else:
        lifecycle.test_warm_start_freezes_new_initial_adapter_and_resets_epoch(tmp_path)


@pytest.mark.parametrize("policy,offload", [("layer_group", "none"), ("selective_mlp", "none")])
def test_unvalidated_combinations_rejected_before_model_load(policy, offload):
    from RL.config import Config
    cfg, _ = load(Path(__file__).parents[1] / "RL/experiments/grpo-offline-k3-smoke.yaml", Path(__file__).parents[1])
    data = cfg.model_dump()
    data["grpo"]["prefix_reuse"]["mode"] = "boundary_grad"
    data["training"].update(checkpoint_policy=policy)
    data["training"]["memory"].update(activation_offload=offload, mlp_chunk_tokens=2)
    with pytest.raises(ValueError, match="boundary_grad"):
        Config.model_validate(data)


def test_disabled_schedule_preserves_legacy_and_backend_changes_reject_resume():
    from RL.config import schedule_contract
    from RL.grpo.contract import validate_resume
    cfg, _ = load(Path(__file__).parents[1] / "RL/experiments/grpo-offline-k3-smoke.yaml", Path(__file__).parents[1])
    legacy = schedule_contract(cfg)
    assert "prefix_reuse" not in legacy["grpo"]
    cfg.grpo.prefix_reuse.mode = "boundary_grad"
    assert schedule_contract(cfg) != legacy
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume(cfg, {}, {"contract": {"schedule": legacy, "input": {}}})


def test_fp32_configuration_contract_and_unsupported_combinations():
    from RL.config import Config, schedule_contract
    root = Path(__file__).parents[1]
    cfg, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    original = schedule_contract(cfg)
    data = cfg.model_dump()
    data["model"]["dtype"] = "float32"
    fp32 = Config.model_validate(data)
    assert schedule_contract(fp32) != original
    for length, flash in [(4096, True), (512, True)]:
        bad = copy.deepcopy(data)
        bad["validation"]["max_sequence_length"] = length
        bad["training"]["memory"].update(exact_length=True, loss_chunk_tokens=16,
            attention_policy="flash_only" if flash else "auto")
        with pytest.raises(ValueError, match="FP32"):
            Config.model_validate(bad)
    bad = copy.deepcopy(data)
    bad["grpo"]["reference_cache"] = {"source":"dataset_probability", "stream":"online_raw",
        "dtype_compatibility":"same_dtype"}
    with pytest.raises(ValueError):
        Config.model_validate(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_generator_releases_branch_and_previous_group_graphs(dtype):
    model, _ = model_pair(True)
    model.to(dtype)
    # Match PEFT production behavior: trainable adapters keep FP32 weights.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    rows = [group(3, 3, "a"), group(5, 2, "b"), group(1, 2, "c")]
    root = Path(__file__).parents[1]
    cfg, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    cfg.grpo.prefix_reuse.mode = "boundary_grad"
    with torch.no_grad():
        anchor = {g.group_id: {bid: model_target_logprobs(model,g.branch_batch(i,"cpu"),2).detach()
                  for i,bid in enumerate(g.branch_ids)} for g in rows}
    class Tracking(GroupObjective):
        previous_roots = ()
        branch_refs = ()
        cleaned_branches = 0
        def begin_group(self, *args):
            gc.collect()
            assert all(ref() is None for ref in self.previous_roots)
            super().begin_group(*args)
            self.previous_roots = tuple(weakref.ref(t) for t in self.prefix.roots)
        def _branch_result(self, *args):
            result = super()._branch_result(*args)
            self.branch_refs = tuple(weakref.ref(t) for t in self.prefix.leaves) + (
                weakref.ref(result.loss),weakref.ref(result.policy_loss),weakref.ref(result.k3_penalty))
            return result
        def after_branch_backward(self):
            super().after_branch_backward()
            gc.collect()
            assert all(ref() is None for ref in self.branch_refs)
            self.cleaned_branches += 1
    objective = Tracking(anchor,cfg.grpo,2)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    state = train_grpo(model,rows,objective,optimizer,scheduler,group_batch_size=2)
    gc.collect()
    assert state["global_step"] == 2 and objective.cleaned_branches == 7
    assert objective.prefix is None and all(ref() is None for ref in objective.previous_roots)


def test_prefix_with_existing_chunked_mlp_and_layer_checkpointing():
    from RL.memory import install_mlp_backend
    oracle, shared = model_pair(True)
    install_mlp_backend(oracle,2); install_mlp_backend(shared,2)
    row = group(65,3)
    prefix = PrefixContext(shared,row.prompt_token_ids[0])
    try:
        for i,completion in enumerate(row.completion_token_ids):
            a=model_target_logprobs(oracle,row.branch_batch(i,"cpu"),2)
            b=prefix.branch_logprobs(shared,completion,2)
            torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-6)
            branch_loss(a,i,3).backward(); branch_loss(b,i,3).backward()
            del a,b
            prefix.after_branch_backward()
        prefix.finish()
        for (name,p),(_,q) in zip(oracle.named_parameters(),shared.named_parameters(),strict=True):
            if p.requires_grad:
                torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4,msg=name)
    finally:
        prefix.close()
