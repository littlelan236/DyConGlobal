"""Frozen KL values, gradients, lifecycle and config identity regressions."""
import copy
from types import SimpleNamespace as NS

import pytest
import torch

from RL.config import Config, Grpo, load, schedule_contract
from RL.grpo.backend import GroupObjective
from RL.grpo.logprobs import model_target_logprobs
from RL.grpo.objective import grpo_objective
from RL.policy_reference import InitialPolicyScorer
from tests.lora_helpers import model_and_targets
from RL.modeling import enable_checkpointing
from tests.test_grpo_prefix import group


def scorer_for(model, tmp_path):
    model.save_pretrained(tmp_path / "adapter")
    return InitialPolicyScorer(model, tmp_path / "adapter")


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_recomputed_kl_matches_independent_model_over_multiple_updates(tmp_path, prefix, checkpoint):
    model, _ = model_and_targets()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_" in name:
                value.uniform_(-.1, .1)
    model.train()
    frozen = copy.deepcopy(model).eval()
    scorer = scorer_for(model, tmp_path)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_" in name:
                value.add_(.025)
    oracle = copy.deepcopy(model)
    if checkpoint:
        enable_checkpointing(model)
        enable_checkpointing(oracle)
    row = group(3, 3)
    settings = Grpo(kl_beta=.02, k3_cap=1., prefix_reuse={"mode": "boundary_grad" if prefix else "disabled"})
    optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.002) for m in (model, oracle)]
    # Intentionally unrelated historical ratio denominator: must not become pi_ref.
    denominator = {row.group_id: {bid: torch.full((len(c),), -4.5)
                                 for bid, c in zip(row.branch_ids, row.completion_token_ids)}}
    reference_before = {k: v.clone() for k, v in scorer.reference.items()}
    for _ in range(2):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        objective = GroupObjective(denominator, settings, 2, reference_scorer=scorer)
        objective.begin_group(model, row)
        try:
            for i, result in enumerate(objective.iter_results(model, row)):
                values = model_target_logprobs(oracle, row.branch_batch(i, "cpu"), 2)
                with torch.no_grad():
                    kl = model_target_logprobs(frozen, row.branch_batch(i, "cpu"), 2)
                expected = grpo_objective(values[None], denominator[row.group_id][str(i)][None],
                    [row.advantages[i]], torch.ones_like(values[None], dtype=torch.bool),
                    kl_beta=.02, k3_cap=1., kl_logp_reference=kl[None])
                torch.testing.assert_close(result.loss, expected.loss / row.group_size, atol=3e-7, rtol=3e-5)
                result.loss.backward()
                (expected.loss / row.group_size).backward()
                if prefix:
                    objective.after_branch_backward()
            objective.finish_group()
        finally:
            objective.close_group()
        for (name, actual), (_, expected) in zip(model.named_parameters(), oracle.named_parameters()):
            if actual.requires_grad:
                assert actual.grad is not None, name
                torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-4)
        for opt in optimizers:
            opt.step()
        assert model.training
        assert objective.kl_reference is None
    for key, value in scorer.reference.items():
        torch.testing.assert_close(value, reference_before[key], atol=0, rtol=0)


def test_scorer_restores_weights_and_mode_after_failure(tmp_path, monkeypatch):
    model, _ = model_and_targets()
    scorer = scorer_for(model, tmp_path)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(.1)
    from peft.utils.save_and_load import get_peft_model_state_dict
    before = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    model.train()
    def fail(*a, **kw):
        raise RuntimeError("injected reference failure")
    monkeypatch.setattr("RL.grpo.logprobs.model_target_logprobs", fail)
    with pytest.raises(RuntimeError, match="injected"):
        scorer.score_group(group(), block_size=2)
    assert model.training
    for key, value in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_kl_reference_is_detached_and_does_not_change_ppo_denominator():
    theta = torch.tensor([[-2., -1.]], requires_grad=True)
    reference = torch.tensor([[-3., -.5]], requires_grad=True)
    denominator = torch.tensor([[-2.2, -1.2]])
    args = (theta, denominator, [1.], torch.ones_like(theta, dtype=torch.bool))
    old = grpo_objective(*args, kl_beta=.02)
    new = grpo_objective(*args, kl_beta=.02, kl_logp_reference=reference)
    torch.testing.assert_close(new.policy_loss, old.policy_loss, atol=0, rtol=0)
    delta = reference.detach() - theta
    torch.testing.assert_close(new.k3_penalty, (torch.expm1(delta) - delta).mean())
    new.loss.backward()
    assert reference.grad is None
    disabled = grpo_objective(*args, kl_beta=0., kl_logp_reference=torch.full_like(theta, float("nan")))
    assert disabled.k3_penalty.item() == 0.


def test_shared_computation_roundtrip_and_resume_identity():
    from pathlib import Path
    root = Path(__file__).parents[1]
    old, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    original = schedule_contract(old)
    assert "kl_reference" not in original
    raw = old.model_dump()
    raw["objective"]["kl"]["computation"] = "recompute"
    new = Config.model_validate(raw)
    expected = schedule_contract(new)
    assert expected.pop("kl_reference") == {"policy": "initial_policy", "computation": "recompute"}
    assert expected == original
    assert Config.model_validate(new.model_dump()) == new
    raw["objective"]["kl"]["computation"] = "typo"
    with pytest.raises(ValueError):
        Config.model_validate(raw)


@pytest.mark.parametrize("source", ["baseline", "dycon"])
@pytest.mark.parametrize("beta,expected_calls", [(0., 0), (.02, 1)])
def test_updater_recompute_overrides_stale_reference_for_both_sources(monkeypatch, source, beta, expected_calls):
    from RL.online_runtime import (OnlineRow, OnlineRolloutBatch, SourceMass, TorchOnlineUpdater,
                                   collate_rows, online_row_masses, online_selected_logprobs, online_torch_objective)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.projection = torch.nn.Linear(4, 8)
        def forward(self, input_ids, attention_mask):
            return NS(logits=self.projection(self.embedding(input_ids).cumsum(1)))
    torch.manual_seed(123)
    model = Model()
    oracle = copy.deepcopy(model)
    row = OnlineRow("row", "task", source, "group", (1, 2, 3), (3,),
                    (-2.,), (float("nan"),), 1., "trajectory", 1, 0, True)
    batch = OnlineRolloutBatch(0, ("task",), "S0", "E0", (row,))
    mass = SourceMass(1., 0.) if source == "dycon" else SourceMass(0., 1.)
    calls = []
    def score(tensors):
        calls.append(1)
        # Distinct from the stored invalid reference and behavior probability.
        return torch.full_like(tensors["input_ids"][:, 1:], -3., dtype=torch.float)
    def make_optimizer(model, settings):
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    monkeypatch.setattr("RL.training.optimizer_and_scheduler", make_optimizer)
    updater = TorchOnlineUpdater(model, NS(grad_clip=1e6), source_mass=mass, kl_beta=beta,
                                 k3_cap=1., reference_scorer=score, kl_reference_mode="recompute")
    updater(batch, 0)
    assert len(calls) == expected_calls
    tensors = collate_rows((row,), device="cpu")
    current, mask = online_selected_logprobs(oracle, tensors)
    loss, _ = online_torch_objective(current, tensors["rollout_logprobs"],
        torch.full_like(current, -3.) if beta else None, torch.tensor([1.]), mask,
        [source], {source: [[0]]}, source_mass=mass, kl_beta=beta, k3_cap=1.)
    loss.backward()
    for actual, expected in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-5)


def test_recomputed_kl_service_resume_keeps_original_frozen_adapter(tmp_path, monkeypatch):
    # Exercise the real primary service, external PPO denominator, checkpoint
    # serialization and resume. The existing oracle also removes the external
    # source before resume, so recovery must use checkpoint-local artifacts.
    from tests import test_grpo_external_reference as lifecycle
    original = lifecycle.configuration
    def configured(*args, **kwargs):
        config = original(*args, **kwargs)
        config.objective.kl.computation = "recompute"
        config.grpo.prefix_reuse.mode = "boundary_grad"
        return config
    monkeypatch.setattr(lifecycle, "configuration", configured)
    lifecycle.test_external_service_resume_uses_local_cache_and_matches_forward_oracle(tmp_path, monkeypatch)
