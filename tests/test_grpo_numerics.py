"""Dense Qwen+LoRA oracle for selected projection, clipping and native targets."""
import copy
from pathlib import Path

import pytest
import torch

from RL.config import load
from RL.grpo.data import prepare, tensor_groups
from RL.grpo.logprobs import model_target_logprobs
from RL.grpo.objective import grpo_objective, selected_logprobs
from tests.lora_helpers import model_and_targets


def fixture_groups():
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    raw, _ = prepare(config)
    return config, list(tensor_groups(raw))


@pytest.mark.parametrize("block_size", [1, 2, 128])
def test_chunked_logprobs_objective_and_lora_gradients_match_dense(block_size):
    _, groups = fixture_groups()
    group = groups[0]
    dense, _ = model_and_targets()
    chunked = copy.deepcopy(dense)
    batch = group.branch_batch(1, "cpu")
    dense.eval(); chunked.eval()
    logits = dense(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    values, mask = selected_logprobs(logits, batch["input_ids"], batch["labels"], batch["loss_mask"])
    oracle = values[mask]
    actual = model_target_logprobs(chunked, batch, block_size)
    torch.testing.assert_close(actual, oracle, atol=1e-6, rtol=1e-6)
    assert len(actual) == len(group.completion_token_ids[1])
    # Perturb reference to test active K3 and ratio derivatives, not just CE.
    reference = oracle.detach() + torch.linspace(-.15, .2, len(oracle))
    dense_result = grpo_objective(oracle[None], reference[None], [-.5], torch.ones_like(oracle[None], dtype=torch.bool))
    chunk_result = grpo_objective(actual[None], reference[None], [-.5], torch.ones_like(actual[None], dtype=torch.bool))
    torch.testing.assert_close(dense_result.loss, chunk_result.loss, atol=1e-7, rtol=1e-6)
    dense_result.loss.backward(); chunk_result.loss.backward()
    for (name, p), (other, q) in zip(dense.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-5)


def test_equal_reference_has_zero_scalar_nonzero_policy_gradient_and_detached_advantage():
    current = torch.tensor([[-3., -2.], [-2., -4.]], requires_grad=True)
    reference = current.detach().clone().requires_grad_(True)
    advantages = torch.tensor([1., -1.], requires_grad=True)
    result = grpo_objective(current, reference, advantages, torch.ones_like(current, dtype=torch.bool))
    assert result.loss.item() == pytest.approx(0.)
    assert result.k3_penalty.item() == pytest.approx(0.)
    result.loss.backward()
    assert current.grad.norm() > 0
    assert reference.grad is None and advantages.grad is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1000., -1000.])
def test_nonfinite_or_overflow_ratio_k3_rejected(bad):
    with pytest.raises(ValueError, match="Nonfinite"):
        grpo_objective(torch.tensor([[bad]], requires_grad=True), torch.zeros(1, 1), [1.], torch.ones(1, 1, dtype=torch.bool))


@pytest.mark.parametrize("adv,ratio,expected", [(1., 1.3, 0.), (-1., .7, 0.), (1., .7, -1.), (-1., 1.3, 1.)])
def test_policy_clip_derivative_sign(adv, ratio, expected):
    theta = torch.tensor([[ratio]], requires_grad=True).log()
    theta.retain_grad()
    result = grpo_objective(theta, torch.zeros_like(theta), [adv], torch.ones_like(theta, dtype=torch.bool), kl_beta=0.)
    result.loss.backward()
    assert torch.sign(theta.grad).item() == expected


def test_variable_branch_lengths_and_groups_have_equal_group_weight():
    theta = torch.tensor([[.05, .1, 0.], [-.1, 0., 0.], [.2, -.2, .1]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False], [True, True, True]])
    adv = [1., -1., 1.]
    result = grpo_objective(theta, torch.zeros_like(theta), adv, mask, group_slices=[slice(0, 2), slice(2, 3)])
    parts = [grpo_objective(theta[i:i+1], torch.zeros_like(theta[i:i+1]), [adv[i]], mask[i:i+1]).loss for i in range(3)]
    torch.testing.assert_close(result.loss, ((parts[0]+parts[1])/2+parts[2])/2)
