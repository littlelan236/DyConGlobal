import pytest
torch = pytest.importorskip("torch")

from RL.grpo.objective import grpo_objective, selected_logprobs


def test_reference_equal_policy_has_zero_k3_and_gradient():
    current = torch.zeros((2, 3), requires_grad=True)
    result = grpo_objective(current, torch.zeros_like(current), [1.0, -1.0],
                            torch.ones_like(current, dtype=torch.bool))
    assert result.k3_penalty.item() == pytest.approx(0.0)
    result.loss.backward()
    assert torch.isfinite(current.grad).all()


def test_ratio_clipping_preserves_negative_advantage_direction():
    current = torch.tensor([[1.0], [-1.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    result = grpo_objective(current, reference, [1.0, -1.0],
                            torch.ones_like(current, dtype=torch.bool), clip_epsilon=.1, kl_beta=0.0)
    result.loss.backward()
    assert current.grad[0] <= 0
    assert current.grad[1] >= 0
    assert result.metrics["clip_fraction"] == pytest.approx(1.0)


def test_selected_logprobs_keep_first_completion_prediction_coordinate():
    logits = torch.full((1, 4, 5), -10.0)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    mask = torch.tensor([[False, False, True, True]])
    logits[0, 1, 3] = 10.0
    logits[0, 2, 4] = 10.0
    values, selected = selected_logprobs(logits, input_ids, labels, mask)
    assert selected.tolist() == [[False, True, True]]
    assert values.shape == (1, 3)
    assert values[0, 1] > -1e-3 and values[0, 2] > -1e-3


def test_k3_is_nonnegative_and_has_expected_derivative():
    theta = torch.tensor([0.2, -0.2, 0.0], requires_grad=True)
    delta = -theta
    value = torch.expm1(delta) - delta
    assert torch.all(value >= 0)
    value.sum().backward()
    torch.testing.assert_close(theta.grad, 1 - torch.exp(delta))
