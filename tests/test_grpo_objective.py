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


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_ratio_guard_only_removes_extreme_token_without_rescaling(advantage):
    current = torch.tensor([[-1000., -0.3, 0.2]], requires_grad=True)
    ref = torch.zeros_like(current)
    result = grpo_objective(current, ref, [advantage], torch.ones_like(current, dtype=torch.bool), ratio_guard_enabled=True)
    other = current.detach()[:, 1:].clone().requires_grad_(True)
    baseline = grpo_objective(other, torch.zeros_like(other), [advantage], torch.ones_like(other, dtype=torch.bool), ratio_guard_enabled=True)
    torch.testing.assert_close(result.loss, baseline.loss * (2 / 3))
    result.loss.backward()
    baseline.loss.backward()
    assert current.grad[0, 0].item() == 0
    torch.testing.assert_close(current.grad[:, 1:], other.grad * (2 / 3))
    assert torch.isfinite(current.grad).all()
    assert result.metrics["ratio_guard_excluded_tokens"] == 1
    assert result.metrics["ratio_min"] == 0


def test_ratio_guard_all_excluded_has_finite_differentiable_zero():
    current = torch.tensor([[-1000., -20.]], requires_grad=True)
    result = grpo_objective(current, torch.zeros_like(current), [-1.], torch.ones_like(current, dtype=torch.bool), ratio_guard_enabled=True)
    assert result.loss.item() == 0
    result.loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_ratio_guard_strict_boundary_and_original_mask():
    # exp(log(1e-3)) rounds slightly below 1e-3 in float32; equality is
    # judged in the actual ratio dtype, not an approximate log threshold.
    x = torch.tensor([[-6., -8., -1000.]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    result = grpo_objective(x, torch.zeros_like(x), [1.], mask, ratio_guard_enabled=True)
    assert result.metrics["ratio_guard_excluded_tokens"] == 1
    assert result.metrics["ratio_guard_eligible_tokens"] == 2
    result.loss.backward()
    assert x.grad[0, 0] != 0
    assert x.grad[0, 1] == 0 and x.grad[0, 2] == 0


def test_ratio_guard_update_counts_are_token_counts():
    from RL.grpo.metrics import summarize_update
    x = torch.tensor([[-8., 0.]], requires_grad=True)
    r = grpo_objective(x, torch.zeros_like(x), [1.], torch.ones_like(x, dtype=torch.bool), ratio_guard_enabled=True)
    summary = summarize_update([[r.metrics, r.metrics]])
    assert summary["ratio_guard_excluded_tokens"] == 2
    assert summary["ratio_guard_eligible_tokens"] == 4
    assert summary["ratio_guard_excluded_fraction"] == .5


def test_ratio_guard_disabled_default_and_custom_threshold():
    x = torch.tensor([[-8., -3.]], requires_grad=True)
    mask = torch.ones_like(x, dtype=torch.bool)
    original = grpo_objective(x, torch.zeros_like(x), [1.], mask)
    assert original.metrics["ratio_guard_excluded_tokens"] == 0
    guarded = grpo_objective(x, torch.zeros_like(x), [1.], mask,
                             ratio_guard_enabled=True, ratio_guard_threshold=.1)
    assert guarded.metrics["ratio_guard_excluded_tokens"] == 2
    assert guarded.loss.item() == 0
    assert original.k3_penalty.item() > 100


@pytest.mark.parametrize("threshold", [0, -1, 1, float('nan'), float('inf')])
def test_ratio_guard_config_rejects_invalid_threshold(threshold):
    from RL.config import Grpo
    with pytest.raises(ValueError):
        Grpo.model_validate({"ratio_guard": {"enabled": True, "threshold": threshold}})


def test_ratio_guard_config_defaults_and_serialization():
    from RL.config import Grpo
    assert not Grpo().ratio_guard.enabled
    configured = Grpo.model_validate({"ratio_guard": {"enabled": True, "threshold": .005}})
    assert configured.model_dump()["ratio_guard"] == {"enabled": True, "threshold": .005}


def test_ratio_guard_equality_is_retained():
    x = torch.tensor([[-3., -4.]], requires_grad=True)
    threshold = float(torch.exp(x.detach())[0, 0])
    result = grpo_objective(x, torch.zeros_like(x), [1.], torch.ones_like(x, dtype=torch.bool),
                            ratio_guard_enabled=True, ratio_guard_threshold=threshold)
    assert result.metrics["ratio_guard_excluded_tokens"] == 1
    result.loss.backward()
    assert x.grad[0, 0] != 0 and x.grad[0, 1] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_ratio_guard_does_not_hide_nonfinite_inputs(value):
    x = torch.tensor([[value]], requires_grad=True)
    with pytest.raises(ValueError, match="Nonfinite"):
        grpo_objective(x, torch.zeros_like(x), [1.], torch.ones_like(x, dtype=torch.bool),
                       ratio_guard_enabled=True)
