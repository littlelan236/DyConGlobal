"""K3 cap comparisons preserve objective weighting, masking and gradients."""
import math

import pytest

torch = pytest.importorskip('torch')

from RL.grpo.k3_diagnostics import K3_CAPS, K3_SCALARS
from RL.grpo.metrics import summarize_update
from RL.grpo.objective import grpo_objective


def evaluate(values, mask=None, groups=None, guard=True):
    theta = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    if mask is None:
        mask = torch.ones_like(theta, dtype=torch.bool)
    else:
        mask = torch.tensor(mask, dtype=torch.bool)
    result = grpo_objective(theta, torch.zeros_like(theta), [1.] * len(values), mask,
                            group_slices=groups, ratio_guard_enabled=guard)
    return theta, result


def test_caps_match_independent_oracle_with_unequal_groups_and_lengths():
    values = [[-6., -8., -1000.], [-1., 0., 2.], [-3., -1000., -1000.]]
    mask = [[True, True, False], [True, True, True], [True, False, False]]
    _, result = evaluate(values, mask, [slice(0, 2), slice(2, 3)])
    # The second token is guard-excluded. Its denominator remains in branch 0.
    raw = [[math.expm1(6.) - 6., 0.],
           [math.expm1(1.) - 1., 0., math.expm1(-2.) + 2.],
           [math.expm1(3.) - 3.]]
    assert result.metrics['k3_diagnostic_tokens'] == 5
    assert result.metrics['k3_penalty_max'] == pytest.approx(raw[0][0])
    for cap in K3_CAPS:
        means = [sum(min(k, cap) for k in branch) / len(branch) for branch in raw]
        expected = ((means[0] + means[1]) / 2 + means[2]) / 2
        exceeded = sum(k > cap for branch in raw for k in branch)
        assert result.metrics[f'k3_cap_{cap}_mean'] == pytest.approx(expected)
        assert result.metrics[f'k3_cap_{cap}_exceeded_tokens'] == exceeded
        assert result.metrics[f'k3_cap_{cap}_exceeded_fraction'] == pytest.approx(exceeded / 5)


def test_branch_microbatches_preserve_group_weight_and_global_token_counts():
    branches = [[-6., -8.], [-1., 0., 2.], [-3.]]
    rows = [evaluate([branch])[1].metrics for branch in branches]
    merged = summarize_update([rows[:2], rows[2:]])
    assert merged['k3_diagnostic_tokens'] == 5
    assert merged['k3_penalty_max'] == max(row['k3_penalty_max'] for row in rows)
    for cap in K3_CAPS:
        mean = f'k3_cap_{cap}_mean'
        expected = ((rows[0][mean] + rows[1][mean]) / 2 + rows[2][mean]) / 2
        assert merged[mean] == pytest.approx(expected)
        count = sum(row[f'k3_cap_{cap}_exceeded_tokens'] for row in rows)
        assert merged[f'k3_cap_{cap}_exceeded_tokens'] == count
        assert merged[f'k3_cap_{cap}_exceeded_fraction'] == pytest.approx(count / 5)


@pytest.mark.parametrize('guard', [True, False])
def test_diagnostics_do_not_change_loss_gradients_or_rng(monkeypatch, guard):
    import RL.grpo.objective as objective

    before = torch.random.get_rng_state().clone()
    theta, result = evaluate([[-6., -8., .3]], guard=guard)
    result.loss.backward()
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(type(result.metrics[key]) in (int, float) for key in K3_SCALARS)
    monkeypatch.setattr(objective, 'token_statistics', lambda *args: {})
    baseline_theta, baseline = evaluate([[-6., -8., .3]], guard=guard)
    baseline.loss.backward()
    torch.testing.assert_close(result.loss, baseline.loss, rtol=0, atol=0)
    torch.testing.assert_close(theta.grad, baseline_theta.grad, rtol=0, atol=0)
    # Large K3 still contributes to the actual loss and gradient; these are diagnostics.
    assert result.k3_penalty.item() > result.metrics['k3_cap_10_mean']
    assert theta.grad[0, 0].abs().item() > 0


def test_all_guard_excluded_has_no_diagnostic_tokens_and_finite_zero_means():
    theta, result = evaluate([[-1000., -20.]])
    assert result.metrics['k3_diagnostic_tokens'] == 0
    assert result.metrics['k3_penalty_max'] == 0
    assert all(result.metrics[key] == 0 for key in K3_SCALARS)
    result.loss.backward()
    assert torch.equal(theta.grad, torch.zeros_like(theta))


def test_guard_disabled_counts_extreme_tokens_instead_of_hiding_them():
    _, guarded = evaluate([[-8., 0.]])
    _, original = evaluate([[-8., 0.]], guard=False)
    assert guarded.metrics['k3_diagnostic_tokens'] == 1
    assert guarded.metrics['k3_cap_10_exceeded_tokens'] == 0
    assert original.metrics['k3_diagnostic_tokens'] == 2
    assert original.metrics['k3_cap_10_exceeded_tokens'] == 1
    assert original.metrics['k3_cap_10_mean'] == 5


@pytest.mark.parametrize('mixed', [True, False])
def test_legacy_or_partial_diagnostics_are_unavailable_instead_of_zero(mixed):
    _, result = evaluate([[-1., 0.]])
    old = {k: v for k, v in result.metrics.items() if k not in K3_SCALARS}
    rows = [old, result.metrics] if mixed else [old]
    merged = summarize_update([rows])
    assert not any(key in merged for key in K3_SCALARS)
