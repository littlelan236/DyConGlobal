"""Per-token hard K3 cap, original policy/weights, and frozen resume identity."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from RL.config import Grpo, load, schedule_contract
from RL.grpo.metrics import summarize_update
from RL.grpo.objective import grpo_objective


def evaluate(theta, *, cap=1., advantages=None, mask=None, groups=None, guard=False, beta=.2):
    if advantages is None:
        advantages = [1.] * len(theta)
    if mask is None:
        mask = torch.ones_like(theta, dtype=torch.bool)
    return grpo_objective(theta, torch.zeros_like(theta), advantages, mask,
                          k3_cap=cap, kl_beta=beta, group_slices=groups,
                          ratio_guard_enabled=guard)


@pytest.mark.parametrize('cap', [0., -1., float('nan'), float('inf'), -float('inf'), True, 1e40])
def test_invalid_caps_rejected_in_schema_and_objective(cap):
    with pytest.raises(ValueError):
        Grpo(k3_cap=cap)
    with pytest.raises(ValueError, match='k3_cap'):
        evaluate(torch.zeros(1, 1), cap=cap)


def test_disabled_cap_matches_legacy_loss_gradients_and_metrics():
    theta = torch.tensor([[-8., -.3, .2]], requires_grad=True)
    default = grpo_objective(theta, torch.zeros_like(theta), [1.], torch.ones_like(theta, dtype=torch.bool))
    explicit_theta = theta.detach().clone().requires_grad_(True)
    explicit = grpo_objective(explicit_theta, torch.zeros_like(theta), [1.],
                              torch.ones_like(theta, dtype=torch.bool), k3_cap=None)
    default.loss.backward()
    explicit.loss.backward()
    torch.testing.assert_close(default.loss, explicit.loss, rtol=0, atol=0)
    torch.testing.assert_close(theta.grad, explicit_theta.grad, rtol=0, atol=0)
    assert set(default.metrics) == set(explicit.metrics)
    assert 'k3_cap' not in explicit.metrics


@pytest.mark.parametrize('advantage', [-1., 0., 1.])
def test_cap_preserves_policy_and_only_removes_saturated_k3_gradient(advantage):
    theta = torch.tensor([[-4., -.3, .2, 3.]], requires_grad=True)
    capped = evaluate(theta, advantages=[advantage])
    policy = evaluate(theta, advantages=[advantage], beta=0.)
    policy_grad = torch.autograd.grad(policy.loss, theta, retain_graph=True)[0]
    capped.loss.backward()
    # Both low and high ratios can saturate K3. The policy gradient is retained.
    torch.testing.assert_close(theta.grad[0, [0, 3]], policy_grad[0, [0, 3]])
    expected_k3_grad = .2 * (1 - torch.exp(-theta.detach()[0, 1:3])) / 4
    torch.testing.assert_close(theta.grad[0, 1:3] - policy_grad[0, 1:3], expected_k3_grad)
    raw = torch.expm1(-theta.detach()) + theta.detach()
    assert capped.k3_penalty.item() == pytest.approx(raw.clamp_max(1.).mean().item())
    uncapped = evaluate(theta, advantages=[advantage], cap=None)
    torch.testing.assert_close(capped.policy_loss, uncapped.policy_loss, rtol=0, atol=0)
    assert capped.metrics['k3_penalty_max'] == pytest.approx(raw.max().item())
    assert capped.metrics['k3_cap_clipped_tokens'] == 2


def test_equality_retains_k3_gradient_and_strict_exceedance_count():
    theta = torch.tensor([[-1.]], requires_grad=True)
    cap = (torch.expm1(-theta.detach()) + theta.detach()).item()
    result = evaluate(theta, cap=cap, advantages=[0.])
    result.loss.backward()
    assert result.metrics['k3_cap_clipped_tokens'] == 0
    assert theta.grad.item() == pytest.approx(.2 * (1 - math.e), rel=1e-6)


@pytest.mark.parametrize('beta', [0., .2])
def test_extreme_finite_log_ratio_caps_before_autograd_overflow(beta):
    theta = torch.tensor([[-1000., -.3]], requires_grad=True)
    result = evaluate(theta, beta=beta)
    result.loss.backward()
    assert result.k3_penalty.item() == pytest.approx((1 + math.expm1(.3) - .3) / 2)
    assert torch.isfinite(theta.grad).all()
    assert theta.grad[0, 0] == 0
    assert result.metrics['ratio_guard_excluded_tokens'] == 0
    assert result.metrics['k3_cap_clipped_tokens'] == 1
    assert result.metrics['k3_raw_nonfinite_tokens'] == 1
    assert result.metrics['k3_raw_penalty_mean'] is None
    assert result.metrics['k3_penalty_max'] is None
    assert result.metrics['k3_capped_penalty_max'] == 1
    merged = summarize_update([[result.metrics]])
    assert merged['k3_raw_penalty_mean'] is None
    assert merged['k3_penalty_max'] is None
    # Public scalars serialize with strict JSON; private tensors are not serialized.
    json.dumps({k: v for k, v in merged.items() if not k.startswith('_')}, allow_nan=False)
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate(theta.detach(), cap=None, beta=beta)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_cap_does_not_hide_invalid_logprobs(bad):
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate(torch.tensor([[bad]]))


def test_unequal_lengths_groups_masks_guard_and_microbatch_gradients():
    values = [[-4., -.3, -1000.], [-.5, .2, .1], [-8., -.2, -1000.]]
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    theta = torch.tensor(values, requires_grad=True)
    result = evaluate(theta, mask=mask, groups=[slice(0, 2), slice(2, 3)], guard=True)
    result.loss.backward()
    raw = [[math.expm1(4)-4, math.expm1(.3)-.3],
           [math.expm1(.5)-.5, math.expm1(-.2)+.2, math.expm1(-.1)+.1],
           [0., math.expm1(.2)-.2]]
    branch_means = [sum(min(k, 1) for k in branch)/len(branch) for branch in raw]
    expected = ((branch_means[0]+branch_means[1])/2+branch_means[2])/2
    assert result.k3_penalty.item() == pytest.approx(expected)
    assert result.metrics['k3_cap_clipped_tokens'] == 1
    assert result.metrics['k3_cap_eligible_tokens'] == 6
    leaves = []
    rows = []
    for i, vals in enumerate(values):
        leaf = torch.tensor([vals], requires_grad=True)
        part = evaluate(leaf, mask=mask[i:i+1], guard=True)
        (part.loss * (.25 if i < 2 else .5)).backward()
        leaves.append(leaf)
        rows.append(part.metrics)
    torch.testing.assert_close(theta.grad, torch.cat([leaf.grad for leaf in leaves]))
    merged = summarize_update([rows[:2], rows[2:]])
    assert merged['k3_penalty_mean'] == pytest.approx(expected)
    assert merged['k3_raw_penalty_mean'] == pytest.approx(result.metrics['k3_raw_penalty_mean'])
    assert merged['k3_cap_clipped_tokens'] == 1
    assert merged['k3_cap_eligible_tokens'] == 6
    assert merged['k3_cap_clipped_fraction'] == pytest.approx(1/6)


def test_all_guarded_keeps_differentiable_zero_and_no_cap_exceedances():
    theta = torch.tensor([[-1000., -20.]], requires_grad=True)
    result = evaluate(theta, guard=True)
    result.loss.backward()
    assert result.loss.item() == 0
    assert torch.equal(theta.grad, torch.zeros_like(theta))
    assert result.metrics['k3_cap_clipped_tokens'] == 0
    assert result.metrics['k3_cap_eligible_tokens'] == 0
    assert result.metrics['k3_cap_clipped_fraction'] == 0


def test_cap_is_forwarded_by_existing_group_backend(monkeypatch):
    from RL.grpo.backend import GroupObjective
    from RL.grpo.groups import GroupBatch
    import RL.grpo.backend as backend
    theta = torch.tensor([-4., -.3], requires_grad=True)
    monkeypatch.setattr(backend, 'model_target_logprobs', lambda *args: theta)
    group = GroupBatch('g', ['b'], [[1, 2]], [[3, 4]], torch.tensor([[1, 2, 3, 4]]),
                       torch.ones(1, 4), torch.tensor([[-100, -100, 3, 4]]),
                       torch.tensor([[False, False, True, True]]), [1.], [1.])
    model = torch.nn.Linear(1, 1)
    objective = GroupObjective({'g': {'b': torch.zeros(2)}}, Grpo(k3_cap=1.), 128)
    result = next(objective.iter_results(model, group))
    assert result.metrics['k3_cap'] == 1.
    assert result.k3_penalty.item() == pytest.approx((1+math.expm1(.3)-.3)/2)


def test_cap_contract_preserves_legacy_and_rejects_changed_resume(tmp_path):
    from RL.grpo.artifacts import write_objective_contract, verify_objective_contract
    project = Path(__file__).resolve().parents[1]
    cfg, _ = load(project/'RL/experiments/grpo-offline-k3-smoke.yaml', project)
    legacy = schedule_contract(cfg)
    assert 'k3_cap' not in legacy['grpo']
    cfg.grpo.k3_cap = 1.
    capped = schedule_contract(cfg)
    assert capped['grpo']['k3_cap'] == 1.
    assert {k: v for k, v in capped['grpo'].items() if k != 'k3_cap'} == legacy['grpo']
    path = tmp_path/'objective.json'
    write_objective_contract(path, capped)
    verify_objective_contract(path, capped)
    with pytest.raises(ValueError):
        verify_objective_contract(path, legacy)
    cfg.grpo.k3_cap = 5.
    with pytest.raises(ValueError):
        verify_objective_contract(path, schedule_contract(cfg))


def test_configuration_remains_torch_free():
    code = "from RL.config import Grpo; import sys; assert Grpo(k3_cap=1.).k3_cap == 1.; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True, timeout=20)


def test_tiny_qwen_capped_training_resume_matches_uninterrupted_and_rejects_changes(tmp_path):
    from tests.test_grpo_prepared import config, dataset, train
    from tests.test_grpo_lifecycle import assert_tree_close
    from RL.modeling import adapter_fingerprint
    dataset(tmp_path, 'a', range(5))
    cfg = config(tmp_path)
    cfg.grpo.k3_cap = 1e-6  # Exercise saturation with a tiny real LoRA model.
    cfg.training.checkpoint_every_updates = 1
    full, full_model, full_result, _ = train(cfg, 'cap-full')
    paused, _, _, _ = train(cfg, 'cap-paused', stop=1)
    checkpoint = paused/'checkpoints/step-000001'
    resumed, resumed_model, result, _ = train(cfg, 'cap-resumed', mode='resume', parent=checkpoint)
    assert result['state'] == full_result['state']
    assert adapter_fingerprint(resumed_model) == adapter_fingerprint(full_model)
    rows = [json.loads(line) for line in (full/'metrics.jsonl').read_text().splitlines()]
    assert any(row['k3_cap_clipped_tokens'] > 0 for row in rows)
    assert all(row['k3_penalty_mean'] <= cfg.grpo.k3_cap * (1+1e-6) for row in rows)
    a = torch.load(full/'checkpoints/step-000003/training-state.pt', weights_only=True)
    b = torch.load(resumed/'checkpoints/step-000003/training-state.pt', weights_only=True)
    for key in ('optimizer', 'scheduler', 'rng'):
        assert_tree_close(a[key], b[key])
    for changed in (None, 1.):
        altered = cfg.model_copy(deep=True)
        altered.grpo.k3_cap = changed
        with pytest.raises(ValueError):
            train(altered, 'bad-cap-' + str(changed), mode='resume', parent=checkpoint)
