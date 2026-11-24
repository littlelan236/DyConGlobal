"""Opt-in numerical diagnostics: reports errors, does not assert backend equivalence."""
import json
import os
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('DYCON_FLA_GPU_TEST') != '1', reason='Explicit idle GPU audit')


def difference(actual, expected):
    delta = actual.double() - expected.double()
    return {'max_abs': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / expected.double().norm().clamp_min(1e-30))}


def test_full_fla_numerical_diagnostics(tmp_path):
    from RL.gdn_backend import install_gdn_backend, reset_for_tests
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.use_deterministic_algorithms(True)
    shape = (1, 65, 2, 128)
    q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(shape[:3], device='cuda', dtype=torch.float32)
    beta = torch.rand(shape[:3], device='cuda', dtype=torch.bfloat16)
    cotangent = torch.randn(shape, device='cuda', dtype=torch.float32)
    leaves = [x.detach().clone().requires_grad_() for x in (q, k, v, g, beta)]
    reset_for_tests()
    binding = install_gdn_backend(SimpleNamespace(gdn_backend='fla_chunk', dtype='bfloat16'))
    out, _ = binding.chunk(*leaves, output_final_state=False, use_qk_l2norm_in_kernel=True)
    grads = torch.autograd.grad((out.float() * cotangent).sum(), leaves)
    precise = [x.double().detach().requires_grad_() for x in (q, k, v, g, beta)]
    qd, kd, vd, gd, bd = precise
    qd = torch.nn.functional.normalize(qd, dim=-1) * shape[-1] ** -0.5
    kd = torch.nn.functional.normalize(kd, dim=-1)
    state = torch.zeros((1, 2, 128, 128), device='cuda', dtype=torch.float64)
    outputs = []
    for t in range(shape[1]):
        state = state * gd[:, t].exp()[..., None, None]
        residual = vd[:, t] - torch.einsum('bhk,bhkv->bhv', kd[:, t], state)
        state = state + torch.einsum('bhk,bhv->bhkv', kd[:, t], residual * bd[:, t, :, None])
        outputs.append(torch.einsum('bhk,bhkv->bhv', qd[:, t], state))
    expected = torch.stack(outputs, dim=1)
    expected_grads = torch.autograd.grad((expected * cotangent.double()).sum(), precise)
    assert all(torch.isfinite(x).all() for x in (out, expected, *grads, *expected_grads))
    report = {'acceptance': 'finite runtime and complete gradients; cross-backend error is report-only',
              'operator': {'shape': shape, 'baseline': 'independent FP64 recurrence; identical inputs/cotangent',
                           'output': difference(out, expected),
                           'gradients': {name: difference(a, b) for name, a, b in zip(('q','k','v','g','beta'), grads, expected_grads)}}}
    from tests.lora_helpers import raw_model
    from tests.test_grpo_prefix import group
    from RL.full_policy import FullPolicy, parameter_scope
    results = {}
    for backend in ('reference_pytorch', 'fla_chunk'):
        reset_for_tests()
        install_gdn_backend(SimpleNamespace(gdn_backend=backend, dtype='bfloat16'))
        model = raw_model().to('cuda', torch.bfloat16)
        inventory = parameter_scope(model)
        policy = FullPolicy(model).train()
        policy.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        batch = group(3, 3).branch_batch(1, 'cuda')
        values = policy(batch, 1)
        (values * torch.linspace(0.5, 1.5, len(values), device='cuda')).sum().backward()
        gradients = {n: p.grad.float().detach().cpu() for n, p in policy.named_parameters() if p.grad is not None}
        assert set(gradients) == {r['name'] for r in inventory['parameters']}
        assert all(torch.isfinite(x).all() for x in gradients.values())
        results[backend] = (values.detach().cpu(), gradients)
        del model, policy
    reference, fla = results['reference_pytorch'], results['fla_chunk']
    delta = torch.cat([(fla[1][n] - reference[1][n]).reshape(-1) for n in reference[1]])
    baseline = torch.cat([x.reshape(-1) for x in reference[1].values()])
    report['tiny_full'] = {'dtype': 'bfloat16', 'target_logprobs': difference(fla[0], reference[0]),
                           'all_gradient_relative_l2': float(delta.norm() / baseline.norm()),
                           'parameter_tensors': len(reference[1]),
                           'per_parameter_gradients': {n: difference(fla[1][n], reference[1][n]) for n in reference[1]}}
    (tmp_path / 'numerical-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'tiny_full'}))
    reset_for_tests()
