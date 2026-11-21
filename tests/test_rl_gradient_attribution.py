import pytest

torch = pytest.importorskip('torch')
from RL.gradient_attribution import GradientAttribution
from RL.online_runtime import online_torch_objective
from RL.online_bootstrap import SourceMass


def test_weighted_gradients_and_no_training_change():
    p = torch.nn.Parameter(torch.tensor([.2, -.4], dtype=torch.float64))
    tracker = GradientAttribution([p])
    expected = {}
    for source, vector, kl_vector in [('dycon', [2., -1.], [.5, .2]),
                                      ('baseline', [-1., 3.], [.1, -.3])]:
        v, k = torch.tensor(vector), torch.tensor(kl_vector)
        kl = (p * k).sum()
        loss = (p * v).sum() + kl
        tracker.backward(loss, source, kl)
        expected[source] = (v + k).double()
    expected['kl'] = torch.tensor([.6, -.1], dtype=torch.float64)
    total = expected['dycon'] + expected['baseline']
    torch.testing.assert_close(p.grad, total)
    result = tracker.summary()
    for name, vector in expected.items():
        component = result['components'][name]
        assert component['norm'] == pytest.approx(vector.norm().item(), rel=1e-6)
        assert component['projection_share'] == pytest.approx((vector @ total / (total @ total)).item(), rel=1e-6)
    assert sum(result['components'][s]['projection_share'] for s in ['dycon','baseline']) == pytest.approx(1.)
    assert result['source_sum_residual_norm'] < 1e-6
    assert not p._backward_hooks


def test_zero_and_opposed_gradients():
    p = torch.nn.Parameter(torch.zeros(2))
    tracker = GradientAttribution([p])
    tracker.backward(p.sum(), 'dycon')
    tracker.backward(-p.sum(), 'baseline')
    result = tracker.summary()
    assert result['total_norm'] == 0
    assert result['components']['dycon']['projection_share'] is None
    assert result['components']['kl']['norm'] == 0
    assert result['source_cosine'] == pytest.approx(-1.)


def test_objective_kl_tensor_matches_weighted_reference():
    theta = torch.tensor([[-2., -3.], [-1., -2.]], requires_grad=True)
    reference = theta.detach() + .3
    terms = {}
    loss, _ = online_torch_objective(theta, theta.detach(), reference,
        torch.tensor([1., -1.]), torch.ones_like(theta, dtype=torch.bool),
        ['dycon','baseline'], {'dycon':[[0]],'baseline':[[1]]},
        source_mass=SourceMass(.3,.7), kl_beta=.2, k3_cap=1., differentiable_terms=terms)
    expected = .2 * (torch.expm1(reference-theta) - (reference-theta)).mean(dim=1)
    expected = expected @ torch.tensor([.3,.7])
    torch.testing.assert_close(terms['weighted_kl'], expected)
    actual = torch.autograd.grad(terms['weighted_kl'], theta, retain_graph=True)[0]
    torch.testing.assert_close(actual, torch.autograd.grad(expected, theta)[0])
    loss.backward()

@pytest.mark.parametrize('kl_beta', [0., .02])
def test_updater_attribution_preserves_optimizer(monkeypatch, kl_beta):
    import copy
    from types import SimpleNamespace
    from dataclasses import replace
    from tests.test_rl_prefix_trace import rows
    from RL.online_runtime import TorchOnlineUpdater, OnlineRolloutBatch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids).cumsum(dim=1)))
    def optimizer(model, settings):
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    monkeypatch.setattr('RL.training.optimizer_and_scheduler', optimizer)
    torch.manual_seed(42)
    a = Model(); b = copy.deepcopy(a)
    source = rows(bool(kl_beta))
    source += (replace(source[0], row_id='dycon', source='dycon', advantage=.7),)
    batch = OnlineRolloutBatch(0, ('task',), 'S0', 'E0', source)
    settings = SimpleNamespace(grad_clip=.01)
    result = TorchOnlineUpdater(a, settings, source_mass=SourceMass(.3,.7),
                kl_beta=kl_beta, baseline_training_mode='prefix_trace')(batch, 0)
    class NoTracking:
        def __init__(self, parameters): pass
        def backward(self, loss, source, weighted_kl=None): loss.backward()
        def summary(self): return {}
    monkeypatch.setattr('RL.gradient_attribution.GradientAttribution', NoTracking)
    TorchOnlineUpdater(b, settings, source_mass=SourceMass(.3,.7),
                kl_beta=kl_beta, baseline_training_mode='prefix_trace')(batch, 0)
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    metrics = result['gradient_attribution']
    assert metrics['total_norm'] == pytest.approx(result['gradient_norm'], rel=1e-6)
    assert metrics['source_sum_residual_norm'] < 1e-6
    assert metrics['components']['kl']['norm'] > 0 if kl_beta else metrics['components']['kl']['norm'] == 0


def test_tensorboard_exports_gradient_components():
    from RL.observability_tensorboard import scalar_values
    values = scalar_values({'gradient_attribution': {
        'total_norm': 2., 'source_cosine': -.3, 'kl_backward_seconds': .4,
        'components': {'kl': {'norm': .1, 'projection_share': -.2, 'cosine_with_total': None}}}})
    assert values['gradient/kl/projection_share'] == -.2
    assert values['gradient/total_norm'] == 2.
    assert values['timing/gradient_kl_backward_seconds'] == .4
    assert 'gradient/kl/cosine_with_total' not in values
