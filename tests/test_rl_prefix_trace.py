from dataclasses import replace
from types import SimpleNamespace
import copy
import pytest

from RL.online_runtime import OnlineRow, online_row_masses, collate_rows, OnlineRolloutBatch, TorchOnlineUpdater
from RL.online_bootstrap import SourceMass
from RL.online_trace import build_training_traces


def rows(kl=False):
    def make(name, ids, output, step):
        return OnlineRow(name, "task", "baseline", "g", tuple(ids), tuple(output),
                         (-2.,) * len(output), (-2.1,) * len(output) if kl else None,
                         -1., "trajectory", 3, step, False)
    return (make("a", [1, 2, 3], [3], 0),
            make("b", [1, 2, 3, 4, 5, 6], [5, 6], 1))


def test_exact_coverage_and_stable_receipt():
    source = rows(True)
    result, weights, receipt = build_training_traces(source, [.1, .2])
    assert len(result) == 1 and weights == pytest.approx([.3])
    assert result[0].supervision_positions == (2, 4, 5)
    assert result[0].completion_token_ids == (3, 5, 6)
    assert source[0].supervision_positions is None
    assert build_training_traces(source, [.1, .2])[2] == receipt
    torch = pytest.importorskip("torch")
    t = collate_rows(result, device=torch.device("cpu"))
    assert t['loss_mask'].tolist() == [[False, False, True, False, True, True]]
    assert t['kl_reference_logprobs'][0, [1, 3, 4]].tolist() == pytest.approx([-2.1]*3)


@pytest.mark.parametrize("budget,break_prefix", [(4, False), (32768, True)])
def test_split_preserves_full_context(budget, break_prefix):
    source = list(rows())
    if break_prefix:
        source[1] = replace(source[1], input_token_ids=(9, 2, 3, 4, 5, 6))
    result, weights, receipt = build_training_traces(source, [.1, .2], max_input_tokens=budget)
    assert len(result) == 2
    assert result[1].input_token_ids == source[1].input_token_ids
    assert sum(weights) == pytest.approx(.3)


def test_invalid_identity_and_mass_rejected():
    a, b = rows()
    for bad in [replace(b, agent_step=0), replace(b, advantage=1.),
                replace(b, rollout_logprobs=()), replace(b, group_id="other")]:
        with pytest.raises(ValueError):
            build_training_traces((a, bad), [.1, .2])
    with pytest.raises(ValueError, match="masses"):
        build_training_traces((a, b), [.1, .3])


def test_manifest_binds_tokens_and_logprobs():
    a, b = rows()
    original = build_training_traces((a, b), [.1, .2])[2]['manifest_sha256']
    altered = replace(b, rollout_logprobs=(-3., -3.))
    assert build_training_traces((a, altered), [.1, .2])[2]['manifest_sha256'] != original


def test_config_mode_and_budget_validation():
    from RL.config import OnlineBootstrap
    values = dict(tasks=['task'], initial_estimator='manifest')
    assert OnlineBootstrap(**values).baseline_training_mode == 'agent_step'
    for extra in [dict(baseline_training_mode='unknown'), dict(max_trace_input_tokens=0)]:
        with pytest.raises(ValueError):
            OnlineBootstrap(**values, **extra)


def test_production_defaults_preserve_explicit_and_legacy(tmp_path):
    from RL.config import online_training_defaults
    raw = dict(algorithm={'name': 'online_bootstrap_grpo_v1'},
               run={'start': {'mode': 'fresh'}, 'attempt_id': 'new'},
               runtime={'output_root': str(tmp_path)},
               online_bootstrap={'state_path': 'state.json'})
    actual = online_training_defaults(raw, tmp_path)
    assert actual['online_bootstrap']['baseline_training_mode'] == 'prefix_trace'
    assert actual['runtime']['bf16_allow_split_k'] is False
    assert 'baseline_training_mode' not in raw['online_bootstrap']
    explicit = copy.deepcopy(raw)
    explicit['online_bootstrap']['baseline_training_mode'] = 'agent_step'
    assert 'bf16_allow_split_k' not in online_training_defaults(explicit, tmp_path)['runtime']
    explicit['online_bootstrap']['baseline_training_mode'] = 'prefix_trace'
    explicit['runtime']['bf16_allow_split_k'] = True
    assert online_training_defaults(explicit, tmp_path)['runtime']['bf16_allow_split_k'] is True
    legacy = copy.deepcopy(raw)
    legacy['run']['start']['mode'] = 'resume'
    assert online_training_defaults(legacy, tmp_path) == legacy
    (tmp_path / 'state.json.campaign.json').write_text('{"contract":{"options":{}}}')
    assert online_training_defaults(raw, tmp_path) == raw


@pytest.mark.parametrize('kl_beta', [0., .02])
def test_loss_gradient_optimizer_equivalence(monkeypatch, kl_beta):
    torch = pytest.importorskip("torch")
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
    # Include a second trajectory and a DyCon group to test source reduction.
    source += (replace(source[0], row_id='other', trajectory_id='other', trajectory_token_count=1),
               replace(source[0], row_id='dycon', source='dycon', advantage=.7))
    batch = OnlineRolloutBatch(0, ('task',), 'S0', 'E0', source)
    settings = SimpleNamespace(grad_clip=1e6)
    old = TorchOnlineUpdater(a, settings, source_mass=SourceMass(), kl_beta=kl_beta)(batch, 0)
    new = TorchOnlineUpdater(b, settings, source_mass=SourceMass(), kl_beta=kl_beta,
                            baseline_training_mode='prefix_trace')(batch, 0)
    assert old['loss'] == pytest.approx(new['loss'], abs=1e-6, rel=1e-5)
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
    assert b.embedding.weight.grad[1].abs().sum() > 0
    assert old['token_count'] == new['token_count']


def test_bound_campaign_inherits_numerics_and_mode(tmp_path):
    import json
    from RL.config import online_training_defaults
    raw = dict(algorithm={'name': 'online_bootstrap_grpo_v1'},
               run={'start': {'mode': 'resume'}}, runtime={},
               online_bootstrap={'state_path': 'state.json'})
    (tmp_path/'state.json.campaign.json').write_text(json.dumps({'contract':{
        'options':{'baseline_training_mode':'prefix_trace','max_trace_input_tokens':32768},
        'numerics':{'bf16_allow_reduced_precision_reduction':False,'bf16_allow_split_k':False}}}))
    result=online_training_defaults(raw,tmp_path)
    assert result['online_bootstrap']['baseline_training_mode']=='prefix_trace'
    assert result['runtime']['bf16_allow_split_k'] is False
    raw['runtime']['bf16_allow_split_k']=True
    with pytest.raises(ValueError, match='numerics differ'):
        online_training_defaults(raw,tmp_path)


def test_completion_budget_merges_steps_beyond_old_context_limit():
    template = rows()[0]
    base = (1,) * 32768
    source = tuple(replace(template, row_id=str(i), agent_step=i,
                           input_token_ids=base + (2, 3) * (i + 1),
                           completion_token_ids=(3,), trajectory_token_count=5)
                   for i in range(5))
    units, weights, receipt = build_training_traces(
        source, [.2] * 5, max_input_tokens=None, max_completion_tokens=2)
    assert [unit.input_token_ids for unit in units] == [
        source[1].input_token_ids, source[3].input_token_ids, source[4].input_token_ids]
    assert [unit.supervision_positions for unit in units] == [
        (32769, 32771), (32773, 32775), (32777,)]
    assert weights == pytest.approx([.4, .4, .2])
    assert receipt["break_reasons"] == {"completion_budget": 2}
