"""Monitoring semantics and observational equivalence, without extra forwards."""
import copy
import json

import pytest
import torch

from RL.grpo.objective import grpo_objective
from RL.grpo.metrics import summarize_update
from RL.grpo.signals import summarize_signals, throughput
from RL.observability import MetricReader
from RL.observability_tensorboard import scalar_values
from RL.optimizer_signals import gradient_signals, parameter_sample, update_signals


def row(ratios, advantage, *, guard=False):
    theta=torch.tensor([ratios],dtype=torch.float64).log().float().requires_grad_()
    result=grpo_objective(theta,torch.zeros_like(theta),[advantage],torch.ones_like(theta,dtype=torch.bool),
        kl_beta=0,clip_epsilon=.1,ratio_guard_enabled=guard)
    return result,theta


def test_saturation_guard_and_active_follow_actual_surrogate_gradient():
    positive,p=row([1.3,.8,.0001,1.],1,guard=True)
    negative,n=row([1.3,.8,.0001,1.],-1,guard=True)
    positive.loss.backward();negative.loss.backward()
    assert p.grad[0,0]==p.grad[0,2]==0
    assert n.grad[0,1]==n.grad[0,2]==0
    assert p.grad[0,1]<0 and n.grad[0,0]>0
    summary=summarize_update([[positive.metrics,negative.metrics]])
    assert summary['policy_saturated_objective_fraction']==.25
    assert summary['ratio_guard_excluded_objective_fraction']==.25
    assert summary['policy_active_objective_fraction']==.5
    assert summary['clip_fraction']==.75
    assert summary['policy_active_tokens']==4
    assert summary['policy_saturated_positive_token_fraction']==.25
    assert summary['policy_saturated_negative_token_fraction']==.25
    assert summary['ratio_guard_excluded_positive_token_fraction']==.25
    assert summary['ratio_guard_excluded_negative_token_fraction']==.25
    assert summary['policy_active_advantage_weighted_fraction']==.5


def test_reduction_group_branch_means_and_token_tail_support():
    a,_=row([1.3]*10,2)
    b,_=row([1.],-1)
    c,_=row([1.],1)
    d,_=row([1.],-3)
    summary=summarize_update([[a.metrics,b.metrics],[c.metrics,d.metrics]])
    assert summary['policy_saturated_objective_fraction']==.25
    assert summary['policy_saturated_advantage_weighted_fraction']==pytest.approx(2/7)
    assert summary['policy_saturated_positive_token_fraction']==pytest.approx(10/11)
    assert summary['ratio_high_token_fraction']==pytest.approx(10/13)
    assert summary['preference_gap']==pytest.approx(torch.log(torch.tensor(1.3)).item()/2)
    assert summary['advantage_logprob_alignment']==pytest.approx(torch.log(torch.tensor(1.3)).item()/2)
    assert summary['preference_positive_group_fraction']==.5
    assert summary['positive_advantage_tokens']==11 and summary['negative_advantage_tokens']==2
    assert summary['log_ratio_p99']==pytest.approx(torch.log(torch.tensor(1.3)).item())


def test_missing_sign_zero_advantage_and_entirely_guarded_groups():
    positive,_=row([1.],1)
    zero,_=row([1.],0)
    summary=summarize_update([[positive.metrics,zero.metrics]])
    assert summary['preference_gap'] is None and summary['preference_group_count']==0
    assert summary['policy_active_negative_token_fraction'] is None
    assert summary['policy_active_objective_fraction']==.5
    a,_=row([.0001],1,guard=True);b,_=row([.0001],-1,guard=True)
    summary=summarize_update([[a.metrics,b.metrics]])
    assert summary['fully_guarded_groups']==1 and summary['fully_guarded_branches']==2
    assert summary['policy_active_tokens']==0
    assert summary['k3_penalty_mean']==0
    assert summary['log_ratio_min']< -9
    assert summarize_signals([[{}]])=={}


def test_masked_padding_never_changes_signal_denominators():
    theta=torch.tensor([[.2,100.,100.],[-.2,-.1,100.]],requires_grad=True)
    mask=torch.tensor([[True,False,False],[True,True,False]])
    result=grpo_objective(theta,torch.zeros_like(theta),[1,-1],mask,kl_beta=0)
    summary=summarize_update([[result.metrics]])
    assert summary['positive_advantage_tokens']==1 and summary['negative_advantage_tokens']==2
    assert summary['log_ratio_max']==pytest.approx(.2)
    assert summary['preference_gap']==pytest.approx(.35)


def test_objective_diagnostics_do_not_change_loss_or_gradient(monkeypatch):
    import RL.grpo.objective as objective
    from RL.training import capture_rng
    from tests.test_grpo_lifecycle import assert_tree_close
    args=(torch.tensor([[.2,-.2,-10.],[-.2,.3,.01]],requires_grad=True),torch.zeros(2,3),[1.,-1.],torch.ones(2,3,dtype=torch.bool))
    before=capture_rng()
    measured=grpo_objective(*args,ratio_guard_enabled=True)
    gradient=torch.autograd.grad(measured.loss,args[0])[0]
    assert_tree_close(before,capture_rng())
    monkeypatch.setattr(objective,'token_signals',lambda *args:{})
    plain=grpo_objective(*args,ratio_guard_enabled=True)
    other=torch.autograd.grad(plain.loss,args[0])[0]
    torch.testing.assert_close(measured.loss,plain.loss,atol=0,rtol=0)
    torch.testing.assert_close(gradient,other,atol=0,rtol=0)
    assert all(not isinstance(v,torch.Tensor) or not v.requires_grad for v in measured.metrics.values())


def test_actual_optimizer_update_sampling_and_rng_are_observational(monkeypatch):
    import RL.training as training
    from tests.test_rl_observability_training import exercise
    from tests.test_grpo_lifecycle import assert_tree_close
    state=training.capture_rng()
    observed=exercise()
    monkeypatch.setattr(training,'gradient_signals',lambda *a:{})
    monkeypatch.setattr(training,'parameter_sample',lambda *a:None)
    monkeypatch.setattr(training,'update_signals',lambda *a:{})
    training.restore_rng(state)
    plain=exercise()
    assert_tree_close(observed,plain)
    p=torch.nn.Parameter(torch.zeros(1000));p.grad=torch.ones_like(p)
    norm=torch.nn.utils.clip_grad_norm_([p],1.)
    stats=gradient_signals([p],norm,1.)
    assert stats['gradient_parameter_coverage']==1
    assert stats['gradient_clip_scale']==pytest.approx(1/(1000**.5+1e-12))
    before=parameter_sample([p],1);after_rng=training.capture_rng()
    optimizer=torch.optim.SGD([p],lr=.1);optimizer.step()
    delta=update_signals([p],before)
    assert delta['parameter_update_sample_relative_norm'] is None
    assert delta['parameter_update_sample_elements']==64
    assert delta['parameter_update_sample_norm']>0
    assert parameter_sample([p],2) is None and parameter_sample([p],10) is not None
    assert_tree_close(after_rng,training.capture_rng())


def test_json_reader_and_tensorboard_tags_preserve_new_fields_at_beta_zero(tmp_path):
    a,_=row([1.3,.8,.0001],1,guard=True);b,_=row([1.,.9],-1,guard=True)
    values={'global_step':1,'seconds':2.,'group_count':1,'branch_count':2,'token_count':5,
            **summarize_update([[a.metrics,b.metrics]])}
    path=tmp_path/'metrics.jsonl';path.write_text(json.dumps(values,allow_nan=False)+'\n')
    projected=list(MetricReader(path).read())[0][0]
    tags=scalar_values(projected,include_k3=False)
    assert tags['learning/preference/gap']==values['preference_gap']
    assert tags['policy/active/objective_fraction']==values['policy_active_objective_fraction']
    assert tags['drift/k3/objective_mean_after_guard']>0
    assert 'train/k3_penalty_mean' not in tags
    assert tags['throughput/completion_tokens_per_second']==2.5
    assert 'probe' not in ' '.join(tags)
    assert not any(k.startswith('_') for k in projected)


def test_legacy_fields_backfill_only_available_diagnostics():
    tags=scalar_values({'global_step':10,'seconds':2.,'token_count':100,'branch_count':4,
         'group_count':1,'positive_logprob_change':.2,'negative_logprob_change':-.1,
         'ratio_guard_excluded_fraction':.02,'k3_penalty_mean':.3},include_k3=False)
    assert tags['learning/logprob/positive_advantage']==.2
    assert tags['drift/k3/objective_mean_after_guard']==.3
    assert tags['policy/guard/excluded_token_fraction']==.02
    assert tags['throughput/completion_tokens_per_second']==50
    assert not any(k.startswith('policy/active/') for k in tags)
    assert 'learning/preference/gap' not in tags


def test_full_training_metrics_are_json_and_keep_native_signals(tmp_path):
    from tests.test_grpo_prepared import dataset,config,train
    dataset(tmp_path,'a',range(3));cfg=config(tmp_path,updates=2)
    path,_,_,_=train(cfg,'signals')
    raw=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
    rows=[row for row,_ in MetricReader(path/'metrics.jsonl').read()]
    assert len(raw)==len(rows)==2
    for row in rows:
        assert row['evaluated_at_global_step']==row['global_step']-1
        assert row['preference_group_count']==row['group_count']
        assert 0<=row['policy_active_objective_fraction']<=1
        assert row['gradient_parameter_coverage']==1
        assert row['completion_tokens_per_second']>0
        assert 'learning/preference/gap' in scalar_values(row)
    assert raw[0]['parameter_update_sample_norm']>0
    assert 'parameter_update_sample_norm' not in raw[1]
    assert not any(k.startswith('_') for row in raw for k in row)
