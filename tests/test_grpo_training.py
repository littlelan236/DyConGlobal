"""Tail normalization, no replay, and rejection before optimizer mutation."""
from types import SimpleNamespace
import copy

import pytest
import torch

from RL.grpo.objective import ObjectiveResult
from RL.training import train_grpo


def group(index, slope):
    return SimpleNamespace(group_id=str(index), group_size=2, loss_mask=torch.tensor([[True],[True]]), slope=slope)


def train(model, groups, *, batch_size=2, stop=None, state=None):
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _:1.)
    seen = []; metrics = []
    def objective(m, g):
        seen.append(g.group_id)
        loss = m.weight.sum() * g.slope
        return ObjectiveResult(loss, loss, loss*0, {})
    result = train_grpo(model, groups, objective, opt, sched, group_batch_size=batch_size,
                       max_updates=stop, grad_clip=100., state=state,
                       on_update=lambda _,m: metrics.append(m))
    return result, seen, metrics


def test_tail_uses_actual_groups_and_epoch_does_not_wrap():
    model = torch.nn.Linear(1,1,bias=False)
    model.weight.data.zero_()
    groups = [group(0,1.),group(1,3.),group(2,10.)]
    state,seen,metrics = train(model, groups, stop=100)
    assert seen == ["0","1","2"]
    assert model.weight.item() == pytest.approx(-1.2)
    assert state["global_step"] == 2 and state["group_cursor"] == 3
    assert [m["group_count"] for m in metrics] == [2,1]


def test_resume_skips_consumed_prefix_without_repeating_groups():
    model = torch.nn.Linear(1,1,bias=False)
    groups = [group(0,1.),group(1,3.),group(2,10.)]
    paused,seen,_ = train(model, groups, stop=1)
    final,remaining,_ = train(model, groups, state=paused)
    assert seen + remaining == ["0","1","2"]
    assert final["group_cursor"] == 3


def test_nonfinite_objective_rejects_whole_window_before_step():
    model = torch.nn.Linear(1,1,bias=False)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="Nonfinite"):
        train(model,[group(0,1.),group(1,float("inf"))])
    torch.testing.assert_close(model.weight, before)
