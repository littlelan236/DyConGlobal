import pytest
from types import SimpleNamespace
from RL.lr_schedule import factor

@pytest.mark.parametrize("scheduler, expected", [
    ("constant", [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]),
    ("linear", [0.25, 0.5, 0.75, 1.0, 2/3, 1/3]),
])
def test_warmup_updates_are_optimizer_steps_and_compose_with_decay(scheduler, expected):
    settings = SimpleNamespace(warmup_updates=4, max_updates=6, scheduler=scheduler)
    assert [factor(step, settings) for step in range(6)] == pytest.approx(expected)

def test_zero_warmup_preserves_existing_schedule():
    settings = SimpleNamespace(warmup_updates=0, max_updates=4, scheduler="linear")
    assert [factor(step, settings) for step in range(5)] == pytest.approx([1, .75, .5, .25, 0])


@pytest.mark.parametrize('mode', ['lora', 'full'])
@pytest.mark.parametrize('kind', ['constant', 'linear'])
def test_both_optimizers_follow_warmup_and_in_memory_restore(mode, kind):
    import copy
    import torch
    from tests.lora_helpers import settings
    from RL.training import optimizer_and_scheduler
    from RL.full_policy import optimizer
    settings = settings(6)
    settings.scheduler = kind
    settings.warmup_updates = 4

    def build():
        model = torch.nn.Linear(2, 2)
        if mode == 'lora':
            opt, schedule = optimizer_and_scheduler(model, settings)
        else:
            inventory = {'parameters':[{'name':n,'weight_decay':True} for n,_ in model.named_parameters()]}
            opt, schedule = optimizer(model, inventory, settings)
        return model, opt, schedule

    model, opt, schedule = build()
    expected = [factor(i, settings)*settings.learning_rate for i in range(7)]
    seen = []
    for step in range(6):
        seen.append(opt.param_groups[0]['lr'])
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        schedule.step()
        if step == 1:
            saved = copy.deepcopy((model.state_dict(), opt.state_dict(), schedule.state_dict()))
    seen.append(opt.param_groups[0]['lr'])
    assert seen == pytest.approx(expected)
    restored, ropt, rschedule = build()
    restored.load_state_dict(saved[0]); ropt.load_state_dict(saved[1]); rschedule.load_state_dict(saved[2])
    for step in range(2,6):
        assert ropt.param_groups[0]['lr'] == pytest.approx(expected[step])
        for p in restored.parameters():
            p.grad = torch.ones_like(p)
        ropt.step(); rschedule.step()
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert schedule.state_dict() == rschedule.state_dict()
