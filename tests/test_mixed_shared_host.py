"""Shared admission covers DyCon prefix lifetimes and mixed baseline windows."""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
from RL.config import Config, Memory
from RL.grpo.backend import GroupObjective
from tests.test_grpo_prefix import group


def memory():
    return Memory(activation_offload="checkpoint_boundary", host_budget_gib=None,
        shared_trace_host_budget_gib=460., offload_host_reserve_gib=1.,
        host_offload={"enabled": True, "allocator_budget_gib": 460., "reclaim_policy": "after_use"})


@pytest.mark.parametrize("failure", [False, True])
def test_group_reservation_covers_prefix_and_releases_on_all_exits(monkeypatch, failure):
    calls = []
    active = [False]
    class Offloader:
        trace_admission = object()
        @contextmanager
        def trace_scope(self, tokens, stage):
            calls.append(("enter", tokens))
            active[0] = True
            try:
                yield
            except BaseException:
                calls.append(("failed", tokens))
                raise
            finally:
                active[0] = False
                calls.append(("exit", tokens))
    class Prefix:
        device = NS(type="cpu")
        metrics = {}
        def __init__(self, *args, **kwargs):
            assert active[0]
            calls.append(("prefix", 0))
            if failure:
                raise RuntimeError("injected failure")
        def finish(self):
            assert active[0]
        def close(self):
            assert active[0]
    monkeypatch.setattr("RL.grpo.prefix.PrefixContext", Prefix)
    row = group(5, 3)
    settings = NS(prefix_reuse=NS(mode="boundary_grad"), kl_beta=0.)
    objective = GroupObjective({}, settings, 128, offloader=Offloader())
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            try:
                objective.begin_group(None, row)
            finally:
                objective.close_group()
        assert ("failed", 10) in calls
    else:
        objective.begin_group(None, row)
        objective.finish_group()
        objective.close_group()
        # Closing an already released group is safe.
        objective.close_group()
    assert calls[0] == ("enter", 10) and calls[-1] == ("exit", 10)
    assert not active[0] and objective.host_scope is None


def test_mixed_config_accepts_original_baseline_shared_memory():
    from pathlib import Path
    from RL.config import load
    root = Path(__file__).parents[1]
    cfg, _ = load(root/"RL/experiments/grpo-offline-k3-smoke.yaml", root)
    raw = cfg.model_dump()
    raw["offline_baseline"] = {"dataset": "/unused"}
    raw["training"]["memory"] = memory().model_dump()
    validated = Config.model_validate(raw)
    assert validated.training.memory.host_budget_gib is None
    assert validated.training.memory.shared_trace_host_budget_gib == 460.
