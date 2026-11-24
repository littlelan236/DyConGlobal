"""Shared frozen 8x16 plans under full-policy FSDP2 (no checkpoint exports)."""
import copy
from types import SimpleNamespace

import pytest
import torch

from RL.config import Config, schedule_contract
from RL.full_policy import FullPolicy, microsteps, parameter_scope
from RL.grpo.execution import PreparedExecution, cursor, validate_state, validate_window
from RL.grpo.prepared import materialize
from tests.test_full_policy import ROOT, full_values
from tests.test_grpo_prepared import dataset


def values(world=2, warmup=0):
    raw = full_values()
    raw['runtime'].pop('gpu', None)
    raw['runtime']['gpus'] = list(range(world))
    raw['grpo'].update(group_batch_size=8, parallel_branch_chunk_size=16,
        parallel_branch_schedule='distinct_parent_two_pass_v1',
        prefix_reuse={'mode': 'disabled'}, top_k=1, top_k_fraction=None)
    raw['training'].update(max_updates=4, warmup_updates=warmup)
    return raw


@pytest.mark.parametrize('world', [2, 4, 8])
def test_complete_frozen_halves_tail_rounds_and_resume_cursor(tmp_path, world):
    dataset(tmp_path, 'native', range(9), branch_count=32)
    raw = values(world)
    raw['algorithm'].update(dataset_root=str(tmp_path/'sources'), dataset_id='native')
    raw['runtime']['output_root'] = str(tmp_path)
    cfg = Config.model_validate(raw)
    reader = materialize(cfg, ROOT, 'full-plan')
    views = [PreparedExecution(reader, rank=r, world_size=world, fsdp_replicate_empty=True)
             for r in range(world)]
    windows = [list(view.windows(0)) for view in views]
    assert [len(w) for w in windows[0]] == [8, 1, 8, 1]
    covered = []
    count = 0
    for step, rank_windows in enumerate(zip(*windows)):
        for rank, window in enumerate(rank_windows):
            validate_window(reader.execution_plan, step, window)
            table = microsteps(window, world)
            assert len(table) == 16*((len(window)+world-1)//world)
            for assignments in table:
                task = assignments[rank]
                if task is not None:
                    g, b = task
                    assert window[g].input_ids is not None
                    covered.append((window[g].group_id, window[g].branch_ids[b]))
                else:
                    assert any(g.input_ids is not None for g in window)
        count += len(rank_windows[0])
        state = {'global_step': step+1, **cursor(reader.execution_plan, count)}
        validate_state(reader.execution_plan, state)
        resumed = list(views[0].windows(count))
        assert len(resumed) == 3-step
        if step < 2:
            assert state['completed_parent_groups'] == 0
    assert len(covered) == len(set(covered)) == 9*32
    assert state['completed_parent_groups'] == state['consumed_groups'] == 9
    assert state['consumed_branches'] == 288


@pytest.mark.parametrize('world', [2, 4, 8])
def test_full_split_config_keeps_topology_and_prefix_constraints(world):
    assert Config.model_validate(values(world)).grpo.parallel_branch_chunk_size == 16
    for mutate in [lambda r: r['runtime'].update(gpus=[0]),
                   lambda r: r['grpo'].update(group_batch_size=4),
                   lambda r: r['grpo'].update(prefix_reuse={'mode':'boundary_grad'}),
                   lambda r: r['grpo'].update(parallel_branch_schedule=None),
                   lambda r: r['runtime'].update(topology_policy='preserve_updates')]:
        raw = values(world)
        mutate(raw)
        with pytest.raises(ValueError):
            Config.model_validate(raw)


def test_warmup_default_contract_and_optin_identity():
    plain = Config.model_validate(full_values())
    assert 'warmup_updates' not in schedule_contract(plain)['training']
    warm = Config.model_validate(values(warmup=2))
    assert schedule_contract(warm)['training']['warmup_updates'] == 2
    raw = values(warmup=5)
    with pytest.raises(ValueError, match='warmup_updates'):
        Config.model_validate(raw)


@pytest.mark.parametrize('world', [2, 4, 8])
@pytest.mark.parametrize('parents', [1, 3, 8])
def test_full_split_gradients_match_independent_branch_average(world, parents, monkeypatch):
    """All parameter gradients, with real model autograd and simulated rank mean."""
    from tests.lora_helpers import raw_model
    from tests.test_grpo_prefix import group
    from RL.full_training import window_update
    from RL.grpo.objective import grpo_objective
    torch.set_num_threads(2)
    original = raw_model()
    inventory = parameter_scope(original)
    groups = [group(1, 16, 'parent'+str(i)) for i in range(parents)]
    from dataclasses import replace
    groups = [replace(g, metadata={'execution_chunk': {'native_branch_ordinals':list(range(16))}})
              for g in groups]
    native = type(groups[0]).branch_batch
    monkeypatch.setattr(type(groups[0]), 'branch_batch', lambda self, b, dev: native(self, b, 'cpu'))
    config = Config.model_validate(values(world))
    refs = {(g.group_id,bid):torch.full((len(g.completion_token_ids[b]),), -4.0)
            for g in groups for b,bid in enumerate(g.branch_ids)}
    oracle = FullPolicy(copy.deepcopy(original))
    loss = 0
    for g in groups:
        for b,bid in enumerate(g.branch_ids):
            scores = oracle(g.branch_batch(b,'cpu'),16)
            result = grpo_objective(scores[None],refs[(g.group_id,bid)][None],[g.advantages[b]],
                torch.ones_like(scores[None],dtype=torch.bool),clip_epsilon=config.grpo.clip_epsilon,
                kl_beta=config.grpo.kl_beta,k3_cap=config.grpo.k3_cap,
                ratio_guard_enabled=config.grpo.ratio_guard.enabled,
                ratio_guard_threshold=config.grpo.ratio_guard.threshold)
            loss = loss+result.loss/(parents*16)
    loss.backward()
    expected_keys = [(g.group_id,bid) for g in groups for bid in g.branch_ids]
    policies=[]
    for rank in range(world):
        def gather(obj):
            # Simulate the row-coverage collective; each policy retains its own
            # local gradients, which are explicitly averaged below.
            present={(r['group_id'],r['branch_id']) for r in obj['rows']}
            padding=[{'group_id':g,'branch_id':b} for g,b in expected_keys if (g,b) not in present]
            return [obj,{'rows':padding,'loss':0.0}]
        policy=FullPolicy(copy.deepcopy(original))
        window_update(policy,groups,refs,config,SimpleNamespace(rank=rank,world_size=world,gather=gather))
        policies.append(policy)
    for name, expected in oracle.named_parameters():
        actual=torch.stack([dict(p.named_parameters())[name].grad for p in policies]).mean(0)
        torch.testing.assert_close(actual,expected.grad,atol=3e-6,rtol=3e-4,msg=name)
