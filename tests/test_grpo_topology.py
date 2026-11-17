"""Physical 2/4/8-worker changes preserve immutable optimizer windows."""
import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.io import read_json, sha256, write_json
from RL.config import Config, attempt_directory, schedule_contract
from RL.parallel import GroupParallel, execution_contract
from RL.topology import (POLICY, configure, tasks, owns_parent, merge_rows,
                         validate_resume, execution_record, assert_rng_unchanged)
from RL.training import capture_rng, train_grpo


def _whole_groups():
    from RL.grpo.groups import GroupBatch
    result = []
    for i, size in enumerate([32, 17, 33, 3, 32, 31, 3, 32, 19, 32, 3, 32, 17]):
        lengths = [1 + b % 3 for b in range(size)]
        mask = torch.zeros(size, 3, dtype=torch.bool)
        for b, length in enumerate(lengths):
            mask[b, :length] = True
        result.append(GroupBatch(str(i), [f'{i}-{b}' for b in range(size)], [[1, 2]]*size,
            [list(range(n)) for n in lengths], None, None, None, mask,
            [float(b) for b in range(size)], [(b-(size-1)/2)/size for b in range(size)]))
    return result


class _Objective:
    def iter_results(self, model, group):
        from RL.grpo.objective import grpo_objective
        for i, bid in enumerate(group.branch_ids):
            ordinal = int(bid.split('-')[1])
            theta = (model.weight[0] + model.weight[1] * torch.arange(1, len(group.completion_token_ids[i])+1)*.03)[None]
            if int(group.group_id) % 2:
                theta = theta + model.conditional
            result = grpo_objective(theta, torch.full_like(theta, -.2), [group.advantages[i]],
                torch.ones_like(theta, dtype=torch.bool), kl_beta=.13)
            result.loss = result.loss / group.group_size
            result.metrics.update(branch_id=bid, native_branch_ordinal=ordinal)
            yield result


def test_task_coverage_tail_weights_and_rank_mapping():
    window = _whole_groups()[:4]
    for world in (2, 4, 8):
        peers = []
        for rank in range(world):
            rows = []
            for index, task, fraction in tasks(window, rank=rank, world_size=world):
                assert fraction/task.group_size/len(window) == pytest.approx(1/(window[index].group_size*len(window)))
                assert owns_parent(index, [g.group_size for g in window], rank, world)
                rows.append((index, [{'branch_id': bid, 'native_branch_ordinal': ordinal}
                    for bid, ordinal in zip(task.branch_ids, task.metadata['physical_task']['native_branch_ordinals'])]))
            peers.append({'rank': rank, 'rows': rows})
        merged = merge_rows(window, peers)
        assert [len(r) for r in merged] == [g.group_size for g in window]
        broken = copy.deepcopy(peers)
        broken[0]['rows'].append(copy.deepcopy(broken[0]['rows'][0]))
        with pytest.raises(ValueError, match='duplicate'):
            merge_rows(window, broken)


def test_training_rng_consumption_is_rejected():
    saved = capture_rng()
    torch.rand(1)
    with pytest.raises(ValueError, match='RNG consumption'):
        assert_rng_unchanged(saved)


def test_nonzero_dropout_and_stochastic_attention_are_rejected():
    from RL.topology import audit_model
    with pytest.raises(ValueError, match='dropout'):
        audit_model(torch.nn.Sequential(torch.nn.Dropout(.1)))
    model = torch.nn.Linear(2,2)
    model.attention_dropout = .1
    with pytest.raises(ValueError, match='stochastic model'):
        audit_model(model)


def test_topology_admission_is_torch_free():
    import subprocess
    import sys
    code = '''
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'torch' or fullname.startswith('torch.'):
   raise AssertionError('topology admission imported Torch')
sys.meta_path.insert(0,NoTorch())
from RL.topology import validate_record
validate_record(dict(version='logical_window_remap_v1',policy='preserve_updates',
 world_size=2,gpu_ids=[0,1],whole_parent_task_branches=16,rng_policy='reject_training_rng_consumption_v1'))
'''
    subprocess.run([sys.executable,'-c',code],check=True,timeout=20)


def test_portable_config_is_opt_in_and_keeps_schedule(tmp_path):
    from tests.test_grpo_prepared import config
    from tests.test_grpo_execution import cfg32
    cfg = config(tmp_path)
    original = schedule_contract(cfg)
    cfg.runtime.gpu = None
    cfg.runtime.gpus = list(range(8))
    with pytest.raises(ValueError, match='8 GPUs'):
        Config.model_validate(cfg.model_dump())
    cfg.runtime.topology_policy = POLICY
    assert schedule_contract(Config.model_validate(cfg.model_dump())) == original
    split = cfg32(tmp_path/'split')
    for count in (2, 4, 8):
        split.runtime.gpus = list(range(count))
        split.runtime.topology_policy = POLICY
        Config.model_validate(split.model_dump())


def test_resume_admission_preserves_all_nonphysical_protocol_fields(tmp_path):
    from tests.test_grpo_prepared import config
    cfg = config(tmp_path)
    cfg.runtime.gpu = None
    cfg.runtime.gpus = [0, 1]
    saved = {'contract': {'parallel': execution_contract(cfg), 'prepared': {}}}
    cfg.run.start.mode = 'resume'
    cfg.runtime.gpus = list(range(8))
    with pytest.raises(ValueError, match='same data-parallel'):
        validate_resume(cfg, saved)
    cfg.runtime.topology_policy = POLICY
    validate_resume(cfg, saved)
    broken = copy.deepcopy(saved)
    broken['contract']['parallel']['gradient_reduction'] = 'mean'
    with pytest.raises(ValueError, match='logical execution'):
        validate_resume(cfg, broken)
    saved['physical_execution'] = execution_record(cfg)
    cfg.runtime.gpus = [0, 1]
    validate_resume(cfg, saved)
    cfg.runtime.topology_policy = 'exact'
    with pytest.raises(ValueError, match='Portable checkpoints'):
        validate_resume(cfg, saved)


def _math_worker(rank, world, folder, stage, previous):
    from tests.test_grpo_parallel import Model
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(Path(folder)/f'store-{stage}').as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        parallel = GroupParallel(rank, world)
        parallel.topology_policy = POLICY
        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.8)
        state = None
        if previous:
            payload = torch.load(previous, weights_only=True)
            model.load_state_dict(payload['model'])
            optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler'])
            state = payload['state']
        gradients, updates = [], []
        state = train_grpo(model, _whole_groups(), _Objective(), optimizer, scheduler,
            state=state, parallel=parallel, group_batch_size=4, grad_clip=.25, max_updates=stage,
            on_gradients=lambda: gradients.append({n: p.grad.clone() if p.grad is not None else None
                                                   for n, p in model.named_parameters()}),
            on_update=lambda s, m: updates.append((s, m)))
        if rank == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'state': state, 'gradients': gradients,
                'updates': updates}, Path(folder)/f'stage-{stage}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _rng_worker(rank, folder):
    from tests.test_grpo_parallel import Model
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(Path(folder)/'rng-store').as_uri(),
                            rank=rank,world_size=2,timeout=timedelta(seconds=45))
    try:
        parallel = GroupParallel(rank,2)
        parallel.topology_policy = POLICY
        model = Model()
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(),lr=.03)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1)
        class Stochastic(_Objective):
            def iter_results(self, model, group):
                for result in super().iter_results(model,group):
                    result.loss = result.loss * torch.rand(())
                    yield result
        with pytest.raises(ValueError,match='RNG consumption'):
            train_grpo(model,_whole_groups(),Stochastic(),optimizer,scheduler,
                       group_batch_size=4,parallel=parallel)
        assert not optimizer.state
        assert scheduler.last_epoch == 0
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(model.state_dict()[key],value) for key,value in before.items())
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_all_workers_reject_random_training_before_optimizer_mutation(tmp_path):
    mp.spawn(_rng_worker,args=(str(tmp_path),),nprocs=2,join=True)


@pytest.mark.parametrize('counts', [(2, 8, 4, 2), (8, 2, 4, 8)])
def test_real_collectives_bidirectional_match_independent_unsplit_oracle(tmp_path, counts):
    from tests.test_grpo_parallel import Model
    torch.set_num_threads(1)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.8)
    gradients, updates = [], []
    expected = train_grpo(model, _whole_groups(), _Objective(), optimizer, scheduler,
        group_batch_size=4, grad_clip=.25,
        on_gradients=lambda: gradients.append({n:p.grad.clone() if p.grad is not None else None
                                               for n,p in model.named_parameters()}),
        on_update=lambda s,m: updates.append((s,m)))
    previous = None
    for stage, world in enumerate(counts, 1):
        mp.spawn(_math_worker, args=(world, str(tmp_path), stage, previous), nprocs=world, join=True)
        previous = str(tmp_path/f'stage-{stage}.pt')
        actual = torch.load(previous, weights_only=True)
        assert actual['state'] == updates[stage-1][0]
        for name, want in gradients[stage-1].items():
            got = actual['gradients'][0][name]
            if want is None:
                assert got is None
            else:
                torch.testing.assert_close(got, want, atol=5e-7, rtol=5e-6)
        assert actual['updates'][0][1]['loss'] == pytest.approx(updates[stage-1][1]['loss'], abs=1e-7)
        for key in ('group_count','branch_count','token_count','policy_loss','k3_penalty_mean'):
            assert actual['updates'][0][1][key] == pytest.approx(updates[stage-1][1][key], abs=1e-7)
    for name, want in model.state_dict().items():
        torch.testing.assert_close(actual['model'][name], want, atol=5e-7, rtol=5e-6)
    for key, value in optimizer.state_dict()['state'].items():
        for field, want in value.items():
            torch.testing.assert_close(actual['optimizer']['state'][key][field], want, atol=5e-7, rtol=5e-6)
    assert actual['scheduler'] == scheduler.state_dict()
    assert actual['state'] == expected


def _service_worker(rank, world, cfg, rendezvous):
    import RL.parallel as parallel_module
    from RL.service import _source, _train_grpo
    from tests.test_grpo_prepared import ROOT, BASE
    from tests.lora_helpers import model_and_targets
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=world, timeout=timedelta(seconds=120))
    try:
        parallel_module._context = GroupParallel(rank, world)
        parent = _source(cfg)
        model, targets = model_and_targets(settings=cfg.lora, adapter=parent['adapter'] if parent else None)
        parallel_module._context.synchronize_model(model)
        directory = attempt_directory(cfg)
        location = directory if rank == 0 else directory/'distributed'/f'rank-{rank}'
        location.mkdir(parents=True, exist_ok=True)
        attempt = {'attempt_id':cfg.run.attempt_id, 'purpose':'engineering_smoke', 'start_mode':cfg.run.start.mode}
        result = _train_grpo(cfg, ROOT, location, BASE, parent, attempt, model, targets,
                             optimizer_settings=cfg.training)
        if rank == 0:
            write_json(directory/'test-result.json', result)
        dist.barrier()
    finally:
        parallel_module._context = None
        dist.destroy_process_group()


@pytest.mark.parametrize('split,streaming', [(False,False), (False,True), (True,False), (True,True)])
def test_tiny_qwen_service_checkpoint_roundtrip_keeps_anchor_cursor_and_optimizer(tmp_path, split, streaming):
    from tests.test_grpo_prepared import config, dataset, train
    from tests.test_grpo_execution import cfg32
    from tests.test_grpo_external_reference import make_source, select
    from tests.test_grpo_lifecycle import assert_tree_close
    from RL.checkpoints import verify_checkpoint
    if split:
        cfg = cfg32(tmp_path, n=9, streaming=streaming)
        counts = (8,2,4,8)
    else:
        dataset(tmp_path,'a',range(13),branch_count=32)
        cfg = config(tmp_path,updates=4)
        cfg.grpo.group_batch_size = 4
        cfg.grpo.prefix_reuse.mode = 'boundary_grad'
        cfg.training.checkpoint_every_updates = 1
        cfg.runtime.data_preparation_async = streaming
        counts = (2,8,4,2)
    cfg.preflight.verify_identity = False
    cfg.preflight.audit_native_tokens = False
    cfg.runtime.gpu = None
    cfg.runtime.gpus = list(range(counts[0]))
    if streaming:
        def native_rows(manifest, rows, path):
            for row in rows:
                row['native_request_id'] = row['branch_sample_id']
        make_source(tmp_path/'probabilities',cfg,mutate=native_rows)
        select(cfg,tmp_path/'probabilities')
    oracle,_,_,_ = train(cfg,'oracle')
    previous = None
    first_contract = None
    preserved = {}
    for stage, world in enumerate(counts,1):
        stage_cfg = cfg.model_copy(deep=True)
        stage_cfg.run.attempt_id = f'physical-{stage}-{world}'
        stage_cfg.runtime.gpus = list(range(world))
        # Start from a historical-style exact checkpoint, then adopt portability.
        stage_cfg.runtime.topology_policy = 'exact' if stage == 1 else POLICY
        stage_cfg.run.start.mode = 'fresh' if previous is None else 'resume'
        stage_cfg.run.start.checkpoint = str(previous) if previous else None
        stage_cfg.validation.stop_after_update = stage if stage<4 else None
        stage_cfg = Config.model_validate(stage_cfg.model_dump())
        if previous:
            validate_resume(stage_cfg,read_json(previous/'state.json'))
        mp.spawn(_service_worker,args=(world,stage_cfg,str(tmp_path/f'service-store-{stage}')),
                 nprocs=world,join=True)
        directory = attempt_directory(stage_cfg)
        previous = directory/'checkpoints'/f'step-{stage:06d}'
        verify_checkpoint(previous)
        if stage > 1:
            transition = read_json(previous/'topology-transition.json')
            assert transition['source_receipt_sha256'] == sha256(stage_cfg.run.start.checkpoint+'/receipt.json')
        metadata = read_json(previous/'state.json')
        if first_contract is None:
            first_contract = metadata['contract']
        else:
            assert metadata['contract'] == first_contract
            assert metadata['physical_execution']['world_size'] == world
        want = torch.load(oracle/'checkpoints'/f'step-{stage:06d}'/'training-state.pt',weights_only=True)
        got = torch.load(previous/'training-state.pt',weights_only=True)
        assert len(got['rank_rng']) == world
        assert got['metadata']['state'] == want['metadata']['state']
        # Different collective orders may perturb floating point values slightly.
        assert_tree_close(got['optimizer'],want['optimizer'])
        assert got['scheduler'] == want['scheduler']
        from safetensors.torch import load_file
        for key, value in load_file(str(oracle/'checkpoints'/f'step-{stage:06d}'/'adapter/adapter_model.safetensors')).items():
            torch.testing.assert_close(load_file(str(previous/'adapter/adapter_model.safetensors'))[key], value,
                                       atol=2e-6,rtol=2e-4)
        for path, digest in preserved.items():
            assert sha256(path) == digest
        preserved.update({p:sha256(p) for p in previous.rglob('*') if p.is_file()})
        if split:
            assert metadata['state']['consumed_execution_chunks'] == [8,9,17,18][stage-1]
            assert metadata['state']['completed_parent_groups'] == [0,0,8,9][stage-1]
