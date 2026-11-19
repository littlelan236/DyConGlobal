"""Opt-in NCCL branch-round verification, without checkpoint/model exports.

Tests invoke the production loader/update/controller. Tensor comparisons stay
in memory; only a compact JSON receipt of test assertions is written.
"""
import json
import os
from pathlib import Path
from datetime import timedelta

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('DYCON_FULL_EXECUTION_GPU') != '1',
                               reason='Requires explicitly selected idle GPUs')


def _worker(rank, gpus, path, backend):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[rank])
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from RL.config import Config
    from RL.full_policy import load, optimizer, global_clip, activation_offloader
    from RL.full_training import window_update
    from RL.memory import MemoryController, attention_context
    from RL.parallel import GroupParallel
    from RL.probe import _configure_numerics
    from RL.grpo.execution import PreparedExecution, validate_window
    from RL.grpo.prepared import PreparedReader
    from core.io import read_json, write_json
    path = Path(path)
    config = Config.model_validate(read_json(path/'config.json'))
    from RL.allocator import configure_cuda_allocator
    configure_cuda_allocator(config.runtime.cuda_allocator, host_offload=config.training.memory.host_offload)
    _configure_numerics(config)
    torch.cuda.set_device(0)
    dist.init_process_group('gloo', init_method=(path/'store').as_uri(), rank=rank,
                            world_size=len(gpus), timeout=timedelta(seconds=180))
    data = dist.new_group(backend='nccl', timeout=timedelta(seconds=180))
    context = GroupParallel(rank, len(gpus), data_group=data)
    mesh = DeviceMesh.from_group(data, 'cuda', mesh_dim_names=('data',))
    policy, inventory = load(config, mesh, rank, len(gpus))
    opt, scheduler = optimizer(policy, inventory, config.training)
    reader = PreparedReader(read_json(path/'prepared.json')['path'])
    groups = PreparedExecution(reader, rank=rank, world_size=len(gpus), fsdp_replicate_empty=True)
    # Independent execution oracle: same shards/optimizer, but ordinary full
    # branch rounds instead of distinct-parent assignment. No checkpoint IO.
    baseline, base_inventory = load(config, mesh, rank, len(gpus))
    base_opt, base_schedule = optimizer(baseline, base_inventory, config.training)
    baseline.train()
    full_windows = iter(PreparedExecution(reader).windows(0))
    offload = activation_offloader(policy, config)
    controller = MemoryController(config, path, rank=rank, world_size=len(gpus),
                                  offloader=offload, device='cuda:0')
    policy.train()
    seen = []
    count = 0
    state = {'global_step':0, **groups.cursor(0)}
    for step, window in enumerate(groups.windows(0)):
        validate_window(reader.execution_plan,step,window)
        refs = {(g.group_id,bid):torch.full((len(g.completion_token_ids[b]),),-4.0)
                for g in window if g.input_ids is not None for b,bid in enumerate(g.branch_ids)}
        opt.zero_grad(set_to_none=True)
        offload.reset()
        controller.decide(window, global_step=step, group_cursor=count, parallel=context)
        controller.begin_attempt(window, global_step=step, attempt_id=0)
        with attention_context('auto'):
            loss, rows, table = window_update(policy,window,refs,config,context,controller)
        norm = global_clip(policy,1.0,data)
        lr = opt.param_groups[0]['lr']
        opt.step(); scheduler.step()
        from dataclasses import replace
        oracle_window = tuple(replace(g,metadata={}) for g in next(full_windows))
        base_refs = {(g.group_id,bid):torch.full((len(g.completion_token_ids[b]),),-4.0)
                     for g in oracle_window for b,bid in enumerate(g.branch_ids)}
        base_opt.zero_grad(set_to_none=True)
        with attention_context('auto'):
            base_loss,_,_ = window_update(baseline,oracle_window,base_refs,config,context)
        base_norm = global_clip(baseline,1.0,data)
        base_opt.step(); base_schedule.step()
        # Fixed same-backend tolerances; FLA comparisons never claim equality
        # with reference-PyTorch GDN execution.
        atol,rtol = (2e-4,2e-2) if backend=='fla_chunk' else (3e-6,3e-4)
        for (name,parameter),other in zip(policy.named_parameters(),baseline.parameters(),strict=True):
            for key in (None,'exp_avg','exp_avg_sq'):
                a = parameter if key is None else opt.state[parameter][key]
                b = other if key is None else base_opt.state[other][key]
                full,expected = a.full_tensor(),b.full_tensor()
                assert full.dtype == torch.float32 and torch.isfinite(full).all()
                torch.testing.assert_close(full,expected,atol=atol,rtol=rtol,msg=f'{name}/{key}')
        assert loss == pytest.approx(base_loss,abs=3e-4,rel=3e-3)
        assert norm == pytest.approx(base_norm,abs=3e-4,rel=3e-3)
        count += len(window)
        state = {'global_step':step+1, **groups.cursor(count)}
        groups.validate_state(state)
        transferred = offload.metrics()
        assert transferred['offload_d2h_bytes'] > 0
        assert transferred['offload_h2d_bytes'] == transferred['offload_d2h_bytes']
        assert transferred['host_pool_reserved_bytes'] == transferred['offload_live_bytes'] == 0
        seen.append({'step':step+1,'branches':len(rows),'microsteps':len(table),'lr':lr,
                     'loss':loss,'grad_norm':norm,'offload':transferred,'state':state})
    controller.close()
    assert state['consumed_groups'] == 9 and state['consumed_branches'] == 288
    assert [row['branches'] for row in seen] == [128,16,128,16]
    assert [row['microsteps'] for row in seen] == [64,16,64,16]
    assert [row['lr'] for row in seen] == pytest.approx([5e-4,1e-3,2e-3/3,1e-3/3])
    if rank == 0:
        write_json(path/'gpu-evidence.json',{'backend':backend,'world_size':len(gpus),'updates':seen})
    dist.barrier(); dist.destroy_process_group()


@pytest.mark.parametrize('backend', ['reference_pytorch','fla_chunk'])
def test_full_two_pass_real_nccl_without_checkpoint(tmp_path, backend):
    from tests.lora_helpers import raw_model
    from tests.test_full_execution import values
    from tests.test_grpo_prepared import dataset
    from tests.test_full_policy import ROOT
    from RL.config import Config
    from RL.grpo.prepared import materialize
    from core.io import write_json
    gpus = [int(v) for v in os.environ['DYCON_FULL_EXECUTION_GPUS'].split(',')]
    assert len(gpus) == 2 and len(set(gpus)) == 2
    torch.set_num_threads(1)
    model = raw_model()
    model.save_pretrained(tmp_path/'tiny-base')
    dataset(tmp_path,'native',range(9),branch_count=32)
    raw = values(2,warmup=2)
    raw['model'].update(path=str(tmp_path/'tiny-base'),
        dtype='bfloat16' if backend=='fla_chunk' else 'float32',gdn_backend=backend)
    raw['runtime'].update(gpus=gpus,output_root=str(tmp_path))
    raw['algorithm'].update(dataset_root=str(tmp_path/'sources'),dataset_id='native')
    raw['training'].update(learning_rate=1e-3,scheduler='linear')
    raw['training']['memory'].update(activation_offload='checkpoint_boundary',host_budget_gib=.05,
        loss_chunk_tokens=16,host_offload={'enabled':True,'allocator_budget_gib':.1,
            'cache_limit_gib':0.0,'reserve_gib':1.0,'reserve_fraction':.01})
    cfg = Config.model_validate(raw)
    reader = materialize(cfg,ROOT,'full-plan')
    # Keep the artifact's actual canonical path for the worker.
    write_json(tmp_path/'prepared.json',{'path':str(reader.path)})
    write_json(tmp_path/'config.json',raw)
    import torch.multiprocessing as mp
    mp.spawn(_worker,args=(gpus,str(tmp_path),backend),nprocs=2,join=True)
    assert (tmp_path/'gpu-evidence.json').exists()
