"""Opt-in bounded CUDA lifecycle and two-rank NCCL transaction evidence.

Pressure inputs are controlled, explicitly labelled, and do not consume the free
memory of other jobs. Each policy starts in a fresh spawned process.
"""
import json
import os
from pathlib import Path
import time
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("DYCON_HOST_OFFLOAD_GPU") != "1",
                               reason="requires selected idle CUDA devices")


def pss():
    values = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in {'Pss', 'Pss_Anon', 'Pss_Shmem', 'Rss'}:
            values[key + '_bytes'] = int(value.split()[0]) * 1024
    return values


def _lifecycle(rank, physical, directory, policy):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical)
    from RL.allocator import configure_cuda_allocator
    from RL.config import CudaAllocator, HostOffload, MemoryPressure
    from RL.host_offload import host_allocator_stats
    from RL.memory import BoundaryOffload, local_saved_tensors
    from RL.resources import host_memory
    managed = policy in {'managed', 'pressure'}
    host = HostOffload(enabled=managed, allocator_budget_gib=0.125, cache_limit_gib=0.001,
                       reserve_gib=8.0, reserve_fraction=0.03)
    configure_cuda_allocator(CudaAllocator(), host_offload=host)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    torch.cuda.set_device(0)
    model = torch.nn.ModuleList([torch.nn.Linear(256, 256) for _ in range(8)]).cuda()
    controller = BoundaryOffload(model, 128*2**20, enabled=policy != 'none', host_settings=host,
        pressure_settings=MemoryPressure(device_headroom_gib=0.01) if policy == 'pressure' else None)
    pressure_index = [0]
    if policy == 'pressure':
        # Controlled alternating pressure inputs exercise real partial transfers
        # without allocating tens of GiB to manufacture a driver-level OOM.
        import RL.host_offload as api
        original = api.pressure_choice
        def controlled(**kwargs):
            pressure_index[0] += 1
            actual = kwargs['driver_free_bytes']
            kwargs['driver_free_bytes'] = 0 if pressure_index[0] % 2 == 0 else 2**60
            return {**original(**kwargs), 'actual_driver_free_bytes': actual,
                    'controlled_pressure_input': True}
        api.pressure_choice = controlled
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    rows = []
    oracle = {}
    lengths = [64, 2048, 64, 3072] * 20
    for step, length in enumerate(lengths):
        controller.reset()
        # Short windows intentionally stay on GPU, including after long windows.
        controller.set_enabled(policy != 'none' and length > 64)
        generator = torch.Generator(device='cuda').manual_seed(900+length)
        x = torch.randn((1, length, 256), generator=generator, device='cuda', requires_grad=True)
        if length not in oracle:
            value = x
            with local_saved_tensors():
                for layer in model: value = torch.tanh(layer(value))
                reference = value.square().mean()
                reference.backward()
            oracle[length] = (float(reference), [p.grad.cpu().clone() for p in model.parameters()], x.grad.cpu().clone())
            optimizer.zero_grad(set_to_none=True)
            x.grad = None
            del value, reference
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        value = x
        for layer in model:
            value = controller.checkpoint(lambda t, layer=layer: torch.tanh(layer(t)), value)
        loss = value.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        duration = time.monotonic()-started
        expected_loss, expected_grads, expected_x = oracle[length]
        torch.testing.assert_close(loss.detach().cpu(), torch.tensor(expected_loss), atol=1e-6, rtol=1e-5)
        for p, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(p.grad.cpu(), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(x.grad.cpu(), expected_x, atol=1e-6, rtol=1e-5)
        peak = torch.cuda.max_memory_allocated()
        del loss, value, x
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        before = host_allocator_stats()
        controller.safe_boundary()
        metrics = controller.metrics()
        assert metrics['offload_live_bytes'] == 0
        if managed and controller.pool is not None:
            assert metrics['host_pool_idle_bytes'] <= controller.pool.cache_limit
            assert metrics['host_allocated_bytes'] <= controller.pool.budget
            assert metrics['host_pool_reserved_bytes'] <= controller.pool.cache_limit
        if policy == 'pressure' and length > 64:
            assert metrics['offload_selected_boundaries'] > 0 and metrics['offload_retained_boundaries'] > 0
        if length == 64: assert metrics['offload_d2h_bytes'] == 0
        row = {'step': step+1, 'length': length, 'seconds': duration, 'gpu_peak_allocated_bytes': peak,
               'before_trim': before, 'metrics': metrics, 'host': host_memory(), 'pss': pss()}
        rows.append(row)
    path = Path(directory)/f'{policy}.json'
    path.write_text(json.dumps({'physical_gpu':physical, 'policy':policy, 'rows':rows,
        'all_gradients_close':True, 'controlled_pressure_inputs':policy == 'pressure'},indent=2))


def test_fresh_process_short_long_short_twenty_rounds(tmp_path):
    evidence = Path(os.environ.get('DYCON_HOST_OFFLOAD_EVIDENCE', str(tmp_path))).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    gpu = int(os.environ.get('DYCON_HOST_OFFLOAD_GPUS', '4,5').split(',')[0])
    for policy in ['none', 'legacy', 'managed', 'pressure']:
        mp.spawn(_lifecycle, args=(gpu, str(evidence), policy), nprocs=1, join=True)
    result = {p:json.loads((evidence/f'{p}.json').read_text()) for p in ['none','legacy','managed','pressure']}
    assert result['legacy']['rows'][-1]['metrics']['host_allocated_bytes'] > 0
    assert result['managed']['rows'][-1]['metrics']['host_pool_reserved_bytes'] <= 0.001*2**30
    for index in [1,3]:
        assert result['pressure']['rows'][index]['gpu_peak_allocated_bytes'] < result['none']['rows'][index]['gpu_peak_allocated_bytes']
    (evidence/'lifecycle-acceptance.json').write_text(json.dumps({'passed':True,'rounds':20,
        'fresh_processes':4,'all_gradients_close':True,'controlled_pressure_inputs':True},indent=2))


def _nccl_worker(rank, selected, directory):
    import torch.distributed as dist
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected[rank])
    from RL.allocator import configure_cuda_allocator
    from RL.config import CudaAllocator, Memory
    from RL.memory import BoundaryOffload, MemoryController
    from RL.parallel import GroupParallel
    from RL.training import train_grpo, seed_everything
    from RL.modeling import adapter_fingerprint  # Production imports modeling before seeding/training.
    from tests.test_lora_oom_recovery import Model, Objective, Windows, groups, settings
    configure_cuda_allocator(CudaAllocator(), host_offload=Memory(host_offload={'enabled':True}).host_offload)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    dist.init_process_group('gloo',init_method=(Path(directory)/'nccl-store').as_uri(),
                            rank=rank,world_size=2,timeout=timedelta(seconds=60))
    data = dist.new_group(backend='nccl',timeout=timedelta(seconds=60))
    try:
        context = GroupParallel(rank,2,data_group=data)
        snapshots = []
        for label, fault in [('direct',False),('retry',True)]:
            seed_everything(110+rank)
            model = Model().cuda()
            cfg = settings('memory_pressure')
            cfg.training.memory.host_offload.reserve_fraction = 0.03
            cfg.training.memory.host_offload.reserve_gib = 8.0
            offload = BoundaryOffload(model,64*2**20,enabled=True,
                host_settings=cfg.training.memory.host_offload, pressure_settings=cfg.training.memory.memory_pressure)
            cfg.training.memory.memory_pressure.device_headroom_gib = 200.0 if rank else 0.001
            class GPUObjective(Objective):
                def iter_results(self, model, group):
                    # Unequal local activation placement with real async copies.
                    # Parameter storage is excluded; this independent checkpoint
                    # input is restored in backward before the existing objective.
                    for item in super().iter_results(model,group):
                        x=torch.ones((1,8+rank*8,2),device='cuda',requires_grad=True)
                        anchor=offload.checkpoint(lambda t:t.sin(),x).sum()*0
                        item.loss=item.loss+anchor
                        yield item
            # Objective injects on group index 2, assigned to rank 0.
            objective = GPUObjective('once' if fault and rank == 0 else None)
            controller=MemoryController(cfg,Path(directory)/f'{label}-{rank}',rank=rank,world_size=2,
                offloader=offload,device='cuda')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.03)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.8)
            seen=[]
            state=train_grpo(model,Windows(groups(5)),objective,optimizer,scheduler,group_batch_size=4,
                grad_clip=100.,parallel=context,memory_controller=controller,
                on_window_start=offload.reset,on_update=lambda s,m:seen.append(m))
            snapshots.append((model.weight.detach().cpu().clone(),state,optimizer.state_dict(),scheduler.state_dict()))
            assert seen[0]['step_attempt'] == int(fault)
            if rank:
                assert seen[0]['offload_selected_boundaries'] > 0
            else:
                assert seen[0]['offload_retained_boundaries'] > 0
        from tests.test_grpo_lifecycle import assert_tree_close
        assert_tree_close(snapshots[0],snapshots[1])
        (Path(directory)/f'nccl-{rank}.json').write_text(json.dumps({'rank':rank,'passed':True,
            'real_nccl_gradients':True,'fault_injected':True,'natural_cuda_oom_observed':False}))
    finally:
        dist.destroy_process_group()


def test_two_cuda_ranks_nccl_and_gloo_transaction(tmp_path):
    evidence=Path(os.environ.get('DYCON_HOST_OFFLOAD_EVIDENCE',str(tmp_path))).resolve()/'dual'
    evidence.mkdir(parents=True,exist_ok=True)
    selected=[int(x) for x in os.environ.get('DYCON_HOST_OFFLOAD_GPUS','4,5').split(',')]
    mp.spawn(_nccl_worker,args=(selected[:2],str(evidence)),nprocs=2,join=True)


def _retained_graph_worker(rank, physical, directory):
    import gc
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical)
    from RL.allocator import configure_cuda_allocator
    from RL.config import CudaAllocator, HostOffload
    from RL.host_offload import host_allocator_stats, TorchHostAllocator, NATIVE_CACHE_BYTES
    from RL.memory import BoundaryOffload
    host = HostOffload(enabled=True, allocator_budget_gib=0.125, cache_limit_gib=0)
    configure_cuda_allocator(CudaAllocator(), host_offload=host)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    # Populate a useful native cache independently of the activation pool.
    small = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    small.fill_(1)
    del small
    baseline = host_allocator_stats()['host_allocated_bytes']
    model = torch.nn.Linear(16, 16).cuda()
    offload = BoundaryOffload(model, 64*2**20, host_settings=host)
    x = torch.randn(1, 64, 16, device='cuda', requires_grad=True)
    output = offload.checkpoint(lambda t: t.sin(), x)
    loss = output.sum()
    loss.backward(retain_graph=True)
    offload.reclaim_completed()
    assert offload.live_bytes > 0 and offload.pool.metrics()['host_pool_leased_bytes'] > 0
    x.grad = None
    other_stream = torch.cuda.Stream()
    other_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other_stream): loss.backward()
    torch.cuda.current_stream().wait_stream(other_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(x.grad, x.cos())
    del loss, output
    gc.collect()
    offload.safe_boundary()
    metrics = offload.pool.metrics()
    assert metrics['host_pool_reserved_bytes'] == metrics['host_pool_pending_bytes'] == 0
    assert 4096 <= metrics['host_native_cache_bytes'] <= baseline + 4096
    assert metrics['host_allocated_bytes'] <= baseline + 4096
    assert TorchHostAllocator.charge(4) > NATIVE_CACHE_BYTES
    offload.pool.close()
    assert offload.pool.close() is None
    (Path(directory)/'retained-graph.json').write_text(json.dumps({'passed':True,
        'repeated_unpack':True,'cross_stream_backward':True,'metrics':metrics},indent=2))


def test_retained_graph_cross_stream_and_small_cache_survive(tmp_path):
    evidence=Path(os.environ.get('DYCON_HOST_OFFLOAD_EVIDENCE',str(tmp_path))).resolve()
    evidence.mkdir(parents=True,exist_ok=True)
    physical=int(os.environ.get('DYCON_HOST_OFFLOAD_GPUS','4,5').split(',')[0])
    mp.spawn(_retained_graph_worker,args=(physical,str(evidence)),nprocs=1,join=True)


def _native_deferred_free_worker(rank, physical, directory):
    import gc
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical)
    from RL.allocator import configure_cuda_allocator
    from RL.config import CudaAllocator, HostOffload
    from RL.host_offload import host_allocator_stats
    from RL.memory import BoundaryOffload
    settings=HostOffload(enabled=True,allocator_budget_gib=0.125,cache_limit_gib=0)
    configure_cuda_allocator(CudaAllocator(),host_offload=settings)
    torch.cuda.set_device(0)
    model=torch.nn.Linear(16,16).cuda()
    offload=BoundaryOffload(model,64*2**20,host_settings=settings)
    source=torch.ones((1,64,16),device='cuda')
    baseline=host_allocator_stats()['host_allocated_bytes']
    hooks=offload.context()
    moved,lease=hooks.pack_hook(source)
    assert moved
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream): restored=hooks.unpack_hook((True,lease))
    offload.events['h2d'][-1][1].synchronize()
    # Our copy events are complete, but native free() will record an event after
    # this unrelated work. This distinguishes native pending frees from pool idle.
    with torch.cuda.stream(stream): torch.cuda._sleep(50_000_000)
    del lease
    gc.collect()
    offload.reclaim_completed()
    assert not offload.pool.blocks and offload.pool.native_release_pending
    offload.safe_boundary()
    metrics=offload.pool.metrics()
    assert not metrics['host_pool_native_release_pending']
    assert metrics['host_allocated_bytes'] <= baseline+8192
    torch.testing.assert_close(restored,source)
    offload.close()
    (Path(directory)/'native-deferred-free.json').write_text(json.dumps({'passed':True,
        'empty_pool_boundary_reclaimed':True,'metrics':metrics},indent=2))


def test_native_deferred_free_is_drained_even_when_pool_is_empty(tmp_path):
    evidence=Path(os.environ.get('DYCON_HOST_OFFLOAD_EVIDENCE',str(tmp_path))).resolve()
    evidence.mkdir(parents=True,exist_ok=True)
    physical=int(os.environ.get('DYCON_HOST_OFFLOAD_GPUS','4,5').split(',')[0])
    mp.spawn(_native_deferred_free_worker,args=(physical,str(evidence)),nprocs=1,join=True)
