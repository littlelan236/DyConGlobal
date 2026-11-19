"""Full-state numerical, storage lifetime and admission regression tests."""
import copy
import gc
import os
from pathlib import Path

import pytest
import torch

from RL.config import AdamMomentOffload, HostOffload, Memory, Config, memory_contract
from RL.optimizer_offload import CPUBacking, StreamedAdamW


def settings(**kw):
    return AdamMomentOffload(mode="memory_pressure", tile_pair_mib=1,
        gpu_scratch_budget_mib=4, pinned_staging_budget_mib=4,
        host_state_budget_gib=.1, node_state_budget_gib=.2, **kw)


def optimizer(parameters, **kw):
    return StreamedAdamW(parameters, lr=1e-3, settings=settings(**kw),
                        host_settings=HostOffload(enabled=True))


def collect(opt, p, key):
    parts = [v.detach().clone().cpu() for _, v in opt.iter_local_state_chunks(p, key)]
    return torch.cat(parts) if parts else torch.empty(0)


@pytest.mark.parametrize("placement", ["gpu", "partial", "cpu", "cycles"])
@pytest.mark.parametrize("schedule", ["constant", "linear"])
def test_fifty_updates_full_values_and_moments(placement, schedule):
    torch.set_num_threads(1)
    torch.manual_seed(312)
    p = [torch.nn.Parameter(torch.randn(131091)), torch.nn.Parameter(torch.randn(7)),
         torch.nn.Parameter(torch.empty(0)), torch.nn.Parameter(torch.randn(()))]
    q = [torch.nn.Parameter(v.detach().clone()) for v in p]
    a = optimizer([{"params": p[:1], "weight_decay": .13}, {"params": p[1:], "weight_decay": 0.}])
    b = torch.optim.AdamW([{"params": q[:1], "weight_decay": .13}, {"params": q[1:], "weight_decay": 0.}], lr=1e-3, foreach=False)
    factor = lambda step: min(1., (step+1)/5) * (1. if schedule == "constant" else (51-step)/51)
    sa, sb = (torch.optim.lr_scheduler.LambdaLR(v, factor) for v in (a, b))
    a.initialize(cpu_target=2**60 if placement == "cpu" else 0)
    if placement == "partial": a.evict(2**20)
    for step in range(50):
        if placement == "cycles" and step % 2 == 0:
            rng = torch.get_rng_state().clone()
            a.evict(2**60)
            assert torch.equal(rng, torch.get_rng_state())
        elif placement == "cycles":
            before = {id(x): collect(a,x,"exp_avg") for x in p}
            a.restore_gpu(2**60, minimum_residency=False)
            for x in p: assert torch.equal(before[id(x)],collect(a,x,"exp_avg"))
            assert a.cpu_bytes() == 0
        for i, (x, y) in enumerate(zip(p, q)):
            grad = torch.randn_like(x) if step % 3 else torch.zeros_like(x)
            x.grad, y.grad = grad.clone(), grad.clone()
        a.step(); b.step(); sa.step(); sb.step()
        for x, y in zip(p, q):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(collect(a,x,key),b.state[y][key].reshape(-1),atol=0,rtol=0)
            assert float(a.state[x]["step"]) == float(b.state[y]["step"]) == step+1
        a.validate()
    stats = a.metrics()
    assert stats["gpu_authoritative_bytes"] + stats["cpu_authoritative_bytes"] == stats["logical_state_bytes"]
    if placement == "cycles": assert stats["cpu_backing_released_bytes"] > 0
    a.close(); a.close()


def test_cpu_backing_alias_prevents_unmap_and_charging_release():
    backing = CPUBacking(100)
    alias = backing.tensor[3:]
    with pytest.raises(BufferError): backing.close()
    assert backing.mapping is not None
    alias.fill_(3)
    del alias
    gc.collect()
    backing.close()
    assert backing.mapping is None


def test_none_grad_and_tiny_state_resume(tmp_path):
    p = torch.nn.Parameter(torch.arange(20,dtype=torch.float32))
    a = optimizer([p]); a.initialize(cpu_target=1000)
    a.step(); assert int(a.state[p]["step"]) == 0
    for _ in range(3): p.grad=torch.ones_like(p); a.step()
    torch.save(a.state_dict(),tmp_path/"tiny.pt")
    q = torch.nn.Parameter(p.detach().clone())
    b = optimizer([q]); b.load_state_dict(torch.load(tmp_path/"tiny.pt",weights_only=True))
    for _ in range(3):
        p.grad=torch.ones_like(p); q.grad=p.grad.clone(); a.step(); b.step()
    assert torch.equal(p,q)
    for key in ("exp_avg","exp_avg_sq"): assert torch.equal(collect(a,p,key),collect(b,q,key))
    with pytest.raises(ValueError,match="transition"):
        optimizer([q]).load_state_dict({"state":{}})
    a.close(); b.close()


def test_streaming_serialization_different_tile_layout(tmp_path):
    from RL.full_checkpoint import write_moment, read_moment
    p = torch.nn.Parameter(torch.randn(150011))
    a = optimizer([p]); a.initialize(cpu_target=2**60)
    p.grad=torch.randn_like(p); a.step()
    q=torch.nn.Parameter(p.detach().clone()); b=optimizer([q]); b.tile_elements=17003; b.initialize()
    for key in ("exp_avg","exp_avg_sq"):
        desc=write_moment(tmp_path,a,p,key,[0 if key=="exp_avg" else 100])
        read_moment(tmp_path,desc,b,q,key)
        assert torch.equal(collect(a,p,key),collect(b,q,key))
        desc["parts"][0]["offset"]=1
        with pytest.raises(ValueError,match="gaps"):
            read_moment(tmp_path,desc,b,q,key)
    a.close();b.close()


def test_twenty_short_long_short_hysteresis_cycles():
    p=torch.nn.Parameter(torch.zeros(300000));a=optimizer([p],restore_stable_windows=3,min_residency_windows=3)
    a.initialize()
    for cycle in range(20):
        a.prepare(extra_bytes=2**20, new_window=True, sample={"driver_free_bytes":12*2**30,"allocated_bytes":0,"reserved_bytes":0})
        assert a.cpu_bytes()>0
        for _ in range(3):
            a.prepare(new_window=True,allow_restore=True,sample={"driver_free_bytes":30*2**30,"allocated_bytes":0,"reserved_bytes":0})
        assert a.cpu_bytes()==0
    assert a.metrics()["restores"] >=20
    a.close()


def test_partial_update_failure_is_terminal(monkeypatch):
    import RL.optimizer_offload as module
    p=torch.nn.Parameter(torch.zeros(150000));a=optimizer([p]);a.initialize();p.grad=torch.ones_like(p)
    original=module.adamw;calls=[]
    def fault(*args,**kw):
        calls.append(1)
        if len(calls)==2: raise RuntimeError("injected")
        return original(*args,**kw)
    monkeypatch.setattr(module,"adamw",fault)
    with pytest.raises(RuntimeError,match="injected"):a.step()
    assert int(a.state[p]["step"])==0 and a.failed
    with pytest.raises(RuntimeError,match="Unusable"):a.step()
    a.close()


def test_schema_default_legacy_and_rejections():
    assert "adam_moment_offload" not in memory_contract(Memory())
    with pytest.raises(ValueError):settings(high_watermark_gib=10)
    with pytest.raises(ValueError):AdamMomentOffload(tile_pair_mib=64,gpu_scratch_budget_mib=64)
    with pytest.raises(ValueError):Memory(adam_moment_offload=settings())
    from tests.test_full_policy import full_values
    raw=full_values();raw["validation"].update(skip_checkpoint_export=True,stop_after_update=None)
    with pytest.raises(ValueError,match="explicit stop"):Config.model_validate(raw)


def _admission_worker(path,ready,go,finish,queue):
    from RL.host_admission import NodeAdmission
    a=NodeAdmission(path,settings(),HostOffload(enabled=True,reserve_gib=.001,reserve_fraction=.001),attempt="test",rank=0)
    ready.put(1);go.wait()
    try:
        token=a.reserve(90*2**20,"adam")
    except RuntimeError:
        queue.put("rejected");return
    queue.put("accepted")
    finish.wait(20)
    a.release(token);a.close()


def test_node_concurrent_pending_reservations(tmp_path):
    import multiprocessing as mp
    context=mp.get_context("spawn")
    ready,queue=context.Queue(),context.Queue()
    go,finish=context.Event(),context.Event()
    workers=[context.Process(target=_admission_worker,args=(str(tmp_path),ready,go,finish,queue)) for _ in range(4)]
    for worker in workers:worker.start()
    try:
        for _ in workers:ready.get(timeout=25)
        go.set()
        results=[queue.get(timeout=25) for _ in workers]
        assert results.count("accepted")==2 and results.count("rejected")==2
    finally:
        finish.set()
        for worker in workers:worker.join(25)
        for worker in workers:
            if worker.is_alive():worker.kill();worker.join()
    assert all(worker.exitcode==0 for worker in workers)


def test_node_admission_pending_charged_release_and_conflict(tmp_path,monkeypatch):
    import RL.host_admission as module
    monkeypatch.setattr(module,"check_host_capacity",lambda s,**kw:kw)
    a=module.NodeAdmission(tmp_path,settings(),HostOffload(),attempt="one",rank=0)
    b=module.NodeAdmission(tmp_path,settings(),HostOffload(),attempt="two",rank=1)
    x=a.reserve(90*2**20,"adam");a.commit(x)
    different=settings();different.node_state_budget_gib=.3
    conflicting=module.NodeAdmission(tmp_path,different,HostOffload(),attempt="conflicting",rank=2)
    with pytest.raises(RuntimeError,match="Conflicting"):conflicting.reserve(1,"pinned")
    conflicting.close()
    y=b.reserve(90*2**20,"adam");b.commit(y)
    with pytest.raises(RuntimeError,match="budget"):b.reserve(40*2**20,"adam")
    assert a.metrics()["node_pending_bytes"]==0
    a.release(x);b.release(y);a.close();b.close()
    entries={"dead":{"pid":999999999,"start":"0","owner":"dead","kind":"adam","size":100,"status":"pending","node_limit":1}}
    (tmp_path/"ledger.json").write_text(__import__('json').dumps(entries))
    c=module.NodeAdmission(tmp_path,settings(),HostOffload(),attempt="c",rank=0)
    assert c.metrics()["node_adam_bytes"]==0
    c.close()


def test_real_mapping_rss_returns_after_restore():
    from RL.resources import host_memory
    before=host_memory()["VmRSS"]
    backing=CPUBacking(16*2**20);high=host_memory()["VmRSS"]
    backing.close();low=host_memory()["VmRSS"]
    assert high-before>48*2**20 and high-low>48*2**20


def test_explicit_optimizer_migration_binds_hashes_and_math():
    from RL.config import schedule_contract
    from RL.optimizer_migration import validate,schedule_matches
    from core.io import hash_json
    from tests.test_full_policy import full_values
    raw=full_values();cfg=Config.model_validate(raw)
    source={"schedule":schedule_contract(cfg)}
    raw["training"]["memory"].update(host_offload={"enabled":True,"allocator_budget_gib":2.})
    # The source must have the same activation and allocator policy; residency
    # migration cannot hide another unrelated configuration change.
    cfg=Config.model_validate(raw);source={"schedule":schedule_contract(cfg)}
    raw["run"]["start"]={"mode":"resume","checkpoint":"/tiny/source"}
    raw["training"]["memory"]["adam_moment_offload"]={"mode":"memory_pressure"}
    cfg=Config.model_validate(raw)
    assert not schedule_matches(cfg,source)
    raw["run"]["start"]["optimizer_backend_migration"]={"source_schedule_sha256":hash_json(source["schedule"]),
        "target_schedule_sha256":hash_json(schedule_contract(cfg)),"reason":"explicit unit migration"}
    cfg=Config.model_validate(raw)
    assert validate(cfg,source)
    changed=cfg.model_copy(deep=True);changed.training.learning_rate*=2
    with pytest.raises(ValueError,match="hash"):validate(changed,source)
    wrong=copy.deepcopy(source);wrong["optimizer_backend"]={"backend":"other"}
    with pytest.raises(ValueError,match="standard"):validate(cfg,wrong)


def test_host_admission_does_not_double_subtract_charged_bytes(tmp_path,monkeypatch):
    import RL.host_admission as module
    calls=[]
    monkeypatch.setattr(module,"check_host_capacity",lambda s,**kw:calls.append(kw["requested_bytes"]))
    a=module.NodeAdmission(tmp_path,settings(),HostOffload(),attempt="test",rank=0)
    one=a.reserve(1024,"adam");a.commit(one)
    two=a.reserve(512,"pinned")
    three=a.reserve(256,"pinned")
    assert calls==[1024,512,768]
    for token in (one,two,three):a.release(token)
    a.close()


def test_host_cgroup_remaining_checked_before_allocation():
    from RL.resources import check_host_capacity
    host={"MemTotal":100*2**30,"MemAvailable":80*2**30,"cgroup_remaining_bytes":2*2**30}
    with pytest.raises(RuntimeError,match="reserve"):
        check_host_capacity(HostOffload(reserve_gib=1.,reserve_fraction=.01),requested_bytes=2*2**30,host=host)


def test_restore_preserves_alias_and_charge_until_last_reader_leaves():
    p=torch.nn.Parameter(torch.randn(25));a=optimizer([p]);a.initialize(cpu_target=1000)
    alias=next(a.iter_local_state_chunks(p,"exp_avg"))[1]
    a.restore_gpu(1000,minimum_residency=False)
    assert a.metrics()["cpu_authoritative_bytes"]==0 and a.cpu_bytes()>0
    assert a.metrics()["pending_backings"]==1 and torch.equal(alias,torch.zeros_like(alias))
    del alias;gc.collect();a.poll()
    assert a.cpu_bytes()==0
    a.close()


def test_mapping_count_is_bounded_before_allocation(monkeypatch):
    monkeypatch.setattr(CPUBacking,"max_mappings",CPUBacking.live_mappings)
    with pytest.raises(RuntimeError,match="mapping-count"):CPUBacking(25)


def test_driver_granularity_rechecks_and_evicts_more(monkeypatch):
    p=torch.nn.Parameter(torch.zeros(600000));a=optimizer([p]);a.initialize()
    values=iter([12*2**30-2**20,12*2**30-1,12*2**30+1])
    monkeypatch.setattr(a,"sample",lambda:{"driver_free_bytes":next(values),"allocated_bytes":0,"reserved_bytes":0})
    a.prepare()
    assert a.last_decision["after"]["driver_free_bytes"]>=12*2**30
    assert a.counters["evictions"]>=4
    a.close()


def test_transfer_fault_keeps_authoritative_source(monkeypatch):
    p=torch.nn.Parameter(torch.randn(25));a=optimizer([p]);a.initialize()
    initial=collect(a,p,"exp_avg")
    def fail(*args):raise RuntimeError("injected transfer")
    monkeypatch.setattr(a,"_copy",fail)
    with pytest.raises(RuntimeError,match="injected transfer"):a.evict(1000)
    assert a.cpu_bytes()==0 and a.metrics()["gpu_authoritative_bytes"]==p.numel()*8
    assert torch.equal(initial,collect(a,p,"exp_avg"))
    a.close()


def test_adam_staging_dirties_shared_pool_after_clean_activation_boundary(monkeypatch):
    from RL.memory import BoundaryOffload
    calls=[]
    class Pool:
        blocks=[]
        native_release_pending=True
        _needs_reclaim=False
        pending_admission_releases=["pending"]
        def safe_boundary(self):
            calls.append("reclaim")
            self.native_release_pending=False
            self.pending_admission_releases=[]
            return {"reclaimed":True}
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:calls.append("sync"))
    offloader=BoundaryOffload(torch.nn.Linear(2,2),1024**2,enabled=True)
    offloader.pool=Pool();offloader._boundary_clean=True
    assert offloader.safe_boundary()=={"reclaimed":True}
    assert calls==["sync","reclaim"]
    assert offloader.safe_boundary() is None


def test_no_export_is_operational_and_preserves_schedule_identity():
    from RL.config import schedule_contract
    from tests.test_full_policy import full_values
    raw=full_values();expected=schedule_contract(Config.model_validate(raw))
    raw["validation"].update(stop_after_update=1,skip_checkpoint_export=True)
    assert schedule_contract(Config.model_validate(raw))==expected
    raw["validation"]["capacity_microsteps"]=1
    with pytest.raises(ValueError,match="Capacity microsteps"):Config.model_validate(raw)
    raw["training"]["memory"].update(adam_moment_offload={"mode":"memory_pressure"},
        host_offload={"enabled":True,"allocator_budget_gib":2.})
    cfg=Config.model_validate(raw);normal=cfg.model_copy(deep=True)
    normal.validation.capacity_microsteps=None
    assert schedule_contract(cfg)==schedule_contract(normal)
