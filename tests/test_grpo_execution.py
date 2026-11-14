"""8x16 frozen scoring, two-pass execution, selected IO and resume invariants."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from core.io import hash_json, read_json, sha256
from RL.config import Config, schedule_contract
from RL.grpo.execution import (build, partition, cursor, validate, validate_state, parent_record,
    from_config, select_raw, PreparedExecution, validate_window)
from RL.grpo.data import tensor_groups, prepare
from RL.grpo.objective import grpo_objective
from RL.grpo.prepared import materialize
from tests.test_grpo_prepared import ROOT, BASE, config, dataset, train


def cfg32(tmp, n=9, streaming=False):
    dataset(tmp, "a", range(n), branch_count=32)
    cfg = config(tmp, updates=2*((n+7)//8))
    cfg.runtime.gpu = None
    cfg.runtime.gpus = list(range(8))
    cfg.runtime.data_preparation_async = streaming
    cfg.grpo.group_batch_size = 8
    cfg.grpo.parallel_branch_chunk_size = 16
    cfg.grpo.parallel_branch_schedule = "distinct_parent_two_pass_v1"
    cfg.grpo.branch_partition_seed = 42
    cfg.grpo.prefix_reuse.mode = "boundary_grad"
    cfg.training.checkpoint_policy = "layer"
    cfg.training.checkpoint_every_updates = 1
    cfg.training.memory.profile_first_update = False
    cfg.preflight.verify_identity = False
    cfg.preflight.audit_native_tokens = False
    return Config.model_validate(cfg.model_dump())


def parents(n):
    return [{"parent_group_id": f"p{i}", "branch_ids": [f"p{i}-b{b}" for b in range(32)],
             "prompt_length": 2+i%17, "completion_counts": [1+(i+b)%7 for b in range(32)],
             "frozen_metadata_sha256": hash_json(i)} for i in range(n)]


def plan_for(n):
    return build(parents(n), partition_seed=42, shuffle_seed=19, shuffle_algorithm="prefix_length_pool128_v1")


@pytest.mark.parametrize("n", [8,9,12,13,16,6773])
def test_two_pass_distinct_parent_windows_exact_epoch_coverage(n):
    plan = plan_for(n)
    assert len(plan["windows"]) == 2*((n+7)//8)
    assert len(plan["chunks"]) == 2*n
    coverage = {}
    for w in plan["windows"]:
        selected = plan["chunks"][w["start"]:w["end"]]
        assert len(selected) == (n%8 or 8) if w["end"] in (n,2*n) else len(selected) == 8
        assert len({c["parent_group_id"] for c in selected}) == len(selected)
        assert all(c["half_index"] == w["round_index"] for c in selected)
        for c in selected:
            coverage.setdefault(c["parent_group_id"], []).extend(c["native_branch_ordinals"])
    assert all(sorted(indices) == list(range(32)) for indices in coverage.values())
    state = {"global_step": len(plan["windows"]), **cursor(plan,2*n)}
    validate_state(plan,state)
    assert state["completed_parent_groups"] == state["distinct_parent_groups_seen"] == n
    first = cursor(plan,n)
    assert first["group_cursor"] == first["completed_parent_groups"] == 0
    assert first["distinct_parent_groups_seen"] == first["consumed_execution_chunks"] == n
    assert cursor(plan,0)["consumed_branches"] == 0


def test_partition_cross_process_independent_of_traversal_and_hash_seed():
    expected = partition("稳定父组", 42)
    code = "import json; from RL.grpo.execution import partition; print(json.dumps(partition('稳定父组',42)))"
    for seed in ["1","999"]:
        out = subprocess.check_output([sys.executable,"-c",code],cwd=ROOT,
                                      env={**os.environ,"PYTHONHASHSEED":seed},text=True,timeout=20)
        assert json.loads(out) == expected
    assert partition("稳定父组",43) != expected
    p = parents(13)
    kwargs = dict(partition_seed=42,shuffle_seed=19,shuffle_algorithm="prefix_length_pool128_v1")
    assert build(p,**kwargs) == build(reversed(p),**kwargs)


@pytest.mark.parametrize("fault", ["half", "branch", "tail", "seed", "world", "hash"])
def test_plan_tampering_fails(fault):
    p = plan_for(9)
    if fault == "half": p["chunks"][0]["half_index"] = 1
    if fault == "branch": p["chunks"][0]["branch_ids"][0] = "other"
    if fault == "tail": p["windows"][1]["end"] += 1
    if fault == "seed": p["partition_seed"] += 1
    if fault == "world": p["world_size"] = 4
    if fault == "hash": p["sha256"] = "0"*64
    with pytest.raises(ValueError,match="plan"):
        validate(p)


@pytest.mark.parametrize("field", ["consumed_execution_chunks", "consumed_half_ids_sha256", "completed_parent_groups",
    "distinct_parent_groups_seen", "group_cursor", "consumed_branches", "consumed_tokens", "execution_round", "global_step"])
def test_partial_parent_checkpoint_cursor_is_strict(field):
    p = plan_for(13)
    for count in [8,13,21,26]:
        s = {"global_step":cursor(p,count)["execution_window_cursor"],**cursor(p,count)}
        validate_state(p,s)
        s[field] = "bad" if isinstance(s[field],str) else s[field]+1
        with pytest.raises(ValueError,match="8x16"):
            validate_state(p,s)


def test_scoring_masks_reference_preserved_and_half_advantage_is_not_recentered(tmp_path):
    cfg=cfg32(tmp_path)
    raw,proof=prepare(cfg)
    p=proof["execution_plan"]
    selected={c["half_index"]:c for c in p["chunks"] if c["parent_group_id"]==raw[0]["group"]["group_id"]}
    full=next(tensor_groups([raw[0]]))
    halves=[next(tensor_groups([select_raw(raw[0],selected[h])])) for h in range(2)]
    for half in halves:
        for i,bid in enumerate(half.branch_ids):
            j=full.branch_ids.index(bid)
            assert half.advantages[i]==full.advantages[j]
            assert half.historical_scores[i]==full.historical_scores[j]
            assert half.completion_token_ids[i]==full.completion_token_ids[j]
            assert torch.equal(half.loss_mask[i],full.loss_mask[j])
    assert any(abs(sum(h.advantages))>1e-8 for h in halves)
    torch.manual_seed(53)
    theta=torch.randn(32,7,requires_grad=True)*.05-2
    theta=theta.detach().requires_grad_()
    ref=theta.detach()-.03
    mask=torch.arange(7)[None,:] < torch.tensor([1+i%7 for i in range(32)])[:,None]
    full_result=grpo_objective(theta,ref,full.advantages,mask,kl_beta=.17)
    losses=[]
    for h in range(2):
        idx=selected[h]["native_branch_ordinals"]
        losses.append(grpo_objective(theta[idx],ref[idx],[full.advantages[i] for i in idx],mask[idx],kl_beta=.17).loss)
    average=sum(losses)/2
    torch.testing.assert_close(average,full_result.loss)
    torch.testing.assert_close(torch.autograd.grad(average,theta,retain_graph=True)[0],
                               torch.autograd.grad(full_result.loss,theta)[0])


@pytest.mark.parametrize("branches",[31,33])
def test_mandatory_branch_count_validation_precedes_model_and_ignores_optional_audit(tmp_path,branches):
    cfg=cfg32(tmp_path)
    # New immutable source fixture with the unsupported shape.
    dataset(tmp_path,"wrong",range(9),branch_count=branches)
    cfg.algorithm.dataset_id="wrong"
    with pytest.raises(ValueError,match="32"):
        prepare(cfg)


@pytest.mark.parametrize("fault",["world","batch","mode","schedule","size","eight_whole"])
def test_invalid_configuration_and_legacy_contract(tmp_path,fault):
    cfg=cfg32(tmp_path)
    data=cfg.model_dump()
    if fault=="world": data["runtime"]["gpus"]=[0,1,2,3]
    if fault=="batch": data["grpo"]["group_batch_size"]=4
    if fault=="mode": data["grpo"]["prefix_reuse"]["mode"]="disabled"
    if fault=="schedule": data["grpo"]["parallel_branch_schedule"]=None
    if fault=="size": data["grpo"]["parallel_branch_chunk_size"]=8
    if fault=="eight_whole":
        data["grpo"].update(parallel_branch_chunk_size=None,parallel_branch_schedule=None)
    with pytest.raises(ValueError): Config.model_validate(data)
    old=config(tmp_path).model_copy(deep=True)
    schedule=schedule_contract(old)
    assert not {"parallel_branch_chunk_size","parallel_branch_schedule","branch_partition_seed"}.intersection(schedule["grpo"])


def test_prepared_streaming_same_plan_selected_io_and_frozen_source(tmp_path,monkeypatch):
    from RL.grpo.streaming import create_plan,StreamingGroups
    from RL.dataset.loader import native_tokens
    cfg=cfg32(tmp_path)
    reader=materialize(cfg,ROOT,"indexed")
    stream_plan=create_plan(cfg)
    assert reader.execution_plan==stream_plan["proof"]["execution_plan"]
    from RL.dataset.artifacts import directory
    source=directory(cfg.algorithm.dataset_root,"rl-dataset","a")
    before={str(p):sha256(p) for p in source.rglob("*") if p.is_file()}
    for rank in [0,7]:
        prepared=PreparedExecution(reader,rank=rank,world_size=8)
        stream=StreamingGroups(cfg,stream_plan,rank=rank,world_size=8)
        reads=[]
        import RL.dataset.loader as loader
        def counted(row,**kwargs):
            reads.append(row["branch_sample_id"])
            return native_tokens(row,**kwargs)
        monkeypatch.setattr(loader,"native_tokens",counted)
        class Source:
            def values(self,row,**kwargs):
                import numpy as np
                return np.full(row["completion_count"],-2,dtype=np.float32)
            def verify_unchanged(self): pass
        stream.source=Source()
        a=iter(prepared.windows(0)); b=iter(stream.windows(0))
        for step in range(4):
            wa,wb=next(a),next(b)
            validate_window(reader.execution_plan,step,wa)
            assert [g.branch_ids for g in wa]==[g.branch_ids for g in wb]
            local=[g for i,g in enumerate(wb) if i%8==rank]
            assert set(stream.reference.values)=={g.group_id for g in local}
            for g in local:
                assert set(stream.reference.values[g.group_id])==set(g.branch_ids)
            stream.commit_window(wb)
        b.close();stream.close()
        expected=[bid for w in reader.execution_plan["windows"]
                  for c in reader.execution_plan["chunks"][w["start"]+rank:w["end"]:8] for bid in c["branch_ids"]]
        assert reads==expected
    assert all(sha256(p)==h for p,h in before.items())


@pytest.mark.parametrize("streaming",[False,True])
def test_real_tiny_qwen_checkpoint_resume_each_window_boundary(tmp_path,streaming):
    from tests.test_grpo_streaming_resume import compare,checkpoint
    from tests.test_grpo_external_reference import make_source,select
    from RL.checkpoints import verify_checkpoint
    cfg=cfg32(tmp_path,streaming=streaming)
    if streaming:
        def native_requests(manifest, rows, path):
            for row in rows:
                row["native_request_id"] = row["branch_sample_id"]
        make_source(tmp_path/"probabilities",cfg,mutate=native_requests)
        select(cfg,tmp_path/"probabilities")
    full,_,_,_=train(cfg,"full")
    previous=None
    for step in range(1,5):
        current,_,result,_=train(cfg,f"stage-{step}",parent=previous,
                               mode="resume" if previous else "fresh",stop=step if step<4 else None)
        compare(full,current,[step])
        previous=checkpoint(current,step)
        verify_checkpoint(previous)
        s=result["state"]
        assert s["consumed_execution_chunks"]==[8,9,17,18][step-1]
        assert s["completed_parent_groups"]==[0,0,8,9][step-1]
    history=read_json(current/"grpo-report.json")["updates"]
    assert [r["branch_count"] for r in history]==[128,16,128,16]
    assert all(len(r["prefix_reuse"])==r["execution_chunk_count"] for r in history)
    assert [r["branch_loss_weight"] for r in history]==[1/128,1/16,1/128,1/16]
    assert all(len({(p["parent_group_id"],p["half_index"]) for p in r["prefix_reuse"]})==len(r["prefix_reuse"]) for r in history)


def test_public_service_eight_gpu_lock_scope_artifact_status_and_tensorboard(tmp_path,monkeypatch):
    from contextlib import contextmanager
    import yaml
    from RL.service import run,_train_grpo
    from RL.artifacts import verify
    from RL.observability import snapshot
    from RL.observability_tensorboard import scalar_values
    from core.io import write_json
    from tests.lora_helpers import model_and_targets
    cfg=cfg32(tmp_path)
    cfg.run.attempt_id="public-8x16-tiny"
    cfg.validation.stop_after_update=1
    file=tmp_path/"training.yaml"
    file.write_text(yaml.safe_dump(cfg.model_dump()))
    base={"files":{"tiny_fixture":True}}
    base["fingerprint"]=hash_json(base["files"])
    from RL.dataset.artifacts import directory,seal
    native=directory(cfg.algorithm.dataset_root,"rl-dataset","a")
    meta=read_json(native/"dataset.json")
    meta["cohort"]["model"]=base
    write_json(native/"dataset.json",meta)
    seal(native,"rl-dataset","a")
    # Dataset identity uses a tiny fixture; service model identity is injected.
    monkeypatch.setattr("RL.service._preflight",lambda *args:base)
    active,acquired=[],[]
    @contextmanager
    def lock(path,**kwargs):
        active.append(path);acquired.append(path)
        try: yield
        finally: active.remove(path)
    monkeypatch.setattr("RL.service.file_lock",lock)
    def launch(config,root,directory,base,parent,attempt):
        assert len([p for p in active if p.parent.name=="lora-gpu"])==8
        model,targets=model_and_targets()
        write_json(directory/"environment.json",{"tiny_fixture":True})
        write_json(directory/"trainable-parameters.json",{"targets":targets,"target_count":len(targets),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)})
        return _train_grpo(config,root,directory,base,parent,attempt,model,targets,optimizer_settings=config.training)
    monkeypatch.setattr("RL.parallel.launch",launch)
    result=run({"project_root":str(ROOT)},file)
    assert result["status"]=="paused" and not active
    assert sorted(int(p.name.split('.')[0]) for p in acquired if p.parent.name=="lora-gpu")==list(range(8))
    assert verify({"project_root":str(ROOT)},"lora-attempt",cfg.run.attempt_id,storage_root=cfg.runtime.output_root)["verified"]
    view=snapshot({"project_root":str(ROOT)},cfg.run.attempt_id,storage_root=cfg.runtime.output_root)
    assert view["consumed_execution_chunks"]==8
    assert view["completed_parent_groups"]==view["committed_consumed_groups"]==0
    assert view["distinct_parent_groups_seen"]==8
    assert view["planned_execution_chunks"]==18
    metric=read_json(Path(result["path"])/"grpo-report.json")["updates"][0]
    projected=scalar_values(metric)
    assert projected["progress/consumed_execution_chunks"]==8
    assert projected["progress/completed_parent_groups"]==0
    assert projected["progress/planned_parent_groups"]==9


def test_eight_worker_supervisor_reaps_processes_restores_signals_and_releases_locks(tmp_path,monkeypatch):
    from contextlib import contextmanager
    import signal
    import yaml
    from RL.service import run
    cfg=cfg32(tmp_path)
    cfg.run.attempt_id="supervisor-8x16-failure"
    file=tmp_path/"training.yaml"
    file.write_text(yaml.safe_dump(cfg.model_dump()))
    monkeypatch.setattr("RL.service._preflight",lambda *args:BASE)
    locks=[]
    @contextmanager
    def lock(path,**kwargs):
        locks.append(path)
        try: yield
        finally: locks.remove(path)
    monkeypatch.setattr("RL.service.file_lock",lock)
    class Process:
        def __init__(self): self.alive=True;self.joined=False
        def is_alive(self): return self.alive
        def terminate(self): self.alive=False
        def join(self,timeout): self.joined=True
        def kill(self): self.alive=False
    processes=[Process() for _ in range(8)]
    class Context:
        def join(self,timeout): raise RuntimeError("injected worker failure")
    context=Context();context.processes=processes
    def start(worker,**kwargs):
        assert kwargs["nprocs"]==8 and kwargs["start_method"]=="spawn"
        assert len([p for p in locks if p.parent.name=="lora-gpu"])==8
        return context
    monkeypatch.setattr("torch.multiprocessing.start_processes",start)
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    before=set((ROOT/"temp").glob("rl-data-parallel-*"))
    with pytest.raises(RuntimeError,match="injected worker failure"):
        run({"project_root":str(ROOT)},file)
    assert not locks and all(not p.alive and p.joined for p in processes)
    assert set((ROOT/"temp").glob("rl-data-parallel-*"))==before
    assert {s:signal.getsignal(s) for s in handlers}==handlers


@pytest.mark.parametrize('n',[7,9])
def test_preflight_is_torch_free_and_structure_cannot_be_skipped(tmp_path,n):
    import yaml
    cfg=cfg32(tmp_path,n=n,streaming=True)
    cfg.run.mode="dry-run"
    cfg.run.attempt_id=f"no-torch-8x16-{n}"
    file=tmp_path/"dry.yaml";file.write_text(yaml.safe_dump(cfg.model_dump()))
    code='''
import importlib.abc,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=="torch" or fullname.startswith("torch."): raise AssertionError("preflight imported Torch")
sys.meta_path.insert(0,NoTorch())
from RL.service import run
import RL.service as service
service.base_identity=lambda *args:{"fingerprint":"a"*64}
service.memory_plan=lambda *args:{}
if sys.argv[2]=="7":
 try: run({"project_root":sys.argv[3]},sys.argv[1])
 except ValueError as exc: assert "at least 8" in str(exc)
 else: raise AssertionError("optional-audit flags bypassed required parent validation")
else:
 result=run({"project_root":sys.argv[3]},sys.argv[1])
 assert result["dataset"]["planned_updates"]==4
 assert result["dataset"]["planned_execution_chunks"]==18
assert "torch" not in sys.modules
'''
    completed=subprocess.run([sys.executable,"-c",code,str(file),str(n),str(ROOT)],cwd=ROOT,
                             text=True,capture_output=True,timeout=40)
    assert completed.returncode==0,completed.stderr[-5000:]
