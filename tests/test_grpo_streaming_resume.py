"""Native streaming resume: independent processes, legacy evidence and failures."""
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from core.io import hash_json, read_json, sha256, write_json
from RL.artifacts import commit_receipt
from RL.checkpoints import verify_checkpoint
from RL.config import Config, attempt_directory
from RL.grpo.stream_resume import resolve_checkpoint, execution_mode
from tests.test_grpo_prepared import ROOT, BASE, dataset, config
from tests.test_grpo_external_reference import make_source, select
from tests.test_grpo_lifecycle import assert_tree_close


def fixture(tmp, n=7):
    dataset(tmp, "a", range(n))
    cfg = config(tmp, updates=(n+1)//2)
    cfg.runtime.data_preparation_async = True
    cfg.training.checkpoint_every_updates = 1
    cfg.preflight.verify_identity = False
    cfg.preflight.audit_native_tokens = False
    def native_rows(manifest, rows, path):
        for row in rows:
            row["native_request_id"] = row["branch_sample_id"]
    make_source(tmp/"probabilities", cfg, mutate=native_rows)
    select(cfg, tmp/"probabilities")
    return cfg


def worker(cfg, *, rank=None, rendezvous=None):
    """Exercise service internals with a tiny real model; production uses CLI."""
    from RL.service import _source, _train_grpo
    from tests.lora_helpers import model_and_targets
    from RL.grpo.backend import GroupObjective
    from RL.training import capture_rng
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    owner = _source(cfg)
    model, targets = model_and_targets(settings=cfg.lora, adapter=owner["adapter"] if owner else None)
    # Every update genuinely consumes three RNG families and affects gradients.
    original = GroupObjective.iter_results
    def stochastic(self, model, group):
        if group.group_id == os.environ.get('DYCON_STREAM_FAILURE_GROUP'):
            write_json(path/'test-injected-failure.json',{'group_id':group.group_id,'error':'mid-window OOM'})
            raise torch.OutOfMemoryError('injected mid-window OOM')
        for result in original(self, model, group):
            factor = .8 + .1*(random.random()+float(np.random.rand())+float(torch.rand(())))
            result.loss = result.loss * factor
            yield result
    GroupObjective.iter_results = stochastic
    path = attempt_directory(cfg)
    if rank is not None:
        import torch.distributed as dist
        import RL.parallel as parallel
        from datetime import timedelta
        dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
                                world_size=2, timeout=timedelta(seconds=60))
        parallel._context = parallel.GroupParallel(rank, 2)
        parallel._context.synchronize_model(model)
        if rank:
            path = path/"distributed"/f"rank-{rank}"
    path.mkdir(parents=True, exist_ok=True)
    attempt = {"attempt_id":cfg.run.attempt_id,"purpose":"engineering_smoke",
               "start_mode":cfg.run.start.mode}
    # Record payload reads for the consumed-prefix/laziness assertion.
    from RL.dataset import loader, probability
    reads = {"native":[], "probability":[]}
    native, values = loader.native_tokens, probability.ProbabilityStream.values
    def counted(row, **kw):
        reads["native"].append(row["group_id"])
        return native(row, **kw)
    def counted_values(self, row, **kw):
        reads["probability"].append(row["group_id"])
        return values(self, row, **kw)
    loader.native_tokens, probability.ProbabilityStream.values = counted, counted_values
    try:
        result = _train_grpo(cfg, ROOT, path, BASE, owner, attempt, model, targets,
                             optimizer_settings=cfg.training)
        write_json(path/"test-payload-reads.json", reads)
        torch.save(capture_rng(), path/"test-final-rng.pt")
        return result
    finally:
        if rank is not None:
            parallel._context = None
            dist.destroy_process_group()


def run(cfg, name, *, parent=None, stop=None, parallel=False, fail_group=None):
    cfg = cfg.model_copy(deep=True)
    cfg.run.attempt_id = name
    cfg.run.start.mode = "resume" if parent else "fresh"
    cfg.run.start.checkpoint = str(parent) if parent else None
    cfg.validation.stop_after_update = stop
    if parallel:
        cfg.runtime.gpu = None
        cfg.runtime.gpus = [0,1]
    file = Path(cfg.runtime.output_root).parent/(name+".yaml")
    file.write_text(yaml.safe_dump(cfg.model_dump()))
    completed = subprocess.run([sys.executable,"-m","tests.test_grpo_streaming_resume",str(file),"parallel" if parallel else "serial"],
        cwd=ROOT, text=True, capture_output=True, timeout=150,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1",
             'DYCON_STREAM_FAILURE_GROUP':fail_group or ''})
    if fail_group:
        assert completed.returncode != 0, completed.stderr[-7000:]
        markers=list(attempt_directory(cfg).rglob('test-injected-failure.json'))
        assert markers and any(read_json(p)['group_id']==fail_group for p in markers), completed.stderr[-7000:]
    else:
        assert completed.returncode == 0, completed.stderr[-7000:]
    return attempt_directory(cfg)


def checkpoint(path, step):
    return path/f"checkpoints/step-{step:06d}"


def compare(left, right, steps, *, parallel=False):
    for step in steps:
        a,b = checkpoint(left,step),checkpoint(right,step)
        assert read_json(a/"state.json")["state"] == read_json(b/"state.json")["state"]
        assert (a/"adapter/adapter_model.safetensors").read_bytes() == (b/"adapter/adapter_model.safetensors").read_bytes()
        pa,pb = [torch.load(p/"training-state.pt",weights_only=True) for p in (a,b)]
        for key in ("optimizer","scheduler","rng", *(("rank_rng",) if parallel else ())):
            assert_tree_close(pa[key],pb[key])
    a,b = [read_json(p/"grpo-report.json")["updates"] for p in (left,right)]
    for x,y in zip(a,b):
        for key in ("loss","policy_loss","ratio_mean","gradient_norm","learning_rate","consumed_tokens"):
            assert x[key] == pytest.approx(y[key], abs=1e-7, rel=1e-6)


@pytest.mark.parametrize("parallel", [False, True])
def test_independent_process_resume_twice_matches_continuous(tmp_path, parallel):
    cfg=fixture(tmp_path)
    full=run(cfg,"full",parallel=parallel)
    paused=run(cfg,"pause",stop=1,parallel=parallel)
    cp=checkpoint(paused,1)
    before={str(p):sha256(p) for p in paused.rglob('*') if p.is_file()}
    resumed=run(cfg,"resume",parent=cp,stop=2,parallel=parallel)
    compare(full,resumed,[2],parallel=parallel)
    # Keep only the independent checkpoint and remove its original attempt.
    detached=tmp_path/"detached"/"checkpoints"/"step-000002"
    shutil.copytree(checkpoint(resumed,2),detached)
    shutil.rmtree(resumed)
    final=run(cfg,"resume-again",parent=detached,parallel=parallel)
    compare(full,final,[3,4],parallel=parallel)
    verify_checkpoint(checkpoint(final,4))
    assert all(sha256(p)==h for p,h in before.items())
    consumed=set(read_json(cp/"grpo/stream-plan.json")["proof"]["eligible_group_ids"][:2])
    reads=read_json(final/"test-payload-reads.json")
    assert not consumed.intersection(reads["native"]+reads["probability"])
    assert not (final/"frozen-groups.json").exists()
    assert not (final/"grpo/reference-logprobs.pt").exists()


def legacy_checkpoint(cfg, tmp):
    paused=run(cfg,"legacy-parent",stop=1)
    cp=checkpoint(paused,1)
    shutil.rmtree(cp/"grpo/reference")
    for name in ("stream-plan.json","source-provenance.json","anchor-tensors.json","stream-resume.json"):
        (cp/"grpo"/name).unlink()
    receipt=read_json(cp/"receipt.json")
    commit_receipt(cp,"lora-checkpoint-v1",**{k:v for k,v in receipt.items()
        if k not in {"files","kind","schema_version","created_at"}})
    (paused/"receipt.json").unlink()
    attempt=read_json(paused/"attempt.json")
    attempt.pop("contract",None);attempt["status"]="failed"
    write_json(paused/"attempt.json",attempt)
    return paused,cp


def test_legacy_failed_parent_readonly_then_self_contained_output(tmp_path):
    cfg=fixture(tmp_path)
    parent,cp=legacy_checkpoint(cfg,tmp_path)
    before={str(p):sha256(p) for p in parent.rglob('*') if p.is_file()}
    verify_checkpoint(cp)
    resolved=resolve_checkpoint(cp,read_json(cp/"state.json")["contract"])
    assert resolved["origin"]["legacy_parent_attempt"]==str(parent)
    result=run(cfg,"legacy-resume",parent=cp,stop=2)
    assert (checkpoint(result,2)/"grpo/stream-resume.json").exists()
    assert all(sha256(p)==h for p,h in before.items())


@pytest.mark.parametrize("fault",["plan","proof","provenance","anchor","tensor","missing"])
def test_legacy_corrupt_evidence_refused(tmp_path,fault):
    cfg=fixture(tmp_path)
    parent,cp=legacy_checkpoint(cfg,tmp_path)
    if fault in {"plan","proof"}:
        p=parent/"grpo/stream-plan.json";value=read_json(p)
        if fault=="plan":value["groups"].reverse()
        else:value["proof"]["branch_count"]+=1
        write_json(p,value)
    elif fault=="provenance":(parent/"grpo/source-provenance.json").write_text('{}')
    elif fault=="missing":(parent/"initial-adapter.json").unlink()
    elif fault=="anchor":write_json(parent/"initial-adapter.json",{"bad":"anchor"})
    else:
        from safetensors.torch import load_file,save_file
        p=parent/"grpo/reference/adapter/adapter_model.safetensors"
        values=load_file(p);next(iter(values.values())).add_(1)
        save_file(values,p)
        # Self-described hashes alone cannot authorize a changed original anchor.
        from RL.artifacts import inventory
        identity=read_json(p.parent/"identity.json")
        files=inventory(p.parent);files.pop("identity.json")
        identity["files"]=files;write_json(p.parent/"identity.json",identity)
        from tests.lora_helpers import model_and_targets
        from RL.grpo.stream_resume import inherit,validate_anchor
        resolved=resolve_checkpoint(cp,read_json(cp/"state.json")["contract"])
        dest=tmp_path/"inherited";inherit(resolved,dest)
        model,_=model_and_targets()
        with pytest.raises(ValueError,match="actual tensor"):
            validate_anchor(model,dest,read_json(cp/"state.json")["contract"])
        return
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):
        verify_checkpoint(cp)


@pytest.mark.parametrize("fault",["missing","tamper","symlink","traversal"])
def test_bundle_never_falls_back_to_parent(tmp_path,fault):
    cfg=fixture(tmp_path);parent=run(cfg,"pause",stop=1);cp=checkpoint(parent,1)
    if fault=="missing":(cp/"grpo/source-provenance.json").unlink()
    elif fault=="tamper":(cp/"grpo/stream-plan.json").write_text('{}')
    elif fault=="symlink":
        p=cp/"grpo/stream-plan.json";p.unlink();p.symlink_to(parent/"grpo/stream-plan.json")
    else:
        p=cp/"grpo/stream-resume.json";value=read_json(p)
        value["files"]["../outside"]={"sha256":"x","bytes":1};write_json(p,value)
    with pytest.raises((ValueError,FileNotFoundError)):
        resolve_checkpoint(cp,read_json(cp/"state.json")["contract"])


@pytest.mark.parametrize("fault",["boundary","prefix","branches","tokens","bounds"])
def test_cursor_validation(tmp_path,fault):
    cfg=fixture(tmp_path)
    from RL.grpo.streaming import create_plan,StreamingGroups
    groups=StreamingGroups(cfg,create_plan(cfg))
    state={"global_step":1,"group_cursor":2,"consumed_groups":2,"consumed_branches":6,
           "consumed_tokens":14,**groups.cursor(2)}
    if fault=="boundary":state["group_cursor"]=1
    elif fault=="prefix":state["consumed_group_ids_sha256"]="bad"
    elif fault=="branches":state["consumed_branches"]+=1
    elif fault=="tokens":state["consumed_tokens"]+=1
    else:state["group_cursor"]=100
    with pytest.raises(ValueError):groups.validate_state(state)


def test_mode_selection_and_unknown_version(tmp_path):
    cfg=fixture(tmp_path)
    assert execution_mode(cfg)=="streaming"
    cfg.run.start.mode="resume"
    assert execution_mode(cfg,{"prepared":{}})=="prepared"
    assert execution_mode(cfg,{})=="legacy"
    frozen={"streaming":{"version":1,"window_queue":2,"assignment":"window_ordinal_mod_world_size"}}
    assert execution_mode(cfg,frozen)=="streaming"
    cfg.runtime.data_preparation_async=False
    with pytest.raises(ValueError,match="requires"):execution_mode(cfg,frozen)
    cfg.runtime.data_preparation_async=True
    for mode in ("warm_start","append_resume"):
        cfg.run.start.mode=mode
        with pytest.raises(ValueError,match="exact resume only"):execution_mode(cfg,frozen)
    frozen["streaming"]["version"]=999
    with pytest.raises(ValueError,match="Unsupported"):execution_mode(cfg,frozen)


@pytest.mark.parametrize("fault",["native_token","metadata","probability","source_index"])
def test_mandatory_payload_checks_with_audits_disabled(tmp_path,fault):
    cfg=fixture(tmp_path)
    from RL.grpo.streaming import StreamingGroups,create_plan,bind_source
    from RL.grpo.reference import adapter_compatibility
    from RL.modeling import adapter_fingerprint
    from RL.checkpoints import export_adapter,adapter_contract
    from tests.lora_helpers import model_and_targets
    from RL.dataset.materialize import read_array
    plan=create_plan(cfg)
    group_id=plan['groups'][0]['group_id']
    model,targets=model_and_targets()
    directory=tmp_path/'preparation'
    write_json(directory/'grpo/stream-plan.json',plan)
    groups=StreamingGroups(cfg,plan)
    row=groups._ordered[0]['members'][0]
    anchor_hash=hash_json(adapter_fingerprint(model))
    identity={'model_fingerprint':BASE['fingerprint'],'tokenizer_fingerprint':'tiny-tokenizer',
              'adapter_hash':anchor_hash,'objective_version':'offline_grpo_k3_v1'}
    bind_source(cfg,groups,directory,BASE,
                adapter_compatibility(model,start_mode='fresh',adapter_hash=anchor_hash),identity)
    if fault=='native_token':
        path=Path(row['native_evidence']['root'])/row['native_evidence']['ledger_ref']
        value=read_json(path);value['output_token_ids'][0]+=1;write_json(path,value)
    elif fault=='metadata':
        path=Path(cfg.algorithm.dataset_root)/'RL/datasets/a/branches.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['historical_score']+=1
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with pytest.raises(ValueError,match='metadata checksum'):
            StreamingGroups(cfg,plan)
        return
    elif fault=='source_index':
        path=groups.source.path/'branches.jsonl'
        path.write_text(path.read_text()+' ')
        with pytest.raises(ValueError,match='metadata changed'):
            groups.close()
        return
    else:
        record=groups.source.records[(row['group_id'],row['branch_sample_id'])]
        ref=record['values_ref']
        path=groups.source.path/ref['shard']
        with path.open('r+b') as stream:
            stream.seek(ref['offset']*4);stream.write(np.array([-1.25],dtype='<f4').tobytes())
    windows=groups.windows(0)
    with pytest.raises(ValueError,match='checksum'):
        try:next(windows)
        finally:windows.close()
    assert groups._thread is None or not groups._thread.is_alive()


def test_prefetch_stop_and_producer_rng_are_bounded(tmp_path):
    cfg=fixture(tmp_path,n=13)
    from RL.grpo.streaming import StreamingGroups,create_plan,attach_source
    from RL.training import capture_rng
    from tests.lora_helpers import model_and_targets
    # Reuse a sealed source binding, then stop with a full two-window queue.
    paused=run(cfg,'pause',stop=1)
    contract=read_json(checkpoint(paused,1)/'state.json')['contract']
    groups=StreamingGroups(cfg,read_json(paused/'grpo/stream-plan.json'))
    attach_source(cfg,groups,contract['grpo']['probability_binding'],directory=paused)
    before=capture_rng()
    windows=groups.windows(2)
    next(windows)
    import time
    deadline=time.monotonic()+3
    while groups._queue.qsize()<2 and time.monotonic()<deadline:time.sleep(.02)
    assert groups._queue.qsize()==2
    windows.close()
    assert not groups._thread.is_alive()
    assert_tree_close(before,capture_rng())


def test_runner_dry_run_and_failed_status_do_not_import_torch(tmp_path):
    cfg=fixture(tmp_path)
    parent,cp=legacy_checkpoint(cfg,tmp_path)
    cfg.run.attempt_id='dry-run'
    cfg.run.mode='dry-run'
    cfg.run.start.mode='resume';cfg.run.start.checkpoint=str(cp)
    cfg.validation.stop_after_update=2
    file=tmp_path/'dry-run.yaml';file.write_text(yaml.safe_dump(cfg.model_dump()))
    write_json(parent/'resolved-config.json',cfg.model_dump())
    # The public dispatcher runs against a tiny identity; no model is required.
    script='''
import importlib.abc,sys,json
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch import')
sys.meta_path.insert(0,NoTorch())
import RL.service
RL.service.base_identity=lambda *args,**kw:{'fingerprint':'a'*64}
from core.cli import main
main(sys.argv[1:])
assert 'torch' not in sys.modules
'''
    for args in (["lora-train","--training-config",str(file)],
                 ["lora-status",parent.name,"--storage-root",cfg.runtime.output_root,"--format","json"]):
        result=subprocess.run([sys.executable,'-c',script,*args],cwd=ROOT,text=True,capture_output=True,timeout=40)
        assert result.returncode==0,result.stderr[-5000:]
        summary=json.loads(result.stdout)
        if args[0]=='lora-train':
            assert summary['streaming_resume']['initial_state']['global_step']==1
            assert summary['streaming_resume']['tensor_validation']=='pending_training_model'
        else:
            assert summary['recoverable_checkpoint_step']==1
            assert summary['status']=='failed'


def test_completed_epoch_and_nonadvancing_stop_refused(tmp_path):
    from RL.grpo.contract import validate_resume
    cfg=fixture(tmp_path)
    parent=run(cfg,'pause',stop=1)
    metadata=read_json(checkpoint(parent,1)/'state.json')
    cfg.validation.stop_after_update=1
    with pytest.raises(ValueError,match='Pause target'):
        validate_resume(cfg,metadata['contract']['input'],metadata)
    cfg.validation.stop_after_update=None
    metadata['state'].update(global_step=4,group_cursor=7)
    with pytest.raises(ValueError,match='already complete'):
        validate_resume(cfg,metadata['contract']['input'],metadata)


def test_missing_rank_rng_and_cuda_topology_refused(tmp_path,monkeypatch):
    from RL.training import restore_rng,capture_rng
    state=capture_rng()
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(ValueError,match='CUDA RNG topology'):
        restore_rng(state)
    monkeypatch.undo()
    cfg=fixture(tmp_path)
    parent=run(cfg,'pause',stop=1)
    cp=checkpoint(parent,1)
    from tests.lora_helpers import model_and_targets
    from RL.training import optimizer_and_scheduler
    from RL.checkpoints import restore_checkpoint
    from RL.parallel import GroupParallel
    model,_=model_and_targets()
    optimizer,scheduler=optimizer_and_scheduler(model,cfg.training)
    metadata=read_json(cp/'state.json')
    monkeypatch.setattr('RL.parallel._context',GroupParallel(0,2))
    with pytest.raises(ValueError,match='rank RNG topology'):
        restore_checkpoint(cp,optimizer,scheduler,metadata['contract'],metadata['adapter_contract'])


@pytest.mark.parametrize('parallel',[False,True])
def test_mid_window_failure_recovers_only_last_checkpoint(tmp_path,parallel):
    cfg=fixture(tmp_path,n=5)
    from RL.grpo.streaming import create_plan
    plan=create_plan(cfg)
    full=run(cfg,'full',parallel=parallel)
    failed=run(cfg,'failed',parallel=parallel,fail_group=plan['groups'][2]['group_id'])
    assert checkpoint(failed,1).is_dir()
    assert not checkpoint(failed,2).exists()
    resumed=run(cfg,'resume',parent=checkpoint(failed,1),parallel=parallel)
    compare(full,resumed,[2,3],parallel=parallel)


def worker_rank(rank, cfg, rendezvous):
    worker(cfg, rank=rank, rendezvous=rendezvous)


if __name__=="__main__":
    cfg=Config.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text()))
    if sys.argv[2]=="parallel":
        import torch.multiprocessing as mp
        rendezvous=str(Path(sys.argv[1]).with_suffix('.store'))
        mp.spawn(worker_rank,args=(cfg,rendezvous),nprocs=2,join=True)
    else:
        worker(cfg)
