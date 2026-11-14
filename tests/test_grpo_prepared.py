"""Prepared storage, sealed-source reuse and real tiny-Qwen append lifecycle."""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
import torch
import yaml

from core.io import hash_json, read_json, sha256, write_json
from RL.config import load, attempt_directory
from RL.dataset.artifacts import REQUIRED, directory, seal
from RL.dataset.schema import spans
from RL.grpo.data import prepare, tensor_groups
from RL.grpo.schedule import LENGTH_SHUFFLE, POOL_GROUPS, order_groups
from RL.grpo.prepared import (PreparedReader, PreparedGroups, binding, materialize, resolve,
                              validate_extension_schedule, validate_schedule)
from RL.grpo.prepared_service import create, read_extension
from RL.modeling import adapter_fingerprint
from RL.service import _source, _train_grpo
from tests.lora_helpers import LORA, model_and_targets

ROOT = Path(__file__).parents[1]
BASE = {"fingerprint":"a"*64}


def dataset(tmp, name, group_numbers, *, changed=None, branch_count=3):
    path = directory(tmp/"sources", "rl-dataset", name)
    path.mkdir(parents=True)
    for filename in REQUIRED["rl-dataset"]:
        (path/filename).write_text("" if filename.endswith(".jsonl") else "{}")
    groups, branches = [], []
    ledgers = tmp/"native"
    ledgers.mkdir(exist_ok=True)
    for number in group_numbers:
        gid = f"g{number}"
        ids = [f"{gid}-b{i}" for i in range(branch_count)]
        groups.append({"group_id":gid, "branch_ids":ids, "expected_branch_count":branch_count, "data_ready":True})
        p = [2,3,4+number%3]
        for i,bid in enumerate(ids):
            c = [5,6+i,7][:2+i%2]
            ledger = ledgers/(bid+".json")
            if not ledger.exists():
                write_json(ledger, {"prompt_token_ids":p, "output_token_ids":c})
            row = {"group_id":gid, "branch_sample_id":bid, "historical_score":[0,2,5][i] if branch_count == 3 else float(i),
                "retained":i<(2 if branch_count == 3 else branch_count//2), "action":"native-action", "prompt_sha256":hash_json(p),
                "completion_sha256":hash_json(c), "prompt_count":len(p), "completion_count":len(c),
                "mask_spans":spans(len(p),len(c)), "tokenizer_fingerprint":"tiny-tokenizer",
                "native_request_id":bid, "session_id":"tiny", "decision_id":gid,
                "branch_id":str(i), "agent_step":number, "message_index":0,
                "native_evidence":{"root":str(ledgers), "ledger_ref":ledger.name}}
            if changed:
                changed(row)
            branches.append(row)
    write_json(path/"dataset.json", {"dataset_id":name, "data_ready":True,
        "cohort":{"model":BASE, "tokenizer_fingerprint":"tiny-tokenizer"}})
    (path/"groups.jsonl").write_text("".join(json.dumps(g)+"\n" for g in groups))
    (path/"branches.jsonl").write_text("".join(json.dumps(r)+"\n" for r in branches))
    seal(path,"rl-dataset",name)
    return path


def config(tmp, dataset_id="a", updates=3):
    cfg,_ = load(ROOT/"RL/experiments/grpo-offline-k3-smoke.yaml",ROOT)
    cfg.lora = LORA.model_copy()
    cfg.runtime.output_root = str(tmp/"artifacts")
    cfg.algorithm.dataset_root = str(tmp/"sources")
    cfg.algorithm.dataset_id = dataset_id
    cfg.grpo.group_batch_size = 2
    cfg.training.max_updates = updates
    cfg.training.learning_rate = .003
    cfg.run.mode = "smoke"
    return cfg


def train(cfg, name, *, parent=None, mode="fresh", stop=None, extension=None):
    cfg = cfg.model_copy(deep=True)
    cfg.run.attempt_id = name
    cfg.run.start.mode = mode
    cfg.run.start.checkpoint = str(parent) if parent else None
    cfg.run.start.extension_manifest = str(extension) if extension else None
    cfg.validation.stop_after_update = stop
    owner = _source(cfg)
    model,targets = model_and_targets(adapter=owner["adapter"] if owner else None)
    path = attempt_directory(cfg); path.mkdir(parents=True)
    attempt = {"attempt_id":name,"purpose":"engineering_smoke","start_mode":mode,
               "parent":{k:v for k,v in owner.items() if k not in {"identity","adapter"}} if owner else None}
    result = _train_grpo(cfg, ROOT, path, BASE, owner, attempt, model, targets, optimizer_settings=cfg.training)
    return path, model, result, cfg


def extension(tmp, cfg, cp, name="extension"):
    cfg = cfg.model_copy(deep=True)
    cfg.run.start.mode = "append_resume"
    cfg.run.start.checkpoint = str(cp)
    cfg.run.start.extension_manifest = str(directory(cfg.runtime.output_root,"rl-extension",name)/"manifest.json")
    file = tmp/(name+".yaml"); file.write_text(yaml.safe_dump(cfg.model_dump()))
    return create(file,{"project_root":str(ROOT)},name+"-prepared",checkpoint=cp,
                  extension_id=name,append_seed=19,shard_groups=2)


def test_shards_match_legacy_order_tensors_and_single_prompt(tmp_path):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    raw, proof = prepare(cfg)
    reader = materialize(cfg,ROOT,"prepared",shard_groups=2)
    assert reader.ids == proof["eligible_group_ids"]
    assert reader.proof["group_order_sha256"] == proof["group_order_sha256"]
    assert len(reader.manifest["shards"]) == 3
    assert sum(p.stat().st_size for p in reader.path.glob("shards/*/prompt-tokens.bin")) == 5*3*8
    expected = list(tensor_groups(raw))
    actual = list(PreparedGroups(reader))
    for left,right in zip(expected,actual):
        assert left.group_id == right.group_id and left.branch_ids == right.branch_ids
        assert tuple(left.advantages) == tuple(right.advantages)
        for key in ("input_ids","attention_mask","labels","loss_mask"):
            torch.testing.assert_close(getattr(left,key),getattr(right,key),atol=0,rtol=0)
    before=sha256(reader.path/"receipt.json")
    with pytest.raises(FileExistsError): materialize(cfg,ROOT,"prepared")
    assert sha256(reader.path/"receipt.json") == before


def test_prefix_length_pool_shuffle_groups_similar_prompts_and_is_deterministic():
    items=[{"id":f"g{i}","length":i} for i in range(128)]
    args={"seed":73,"batch_size":4,"algorithm":LENGTH_SHUFFLE,
          "prompt_length":lambda row:row["length"]}
    first=order_groups(items,**args); second=order_groups(items,**args)
    assert [row["id"] for row in first]==[row["id"] for row in second]
    assert sorted(row["id"] for row in first)==sorted(row["id"] for row in items)
    windows=[first[i:i+4] for i in range(0,len(first),4)]
    assert all(len(window)==4 for window in windows)
    assert all(max(x["length"] for x in w)-min(x["length"] for x in w)==3 for w in windows)
    assert all([x["length"] for x in w]==sorted(x["length"] for x in w) for w in windows)
    assert [w[0]["length"] for w in windows]!=sorted(w[0]["length"] for w in windows)


def test_prefix_length_pool_shuffle_retains_one_final_partial_window():
    items=[{"id":f"g{i}","length":i} for i in range(131)]
    result=order_groups(items,seed=7,batch_size=4,algorithm=LENGTH_SHUFFLE,
                        prompt_length=lambda row:row["length"])
    windows=[result[i:i+4] for i in range(0,len(result),4)]
    assert [len(window) for window in windows]==[4]*32+[3]
    assert len({row["id"] for row in result})==len(items)
    assert sorted(row["id"] for row in result)==sorted(row["id"] for row in items)


def test_legacy_shuffle_schedule_remains_reconstructible():
    from RL.grpo.prepared import segment, validate_schedule
    ids=[f"g{i}" for i in range(9)]
    old=segment(ids,4,17,0)
    assert old["shuffle_algorithm"]=="stable_random_v1"
    schedule={"schema_version":1,"schedule_id":"legacy","segments":[old],
              "parent_schedule_hash":None,"parent_last_segment_hash":None,"extension_request_id":None}
    schedule["sha256"]=hash_json(schedule)
    assert validate_schedule(schedule)==schedule


@pytest.mark.parametrize("fault",["receipt","token","missing","reorder","offset","duplicate_branch"])
def test_prepared_corruption_refused(tmp_path,fault):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    reader=materialize(cfg,ROOT,"prepared",shard_groups=2)
    path=reader.path
    if fault=="receipt": (path/"receipt.json").write_text("{}")
    elif fault=="token": next(path.glob("shards/*/completion-tokens.bin")).write_bytes(b"bad")
    elif fault=="missing": next(path.glob("shards/*/prompt-tokens.bin")).unlink()
    elif fault=="reorder":
        m=read_json(path/"manifest.json"); m["shards"].reverse(); write_json(path/"manifest.json",m)
    else:
        with sqlite3.connect(path/"inventory.sqlite3") as db:
            gid,payload=db.execute("SELECT id,payload FROM groups WHERE eligible=1 LIMIT 1").fetchone()
            value=json.loads(payload)
            if fault=="offset": value["storage"]["prompt_offset"]=10**10
            else: value["members"][1]["branch_sample_id"]=value["members"][0]["branch_sample_id"]
            db.execute("UPDATE groups SET payload=? WHERE id=?",(json.dumps(value),gid))
        # Even a re-sealed malformed index must fail at the checked window read.
        seal(path,"rl-prepared",path.name)
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):
        bad=PreparedReader(path)
        list(PreparedGroups(bad))


def test_failed_materialization_is_not_a_consumer_artifact(tmp_path,monkeypatch):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    def fail(*args,**kwargs): raise RuntimeError("injected write failure")
    monkeypatch.setattr("RL.grpo.prepared.freeze_group",fail)
    with pytest.raises(RuntimeError,match="injected"): materialize(cfg,ROOT,"failed")
    path=directory(cfg.runtime.output_root,"rl-prepared","failed")
    assert not path.exists()
    assert not (path.with_name(".failed.staging")/"receipt.json").exists()


def test_exact_resume_opens_no_source_or_consumed_tokens(tmp_path,monkeypatch):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    full,full_model,full_result,_=train(cfg,"full")
    paused,_,_,_=train(cfg,"paused",stop=1)
    def forbidden(*args,**kwargs): raise AssertionError("exact resume rescanned source/ledger/probabilities")
    monkeypatch.setattr("RL.grpo.prepared.source_groups",forbidden)
    monkeypatch.setattr("RL.dataset.loader.native_tokens",forbidden)
    monkeypatch.setattr("RL.dataset.probability.open_probability_stream",forbidden)
    monkeypatch.setattr("RL.grpo.data.prepare",forbidden)
    resumed,resumed_model,result,_=train(cfg,"resumed",mode="resume",parent=paused/"checkpoints/step-000001")
    assert result["state"]==full_result["state"]
    assert adapter_fingerprint(resumed_model)==adapter_fingerprint(full_model)
    report=read_json(resumed/"grpo-report.json")
    assert report["preparation"]["prepared_cache_hit"]
    assert report["preparation"]["native_scan_groups"]==0
    assert report["preparation"]["reference_new_groups"]==0
    assert not (resumed/"frozen-groups.json").exists()
    from tests.test_grpo_lifecycle import assert_tree_close
    a=torch.load(full/"checkpoints/step-000003/training-state.pt",weights_only=True)
    b=torch.load(resumed/"checkpoints/step-000003/training-state.pt",weights_only=True)
    for key in ("optimizer","scheduler","rng"): assert_tree_close(a[key],b[key])


@pytest.mark.parametrize("stop",[1,2,None])
def test_append_consumes_old_remainder_then_new_segment_once(tmp_path,monkeypatch,stop):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    paused,_,_,_=train(cfg,"parent",stop=stop)
    step=stop or 3; cp=paused/f"checkpoints/step-{step:06d}"
    old=PreparedReader(resolve(read_json(cp/"state.json")["contract"]["prepared"]))
    parent_hashes={str(p):sha256(p) for p in paused.rglob("*") if p.is_file()}
    dataset(tmp_path,"b",range(8))
    childcfg=config(tmp_path,"b",updates=5)
    reads=[]
    from RL.dataset.loader import native_tokens
    def counted(row,**kwargs):
        reads.append(row["group_id"]); return native_tokens(row,**kwargs)
    monkeypatch.setattr("RL.dataset.loader.native_tokens",counted)
    prepared=extension(tmp_path,childcfg,cp)
    assert set(reads)=={"g5","g6","g7"} and len(reads)==9
    request,_,child,_=read_extension(prepared["extension_manifest"])
    assert child.schedule["segments"][:1]==old.schedule["segments"]
    assert child.ids[:5]==old.ids
    assert [len(w) for w in child.windows()]==[2,2,1,2,1]
    forwards=[]
    from RL.grpo.reference import build_reference
    def counted_reference(model,groups,*args,**kwargs):
        forwards.extend(g.group_id for g in groups)
        return build_reference(model,groups,*args,**kwargs)
    monkeypatch.setattr("RL.grpo.reference.build_reference",counted_reference)
    appended,_,result,_=train(childcfg,"appended",mode="append_resume",parent=cp,
                            extension=prepared["extension_manifest"])
    assert set(forwards)=={"g5","g6","g7"} and len(forwards)==3
    assert result["state"]["group_cursor"]==8 and result["state"]["global_step"]==5
    assert result["state"]["consumed_group_ids_sha256"]==hash_json(child.ids)
    report=read_json(appended/"grpo-report.json")
    assert report["extension"]["new_groups"]==3
    assert report["extension"]["new_tokens"]==21
    assert [r["group_count"] for r in report["updates"]]==[2,2,1,2,1]
    assert report["preparation"]["reference_reused_groups"]==5
    assert all(sha256(p)==digest for p,digest in parent_hashes.items())


@pytest.mark.parametrize("fault",["score","retained","branch","token_hash","branch_collision"])
def test_append_rejects_old_identity_changes(tmp_path,fault):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    parent=materialize(cfg,ROOT,"parent")
    def change(row):
        if row["group_id"]=="g0":
            if fault=="score": row["historical_score"]+=.1
            if fault=="retained": row["retained"]=not row["retained"]
            if fault=="branch": row["branch_sample_id"]+="changed"
            if fault=="token_hash": row["completion_sha256"]="b"*64
        if fault=="branch_collision" and row["group_id"]=="g5": row["branch_sample_id"]="g0-b0"
    dataset(tmp_path,"b",range(8),changed=change)
    with pytest.raises(ValueError,match="existing|conflicts|order"):
        materialize(config(tmp_path,"b",updates=5),ROOT,"child",parent=parent,append_seed=19,extension_id="x")


def test_schedule_insertion_and_linear_extension_refused(tmp_path):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    paused,_,_,_=train(cfg,"parent",stop=1)
    dataset(tmp_path,"b",range(8)); childcfg=config(tmp_path,"b",updates=5)
    childcfg.training.scheduler="linear"
    with pytest.raises(ValueError,match="identity|constant"):
        extension(tmp_path,childcfg,paused/"checkpoints/step-000001")


def test_cli_preparation_and_verification_are_torch_free(tmp_path):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    file=tmp_path/"prepare.yaml"; file.write_text(yaml.safe_dump(cfg.model_dump()))
    script='''
import importlib.abc,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=="torch" or fullname.startswith("torch."): raise AssertionError("prepared CLI imported Torch")
sys.meta_path.insert(0,NoTorch())
from core.cli import main
main(sys.argv[1:])
assert "torch" not in sys.modules
'''
    result=subprocess.run([sys.executable,"-c",script,"lora-prepare","--training-config",str(file),
        "--prepared-id","cli-prepared","--shard-groups","2"],cwd=ROOT,text=True,capture_output=True,timeout=45)
    assert result.returncode==0,result.stderr[-3000:]
    summary=json.loads(result.stdout)
    assert summary["group_count"]==5 and summary["local_shards"]==3
    for action in ("show","verify"):
        result=subprocess.run([sys.executable,"-c",script,"artifact",action,"cli-prepared",
            "--scope","rl-prepared","--storage-root",cfg.runtime.output_root],
            cwd=ROOT,text=True,capture_output=True,timeout=45)
        assert result.returncode==0,result.stderr[-3000:]
        assert json.loads(result.stdout)["group_count"]==5


@pytest.mark.parametrize("fault",["shuffle","advantage","batch","sequence","cursor_hash"])
def test_prepared_identity_changes_are_rejected(tmp_path,fault):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    reader=materialize(cfg,ROOT,"prepared")
    from RL.grpo.prepared import validate_config
    if fault=="cursor_hash":
        state={"global_step":1,"group_cursor":2,"consumed_groups":2,"consumed_branches":6,
               "consumed_tokens":sum(reader.records[g]["token_count"] for g in reader.ids[:2]),**reader.cursor(2)}
        state["consumed_group_ids_sha256"]="b"*64
        with pytest.raises(ValueError,match="ID hash"): reader.validate_state(state)
    else:
        if fault=="shuffle":cfg.grpo.shuffle_seed+=1
        if fault=="advantage":cfg.grpo.boundary_tie_sigma=.1
        if fault=="batch":cfg.grpo.group_batch_size=3
        if fault=="sequence":cfg.validation.max_sequence_length+=1
        with pytest.raises(ValueError,match="identity"):validate_config(reader,cfg)


def test_prepared_producer_source_fingerprint_is_not_a_compatibility_gate(tmp_path):
    dataset(tmp_path,"a",range(5)); cfg=config(tmp_path)
    reader=materialize(cfg,ROOT,"prepared")
    manifest=read_json(reader.path/"manifest.json")
    manifest["producer_fingerprint"]="b"*64
    write_json(reader.path/"manifest.json",manifest)
    seal(reader.path,"rl-prepared",reader.path.name)
    from RL.config import PreparedBinding
    cfg.prepared_data=PreparedBinding.model_validate(binding(reader.path))
    from RL.grpo.prepared_service import for_training
    loaded,_,_,_=for_training(cfg,ROOT,None)
    assert loaded.schedule["sha256"]==reader.schedule["sha256"]


def test_external_append_imports_only_new_groups_and_freezes_anchor(tmp_path,monkeypatch):
    from tests.test_grpo_external_reference import make_source,select
    from RL.dataset.probability import ProbabilityStream
    dataset(tmp_path,"a",range(5));cfg=config(tmp_path)
    def native_rows(manifest,rows,path):
        for row in rows:row["native_request_id"]=row["branch_sample_id"]
    source_a=make_source(tmp_path/"probabilities",cfg,name="raw-a",mutate=native_rows)
    select(cfg,tmp_path/"probabilities",name="raw-a")
    parent,_,_,_=train(cfg,"parent",stop=1)
    cp=parent/"checkpoints/step-000001"
    dataset(tmp_path,"b",range(8));childcfg=config(tmp_path,"b",updates=5)
    producer_cfg=childcfg.model_copy(deep=True);producer_cfg.training.max_updates=4
    # The probability source may include all dataset branches; only new groups
    # are imported. Existing values remain owned by the sealed parent reference.
    make_source(tmp_path/"probabilities",producer_cfg,name="raw-b",mutate=native_rows)
    select(childcfg,tmp_path/"probabilities",name="raw-b")
    prepared=extension(tmp_path,childcfg,cp)
    reads=[]
    original=ProbabilityStream.values
    def counted(self,row,**kwargs):
        reads.append(row["group_id"]);return original(self,row,**kwargs)
    monkeypatch.setattr(ProbabilityStream,"values",counted)
    def forbidden(*args,**kwargs):raise AssertionError("external append performed model reference forward")
    monkeypatch.setattr("RL.grpo.reference.build_reference",forbidden)
    appended,_,result,_=train(childcfg,"appended",mode="append_resume",parent=cp,
                             extension=prepared["extension_manifest"])
    assert result["state"]["group_cursor"]==8
    assert set(reads)=={"g5","g6","g7"} and len(reads)==9
    assert read_json(appended/"grpo-contract.json")["grpo"]["reference_identity"]==read_json(cp/"state.json")["contract"]["grpo"]["reference_identity"]


def test_exact_resume_inside_appended_segment_matches_uninterrupted_extension(tmp_path):
    dataset(tmp_path,"a",range(5));cfg=config(tmp_path)
    parent,_,_,_=train(cfg,"parent",stop=1)
    cp=parent/"checkpoints/step-000001"
    dataset(tmp_path,"b",range(8));childcfg=config(tmp_path,"b",updates=5)
    prepared=extension(tmp_path,childcfg,cp)
    full,full_model,full_result,_=train(childcfg,"full-extension",mode="append_resume",parent=cp,
                                      extension=prepared["extension_manifest"])
    paused,_,_,_=train(childcfg,"partial-extension",mode="append_resume",parent=cp,stop=4,
                       extension=prepared["extension_manifest"])
    resumed,resumed_model,result,_=train(childcfg,"exact-extension",mode="resume",
                                        parent=paused/"checkpoints/step-000004")
    assert result["state"]==full_result["state"]
    assert adapter_fingerprint(resumed_model)==adapter_fingerprint(full_model)
    report=read_json(resumed/"grpo-report.json")
    assert report["preparation"]["native_scan_groups"]==0 and report["preparation"]["reference_new_groups"]==0


def test_legacy_frozen_groups_explicit_import_retains_order(tmp_path):
    dataset(tmp_path,"a",range(5));cfg=config(tmp_path)
    raw,proof=prepare(cfg)
    class Frozen:
        def __iter__(self):return iter(copy.deepcopy(raw))
    frozen=Frozen();frozen.proof=proof
    reader=materialize(cfg,ROOT,"migrated",legacy=frozen,source_attempt="legacy-attempt")
    assert reader.ids==proof["eligible_group_ids"]
    assert reader.proof["group_order_sha256"]==proof["group_order_sha256"]
    assert reader.manifest["source_attempt"]=="legacy-attempt"
    frozen.proof=copy.deepcopy(proof);frozen.proof["group_order_sha256"]="b"*64
    with pytest.raises(ValueError,match="Legacy"):
        materialize(cfg,ROOT,"bad-migration",legacy=frozen)
    assert not directory(cfg.runtime.output_root,"rl-prepared","bad-migration").exists()
