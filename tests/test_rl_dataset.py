"""Real-schema token-only fixtures exercise evidence and causal target boundaries."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.io import hash_json, read_json, sha256, write_json
from steering_fixtures import make_source
from RL.dataset.build import build_dataset, check_dataset, iter_rows
from RL.dataset.loader import iter_groups, collate_group
from RL.dataset.materialize import materialize
from RL.dataset.config import load
from RL.dataset.replay.validation import validate_values, unsupported_request


def token_source(tmp_path, *, bad_prompt=False, finish="stop", no_action=False):
    f = make_source(tmp_path, no_closes=True)
    d = f["decision"]; run=f["run"]; cfgpath=tmp_path/"runs"/run/"resolved-config.json"
    cfg=read_json(cfgpath); cfg["generation"]={"temperature":.6,"top_p":.95,"top_k":20,"min_p":0.,"presence_penalty":0.,"repetition_penalty":1.,"thinking_token_budget":None}
    write_json(cfgpath,cfg)
    for bid,c in enumerate(d["candidates"]):
        p=[10,11,12] if not (bad_prompt and bid==1) else [10,11,13]
        output=[20,30+bid,42]
        q={"request_id":c["native_request_id"],"session_id":f["sid"],"decision_id":d["decision_id"],"branch_id":bid,"step_id":0,"message_index":0,"parent_prefix_sha256":"parent-sha","query_id":"query-unit","api_attempt_id":f"api-{bid}","tokenizer_fingerprint":"unit-tokenizer","prompt_token_ids":p,"output_token_ids":output,"request_sha256":hash_json(p),"output_sha256":hash_json(output),"finish_reason":finish if bid==1 else "stop","run_id":run}
        ledger=f["nas"]/"requests"/f"{bid}.json";write_json(ledger,q)
        c.update(ledger_ref=f"requests/{bid}.json",ledger_sha256=sha256(ledger),api_attempt_id=f"api-{bid}",seed=42+bid,output_sha256=q["output_sha256"])
        if no_action and bid==1:c["response"]={"choices":[{"finish_reason":"stop","message":{"role":"assistant"}}]}
    write_json(f["session"]/"parallel/decisions/dec-unit.json",d)
    trajectory=tmp_path/f["row"]["trajectory_path"];t=read_json(trajectory);emb=copy.deepcopy(d)
    for c in emb["candidates"]:c.pop("response")
    t["messages"][0]["extra"]["dycon_parallel"]=emb;write_json(trajectory,t);tsha=sha256(trajectory)
    evpath=tmp_path/"runs"/run/"evidence.json";ev=read_json(evpath);ev["rollouts"][f["rid"]]["trajectory_sha256"]=tsha;write_json(evpath,ev)
    final=read_json(f["session"]/"finalization.json");final["trajectory_sha256"]=tsha;write_json(f["session"]/"finalization.json",final)
    index=read_json(f["session"]/"parallel/branch-index.json")
    for r in index:r["trajectory_sha256"]=tsha
    write_json(f["session"]/"parallel/branch-index.json",index)
    receipt=read_json(f["session"]/"parallel/branch-receipt.json");receipt["trajectory_sha256"]=tsha;receipt["files"]={k:sha256(f["nas"]/k) for k in receipt["files"]};write_json(f["session"]/"parallel/branch-receipt.json",receipt)
    pubpath=tmp_path/"runs"/run/"hidden/publication.json";pub=read_json(pubpath);pub["schema_version"]=1;pub["state"]="published"
    for ent in pub["sessions"]:
        ent["trajectory_sha256"]=tsha
        for name,proof in ent["files"].items():proof["sha256"]=sha256(f["session"]/name)
    write_json(pubpath,pub)
    config={"schema_version":1,"dataset_id":"unit-dataset","sources":{"runs":[run],"all_submitted":True},"audit":{"enabled":True},"storage":{"dataset_root":str(tmp_path/"out"),"cache_root":str(tmp_path/"cache"),"max_shard_bytes":1024,"memory_budget_gib":1.0}}
    file=tmp_path/"data.yaml";file.write_text(yaml.safe_dump(config),encoding="utf-8")
    return f,file,config


def build(fixture):
    f,file,_=fixture
    return Path(build_dataset(f["cfg"],file)["path"])


def test_complete_all_candidates_and_original_target_coordinates(tmp_path):
    fixture=token_source(tmp_path);ds=build(fixture);report=check_dataset(ds)
    assert report["branches"]==6 and report["groups"]==1
    groups=list(iter_groups(ds));assert len(groups)==1
    members=groups[0]["members"]
    assert len(members)==6 and sum(r["selected"] for r in members)==1
    assert any(not r["selected"] and not r["retained"] for r in members)
    b=collate_group(groups[0],as_tensors=False)
    assert b["input_ids"][0]==[10,11,12,20,30,42]
    assert b["labels"][0]==[-100,-100,-100,20,30,42]
    assert b["loss_mask"][0]==[False,False,False,True,True,True]
    assert read_json(ds/"dataset.json")["grpo_ready"] is False


@pytest.mark.parametrize("options,reason",[({"finish":"length"},"incomplete_native_finish"),({"finish":"abort"},"incomplete_native_finish"),({"no_action":True},"missing_tool_calls")])
def test_incomplete_group_keeps_every_slot(tmp_path,options,reason):
    ds=build(token_source(tmp_path,**options));rows=list(iter_rows(ds/"branches.jsonl"))
    assert len(rows)==6 and len(list(iter_rows(ds/"groups.jsonl")))==1
    assert not list(iter_groups(ds))
    assert any(r["exclusion_reason"]==reason for r in rows)


def test_prompt_mismatch_is_hard_failure(tmp_path):
    with pytest.raises(ValueError,match="different actual native prompts"):
        build(token_source(tmp_path,bad_prompt=True))
    assert not (tmp_path/"out/RL/datasets/unit-dataset/receipt.json").exists()


def test_ledger_tamper_is_not_data_exclusion(tmp_path):
    fixture=token_source(tmp_path);f,_,_=fixture
    (f["nas"]/"requests/1.json").write_text("{}")
    with pytest.raises(ValueError,match="checksum"):
        build(fixture)


def test_audit_off_is_default_and_skips_requested_identity_and_token_checks(tmp_path):
    fixture=token_source(tmp_path,bad_prompt=True);f,file,cfg=fixture
    cfg.pop("audit");file.write_text(yaml.safe_dump(cfg),encoding="utf-8")
    f["nas"].joinpath("requests/1.json").write_text("{}")
    dataset=Path(build_dataset(f["cfg"],file)["path"])
    assert read_json(dataset/"dataset.json")["audit"]["enabled"] is False
    assert not list(iter_groups(dataset))


def test_source_mutation_after_build_is_rejected(tmp_path):
    fixture=token_source(tmp_path);ds=build(fixture)
    (fixture[0]["nas"]/"requests/1.json").write_text("{}")
    assert check_dataset(ds,"metadata")["status"]=="verified"
    with pytest.raises(ValueError,match="dependency"):
        check_dataset(ds,"sources")
    with pytest.raises(ValueError,match="checksum"):
        list(iter_groups(ds))


def test_no_hidden_tensor_needed_and_compressed_response_is_resolved(tmp_path):
    fixture=token_source(tmp_path);f,_,_=fixture
    for p in (f["nas"]/"raw").glob("*"):p.unlink()
    ds=build(fixture)
    assert len(list(iter_groups(ds))[0]["members"])==6


def test_cache_is_mmap_without_pickle_and_matches_every_native_token(tmp_path):
    ds=build(token_source(tmp_path));r=materialize(ds,tmp_path/"cache");cache=Path(r["path"])
    groups=list(iter_groups(ds,cache=cache));native=list(iter_groups(ds))
    assert [r["completion_token_ids"] for r in groups[0]["members"]]==[r["completion_token_ids"] for r in native[0]["members"]]
    assert all(p.suffix!=".npz" for p in cache.rglob("*"))
    assert materialize(ds,tmp_path/"cache")["reused"]
    shard=next((cache/"shards").iterdir())
    assert np.memmap(shard,dtype="<u4",mode="r").ndim==1
    with shard.open("ab") as stream:stream.write(b"bad")
    with pytest.raises(ValueError,match="inventory/hash"):
        list(iter_groups(ds,cache=cache))


def test_explicit_length_budget_rejects_whole_group(tmp_path):
    ds=build(token_source(tmp_path))
    with pytest.raises(ValueError,match="no truncation"):
        list(iter_groups(ds,max_sequence_length=5))


def test_duplicate_yaml_keys_unknown_fields_and_missing_selector(tmp_path):
    fixture=token_source(tmp_path);_,file,cfg=fixture
    file.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ValueError,match="Duplicate"):load(file,tmp_path)
    cfg["unexpected"]=True;file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load(file,tmp_path)
    del cfg["unexpected"];cfg["sources"].pop("all_submitted");file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match="selector"):load(file,tmp_path)


def test_zero_support_nan_and_trace_constraints():
    assert validate_values([-1.,float("-inf")],2)==[-1.,float("-inf")]
    for value in (float("nan"),float("inf"),1.):
        with pytest.raises(ValueError):validate_values([value],1)
    with pytest.raises(ValueError):validate_values([-.2],2)
    assert unsupported_request({"thinking_token_budget":12})==["thinking_token_budget"]


def test_audit_off_accepts_legacy_publication_sessions_mapping(tmp_path):
    fixture = token_source(tmp_path)
    f, file, cfg = fixture
    cfg["audit"]["enabled"] = False
    file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pubpath = tmp_path / "runs" / f["run"] / "hidden/publication.json"
    publication = read_json(pubpath)
    publication["sessions"] = {entry["session_id"]: entry for entry in publication["sessions"]}
    write_json(pubpath, publication)
    dataset = Path(build_dataset(f["cfg"], file)["path"])
    group = next(iter_groups(dataset))
    assert len(group["members"]) == 6
    assert all(member["prompt_token_ids"] == [10, 11, 12] for member in group["members"])
    assert check_dataset(dataset, "full")["status"] == "verified"


@pytest.mark.parametrize("missing", ["finalization.json", "session_directory"])
def test_missing_session_evidence_preserves_branch_inventory(tmp_path, missing):
    import shutil
    fixture = token_source(tmp_path)
    f, file, cfg = fixture
    cfg["audit"]["enabled"] = False
    file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if missing == "session_directory":
        shutil.rmtree(f["session"])
    else:
        (f["session"] / missing).unlink()
    dataset = Path(build_dataset(f["cfg"], file)["path"])
    rows = list(iter_rows(dataset / "branches.jsonl"))
    assert len(rows) == 6
    assert all(row["exclusion_reason"] == "native_session_evidence_unavailable" for row in rows)
    assert not list(iter_groups(dataset))
    assert check_dataset(dataset, "full")["branches"] == 6
