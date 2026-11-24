"""Fixed probe: known preference math, frozen payloads, real tiny-model forwards and CLI service."""
import argparse
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from core.io import hash_json, read_json, sha256, write_json
from RL.probe import _output_path, _publish, evaluate, run
from RL.probe_data import ProbeGroups
from RL.probe_metrics import TokenChanges, aggregate, compare, group_metrics, token_change_metrics
from tests.test_grpo_prepared import BASE, ROOT, config, dataset, train
from tests.lora_helpers import model_and_targets


def branch(bid, advantage, logp, reference, count):
    return {"branch_id":bid,"advantage":advantage,"logprob_mean":logp,
            "reference_logprob_mean":reference,"logprob_change":logp-reference,"token_count":count}


def test_preference_math_weights_branches_and_groups_not_lengths():
    a = [branch("a", 1, -1, -3, 100), branch("b", 1, -3, -3, 1), branch("c", -2, -5, -4, 2)]
    metrics = group_metrics(a)
    assert metrics["positive_logprob_mean"] == -2
    assert metrics["preference_gap"] == 3
    assert metrics["reference_preference_gap"] == 1
    assert metrics["preference_gap_change"] == 2
    assert metrics["advantage_logprob_alignment"] == pytest.approx(4/3)
    assert metrics["pairwise_preference_rate"] == 1
    b = [branch("d", 1, -2, -3, 5), branch("e", -1, -2, -3, 1000)]
    result, support = aggregate([{"metrics":metrics,"branches":a}, {"metrics":group_metrics(b),"branches":b}])
    assert result["preference_gap_change"] == 1
    assert result["pairwise_preference_rate"] == .75
    assert support["token_count"] == 1108
    assert support["comparable_group_count"] == 2


def test_missing_sign_is_null_and_support_is_explicit():
    rows = [branch("a",1,-2,-3,2), branch("b",0,-4,-4,3)]
    m = group_metrics(rows)
    assert m["negative_logprob_change"] is None
    assert m["preference_gap"] is None
    metrics, support = aggregate([{"metrics":m,"branches":rows}])
    assert metrics["improved_group_fraction"] is None
    assert support["comparable_group_count"] == 0


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1)
    dataset(tmp_path, "a", range(3))
    cfg = config(tmp_path, updates=2)
    cfg.training.checkpoint_every_updates = 1
    path, _, _, cfg = train(cfg, "probe-source")
    write_json(path/"resolved-config.json", cfg.model_dump())
    cp = path/"checkpoints/step-000001"
    return cfg, path, cp, read_json(cp/"state.json")["contract"]


def test_prepared_probe_is_repeatable_and_matches_native_logits(prepared):
    from RL.grpo.objective import selected_logprobs
    from RL.modeling import frozen_checksums
    cfg, path, cp, contract = prepared
    model, _ = model_and_targets(settings=cfg.lora, adapter=cp/"adapter")
    before = {str(p):sha256(p) for p in path.rglob("*") if p.is_file()}
    state = {k:v.clone() for k,v in model.state_dict().items()}
    protocol = {"attention_policy":"auto","projection_block_size":2}
    ids = contract["input"]["eligible_group_ids"][:2]
    def score():
        with ProbeGroups(cfg,cp,contract,ids) as groups:
            return evaluate(model,groups,protocol)
    reports, identity, token_changes = score()
    assert score() == (reports,identity,token_changes)
    assert token_changes["all"]["token_count"] == sum(
        b["token_count"] for g in reports for b in g["branches"])
    assert token_changes["all"]["token_count"] == sum(
        g["token_changes"]["all"]["token_count"] for g in reports)
    with ProbeGroups(cfg,cp,contract,ids[:1]) as groups:
        group, _ = next(iter(groups))
    batch = group.branch_batch(0,"cpu")
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logp,mask = selected_logprobs(logits,batch["input_ids"],batch["labels"],batch["loss_mask"])
    assert reports[0]["branches"][0]["logprob_mean"] == pytest.approx(float(logp[mask].double().mean()),abs=1e-6)
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,state[k],atol=0,rtol=0)
    assert all(p.grad is None for p in model.parameters())
    assert before == {str(p):sha256(p) for p in path.rglob("*") if p.is_file()}
    assert reports[0]["group_id"] == ids[0]


def test_baseline_compares_only_compatible_fixed_probe():
    r = {"schema_version":1,"kind":"fixed_grpo_preference","probe_identity":{"groups":["a"],"protocol":{}},
         "checkpoint":{"global_step":2},"aggregate":{"preference_gap":3.,"missing":None},
         "groups":[{"group_id":"a","metrics":{"preference_gap":3.}}]}
    previous = copy.deepcopy(r)
    previous["aggregate"]["preference_gap"] = 1.
    previous["groups"][0]["metrics"]["preference_gap"] = 1.
    assert compare(r,previous)["aggregate_delta"] == {"preference_gap":2.,"missing":None}
    for field in ("groups","protocol"):
        bad=copy.deepcopy(previous); bad["probe_identity"][field] = "changed"
        with pytest.raises(ValueError,match="incompatible"):
            compare(r,bad)


def test_cli_validation_and_atomic_immutable_output(prepared,tmp_path):
    from RL.cli import register
    cfg,path,cp,contract=prepared
    parser=argparse.ArgumentParser(); register(parser.add_subparsers(dest="command"))
    args=parser.parse_args(["lora-probe","--checkpoint",str(cp),"--group-id","g0","--output","r.json","--dry-run"])
    assert args.command=="lora-probe" and args.dry_run
    root={"project_root":str(tmp_path)}
    result=run(root,checkpoint=cp,group_ids=["g0"],output="report.json",dry_run=True)
    assert not result["model_loaded"] and not (tmp_path/"report.json").exists()
    for ids in ([],["g0","g0"],["unknown"]):
        with pytest.raises(ValueError):
            run(root,checkpoint=cp,group_ids=ids,output="report.json",dry_run=True)
    with pytest.raises(ValueError,match="outside"):
        run(root,checkpoint=cp,group_ids=["g0"],output=path/"report.json",dry_run=True)
    with pytest.raises(ValueError,match="gpu"):
        run(root,checkpoint=cp,group_ids=["g0"],output="report.json")
    output=tmp_path/"report.json"
    _publish(output,{"value":1})
    with pytest.raises(FileExistsError): _publish(output,{"value":2})
    assert read_json(output)=={"value":1}
    assert not list(tmp_path.glob(".probe-*"))


def test_streaming_probe_reads_only_fixed_parent_and_no_progress(tmp_path,monkeypatch):
    from tests.test_grpo_streaming_resume import fixture, worker
    from RL.dataset import loader
    cfg=fixture(tmp_path,n=3)
    cfg.run.attempt_id="probe-stream"
    worker(cfg)
    from RL.config import attempt_directory
    path=attempt_directory(cfg); cp=path/"checkpoints/step-000001"
    contract=read_json(cp/"state.json")["contract"]
    ids=contract["input"]["eligible_group_ids"][:1]
    reads=[]; native=loader.native_tokens
    def counted(row,**kwargs):
        reads.append(row["group_id"]); return native(row,**kwargs)
    monkeypatch.setattr(loader,"native_tokens",counted)
    before={str(p):sha256(p) for p in path.rglob("*") if p.is_file()}
    with ProbeGroups(cfg,cp,contract,ids) as groups:
        loaded=list(groups)
        assert groups.stream._thread is None
        assert groups.stream.consumed_windows==groups.stream.prepared_windows==0
    assert set(reads)==set(ids)
    assert loaded[0][0].group_id==ids[0]
    assert len(loaded[0][1])==loaded[0][0].group_size
    assert before=={str(p):sha256(p) for p in path.rglob("*") if p.is_file()}


def test_legacy_stream_scan_uses_frozen_values_and_rejects_tampering(tmp_path):
    from RL.grpo.data import prepare
    from RL.grpo.reference import _write_reference, ReferenceIdentity
    cfg=config(tmp_path); dataset(tmp_path,"a",range(5))
    raw,proof=prepare(cfg)
    from RL.grpo.data import tensor_groups
    groups=list(tensor_groups(raw))
    owner=tmp_path/"owner"; cp=owner/"checkpoints/step-000001"; cp.mkdir(parents=True)
    write_json(owner/"frozen-groups.json",raw)
    from RL.probe_data import ProbeGroups
    selected=[groups[-1].group_id]
    values={g.group_id:{bid:torch.full((len(g.completion_token_ids[i]),),-2.)
                       for i,bid in enumerate(g.branch_ids)} for g in groups}
    (cp/"grpo/reference/adapter").mkdir(parents=True)
    write_json(cp/"grpo/reference/adapter/identity.json",{})
    _,ref=_write_reference(cp,groups,proof,ReferenceIdentity("a","b","c").as_dict(),values)
    contract={"input":proof,"grpo":ref}
    with ProbeGroups(cfg,cp,contract,selected) as reader:
        loaded=list(reader)
    assert [g.group_id for g,_ in loaded]==selected
    assert tuple(loaded[0][0].advantages)==tuple(raw[-1]["frozen_advantages"])
    raw[0]["frozen_advantages"][0]+=1
    write_json(owner/"frozen-groups.json",raw)
    with ProbeGroups(cfg,cp,contract,selected) as reader:
        with pytest.raises(ValueError,match="checksum"):
            list(reader)


def test_reference_subset_rejects_wrong_token_identity(prepared):
    from RL.grpo.prepared import PreparedReader, resolve
    from RL.grpo.data import tensor_groups
    from RL.grpo.reference_artifact import ReferenceReader
    cfg,path,cp,contract=prepared
    prepared=PreparedReader(resolve(contract["prepared"]))
    reference=ReferenceReader(resolve(contract["grpo"]["reference_artifact"],"rl-reference"),prepared=prepared)
    group=next(tensor_groups([prepared.raw(prepared.ids[0])]))
    group.completion_token_ids[0][0]+=1
    with pytest.raises(ValueError,match="identity"):
        reference.for_group(group)


def test_service_json_checkpoint_comparison_and_mismatch(prepared,tmp_path,monkeypatch):
    cfg,path,cp,contract=prepared
    monkeypatch.setattr("RL.service._select_gpu",lambda config:{"physical_gpu":0,"test_cpu":True})
    monkeypatch.setattr("RL.probe._configure_numerics",lambda config:None)
    monkeypatch.setattr("RL.artifacts.base_identity",lambda *a,**kw:BASE)
    monkeypatch.setattr("RL.modeling.load_base",lambda config:(None,None))
    monkeypatch.setattr("RL.modeling.inject",lambda model,settings,adapter,expected_targets:
                        model_and_targets(settings=settings,adapter=adapter))
    root={"project_root":str(tmp_path)}
    first=run(root,checkpoint=cp,group_ids=["g0","g1"],gpu="0",output="first.json")
    report=read_json(first["output"])
    assert report["support"]["group_count"]==2
    assert report["checkpoint"]["global_step"]==1
    assert report["probe_sha256"]==hash_json(report["probe_identity"])
    assert report["token_changes"]["all"]["token_count"]==report["support"]["token_count"]
    assert report["token_changes"]["version"]==1
    assert all("token_changes" in g for g in report["groups"])
    assert report['training_schedule']['checkpoint_step'] == 1
    assert [g['group_id'] for g in report['training_schedule']['groups']] == ['g0', 'g1']
    same=run(root,checkpoint=cp,baseline_report="first.json",gpu="0",output="same.json")
    assert all(v==0 or v is None for v in same["comparison"].values())
    assert read_json(same["output"])["token_changes"]==report["token_changes"]
    # Reports predating the additive statistics remain valid baselines and keep their identity.
    legacy=copy.deepcopy(report)
    del legacy["token_changes"]
    for group in legacy["groups"]:
        del group["token_changes"]
    write_json(tmp_path/"legacy.json",legacy)
    legacy_result=run(root,checkpoint=cp,baseline_report="legacy.json",gpu="0",output="legacy-next.json")
    assert read_json(legacy_result["output"])["token_changes"]==report["token_changes"]
    assert legacy_result["probe_sha256"]==report["probe_sha256"]
    second=run(root,checkpoint=path/"checkpoints/step-000002",baseline_report="first.json",gpu="0",output="second.json")
    later=read_json(second["output"])
    assert later["probe_sha256"]==report["probe_sha256"]
    assert later["comparison"]["aggregate_delta"]["preference_gap"]==pytest.approx(
        later["aggregate"]["preference_gap"]-report["aggregate"]["preference_gap"])
    report["probe_identity"]["protocol"]["dtype"]="changed"
    write_json(tmp_path/"bad.json",report)
    with pytest.raises(ValueError,match="protocol"):
        run(root,checkpoint=cp,baseline_report="bad.json",gpu="0",output="rejected.json")
    assert not (tmp_path/"rejected.json").exists()


def test_runner_cli_dry_run_on_real_checkpoint_without_torch(prepared,tmp_path):
    import json
    import subprocess
    cfg,path,cp,contract=prepared
    command=[str(ROOT/".venvs/runner/bin/dycon"),"lora-probe","--checkpoint",str(cp),
             "--group-id","g0","--output",str(tmp_path/"dry.json"),"--dry-run"]
    completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=25)
    assert completed.returncode==0,completed.stderr[-2000:]
    assert json.loads(completed.stdout)["status"]=="metadata_validated"
    assert not (tmp_path/"dry.json").exists()


def test_streaming_8x16_probe_loads_all_frozen_parent_branches(tmp_path):
    import numpy as np
    from tests.test_grpo_execution import cfg32
    from RL.grpo.streaming import StreamingGroups, create_plan
    cfg=cfg32(tmp_path,n=8,streaming=True)
    stream=StreamingGroups(cfg,create_plan(cfg))
    class Source:
        def values(self,row,**kwargs): return np.full(row["completion_count"],-2,dtype=np.float32)
        def verify_unchanged(self): pass
    stream.source=Source()
    gid=stream.plan["groups"][0]["group_id"]
    group,reference=stream.load_group(gid)
    assert group.group_size==len(reference)==32
    assert tuple(group.advantages)==tuple(stream.plan["groups"][0]["frozen_advantages"])
    assert "execution_chunk" not in group.metadata
    assert stream.consumed_windows==stream.prepared_windows==0 and stream._thread is None
    with pytest.raises(ValueError,match="Unknown"):
        stream.load_group(gid+"-half-0")
    stream.close()


def test_token_change_quantiles_and_cancellation():
    metrics=token_change_metrics(torch.tensor([-2.,-1.,0.,1.,2.],dtype=torch.float64))
    assert metrics["logprob_change_mean"]==0
    assert metrics["abs_logprob_change_mean"]==pytest.approx(1.2)
    assert metrics["logprob_change_quantiles"]["p25"]==-1
    assert metrics["logprob_change_quantiles"]["p95"]==pytest.approx(1.8)
    assert metrics["abs_logprob_change_quantiles"]["p50"]==1
    assert metrics["top_abs_change"]["top_1pct"]=={"token_count":1,"abs_change_share":pytest.approx(1/3)}


def test_token_change_concentration_distinguishes_uniform_and_sparse_changes():
    uniform=token_change_metrics(torch.ones(100,dtype=torch.float64))
    sparse=token_change_metrics(torch.tensor([100.]+[0.]*99,dtype=torch.float64))
    assert uniform["logprob_change_mean"]==sparse["logprob_change_mean"]==1
    for percent in (1,5,10):
        assert uniform["top_abs_change"][f"top_{percent}pct"]["abs_change_share"]==pytest.approx(percent/100)
        assert sparse["top_abs_change"][f"top_{percent}pct"]["abs_change_share"]==1
    rounded=token_change_metrics(torch.ones(101))
    assert [row["token_count"] for row in rounded["top_abs_change"].values()]==[2,6,11]


def test_relative_probability_thresholds_are_strict_directional_and_overflow_safe():
    values=[math.log1p(.1),math.log1p(-.1),math.log1p(.5),math.log1p(-.5),0.,1000.,-1000.]
    metrics=token_change_metrics(torch.tensor(values,dtype=torch.float64))["relative_probability_change"]
    assert metrics["gt_10pct"]=={"increase_fraction":pytest.approx(2/7),
                                    "decrease_fraction":pytest.approx(2/7),"either_fraction":pytest.approx(4/7)}
    assert metrics["gt_50pct"]=={"increase_fraction":pytest.approx(1/7),
                                    "decrease_fraction":pytest.approx(1/7),"either_fraction":pytest.approx(2/7)}
    # Compare the log-space counts with direct relative probabilities for moderate values.
    changes=torch.tensor([-.8,-.2,-.05,0,.04,.2,.8],dtype=torch.float64)
    direct=token_change_metrics(changes)["relative_probability_change"]
    for percent in (10,50):
        expected=float((changes.expm1().abs()>percent/100).double().mean())
        assert direct[f"gt_{percent}pct"]["either_fraction"]==expected


def test_token_changes_pool_tokens_by_sign_and_handle_zero_or_missing_changes():
    pool=TokenChanges()
    pool.add(torch.tensor([2.]),1)
    pool.add(torch.zeros(9),1)
    other=TokenChanges()
    other.add(torch.tensor([-1.,-1.]),-1)
    other.add(torch.tensor([4.]),0)
    pool.extend(other)
    result=pool.summary()
    assert result["positive"]["logprob_change_mean"]==pytest.approx(.2)
    assert result["positive"]["token_count"]==10
    assert result["negative"]["logprob_change_mean"]==-1
    assert result["zero"]["token_count"]==1
    assert result["all"]["token_count"]==13
    assert result["all"]["logprob_change_mean"]==pytest.approx(4/13)
    zero=token_change_metrics(torch.zeros(10))
    assert zero["abs_logprob_change_mean"]==0
    assert zero["top_abs_change"]["top_10pct"]["abs_change_share"] is None
    assert zero["relative_probability_change"]["gt_10pct"]["either_fraction"]==0
    empty=TokenChanges().summary()
    assert empty["all"]["token_count"]==0
    assert empty["positive"]["logprob_change_mean"] is None
    assert empty["negative"]["logprob_change_quantiles"]["p50"] is None
    assert empty["zero"]["relative_probability_change"]["gt_50pct"]["either_fraction"] is None
    json.dumps([result,zero,empty],allow_nan=False)


@pytest.mark.parametrize("values",[[float("nan")],[float("inf")],[[1.]]])
def test_token_change_statistics_reject_invalid_values(values):
    with pytest.raises(ValueError,match="finite 1D"):
        token_change_metrics(torch.tensor(values))


def test_evaluate_collects_exact_global_tokens_without_extra_forward(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    rows={"p":torch.tensor([-1.,-3.]),"n":torch.tensor([-3.]),"z":torch.tensor([-2.])}
    def scorer(model,batch,block):
        calls.append(batch)
        return rows[batch]
    monkeypatch.setattr("RL.grpo.logprobs.model_target_logprobs",scorer)
    monkeypatch.setattr("RL.probe_data.group_identity",lambda group,ref:{"group_id":group.group_id})
    def group(gid,bids,advantages):
        return SimpleNamespace(group_id=gid,branch_ids=bids,advantages=advantages,
                               branch_batch=lambda i,device:bids[i])
    groups=[(group("a",["p","n"],[1.,-1.]),{"p":torch.tensor([-2.,-2.]),"n":torch.tensor([-2.])}),
            (group("b",["z"],[0.]),{"z":torch.tensor([-2.])})]
    reports,identity,stats=evaluate(torch.nn.Linear(1,1),groups,
                                    {"attention_policy":"auto","projection_block_size":2})
    assert calls==["p","n","z"]
    assert len(identity)==2
    assert stats["all"]["logprob_change_mean"]==-.25
    assert stats["all"]["abs_logprob_change_mean"]==.75
    assert stats["all"]["logprob_change_quantiles"]["p50"]==-.5
    assert reports[0]["token_changes"]["all"]["token_count"]==3
    assert stats["positive"]["token_count"]==2
    assert stats["negative"]["token_count"]==1
    assert stats["zero"]["token_count"]==1
    assert stats["all"]["top_abs_change"]["top_1pct"]["abs_change_share"]==pytest.approx(1/3)
    assert reports[0]["metrics"]["logprob_change"]==-.5  # Existing branch mean is preserved.
