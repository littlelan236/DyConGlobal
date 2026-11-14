"""Opt-in Qwen3.5-9B 8x16 public CLI acceptance; requires eight idle GPUs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest
import torch
import yaml

from core.io import hash_json,read_json,write_json
from RL.artifacts import base_identity
from RL.dataset.artifacts import directory,seal
from tests.test_grpo_execution import cfg32
from tests.test_grpo_external_reference import make_source,select
from tests.test_grpo_prepared import ROOT
from tests.test_grpo_lifecycle import assert_tree_close

pytestmark=pytest.mark.skipif(os.environ.get("DYCON_GRPO_8X16_GPU")!="1",
    reason="requires eight explicitly selected idle physical GPUs and local Qwen3.5-9B")


def _fixture(evidence,n,prompt_length=3):
    cfg=cfg32(evidence,n=n,streaming=True)
    selected=[int(x) for x in os.environ["DYCON_GRPO_GPUS"].split(',')]
    assert len(selected)==len(set(selected))==8
    cfg.runtime.gpus=selected
    cfg.model.path=os.environ.get("DYCON_8X16_MODEL","/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B")
    base=base_identity(cfg.model.path,ROOT,False)
    historical=Path(os.environ.get("DYCON_8X16_IDENTITY_ATTEMPT",
        "/data3/wangtianyi/DyConGlobal/RL/checkpoints/grpo-swe169-consensus-half-prefix-pool128-kl-off-step-stream-fresh-20261113-v10"))
    tokenizer=read_json(historical/"input-identity.json")["tokenizer_fingerprint"]
    native=directory(cfg.algorithm.dataset_root,"rl-dataset","a")
    meta=read_json(native/"dataset.json")
    meta["cohort"].update(model=base,tokenizer_fingerprint=tokenizer)
    write_json(native/"dataset.json",meta)
    rows=[json.loads(line) for line in (native/"branches.jsonl").read_text().splitlines()]
    from RL.dataset.schema import spans
    for row in rows:
        row["tokenizer_fingerprint"]=tokenizer
        ledger=Path(row["native_evidence"]["root"])/row["native_evidence"]["ledger_ref"]
        payload=read_json(ledger)
        payload["prompt_token_ids"]=[2,3]+[4]*(prompt_length-2)
        write_json(ledger,payload)
        row.update(prompt_count=prompt_length,prompt_sha256=hash_json(payload["prompt_token_ids"]),
                   mask_spans=spans(prompt_length,row["completion_count"]))
    (native/"branches.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows))
    seal(native,"rl-dataset","a")
    def scoring(manifest,branches,path):
        manifest["scoring_identity"].update(checkpoint_fingerprint=base["fingerprint"],tokenizer_fingerprint=tokenizer)
        for row in branches:
            row["native_request_id"]=row["branch_sample_id"]
            row["scoring_identity_sha256"]=hash_json(manifest["scoring_identity"])
    make_source(evidence/"probabilities",cfg,mutate=scoring)
    select(cfg,evidence/"probabilities")
    cfg.runtime.distributed_timeout_seconds=3600
    cfg.preflight.check_resources=True
    cfg.preflight.verify_checkpoint=True
    cfg.validation.max_sequence_length=max(256,prompt_length+4)
    cfg.training.memory.attention_policy="flash_only"
    return cfg


def _training(cfg,evidence,label,*,parent=None,stop=None):
    item=cfg.model_copy(deep=True)
    item.run.attempt_id="8x16-"+label+"-"+uuid.uuid4().hex[:10]
    item.run.start.mode="resume" if parent else "fresh"
    item.run.start.checkpoint=str(parent) if parent else None
    item.validation.stop_after_update=stop
    file=evidence/(item.run.attempt_id+".yaml")
    file.write_text(yaml.safe_dump(item.model_dump()))
    log=evidence/(item.run.attempt_id+".log")
    started=time.monotonic()
    with log.open('w') as out:
        proc=subprocess.run([str(ROOT/'.venvs/rl/bin/dycon'),'lora-train','--training-config',str(file)],
            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=7200)
    assert proc.returncode==0,log.read_text()[-10000:]
    elapsed=time.monotonic()-started
    path=Path(item.runtime.output_root)/'RL/checkpoints'/item.run.attempt_id
    verify_log=log.with_suffix('.verify.log')
    with verify_log.open('w') as out:
        proc=subprocess.run([str(ROOT/'.venvs/runner/bin/dycon'),'artifact','verify',item.run.attempt_id,
            '--scope','lora-attempt','--storage-root',item.runtime.output_root],cwd=ROOT,
            stdout=out,stderr=subprocess.STDOUT,timeout=300)
    assert proc.returncode==0,verify_log.read_text()[-8000:]
    return path,elapsed


@pytest.mark.parametrize('streaming',[True,False])
def test_real_eight_gpu_native_and_prepared_cross_round_tail_resume_and_artifact(tmp_path,streaming):
    evidence=Path(os.environ.get('DYCON_8X16_EVIDENCE',str(tmp_path)))/('streaming' if streaming else 'prepared')
    evidence.mkdir(parents=True,exist_ok=False)
    cfg=_fixture(evidence,9)
    cfg.runtime.data_preparation_async=streaming
    full,_=_training(cfg,evidence,'full')
    previous=None
    for step in range(1,5):
        current,_=_training(cfg,evidence,'resume-'+str(step),parent=previous,stop=step if step<4 else None)
        previous=current/f'checkpoints/step-{step:06d}'
        want=full/f'checkpoints/step-{step:06d}'
        a,b=[torch.load(p/'training-state.pt',map_location='cpu',weights_only=True) for p in (want,previous)]
        assert a['metadata']['state']==b['metadata']['state']
        assert len(b['rank_rng'])==8
        for key in ('optimizer','scheduler','rng','rank_rng'): assert_tree_close(a[key],b[key])
        assert read_json(want/'adapter/identity.json')==read_json(previous/'adapter/identity.json')
        from safetensors.torch import load_file
        assert_tree_close(load_file(str(want/'adapter/adapter_model.safetensors')),
                          load_file(str(previous/'adapter/adapter_model.safetensors')))
    rows=read_json(current/'grpo-report.json')['updates']
    assert [r['branch_count'] for r in rows]==[128,16,128,16]
    for row in rows:
        assert row['parallel']['world_size']==8
        assert len(row['prefix_reuse'])==row['parent_group_count']
        assert all(p['branch_count']==16 for p in row['prefix_reuse'])
    write_json(evidence/'acceptance.json',{'status':'passed','streaming':streaming,'full':str(full),
        'resumed':str(current),'source':'engineered sealed external raw probability fixture',
        'real_long_v10_prefix_capacity_verified':False,'performance_gate_verified':False})


def test_same_data_branch_budget_performance_twice_short_and_long_prefixes(tmp_path):
    if os.environ.get('DYCON_8X16_PERFORMANCE')!='1':
        pytest.skip('explicit performance acceptance flag required')
    import statistics
    evidence=Path(os.environ.get('DYCON_8X16_EVIDENCE',str(tmp_path)))/'performance'
    evidence.mkdir(parents=True,exist_ok=False)
    results=[]
    for length in [64,int(os.environ.get('DYCON_8X16_LONG_PREFIX','4096'))]:
        folder=evidence/f'prefix-{length}'
        folder.mkdir()
        cfg=_fixture(folder,16,prompt_length=length)
        for repeat in range(2):
            for mode in ('4x32','8x16'):
                item=cfg.model_copy(deep=True)
                if mode=='4x32':
                    item.runtime.gpus=item.runtime.gpus[:4]
                    item.grpo.group_batch_size=4
                    item.grpo.parallel_branch_chunk_size=None
                    item.grpo.parallel_branch_schedule=None
                    item.training.max_updates=4
                path,total=_training(item,folder,f'{mode}-repeat-{repeat}')
                rows=read_json(path/'grpo-report.json')['updates']
                elapsed=[r['seconds'] for r in rows[1:]]
                count=sum(r['branch_count'] for r in rows)
                assert count==512
                results.append({'mode':mode,'repeat':repeat,'prompt_length':length,'attempt':str(path),
                    'branch_count':count,'elapsed_including_startup_checkpoint_seconds':total,
                    'update_seconds_total':sum(r['seconds'] for r in rows),
                    'steady_state_branches_per_second':sum(r['branch_count'] for r in rows[1:])/sum(elapsed),
                    'steady_state_median_update_seconds':statistics.median(elapsed),
                    'steady_state_p90_update_seconds':float(torch.quantile(torch.tensor(elapsed),.9)),
                    'updates':rows,'capacity_scope':'synthetic lengths; separate v10 long-prefix acceptance required'})
                write_json(evidence/'performance.json',{'results':results,'quality_verified':False})
