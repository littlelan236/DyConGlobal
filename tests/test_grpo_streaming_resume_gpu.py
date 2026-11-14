"""Opt-in native streaming acceptance through the public 9B training CLI."""
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

from core.io import hash_json, read_json, write_json
from RL.artifacts import base_identity
from RL.dataset.artifacts import seal
from tests.test_grpo_prepared import ROOT, dataset, config
from tests.test_grpo_external_reference import make_source, select
from tests.test_grpo_lifecycle import assert_tree_close

pytestmark = pytest.mark.skipif(os.environ.get("DYCON_STREAMING_GPU") != "1",
                               reason="requires explicit idle physical GPUs and the local 9B model")


@pytest.mark.parametrize("world_size",[1,2])
def test_native_streaming_public_cli_full_pause_resume_twice(tmp_path,world_size):
    selected=[int(x) for x in os.environ["DYCON_GRPO_GPUS"].split(',')]
    assert len(selected)>=world_size
    evidence=Path(os.environ.get("DYCON_GRPO_EVIDENCE",str(tmp_path)))/f"world-{world_size}"
    evidence.mkdir(parents=True,exist_ok=False)
    cfg=config(evidence,updates=4)
    cfg.model.path=os.environ.get("DYCON_STREAMING_MODEL", "/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B")
    base=base_identity(cfg.model.path,ROOT,False)
    native=dataset(evidence,"native-stream",range(7))
    meta=read_json(native/"dataset.json")
    meta["cohort"]["model"]=base
    # The model/tokenizer fingerprints come from the installed frozen model.
    historical=Path(os.environ.get("DYCON_STREAMING_IDENTITY_ATTEMPT",
        "/data3/wangtianyi/DyConGlobal/RL/checkpoints/grpo-swe169-consensus-half-prefix-pool128-kl-off-step-stream-fresh-20261113-v10"))
    tokenizer=read_json(historical/"input-identity.json")["tokenizer_fingerprint"]
    meta["cohort"]["tokenizer_fingerprint"]=tokenizer
    write_json(native/"dataset.json",meta)
    rows=[json.loads(line) for line in (native/"branches.jsonl").read_text().splitlines()]
    for row in rows:row["tokenizer_fingerprint"]=tokenizer
    (native/"branches.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows))
    seal(native,"rl-dataset","native-stream")
    cfg.algorithm.dataset_id="native-stream"
    def native_source(manifest,rows,path):
        manifest["scoring_identity"]["checkpoint_fingerprint"]=base["fingerprint"]
        manifest["scoring_identity"]["tokenizer_fingerprint"]=tokenizer
        for row in rows:
            row["native_request_id"]=row["branch_sample_id"]
            row["scoring_identity_sha256"]=hash_json(manifest["scoring_identity"])
    make_source(evidence/"probabilities",cfg,mutate=native_source)
    select(cfg,evidence/"probabilities")
    cfg.runtime.data_preparation_async=True
    cfg.runtime.output_root=str(evidence/"artifacts")
    cfg.runtime.distributed_timeout_seconds=180
    cfg.runtime.gpu=selected[0] if world_size==1 else None
    cfg.runtime.gpus=selected[:2] if world_size==2 else None
    cfg.preflight.check_resources=True
    cfg.preflight.verify_checkpoint=True
    cfg.grpo.prefix_reuse.mode="boundary_grad"
    cfg.grpo.ratio_guard.enabled=False
    cfg.training.checkpoint_every_updates=1
    cfg.training.memory.attention_policy="flash_only"
    cfg.training.memory.profile_first_update=False
    cfg.validation.max_sequence_length=256
    prefix="native-stream-gpu-"+uuid.uuid4().hex[:10]
    timing={}
    def training(label,*,parent=None,stop=None):
        item=cfg.model_copy(deep=True)
        item.run.attempt_id=prefix+'-'+label
        item.run.start.mode="resume" if parent else "fresh"
        item.run.start.checkpoint=str(parent) if parent else None
        item.validation.stop_after_update=stop
        file=evidence/(label+'.yaml');file.write_text(yaml.safe_dump(item.model_dump()))
        log=evidence/(label+'.log')
        started=time.monotonic()
        with log.open('w') as out:
            proc=subprocess.run([str(ROOT/'.venvs/rl/bin/dycon'),'lora-train','--training-config',str(file)],
                cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420)
        assert proc.returncode==0,log.read_text()[-9000:]
        timing[label]=time.monotonic()-started
        path=evidence/'artifacts/RL/checkpoints'/item.run.attempt_id
        with (evidence/(label+'-verify.log')).open('w') as out:
            proc=subprocess.run([str(ROOT/'.venvs/runner/bin/dycon'),'artifact','verify',item.run.attempt_id,
                '--scope','lora-attempt','--storage-root',item.runtime.output_root],
                cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=120)
        assert proc.returncode==0,(evidence/(label+'-verify.log')).read_text()[-6000:]
        print(f'world={world_size} {label} passed ({timing[label]:.1f}s)',flush=True)
        return path
    full=training('full')
    paused=training('paused',stop=1)
    resumed=training('resumed',parent=paused/'checkpoints/step-000001',stop=2)
    again=training('again',parent=resumed/'checkpoints/step-000002')
    def compare(left,right,step):
        a,b=[p/f'checkpoints/step-{step:06d}' for p in (left,right)]
        pa,pb=[torch.load(p/'training-state.pt',map_location='cpu',weights_only=True) for p in (a,b)]
        assert pa['metadata']['state']==pb['metadata']['state']
        for key in ('optimizer','scheduler','rng',*(['rank_rng'] if world_size==2 else [])):
            assert_tree_close(pa[key],pb[key])
        from safetensors.torch import load_file
        sa,sb=[load_file(str(p/'adapter/adapter_model.safetensors')) for p in (a,b)]
        for key in sa:
            torch.testing.assert_close(sa[key],sb[key],atol=2e-5,rtol=2e-3)
    compare(full,resumed,2)
    compare(full,again,3);compare(full,again,4)
    a,b=[read_json(p/'grpo-report.json')['updates'] for p in (full,again)]
    for x,y in zip(a,b):
        for key in ('loss','policy_loss','ratio_mean','gradient_norm'):
            assert x[key]==pytest.approx(y[key],abs=.05,rel=.01)
    if world_size==2:
        assert b[-1]['parallel']['ranks'][1]['group_count']==0
    write_json(evidence/'acceptance.json',{'status':'passed','world_size':world_size,'gpus':selected[:world_size],
        'full':str(full),'paused':str(paused),'resumed':str(resumed),'again':str(again),'timing_seconds':timing,
        'native_streaming':True,'source':'engineered sealed raw external probability fixture',
        'real_long_context_capacity_verified':False})
