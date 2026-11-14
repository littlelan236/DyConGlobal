"""Opt-in real Qwen3.5-9B prepared/exact/append acceptance through public CLI."""
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

from core.io import read_json,write_json,sha256

pytestmark=pytest.mark.skipif(os.environ.get("DYCON_GRPO_PREPARED_GPU")!="1",
                            reason="requires explicit idle-GPU real Qwen3.5-9B acceptance")


def test_public_cli_real_9b_prepared_exact_and_append(tmp_path,monkeypatch):
    from tests import test_grpo_prepared as fixtures
    from tests.test_grpo_lifecycle import assert_tree_close
    from RL.artifacts import base_identity
    root=Path(__file__).parents[1]
    evidence=Path(os.environ.get("DYCON_GRPO_EVIDENCE",str(tmp_path)))
    evidence.mkdir(parents=True,exist_ok=True)
    cfg=fixtures.config(evidence)
    cfg.model.path=os.environ.get("DYCON_GRPO_MODEL",cfg.model.path)
    cfg.runtime.gpu=int(os.environ.get("DYCON_GRPO_GPU","1"))
    cfg.runtime.output_root=str(evidence/"artifacts")
    cfg.preflight.check_resources=True
    cfg.preflight.verify_checkpoint=True
    base=base_identity(cfg.model.path,root,False)
    monkeypatch.setattr(fixtures,"BASE",base)
    fixtures.dataset(evidence,"a",range(5))
    fixtures.dataset(evidence,"b",range(8))
    prefix="prepared-gpu-"+uuid.uuid4().hex[:8]
    timings={}
    def cli(label,args,timeout=300):
        start=time.perf_counter()
        with (evidence/(label+".log")).open("w") as output:
            completed=subprocess.run([str(root/".venvs/rl/bin/dycon"),*args],cwd=root,
                stdout=output,stderr=subprocess.STDOUT,timeout=timeout)
        assert completed.returncode==0,(evidence/(label+".log")).read_text()[-6000:]
        timings[label]=time.perf_counter()-start
    def training(label,config):
        config=config.model_copy(deep=True);config.run.attempt_id=prefix+"-"+label
        file=evidence/(label+".yaml");file.write_text(yaml.safe_dump(config.model_dump()))
        cli(label,["lora-train","--training-config",str(file)])
        path=Path(config.runtime.output_root)/"RL/checkpoints"/config.run.attempt_id
        cli(label+"-verify",["artifact","verify",config.run.attempt_id,"--scope","lora-attempt",
                             "--storage-root",config.runtime.output_root],timeout=120)
        return path
    full=training("full",cfg)
    paused_cfg=cfg.model_copy(deep=True);paused_cfg.validation.stop_after_update=1
    paused=training("paused",paused_cfg)
    checkpoint=paused/"checkpoints/step-000001"
    parent_receipt=sha256(checkpoint/"receipt.json")
    resumed_cfg=cfg.model_copy(deep=True)
    resumed_cfg.run.start.mode="resume";resumed_cfg.run.start.checkpoint=str(checkpoint)
    resumed=training("resumed",resumed_cfg)
    assert read_json(full/"final-adapter.json")==read_json(resumed/"final-adapter.json")
    left=torch.load(full/"checkpoints/step-000003/training-state.pt",weights_only=True,map_location="cpu")
    right=torch.load(resumed/"checkpoints/step-000003/training-state.pt",weights_only=True,map_location="cpu")
    for key in ("optimizer","scheduler","rng"):assert_tree_close(left[key],right[key])
    child=cfg.model_copy(deep=True)
    child.algorithm.dataset_id="b";child.training.max_updates=5
    child.run.start.mode="append_resume";child.run.start.checkpoint=str(checkpoint)
    extension_id=prefix+"-extension"
    child.run.start.extension_manifest=str(Path(child.runtime.output_root)/"RL/extensions"/extension_id/"manifest.json")
    file=evidence/"append-prepare.yaml";file.write_text(yaml.safe_dump(child.model_dump()))
    cli("append-prepare",["lora-prepare","--training-config",str(file),"--prepared-id",prefix+"-append-data",
        "--parent-checkpoint",str(checkpoint),"--extension-id",extension_id,"--append-seed","19","--shard-groups","2"])
    appended=training("appended",child)
    for scope,identity in (("rl-prepared",prefix+"-append-data"),("rl-extension",extension_id)):
        cli(scope+"-verify",["artifact","verify",identity,"--scope",scope,"--storage-root",cfg.runtime.output_root])
    report=read_json(appended/"grpo-report.json")
    assert report["state"]["global_step"]==5 and report["state"]["consumed_groups"]==8
    assert [r["group_count"] for r in report["updates"]]==[2,2,1,2,1]
    exact=read_json(resumed/"grpo-report.json")["preparation"]
    assert exact["prepared_cache_hit"] and exact["native_scan_groups"]==0
    assert exact["reference_new_groups"]==0
    assert report["preparation"]["reference_reused_groups"]==5
    assert report["preparation"]["reference_new_groups"]==3
    assert sha256(checkpoint/"receipt.json")==parent_receipt
    write_json(evidence/"acceptance.json",{"model":"Qwen3.5-9B","gpu":cfg.runtime.gpu,
        "full":str(full),"paused":str(paused),"resumed":str(resumed),"appended":str(appended),
        "state":report["state"],"timings":timings,"exact_preparation":exact,
        "append_preparation":report["preparation"],"parent_receipt_unchanged":True,
        "adapter_optimizer_scheduler_rng_equal":True,"context_capacity_acceptance":False})
