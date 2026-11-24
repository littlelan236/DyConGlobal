"""Opt-in CUDA numerical, physical-memory and formal FSDP2 acceptance."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

pytestmark = pytest.mark.skipif(os.environ.get("DYCON_ADAM_OFFLOAD_GPU") != "1",
                               reason="Explicit idle GPU pair required")

ROOT = Path(__file__).parents[1]


def test_cuda_fifty_updates_events_reclaim_and_budget(tmp_path):
    code = r'''
import json,torch,os
from pathlib import Path
from RL.config import AdamMomentOffload,HostOffload,CudaAllocator
from RL.allocator import configure_cuda_allocator
from RL.host_offload import ManagedHostPool
from RL.optimizer_offload import StreamedAdamW
from RL.resources import host_memory
from tests.test_optimizer_offload import collect
host=HostOffload(enabled=True,allocator_budget_gib=.05,cache_limit_gib=0.,reserve_gib=1.,reserve_fraction=.01)
configure_cuda_allocator(CudaAllocator(),host_offload=host)
torch.set_num_threads(1);torch.cuda.set_device(0)
pool=ManagedHostPool(host)
cfg=AdamMomentOffload(mode='memory_pressure',tile_pair_mib=1,gpu_scratch_budget_mib=4,
 pinned_staging_budget_mib=4,host_state_budget_gib=.1,node_state_budget_gib=.2)
torch.manual_seed(17)
p=torch.nn.Parameter(torch.randn(150011,device='cuda'))
q=torch.nn.Parameter(p.detach().clone())
a=StreamedAdamW([{'params':[p],'weight_decay':.13}],lr=1e-3,settings=cfg,host_settings=host,pool=pool)
b=torch.optim.AdamW([q],lr=1e-3,weight_decay=.13,foreach=False)
a.initialize()
maximum={k:0. for k in ['parameter','exp_avg','exp_avg_sq']}
for step in range(50):
 if step%2==0:
  a.evict(2**60)
 else:
  a.restore_gpu(2**60,minimum_residency=False)
  assert a.cpu_bytes()==0
 p.grad=torch.randn_like(p);q.grad=p.grad.clone()
 a.step();b.step()
 torch.testing.assert_close(p,q,atol=0,rtol=0)
 for key in ['exp_avg','exp_avg_sq']:
  actual=collect(a,p,key);expected=b.state[q][key].cpu()
  torch.testing.assert_close(actual,expected,atol=0,rtol=0)
  maximum[key]=max(maximum[key],float((actual-expected).abs().max()))
 a.validate();torch.cuda.synchronize();pool.safe_boundary()
 assert pool.metrics()['host_pool_reserved_bytes']==0
assert a.metrics()['step_h2d_bytes']==a.metrics()['step_d2h_bytes']>0
a.close();pool.close()
# GPU-to-CPU-to-GPU physical release: no CPU stand-in for the target.
pool=ManagedHostPool(host)
p=torch.nn.Parameter(torch.zeros(8*2**20,device='cuda'))
a=StreamedAdamW([p],lr=1e-3,settings=cfg,host_settings=host,pool=pool)
a.initialize();torch.cuda.synchronize()
allocated=torch.cuda.memory_allocated();rss=host_memory()['VmRSS']
a.evict(2**60);torch.cuda.synchronize();pool.safe_boundary()
offallocated=torch.cuda.memory_allocated();offrss=host_memory()['VmRSS']
assert allocated-offallocated>=63*2**20
a.restore_gpu(2**60,minimum_residency=False);torch.cuda.synchronize();pool.safe_boundary()
restoredrss=host_memory()['VmRSS']
assert offrss-restoredrss>48*2**20 and a.cpu_bytes()==0
evidence={'max_abs_error':maximum,'allocated_before':allocated,'allocated_offloaded':offallocated,
 'rss_before':rss,'rss_offloaded':offrss,'rss_restored':restoredrss,'metrics':a.metrics(),'host':pool.metrics()}
a.close();pool.close()
Path(os.environ['ADAM_TEST_OUTPUT']).write_text(json.dumps(evidence))
'''
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=env.get("DYCON_ADAM_GPUS", "4,5").split(",")[0],
               ADAM_TEST_OUTPUT=str(tmp_path/"cuda-evidence.json"))
    result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize("backend",["reference_pytorch","fla_chunk"])
def test_formal_tiny_full_mixed_residency_resume_and_no_export(tmp_path,backend):
    from tests.lora_helpers import raw_model
    from tests.test_full_policy import full_values
    from tests.test_full_fsdp2_gpu import cli,checkpoint_tensors
    from core.io import read_json
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.set_num_threads(1)
    model_path=tmp_path/"tiny-base"
    raw_model().to(torch.bfloat16).save_pretrained(model_path)
    tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i):i for i in range(64)},unk_token="0")),
        unk_token="0",pad_token="0",eos_token="1")
    tok.save_pretrained(model_path)
    gpus=[int(x) for x in os.environ.get("DYCON_ADAM_GPUS","4,5").split(",")]
    raw=full_values()
    raw["run"].update(mode="smoke")
    raw["model"].update(path=str(model_path),dtype="bfloat16",gdn_backend=backend)
    raw["runtime"]={"gpus":gpus,"output_root":str(tmp_path),"distributed_timeout_seconds":90}
    raw["algorithm"]["dataset_root"]=str(tmp_path)
    raw["training"].update(learning_rate=1e-3,warmup_updates=2,checkpoint_every_updates=2)
    raw["training"]["memory"].update(activation_offload="memory_pressure",loss_chunk_tokens=16,
        memory_pressure={"device_headroom_gib":200.},
        host_offload={"enabled":True,"allocator_budget_gib":.1,"cache_limit_gib":0.,"reserve_gib":1.,"reserve_fraction":.01})
    raw["grpo"].update(top_k=1,k3_cap=1.,kl_beta=0.,ratio_guard={"enabled":False})
    raw["preflight"]={"verify_identity":True,"check_resources":True}
    def run(label,current,expected=0):
        current["run"]["attempt_id"]=label
        cfg=tmp_path/(label+".yaml");cfg.write_text(yaml.safe_dump(current))
        cli(tmp_path,label,["rl-train","--training-config",str(cfg)],timeout=240,expected=expected)
        return tmp_path/"RL/checkpoints"/label
    oracle=run("oracle",copy.deepcopy(raw))/"checkpoints/step-000002"
    enabled=copy.deepcopy(raw)
    enabled["training"]["memory"]["adam_moment_offload"]={"mode":"memory_pressure","tile_pair_mib":1,
        "gpu_scratch_budget_mib":4,"pinned_staging_budget_mib":4,"host_state_budget_gib":.1,
        "node_state_budget_gib":64.,"low_watermark_gib":94.6,"high_watermark_gib":94.8}
    # Tiny model naturally fits. A calibrated high threshold may place only part
    # on CPU depending on CUDA/NCCL baseline. Explicit migration coverage is in
    # the standalone test and mixed-rank 8x16 worker test below.
    enabled["training"]["memory"]["adam_moment_offload"].update(low_watermark_gib=.1,high_watermark_gib=.2)
    full_attempt=run("enabled",copy.deepcopy(enabled));full=full_attempt/"checkpoints/step-000002"
    def moment_values(cp,rank,name,key):
        from safetensors.torch import load_file
        folder=cp/f"ranks/{rank}";desc=read_json(folder/"manifest.json")["parameters"][name]["optimizer"][key]
        return torch.cat([load_file(str(folder/p["file"]))["value"] for p in desc["parts"]]) if desc["parts"] else torch.empty(0)
    def compare(a,b):
        for name,value in checkpoint_tensors(a).items():
            torch.testing.assert_close(value,checkpoint_tensors(b)[name],atol=0,rtol=0)
            for rank in range(2):
                for key in ("exp_avg","exp_avg_sq"):
                    torch.testing.assert_close(moment_values(a,rank,name,key),moment_values(b,rank,name,key),atol=0,rtol=0)
    compare(full,oracle)
    pause=copy.deepcopy(enabled);pause["validation"]["stop_after_update"]=1
    source=run("pause",pause)/"checkpoints/step-000001"
    resume=copy.deepcopy(enabled);resume["run"]["start"]={"mode":"resume","checkpoint":str(source)}
    final=run("resume",resume)/"checkpoints/step-000002";compare(final,full)
    # Explicit transition preserves source moments, step, warmup and data binding.
    legacy_pause=copy.deepcopy(raw);legacy_pause["validation"]["stop_after_update"]=1
    legacy_source=run("legacy-pause",legacy_pause)/"checkpoints/step-000001"
    migration=copy.deepcopy(enabled)
    migration["run"]["start"]={"mode":"resume","checkpoint":str(legacy_source)}
    from RL.config import Config,schedule_contract
    from core.io import hash_json
    migration["run"]["start"]["optimizer_backend_migration"]={
        "source_schedule_sha256":hash_json(read_json(legacy_source/"state.json")["contract"]["schedule"]),
        "target_schedule_sha256":hash_json(schedule_contract(Config.model_validate(migration))),
        "reason":"explicit tiny numerical migration acceptance"}
    migrated=run("migrated",migration)/"checkpoints/step-000002";compare(migrated,oracle)
    short=copy.deepcopy(enabled);short["validation"].update(stop_after_update=1,skip_checkpoint_export=True)
    attempt=run("no-export",short)
    assert not (attempt/"checkpoints").exists()
    meta=read_json(attempt/"attempt.json")
    assert meta["resumable"] is False and meta["final_checkpoint"] is None
    cli(tmp_path,"verify-no-export",["artifact","verify","no-export","--scope","rl-full-attempt","--storage-root",str(tmp_path)])
    probe=copy.deepcopy(short);probe["validation"]["capacity_microsteps"]=1
    attempt=run("capacity",probe)
    meta=read_json(attempt/"attempt.json");observation=read_json(attempt/"capacity.json")
    assert meta["capacity_only"] and meta["final_state"]["global_step"]==0
    assert observation["committed_updates"]==0 and observation["optimizer_step_applied"] is False
    assert observation["microsteps_executed"]==1 and len(observation["ranks"])==2
    assert not (attempt/"checkpoints").exists() and not (attempt/"metrics.jsonl").exists()
    cli(tmp_path,"verify-capacity",["artifact","verify","capacity","--scope","rl-full-attempt","--storage-root",str(tmp_path)])
    (tmp_path/"full-evidence.json").write_text(json.dumps({"backend":backend,"oracle":str(oracle),"enabled":str(full),"resume":str(final),"no_export":str(attempt)}))
