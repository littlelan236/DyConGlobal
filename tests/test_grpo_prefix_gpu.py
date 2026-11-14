"""Opt-in real 9B numerical/performance audits and formal CLI MIG lifecycle."""
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid

import pytest

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("DYCON_GRPO_PREFIX_GPU") != "1",
                              reason="real 9B prefix acceptance requires explicit opt-in")


def evidence():
    path = Path(os.environ["DYCON_GRPO_EVIDENCE"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_real_9b_all_gradients_and_branch_scaling():
    from RL.config import load
    from RL.service import _select_gpu
    config, _ = load(ROOT / "RL/experiments/grpo-offline-k3-smoke.yaml", ROOT)
    config.runtime.gpu = os.environ["DYCON_GRPO_GPU"]
    device = _select_gpu(config)
    assert device["device_kind"] == "mig"
    import torch
    from RL.grpo.logprobs import model_target_logprobs
    from RL.grpo.prefix import PrefixContext
    from RL.memory import attention_context
    from RL.modeling import inject, load_base, enable_checkpointing
    from RL.training import seed_everything
    from tests.test_grpo_prefix import group

    output = evidence()
    fp32 = os.environ.get("DYCON_GRPO_AUDIT_PRECISION") == "float32"
    accept_bf16_rounding = not fp32 and os.environ.get("DYCON_GRPO_ACCEPT_BF16_ROUNDING") == "1"
    if fp32:
        config.model.dtype = "float32"
    tolerances = {"logprob_atol": .02, "logprob_rtol": .002,
                  "loss_atol": .001, "loss_rtol": .002,
                  "gradient_atol": 2e-5, "gradient_rtol": 2e-3}
    if fp32:
        tolerances = {"logprob_atol": 2e-5, "logprob_rtol": 2e-5,
                      "loss_atol": 2e-6, "loss_rtol": 2e-5,
                      "gradient_atol": 2e-6, "gradient_rtol": 2e-4}
    (output / "tolerances.json").write_text(json.dumps(tolerances, indent=2))
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    seed_everything(42)
    started = time.perf_counter()
    model, _ = load_base(config)
    model, _ = inject(model, config.lora)
    enable_checkpointing(model)
    model.train()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.uniform_(-.002, .002)
    torch.cuda.synchronize()
    loading = time.perf_counter() - started
    assert torch.cuda.device_count() == 1
    from RL.grpo.objective import grpo_objective

    def run(row, enabled, anchors, capture=False):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        prefix = None
        values = []; total_loss = 0.
        try:
            with attention_context("auto" if fp32 else "flash_only"):
                if enabled:
                    prefix = PrefixContext(model, row.prompt_token_ids[0])
                for i, completion in enumerate(row.completion_token_ids):
                    current = (prefix.branch_logprobs(model, completion, 4) if enabled else
                               model_target_logprobs(model, row.branch_batch(i, "cuda"), 4))
                    result = grpo_objective(current[None], anchors[i][None].cuda(), [row.advantages[i]],
                        torch.ones_like(current[None], dtype=torch.bool))
                    loss = result.loss / row.group_size
                    if capture:
                        values.append(current.detach().cpu())
                    total_loss += float(loss.detach())
                    loss.backward()
                    del result, loss, current
                    if prefix:
                        prefix.after_branch_backward()
                if prefix:
                    prefix.finish()
            torch.cuda.synchronize()
            record = {"seconds": time.perf_counter() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "loss": total_loss}
            if prefix:
                record["prefix"] = dict(prefix.metrics)
            gradients = {n: p.grad.detach().float().cpu().clone() for n,p in model.named_parameters()
                         if p.requires_grad} if capture else None
        finally:
            if prefix:
                prefix.close()
            model.zero_grad(set_to_none=True)
        return record, values, gradients

    def anchor(row):
        with torch.no_grad(), attention_context("auto" if fp32 else "flash_only"):
            return [model_target_logprobs(model, row.branch_batch(i, "cuda"), 4).cpu() +
                    torch.linspace(-.2, .2, len(c)) for i,c in enumerate(row.completion_token_ids)]

    def compare(row, anchors, label):
        baseline, a, grads = run(row, False, anchors, True)
        shared, b, actual = run(row, True, anchors, True)
        failures = []
        log_error = max(float((x-y).abs().max()) for x,y in zip(a,b,strict=True))
        for x,y in zip(a,b,strict=True):
            try:
                torch.testing.assert_close(x,y,atol=tolerances["logprob_atol"],rtol=tolerances["logprob_rtol"])
            except AssertionError:
                failures.append("target logprobs")
        try:
            torch.testing.assert_close(torch.tensor(baseline["loss"]), torch.tensor(shared["loss"]),
                atol=tolerances["loss_atol"],rtol=tolerances["loss_rtol"])
        except AssertionError:
            failures.append("loss")
        rows = []
        assert grads.keys() == actual.keys()
        for name, ref in grads.items():
            value = actual[name]
            assert torch.isfinite(ref).all() and torch.isfinite(value).all(), name
            difference = (value-ref).abs()
            allowed = tolerances["gradient_atol"] + tolerances["gradient_rtol"] * ref.abs()
            count = int((difference > allowed).sum())
            rows.append({"name":name, "max_abs_error":float(difference.max()),
                         "max_relative_error_above_1e-7":float((difference/ref.abs().clamp_min(1e-7)).max()),
                         "relative_l2_error":float(difference.norm()/ref.norm().clamp_min(1e-12)),
                         "failed_elements":count,"elements":ref.numel(),
                         "reference_norm":float(ref.norm()),"actual_norm":float(value.norm())})
            if count:
                failures.append(name)
        summary = {"label":label,"P":len(row.prompt_token_ids[0]),"N":row.group_size,
                   "max_logprob_absolute_error":log_error,"baseline":baseline,"shared":shared,
                   "worst_parameter":max(rows,key=lambda r:r["max_abs_error"]),
                   "parameter_errors":rows,"failures":failures,"device":device,"loading_seconds":loading,
                   "strict_tolerance_passed":not failures,"bf16_rounding_accepted_by_user":accept_bf16_rounding,
                   "failed_gradient_elements":sum(r["failed_elements"] for r in rows),
                   "total_gradient_elements":sum(r["elements"] for r in rows),
                   "all_trainable_parameters_nonzero":all(r["reference_norm"]>0 and r["actual_norm"]>0 for r in rows)}
        (output / (label+"-numerics.json")).write_text(json.dumps(summary,indent=2))
        print(label, "max logprob",log_error,"worst gradient",summary["worst_parameter"],flush=True)
        assert not failures or accept_bf16_rounding, failures[:12]
        assert summary["all_trainable_parameters_nonzero"]

    compare(group(65,3), anchor(group(65,3)), "variable-completion")
    if fp32 and os.environ.get("DYCON_GRPO_RUN_PERFORMANCE") != "1":
        compare(group(5,32), anchor(group(5,32)), "branches-32")
        return  # FP32 is an independent mathematical audit, never BF16 performance acceptance.
    perf = []
    prompt_length = int(os.environ.get("DYCON_GRPO_PREFIX_LENGTH", "128"))
    assert 1 <= prompt_length <= 2048
    for n in [2,4,8,32]:
        # Fixed P and C=8, identical model and computation policy for both modes.
        row = group(prompt_length,n)
        completions = [[3+(i*7+j)%50 for j in range(8)] for i in range(n)]
        ids = torch.tensor([p+c for p,c in zip(row.prompt_token_ids,completions)])
        labels=ids.clone(); labels[:,:prompt_length]=-100
        from dataclasses import replace
        row = replace(row, completion_token_ids=completions,input_ids=ids,
                      labels=labels,loss_mask=labels!=-100,attention_mask=torch.ones_like(ids))
        anchor_started = time.perf_counter()
        anchors = anchor(row)
        anchor_seconds = time.perf_counter() - anchor_started
        compare(row,anchors,"branches-"+str(n))
        for enabled in [False,True]:
            run(row,enabled,anchors)  # Independent warmup, excluded.
            repetitions = [run(row,enabled,anchors)[0] for _ in range(3)]
            seconds = [r["seconds"] for r in repetitions]
            perf.append({"N":n,"mode":"boundary_grad" if enabled else "disabled",
                         "anchor_seconds": anchor_seconds,
                         "median_seconds":statistics.median(seconds),
                         "min_seconds":min(seconds),"max_seconds":max(seconds),
                         "repetitions":repetitions})
            (output / "performance.json").write_text(json.dumps({"device":device,"P":prompt_length,"C":8,
                "layer_checkpointing":True,"precision":"float32" if fp32 else "bfloat16",
                "attention":"auto" if fp32 else "flash_only","loading_seconds":loading,
                "anchor_preparation_excluded":True,"measurements":perf},indent=2))
            print("PERF",n,enabled,statistics.median(seconds),flush=True)


def test_real_mig_cli_two_updates_pause_resume_and_warm_start():
    import torch
    import yaml
    from core.io import read_json,sha256
    from tests.test_grpo_lifecycle import assert_tree_close
    output=evidence()
    storage = Path(os.environ.get("DYCON_GRPO_ARTIFACT_ROOT", str(output / "artifacts")))
    config=yaml.safe_load((ROOT/"RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    config["run"]["mode"]="smoke"
    config["runtime"].update(gpu=os.environ["DYCON_GRPO_GPU"],output_root=str(storage))
    config["grpo"]["prefix_reuse"]={"mode":"boundary_grad"}
    fp32 = os.environ.get("DYCON_GRPO_AUDIT_PRECISION") == "float32"
    config["model"]["dtype"] = "float32" if fp32 else "bfloat16"
    config["training"]["memory"].update(attention_policy="auto" if fp32 else "flash_only",profile_first_update=False)
    prefix="grpo-prefix-"+uuid.uuid4().hex[:10]
    timings={}
    def cli(label,value):
        value=copy.deepcopy(value); value["run"]["attempt_id"]=prefix+"-"+label
        file=output/(label+".yaml"); file.write_text(yaml.safe_dump(value))
        started=time.perf_counter()
        with (output/(label+".log")).open("w") as log:
            result=subprocess.run([str(ROOT/".venvs/rl/bin/dycon"),"lora-train","--training-config",str(file)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        assert result.returncode==0,(output/(label+".log")).read_text()[-7000:]
        timings[label]=time.perf_counter()-started
        path=storage/"RL/checkpoints"/value["run"]["attempt_id"]
        with (output/(label+"-verify.log")).open("w") as log:
            verified=subprocess.run([str(ROOT/".venvs/runner/bin/dycon"),"artifact","verify",value["run"]["attempt_id"],
                "--scope","lora-attempt","--storage-root",str(storage)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        assert verified.returncode==0,(output/(label+"-verify.log")).read_text()[-4000:]
        print("CLI",label,"passed",timings[label],flush=True)
        return path
    baseline=cli("continuous",config)
    paused_config=copy.deepcopy(config); paused_config["validation"]["stop_after_update"]=1
    paused=cli("paused",paused_config)
    resumed_config=copy.deepcopy(config)
    resumed_config["run"]["start"]={"mode":"resume","checkpoint":str(paused/"checkpoints/step-000001")}
    resumed=cli("resumed",resumed_config)
    assert read_json(baseline/"final-adapter.json")==read_json(resumed/"final-adapter.json")
    a=torch.load(baseline/"checkpoints/step-000002/training-state.pt",weights_only=True)
    b=torch.load(resumed/"checkpoints/step-000002/training-state.pt",weights_only=True)
    for key in ["optimizer","scheduler","rng"]:
        assert_tree_close(a[key],b[key])
    assert sha256(paused/"grpo/reference-logprobs.pt")==sha256(resumed/"grpo/reference-logprobs.pt")
    assert read_json(resumed/"attempt.json")["final_state"]==read_json(baseline/"attempt.json")["final_state"]
    warm_config=copy.deepcopy(config)
    warm_config["run"]["start"]={"mode":"warm_start","checkpoint":str(baseline/"adapter")}
    warm=cli("warm",warm_config)
    (output/"cli-acceptance.json").write_text(json.dumps({"attempts":dict(continuous=str(baseline),paused=str(paused),
        resumed=str(resumed),warm_start=str(warm)),"cli_seconds":timings,"adapter_identical":True,
        "optimizer_scheduler_rng_equal":True,"anchor_identical":True,"state_identical":True},indent=2))


def test_real_bf16_prefix_shape_diagnostics():
    """Measure causally identical prompt activations before changing any policy."""
    from RL.config import load
    from RL.service import _select_gpu
    config, _ = load(ROOT / "RL/experiments/grpo-offline-k3-smoke.yaml", ROOT)
    config.runtime.gpu = os.environ["DYCON_GRPO_GPU"]
    device = _select_gpu(config)
    import torch
    from RL.memory import text_backbone, attention_context
    from RL.modeling import inject, load_base
    from RL.qwen_boundary import segment
    from RL.training import seed_everything
    from tests.test_grpo_prefix import group
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    seed_everything(42)
    model, _ = load_base(config)
    model, _ = inject(model, config.lora)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if "lora_B" in name:
                p.uniform_(-.002,.002)
    model.train()
    text = text_backbone(model)
    row = group(65,3)
    captures = {}
    handles = []
    for index,layer in enumerate(text.layers):
        modules = {"input_norm":layer.input_layernorm, "post_attention_norm":layer.post_attention_layernorm,
                   "mlp":layer.mlp, "attention":getattr(layer,"linear_attn",getattr(layer,"self_attn",None))}
        if hasattr(layer,"linear_attn"):
            modules["qkv_projection"] = layer.linear_attn.in_proj_qkv
            modules["z_projection"] = layer.linear_attn.in_proj_z
            modules["b_projection"] = layer.linear_attn.in_proj_b
            modules["a_projection"] = layer.linear_attn.in_proj_a
            def inspect_core(module,args,output,index=index,attn=layer.linear_attn):
                captures[f"layer.{index}.gdn_core"] = args[0].detach().reshape(1,-1,attn.value_dim).float().cpu().clone()
                captures[f"layer.{index}.gdn_gated_norm"] = output.detach().reshape(1,-1,attn.value_dim).float().cpu().clone()
            handles.append(layer.linear_attn.norm.register_forward_hook(inspect_core))
        for name,module in modules.items():
            def hook(module,args,output,key=f"layer.{index}.{name}"):
                tensor = output[0] if isinstance(output,tuple) else output
                captures[key] = tensor.detach().float().cpu().clone()
            handles.append(module.register_forward_hook(hook))
    observations = {}
    try:
        with torch.no_grad(),attention_context("flash_only"):
            for index in [0,1,2,2]:
                captures = {}
                batch = row.branch_batch(index,"cuda")
                hidden = text(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],use_cache=False).last_hidden_state
                captures["final_hidden"] = hidden.detach().float().cpu().clone()
                observations["full-C"+str(len(row.completion_token_ids[index]))+("-repeat" if "full-C5" in observations and index==2 else "")] = captures
            captures = {}
            hidden,states = segment(model,torch.tensor([row.prompt_token_ids[0]],device="cuda"))
            captures["final_hidden"] = hidden.detach().float().cpu().clone()
            observations["shared-prefix"] = captures
            captures = {}
            hidden,_ = segment(model,torch.tensor([row.completion_token_ids[2][:-1]],device="cuda"),states,65)
            captures["final_hidden"] = hidden.detach().float().cpu().clone()
            observations["shared-suffix"] = captures
    finally:
        for handle in handles:
            handle.remove()
    reference=observations["full-C5"]
    result={"device":device,"P":65,"comparisons":{}}
    for label,samples in observations.items():
        rows=[]
        for key,ref in reference.items():
            if label == "shared-suffix":
                ref=ref[:,65:69]; value=samples[key]
            else:
                ref=ref[:,64]; value=samples[key][:,64]
            delta=(value-ref).abs()
            rows.append({"module":key,"max_abs_error":float(delta.max()),
                         "relative_l2":float(delta.norm()/ref.norm().clamp_min(1e-12)),
                         "different_elements":int((delta!=0).sum()),"elements":ref.numel()})
        result["comparisons"][label]=rows
        different=[r for r in rows if r["different_elements"]]
        print("BF16-SHAPE",label,"first differences",different[:5],"final",rows[-1],flush=True)
    (evidence()/"bf16-shape-diagnostics.json").write_text(json.dumps(result,indent=2))
    assert all(row["different_elements"]==0 for row in result["comparisons"]["full-C5-repeat"])
