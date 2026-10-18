"""Native FP16 capture behavior and sampler diagnostic, isolated from HTTP.

Use --dycon-general-plugin to test the project's optional cache adapter.
GPU work is opt-in through this CLI. Probes introduce synchronization, so
repeat with --no-probe to exclude instrumentation masking a timing defect.
"""

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path


def install_probe(worker):
    import torch

    runner = worker.model_runner
    records = []
    worker._dycon_diag = {"label": "unarmed", "records": records}

    def summary(tensor):
        tensor = tensor.detach().float()
        finite = torch.isfinite(tensor)
        token198 = tensor[:, 198].cpu().tolist() if tensor.shape[-1] > 198 else []
        return {
            "shape": list(tensor.shape),
            "finite": int(finite.sum().item()),
            "nan": int(torch.isnan(tensor).sum().item()),
            "posinf": int(torch.isposinf(tensor).sum().item()),
            "neginf": int(torch.isneginf(tensor).sum().item()),
            "finite_norm": float(torch.where(finite, tensor, 0).norm().item()),
            "argmax": tensor.argmax(-1).cpu().tolist(),
            "token198": [
                value if value == value and abs(value) != float("inf") else str(value) for value in token198
            ],
        }

    original_logits = runner.model.compute_logits

    def compute_logits(hidden_states, *args, **kwargs):
        if len(records) < 160:
            records.append(
                {"stage": "hidden", "label": worker._dycon_diag["label"], "values": summary(hidden_states)}
            )
        return original_logits(hidden_states, *args, **kwargs)

    runner.model.compute_logits = compute_logits
    sampler = runner.sampler
    original_sample = sampler.sample

    def sample(logits, *args, **kwargs):
        row = None
        if len(records) < 160:
            mapping = args[2]
            bias = sampler.logit_bias_state
            row = {
                "stage": "sample",
                "label": worker._dycon_diag["label"],
                "raw": summary(logits),
                "indices": mapping.tolist(),
                "allowed_counts": bias.num_allowed_token_ids.np[mapping].tolist(),
                "allowed_first": bias.allowed_token_ids.gpu[mapping, 0].cpu().tolist(),
            }
        sampled, processed = original_sample(logits, *args, **kwargs)
        if row is not None:
            row["processed"] = summary(processed)
            row["sampled"] = sampled.cpu().tolist()
            records.append(row)
        return sampled, processed

    sampler.sample = sample
    return {
        "runner": type(runner).__name__,
        "model_state": type(runner.model_state).__name__,
        "uses_inputs_embeds": runner.uses_inputs_embeds,
        "supports_mm_inputs": runner.supports_mm_inputs,
        "cache_mode": worker.cache_config.mamba_cache_mode,
        "mamba_block_size": worker.cache_config.mamba_block_size,
        "use_kda_recoverssm": worker.cache_config.use_kda_recoverssm,
    }


def arm_probe(worker, label):
    worker._dycon_diag["label"] = label
    worker._dycon_diag["records"].clear()
    return True


def read_probe(worker):
    return worker._dycon_diag["records"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--apc", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dycon-general-plugin", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        VLLM_PLUGINS="dycon_capture" if args.dycon_general_plugin else "",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TOKENIZERS_PARALLELISM="false",
        VLLM_BATCH_INVARIANT="0",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1",
    )
    ninjas = list((Path.home() / ".cache/uv/archive-v0").glob("*/ninja-*.data/scripts/ninja"))
    if ninjas:
        os.environ["PATH"] = str(ninjas[0].parent) + os.pathsep + os.environ["PATH"]
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.capture import SelectSpec

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is 1+1?"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if isinstance(prompt, Mapping):
        prompt = prompt["input_ids"]
    prompt = list(prompt)
    assert prompt and all(isinstance(token, int) for token in prompt), repr(prompt)
    close = tok.encode("</think>", add_special_tokens=False)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text_config = getattr(config, "text_config", config)
    layer_count = text_config.num_hidden_layers
    sites = [0, layer_count // 2 - 1, layer_count]
    if args.preflight:
        print(json.dumps({"prompt": prompt, "close": close}))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {"args": vars(args), "cases": [], "started": time.time()}
    engine_args = dict(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=len(args.gpus.split(",")),
        enforce_eager=args.eager,
        async_scheduling=False,
        enable_prefix_caching=args.apc,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        kv_cache_memory_bytes=4 * 1024**3,
        language_model_only=args.language_model_only,
        generation_config="vllm",
        seed=42,
        enable_steer_vector=True,
        steer_algorithms="direct",
        steer_graph_mode="split" if args.eager else "in_graph",
        steer_vector_dtype="float32",
    )
    if not args.eager:
        engine_args["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        }
    if args.prefill_chunk:
        engine_args["long_prefill_token_threshold"] = args.prefill_chunk
    try:
        llm = LLM(**engine_args)
        if not args.no_probe:
            report["workers"] = llm.collective_rpc(install_probe)
        report["prompt"] = prompt
        report["close"] = close

        def case(name, capture, allowed=True):
            if capture is None:
                llm.collective_rpc("stop_capture", args=("tact_states",))
            else:
                ids = [1] if capture == "never" else [198, 271] + close
                llm.collective_rpc(
                    "start_capture",
                    args=("tact_states",),
                    kwargs={
                        "layers": sites,
                        "dtype": "float32",
                        "select": SelectSpec(generation_tokens=ids).to_wire(),
                        "budget_bytes": 67108864,
                        "device_budget_bytes": 67108864,
                    },
                )
            if not args.no_probe:
                llm.collective_rpc(arm_probe, args=(name,))
            params = SamplingParams(
                temperature=0,
                max_tokens=16 if allowed else 48,
                ignore_eos=True,
                allowed_token_ids=[198] if allowed else None,
            )
            outputs = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
            row = {
                "name": name,
                "tokens": list(outputs[0].outputs[0].token_ids),
                "text": outputs[0].outputs[0].text,
                "cached": getattr(outputs[0], "num_cached_tokens", 0),
                "status": llm.collective_rpc("capture_status", args=("tact_states",)),
                "probes": [] if args.no_probe else llm.collective_rpc(read_probe),
            }
            if capture is not None:
                raw = llm.collective_rpc("fetch_captured", args=("tact_states",), kwargs={"clear": True})
                row["fetch_nonempty"] = [bool(x) for x in raw]
            report["cases"].append(row)
            (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps({k: row[k] for k in ["name", "tokens", "cached"]}), flush=True)

        case("off_fixed", None)
        case("never_fixed", "never")
        case("on_fixed", "on")
        case("on_fixed_repeat", "on")
        case("off_fixed_after", None)
        case("off_natural", None, False)
        case("on_natural", "on", False)
        report["completed"] = True
        fixed = [case for case in report["cases"] if "fixed" in case["name"]]
        natural = [case for case in report["cases"] if "natural" in case["name"]]
        report["passed"] = (
            all(case["tokens"] == [198] * 16 for case in fixed)
            and natural[0]["tokens"] == natural[1]["tokens"]
            and all(
                not event["raw"]["nan"] and not event["raw"]["posinf"] and not event["raw"]["neginf"]
                for case in report["cases"]
                for worker in case["probes"]
                for event in worker
                if event["stage"] == "sample"
            )
        )
    finally:
        report["finished"] = time.time()
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit("Capture behavior or finite-logits gate failed; inspect report.json")


if __name__ == "__main__":
    main()
