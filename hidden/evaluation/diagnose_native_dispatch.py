"""Trace real native batch dispatch with capture and steering disabled.

Only observation wrappers are installed after engine initialization. They do
not change scheduler decisions, graph descriptors, or forward arguments.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


def install_dispatch_probe(worker):
    import vllm.v1.worker.gpu.model_runner as module

    runner = worker.model_runner
    trace = {"batches": [], "current": None}
    worker._native_dispatch_trace = trace
    execute = runner.execute_model
    gather = runner.gather_batch_req_state
    dispatch = module.dispatch_cg_and_sync_dp
    context = module.set_forward_context

    def observed_execute(scheduler_output, *args, **kwargs):
        row = {
            "scheduled": dict(scheduler_output.num_scheduled_tokens),
            "compiled_calls": [],
            "forward_contexts": [],
        }
        trace["current"] = row
        try:
            result = execute(scheduler_output, *args, **kwargs)
            trace["batches"].append(row)
            return result
        finally:
            trace["current"] = None

    def observed_gather(*args, **kwargs):
        result = gather(*args, **kwargs)
        state, uniform = result
        row = trace["current"]
        if row is not None and state is not None:
            flags = state.is_prefilling_np.tolist()
            row.update(
                req_ids=list(state.req_ids),
                is_prefilling=flags,
                query_lengths=state.num_scheduled_tokens.tolist(),
                uniform_token_count=None if uniform is None else int(uniform),
                batch_kind="mixed"
                if any(flags) and not all(flags)
                else "prefill"
                if all(flags)
                else "decode",
            )
        return result

    def observed_dispatch(*args, **kwargs):
        result = dispatch(*args, **kwargs)
        row = trace["current"]
        if row is not None:
            row.update(
                cg_mode=result[0].cg_mode.name,
                need_eager=bool(kwargs.get("need_eager", False)),
                padded_tokens=result[0].num_tokens,
            )
        return result

    def observed_context(*args, **kwargs):
        row = trace["current"]
        if row is not None:
            mode = kwargs.get("cudagraph_runtime_mode")
            row["forward_contexts"].append(
                {
                    "cg_mode": mode.name if mode is not None else "NONE",
                    "skip_compiled": bool(kwargs.get("skip_compiled", False)),
                }
            )
        return context(*args, **kwargs)

    compiled_modules = []
    for name, model in runner.model.named_modules():
        compiled = getattr(model, "aot_compiled_fn", None)
        if compiled is None:
            continue
        compiled_modules.append(name)

        def observed_compiled(*args, _fn=compiled, _name=name, **kwargs):
            row = trace["current"]
            if row is not None:
                row["compiled_calls"].append(_name)
            return _fn(*args, **kwargs)

        model.aot_compiled_fn = observed_compiled
    runner.execute_model = observed_execute
    runner.gather_batch_req_state = observed_gather
    module.dispatch_cg_and_sync_dp = observed_dispatch
    module.set_forward_context = observed_context
    return {
        "runner": type(runner).__name__,
        "compiled_modules": compiled_modules,
        "resolved_graph_mode": runner.compilation_config.cudagraph_mode.name,
        "compilation_mode": runner.compilation_config.mode.name,
    }


def read_dispatch_probe(worker, clear=False):
    result = list(worker._native_dispatch_trace["batches"])
    if clear:
        worker._native_dispatch_trace["batches"].clear()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--attention-backend")
    parser.add_argument("--graph-mode", choices=["AUTO", "FULL", "FULL_DECODE_ONLY"], default="AUTO")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        VLLM_PLUGINS="",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1",
        TOKENIZERS_PARALLELISM="false",
    )
    os.environ["VLLM_CACHE_ROOT"] = str(Path(args.output).absolute() / "runtime-cache")
    if args.runtime_root:
        runtime = str(Path(args.runtime_root).absolute())
        sys.path.insert(0, runtime)
        os.environ["PYTHONPATH"] = runtime + os.pathsep + os.environ.get("PYTHONPATH", "")
    ninjas = list((Path.home() / ".cache/uv/archive-v0").glob("*/ninja-*.data/scripts/ninja"))
    if ninjas:
        os.environ["PATH"] = str(ninjas[0].parent) + os.pathsep + os.environ["PATH"]

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def prompt(long):
        value = tok.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Answer carefully. "
                    + ("Repository note: inspect tests before editing. " * 180 if long else ""),
                },
                {"role": "user", "content": "What is 1+1?"},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return list(value["input_ids"] if isinstance(value, Mapping) else value)

    prompts = [prompt(False), prompt(True)]
    runtime_path = Path(vllm.__file__).parent
    report = {
        "args": vars(args),
        "vllm_version": vllm.__version__,
        "vllm_path": str(runtime_path),
        "model_runner_sha256": hashlib.sha256(
            (runtime_path / "v1/worker/gpu/model_runner.py").read_bytes()
        ).hexdigest(),
        "prompt_lengths": list(map(len, prompts)),
        "capture_enabled": False,
        "steering_enabled": False,
        "dtype": "float16",
        "passed": False,
    }
    if args.preflight:
        print(json.dumps(report), flush=True)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    compilation = {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}
    if args.graph_mode != "AUTO":
        compilation["cudagraph_mode"] = args.graph_mode
    try:
        attention = {"backend": args.attention_backend} if args.attention_backend else {}
        llm = LLM(
            model=args.model,
            dtype="float16",
            tensor_parallel_size=len(args.gpus.split(",")),
            async_scheduling=False,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            language_model_only=True,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            max_model_len=4096,
            gpu_memory_utilization=0.6,
            kv_cache_memory_bytes=4 * 1024**3,
            generation_config="vllm",
            seed=42,
            compilation_config=compilation,
            attention_config=attention,
        )
        report["workers"] = llm.collective_rpc(install_dispatch_probe)
        report["runs"] = []
        for name in ["cold", "warm"]:
            started = time.perf_counter()
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in prompts],
                SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, allowed_token_ids=[198]),
                use_tqdm=False,
            )
            seconds = time.perf_counter() - started
            batches = llm.collective_rpc(read_dispatch_probe, kwargs={"clear": True})
            run = {
                "name": name,
                "seconds": seconds,
                "batches": batches,
                "tokens": [list(output.outputs[0].token_ids) for output in outputs],
                "cached_tokens": [getattr(output, "num_cached_tokens", 0) for output in outputs],
            }
            run["counts"] = [
                dict(
                    Counter(
                        (row.get("batch_kind", "unknown") + ":" + row.get("cg_mode", "unknown"))
                        for row in worker
                    )
                )
                for worker in batches
            ]
            report["runs"].append(run)
            (out / "report.json").write_text(json.dumps(report, indent=2))
            print(
                json.dumps({key: run[key] for key in ["name", "seconds", "counts", "cached_tokens"]}),
                flush=True,
            )
            assert all(tokens == [198] * 32 for tokens in run["tokens"])
        assert any(row.get("batch_kind") == "mixed" for row in report["runs"][0]["batches"][0]), (
            "No actual mixed batch was observed"
        )
        report["passed"] = True
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
