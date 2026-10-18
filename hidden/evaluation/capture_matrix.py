"""Native online all-site acceptance using native trace replay for controlled events.

Natural greedy requests independently test capture-on/off behavior. Controlled
traces exercise newline/close/follow-up and mixed batches, not task resolution.
"""

import argparse
from collections.abc import Mapping
import os
from pathlib import Path

import numpy as np

from core.config import resolve
from core.io import hash_json, read_json, write_json
from hidden.backends.vllm.capture import native_layers
from hidden.backends.vllm.online import capture_request_metadata, tokenizer_newline_inventory
from runtime.serving.capture import OnlineCaptureController
from runtime.serving.capture_worker import execution_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precision-only", action="store_true")
    args = p.parse_args()
    cfg = resolve(args.config, args.set)
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(map(str, cfg["serving"]["gpus"])),
                      VLLM_PLUGINS="", VLLM_USE_V2_MODEL_RUNNER="1",
                      VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false")
    ninjas = list((Path.home() / ".cache/uv/archive-v0").glob("*/ninja-*.data/scripts/ninja"))
    if ninjas:
        os.environ["PATH"] = str(ninjas[0].parent) + os.pathsep + os.environ["PATH"]
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_CACHE_ROOT"] = str(args.output / "runtime-cache")
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.capture import SelectSpec, match_capture_request_id
    from vllm.model_hooks.capture.serialization import assemble_captured

    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True)
    config = AutoConfig.from_pretrained(cfg["model"]["path"], local_files_only=True)
    text = getattr(config, "text_config", config)
    layers = list(range(text.num_hidden_layers))
    sites = native_layers(layers, text.num_hidden_layers)
    newline_ids, _ = tokenizer_newline_inventory(tok)
    close = tok.encode("</think>", add_special_tokens=False)
    selector = SelectSpec(generation_tokens=newline_ids + close).to_wire()
    tokenizer_hash = hash_json({"newline_ids": newline_ids, "close_ids": close})

    def prompt(long):
        value = tok.apply_chat_template([
            {"role": "system", "content": "Inspect the repository and explain the minimal test fix. " +
             ("Repository note: inspect tests before editing. " * 180 if long else "")},
            {"role": "user", "content": "What is 1+1? Answer briefly."},
        ], tokenize=True, add_generation_prompt=True, enable_thinking=True)
        return list(value["input_ids"] if isinstance(value, Mapping) else value)

    prompts = [prompt(False), prompt(True)]
    trace = tok.encode("Inspect the failing test.\n\nCheck the implementation.\n\nApply a minimal fix.\n\n",
                       add_special_tokens=False) + close + tok.encode("\nAnswer: 2.\n\n", add_special_tokens=False)
    report = {"passed": False, "dtype": cfg["serving"]["dtype"], "model_fingerprint": cfg["model"]["fingerprint"],
              "model_path": cfg["model"]["path"], "layers": layers, "native_sites": sites,
              "tp": cfg["serving"]["tensor_parallel_size"], "scenarios": [],
              "controlled_generation": "native_trace_decode_token_ids"}
    controller = OnlineCaptureController(args.output / "manifests", {
        "model_fingerprint": cfg["model"]["fingerprint"], "serving_dtype": cfg["serving"]["dtype"],
        "tensor_parallel_size": cfg["serving"]["tensor_parallel_size"],
        "export": {"root": str(args.output / "capture"), "queue_size": 256}})
    engine_args = dict(model=cfg["model"]["path"], dtype=cfg["serving"]["dtype"], language_model_only=True,
                       tensor_parallel_size=cfg["serving"]["tensor_parallel_size"],
                       async_scheduling=False, enable_prefix_caching=True, enable_chunked_prefill=True,
                       max_model_len=4096, max_num_seqs=4, max_num_batched_tokens=512,
                       gpu_memory_utilization=0.6, kv_cache_memory_bytes=4 * 1024**3,
                       generation_config="vllm", seed=42, enable_trace_replay=True,
                       enable_steer_vector=True, steer_algorithms="direct", steer_graph_mode="in_graph",
                       steer_vector_dtype="float32",
                       worker_extension_cls="runtime.serving.capture_worker.CaptureWorkerExtension",
                       compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE",
                                           "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]})
    if "Moe" in str(config.architectures):
        engine_args["kernel_config"] = {"moe_backend": "triton"}
    steering = False
    if cfg["steering"]["enabled"]:
        from vllm.model_hooks.steering.api import ApplySpec, SteeringSpec, VectorSpec
        from vllm.model_hooks.steering.payloads import DirectionVector

        vec = np.random.default_rng(42).normal(0, 0.1, text.hidden_size).astype(np.float32)
        layer = text.num_hidden_layers // 2 - 1
        np.save(args.output / "steering-vector.npy", vec)
        import base64

        wire = DirectionVector({layer: vec}).to_wire()
        for tensor in wire["tensors"].values():
            tensor["data"] = base64.b64encode(tensor["data"]).decode()
        write_json(args.output / "steering.json", {"layer": layer, "scale": 1.0,
                    "vector": "steering-vector.npy", "model_fingerprint": cfg["model"]["fingerprint"],
                    "steering": {"vectors": [{"data": wire, "layers": [layer], "scale": 1.0,
                                             "algorithm": "direct", "apply": {"generation": "all"}}]}})
        steering = SteeringSpec(vectors=[VectorSpec(data=DirectionVector({layer: vec}), layers=[layer],
                               apply=ApplySpec(generation="all"), scale=1.0)])
    llm = None
    intervals = {}
    try:
        llm = LLM(**engine_args)
        report["attested_model_dtype"] = str(llm.llm_engine.vllm_config.model_config.dtype)
        if args.precision_only:
            values = {}
            for name, choice in [("baseline", False), ("steering", steering)]:
                outputs = llm.generate([{"prompt_token_ids": prompts[0]}],
                    SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, logprobs=1),
                    steering=choice, use_tqdm=False)
                result = outputs[0].outputs[0]
                assert result.token_ids and all(np.isfinite(item.logprob) for row in result.logprobs for item in row.values())
                values[name] = {"tokens": list(result.token_ids), "selected_logprobs": [
                    row[token_id].logprob for token_id, row in zip(result.token_ids, result.logprobs, strict=True)]}
            report["precision_matrix"] = values
            report["passed"] = True
            return
        llm.collective_rpc("configure_capture_execution")

        def start():
            llm.collective_rpc("start_capture", args=("tact_states",), kwargs={
                "layers": sites, "select": selector, "dtype": "float32",
                "budget_bytes": 268435456, "device_budget_bytes": 268435456})

        def run(name, ids, controlled=True, enabled=True, stop_at_close=False):
            if enabled:
                start()
            else:
                llm.collective_rpc("stop_capture", args=("tact_states",))
            scripted = trace[:trace.index(close[0]) + 1] if stop_at_close else trace
            params = SamplingParams(temperature=0, max_tokens=len(scripted) if controlled else 64,
                                    ignore_eos=True, trace_decode_token_ids=scripted if controlled else None)
            outputs = llm.generate([{"prompt_token_ids": value} for value in ids], params,
                                   steering=steering, use_tqdm=False)
            raw = llm.collective_rpc("fetch_captured", args=("tact_states",), kwargs={"clear": True}) if enabled else []
            for worker in llm.collective_rpc("fetch_capture_execution"):
                for rid, values in worker.items():
                    intervals.setdefault(rid, []).extend(values)
            if enabled:
                tensors, meta, _ = assemble_captured(raw, tp_size=cfg["serving"]["tensor_parallel_size"])
                native_ids = set(meta[next(iter(meta))].req_ids) if meta else set()
                for i, output in enumerate(outputs):
                    matches = [rid for rid in native_ids if match_capture_request_id(rid, output.request_id)]
                    if len(matches) > 1:
                        raise ValueError("Native capture request identity is ambiguous")
                    native_id = matches[0] if matches else output.request_id
                    md = capture_request_metadata(f"{name}-{i}", f"diagnostic-task-{i}", name, f"{name}-{i}",
                        0, ["reasoning_newline", "think_close"], newline_token_ids=newline_ids,
                        close_token_ids=close, tokenizer_fingerprint=tokenizer_hash)
                    md.update(prompt_token_ids=list(output.prompt_token_ids), message_index=2,
                              request_sha256=hash_json(list(output.prompt_token_ids)))
                    controller.start(md)
                    controller.bind(native_id, md)
                    controller.record_output({"request_id": native_id,
                        "prompt_token_ids": list(output.prompt_token_ids),
                        "output_token_ids": list(output.outputs[0].token_ids),
                        "finish_reason": output.outputs[0].finish_reason,
                        "num_cached_tokens": getattr(output, "num_cached_tokens", 0) or 0})
                if meta:
                    labels = meta[next(iter(meta))]
                    for n, (rid, pos, token_id) in enumerate(zip(labels.req_ids, labels.positions.tolist(),
                                                               labels.token_ids.tolist(), strict=True)):
                        values = {f"layer_{site}": tensor[n].float().cpu().numpy() for site, tensor in tensors.items()}
                        assert all(np.isfinite(value).all() for value in values.values())
                        execution = execution_at(intervals, rid, pos)
                        controller.record(rid, values, anchor="think_close" if token_id in close else "reasoning_newline",
                            position=pos, absolute_position=pos, token_id=token_id,
                            phase="generation_input", generation_index=pos - len(controller.request_outputs[rid]["prompt_token_ids"]),
                            token_ref=[controller.request_outputs[rid]["rollout_id"], 0, "generation_input", pos, token_id],
                            graph_replays=int(execution["graph_execution"] == "full"), **execution)
                controller.queue.queue.join()
            row = {"name": name, "enabled": enabled,
                   "tokens": [list(output.outputs[0].token_ids) for output in outputs],
                   "finish_reasons": [output.outputs[0].finish_reason for output in outputs],
                   "cached_tokens": [getattr(output, "num_cached_tokens", 0) for output in outputs]}
            report["scenarios"].append(row)
            return row

        plain = run("natural-control", prompts[:1], False, False)
        captured = run("natural-online", prompts[:1], False, True)
        assert plain["tokens"] == captured["tokens"] and plain["finish_reasons"] == captured["finish_reasons"]
        llm.reset_prefix_cache()
        cold = run("mixed-cold", prompts)
        warm = run("mixed-warm", prompts)
        ended = run("close-without-followup", prompts[:1], stop_at_close=True)
        assert cold["tokens"] == warm["tokens"] == [trace, trace]
        assert ended["tokens"][0][-1] == close[0]
        report["workers"] = llm.collective_rpc("capture_status", args=("tact_states",))
        assert all(worker["graph_replays"] > 0 for worker in report["workers"])
        assert max(warm["cached_tokens"]) > 0
        report["passed"] = True
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        controller.close()
        write_json(args.output / "report.json", report)
        print({"passed": report["passed"], "sites": len(sites), "scenarios": len(report["scenarios"])}, flush=True)


if __name__ == "__main__":
    main()
