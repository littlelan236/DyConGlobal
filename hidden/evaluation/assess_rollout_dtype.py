"""Paired dtype assessment on real coding prompts, without executing tools.

Latency runs use native rollout graph selection with capture disabled. Accuracy
compares states captured during generation with prefill replay of exact emitted
token IDs. This is a model-serving benchmark, not a SWE-bench outcome estimate.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from core.config import resolve
from core.io import hash_json, read_json, write_json
from hidden.backends.vllm.engine import (
    configure_environment,
    engine_args,
    gpu_reservation,
    verify_runtime,
)


def vector_errors(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.sum(a * b, axis=-1) / np.maximum(norms, 1e-20)
    relative = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-20)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite reference/capture")
    return cosine, relative


def summarize_errors(actual, reference, rows):
    cosine, relative = vector_errors(actual, reference)
    result = {}
    for name, mask in [("all", np.ones(len(rows), bool))] + [
        (kind, np.array([r["kind"] == kind for r in rows])) for kind in sorted({r["kind"] for r in rows})
    ]:
        if not mask.any():
            continue
        c, e = cosine[mask].ravel(), relative[mask].ravel()
        result[name] = {
            "samples": int(mask.sum()),
            "site_vectors": len(e),
            "cosine_min": float(c.min()),
            "cosine_p50": float(np.median(c)),
            "relative_l2_max": float(e.max()),
            "relative_l2_p50": float(np.median(e)),
            "relative_l2_p95": float(np.quantile(e, 0.95)),
        }
    return result


def capture_generation(llm, prompts, layers, select, sampling):
    from easysteer.capture.fetch import fetch_pages
    from easysteer.capture.result import CaptureResult
    from vllm.capture import SelectSpec, validate_capture_topology

    stream = "tact_states"

    def rpc(method, *args, **kwargs):
        return llm.collective_rpc(method, args=args, kwargs=kwargs)

    tp = validate_capture_topology(rpc("capture_status", stream))
    before = rpc("capture_status", stream)
    try:
        if not all(
            rpc(
                "start_capture",
                stream,
                layers=layers,
                dtype="float32",
                select=SelectSpec.from_wire(select).to_wire(),
                budget_bytes=512 * 1024 * 1024,
                device_budget_bytes=512 * 1024 * 1024,
            )
        ):
            raise RuntimeError("Capture start failed")
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        status = rpc("capture_status", stream)
        if any(s["tokens_dropped"] or s["cache_elided_reqs"] for s in status):
            raise RuntimeError("Incomplete or cache-elided capture")
        tensors, meta, layouts = fetch_pages(rpc, stream, status, tp, 16 * 1024 * 1024, None)
        result = CaptureResult(tensors, meta, outputs, layouts=layouts, component=stream)
        mode = {
            key: max(s.get(key, 0) for s in status) - max(s.get(key, 0) for s in before)
            for key in ["eager_capture_forwards", "graph_replays"]
        }
        return result, mode
    finally:
        rpc("stop_capture", stream)


def generated_references(result, prompts, native, close_id):
    rows, states = [], []
    for i, output in enumerate(result.outputs):
        prompt = prompts[i]["prefix_token_ids"]
        if list(output.prompt_token_ids) != prompt:
            raise ValueError("Online prompt mismatch")
        generated = list(output.outputs[0].token_ids)
        all_ids = prompt + generated
        positions, token_ids = result.sample_positions(i), result.sample_token_ids(i)
        for j, (position, token) in enumerate(zip(positions, token_ids, strict=True)):
            if position >= len(all_ids) or all_ids[position] != token:
                raise ValueError("Online row label does not match exact emitted token IDs")
            prefix = all_ids[: position + 1]
            kind = (
                "think_open"
                if position == len(prompt) - 1
                else ("think_close" if token == close_id else "decode_token")
            )
            rows.append(
                {
                    "sample_id": f"{prompts[i]['sample_id']}:online:{position}",
                    "task_id": prompts[i]["task_id"],
                    "rollout_id": prompts[i]["rollout_id"],
                    "source_prompt": prompts[i]["sample_id"],
                    "kind": kind,
                    "prefix_token_ids": prefix,
                    "prefix_sha256": hash_json(prefix),
                    "position": position,
                    "token_id": token,
                }
            )
            states.append(np.stack([result.sample_rows(i, site)[j].float().numpy() for site in native]))
    if not rows:
        raise ValueError("Online generation yielded no captured states")
    return rows, np.stack(states)


def replay(llm, rows, native, reference):
    from vllm import SamplingParams

    result, mode = capture_generation(
        llm,
        [{"prompt_token_ids": r["prefix_token_ids"]} for r in rows],
        native,
        {"prompt_positions": [-1]},
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    values = []
    for i, row in enumerate(rows):
        if result.sample_positions(i) != [row["position"]] or result.sample_token_ids(i) != [row["token_id"]]:
            raise ValueError("Replay did not capture the exact reference token")
        values.append(np.stack([result.sample_rows(i, site)[0].float().numpy() for site in native]))
    values = np.stack(values)
    return {
        "errors": summarize_errors(values, reference, rows),
        "execution": mode,
        "cached_tokens": sum(o.num_cached_tokens or 0 for o in result.outputs),
        "prompt_tokens": sum(len(r["prefix_token_ids"]) for r in rows),
    }, values


def timed_generate(llm, prompts, sampling):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    output = llm.generate(prompts, sampling, use_tqdm=False)
    seconds = time.perf_counter() - start
    tokens = sum(len(o.outputs[0].token_ids) for o in output)
    return {
        "seconds": seconds,
        "completion_tokens": tokens,
        "tokens_per_second": tokens / seconds,
        "cached_tokens": sum(o.num_cached_tokens or 0 for o in output),
    }, output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bf16-reference", type=Path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tokens", type=int, default=512)
    a = p.parse_args()
    cfg = resolve(a.config, a.set)
    configure_environment(cfg)
    a.output.mkdir(parents=True, exist_ok=False)
    prompts = read_json(a.inputs)
    report = {
        "dtype": cfg["hidden"]["dtype"],
        "model_fingerprint": cfg["model"]["fingerprint"],
        "scope": "fixed real coding prompts; no tool execution or task outcome measurement",
        "tokens_per_request": a.tokens,
        "latency": {},
    }
    with gpu_reservation(cfg["project_root"], cfg["hidden"]["vllm"]["gpus"]):
        from core.model_identity import verify_model

        verify_model(cfg)
        report["runtime"] = verify_runtime(cfg["project_root"])
        from vllm import LLM, SamplingParams

        kwargs = engine_args(cfg)
        kwargs["enforce_eager"] = False  # diagnostic only: match ordinary rollout graph selection
        report["engine_args"] = kwargs
        start = time.perf_counter()
        llm = LLM(**kwargs)
        report["startup_seconds"] = time.perf_counter() - start
        config = llm.llm_engine.vllm_config
        report["actual_compilation"] = {
            "mode": str(config.compilation_config.mode),
            "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        }
        layer_count = config.model_config.hf_text_config.num_hidden_layers
        native = [0, layer_count // 2 - 1, layer_count]
        close = llm.get_tokenizer().encode("</think>", add_special_tokens=False)
        if len(close) != 1:
            raise ValueError("This diagnostic requires a single-token reasoning close marker")
        close_id = close[0]
        report["native_sites"] = native
        fixed = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=0,
            max_tokens=a.tokens,
            min_tokens=a.tokens,
            ignore_eos=True,
        )
        try:
            for concurrency in [1, 4, 12]:
                selected = [prompts[i] for i in range(concurrency)]
                batch = [{"prompt_token_ids": x["prefix_token_ids"]} for x in selected]
                assert llm.reset_prefix_cache()
                cold, _ = timed_generate(llm, batch, SamplingParams(temperature=0, max_tokens=1))
                timed_generate(llm, batch, fixed)  # warm decode kernels; not measured
                rounds = []
                for _ in range(a.rounds):
                    value, outputs = timed_generate(llm, batch, fixed)
                    if value["completion_tokens"] != concurrency * a.tokens:
                        raise ValueError("Fixed-token latency comparison must have identical output counts")
                    rounds.append(value)
                report["latency"][str(concurrency)] = {
                    "prompt_lengths": [len(x["prefix_token_ids"]) for x in selected],
                    "cold_prefill": cold,
                    "rounds": rounds,
                    "median_seconds": statistics.median(x["seconds"] for x in rounds),
                }
                write_json(a.output / "report.json", report)
            assert llm.reset_prefix_cache()
            selected = [prompts[i] for i in [0, 4, 8, 11]]
            online, mode = capture_generation(
                llm,
                [{"prompt_token_ids": x["prefix_token_ids"]} for x in selected],
                native,
                {
                    "prompt_positions": [-1],
                    "generation_positions": [0, 63, 127, 255, 511],
                    "generation_tokens": [close_id],
                },
                SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=0, max_tokens=1024),
            )
            rows, reference = generated_references(online, selected, native, close_id)
            write_json(a.output / "reference-prefixes.json", rows)
            np.savez_compressed(a.output / "online-reference.npz", hidden=reference)
            report["online"] = {
                "execution": mode,
                "samples": len(rows),
                "generated_lengths": [len(o.outputs[0].token_ids) for o in online.outputs],
                "kinds": {kind: sum(r["kind"] == kind for r in rows) for kind in {r["kind"] for r in rows}},
            }
            del online
            assert llm.reset_prefix_cache()
            report["cold_replay"], values = replay(llm, rows, native, reference)
            np.savez_compressed(a.output / "cold-replay.npz", hidden=values)
            report["warm_replay"], values = replay(llm, rows, native, reference)
            np.savez_compressed(a.output / "warm-replay.npz", hidden=values)
            if a.bf16_reference:
                old_rows = read_json(a.bf16_reference / "reference-prefixes.json")
                with np.load(a.bf16_reference / "online-reference.npz") as data:
                    old_reference = data["hidden"]
                assert llm.reset_prefix_cache()
                report["fp16_replay_of_bf16_online"], _ = replay(llm, old_rows, native, old_reference)
            report["completed"] = True
        finally:
            write_json(a.output / "report.json", report)
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
