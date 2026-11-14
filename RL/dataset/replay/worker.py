"""Private raw trace worker, called only by the formal dispatcher."""
from __future__ import annotations

import sys
import signal
import time
from collections import deque
from hashlib import sha256 as hash_bytes
from pathlib import Path
from types import SimpleNamespace

from core.io import hash_json, read_json, sha256, write_json
from core.model_identity import verify_model
from hidden.backends.vllm.engine import configure_environment, verify_runtime
from ..config import ReplayConfig
from .artifacts import ReplayWriter
from .output import AsyncWriter
from .inputs import Prefetch
from .validation import raw_contract, raw_policy, target_logprobs


def engine_options(model, runtime):
    compilation = {"cudagraph_mode": runtime.cudagraph_mode}
    if runtime.cudagraph_capture_sizes is not None:
        compilation["cudagraph_capture_sizes"] = list(runtime.cudagraph_capture_sizes)
    return dict(model=model["model"]["path"], trust_remote_code=model["model"].get("trust_remote_code", False),
                dtype="half", tensor_parallel_size=runtime.tensor_parallel_size,
                enable_trace_replay=True, logprobs_mode="raw_logprobs", generation_config="vllm",
                enforce_eager=runtime.enforce_eager, compilation_config=compilation,
                async_scheduling=False, language_model_only=True,
                enable_prefix_caching=runtime.enable_prefix_caching,
                enable_chunked_prefill=runtime.enable_chunked_prefill,
                mamba_cache_mode=runtime.mamba_cache_mode,
                max_num_seqs=runtime.max_num_seqs, max_num_batched_tokens=runtime.max_num_batched_tokens,
                max_model_len=runtime.max_model_len, kv_cache_memory_bytes=runtime.kv_cache_memory_bytes,
                seed=0)


def sampling_options(row):
    return {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
            "logit_bias": None, "allowed_token_ids": None, "structured_outputs": None,
            "stop": [], "stop_token_ids": [], "min_tokens": 0, "ignore_eos": True,
            "n": 1, "logprobs": 0, "prompt_logprobs": None,
            "trace_decode_token_ids": row["completion_token_ids"], "max_tokens": row["completion_count"],
            "detokenize": False, "seed": 0}


def execute_batches(engine, sampling_class, rows, size, writer, policy_for, model_key):
    """Compatibility adapter; production uses continuous request coordination."""
    from itertools import islice
    iterator = iter(rows)
    while batch := list(islice(iterator, size)):
        params = [sampling_class(**sampling_options(row)) for row in batch]
        outputs = engine.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row in batch], params, use_tqdm=False)
        if len(outputs) != len(batch): raise ValueError("Replay batch output count mismatch")
        for row, output in zip(batch, outputs):
            policy = policy_for(row); values = target_logprobs(output, row["completion_token_ids"])
            writer.put(row, {"token_ids": list(output.outputs[0].token_ids)}, values,
                       policy_sha256=policy["policy_sha256"], model_sha256=model_key)


def _queue_metrics(engine):
    running = waiting = 0
    try:
        from vllm.v1.metrics.reader import Gauge
        for metric in engine.llm_engine.get_metrics():
            if isinstance(metric, Gauge) and metric.name == "vllm:num_requests_running": running += int(metric.value)
            if isinstance(metric, Gauge) and metric.name == "vllm:num_requests_waiting": waiting += int(metric.value)
    except Exception:
        pass
    return running, waiting


def execute_continuous(engine, sampling_class, groups, writer, policy_for, model_key, config, runtime_json, vocab_size):
    llm_engine = engine.llm_engine
    ready = deque(); group_remaining = {}; group_rows = {}; request_rows = {}
    group_leaders = {}
    submitted = completed = active_tokens = 0; metric_samples = []; start = time.monotonic(); first_result = None
    exhausted = False; next_group_id = 0
    leader_done = set(); group_admitted = set(); last_sample = start

    def add_group(group):
        nonlocal next_group_id
        gid = next_group_id; next_group_id += 1
        group_remaining[gid] = len(group["members"]); group_rows[gid] = group
        group_leaders[gid] = group["members"][0]["branch_sample_id"]
        for row in group["members"]: ready.append((gid, row))

    while ready or not exhausted or llm_engine.has_unfinished_requests():
        # Fill the engine's actual admission window, not a fixed generate() barrier.
        writer.check()
        deferred = None; skipped_group_ids = set(); searched = 0; batch_at_start = bool(request_rows)
        while ready and len(request_rows) < min(config.runtime.max_num_seqs, config.scheduling.max_inflight_requests):
            if config.scheduling.mode == "batch" and batch_at_start: break
            gid, row = ready.popleft()
            is_leader = row["branch_sample_id"] == group_leaders[gid]
            if config.scheduling.prefix_strategy == "leader_finished" and not is_leader and gid in group_admitted and gid not in leader_done:
                ready.append((gid, row)); searched += 1
                if searched >= len(ready): break
                continue
            cost = row["prompt_count"] + row["completion_count"]
            if active_tokens + cost > config.scheduling.max_inflight_tokens:
                deferred = (gid, row); break
            reqid = "raw-" + hash_bytes((config.replay_id + "\0" + row["branch_sample_id"]).encode()).hexdigest()[:40]
            if row["branch_sample_id"] in writer.finished:
                writer.verify_completed(row, policy_sha256=policy_for(row)["policy_sha256"], model_sha256=model_key)
                group_remaining[gid] -= 1; skipped_group_ids.add(gid)
                if is_leader: leader_done.add(gid)
                continue
            for tokens, expected in ((row["prompt_token_ids"], row["prompt_count"]), (row["completion_token_ids"], row["completion_count"])):
                if len(tokens) != expected or not tokens or any(type(t) is not int or t < 0 for t in tokens): raise ValueError("Invalid native token payload")
            params = sampling_class(**sampling_options(row))
            params._validate_trace_replay(SimpleNamespace(get_vocab_size=lambda: vocab_size), None)
            llm_engine.add_request(reqid, {"prompt_token_ids": row["prompt_token_ids"]}, params)
            request_rows[reqid] = (gid, row); submitted += 1; active_tokens += cost
            group_admitted.add(gid)
        if deferred is not None: ready.appendleft(deferred)
        for gid in skipped_group_ids:
            if group_remaining[gid] == 0:
                groups.release(group_rows.pop(gid)); group_remaining.pop(gid)
        if not exhausted and len(group_remaining) < config.scheduling.max_active_groups:
            group = groups.get(block=not request_rows and not ready)
            if group is groups.END: exhausted = True
            elif group is not None: add_group(group)
        outputs = llm_engine.step() if llm_engine.has_unfinished_requests() else []
        for output in outputs:
            reqid = output.request_id
            pair = request_rows.get(reqid)
            if pair is None: raise ValueError("Runtime returned unknown replay request ID")
            gid, row = pair
            if not output.finished: continue
            values = target_logprobs(output, row["completion_token_ids"])
            writer.put(row, {"token_ids": list(output.outputs[0].token_ids)}, values,
                       policy_sha256=policy_for(row)["policy_sha256"], model_sha256=model_key)
            del request_rows[reqid]; completed += 1; group_remaining[gid] -= 1
            if row["branch_sample_id"] == group_leaders[gid]: leader_done.add(gid)
            active_tokens -= row["prompt_count"] + row["completion_count"]
            if first_result is None: first_result = time.monotonic() - start
            if group_remaining[gid] == 0:
                groups.release(group_rows.pop(gid)); group_remaining.pop(gid)
        if time.monotonic() - last_sample >= 5:
            metric_samples.append({"elapsed_seconds": round(time.monotonic()-start, 3),
                                   "submitted": submitted, "completed": completed,
                                   "queue": _queue_metrics(engine),
                                   "memory_gib": round(groups.peak / 1024**3, 3)})
            last_sample = time.monotonic()
        if not outputs and not ready and not exhausted and not llm_engine.has_unfinished_requests(): time.sleep(.005)
    writer.close()
    runtime_json.update(queue_metrics={"submitted": submitted, "completed": completed,
                                      "first_result_seconds": first_result,
                                      "elapsed_seconds": time.monotonic()-start,
                                      "prefetch_peak_bytes": groups.peak,
                                      "writer_queue_peak": writer.peak,
                                      "writer_seconds": writer.seconds,
                                      "samples": metric_samples[-1000:]})


def run(job_path):
    def interrupted(signum, frame): raise InterruptedError(f"Raw replay worker interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    job = read_json(job_path); replay = ReplayConfig.model_validate(job["replay_config"])
    if not job.get('partition_worker') and (replay.runtime.parallel_workers > 1 or replay.resume_from):
        from .parallel import run as run_parallel
        return run_parallel(job_path)
    root = Path(job["project_root"]); dataset = Path(job["dataset"]); stage = Path(job["stage"])
    if sha256(dataset/"receipt.json") != job["identity"]["dataset_receipt_sha256"]:
        raise ValueError("Dataset receipt changed before worker startup")
    if job["plan"].get("cache") and sha256(Path(job["plan"]["cache"])/"receipt.json") != job["identity"]["cache_receipt_sha256"]:
        raise ValueError("Token cache receipt changed after preflight")
    if replay.selection.coverage_plan:
        from ..probability_artifacts import replay_selection
        frozen = replay_selection(replay.selection.coverage_plan, dataset)
        if frozen["summary"]["receipt_sha256"] != job["identity"]["coverage_plan_receipt_sha256"]:
            raise ValueError("Coverage plan receipt changed before worker startup")
    model_key, model = next(iter(job["plan"]["models"].items()))
    cfg = {"project_root": str(root), "model": model["model"], "serving": {"python": ".venvs/easysteer/bin/python"},
           "hidden": {"vllm": {"python": replay.runtime.python, "gpus": replay.runtime.gpus}}}
    configure_environment(cfg)
    runtime = verify_runtime(root)
    identity = verify_model(cfg, deep=True)
    model_config = read_json(Path(model["model"]["path"])/"config.json")
    if model_config.get("quantization_config"): raise ValueError("Quantized scoring model is not supported")
    from vllm import LLM, SamplingParams
    from vllm.config import ModelConfig
    from inspect import signature
    if "trace_decode_token_ids" not in signature(SamplingParams).parameters: raise ValueError("Pinned runtime has no native trace sampling support")
    if "enable_trace_replay" not in signature(ModelConfig).parameters: raise ValueError("Pinned runtime has no trace engine support")
    runtime.update(backend="native_vllm_trace_v1", logprobs_mode="raw_logprobs", enable_trace_replay=True,
                   engine_options=engine_options(model, replay.runtime), model_inventory_sha256=hash_json(identity["files"]),
                   cudagraph_requested=replay.runtime.cudagraph_mode,
                   cudagraph_requested_sizes=replay.runtime.cudagraph_capture_sizes,
                   prefix_cache={"enabled": replay.runtime.enable_prefix_caching,
                                 "mamba_cache_mode": replay.runtime.mamba_cache_mode},
                   replay_implementation={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")})
    if job["recover"] and (stage/"runtime-identity.json").exists():
        if read_json(stage/"runtime-identity.json") != runtime: raise ValueError("Recovery runtime/model implementation identity changed")
    else: write_json(stage/"runtime-identity.json", runtime)
    disk_writer = ReplayWriter(stage, dataset, job["identity"], replay.storage.max_shard_bytes,
                               recover=job["recover"], commit_every_branches=replay.storage.commit_every_branches,
                               commit_interval_seconds=replay.storage.commit_interval_seconds)
    writer = AsyncWriter(disk_writer, replay.scheduling.max_inflight_requests)
    policy = raw_policy()
    def policy_for(row): return policy

    vocab = model_config.get("text_config", model_config)["vocab_size"]
    # Read each ledger once in bounded metadata workers while the single cache
    # iterator stays confined to the producer thread.
    prefetch = Prefetch(dataset, Path(job["plan"]["cache"]) if job["plan"].get("cache") else None, replay, vocab_size=vocab,
                        group_ids=job.get('assigned_group_ids'), completed_ids=job.get('completed_branch_ids', ()))
    try:
        started = time.monotonic()
        options = engine_options(model, replay.runtime)
        options["worker_extension_cls"] = "RL.dataset.replay.observation.ReplayObservation"
        options["disable_log_stats"] = False
        engine = LLM(**options)
        try:
            effective = engine.llm_engine.vllm_config
            if not effective.model_config.enable_trace_replay or effective.model_config.logprobs_mode != "raw_logprobs":
                raise ValueError("Engine did not activate raw trace logprobs")
            compilation = effective.compilation_config
            effective_mode = getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode))
            if effective_mode != replay.runtime.cudagraph_mode:
                raise ValueError(f"Requested CUDA graph mode {replay.runtime.cudagraph_mode} resolved to {effective_mode}")
            if replay.runtime.enable_prefix_caching and not effective.cache_config.enable_prefix_caching:
                raise ValueError("Prefix caching request was not activated")
            runtime.update(cudagraph_effective=effective_mode,
                           cudagraph_capture_sizes=compilation.cudagraph_capture_sizes,
                           mamba_cache_mode=effective.cache_config.mamba_cache_mode,
                           prefix_cache_effective=effective.cache_config.enable_prefix_caching)
            runtime["startup_and_capture_seconds"] = time.monotonic() - started
            observations = engine.collective_rpc("replay_observe_start")
            if any(v["effective_mode"] != replay.runtime.cudagraph_mode for v in observations):
                raise ValueError("Worker graph mode differs from requested mode: " + str(observations))
            write_json(stage/"execution.json", runtime)
            groups = prefetch
            execute_continuous(engine, SamplingParams, groups, writer, policy_for, model_key, replay, runtime, vocab)
            runtime["dispatch_observations"] = engine.collective_rpc("replay_observe_snapshot")
            if replay.runtime.cudagraph_mode == "FULL" and any(not o["actual_full_dispatch"].get("FULL") for o in runtime["dispatch_observations"]):
                raise ValueError("No actual FULL graph dispatch observed; production target not validated")
            if job.get('partition_worker'):
                # The coordinator owns the one final artifact and full coverage
                # check. Child journals contain only this worker's new results.
                disk_writer.checkpoint('complete')
            else:
                writer.finish(raw_contract(), {policy["policy_sha256"]: policy}, job["plan"]["models"], runtime)
            write_json(stage/"execution.json", runtime)
        finally:
            shutdown = getattr(engine.llm_engine, "shutdown", None) or getattr(engine.llm_engine.engine_core, "shutdown", None)
            if shutdown is not None: shutdown()
    finally:
        try: writer.close()
        finally: prefetch.close()


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("Private replay worker requires dispatcher job path")
    run(sys.argv[1])
