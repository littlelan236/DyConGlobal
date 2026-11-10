"""Private isolated worker. Invoked only by the formal rl-dataset dispatcher."""
from __future__ import annotations

from itertools import islice
from pathlib import Path
import sys

from core.io import hash_json, owned_path, read_json, sha256, write_json
from core.model_identity import verify_model
from hidden.backends.vllm.engine import configure_environment, verify_runtime
from ..config import ReplayConfig
from ..loader import iter_groups
from .artifacts import ReplayWriter
from .service import request_policy
from .validation import target_logprobs


def engine_options(model, runtime):
    return dict(model=model["model"]["path"], trust_remote_code=model["model"].get("trust_remote_code", False),
                dtype=model["dtype"], tensor_parallel_size=runtime.tensor_parallel_size,
                enable_trace_replay=True, logprobs_mode="processed_logprobs", generation_config="vllm",
                enforce_eager=True, async_scheduling=False, language_model_only=True,
                enable_prefix_caching=runtime.enable_prefix_caching,
                enable_chunked_prefill=runtime.enable_chunked_prefill,
                max_num_seqs=runtime.max_num_seqs, max_num_batched_tokens=runtime.max_num_batched_tokens,
                max_model_len=runtime.max_model_len, kv_cache_memory_bytes=runtime.kv_cache_memory_bytes,
                seed=0)


def sampling_options(row, policy):
    parameters = dict(policy["parameters"])
    if row.get("seed") is not None: parameters["seed"] = row["seed"]
    # min_tokens suppresses EOS/stop before its boundary. It must remain historical.
    if parameters["min_tokens"] > row["completion_count"]:
        raise ValueError("Historical min_tokens exceeds trace length")
    return {**parameters, "n": 1, "logprobs": 0, "prompt_logprobs": None,
            "trace_decode_token_ids": row["completion_token_ids"], "max_tokens": row["completion_count"],
            "detokenize": False}


def execute_batches(engine, sampling_class, rows, size, writer, policy_for, model_key):
    """Bound CPU output retention to max_num_seqs; write every completed branch."""
    iterator = iter(rows)
    while batch := list(islice(iterator, size)):
        policies = [policy_for(row) for row in batch]
        params = [sampling_class(**sampling_options(row, policy)) for row, policy in zip(batch, policies)]
        outputs = engine.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row in batch], params, use_tqdm=False)
        if len(outputs) != len(batch): raise ValueError("Replay batch output count mismatch")
        for row, policy, output in zip(batch, policies, outputs):
            values = target_logprobs(output, row["completion_token_ids"])
            writer.put(row, {"token_ids": list(output.outputs[0].token_ids)}, values,
                       policy_sha256=policy["policy_sha256"], model_sha256=model_key)


def run(job_path):
    job = read_json(job_path); replay = ReplayConfig.model_validate(job["replay_config"])
    root = Path(job["project_root"]); dataset = Path(job["dataset"]); stage = Path(job["stage"])
    if sha256(dataset/"receipt.json") != job["identity"]["dataset_receipt_sha256"]:
        raise ValueError("Dataset receipt changed before worker startup")
    model_key, model = next(iter(job["plan"]["models"].items()))
    cfg = {"project_root": str(root), "model": model["model"],
           "serving": {"python": ".venvs/easysteer/bin/python"},
           "hidden": {"vllm": {"python": replay.runtime.python, "gpus": replay.runtime.gpus}}}
    configure_environment(cfg)  # Must precede imports of Torch/vLLM.
    runtime = verify_runtime(root)
    identity = verify_model(cfg, deep=True)
    config = read_json(Path(model["model"]["path"])/"config.json")
    if config.get("quantization_config"): raise ValueError("Historical quantized model is not supported by this replay backend")
    from vllm import LLM, SamplingParams
    from vllm.config import ModelConfig
    from inspect import signature
    if "trace_decode_token_ids" not in signature(SamplingParams).parameters:
        raise ValueError("Pinned runtime has no native trace sampling support")
    if "enable_trace_replay" not in signature(ModelConfig).parameters:
        raise ValueError("Pinned runtime has no trace engine support")
    runtime.update(backend="native_vllm_trace_v1", logprobs_mode="processed_logprobs", enable_trace_replay=True,
                   engine_options=engine_options(model, replay.runtime), model_inventory_sha256=hash_json(identity["files"]),
                   replay_implementation={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")})
    if job["recover"] and (stage/"runtime-identity.json").exists():
        if read_json(stage/"runtime-identity.json") != runtime: raise ValueError("Recovery runtime/model implementation identity changed")
    else: write_json(stage/"runtime-identity.json", runtime)
    writer = ReplayWriter(stage, dataset, job["identity"], replay.storage.max_shard_bytes, recover=job["recover"])
    sources = {}
    def policy_for(row):
        run_id = row["source_run_id"]
        if run_id not in sources: sources[run_id] = read_json(owned_path(dataset, f"source-evidence/{run_id}/resolved-config.json"))
        policy = request_policy(row, sources[run_id], job["plan"]["sampling"])
        if policy["policy_sha256"] not in job["plan"]["policies"]: raise ValueError("Native sampling policy changed after preflight")
        return policy
    from types import SimpleNamespace
    vocab = config.get("text_config",config)["vocab_size"]
    native_model = SimpleNamespace(get_vocab_size=lambda:vocab)
    # Native compatibility validation precedes GPU/model construction.
    for group in iter_groups(dataset):
        for row in group["members"]:
            if max(row["prompt_token_ids"]+row["completion_token_ids"]) >= vocab:
                raise ValueError("Native token IDs exceed historical model vocabulary")
            params = SamplingParams(**sampling_options(row,policy_for(row)))
            params._validate_trace_replay(native_model,None)
            if row['branch_sample_id'] in writer.finished:
                writer.verify_completed(row,policy_sha256=policy_for(row)['policy_sha256'],model_sha256=model_key)
    rows = (row for group in iter_groups(dataset) for row in group["members"] if row["branch_sample_id"] not in writer.finished)
    # The parent holds the shared GPU reservation until all child processes exit.
    engine = LLM(**engine_options(model, replay.runtime))
    try:
        effective = engine.llm_engine.vllm_config.model_config
        if not effective.enable_trace_replay or effective.logprobs_mode != "processed_logprobs":
            raise ValueError("Engine did not activate trace replay and processed logprobs")
        execute_batches(engine, SamplingParams, rows, replay.runtime.max_num_seqs, writer, policy_for, model_key)
        writer.finish(job["plan"]["sampling"], job["plan"]["policies"], job["plan"]["models"], runtime)
    finally:
        shutdown = getattr(engine.llm_engine, "shutdown", None)
        if shutdown is not None: shutdown()


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("Private replay worker requires dispatcher job path")
    run(sys.argv[1])
