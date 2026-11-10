"""Formal dispatcher: immutable inputs, bounded isolated child and atomic sidecar."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess

from core.io import hash_json, owned_path, read_json, sha256, write_json
from ..artifacts import directory, publication, require_space, seal, verify_local
from ..config import load
from ..loader import iter_groups
from .validation import effective_policy, root_sampling, DEFAULTS, UNSUPPORTED, ROOT_KEYS


def request_policy(row, source, sampling):
    value = read_json(owned_path(row["native_evidence"]["root"], row["native_evidence"]["ledger_ref"]))
    request = value.get("sampling_params") or value.get("sampling_parameters") or value.get("request_params") or {}
    allowed = set(DEFAULTS) | set(UNSUPPORTED) | set(ROOT_KEYS) | {"seed", "n", "extra_body", "prompt_logprobs"}
    # Some schemas store the original generation options at the top level.
    request = {**{k: v for k, v in value.items() if k in allowed}, **request}
    policy = effective_policy(source, sampling, request)
    policy["policy_sha256"] = hash_json(policy["parameters"])
    return policy


def prepare(dataset, root_config, replay_config):
    dataset = Path(dataset); verify_local(dataset, "rl-dataset")
    sampling = root_sampling(root_config)
    sources = {}; policies = {}; models = {}; counts = {"groups": 0, "branches": 0, "completion_tokens": 0, "max_sequence_length": 0}
    blocked = []; tokenizer = set()
    # Stream complete groups; no list of prompts/completions is retained.
    for group in iter_groups(dataset):
        counts["groups"] += 1
        for row in group["members"]:
            counts["branches"] += 1; counts["completion_tokens"] += row["completion_count"]
            length = row["prompt_count"] + row["completion_count"]
            counts["max_sequence_length"] = max(counts["max_sequence_length"], length)
            run = row["source_run_id"]
            if run not in sources: sources[run] = read_json(owned_path(dataset, f"source-evidence/{run}/resolved-config.json"))
            source = sources[run]
            model = source["model"]
            identity = {"model": model, "dtype": source.get("serving", {}).get("dtype")}
            model_key = hash_json(identity); models[model_key] = identity
            tokenizer.add(row["tokenizer_fingerprint"])
            try:
                policy = request_policy(row, source, sampling)
                policies[policy["policy_sha256"]] = policy
                if policy["parameters"]["min_tokens"] > row["completion_count"]: raise ValueError("Historical min_tokens exceeds trace length")
                if length > replay_config.runtime.max_model_len: raise ValueError("Sequence exceeds runtime.max_model_len; no truncation")
                if row["completion_count"] * 4 > replay_config.storage.max_shard_bytes: raise ValueError("Completion exceeds shard byte budget")
            except ValueError as exc:
                if len(blocked) < 20: blocked.append({"branch_sample_id": row["branch_sample_id"], "reason": str(exc)})
    if len(models) != 1 or len(tokenizer) != 1: blocked.append({"reason": "Replay requires one historical model/dtype/tokenizer cohort per dataset"})
    if not counts["branches"]: blocked.append({"reason": "No complete native groups are eligible"})
    return {"sampling": sampling, "models": models, "policies": policies, "counts": counts,
            "blocked": blocked, "tokenizers": sorted(tokenizer)}


def dry_run(dataset, root_config, replay_config):
    plan = prepare(dataset, root_config, replay_config)
    return {"status": "blocked" if plan["blocked"] else "dry-run", **plan["counts"],
            "sampling": plan["sampling"], "historical_models": list(plan["models"].values()),
            "policy_count": len(plan["policies"]), "blocked": plan["blocked"],
            "backend": replay_config.runtime.backend, "gpus": replay_config.runtime.gpus,
            "model_load": False, "receipt": False, "runtime_capabilities_verified": False,
            "historical_behavior_verified": False}


def dispatch(python, job, log, seconds, root):
    """Keep child logs on disk and kill the entire worker group on failure."""
    env = os.environ.copy(); env["PYTHONPATH"] = str(root)
    with Path(log).open("ab") as stream:
        process = subprocess.Popen([python, "-m", "RL.dataset.replay.worker", str(job)],
                                   cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
            if code: raise RuntimeError(f"Replay worker exited {code}; see {log}")
        finally:
            # vLLM may have spawned descendants, including on an error path.
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass


def replay_logprobs(cfg, replay_config_file, *, dry=False, recover=False):
    replay, source = load(replay_config_file, cfg["project_root"], replay=True)
    dataset = directory(replay.dataset.root, "rl-dataset", replay.dataset.id)
    if dry:
        if recover: raise ValueError("Use recover without dry-run to continue the frozen snapshot")
        return dry_run(dataset, cfg, replay)
    destination = directory(replay.storage.output_root, "rl-replay", replay.replay_id)
    stage = destination.with_name("." + replay.replay_id + ".staging")
    if recover:
        frozen = read_json(stage/"job.json")
        if frozen["replay_config"] != replay.model_dump(): raise ValueError("Recovery replay configuration differs")
        # Resume never picks up edited root sampling values.
        sampling_cfg = {"generation": frozen["plan"]["sampling"], "config_source_sha256": frozen["plan"]["sampling"].get("root_config_sha256")}
    else: sampling_cfg = cfg
    plan = prepare(dataset, sampling_cfg, replay)
    if plan["blocked"]: raise ValueError("Replay preflight blocked: " + str(plan["blocked"]))
    identity = {"replay_id": replay.replay_id, "dataset_receipt_sha256": sha256(dataset/"receipt.json"),
                "sampling": plan["sampling"], "models": plan["models"], "policies": plan["policies"],
                "runtime_options": replay.runtime.model_dump()}
    if recover and frozen["identity"] != identity: raise ValueError("Recovery model/policy/dataset identity differs")
    require_space(replay.storage.output_root, plan["counts"]["completion_tokens"]*4)
    with publication(replay.storage.output_root, "rl-replay", replay.replay_id, recover=recover, signature=identity) as (stage, destination):
        job = {"project_root": cfg["project_root"], "dataset": str(dataset), "stage": str(stage),
               "plan": plan, "identity": identity, "replay_config": replay.model_dump(), "recover": recover}
        write_json(stage/"job.json", job)
        (stage/"source-config.yaml").write_bytes(source.read_bytes())
        from hidden.backends.vllm.engine import gpu_reservation
        with gpu_reservation(cfg["project_root"], replay.runtime.gpus):
            dispatch(replay.runtime.python, stage/"job.json", stage/"worker.log", replay.runtime.subprocess_seconds, cfg["project_root"])
        seal(stage, "rl-replay", replay.replay_id, dataset_receipt_sha256=identity["dataset_receipt_sha256"], probability_semantics="processed_sampling_v1")
        verify_local(stage, "rl-replay")
    return {"status": "replay_estimate", "path": str(destination), "replay_id": replay.replay_id,
            **plan["counts"], "historical_behavior_verified": False}
