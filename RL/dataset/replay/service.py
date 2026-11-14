"""Formal raw-only dispatcher: frozen inputs, isolated child, atomic publication."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

from core.io import hash_json, owned_path, read_json, sha256, write_json
from ..artifacts import directory, publication, require_space, seal, verify_local
from ..config import load
from .inputs import selected_groups
from .validation import RAW_SEMANTICS, raw_contract, raw_policy


def request_policy(row=None, source=None, sampling=None):
    """Historical transforms are provenance, never installed on a raw request."""
    return raw_policy()


def implementation_identity():
    base = Path(__file__).resolve().parents[1]
    files = list((base / "replay").glob("*.py")) + [base / name for name in ("config.py", "materialize.py", "loader.py")]
    return {str(p.relative_to(base)): sha256(p) for p in sorted(files)}


def prepare(dataset, root_config, replay_config):
    dataset = Path(dataset); verify_local(dataset, "rl-dataset")
    sources = {}; models = {}; tokenizers = set(); blocked = []
    counts = {"groups": 0, "branches": 0, "completion_tokens": 0, "max_sequence_length": 0}
    selection_hash = hashlib.sha256()
    for group in selected_groups(dataset, replay_config, with_tokens=replay_config.preflight.enabled):
        counts["groups"] += 1
        for row in group["members"]:
            selection_hash.update((row["branch_sample_id"] + "\n").encode())
            counts["branches"] += 1; counts["completion_tokens"] += row["completion_count"]
            length = row["prompt_count"] + row["completion_count"]
            counts["max_sequence_length"] = max(counts["max_sequence_length"], length)
            run = row["source_run_id"]
            if run not in sources: sources[run] = read_json(owned_path(dataset, f"source-evidence/{run}/resolved-config.json"))
            source = sources[run]
            model = {k: v for k, v in source["model"].items() if k != "verify_identity"}
            dtype = source.get("serving", {}).get("dtype")
            dtype = {"half": "float16", "fp16": "float16"}.get(dtype, dtype)
            identity = {"model": model, "dtype": dtype}
            models[hash_json(identity)] = identity; tokenizers.add(row["tokenizer_fingerprint"])
            reason = None
            if dtype != "float16": reason = "Raw replay requires the frozen float16 checkpoint cohort"
            if (source.get("steering") or {}).get("enabled"): reason = "Historical active steering model functions are unsupported"
            if any(model.get(k) or source.get("serving", {}).get(k) for k in ("adapter", "lora", "lora_request", "quantization", "speculative_config")):
                reason = "Adapters, quantization and speculative replay are unsupported"
            if length > replay_config.runtime.max_model_len: reason = "Sequence exceeds runtime.max_model_len; no truncation"
            if length > replay_config.scheduling.max_inflight_tokens: reason = "Sequence exceeds scheduling.max_inflight_tokens"
            if row["completion_count"] * 4 > replay_config.storage.max_shard_bytes: reason = "Completion exceeds shard byte budget"
            if reason and len(blocked) < 20: blocked.append({"branch_sample_id": row["branch_sample_id"], "reason": reason})
    if counts["branches"] and (len(models) != 1 or len(tokenizers) != 1): blocked.append({"reason": "Replay requires one historical model/dtype/tokenizer cohort"})
    if not counts["branches"] and not replay_config.selection.coverage_plan: blocked.append({"reason": "No complete native groups are eligible"})
    if replay_config.selection.coverage_plan:
        from ..probability_artifacts import replay_selection
        frozen = replay_selection(replay_config.selection.coverage_plan, dataset)["plan"]
        function = frozen["scoring_function_identity"]
        expected_dtype = next(iter(frozen["execution_cohorts"].values()))["collector_dtype"]
        if counts["branches"] and (function["adapter_fingerprint"] is not None
                or function["effective_model_overrides"] != {}
                or function["tokenizer_fingerprint"] not in tokenizers
                or any(m["model"]["fingerprint"] != function["checkpoint_fingerprint"]
                       or m["dtype"] != expected_dtype for m in models.values())):
            blocked.append({"reason": "Missing replay model function/dtype differs from frozen probability plan"})
    cache = None; cache_receipt = None
    if replay_config.token_cache:
        from ..materialize import CacheReader
        binding = replay_config.token_cache
        cache = directory(binding.root, "rl-cache", binding.id)
        CacheReader(cache, sha256(dataset / "receipt.json"), prompt_budget_bytes=0)
        cache_receipt = sha256(cache / "receipt.json")
    policy = raw_policy()
    return {"sampling": raw_contract(), "models": models, "policies": {policy["policy_sha256"]: policy},
            "counts": counts, "blocked": blocked, "tokenizers": sorted(tokenizers),
            "cache": str(cache) if cache else None, "cache_receipt_sha256": cache_receipt,
            "selection_sha256": selection_hash.hexdigest()}


def dry_run(dataset, root_config, replay_config):
    plan = prepare(dataset, root_config, replay_config)
    migration = None
    if replay_config.resume_from:
        from .parallel import source_path, check_source
        origin = source_path(replay_config.resume_from)
        source_job = read_json(origin / 'job.json')
        source_identity = source_job['identity']
        for key, expected in (('dataset_receipt_sha256', sha256(Path(dataset) / 'receipt.json')),
                              ('cache_receipt_sha256', plan['cache_receipt_sha256']),
                              ('selection_sha256', plan['selection_sha256']),
                              ('raw_contract_sha256', hash_json(raw_contract())), ('models', plan['models'])):
            if source_identity.get(key) != expected:
                raise ValueError('Resume source preflight identity differs: ' + key)
        # Also validate engine settings and the pinned runtime before a user
        # stops the old job; source state/cleanup is enforced at actual import.
        pin = Path(root_config['project_root']) / 'vendor/easysteer-runtime.json'
        expected = {**source_identity, 'group_ids': replay_config.selection.group_ids,
                    'runtime_pin_sha256': sha256(pin) if pin.exists() else None,
                    'coverage_plan_sha256': replay_config.selection.coverage_plan.sha256 if replay_config.selection.coverage_plan else None}
        check_source(origin, expected, replay_config)
        migration = {'id': replay_config.resume_from.id, 'path': str(origin),
                     'committed_branches': read_json(origin / 'journal.json').get('branch_count'),
                     'source_must_be_stopped_before_start': True}
    return {"status": "blocked" if plan["blocked"] else "dry-run", **plan["counts"],
            "sampling": plan["sampling"], "historical_models": list(plan["models"].values()),
            "policy_count": len(plan["policies"]), "blocked": plan["blocked"],
            "backend": replay_config.runtime.backend, "gpus": replay_config.runtime.gpus,
            "parallel_workers": replay_config.runtime.parallel_workers,
            "tensor_parallel_size": replay_config.runtime.tensor_parallel_size,
            "resume_from": migration,
            "cache_receipt_sha256": plan["cache_receipt_sha256"], "selection_sha256": plan["selection_sha256"],
            "model_load": False, "receipt": False, "full_preflight_enabled": replay_config.preflight.enabled,
            "runtime_capabilities_verified": False, "historical_behavior_verified": False}


def dispatch(python, job, log, seconds, root):
    """Own the worker process group until bounded shutdown has completed."""
    env = os.environ.copy(); env["PYTHONPATH"] = str(root)
    with Path(log).open("ab") as stream:
        process = subprocess.Popen([python, "-m", "RL.dataset.replay.worker", str(job)],
                                   cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
            if code: raise RuntimeError(f"Replay worker exited {code}; see {log}")
        finally:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                members = []
                for stat in Path("/proc").glob("[0-9]*/stat"):
                    try:
                        fields = stat.read_text().rsplit(")", 1)[1].split()
                        if int(fields[2]) == process.pid and fields[0] != "Z": members.append(stat.parent.name)
                    except (OSError, ValueError, IndexError): continue
                if not members: break
                time.sleep(.1)
            else: raise RuntimeError("Replay GPU process group cleanup not confirmed: " + str(members))
            write_json(Path(log).with_name("cleanup.json"), {"worker_pid": process.pid, "process_group_empty": True})


def replay_logprobs(cfg, replay_config_file, *, dry=False, recover=False):
    replay, source = load(replay_config_file, cfg["project_root"], replay=True)
    dataset = directory(replay.dataset.root, "rl-dataset", replay.dataset.id)
    if dry:
        if recover: raise ValueError("Use recover without dry-run")
        return dry_run(dataset, cfg, replay)
    if replay.selection.coverage_plan:
        from ..probability_artifacts import replay_selection
        frozen_plan = replay_selection(replay.selection.coverage_plan, dataset)
        declared = [s["spec"] for s in frozen_plan["plan"]["sources"] if s["spec"]["role"] == "backfill"]
        if declared != [{"kind": "replay", "id": replay.replay_id, "root": replay.storage.output_root, "stream": None, "role": "backfill"}]:
            raise ValueError("Replay ID/root differ from planned backfill")
        if not frozen_plan["missing"]:
            return {"status": "no_missing", "branches": 0, "model_load": False, "gpu_worker_started": False,
                    "coverage_plan_sha256": replay.selection.coverage_plan.sha256}
    destination = directory(replay.storage.output_root, "rl-replay", replay.replay_id)
    stage = destination.with_name("." + replay.replay_id + ".staging")
    if recover:
        frozen = read_json(stage / "job.json")
        if frozen["replay_config"].get("schema_version") != 2: raise ValueError("Processed staging is read-only; create a new raw replay ID")
        if frozen["replay_config"] != replay.model_dump(): raise ValueError("Recovery replay configuration differs")
    plan = prepare(dataset, {}, replay)
    if plan["blocked"]: raise ValueError("Replay preflight blocked: " + str(plan["blocked"]))
    pin = Path(cfg["project_root"]) / "vendor/easysteer-runtime.json"
    identity = {"schema_version": 2, "replay_id": replay.replay_id,
                "probability_semantics": RAW_SEMANTICS, "dataset_receipt_sha256": sha256(dataset / "receipt.json"),
                "cache_receipt_sha256": plan["cache_receipt_sha256"], "selection_sha256": plan["selection_sha256"],
                "group_ids": replay.selection.group_ids,
                "raw_contract_sha256": hash_json(raw_contract()), "models": plan["models"],
                "runtime_options": replay.runtime.model_dump(), "scheduling": replay.scheduling.model_dump(),
                "io": replay.io.model_dump(), "storage": replay.storage.model_dump(),
                "implementation": implementation_identity(), "runtime_pin_sha256": sha256(pin) if pin.exists() else None}
    if replay.selection.coverage_plan:
        identity.update(coverage_plan_id=replay.selection.coverage_plan.id,
                        coverage_plan_sha256=replay.selection.coverage_plan.sha256,
                        coverage_plan_receipt_sha256=frozen_plan["summary"].get("receipt_sha256"),
                        branch_keys=[[r["group_id"], r["branch_sample_id"]] for r in frozen_plan["missing"]])
    if replay.resume_from:
        identity['resume_from'] = replay.resume_from.model_dump()
    if recover and frozen["identity"] != identity: raise ValueError("Recovery model/runtime/dataset/implementation identity differs")
    require_space(replay.storage.output_root, plan["counts"]["completion_tokens"] * 4)
    previous = {}; interrupted = {"signal": None}
    def interrupt(signum, frame):
        interrupted["signal"] = signum
        raise InterruptedError(f"Replay interrupted by signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT): previous[sig] = signal.signal(sig, interrupt)
    try:
        with publication(replay.storage.output_root, "rl-replay", replay.replay_id, recover=recover, signature=identity) as (stage, destination):
            job = {"project_root": cfg["project_root"], "dataset": str(dataset), "stage": str(stage),
                   "plan": plan, "identity": identity, "replay_config": replay.model_dump(), "recover": recover}
            write_json(stage / "job.json", job)
            (stage / "source-config.yaml").write_bytes(source.read_bytes())
            write_json(stage / "state.json", {"state": "running", "pid": os.getpid()})
            (stage / 'cleanup.json').unlink(missing_ok=True)
            from hidden.backends.vllm.engine import gpu_reservation
            try:
                with gpu_reservation(cfg["project_root"], replay.runtime.gpus):
                    dispatch(replay.runtime.python, stage / "job.json", stage / "worker.log", replay.runtime.subprocess_seconds, cfg["project_root"])
                if replay.selection.coverage_plan:
                    current = replay_selection(replay.selection.coverage_plan, dataset)
                    if current["summary"]["receipt_sha256"] != identity["coverage_plan_receipt_sha256"]:
                        raise ValueError("Coverage plan receipt changed during replay")
                if interrupted["signal"] is not None: raise InterruptedError(f"Replay interrupted by signal {interrupted['signal']}")
                from .artifacts import check_replay
                write_json(stage / "state.json", {"state": "complete", "exit_code": 0})
                seal(stage, "rl-replay", replay.replay_id, dataset_receipt_sha256=identity["dataset_receipt_sha256"], probability_semantics=RAW_SEMANTICS)
                check_replay(stage, dataset_path=dataset, with_values=True)
            except BaseException as error:
                was_interrupted = interrupted["signal"] is not None or isinstance(error, (InterruptedError, KeyboardInterrupt))
                write_json(stage / "state.json", {"state": "interrupted" if was_interrupted else "failed", "error": str(error), "signal": interrupted["signal"], "exit_code": 143 if interrupted["signal"] == signal.SIGTERM else 130 if interrupted["signal"] == signal.SIGINT else 1})
                raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
    return {"status": "replay_estimate", "path": str(destination), "replay_id": replay.replay_id,
            **plan["counts"], "historical_behavior_verified": False, "probability_semantics": RAW_SEMANTICS}
