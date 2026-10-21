"""Resumable extraction queue backed exclusively by one vLLM TP engine."""

from __future__ import annotations

import argparse
import copy
import os
import time
import traceback
from pathlib import Path

from core.io import hash_json, read_json, write_json
from core.locking import file_lock
from core.model_identity import verify_model
from enrichment.trajectory import enrich
from hidden.extract import extract_hidden

BATCH = "anchors-20261017"


def frozen_config(bundle, model, anchors):
    cfg = copy.deepcopy(bundle["configs"][model + "|" + ",".join(anchors)])
    expected = cfg.pop("config_hash")
    if hash_json(cfg) != expected:
        raise ValueError("Frozen extraction config checksum mismatch")
    cfg["config_hash"] = expected
    if cfg["hidden"]["adapter"] != "vllm_eager":
        raise ValueError("Extraction queue only supports vllm_eager")
    return cfg


def failure_scope(exc):
    row_errors = (
        "Prompt count mismatch:", "Actual prompt IDs differ", "No recoverable anchors",
        "Terminal message inside a model request",
    )
    return "trajectory" if isinstance(exc, ValueError) and any(
        str(exc).startswith(message) for message in row_errors
    ) else "system"


def update(root, owner, key, value):
    with file_lock(root / "state/product-manifests.lock"):
        path = root / owner / "manifest.json"
        manifest = read_json(path)
        manifest.setdefault("extraction_jobs", {})[key] = value
        write_json(path, manifest)


def _worker(root, snapshot, gpus):
    bundle = read_json(snapshot)
    models = bundle["models"]
    queue = root / "state" / BATCH / "queue.json"
    for key in bundle["configs"]:
        model, anchors = key.split("|", 1)
        frozen_config(bundle, model, anchors.split(","))

    from hidden.backends.vllm.engine import configure_environment, gpu_reservation

    first = frozen_config(bundle, next(iter(models)), ["think_open"])
    first["hidden"]["vllm"]["gpus"] = list(gpus)
    configure_environment(first)
    import gc

    from transformers import AutoTokenizer

    from hidden.adapters.vllm import load_adapter

    cached_model = None
    adapter = None
    tokenizer = None
    verified = set()
    try:
        with gpu_reservation(root, list(gpus)):
            while True:
                with file_lock(root / "state" / BATCH / "queue.lock"):
                    jobs = read_json(queue)
                    pending = [j for j in jobs if j["status"] in {"pending", "interrupted"}]
                    if not pending:
                        break
                    same = [j for j in pending if j["model"] == cached_model]
                    job = (same or pending)[0]
                    job.update(status="running", gpus=list(gpus), pid=os.getpid(), started=time.time())
                    write_json(queue, jobs)
                    job = dict(job)

                rid, model = job["rollout_id"], job["model"]
                dataset = BATCH + "-" + rid
                base = {
                    "batch": BATCH, "rollout_id": rid, "task_id": job["task_id"],
                    "run_id": job["run_id"], "trajectory_sha256": job["trajectory_sha256"],
                    "anchors": [{"name": a, "text": "<think>" if a == "think_open" else "</think>"}
                                for a in job["anchors"]],
                    "model": models[model], "gpus": list(gpus), "dataset_id": dataset,
                    "capture_id": dataset, "status": "running",
                    "config_snapshot": str(snapshot.relative_to(root)),
                }
                for owner in ["enrichment", "hidden"]:
                    update(root, owner, rid, base)
                status, error, fatal = "pending", None, False
                try:
                    cfg = frozen_config(bundle, model, job["anchors"])
                    cfg["hidden"]["vllm"]["gpus"] = list(gpus)
                    if model not in verified:
                        verify_model(cfg)
                        verified.add(model)
                    if cached_model != model:
                        if adapter is not None:
                            adapter.close()
                            adapter = None
                        tokenizer = AutoTokenizer.from_pretrained(
                            models[model]["path"], local_files_only=True,
                            trust_remote_code=cfg["model"]["trust_remote_code"],
                        )
                        cached_model = model
                    from application.execution.evidence import RunEvidence
                    source_run = job.get("source_run_id") or job.get("run_id")
                    if not source_run:
                        raise ValueError("Extraction job requires an explicit source run")
                    selected = RunEvidence(root, source_run).select([rid])
                    enrich(cfg, dataset, tokenizer, selection=selected)
                    update(root, "enrichment", rid, {**base, "status": "completed",
                                                       "receipt": "enrichment/outputs/" + dataset + "/receipt.json"})
                    if adapter is None:
                        adapter = load_adapter(cfg)
                    extract_hidden(cfg, dataset, dataset, adapter)
                    update(root, "hidden", rid, {**base, "status": "completed",
                                                  "receipt": "hidden/outputs/" + dataset + "/receipt.json"})
                    status = "completed"
                except Exception as exc:
                    error = traceback.format_exc()
                    scope = failure_scope(exc)
                    status, fatal = ("failed", False) if scope == "trajectory" else ("pending", True)
                    update(root, "hidden", rid, {**base, "status": status, "error": error})
                finally:
                    with file_lock(root / "state" / BATCH / "queue.lock"):
                        current = read_json(queue)
                        target = next(x for x in current if x["rollout_id"] == rid)
                        target.update(status=status, error=error, finished=time.time())
                        if error:
                            target.setdefault("failure_history", []).append({
                                "error": error, "gpus": list(gpus), "pid": os.getpid(),
                                "at": time.time(), "scope": "system" if fatal else "trajectory",
                            })
                        write_json(queue, current)
                if fatal:
                    raise RuntimeError("System error; worker stopped before claiming another job")
    finally:
        if adapter is not None:
            adapter.close()
        gc.collect()


def worker(root, snapshot, gpus):
    _worker(root, snapshot, gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    with file_lock(args.root / "state" / BATCH / "worker.lock", nonblocking=True):
        worker(args.root, args.snapshot, args.gpus)
