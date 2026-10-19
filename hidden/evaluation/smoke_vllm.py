"""Explicit GPU integration check: mixed lengths, chunks, hot APC and reordering.

Run only in hidden.vllm.python. Real engine errors fail the check; no mocks or
silent cache-off fallback. Outputs are kept separately from training captures.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from core.config import resolve
from core.io import hash_json, read_json, write_json
from hidden.backends.vllm.engine import configure_environment, gpu_reservation


def compare(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / np.maximum(denom, 1e-20)
    relative = np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-20)
    return {
        "min_cosine": float(cosine.min()),
        "max_relative_l2": float(relative.max()),
        "max_absolute": float(np.abs(a - b).max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = resolve(a.config, a.set)
    configure_environment(cfg)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    options = cfg["hidden"]["vllm"]
    with gpu_reservation(
        cfg["project_root"],
        options["gpus"],
        allow_shared=options.get("allow_shared_gpus", False),
        min_free_bytes=options.get("min_free_gpu_bytes", 0),
    ):
        from transformers import AutoTokenizer

        from hidden.adapters.vllm import load_adapter

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True)
        body = tokenizer.encode(
            "Inspect the repository and fix the failing test.\n", add_special_tokens=False
        )
        rows = []
        for i, (length, anchor) in enumerate([(129, "<think>"), (2049, "</think>"), (4097, "<think>")]):
            marker = tokenizer.encode(anchor, add_special_tokens=False)
            ids = (body * ((length // len(body)) + 1))[: length - len(marker)] + marker
            rows.append({"sample_id": f"smoke-{i}", "prefix_token_ids": ids, "prefix_sha256": hash_json(ids)})
        write_json(out / "inputs.json", rows)
        layers = cfg["hidden"]["layers"]
        start = time.monotonic()
        adapter = load_adapter(cfg)
        report = {
            "runtime": adapter.metadata,
            "layers": layers,
            "model_fingerprint": adapter.fingerprint,
            "dtype": cfg["hidden"]["dtype"],
        }
        try:
            cold = adapter.capture_batch(rows, layers, cfg["hidden"]["chunk_size"])
            hot = adapter.capture_batch(rows, layers, cfg["hidden"]["chunk_size"])
            reordered = adapter.capture_batch(list(reversed(rows)), layers, cfg["hidden"]["chunk_size"])
            single = {r["sample_id"]: adapter.capture_batch([r], layers, 1)[r["sample_id"]] for r in rows}
            report["comparisons"] = {}
            for row in rows:
                sid = row["sample_id"]
                report["comparisons"][sid] = {
                    key: compare(value[sid], cold[sid])
                    for key, value in [("hot", hot), ("reordered", reordered), ("single", single)]
                }
                np.savez_compressed(out / (sid + ".npz"), hidden=cold[sid])
            report["windows"] = adapter.metrics
            if cfg["hidden"]["vllm"]["enable_prefix_caching"] and not adapter.metrics[1]["cached_tokens"]:
                raise AssertionError("APC enabled but repeated prompts produced no measured cache hits")
            if any(
                c["min_cosine"] < 0.999 or c["max_relative_l2"] > 0.05
                for sample in report["comparisons"].values()
                for c in sample.values()
            ):
                raise AssertionError("Batch/cache result drift exceeds smoke tolerances")
            report["passed"] = True
        finally:
            report["seconds"] = time.monotonic() - start
            write_json(out / "report.json", report)
            adapter.close()
    print(read_json(out / "report.json"))


if __name__ == "__main__":
    main()
