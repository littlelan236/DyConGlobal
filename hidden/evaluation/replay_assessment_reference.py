"""Replay exact online references with the existing eager extraction adapter."""

import argparse
from pathlib import Path

import numpy as np

from core.config import resolve
from core.io import read_json, write_json
from hidden.backends.vllm.engine import configure_environment, gpu_reservation
from hidden.evaluation.assess_rollout_dtype import summarize_errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = resolve(a.config, a.set)
    configure_environment(cfg)
    source = read_json(a.reference / "report.json")
    if (
        source["dtype"] != cfg["hidden"]["dtype"]
        or source["model_fingerprint"] != cfg["model"]["fingerprint"]
    ):
        raise ValueError("Reference and replay must match checkpoint and dtype")
    rows = read_json(a.reference / "reference-prefixes.json")
    with np.load(a.reference / "online-reference.npz") as data:
        reference = data["hidden"]
    a.output.mkdir(parents=True, exist_ok=False)
    report = {
        "dtype": cfg["hidden"]["dtype"],
        "source": str(a.reference),
        "scope": "compiled TP4 generation versus actual eager extraction adapter",
    }
    with gpu_reservation(cfg["project_root"], cfg["hidden"]["vllm"]["gpus"]):
        from hidden.adapters.vllm import load_adapter

        adapter = load_adapter(cfg)
        report["backend"] = adapter.metadata
        try:
            captured = {}
            for start in range(0, len(rows), adapter.batch_size):
                captured.update(
                    adapter.capture_batch(
                        rows[start : start + adapter.batch_size],
                        cfg["hidden"]["layers"],
                        cfg["hidden"]["chunk_size"],
                    )
                )
            values = np.stack([captured[row["sample_id"]] for row in rows])
            report["replay_policy"] = "trajectory_apc"
            report["comparison"] = summarize_errors(values, reference, rows)
            report["windows"] = adapter.metrics
            np.savez_compressed(a.output / "trajectory-apc.npz", hidden=values)
            report["completed"] = True
        finally:
            write_json(a.output / "report.json", report)
            adapter.close()


if __name__ == "__main__":
    main()
