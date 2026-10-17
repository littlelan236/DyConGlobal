"""Compare an actual vLLM smoke capture with HF at identical tokens and sites."""

import argparse
import os
from pathlib import Path

import numpy as np

from core.config import resolve
from core.io import read_json, write_json
from hidden.backends.vllm.engine import gpu_reservation
from hidden.evaluation.smoke_vllm import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--smoke-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    a = p.parse_args()
    cfg = resolve(a.config, a.set)
    cfg["hidden"]["device"] = "cuda:0"
    report = read_json(a.smoke_dir / "report.json")
    if not report.get("passed"):
        raise ValueError("vLLM smoke must pass before HF comparison")
    if (
        report["model_fingerprint"] != cfg["model"]["fingerprint"]
        or report["dtype"] != cfg["hidden"]["dtype"]
    ):
        raise ValueError("HF comparison must use the same checkpoint and dtype")
    cfg["hidden"]["layers"] = report["layers"]
    with gpu_reservation(cfg["project_root"], [a.gpu]):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        from hidden.adapters.hf import load_adapter

        adapter = load_adapter(cfg)
        metrics = {}
        for row in read_json(a.smoke_dir / "inputs.json"):
            expected = adapter.capture(row["prefix_token_ids"], report["layers"], cfg["hidden"]["chunk_size"])
            with np.load(a.smoke_dir / (row["sample_id"] + ".npz")) as data:
                actual = data["hidden"]
            metrics[row["sample_id"]] = compare(actual, expected)
        result = {
            "model_fingerprint": cfg["model"]["fingerprint"],
            "layers": report["layers"],
            "metrics": metrics,
            "purpose": "backend numerical audit; no bitwise-equality claim",
        }
        write_json(a.smoke_dir / "hf-comparison.json", result)
        print(result)


if __name__ == "__main__":
    main()
