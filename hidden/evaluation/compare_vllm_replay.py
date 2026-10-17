"""Compare online FULL capture with a native vLLM FP16 replay."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.io import read_json, write_json


def _load_index(root: Path):
    """Read either the offline aggregate index or async raw sidecars."""
    aggregate = root / "index.json"
    if aggregate.exists():
        return {row["sample_id"]: row for row in read_json(aggregate)}
    sidecars = root / "index"
    if sidecars.is_dir():
        rows = {}
        for path in sorted(sidecars.glob("*.json")):
            row = read_json(path)
            row.setdefault("path", row.get("tensor"))
            rows[row["sample_id"]] = row
        return rows
    raise FileNotFoundError(f"No capture index under {root}")


def _tensor(data):
    if "hidden" in data:
        return data["hidden"]
    names = sorted(data.files)
    if not names:
        raise ValueError("Capture tensor archive is empty")
    return data[names[0]]


def row_metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    if actual.shape != reference.shape:
        raise ValueError(f"Hidden shapes differ: {actual.shape} != {reference.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite hidden values")
    delta = actual - reference
    denom = max(float(np.linalg.norm(reference)), 1e-12)
    cosine = float(np.dot(actual, reference) / max(float(np.linalg.norm(actual) * np.linalg.norm(reference)), 1e-12))
    return {"cosine": cosine, "relative_l2": float(np.linalg.norm(delta) / denom), "max_abs": float(np.max(np.abs(delta)))}


def compare_replay(online_dir, replay_dir, *, model_family="dense", dtype="float16"):
    if dtype != "float16":
        raise ValueError("vLLM replay acceptance is FP16-only")
    online = Path(online_dir)
    replay = Path(replay_dir)
    online_index = _load_index(online)
    replay_index = _load_index(replay)
    if set(online_index) != set(replay_index):
        raise ValueError("Online and replay sample IDs differ")
    threshold = (0.999, 0.05) if model_family == "dense" else (0.997, 0.08)
    metrics = {}
    for sample_id, row in online_index.items():
        replay_row = replay_index[sample_id]
        if row.get("token_id") is not None and replay_row.get("token_id") != row["token_id"]:
            raise ValueError(f"Token ID mismatch for {sample_id}")
        if row.get("absolute_position") is not None and replay_row.get("absolute_position") != row["absolute_position"]:
            raise ValueError(f"Absolute position mismatch for {sample_id}")
        if row.get("token_ref") is not None and replay_row.get("token_ref") != row["token_ref"]:
            raise ValueError(f"Token reference mismatch for {sample_id}")
        with np.load(online / row["path"]) as actual_data, np.load(replay / replay_row["path"]) as replay_data:
            metrics[sample_id] = row_metrics(_tensor(actual_data), _tensor(replay_data))
    worst_cosine = min(x["cosine"] for x in metrics.values())
    worst_l2 = max(x["relative_l2"] for x in metrics.values())
    return {"dtype": dtype, "model_family": model_family, "samples": len(metrics), "worst_cosine": worst_cosine, "worst_relative_l2": worst_l2, "cosine_threshold": threshold[0], "relative_l2_threshold": threshold[1], "passed": worst_cosine >= threshold[0] and worst_l2 <= threshold[1], "metrics": metrics, "reference": "vllm_native_replay"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=["dense", "moe"], default="dense")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_replay(args.online_dir, args.replay_dir, model_family=args.model_family)
    write_json(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
