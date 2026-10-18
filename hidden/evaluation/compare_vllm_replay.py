"""Compare online FULL capture with a native vLLM FP16 replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.io import read_json, write_json
from hidden.artifacts import event_records, read_arrays


def _load_index(root: Path):
    """Read either the offline aggregate index or async raw sidecars."""
    aggregate = root / "index.json"
    if aggregate.exists():
        source = read_json(aggregate)
        rows = {row["sample_id"]: row for row in source}
        if len(rows) != len(source):
            raise ValueError("Duplicate sample identities in capture index")
        return rows
    sidecars = root / "index"
    if sidecars.is_dir():
        rows = {}
        for path, row in event_records(root):
            row.setdefault("path", row.get("tensor"))
            rows[row["sample_id"]] = row
        return rows
    raise FileNotFoundError(f"No capture index under {root}")


def _tensors(data, row):
    if "hidden" in data:
        values = data["hidden"]
        sites = row.get("native_layers")
        if sites is not None:
            if values.shape[0] != len(sites):
                raise ValueError("Hidden layer axis differs from site metadata")
            return {f"layer_{site}": values[i] for i, site in enumerate(sites)}
        return {"hidden": values}
    names = sorted(data.files if hasattr(data, "files") else data)
    if not names:
        raise ValueError("Capture tensor archive is empty")
    return {name: data[name] for name in names}


def row_metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if actual.shape != reference.shape:
        raise ValueError(f"Hidden shapes differ: {actual.shape} != {reference.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite hidden values")
    delta = actual - reference
    denom = max(float(np.linalg.norm(reference)), 1e-12)
    cosine = float(
        np.dot(actual, reference) / max(float(np.linalg.norm(actual) * np.linalg.norm(reference)), 1e-12)
    )
    return {
        "cosine": cosine,
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "max_abs": float(np.max(np.abs(delta))),
    }


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
    metrics, by_layer, by_group, by_condition = {}, {}, {}, {}
    for sample_id, row in online_index.items():
        replay_row = replay_index[sample_id]
        for key in [
            "token_id",
            "absolute_position",
            "token_ref",
            "prefix_sha256",
            "request_sha256",
            "trajectory_sha256",
            "tokenizer_fingerprint",
            "model_fingerprint",
            "serving_dtype",
        ]:
            if row.get(key) is not None and replay_row.get(key) != row[key]:
                raise ValueError(f"{key} mismatch for {sample_id}")
        actual = _tensors(read_arrays(online, row), row)
        reference = _tensors(read_arrays(replay, replay_row), replay_row)
        if actual.keys() != reference.keys():
            raise ValueError(f"Layer/site mismatch for {sample_id}")
        layers = {site: row_metrics(actual[site], reference[site]) for site in actual}
        metrics[sample_id] = {
            "cosine": min(value["cosine"] for value in layers.values()),
            "relative_l2": max(value["relative_l2"] for value in layers.values()),
            "max_abs": max(value["max_abs"] for value in layers.values()),
            "layers": layers,
        }
        group = row.get("evaluation_group", row.get("session_id", row.get("anchor", "all")))
        for site, value in layers.items():
            by_layer.setdefault(site, []).append(value)
            by_group.setdefault(group, {}).setdefault(site, []).append(value)
            for condition in ["anchor", "graph_execution", "batch_kind"]:
                label = row.get(condition) or "unknown"
                by_condition.setdefault(condition, {}).setdefault(label, {}).setdefault(site, []).append(
                    value
                )
    if not metrics:
        raise ValueError("No aligned events to compare")

    def summarize(values):
        return {
            "rows": len(values),
            "min_cosine": min(x["cosine"] for x in values),
            "max_relative_l2": max(x["relative_l2"] for x in values),
            "median_relative_l2": float(np.median([x["relative_l2"] for x in values])),
            "p95_relative_l2": float(np.percentile([x["relative_l2"] for x in values], 95)),
            "max_abs": max(x["max_abs"] for x in values),
        }

    worst_cosine = min(x["cosine"] for x in metrics.values())
    worst_l2 = max(x["relative_l2"] for x in metrics.values())
    return {
        "dtype": dtype,
        "model_family": model_family,
        "samples": len(metrics),
        "worst_cosine": worst_cosine,
        "worst_relative_l2": worst_l2,
        "cosine_threshold": threshold[0],
        "relative_l2_threshold": threshold[1],
        "passed": worst_cosine >= threshold[0] and worst_l2 <= threshold[1],
        "metrics": metrics,
        "layers": {site: summarize(values) for site, values in by_layer.items()},
        "groups": {
            group: {site: summarize(values) for site, values in sites.items()}
            for group, sites in by_group.items()
        },
        "conditions": {
            condition: {
                label: {site: summarize(values) for site, values in sites.items()}
                for label, sites in labels.items()
            }
            for condition, labels in by_condition.items()
        },
        "reference": "vllm_native_replay",
    }


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
