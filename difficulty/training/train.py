"""Training utilities for remaining reasoning workload estimators."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from core.io import artifact_dir, read_json
from difficulty.models.estimators import build_estimator, inverse_target, transform_target

TARGETS = {
    "remaining_reasoning_tokens_log1p": "remaining_reasoning_tokens",
    "remaining_steps_log1p": "remaining_steps",
    "remaining_steps_linear": "remaining_steps",
}


def load_training_data(cfg, index_id, target, split="all"):
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    idx_path = artifact_dir(cfg, "difficulty", index_id, "datasets") / "index.json"
    idx = read_json(idx_path)
    source = artifact_dir(cfg, "enrichment", idx["dataset_id"])
    labels = {}
    for name in {r["label_shard"] for r in idx["samples"]}:
        labels.update({r["sample_id"]: r for r in read_json(source / name)})
    rows = []
    for item in idx["samples"]:
        row = labels[item["sample_id"]]
        value = row["labels"][TARGETS[target]]
        if value is None:
            continue
        with np.load(artifact_dir(cfg, "hidden", idx["capture_id"], "outputs") / item["hidden_path"]) as f:
            hidden = f["hidden"]
        rows.append(
            {
                "sample_id": item["sample_id"],
                "rollout_id": row["rollout_id"],
                "hidden": hidden,
                "target": float(value),
            }
        )
    if not rows:
        raise ValueError("No labeled hidden samples")
    groups = defaultdict(list)
    for r in rows:
        groups[r["rollout_id"]].append(r)
    keys = sorted(groups)
    n = max(1, int(round(len(keys) * 0.8)))
    train_keys = set(keys[:n]) if split == "train" else set(keys[n:]) if split == "val" else set(keys)
    selected = [r for r in rows if r["rollout_id"] in train_keys]
    if split == "val" and not selected:
        raise ValueError("Validation split is empty")
    x = np.stack([r["hidden"] for r in selected])
    y = np.asarray([r["target"] for r in selected])
    return x, transform_target(y, target), [r["sample_id"] for r in selected]


def regression_report(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = y - p
    ss = ((y - y.mean()) ** 2).sum()
    return {
        "count": int(len(y)),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e * e).mean())),
        "r2": None if ss == 0 else float(1 - (e * e).sum() / ss),
    }


def train_estimators(cfg, index_id, target, kinds=("ridge", "ffn_mlp", "ridge_plus_ffn"), **kwargs):
    x, y, _ = load_training_data(cfg, index_id, target, "train")
    xv, yv, _ = load_training_data(cfg, index_id, target, "val")
    results = {}
    for kind in kinds:
        model = build_estimator(kind, **kwargs.get(kind, {}))
        model.fit(x, y)
        pred = model.predict(xv)
        results[kind] = {
            "model": model,
            "metrics_transformed": regression_report(yv, pred),
            "predictions": inverse_target(pred, target).tolist(),
            "targets": inverse_target(yv, target).tolist(),
        }
    return results
