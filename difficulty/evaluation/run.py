"""Evaluate explicit predictions against immutable enrichment labels; no estimator choice."""

from collections import defaultdict

import numpy as np

from core.io import artifact_dir, hash_json, owned_path, read_json, sha256, write_json
from difficulty.evaluation.metrics import regression_metrics
from difficulty.evaluation.visualize import plot_trajectory
from difficulty.datasets.paths import dataset_directory, source_directory


def evaluate_predictions(cfg, index_id, prediction_file, evaluation_id):
    index_path = dataset_directory(cfg, index_id) / "index.json"
    index = read_json(index_path)
    enrichment = source_directory(cfg, index, index_path.parent, "enrichment")
    if sha256(enrichment / "receipt.json") != index["source_receipt_sha256"]:
        raise ValueError("Evaluation labels no longer match the frozen dataset")
    labels = {}
    for name in {x["label_shard"] for x in index["samples"]}:
        for row in read_json(enrichment / name):
            labels[row["sample_id"]] = row
    allowed = {row["sample_id"] for row in index["samples"]}
    predictions = read_json(prediction_file)
    seen, groups, by_rollout, missing = set(), defaultdict(list), defaultdict(list), 0
    for prediction in predictions:
        sample_id, target = prediction["sample_id"], prediction["target"]
        if sample_id not in allowed or (sample_id, target) in seen:
            raise ValueError("Unknown or duplicate prediction sample/target")
        if target not in {"remaining_steps", "remaining_reasoning_tokens"}:
            raise ValueError("Unknown remaining workload target")
        seen.add((sample_id, target))
        row = labels[sample_id]
        y, p = row["labels"][target], prediction["prediction"]
        if not isinstance(p, (int, float)) or not np.isfinite(p):
            raise ValueError("Predictions must be finite scalars")
        if y is None:
            missing += 1
            continue
        item = {"sample_id": sample_id, "step_id": row["step_id"], "target": y, "prediction": p}
        groups[(target, row["anchor"])].append(item)
        by_rollout[(row["rollout_id"], target, row["anchor"])].append(item)
    out = artifact_dir(cfg, "difficulty", evaluation_id, "evaluations")
    if out.exists():
        raise FileExistsError(out)
    report = {
        "index_id": index_id,
        "index_sha256": sha256(index_path),
        "prediction_sha256": sha256(prediction_file),
        "source_files": {str(index_path): sha256(index_path), str(prediction_file): sha256(prediction_file)},
        "unlabeled_predictions": missing,
        "supplied_predictions": len(predictions),
        "available_samples": len(allowed),
        "metrics": {},
    }
    for key, rows in groups.items():
        y, p = [r["target"] for r in rows], [r["prediction"] for r in rows]
        metric = regression_metrics(y, p)
        metric["log1p"] = regression_metrics(np.log1p(y), np.log1p(p)) if min(p) >= 0 else None
        metric["negative_predictions"] = sum(v < 0 for v in p)
        report["metrics"]["/".join(key)] = metric
    for key, rows in by_rollout.items():
        plot_trajectory(
            sorted(rows, key=lambda r: r["step_id"]),
            out / (hash_json(key)[:24] + ".png"),
            title=" / ".join(key),
        )
    write_json(out / "metrics.json", report)
    write_json(out / "receipt.json", {
        "schema_version": 1, "state": "complete", "index_sha256": sha256(index_path),
        "prediction_sha256": sha256(prediction_file),
        "source_files": {str(index_path): sha256(index_path),
                         str(prediction_file): sha256(prediction_file),
                         str(enrichment / "receipt.json"): sha256(enrichment / "receipt.json")},
        "files": {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*"))
                  if path.is_file()},
    })
    return out
