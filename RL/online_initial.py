"""Portable, checksummed bootstrap inputs from a published predictor and its dataset.

Preparation uses the ordinary difficulty selection/feature readers on the source
host. Only the selected public site is exported; the initial predictor bytes and
historical split assignments are preserved, independently of pool eviction.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import numpy as np

from core.io import hash_json, read_json, sha256, write_json


def prepare_initial(cfg, predictor, output, *, feature_family="same_base_cross_version_v1"):
    from runtime.serving.parallel import artifact_contract
    from difficulty.datasets.submission import load_selection, iter_features, feature_contract
    from difficulty.models.estimators import transform_target
    from difficulty.datasets.paths import dataset_directory
    source = Path(predictor).resolve()
    contract = artifact_contract(str(source), sha256(source), cfg["model"]["fingerprint"])
    if contract["loader"] != "runtime.serving.ridge_provider:load" or len(contract["sites"]) != 1:
        raise ValueError("native initialization currently supports a published single-site Ridge")
    provenance = contract["provenance"]
    experiment = read_json(source.parent / "source-experiment.json")
    settings = SimpleNamespace(**experiment["settings"])
    rollouts, samples, evidence = load_selection(cfg, settings)
    dataset = dataset_directory(cfg, settings.dataset_id)
    split = read_json(source.parent / "source-split.json")
    if split["dataset_receipt_sha256"] != settings.dataset_receipt_sha256:
        raise ValueError("predictor split and dataset receipt differ")
    assignments = {sid: ("val" if role == "validation" else role)
                   for role, ids in split["sample_ids"].items() for sid in ids}
    if set(assignments) != {s["sample_id"] for s in samples}:
        raise ValueError("predictor split does not cover the complete initial pool")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name("." + output.name + ".preparing")
    staging.mkdir(parents=True, exist_ok=False)
    sites = feature_contract(samples)["native_sites"]
    site = int(contract["sites"][0])
    column = sites.index(site)
    width = int(contract["hidden_size"])
    hidden = np.lib.format.open_memmap(staging / "hidden.npy", mode="w+", dtype=np.float32,
                                      shape=(len(samples), width))
    for index, features in iter_features(samples):
        vector = features[column]
        if vector.shape != (width,) or not np.isfinite(vector).all():
            raise ValueError("invalid initial hidden vector")
        hidden[index] = vector
    hidden.flush()
    del hidden
    tokenizers = {sample["tokenizer_fingerprint"] for sample in samples}
    if len(tokenizers) != 1:
        raise ValueError("initial dataset mixes tokenizers")
    feature = {"site": "layer_" + str(site), "width": width, "dtype": "float32",
               "feature_family": feature_family, "base_fingerprint": contract["model_fingerprint"],
               "tokenizer_fingerprint": next(iter(tokenizers)), "target": contract["target"]}
    field = "remaining_reasoning_tokens_to_submission" if "reasoning_tokens" in contract["target"] else "remaining_steps_to_submission"
    with (staging / "points.jsonl").open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            point = {"point_id": sample["sample_id"], "target": float(transform_target([sample[field]], contract["target"])[0]),
                     "model_fingerprint": sample["model_fingerprint"], "birth_step": 0,
                     "split": assignments[sample["sample_id"]],
                     **{k: v for k, v in feature.items() if k != "target"}, "hidden_row": index}
            handle.write(json.dumps(point, allow_nan=False) + "\n")
    shutil.copytree(source.parent, staging / "predictor")
    manifest = {"schema": "online_initial_dataset_v1", "count": len(samples), "feature": feature,
                "fit_spec": {"model_kind": "ridge", "alpha": float(provenance["alpha"]),
                             "target": contract["target"], "split_policy": "train_only"},
                "initial_predictor": "predictor/manifest.json", "initial_fit_scope": provenance["fit_scope"],
                "new_point_split": {"policy": "stable_hash_v1", "seed": split["seed"],
                                    "validation_fraction": split["validation_fraction"],
                                    "test_fraction": split["test_fraction"]},
                "source": {"predictor_sha256": sha256(source), "dataset_id": settings.dataset_id,
                           "dataset_receipt_sha256": settings.dataset_receipt_sha256,
                           "split_sha256": sha256(source.parent / "source-split.json"),
                           "sample_order_sha256": hash_json([s["sample_id"] for s in samples])},
                "files": {str(p.relative_to(staging)): sha256(p) for p in staging.rglob("*") if p.is_file()}}
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "acceptance-tasks.json", sorted([
        {k: row[k] for k in ("instance_id", "rollout_id", "run_id", "trajectory_sha256", "canonical_valid_assistant_steps")}
        for row in rollouts if row.get("resolved") is True], key=lambda r: r["canonical_valid_assistant_steps"])[:8])
    staging.rename(output)
    return {"manifest": str(output / "manifest.json"), "count": len(samples), "feature": feature,
            "initial_predictor_sha256": sha256(source)}


def load_initial(path):
    """Return exact initial pool plus immutable predictor/fit metadata."""
    from RL.online_bootstrap import HiddenPoint
    path = Path(path).resolve()
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != "online_initial_dataset_v1":
        points = value.get("points") if isinstance(value, dict) else value
        if not isinstance(points, list) or not points:
            raise ValueError("initial estimator requires a prepared dataset bundle or explicit point manifest")
        return [HiddenPoint(**point) for point in points], value if isinstance(value, dict) else {}
    from core.io import owned_path
    if not {"hidden.npy", "points.jsonl", value["initial_predictor"]} <= value["files"].keys():
        raise ValueError("initial estimator bundle omits required file bindings")
    for name, digest in value["files"].items():
        if sha256(owned_path(path.parent, name)) != digest:
            raise ValueError("initial estimator bundle checksum mismatch: " + name)
    hidden = np.load(path.parent / "hidden.npy", mmap_mode="r", allow_pickle=False)
    points, indices = [], []
    with (path.parent / "points.jsonl").open() as handle:
        for line in handle:
            point = json.loads(line)
            index = point.pop("hidden_row")
            if type(index) is not int or not 0 <= index < len(hidden):
                raise ValueError("initial estimator hidden row is out of range")
            indices.append(index)
            points.append(HiddenPoint(**point, hidden=hidden[index].tolist()))
    if (len(points) != value["count"] or hidden.shape != (len(points), value["feature"]["width"])
            or sorted(indices) != list(range(len(points)))
            or len({point.point_id for point in points}) != len(points)):
        raise ValueError("initial estimator bundle count/width mismatch")
    value["initial_predictor"] = str(owned_path(path.parent, value["initial_predictor"]))
    return points, value


def new_point_split(point_id, spec):
    """Assign once by stable identity, never condition eviction on split."""
    if not spec:
        return "train"
    if spec.get("policy") != "stable_hash_v1":
        raise ValueError("unsupported new estimator point split policy")
    draw = int(hash_json([spec["seed"], point_id])[:16], 16) / 2**64
    return ("test" if draw < spec["test_fraction"] else "val"
            if draw < spec["test_fraction"] + spec["validation_fraction"] else "train")
