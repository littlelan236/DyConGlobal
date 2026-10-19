"""Idempotent model-bound extraction from enrichment artifacts."""

from pathlib import Path

import numpy as np

from core.io import artifact_dir, hash_json, identifier, read_json, sha256, write_json
from core.locking import exclusive_artifact, file_lock


def register_capture(cfg, out, receipt):
    """Append new backend products without rewriting migrated legacy entries."""
    if "backend" not in receipt:
        return
    root = Path(cfg["project_root"])
    source = read_json(artifact_dir(cfg, "enrichment", receipt["dataset_id"]) / "receipt.json")
    selection = source["selection"]
    index = read_json(out / "index.json")
    bindings = {}
    for item in index:
        rid = item["rollout_id"]
        row = selection["rollouts"][rid]
        binding = bindings.setdefault(
            rid,
            {
                "task_id": selection["tasks"][row["task_key"]]["instance_id"],
                "run_id": row["run_id"],
                "trajectory_sha256": row["trajectory_sha256"],
                "anchors": [],
            },
        )
        anchor = {"name": item["anchor"], "text": "<think>" if item["anchor"] == "think_open" else "</think>"}
        if anchor not in binding["anchors"]:
            binding["anchors"].append(anchor)
    entry = {
        "dataset_id": receipt["dataset_id"],
        "capture_id": receipt["capture_id"],
        "status": "completed",
        "backend": receipt["backend"],
        "rollouts": bindings,
        "receipt": str((out / "receipt.json").relative_to(root)),
        "receipt_sha256": sha256(out / "receipt.json"),
    }
    with file_lock(root / "state/product-manifests.lock"):
        for owner in ["enrichment", "hidden"]:
            path = root / owner / "manifest.json"
            manifest = read_json(path) if path.exists() else {"schema_version": 1}
            manifest.setdefault("native_captures", {})[receipt["capture_id"]] = entry
            write_json(path, manifest)


@exclusive_artifact("hidden", "capture_id")
def extract_hidden(cfg, dataset_id, capture_id, adapter):
    source = artifact_dir(cfg, "enrichment", dataset_id)
    receipt = read_json(source / "receipt.json")
    fingerprint = cfg["model"]["fingerprint"]
    if not fingerprint or adapter.fingerprint != fingerprint or receipt["model_fingerprint"] != fingerprint:
        raise ValueError("Capture checkpoint differs from enrichment or configured model")
    layers = cfg["hidden"]["layers"]
    if not layers:
        raise ValueError("Configure hidden.layers explicitly; estimator layer selection is not assumed")
    out = artifact_dir(cfg, "hidden", capture_id)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"source": sha256(source / "receipt.json"), "hidden": cfg["hidden"], "model": fingerprint}
    if hasattr(adapter, "metadata"):
        inputs["backend"] = adapter.metadata
    identity = hash_json(inputs)
    if (out / "receipt.json").exists():
        existing = read_json(out / "receipt.json")
        if existing["input_identity"] != identity:
            raise ValueError("Capture ID already has different inputs")
        for name, digest in existing["files"].items():
            if sha256(out / name) != digest:
                raise ValueError("Capture artifact checksum mismatch")
        register_capture(cfg, out, existing)
        return out
    files, index = {}, []
    metrics_start = len(getattr(adapter, "metrics", []))
    pending = []
    batch_size = getattr(adapter, "batch_size", 1)

    def store(row, sample_hash, values):
        if values.shape != (len(layers), adapter.hidden_size) or not np.isfinite(values).all():
            raise ValueError("Hidden shape or values are invalid")
        tensor_name = f"{sample_hash}.npz"
        tmp = out / (sample_hash + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, hidden=values.astype(np.float32))
        tmp.replace(out / tensor_name)
        write_json(
            out / (sample_hash + ".json"),
            {
                "sample_id": row["sample_id"],
                "sha256": sha256(out / tensor_name),
                "input_identity": identity,
            },
        )

    def flush():
        if not pending:
            return
        rows = [row for row, _ in pending]
        if not hasattr(adapter, "capture_batch"):
            raise TypeError("Offline hidden extraction requires the vLLM batch adapter")
        values = adapter.capture_batch(rows, layers, cfg["hidden"]["chunk_size"])
        if set(values) != {row["sample_id"] for row in rows}:
            raise ValueError("Batch capture returned mismatched sample identities")
        for row, sample_hash in pending:
            store(row, sample_hash, values[row["sample_id"]])
        pending.clear()

    seen = set()
    for name, expected in receipt["files"].items():
        if sha256(source / name) != expected:
            raise ValueError("Enrichment shard changed")
        rows = read_json(source / name)
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError("Duplicate sample_id in enrichment dataset")
            seen.add(sample_id)
            if hash_json(row["prefix_token_ids"]) != row["prefix_sha256"]:
                raise ValueError("Prefix checksum mismatch")
            sample_hash = hash_json(
                {"sample": sample_id, "prefix": row["prefix_sha256"], "capture": identity}
            )
            tensor_name = f"{sample_hash}.npz"
            if (out / tensor_name).exists() and (out / (sample_hash + ".json")).exists():
                old = read_json(out / (sample_hash + ".json"))
                if (
                    old["sha256"] != sha256(out / tensor_name)
                    or old["sample_id"] != sample_id
                    or old["input_identity"] != identity
                ):
                    raise ValueError("Existing partial capture is corrupt")
            else:
                pending.append((row, sample_hash))
                if len(pending) >= batch_size:
                    flush()
            index.append(
                {
                    "sample_id": sample_id,
                    "rollout_id": row["rollout_id"],
                    "step_id": row["step_id"],
                    "anchor": row["anchor"],
                    "occurrence": row["occurrence"],
                    "path": tensor_name,
                    "prefix_sha256": row["prefix_sha256"],
                    "anchor_token_id": row["prefix_token_ids"][-1],
                    "anchor_position": len(row["prefix_token_ids"]) - 1,
                }
            )
    flush()
    for item in index:
        files[item["path"]] = sha256(out / item["path"])
    if not index:
        raise ValueError("No recoverable anchors in the selected enrichment dataset")
    write_json(out / "index.json", index)
    files["index.json"] = sha256(out / "index.json")
    receipt = {
        "schema_version": 1,
        "capture_id": identifier(capture_id),
        "dataset_id": dataset_id,
        "input_identity": identity,
        "model_fingerprint": fingerprint,
        "source_receipt_sha256": sha256(source / "receipt.json"),
        "layers": layers,
        "sites": [adapter.sites()[i] for i in layers],
        "hidden_size": adapter.hidden_size,
        "storage_dtype": "float32",
        "replay": "native_trajectory_apc",
        "files": files,
    }
    receipt["backend"] = adapter.metadata
    receipt["execution_metrics"] = getattr(adapter, "metrics", [])[metrics_start:]
    write_json(out / "receipt.json", receipt)
    register_capture(cfg, out, receipt)
    return out
