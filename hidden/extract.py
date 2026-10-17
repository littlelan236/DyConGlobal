"""Idempotent model-bound extraction from enrichment artifacts."""


import numpy as np

from core.io import artifact_dir, hash_json, identifier, read_json, sha256, write_json
from core.locking import exclusive_artifact


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
    identity = hash_json(
        {"source": sha256(source / "receipt.json"), "hidden": cfg["hidden"], "model": fingerprint}
    )
    if (out / "receipt.json").exists():
        existing = read_json(out / "receipt.json")
        if existing["input_identity"] != identity:
            raise ValueError("Capture ID already has different inputs")
        for name, digest in existing["files"].items():
            if sha256(out / name) != digest:
                raise ValueError("Capture artifact checksum mismatch")
        return out
    files, index = {}, []
    for name, expected in receipt["files"].items():
        if sha256(source / name) != expected:
            raise ValueError("Enrichment shard changed")
        rows = read_json(source / name)
        for row in rows:
            sample_id = row["sample_id"]
            sample_hash = hash_json(
                {"sample": sample_id, "prefix": row["prefix_sha256"], "capture": identity}
            )
            tensor_name = f"{sample_hash}.npz"
            if (out / tensor_name).exists() and (out / (sample_hash + ".json")).exists():
                old = read_json(out / (sample_hash + ".json"))
                if old["sha256"] != sha256(out / tensor_name):
                    raise ValueError("Existing partial capture is corrupt")
            else:
                if hash_json(row["prefix_token_ids"]) != row["prefix_sha256"]:
                    raise ValueError("Prefix checksum mismatch")
                values = adapter.capture(row["prefix_token_ids"], layers, cfg["hidden"]["chunk_size"])
                if values.shape != (len(layers), adapter.hidden_size) or not np.isfinite(values).all():
                    raise ValueError("Hidden shape or values are invalid")
                # Keep fp32 on disk: fp16 storage could overflow otherwise finite states.
                tmp = out / (sample_hash + ".tmp")
                with tmp.open("wb") as f:
                    np.savez_compressed(f, hidden=values.astype(np.float32))
                tmp.replace(out / tensor_name)
                write_json(
                    out / (sample_hash + ".json"),
                    {"sample_id": sample_id, "sha256": sha256(out / tensor_name), "input_identity": identity},
                )
            files[tensor_name] = sha256(out / tensor_name)
            index.append(
                {
                    "sample_id": sample_id,
                    "rollout_id": row["rollout_id"],
                    "step_id": row["step_id"],
                    "anchor": row["anchor"],
                    "occurrence": row["occurrence"],
                    "path": tensor_name,
                    "prefix_sha256": row["prefix_sha256"],
                }
            )
    if not index:
        raise ValueError("No recoverable anchors in the selected enrichment dataset")
    write_json(out / "index.json", index)
    files["index.json"] = sha256(out / "index.json")
    write_json(
        out / "receipt.json",
        {
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
            "replay": "independent_causal_prefix",
            "files": files,
        },
    )
    return out
