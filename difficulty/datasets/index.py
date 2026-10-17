"""Exact sample-ID join without duplicating labels, splits or hidden tensors."""

from core.io import artifact_dir, read_json, sha256, write_json


def build_index(cfg, dataset_id, capture_id, index_id):
    source = artifact_dir(cfg, "enrichment", dataset_id)
    capture = artifact_dir(cfg, "hidden", capture_id)
    er = read_json(source / "receipt.json")
    hr = read_json(capture / "receipt.json")
    if hr["source_receipt_sha256"] != sha256(source / "receipt.json"):
        raise ValueError("Capture was created from a different enrichment dataset")
    labels = {}
    for name, expected in er["files"].items():
        if sha256(source / name) != expected:
            raise ValueError("Enrichment checksum mismatch")
        labels.update(
            {
                row["sample_id"]: {"label_shard": name, "rollout_id": row["rollout_id"]}
                for row in read_json(source / name)
            }
        )
    samples = []
    for row in read_json(capture / "index.json"):
        sample = row["sample_id"]
        if sample not in labels:
            raise ValueError("Capture sample has no matching label record")
        if sha256(capture / row["path"]) != hr["files"][row["path"]]:
            raise ValueError("Hidden checksum mismatch")
        samples.append(
            {"sample_id": sample, "label_shard": labels[sample]["label_shard"], "hidden_path": row["path"]}
        )
    if not samples:
        raise ValueError("No samples to join")
    out = artifact_dir(cfg, "difficulty", index_id, "datasets")
    if out.exists():
        raise FileExistsError(out)
    write_json(
        out / "index.json",
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "capture_id": capture_id,
            "source_receipt_sha256": sha256(source / "receipt.json"),
            "capture_receipt_sha256": sha256(capture / "receipt.json"),
            "samples": samples,
        },
    )
    return out
