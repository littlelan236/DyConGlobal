"""Exact sample-ID join without duplicating labels, splits or hidden tensors."""

from core.io import artifact_dir, read_json, sha256, write_json
from core.locking import exclusive_artifact


@exclusive_artifact("difficulty-dataset", "index_id")
def build_index(cfg, dataset_id, capture_id, index_id, rollout_ids=(), rollout_list=None):
    source_run = cfg.get("source_run_id") or cfg.get("run_id")
    if not source_run:
        raise ValueError("Dataset trajectory selection requires an explicit source run")
    from difficulty.datasets.selection import select_run
    scope = select_run(cfg["project_root"], source_run, rollout_ids, rollout_list)
    selected_ids = set(scope["rollout_ids"])
    from application.execution.evidence import RunEvidence
    all_run_ids = set(RunEvidence(cfg["project_root"], source_run).read()["rollouts"])

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
        if row["rollout_id"] not in all_run_ids:
            raise ValueError("Capture trajectory belongs to another source run")
        if row["rollout_id"] not in selected_ids:
            continue
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
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "index.json",
        {
            "schema_version": 1,
            "source_run_id": source_run, "rollout_ids": scope["rollout_ids"],
            "dataset_id": dataset_id,
            "capture_id": capture_id,
            "label_root": str(source.relative_to(cfg["project_root"])),
            "capture_root": str(capture.relative_to(cfg["project_root"])),
            "source_receipt_sha256": sha256(source / "receipt.json"),
            "capture_receipt_sha256": sha256(capture / "receipt.json"),
            "samples": samples,
        },
    )
    write_json(out / "receipt.json", {
        "schema_version": 1, "state": "complete", "dataset_id": index_id,
        "source_receipt_sha256": sha256(source / "receipt.json"),
        "capture_receipt_sha256": sha256(capture / "receipt.json"),
        "source_files": {str(source / "receipt.json"): sha256(source / "receipt.json"),
                         str(capture / "receipt.json"): sha256(capture / "receipt.json")},
        "files": {"index.json": sha256(out / "index.json")},
    })
    return out
