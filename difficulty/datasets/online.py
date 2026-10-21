"""Create task-bound label views referencing packed online hidden rows."""

import math
from pathlib import Path

from catalog.store import Catalog
from core.io import artifact_dir, read_json, sha256, write_json


def build_published_index(cfg, run_id, index_id):
    """Reference published shards directly; no implicit packed tensor copy."""
    from core.config import load_snapshot

    frozen = load_snapshot(cfg["project_root"], run_id)
    trace = Path(cfg["project_root"]) / "runs" / run_id / "enrichment/online"
    receipt = read_json(trace / "receipt.json")
    for name, digest in receipt["files"].items():
        if sha256(trace / name) != digest:
            raise ValueError("Published token-event labels changed")
    catalog = Catalog(cfg["project_root"]).read()
    samples = []
    for event in read_json(trace / "index.json"):
        rollout = catalog["rollouts"][event["rollout_id"]]
        if rollout["trajectory_sha256"] != event["trajectory_sha256"] or rollout["provenance"].get(
            "quarantined"
        ):
            raise ValueError("Published event does not bind an eligible canonical trajectory")
        eligible = rollout["resolved"] is True and not event["censored"]
        samples.append(
            {
                **event,
                "partition": catalog["tasks"][rollout["task_key"]]["partition"],
                "labels": {
                    "remaining_steps": event["remaining_steps"] if eligible else None,
                    "remaining_reasoning_tokens": event["remaining_reasoning_tokens_after"]
                    if eligible
                    else None,
                },
                "selected_evaluation_id": rollout["selected_evaluation_id"],
            }
        )
    source = Path(frozen["paths"]["nas_root"]) / frozen["capture"]["export"]["root"] / run_id
    out = artifact_dir(cfg, "difficulty", index_id, "datasets")
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "index.json",
        {
            "schema_version": 3,
            "dataset_id": run_id,
            "capture_id": run_id,
            "hidden_root": str(source),
            "samples": samples,
            "source_receipt_sha256": sha256(trace / "receipt.json"),
            "split_authority": "catalog/manifest.json task partition",
            "catalog_revision": catalog["catalog_revision"],
        },
    )
    return out


def build_online_index(cfg, capture_id, index_id):
    source = artifact_dir(cfg, "hidden", capture_id)
    receipt = read_json(source / "receipt.json")
    if any(sha256(source / name) != digest for name, digest in receipt["files"].items()):
        raise ValueError("Packed online capture checksum mismatch")
    catalog = Catalog(cfg["project_root"]).read()
    partitions = {task["instance_id"]: task["partition"] for task in catalog["tasks"].values()}
    rows = []
    for event in read_json(source / "index.json"):
        row = dict(event)
        row["remaining_steps_log1p"] = math.log1p(row["remaining_steps"])
        row["remaining_reasoning_tokens_log1p"] = math.log1p(row["remaining_reasoning_tokens_after"])
        row["partition"] = partitions.get(row["task_id"], "diagnostic")
        row["hidden_path"] = str((source / row["path"]).relative_to(Path(cfg["project_root"])))
        rows.append(row)
    out = artifact_dir(cfg, "difficulty", index_id, "datasets")
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "index.json",
        {
            "schema_version": 2,
            "capture_id": capture_id,
            "capture_receipt_sha256": sha256(source / "receipt.json"),
            "samples": rows,
            "split_authority": "catalog/manifest.json task partition; no token-level random split",
        },
    )
    action = artifact_dir(cfg, "action", index_id, "datasets")
    write_json(
        action / "index.json",
        {
            "shared_index": str((out / "index.json").relative_to(cfg["project_root"])),
            "shared_index_sha256": sha256(out / "index.json"),
            "taxonomy_version": None,
            "samples": [{"sample_id": row["sample_id"], "action_label": None} for row in rows],
        },
    )
    return out
