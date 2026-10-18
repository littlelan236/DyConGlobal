"""Recover raw sidecars and deterministically pack eligible online hidden events."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.io import artifact_dir, hash_json, read_json, sha256, write_json
from hidden.artifacts import read_arrays


def recover_raw_index(export_root):
    root = Path(export_root)
    restored, valid, corrupt, requested = [], set(), [], set()
    journal = root / "queue.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                requested.add(json.loads(line)["sample_id"])
            except (ValueError, KeyError):
                continue  # A killed writer may leave a partial final journal line.
    for sidecar in sorted((root / "raw").glob("*.json")):
        row = read_json(sidecar)
        tensor = root / row["tensor"]
        if not tensor.is_file() or sha256(tensor) != row["sha256"]:
            corrupt.append(row["sample_id"])
            continue
        valid.add(row["sample_id"])
        target = root / "index" / (row["sample_id"] + ".json")
        if not target.exists():
            write_json(target, row)
            restored.append(row["sample_id"])
        elif read_json(target) != row:
            raise ValueError("Raw sidecar and capture index disagree")
    orphan = [path.stem for path in (root / "raw").glob("*.npz") if path.stem not in valid]
    return {
        "restored_index": restored,
        "valid_raw_events": len(valid),
        "corrupt_events": corrupt,
        "orphan_tensors": sorted(orphan),
        "unpersisted_queued_events": sorted(requested - valid),
    }


def pack_online(cfg, export_root, token_trace_dir, capture_id):
    source = Path(export_root)
    trace = Path(token_trace_dir)
    receipt = read_json(trace / "receipt.json")
    for name, digest in receipt["files"].items():
        if sha256(trace / name) != digest:
            raise ValueError("Finalized token-event index changed")
    rows = read_json(trace / "index.json")
    attestation_path = source.parent / "report.json"
    attestation = read_json(attestation_path) if attestation_path.exists() else {}
    for row in rows:
        fingerprint = row.get("model_fingerprint") or attestation.get("model_fingerprint")
        dtype = row.get("serving_dtype") or attestation.get("dtype")
        if fingerprint != cfg["model"]["fingerprint"] or dtype != cfg["serving"]["dtype"]:
            raise ValueError("Online checkpoint/precision attestation mismatch")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate finalized event identity")
    if any(row["alignment_status"] != "exact" for row in rows):
        raise ValueError("Packing requires exact token alignment")
    groups = defaultdict(list)
    for row in rows:
        groups[row["rollout_id"]].append(row)
    out = artifact_dir(cfg, "hidden", capture_id)
    out.mkdir(parents=True, exist_ok=False)
    index, files = [], {}
    for rollout_id, events in sorted(groups.items()):
        events.sort(key=lambda row: (row["step_id"], row["absolute_position"], row["sample_id"]))
        values, sites = [], None
        for row in events:
            arrays = read_arrays(source, row)
            current = sorted(arrays, key=lambda key: int(key.removeprefix("layer_")))
            if sites is not None and current != sites:
                raise ValueError("Capture sites changed within a trajectory")
            sites = current
            value = np.stack([arrays[key] for key in sites])
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite online hidden values")
            values.append(value)
        name = hash_json(rollout_id)[:24] + ".npz"
        np.savez_compressed(out / name, hidden=np.stack(values))
        files[name] = sha256(out / name)
        for offset, row in enumerate(events):
            item = {
                **row,
                "path": name,
                "row": offset,
                "native_layers": [int(site.removeprefix("layer_")) for site in sites],
                "raw_tensor_sha256": row["sha256"],
                "sha256": files[name],
            }
            if "tensor_locator" in item:
                item["raw_tensor_locator"] = item.pop("tensor_locator")
            index.append(item)
    write_json(out / "index.json", index)
    files["index.json"] = sha256(out / "index.json")
    write_json(
        out / "receipt.json",
        {
            "schema_version": 1,
            "capture_id": capture_id,
            "source_root": str(source),
            "token_trace_dir": str(trace),
            "source_receipt_sha256": sha256(trace / "receipt.json"),
            "events": len(index),
            "model_fingerprint": cfg["model"]["fingerprint"],
            "files": files,
        },
    )
    return out
