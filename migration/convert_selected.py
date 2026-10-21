"""One-shot conversion of explicitly selected legacy artifacts.

The new runtime never imports this module. Every copied artifact gets a new
run-local descriptor and a provenance record; unselected legacy data is ignored.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.io import hash_json, identifier, now, sha256, write_json


def convert(root, source_manifest, target_run, selection_file):
    root = Path(root).resolve()
    source = Path(source_manifest).resolve()
    run = root / "runs" / identifier(target_run)
    selection = __import__("json").loads(Path(selection_file).read_text(encoding="utf-8"))
    rows = selection.get("artifacts", selection)
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection must contain a non-empty artifacts list")
    converted = []
    for row in rows:
        kind = row["kind"]
        source_spec = Path(row["source_path"])
        source_path = (source_spec if source_spec.is_absolute() else root / source_spec).resolve()
        if not source_path.is_file():
            raise ValueError(f"Invalid or missing source artifact: {source_path}")
        if row.get("source_sha256") and sha256(source_path) != row["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {source_path}")
        artifact_id = identifier(row["artifact_id"])
        if kind == "trajectory":
            target = run / "rollout" / "trajectories" / artifact_id / "raw.traj.json"
        elif kind == "patch":
            target = run / "rollout" / "patches" / artifact_id / "patch.diff"
        elif kind == "enrichment":
            target = run / "enrichment" / row.get("relative_path", f"{artifact_id}.json")
        elif kind == "hidden_metadata":
            target = run / "hidden" / row.get("relative_path", f"{artifact_id}.json")
        elif kind == "harness":
            target = run / "harness" / row.get("relative_path", f"{artifact_id}.json")
        else:
            raise ValueError(f"Unsupported artifact kind: {kind}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        converted.append({
            "kind": kind,
            "artifact_id": artifact_id,
            "source_path": str(source_path.relative_to(root)),
            "source_sha256": sha256(source_path),
            "local_locator": str(target.relative_to(root)),
            "content_sha256": sha256(target),
            "origin": "migrated",
        })
    receipt = {
        "schema_version": 1,
        "migration_run_id": target_run,
        "source_manifest": str(source),
        "selection_sha256": hash_json(selection),
        "converted": converted,
        "finished_at": now(),
    }
    write_json(run / "migration-receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--target-run", required=True)
    p.add_argument("--selection", required=True)
    a = p.parse_args()
    import json
    print(json.dumps(convert(a.root, a.source_manifest, a.target_run, a.selection), indent=2))


if __name__ == "__main__":
    main()
