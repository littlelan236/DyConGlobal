"""Read-only inventory for explicit legacy artifact selection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from core.io import now, sha256, write_json

SHA = re.compile(r"[0-9a-f]{64}")


def digests(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if "sha" in str(key).lower() and isinstance(item, str) and SHA.fullmatch(item):
                yield item
            yield from digests(item)
    elif isinstance(value, list):
        for item in value:
            yield from digests(item)


def inventory(root: Path):
    root = root.resolve()
    manifest = json.loads((root / "catalog/manifest.json").read_text(encoding="utf-8"))
    reference_sources = []
    reference_hashes = set()
    for base in (root / "difficulty", root / "enrichment", root / "hidden", root / "artifacts"):
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            if any(part in {"datasets", "outputs", "token-traces", "cache", "__pycache__"} for part in path.parts):
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            found = set(digests(value))
            if found:
                reference_sources.append({"path": str(path.relative_to(root)), "sha256_refs": len(found)})
                reference_hashes.update(found)
    rows = []
    for rollout_id, row in manifest["rollouts"].items():
        task = manifest["tasks"].get(row["task_key"], {})
        paths = {}
        for kind in ("trajectory", "patch"):
            rel = row[f"{kind}_path"]
            path = (root / rel).resolve()
            present = path.is_file()
            actual = sha256(path) if present else None
            expected = row[f"{kind}_sha256"]
            paths[kind] = {
                "path": rel,
                "exists": present,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_valid": present and actual == expected,
                "referenced_by_hash": expected in reference_hashes,
            }
        rows.append({
            "rollout_id": rollout_id,
            "task_key": row["task_key"],
            "instance_id": task.get("instance_id"),
            "partition": task.get("partition"),
            "run_id": row["run_id"],
            "attempt_id": row["attempt_id"],
            "migration_id": row.get("provenance", {}).get("migration_id"),
            "source": row.get("provenance", {}).get("source"),
            "model_fingerprint": row["model_fingerprint"],
            "precision_evidence": "unknown",
            "evaluation_status": row["evaluation_status"],
            "selected_evaluation_id": row["selected_evaluation_id"],
            "trajectory": paths["trajectory"],
            "patch": paths["patch"],
        })
    return {
        "schema_version": 1,
        "generated_at": now(),
        "catalog_revision": manifest["catalog_revision"],
        "rollout_count": len(rows),
        "reference_scan": reference_sources,
        "rollouts": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = inventory(Path(a.root))
    write_json(a.output, result)
    print(json.dumps({"output": str(Path(a.output).resolve()), "rollouts": result["rollout_count"],
                      "reference_files": len(result["reference_scan"])}, indent=2))


if __name__ == "__main__":
    main()
