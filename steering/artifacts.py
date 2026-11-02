"""Checksummed immutable offline artifacts; no serving/runtime side effects."""

from __future__ import annotations

import json
from pathlib import Path

from core.io import hash_json, identifier, owned_path, read_json, sha256, write_json

SCHEMAS = {
    "steering-dataset": "consensus-branch-close-v1",
    "steering-cache": "branch-close-float32-v1",
    "steering-analysis": "binary-lda-ld1-v1",
}
REQUIRED = {
    "steering-dataset": {
        "dataset.json",
        "selection.json",
        "rollouts.json",
        "decisions.jsonl",
        "samples.jsonl",
        "excluded.jsonl",
        "summary.json",
        "dependencies.json",
    },
    "steering-cache": {"cache.json"},
    "steering-analysis": {
        "analysis.json",
        "selection.json",
        "points-metadata.jsonl",
        "points.csv",
        "summary.json",
        "report.md",
        "index.html",
    },
}


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def seal(directory, kind, artifact_id, identity):
    directory = Path(directory)
    identifier(artifact_id)
    files = {
        p.relative_to(directory).as_posix(): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "receipt.json"
    }
    if not REQUIRED[kind] <= files.keys():
        raise ValueError(f"Incomplete {kind}: {sorted(REQUIRED[kind] - files.keys())}")
    receipt = {
        "schema_version": 1,
        "schema": SCHEMAS[kind],
        "kind": kind,
        "artifact_id": artifact_id,
        "identity": identity,
        "files": files,
        "files_sha256": hash_json(files),
    }
    write_json(directory / "receipt.json", receipt)
    return receipt


def verify_local(directory, kind=None, artifact_id=None):
    directory = Path(directory).resolve()
    receipt = read_json(directory / "receipt.json")
    actual_kind = receipt.get("kind")
    if actual_kind not in SCHEMAS or receipt.get("schema_version") != 1:
        raise ValueError("Unsupported steering artifact schema")
    if receipt.get("schema") != SCHEMAS[actual_kind]:
        raise ValueError("Steering receipt schema mismatch")
    if kind is not None and actual_kind != kind:
        raise ValueError("Steering receipt kind mismatch")
    if artifact_id is not None and receipt.get("artifact_id") != identifier(artifact_id):
        raise ValueError("Steering receipt ID mismatch")
    files = receipt.get("files")
    if not isinstance(files, dict) or not REQUIRED[actual_kind] <= files.keys():
        raise ValueError("Steering receipt omits required files")
    if hash_json(files) != receipt.get("files_sha256"):
        raise ValueError("Steering receipt file index checksum mismatch")
    for relative, digest in files.items():
        path = owned_path(directory, relative)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Steering artifact checksum mismatch: {relative}")
    return receipt


def verify_dependencies(directory):
    dependencies = read_json(Path(directory) / "dependencies.json")
    for path, digest in dependencies["files"].items():
        if not Path(path).is_absolute() or not Path(path).is_file():
            raise ValueError(f"Frozen source dependency missing: {path}")
        if sha256(path) != digest:
            raise ValueError(f"Frozen source checksum mismatch: {path}")
    return len(dependencies["files"])


def show(cfg, scope, artifact_id, *, storage_root=None, with_tensors=False):
    from steering.datasets.paths import locate

    kind = "steering_dataset" if scope == "steering-dataset" else "steering_analysis"
    directory = locate(cfg, kind, artifact_id, storage_root)
    if scope == "steering-dataset":
        from steering.datasets.build import check_dataset

        return check_dataset(directory, with_tensors=with_tensors)
    from steering.visualization.service import check_analysis

    return check_analysis(directory)


verify = show
