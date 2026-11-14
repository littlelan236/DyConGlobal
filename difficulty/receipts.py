"""Shared receipt verification for difficulty datasets and their explicit parents."""
from pathlib import Path

from core.io import owned_path, read_json, sha256


def verify_receipt(root, directory, *, expected_sha256=None, seen=None):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    path = directory / "receipt.json"
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Parent receipt changed: {path}")
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic artifact dependency: {path}")
    seen = seen | {path}
    receipt = read_json(path)
    if receipt.get("state") != "complete" or not receipt.get("files"):
        raise ValueError(f"Incomplete artifact receipt: {path}")
    count = 0
    for name, expected in receipt["files"].items():
        if sha256(owned_path(directory, name)) != expected:
            raise ValueError(f"Artifact checksum mismatch: {directory}/{name}")
        count += 1
    evidence = directory / "source-evidence.json"
    sources = dict(receipt.get("source_files", {}))
    if "source-evidence.json" in receipt["files"]:
        sources.update(read_json(evidence).get("source_files", {}))
    for source, expected in sources.items():
        # Sources may be explicit NAS locators as well as project-local files.
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = owned_path(root, source)
        if sha256(source_path) != expected:
            raise ValueError(f"Bound source checksum mismatch: {source_path}")
        count += 1
    for parent in receipt.get("parents", []):
        from difficulty.storage import receipt_parent
        parent_path = receipt_parent(root, parent["path"])
        _, checked = verify_receipt(root, parent_path, expected_sha256=parent["receipt_sha256"], seen=seen)
        count += checked
    return receipt, count
