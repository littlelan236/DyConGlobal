"""Shared, hash-checked readers for legacy loose files and tar shard members."""

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np

from core.io import owned_path, read_json, sha256


def artifact_bytes(root, row):
    locator = row.get("tensor_locator")
    if locator:
        path = owned_path(root, locator["shard"])
        # Member hashes are checked on every read; the complete shard is checked
        # once at publication (and explicitly by dataset audits).
        with tarfile.open(path, "r:") as archive:
            member = archive.getmember(locator["member"])
            if not member.isfile():
                raise ValueError("Tensor member must be a regular file")
            data = archive.extractfile(member).read()
        digest = locator["member_sha256"]
    else:
        data = owned_path(root, row.get("path", row.get("tensor"))).read_bytes()
        digest = row.get("sha256")
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Hidden tensor checksum mismatch")
    return data


def read_arrays(root, row):
    with np.load(io.BytesIO(artifact_bytes(root, row)), allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def read_request(root, evidence):
    if evidence.get("ledger_ref"):
        path = owned_path(root, evidence["ledger_ref"])
        if sha256(path) != evidence["ledger_sha256"]:
            raise ValueError("Request ledger checksum mismatch")
        value = read_json(path)
        if value.get("inventory_ref"):
            inventory = read_json(owned_path(root, value["inventory_ref"]))
            if hash_inventory(inventory) != value["inventory_sha256"]:
                raise ValueError("Request inventory checksum mismatch")
            needed = {
                k: inventory[k]
                for k in ("newline_token_ids", "close_token_ids", "tokenizer_fingerprint")
                if k in inventory
            }
            value = {**needed, **value}
        return {**evidence, **value}
    return evidence


def hash_inventory(value):
    from core.io import hash_json

    return hash_json(value)


def event_records(root):
    seen = {}
    for path in sorted((Path(root) / "index").glob("*.json")):
        value = read_json(path)
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            sid = row["sample_id"]
            if sid not in seen:
                seen[sid] = row
                yield path, row


def response_records(trajectory):
    """Yield successful and FormatError model responses, preserving message index."""
    for index, message in enumerate(trajectory.get("messages", [])):
        extra = message.get("extra") or {}
        if isinstance(extra.get("response"), dict):
            yield index, message, extra["response"]


def read_branch_index(root, session_id):
    """Read a frozen all-candidate view, without promoting predictions to GT.

    Each close_events item can be passed directly to read_arrays(root, event).
    NAS publication is required: no in-flight spool data are silently accepted.
    """
    from core.io import identifier

    base = Path(root) / "sessions" / identifier(session_id) / "parallel"
    receipt = read_json(base / "branch-receipt.json")
    for relative, digest in receipt["files"].items():
        if sha256(owned_path(root, relative)) != digest:
            raise ValueError("Branch evidence checksum mismatch")
    rows = read_json(base / "branch-index.json")
    coverage = read_json(base / "branch-coverage.json")
    return {"samples": rows, "coverage": coverage, "receipt": receipt}
