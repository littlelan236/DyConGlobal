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


def read_repaired_session(root, repair_id, session_id, *, verify_dependencies=True):
    """Read a published pointwise repair without implicitly enabling training.

    Pass returned point rows to read_arrays(root, row). Recovery bindings remain
    explicit; identity-only and orphan points are not native-exact token evidence.
    """
    from core.io import identifier
    root = Path(root).resolve()
    base = root / 'capture-repairs' / identifier(repair_id) / 'sessions' / identifier(session_id)
    receipt = read_json(base / 'receipt.json')
    if receipt['repair_id'] != repair_id or receipt['session_id'] != session_id:
        raise ValueError('Repaired session receipt identity mismatch')
    files = dict(receipt['files'])
    if verify_dependencies:
        files.update(receipt['dependencies'])
    for relative, digest in files.items():
        if sha256(owned_path(root, relative)) != digest:
            raise ValueError('Repaired session checksum mismatch: ' + relative)
    index = read_json(base / 'point-index.json')
    if Path(index['tensor_root']).resolve() != root or index['session_id'] != session_id:
        raise ValueError('Repaired tensor root/session identity mismatch')
    return {'points': index['points'], 'coverage': read_json(base/'coverage.json'), 'receipt': receipt}


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


def matching_request_receipts(session_id, message, response, receipts):
    """Shared exact native identity join for live preparation and offline repair."""
    metadata = (message.get("extra", {}).get("request") or {}).get("dycon_capture", {})
    return [
        row for row in receipts
        if row["session_id"] == session_id
        and (not metadata.get("decision_id") or (
            row.get("decision_id") == metadata["decision_id"]
            and row.get("branch_id") == metadata["branch_id"]))
        and (row.get("query_id") == metadata["query_id"] if metadata.get("query_id")
             else row["step_id"] == metadata.get("step_id"))
        and (not metadata.get("api_attempt_id") or row.get("api_attempt_id") == metadata["api_attempt_id"])
        and ((metadata.get("decision_id") and row.get("decision_id") == metadata["decision_id"])
             or not response.get("id") or row["request_id"] == response["id"]
             or row["request_id"].startswith(response["id"] + "-"))
    ]


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
    classification = base / "branch-classification-v2.json"
    if classification.exists():
        proof = read_json(base / "branch-classification-v2-receipt.json")
        for relative, digest in proof["files"].items():
            if sha256(owned_path(root, relative)) != digest:
                raise ValueError("Branch classification receipt checksum mismatch")
        revision = read_json(classification)
        for relative, digest in revision["source_files"].items():
            if sha256(owned_path(root, relative)) != digest:
                raise ValueError("Branch classification evidence checksum mismatch")
        coverage = revision["coverage"]
        states = {(r["decision_id"], r["branch_id"]): r["close_capture"] for r in revision["branches"]}
        rows = [{**row, "close_capture": states[(row["decision_id"], row["branch_id"])]} for row in rows]
    return {"samples": rows, "coverage": coverage, "receipt": receipt}
