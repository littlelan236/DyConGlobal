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
    native_id = response.get('dycon_native_request_id')
    return [
        row for row in receipts
        if row["session_id"] == session_id
        and (native_id is None or row['request_id'] == native_id)
        and (not metadata.get("decision_id") or (
            row.get("decision_id") == metadata["decision_id"]
            and row.get("branch_id") == metadata["branch_id"]))
        and (row.get("query_id") == metadata["query_id"] if metadata.get("query_id")
             else row["step_id"] == metadata.get("step_id"))
        and (not metadata.get("api_attempt_id") or row.get("api_attempt_id") == metadata["api_attempt_id"])
        and (native_id is not None or not response.get("id") or row["request_id"] == response["id"]
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


def read_parallel_proof(root, session_id, *, rollout_id, trajectory_sha256):
    """Validate the original winner and candidate inventory against its receipt."""
    from core.io import identifier

    root = Path(root).resolve()
    base = root / "sessions" / identifier(session_id) / "parallel"
    view = read_branch_index(root, session_id)
    receipt = view["receipt"]
    if (receipt.get("rollout_id") != rollout_id
            or receipt.get("trajectory_sha256") != trajectory_sha256):
        raise ValueError("Branch receipt does not bind the canonical trajectory")
    files = receipt["files"]
    for name in ("branch-index.json", "branch-coverage.json"):
        if (base / name).relative_to(root).as_posix() not in files:
            raise ValueError("Branch receipt omits " + name)
    indexed = {}
    for row in view["samples"]:
        key = (row.get("decision_id"), row.get("branch_id"))
        if key[0] is None or key[1] is None or key in indexed:
            raise ValueError("Missing or duplicate branch identity")
        if row.get("rollout_id") != rollout_id or row.get("trajectory_sha256") != trajectory_sha256:
            raise ValueError("Branch index does not bind the canonical trajectory")
        indexed[key] = row
    decisions = []
    for relative in sorted(files):
        path = owned_path(root, relative)
        if path.parent != base / "decisions" or path.suffix != ".json":
            continue
        decision = read_json(path)
        if decision.get("session_id") != session_id:
            raise ValueError("Decision belongs to another session")
        decisions.append(decision)
    decision_ids, requests, checked = set(), set(), set()
    for decision in decisions:
        did = decision.get("decision_id")
        if not did or did in decision_ids:
            raise ValueError("Missing or duplicate decision identity")
        decision_ids.add(did)
        candidates = decision.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != decision.get("branch_count"):
            raise ValueError("Incomplete decision candidate inventory")
        ids = [candidate.get("branch_id") for candidate in candidates]
        if any(bid is None for bid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Missing or duplicate decision candidate")
        winner = decision.get("selected_branch_id")
        if winner is not None and winner not in ids:
            raise ValueError("Selected branch is absent from candidates")
        for candidate in candidates:
            key = (did, candidate["branch_id"])
            row = indexed.get(key)
            if (row is None or row.get("native_request_id") != candidate.get("native_request_id")
                    or row.get("score") != candidate.get("score")
                    or row.get("selected") != (candidate["branch_id"] == winner)):
                raise ValueError("Decision disagrees with its published branch index")
            request = candidate.get("native_request_id")
            if request:
                if request in requests:
                    raise ValueError("Native request is bound to multiple candidates")
                requests.add(request)
            checked.add(key)
    if checked != set(indexed):
        raise ValueError("Branch index contains unbound candidates")
    return {**view, "decisions": decisions}


def iter_arrays_grouped(samples):
    """Yield (key, arrays) from (key, root, row), opening each tar shard once.

    Reuses the same NPZ/checksum contract as read_arrays; row order is explicitly
    returned by key, rather than being inferred from physical I/O order.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for key, root, row in samples:
        locator = row.get("tensor_locator")
        relative = locator["shard"] if locator else row.get("path", row.get("tensor"))
        if not relative:
            raise ValueError("Hidden row has no tensor locator")
        groups[(str(Path(root).resolve()), relative, bool(locator))].append((key, row))
    for (root, relative, sharded), rows in sorted(groups.items()):
        if not sharded:
            digests = {row.get("sha256") for _, row in rows}
            if None in digests or len(digests) != 1:
                raise ValueError("Loose hidden file has missing/conflicting SHA")
            arrays = read_arrays(root, rows[0][1])
            for key, _row in rows:
                yield key, arrays
            continue
        digests = {row["tensor_locator"].get("shard_sha256") for _, row in rows}
        path = owned_path(root, relative)
        if None in digests or len(digests) != 1 or sha256(path) != next(iter(digests)):
            raise ValueError("Hidden tar shard checksum mismatch")
        with tarfile.open(path, "r:") as archive:
            members = {}
            wanted = {row["tensor_locator"]["member"] for _, row in rows}
            for member in archive.getmembers():
                if member.name in wanted:
                    if member.name in members:
                        raise ValueError("Duplicate hidden member name in tar shard")
                    members[member.name] = member
            for key, row in rows:
                locator = row["tensor_locator"]
                member = members.get(locator["member"])
                if member is None or not member.isfile():
                    raise ValueError("Hidden tar member is missing or not a regular file")
                data = archive.extractfile(member).read()
                if (locator["member_sha256"] != row.get("sha256") or
                        hashlib.sha256(data).hexdigest() != locator["member_sha256"]):
                    raise ValueError("Hidden tensor member checksum mismatch")
                with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                    yield key, {name: arrays[name] for name in arrays.files}
