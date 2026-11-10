"""Read-only token publication adapter. Hidden tensors are deliberately not required."""
from pathlib import Path

from core.io import identifier, owned_path, read_json, sha256
from hidden.artifacts import read_request


def load_token_session(source, cfg, sid, dependencies, *, audit=False):
    sid = identifier(sid)
    if sid.startswith(("startup", "smoke")):
        return None, "startup_or_smoke_session"
    export = cfg.get("capture", {}).get("export", {})
    if not export.get("enabled"):
        return None, "capture_export_disabled"
    nas = owned_path(cfg["paths"]["nas_root"], str(Path(export["root"]) / source["source_run_id"]))
    pubpath = Path(source["run_directory"]) / "hidden/publication.json"
    if audit and not pubpath.is_file():
        return None, "publication_proof_unavailable"
    pub = read_json(pubpath) if pubpath.is_file() else {}
    if audit and (pub.get("schema_version") != 1 or pub.get("state") not in (None, "published")
            or pub.get("run_id") != source["source_run_id"] or Path(pub["nas_locator"]).resolve() != nas):
        raise ValueError("Token publication schema/state/run/root mismatch")
    # Publication entries are only required by the opted-in source audit.
    # Legacy publications may use a different sessions representation.
    entries = [x for x in pub.get("sessions", []) if x.get("session_id") == sid] if audit else []
    if audit and not entries:
        return None, "unpublished_session"
    if audit and len(entries) != 1:
        raise ValueError("Ambiguous published session")
    entry = entries[0] if entries else {}
    for key in ("rollout_id", "trajectory_sha256") if audit else ():
        if entry.get(key) != source[key]:
            raise ValueError("Published session canonical identity mismatch: " + key)
    base = nas / "sessions" / sid
    required = {"session.json", "receipt.json", "finalization.json", "parallel/branch-index.json", "parallel/branch-coverage.json", "parallel/branch-receipt.json"}
    if audit and not required <= entry.get("files", {}).keys():
        return None, "publication_token_proof_incomplete"
    # The publication entry is frozen by the dataset; later unrelated sessions may be appended.
    for name in sorted(required) if audit else ():
        proof = entry["files"][name]
        path = owned_path(base, name)
        if Path(proof["nas_locator"]).resolve() != path:
            raise ValueError("Publication file locator escapes exact session")
        if audit:
            if sha256(path) != proof["sha256"]:
                raise ValueError("Published session file checksum mismatch")
        dependencies[str(path.resolve())] = proof["sha256"]
    # Missing native session evidence is an inventory exclusion, not an I/O crash.
    native_required = ("finalization.json", "receipt.json", "parallel/branch-receipt.json", "parallel/branch-index.json")
    if any(not (base / name).is_file() for name in native_required):
        return None, "native_session_evidence_unavailable"
    final = read_json(base / "finalization.json")
    for key, expected in (("session_id", sid), ("rollout_id", source["rollout_id"]),
                          ("trajectory_sha256", source["trajectory_sha256"]), ("model_fingerprint", source["model_fingerprint"])):
        if audit and final.get(key) != expected:
            raise ValueError("Finalization identity mismatch: " + key)
    if audit and Path(final["trajectory_path"]).resolve() != Path(source["trajectory_path"]).resolve():
        raise ValueError("Finalization trajectory path mismatch")
    stopped = read_json(base / "receipt.json")
    if audit and (stopped.get("session_id") != sid or stopped.get("status") not in {"stopped", "complete", "finalized"}):
        return None, "session_not_finished"
    receipt = read_json(base / "parallel/branch-receipt.json")
    if audit:
        for key in ("rollout_id", "trajectory_sha256"):
            if receipt.get(key) != source[key]:
                raise ValueError("Branch receipt canonical identity mismatch")
    # Only token/action evidence belongs to this contract, never hidden tensor availability.
    names = receipt["files"] if audit else {"sessions/%s/parallel/branch-index.json" % sid: None,
                                             "sessions/%s/parallel/branch-coverage.json" % sid: None}
    for relative, digest in names.items():
        path = owned_path(nas, relative)
        if audit:
            if not path.is_relative_to(base / "parallel"):
                raise ValueError("Branch receipt file is outside exact parallel session")
            if sha256(path) != digest:
                raise ValueError("Branch receipt source checksum mismatch")
        dependencies[str(path.resolve())] = digest
    rows = read_json(base / "parallel/branch-index.json")
    index = {}
    grouped = {}
    for row in rows:
        key = (row.get("decision_id"), row.get("branch_id"))
        if audit and (key in index or row.get("rollout_id") != source["rollout_id"] or row.get("trajectory_sha256") != source["trajectory_sha256"]):
            raise ValueError("Branch-index duplicate or canonical identity mismatch")
        index[key] = row
        grouped.setdefault(key[0], []).append(row)
    inventory_path = nas / "inventory.json"
    inventory = read_json(inventory_path)
    inventory_digest = sha256(inventory_path) if audit else None
    dependencies[str(inventory_path.resolve())] = inventory_digest
    return {"root": nas, "base": base, "receipt": receipt, "index": index, "grouped": grouped,
            "inventory": inventory, "finalization": final, "publication": entry,
            "proof_version": "published_token_session_v1", "audit_enabled": audit}, None


def native_branch(session, decision, candidate, message_index, dependencies):
    """Missing data is returned as a reason; conflicting identities always raise."""
    if not candidate.get("ledger_ref") or not candidate.get("native_request_id"):
        return None, "missing_native_request_evidence"
    ledger = owned_path(session["root"], candidate["ledger_ref"])
    if not ledger.is_file():
        return None, "native_ledger_missing"
    if session["audit_enabled"]:
        if sha256(ledger) != candidate["ledger_sha256"]:
            raise ValueError("Request ledger checksum mismatch")
        value = read_request(session["root"], candidate)
    else:
        value = {**candidate, **read_json(ledger)}
    dependencies[str(ledger.resolve())] = candidate.get("ledger_sha256")
    expected = {"request_id": candidate["native_request_id"], "session_id": decision["session_id"],
                "decision_id": decision["decision_id"], "branch_id": candidate["branch_id"],
                "step_id": decision["step_id"], "parent_prefix_sha256": decision["parent_prefix_sha256"],
                "query_id": decision.get("query_id"), "message_index": message_index,
                "api_attempt_id": candidate.get("api_attempt_id") or (candidate.get("score_identity") or {}).get("api_attempt_id")}
    for key, expected_value in expected.items():
        if not session["audit_enabled"]:
            break
        if expected_value is not None and value.get(key) != expected_value:
            raise ValueError("Native request identity mismatch: " + key)
    row = session["index"].get((decision["decision_id"], candidate["branch_id"]))
    if row is None:
        return None, "missing_branch_index"
    if session["audit_enabled"] and (row.get("native_request_id") != candidate["native_request_id"] or row.get("selected") != (candidate["branch_id"] == decision.get("selected_branch_id"))):
        raise ValueError("Branch-index native/selected identity mismatch")
    if session["audit_enabled"] and value.get("tokenizer_fingerprint") != session["inventory"]["tokenizer_fingerprint"]:
        raise ValueError("Native tokenizer/inventory mismatch")
    for key in ("request_sha256", "output_sha256"):
        if session["audit_enabled"] and candidate.get(key) is not None and value.get(key) != candidate[key]:
            raise ValueError("Candidate/native hash mismatch: " + key)
    if not value.get("prompt_token_ids") or not value.get("output_token_ids"):
        return None, "missing_native_tokens"
    if value.get("finish_reason") not in {"stop", "tool_calls", "function_call"}:
        return value, "incomplete_native_finish"
    return value, None
