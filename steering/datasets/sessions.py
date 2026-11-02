"""Published-session evidence only; no spool or startup discovery."""

from __future__ import annotations

from pathlib import Path

from core.io import hash_json, identifier, owned_path, read_json, sha256, write_json
from hidden.artifacts import read_branch_index


def add_dependency(dependencies, path, expected=None):
    path = Path(path).resolve()
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Source evidence checksum mismatch: {path}")
    if str(path) in dependencies and dependencies[str(path)] != digest:
        raise ValueError("Source evidence changed while building dataset")
    dependencies[str(path)] = digest
    return digest


def load_session(source, cfg, session_id, publication, snapshot, dependencies):
    sid = identifier(session_id)
    if sid.startswith(("startup", "smoke")):
        return None, "startup_or_smoke_session"
    export = cfg.get("capture", {}).get("export", {})
    if not export.get("enabled"):
        return None, "capture_export_disabled"
    nas = owned_path(cfg["paths"]["nas_root"], str(Path(export["root"]) / source["source_run_id"]))
    if publication is None:
        return None, "publication_not_acknowledged"
    if (
        publication.get("run_id") != source["source_run_id"]
        or Path(publication["nas_locator"]).resolve() != nas
    ):
        raise ValueError("Publication root/run differs from the source frozen config")
    entries = publication.get("sessions", [])
    matches = [
        entry
        for entry in entries
        if entry.get("rollout_id") == source["rollout_id"]
        and entry.get("trajectory_sha256") == source["trajectory_sha256"]
    ]
    same_sid = [entry for entry in entries if entry.get("session_id") == sid]
    if same_sid and any(entry not in matches for entry in same_sid):
        raise ValueError("Published session is bound to a different canonical trajectory")
    if not matches:
        return None, "unpublished_or_unfinalized_session"
    if len(matches) != 1 or matches[0].get("session_id") != sid:
        raise ValueError("Canonical rollout must bind one unambiguous finalized session")
    entry = matches[0]
    required = {
        "session.json",
        "finalization.json",
        "receipt.json",
        "parallel/branch-index.json",
        "parallel/branch-coverage.json",
        "parallel/branch-receipt.json",
    }
    if not required <= entry.get("files", {}).keys():
        raise ValueError("Published session omits required evidence files")
    base = nas / "sessions" / sid
    for name, proof in entry["files"].items():
        expected = owned_path(base, name)
        if Path(proof["nas_locator"]).resolve() != expected:
            raise ValueError("Published session file locator is not within the exact session")
        add_dependency(dependencies, expected, proof["sha256"])
    finalization = read_json(base / "finalization.json")
    for key, expected in (
        ("session_id", sid),
        ("rollout_id", source["rollout_id"]),
        ("trajectory_sha256", source["trajectory_sha256"]),
        ("model_fingerprint", source["model_fingerprint"]),
    ):
        if finalization.get(key) != expected:
            raise ValueError(f"Finalization identity mismatch: {key}")
    if Path(finalization["trajectory_path"]).resolve() != Path(source["trajectory_path"]).resolve():
        raise ValueError("Finalization trajectory path differs from canonical binding")
    session_receipt = read_json(base / "receipt.json")
    if session_receipt.get("session_id") != sid:
        raise ValueError("Session receipt identity mismatch")
    if session_receipt.get("status") not in {"stopped", "finalized", "complete"}:
        return None, "session_not_finished"
    view = read_branch_index(nas, sid)
    receipt = view["receipt"]
    for key in ("rollout_id", "trajectory_sha256"):
        if receipt.get(key) != source[key]:
            raise ValueError("Branch receipt canonical trajectory identity mismatch")
    for relative, digest in receipt["files"].items():
        add_dependency(dependencies, owned_path(nas, relative), digest)
    index, grouped = {}, {}
    for row in view["samples"]:
        key = (row["decision_id"], row["branch_id"])
        if key in index:
            raise ValueError("Duplicate decision/branch binding in published index")
        if (
            row.get("rollout_id") != source["rollout_id"]
            or row.get("trajectory_sha256") != source["trajectory_sha256"]
        ):
            raise ValueError("Branch-index canonical rollout binding mismatch")
        index[key] = row
        grouped.setdefault(row["decision_id"], []).append(row)
    inventory_path = nas / "inventory.json"
    inventory = read_json(inventory_path)
    add_dependency(dependencies, inventory_path)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("publication.json", entry),
        ("finalization.json", finalization),
        ("branch-receipt.json", receipt),
        ("coverage.json", view["coverage"]),
        ("inventory.json", inventory),
    ):
        write_json(snapshot / name, value)
    revision = base / "parallel/branch-classification-v2.json"
    classification_identity = None
    if revision.exists():
        proof_path = revision.with_name("branch-classification-v2-receipt.json")
        proof, revision_value = read_json(proof_path), read_json(revision)
        for relative, digest in {**proof["files"], **revision_value["source_files"]}.items():
            add_dependency(dependencies, owned_path(nas, relative), digest)
        add_dependency(dependencies, proof_path)
        classification_identity = {"revision_sha256": sha256(revision), "receipt_sha256": sha256(proof_path)}
        write_json(snapshot / "classification-receipt.json", proof)
        write_json(snapshot / "classification.json", revision_value)
    return {
        "root": nas,
        "base": base,
        "publication": entry,
        "finalization": finalization,
        "index": index,
        "grouped": grouped,
        "receipt": receipt,
        "inventory": inventory,
        "classification": classification_identity,
    }, None


def full_decision(session, embedded):
    did = identifier(embedded["decision_id"])
    path = session["base"] / "parallel/decisions" / (did + ".json")
    relative = path.relative_to(session["root"]).as_posix()
    if relative not in session["receipt"]["files"]:
        raise ValueError("Full decision is not covered by the published branch receipt")
    decision = read_json(path)
    for key, value in embedded.items():
        expected = decision.get(key)
        if key == "candidates":
            expected = [
                {k: v for k, v in candidate.items() if k != "response"}
                for candidate in decision.get("candidates", [])
            ]
            value = [{k: v for k, v in candidate.items() if k != "response"} for candidate in value]
        if expected != value:
            raise ValueError(f"Canonical embedded/full decision mismatch: {did}/{key}")
    count = decision.get("branch_count")
    candidates = decision.get("candidates")
    if type(count) is not int or count < 1 or not isinstance(candidates, list):
        raise ValueError("Decision has no complete candidate inventory")
    ids = [row.get("branch_id") for row in candidates]
    if any(type(bid) is not int for bid in ids) or sorted(ids) != list(range(count)):
        raise ValueError("Candidate IDs must cover 0..branch_count-1 exactly once")
    if any(row["branch_id"] not in ids for row in session["grouped"].get(did, [])):
        raise ValueError("Branch-index contains an undeclared candidate")
    return decision, sha256(path)


def choose_close(candidate, index_row, decision, session, message_index):
    if index_row is None:
        return None, "missing_branch_index", None
    if index_row.get("selected") != (candidate["branch_id"] == decision.get("selected_branch_id")):
        raise ValueError("Branch-index selected flag differs from historical decision")
    native = candidate.get("native_request_id")
    if native is None:
        return None, "missing_native_request_identity", None
    if index_row.get("native_request_id") != native:
        raise ValueError("Branch-index request identity differs from candidate")
    events = index_row.get("close_events") or []
    if len({event.get("sample_id") for event in events}) != len(events):
        raise ValueError("Duplicate close sample IDs in branch-index")
    sample_id = candidate.get("close_sample_id")
    if sample_id is not None:
        matches = [event for event in events if event.get("sample_id") == sample_id]
        if not matches:
            return None, "close_generated_hidden_missing", "explicit_close_sample_id"
        event, method = matches[0], "explicit_close_sample_id"
    else:
        if len(events) > 1:
            return None, "ambiguous_close_events", None
        if not events:
            reason = (
                "close_generated_hidden_missing"
                if candidate.get("absolute_position") is not None
                else "close_not_generated"
            )
            return None, reason, None
        event, method = events[0], "unique_full_identity"
    expected = {
        "session_id": decision["session_id"],
        "decision_id": decision["decision_id"],
        "branch_id": candidate["branch_id"],
        "request_id": native,
        "message_index": message_index,
        "step_id": decision.get("step_id"),
        "parent_prefix_sha256": decision.get("parent_prefix_sha256"),
        "query_id": decision.get("query_id"),
    }
    expected.update({k: v for k, v in (candidate.get("score_identity") or {}).items() if v is not None})
    for key, value in expected.items():
        if value is not None and event.get(key) is None:
            return None, "missing_close_identity_" + key, method
        if value is not None and event.get(key) != value:
            raise ValueError(f"Close event identity mismatch: {key}")
    if event.get("anchor") != "think_close" or event.get("phase") not in {
        "generation_input",
        "generation_output",
    }:
        raise ValueError("Bound close event is not a generated reasoning close")
    for key in ("absolute_position", "token_id"):
        if type(event.get(key)) is not int:
            return None, "missing_close_token_coordinates", method
        if candidate.get(key) is not None and candidate[key] != event[key]:
            raise ValueError(f"Candidate/hidden token coordinate mismatch: {key}")
    if event["token_id"] not in session["inventory"]["close_token_ids"]:
        raise ValueError("Hidden token is not a close token in the frozen inventory")
    if event.get("tokenizer_fingerprint") != session["inventory"]["tokenizer_fingerprint"]:
        raise ValueError("Hidden/token inventory tokenizer fingerprint mismatch")
    expected_sample = hash_json(
        {
            "session": decision["session_id"],
            "request": native,
            "anchor": "think_close",
            "position": event["absolute_position"],
        }
    )[:32]
    if event.get("sample_id") != expected_sample:
        raise ValueError(
            "Hidden sample ID mismatch: request="
            + str(native)
            + ", position="
            + str(event.get("absolute_position"))
            + ", actual="
            + str(event.get("sample_id"))
            + ", expected="
            + str(expected_sample)
        )
    if event.get("status") != "persisted":
        return None, "close_generated_hidden_not_persisted", method
    locator = event.get("tensor_locator")
    if locator:
        if locator.get("member_sha256") != event.get("sha256") or not locator.get("shard_sha256"):
            raise ValueError("Hidden locator/member checksums disagree or are missing")
        path = owned_path(session["root"], locator["shard"])
    else:
        if not event.get("sha256") or not (event.get("tensor") or event.get("path")):
            return None, "missing_hidden_locator_or_checksum", method
        path = owned_path(session["root"], event.get("tensor", event.get("path")))
    if not path.is_file():
        return None, "published_hidden_file_missing", method
    return event, None, method
