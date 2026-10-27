"""Offline repair of retained close-only capture evidence."""

from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import numpy as np

from application.execution.evidence import RunEvidence
from application.execution.run_plan import load_plan
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_run
from enrichment.online_trace import reasoning_targets
from hidden.artifacts import artifact_bytes, event_records, matching_request_receipts, read_arrays, read_request, response_records
from runtime.serving.spool import CaptureSpool


class EvidenceConflict(ValueError):
    """Repair refuses contradictory or incomplete identity evidence."""


def _session(result):
    return identifier("cap-" + result["sample_key"] + "-" + result["attempt_id"])


def _requests(roots):
    output = {}
    proofs = {}
    for root in roots:
        paths = sorted({*(root / "requests").glob("*.json"), *(root / "sessions").glob("*/requests/*.json")})
        for path in paths:
            digest = sha256(path)
            proofs[str(path)] = digest
            value = read_request(root, {"ledger_ref": str(path.relative_to(root)), "ledger_sha256": digest})
            if value.get("inventory_ref"):
                inventory = owned_path(root, value["inventory_ref"])
                proofs[str(inventory)] = sha256(inventory)
            if value.get("request_sha256") != hash_json(value["prompt_token_ids"]):
                raise EvidenceConflict("Native prompt token hash mismatch")
            if value.get("output_sha256") != hash_json(value["output_token_ids"]):
                raise EvidenceConflict("Native output token hash mismatch")
            if value["request_id"] in output and output[value["request_id"]] != value:
                raise EvidenceConflict("Conflicting native request receipt")
            output[value["request_id"]] = value
    return output, proofs


def _events(roots):
    output = {}
    proofs = {}
    for root in roots:
        for path in sorted((root / "index").glob("*.json")):
            value = read_json(path)
            proofs[str(path)] = sha256(path)
            for row in value if isinstance(value, list) else [value]:
                prior = output.get(row["sample_id"], [])
                if prior:
                    previous = prior[0][1]
                    identity = ("session_id", "run_id", "task_id", "request_id", "query_id",
                                "api_attempt_id", "absolute_position", "anchor", "token_id", "sha256")
                    if any(previous.get(k) != row.get(k) for k in identity):
                        raise EvidenceConflict("Conflicting duplicate sample metadata")
                output.setdefault(row["sample_id"], []).append((root, row))
    return output, proofs


def _audit(root, run_id, cfg, evidence, result_path, roots, requests, events):
    result = read_json(result_path)
    rid = identifier(result["rollout_id"])
    row = evidence["rollouts"][rid]
    task = evidence["tasks"][row["task_key"]]
    trajectory = owned_path(root, row["trajectory_path"])
    if not trajectory.is_relative_to(root / "runs" / run_id):
        raise EvidenceConflict("Canonical trajectory belongs to another run")
    if row["run_id"] != run_id or sha256(trajectory) != row["trajectory_sha256"]:
        raise EvidenceConflict("Canonical trajectory identity/checksum mismatch")
    if result["instance_id"] != task["instance_id"] or result["attempt_id"] != row["attempt_id"]:
        raise EvidenceConflict("Sample result belongs to another task/attempt")
    if row.get("sample_key") not in {None, result["sample_key"]}:
        raise EvidenceConflict("Sample result/evidence identity mismatch")
    sid = _session(result)
    session_files = [base / "sessions" / sid / "session.json" for base in roots]
    session_paths = [p for p in session_files if p.exists()]
    if not session_paths:
        raise EvidenceConflict("Missing session metadata: " + sid)
    session = read_json(session_paths[0])
    if any(read_json(path) != session for path in session_paths):
        raise EvidenceConflict("Conflicting session metadata")
    if session["session_id"] != sid or session["metadata"]["session_id"] != sid:
        raise EvidenceConflict("Session ID mismatch")
    if session["metadata"]["run_id"] != run_id or session["metadata"]["task_id"] != task["instance_id"]:
        raise EvidenceConflict("Session belongs to another run/task")
    responses = list(response_records(read_json(trajectory)))
    bindings, missing, expected, observed, selected = [], [], set(), set(), []
    action_step, used_requests = 0, set()
    tensor_proofs = {str(trajectory): sha256(trajectory), str(result_path): sha256(result_path)}
    tensor_proofs.update({str(path): sha256(path) for path in session_paths})
    session_requests = [r for r in requests.values() if r.get("session_id") == sid]
    event_lookup = {}
    for event_rows in events.values():
        for base, event in event_rows:
            if event.get("session_id") == sid and event.get("status") != "failed":
                key = (event.get("request_id"), event.get("absolute_position"), event.get("anchor"))
                event_lookup.setdefault(key, []).append((base, event))
    for index, message, response in responses:
        md = (message.get("extra", {}).get("request") or {}).get("dycon_capture", {})
        if md.get("session_id") != sid or md.get("run_id") != run_id or md.get("task_id") != task["instance_id"]:
            raise EvidenceConflict("Canonical response capture metadata mismatch")
        valid_action = message.get("role") == "assistant"
        candidates = matching_request_receipts(sid, message, response, session_requests)
        if len(candidates) != 1:
            missing.append({"message_index": index, "candidate_count": len(candidates)})
            action_step += int(valid_action)
            continue
        request = candidates[0]
        if request["request_id"] in used_requests:
            raise EvidenceConflict("Duplicate native request binding")
        used_requests.add(request["request_id"])
        if request.get("run_id") != run_id or request.get("task_id") != task["instance_id"]:
            raise EvidenceConflict("Native request belongs to another task/run")
        if request.get("anchors") != ["think_close"] or not request.get("close_token_ids"):
            raise EvidenceConflict("Missing original close-only token inventory")
        if response.get("dycon_native_request_id") not in {None, request["request_id"]}:
            raise EvidenceConflict("Existing native request binding conflicts")
        if response.get("prompt_token_ids") is not None and response["prompt_token_ids"] != request["prompt_token_ids"]:
            raise EvidenceConflict("Canonical/native prompt token disagreement")
        choice = (response.get("choices") or [{}])[0]
        token_ids = choice.get("token_ids", (choice.get("provider_specific_fields") or {}).get("token_ids"))
        if token_ids is not None and token_ids != request["output_token_ids"]:
            raise EvidenceConflict("Canonical/native output token disagreement")
        bindings.append({"message_index": index, "agent_step_index": action_step,
                         "is_action_response": valid_action, "request": request})
        action_step += int(valid_action)
        targets, availability = reasoning_targets(request, request.get("open_token_ids", [248068]),
                                                   request.get("close_token_ids", [248069]))
        expected.update((request["request_id"], pos, anchor) for pos, anchor in targets)
        for pos, anchor in targets:
            candidates = event_lookup.get((request["request_id"], pos, anchor), [])
            if len({e["sample_id"] for _, e in candidates}) > 1:
                raise EvidenceConflict("Ambiguous capture event")
            for base, event in candidates:
                try:
                    arrays = read_arrays(base, event)
                except FileNotFoundError:
                    continue
                tokens = request["prompt_token_ids"] + request["output_token_ids"]
                if pos >= len(tokens) - 1 or tokens[pos] != event["token_id"]:
                    raise EvidenceConflict("Target token coordinate/followup mismatch")
                if event.get("request_sha256") != request["request_sha256"]:
                    raise EvidenceConflict("Captured tensor prompt hash mismatch")
                for field in ("query_id", "api_attempt_id", "task_id", "run_id"):
                    if event.get(field) != request.get(field):
                        raise EvidenceConflict("Captured tensor native identity mismatch: " + field)
                if event.get("model_fingerprint") not in {None, row["model_fingerprint"]}:
                    raise EvidenceConflict("Captured tensor model mismatch")
                locator = event.get("tensor_locator")
                tensor_path = owned_path(base, locator["shard"] if locator else event.get("path", event.get("tensor")))
                if str(tensor_path) not in tensor_proofs:
                    tensor_proofs[str(tensor_path)] = sha256(tensor_path)
                if locator and tensor_proofs[str(tensor_path)] != locator["shard_sha256"]:
                    raise EvidenceConflict("Tensor shard checksum mismatch")
                if len(arrays) != len(cfg["capture"]["layers"]):
                    raise EvidenceConflict("Incomplete tensor layers")
                if event.get("storage_shapes") and set(arrays) != set(event["storage_shapes"]):
                    raise EvidenceConflict("Tensor layer names mismatch")
                for name, array in arrays.items():
                    if not array.size or not np.isfinite(array).all():
                        raise EvidenceConflict("Invalid tensor data")
                    if name in event.get("storage_shapes", {}) and list(array.shape) != event["storage_shapes"][name]:
                        raise EvidenceConflict("Tensor shape mismatch")
                    if name in event.get("storage_dtypes", {}) and str(array.dtype) != event["storage_dtypes"][name]:
                        raise EvidenceConflict("Tensor dtype mismatch")
                observed.add((request["request_id"], pos, anchor))
                selected.append((base, event))
                break
    coverage = {"expected": len(expected), "observed": len(observed), "missing": sorted(expected - observed),
                "unbound_responses": len(missing), "expected_count_complete": not missing}
    return {"rollout_id": rid, "session_id": sid, "task_id": task["instance_id"],
            "trajectory_path": str(trajectory), "trajectory_sha256": row["trajectory_sha256"],
            "model_fingerprint": row["model_fingerprint"], "bindings": bindings, "missing_bindings": missing,
            "coverage": coverage, "complete_candidate": bool(responses) and not missing and not coverage["missing"],
            "session": session, "events": selected, "source_proofs": tensor_proofs,
            "original_capture_errors": result.get("capture_errors", [])}


def _public(audit):
    return {key: value for key, value in audit.items() if key not in {
        "bindings", "events", "session", "source_proofs"}}


def _reuse_finalization(roots, sid, proposed, coverage):
    """Reuse identical scientific evidence without changing its original provenance."""
    ignored = {"repair_id", "repaired_at"}
    selected = None
    for base in roots:
        path = base / "sessions" / sid / "finalization.json"
        if path.exists():
            prior = read_json(path)
            if ({k: v for k, v in prior.items() if k not in ignored}
                    != {k: v for k, v in proposed.items() if k not in ignored}):
                raise EvidenceConflict("Existing finalization evidence conflicts: " + sid)
            if selected is not None and prior != selected:
                raise EvidenceConflict("Existing finalization provenance conflicts: " + sid)
            selected = prior
        path = base / "sessions" / sid / "coverage.json"
        if path.exists() and read_json(path) != coverage:
            raise EvidenceConflict("Existing coverage evidence conflicts: " + sid)
    return selected if selected is not None else proposed


@exclusive_run
def repair_online(cfg, run_id, *, rollout_ids=(), session_ids=(), apply=False, repair_id=None, pointwise=False):
    """Audit partial sessions; explicit apply writes a separate repair namespace."""
    root = Path(cfg["project_root"]).resolve()
    loaded = load_plan(root, run_id)
    cfg = loaded["config"]
    if cfg.get("parallel", {}).get("enabled") and cfg["parallel"].get("branch_count", 1) > 1:
        raise ValueError("Repair supports close-only capture without parallel branches")
    if cfg["capture"]["anchors"] != ["think_close"] or not cfg["capture"]["export"]["enabled"]:
        raise ValueError("Repair requires persisted close-only capture")
    if apply and not repair_id:
        raise ValueError("--apply requires --repair-id")
    if pointwise:
        from hidden.repair_points import recover_points
        return recover_points(cfg, run_id, rollout_ids=rollout_ids, session_ids=session_ids,
                              apply=apply, repair_id=repair_id)
    evidence = RunEvidence(root, run_id).read()
    spool = root / cfg["capture"]["export"]["spool_root"] / run_id
    nas = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
    roots = [p for p in (spool, nas) if p.is_dir()]
    print("capture repair: verifying native request receipts", flush=True)
    requests, request_proofs = _requests(roots)
    print("capture repair: verifying event metadata", flush=True)
    events, event_proofs = _events(roots)
    selected_ids, selected_sessions = set(rollout_ids), set(session_ids)
    audits = []
    for path in sorted((loaded["run"] / "rollout/tasks").glob("*/repetitions/*/result.json")):
        result = read_json(path)
        if result.get("capture_status") != "partial":
            continue
        sid = _session(result)
        if selected_ids and result["rollout_id"] not in selected_ids:
            continue
        if selected_sessions and sid not in selected_sessions:
            continue
        audits.append(_audit(root, run_id, cfg, evidence, path, roots, requests, events))
        print("capture repair: audited " + str(len(audits)) + " session(s): " + sid, flush=True)
    if (selected_ids or selected_sessions) and not audits:
        raise ValueError("No selected partial sessions")
    report = {"schema_version": 1, "run_id": run_id, "repair_id": repair_id,
              "apply": apply, "source_proofs": {**request_proofs, **event_proofs},
              "summary": dict(Counter("repairable" if a["complete_candidate"] else "partial" for a in audits)),
              "sessions": [_public(a) for a in audits], "created_at": now()}
    if not apply:
        return report
    out = root / "runs" / run_id / "hidden" / "capture-repairs" / identifier(repair_id)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "audit.json", report)
    write_json(out / "progress.json", {"stage": "publishing", "total": len(audits),
                                      "published": [], "updated_at": now()})
    spool = CaptureSpool(root, cfg["capture"]["export"], run_id,
                         cfg["paths"]["nas_root"], cfg["images"]["reserve_bytes"])
    try:
        spool.recover()
        published = []
        for audit in audits:
            if not audit["complete_candidate"]:
                continue
            sid = audit["session_id"]
            finalization = {
                "schema_version": 2,
                "run_id": run_id,
                "rollout_id": audit["rollout_id"],
                "trajectory_path": audit["trajectory_path"],
                "trajectory_sha256": audit["trajectory_sha256"],
                "model_fingerprint": audit["model_fingerprint"],
                "accepted_native_request_ids": [b["request"]["request_id"] for b in audit["bindings"]],
                "responses": {b["request"]["request_id"]: {k: b[k] for k in
                    ("message_index", "agent_step_index", "is_action_response")}
                    for b in audit["bindings"]},
                "repair_id": repair_id,
                "repaired_at": now(),
            }
            finalization = _reuse_finalization(roots, sid, finalization, audit["coverage"])
            spool.write_json(f"sessions/{sid}/finalization.json", finalization, immutable=True)
            spool.write_json(f"sessions/{sid}/coverage.json", audit["coverage"], immutable=True)
            published.append({"session_id": sid, "published": spool.publish_session(
                sid, cfg["capture"]["export"]["upload_max_elapsed_seconds"]
            )})
            write_json(out / "progress.json", {"stage": "publishing", "total": len(audits),
                                              "published": published, "updated_at": now()})
            print("capture repair: publication " + str(len(published)) + "/" + str(len(audits))
                  + " " + sid + " " + str(published[-1]["published"]), flush=True)
        report["published"] = published
    finally:
        spool.close(cfg["capture"]["export"]["upload_max_elapsed_seconds"])
    # Persist the immutable audit after publication attempts; producer evidence
    # remains unchanged and consumers can inspect this repair receipt separately.
    write_json(out / "report.json", report)
    write_json(out / "source-proofs.json", report["source_proofs"])
    return report
