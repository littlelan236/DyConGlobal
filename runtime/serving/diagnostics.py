"""Bounded capture diagnostics, independent of capture outcome decisions."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

from core.io import write_json

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_SAFE_MESSAGE = re.compile(r"^[A-Za-z0-9_ .:/-]{1,160}$")
_SENSITIVE = re.compile(
    r"(?i)authorization|api.?key|password|secret|bearer|prompt|output|"
    r"tensor|token|payload|credential|sk-|xox"
)
_OPERATIONAL_MESSAGES = frozenset({"capture failed", "server failed", "request failed"})


def sanitize_error_text(value, limit=512):
    """Retain terse operational descriptions, never arbitrary exception text."""
    text = str(value or "")
    if text not in _OPERATIONAL_MESSAGES or not _SAFE_MESSAGE.fullmatch(text) or _SENSITIVE.search(text):
        return "[redacted diagnostic text]"
    return text.encode("utf-8")[:limit].decode("utf-8", "ignore")


def safe_response_summary(body, limit=512):
    """Copy only known scalar error fields from JSON; omit opaque response bodies."""
    try:
        value = json.loads(bytes(body).decode("utf-8"))
    except (TypeError, UnicodeDecodeError, ValueError):
        return ""
    if not isinstance(value, dict):
        return ""
    safe = {}
    for key in ("error_type", "reason_code", "stage", "correlation_id"):
        item = value.get(key)
        if isinstance(item, str) and _IDENTIFIER.fullmatch(item) and not _SENSITIVE.search(item):
            safe[key] = item
    for key in ("error", "detail", "message"):
        item = value.get(key)
        if isinstance(item, str):
            safe[key] = sanitize_error_text(item, 160)
    encoded = json.dumps(safe, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > limit:
        for key in ("message", "detail", "error"):
            safe.pop(key, None)
        encoded = json.dumps(safe, ensure_ascii=True, separators=(",", ":"))
    return encoded if len(encoded.encode("utf-8")) <= limit else "{}"


def response_error_type(summary):
    try:
        value = json.loads(summary)
        return value.get("error_type") if isinstance(value, dict) else None
    except (TypeError, ValueError):
        return None


def safe_stack(exc):
    """Record stack locations only; source lines may contain user data."""
    return [
        {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
        for frame in traceback.extract_tb(exc.__traceback__)[-8:]
    ]


def write_attempt_diagnostics(output, identity, finalized=None, errors=()):
    """Write a small derived artifact without affecting the original outcome."""
    try:
        summary = {
            "schema_version": 1,
            **identity,
            "errors": [item for item in errors if isinstance(item, dict)][:32],
        }
        server = (finalized or {}).get("capture_diagnostics") or {}
        stages = server.get("stages")
        if isinstance(stages, dict):
            summary.update(stages)
        write_json(Path(output) / "capture-diagnostics.json", summary)
    except Exception:
        pass


def bounded_records(project_root, run_id, execution_id, records, *, kind="point", limit=256):
    """Share shutdown-only anomaly reservations across sessions and processes."""
    from core.io import hash_json, identifier, read_json, run_path
    from core.locking import file_lock

    import os
    execution_id = execution_id or os.environ.get("DYCON_EXECUTION_ATTEMPT") or "standalone"
    base = run_path(project_root, identifier(str(run_id))) / "execution" / "attempts" / identifier(str(execution_id))
    path = base / "capture-diagnostic-budget.json"
    kept = []
    with file_lock(base / "capture-diagnostic-budget.lock"):
        value = read_json(path) if path.exists() else {"schema_version": 1, "keys": []}
        keys = set(value["keys"])
        for record in records:
            identity = ([record.get("request_id"), record.get("absolute_position"), record.get("anchor")]
                        if kind == "point" else [record.get("correlation_id")])
            key = hash_json([kind, identity])
            if key in keys or len(keys) < limit:
                keys.add(key)
                kept.append(record)
        value.update(keys=sorted(keys), retained_count=len(keys))
        write_json(path, value)
    return kept


def cap_stage_summary(project_root, summary):
    records = summary.get("missing", [])
    try:
        kept = bounded_records(project_root, summary["run_id"], summary.get("execution_attempt_id"), records)
    except Exception:
        kept = []
        summary["reason_code"] = "diagnostic_budget_unavailable"
    summary["missing"] = kept
    summary["truncated_count"] = max(0, summary.get("missing_count", len(records)) - len(kept))
    summary["by_request"] = {}
    return summary


def queue_summary(snapshot):
    """Counts only: never copy another session or arbitrary writer failures."""
    return {key: snapshot.get(key) for key in ("queued", "persisted", "failed_count",
        "spill_pending", "pending_total", "submitted", "spilled_total", "session_metrics")}


def write_index_diagnostics(root, output, requests, expected, observed, persisted):
    """Use the actual accepted index; unknown upstream evidence stays unknown."""
    from collections import Counter
    from core.io import read_json
    import os

    stages = ("expected", "selector_seen", "capture_row", "submit", "persist", "indexed")
    totals = dict.fromkeys(stages, 0)
    anchors, sessions, source = {}, {}, {}
    missing, reasons, unknown = [], Counter(), Counter()
    attempt_documents = None
    for rid, position, anchor in sorted(expected):
        request = requests[rid]
        sid = request["session_id"]
        if sid not in source:
            path = Path(root) / "sessions" / sid / "finalization-diagnostics.json"
            try:
                source[sid] = read_json(path).get("stages", {})
            except Exception:
                source[sid] = {}
        document = source[sid]
        point = (rid, position, anchor)
        indexed = point in observed
        stored = point in persisted
        if not indexed and not document:
            if attempt_documents is None:
                attempt_documents = {}
                # Publication can precede derived controller diagnostics. Join
                # the local physical attempts by exact session identity instead.
                for parent in Path(output).resolve().parents:
                    if (parent / "plan.json").exists():
                        for path in (parent / "rollout/tasks").glob("*/repetitions/*/attempts/*/capture-diagnostics.json"):
                            try:
                                value = read_json(path)
                                if value.get("session_id"):
                                    attempt_documents[value["session_id"]] = value
                            except Exception:
                                continue
                        break
            document = source[sid] = attempt_documents.get(sid, {})
        match = next((row for row in document.get("missing", [])
                      if row.get("request_id") == rid and row.get("absolute_position") == position
                      and row.get("anchor") == anchor), None)
        stage, reason = (("indexed", "finalizer_join_missing") if stored else
                         (match.get("stage"), match.get("reason_code")) if match else
                         ("unknown", "stage_evidence_unavailable"))
        known = {"expected", "indexed"} if indexed else {"expected"}
        if indexed or stored:
            known.update(stages[:-1])
        elif match and stage in stages and match.get("correlation_confidence") == "exact":
            known.update(stages[:stages.index(stage)])
        else:
            for name in stages[1:-1]:
                unknown[name] += 1
        for counts in (totals, anchors.setdefault(anchor, dict.fromkeys(stages, 0)),
                       sessions.setdefault(sid, dict.fromkeys(stages, 0))):
            for name in known:
                counts[name] += 1
        if not indexed:
            reasons[reason] += 1
            if len(missing) < 256:
                missing.append({**(match or {}), "request_id": rid, "absolute_position": position,
                    "anchor": anchor, "session_id": sid, "stage": stage, "reason_code": reason,
                    "run_id": request.get("run_id"),
                    "execution_attempt_id": document.get("execution_attempt_id"),
                    "sample_key": document.get("sample_key"), "attempt_id": document.get("attempt_id")})
    # Existing sessions without identity still get a bounded local summary.
    if missing:
        bounded = []
        for sid in dict.fromkeys(row["session_id"] for row in missing):
            rows = [row for row in missing if row["session_id"] == sid]
            identity = source[sid]
            project = os.environ.get("DYCON_CAPTURE_PROJECT")
            if not project:
                for parent in Path(output).resolve().parents:
                    if (parent / "resolved-config.json").exists():
                        project = read_json(parent / "resolved-config.json").get("project_root")
                        break
            if project and rows[0].get("run_id"):
                try:
                    rows = bounded_records(project, rows[0]["run_id"], identity.get("execution_attempt_id"), rows)
                except Exception:
                    rows = []
            bounded.extend(rows)
        missing = bounded
    value = {"schema_version": 1, "stage": "indexed", "indexed_scope": "accepted_online_index",
             "stages": totals, "by_anchor": anchors, "by_session": sessions,
             "unavailable_by_stage": dict(unknown), "reason_counts": dict(reasons),
             "missing_count": len(expected - observed), "unexpected_count": len(observed - expected),
             "missing": missing, "truncated_count": max(0, len(expected - observed) - len(missing))}
    write_json(Path(output) / "capture-diagnostics.json", value)
    return value
