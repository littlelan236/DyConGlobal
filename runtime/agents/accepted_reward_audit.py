"""Read-only, run-scoped audit of accepted SkillsBench rewards below one."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from application.execution.run_plan import load_plan
from application.execution.status import _sample_status_projection
from core.io import hash_json, read_json, run_path, sha256, write_json
from runtime.agents.verifier_audit import (
    POLICY,
    classify_attempt,
    finite_reward,
)
from runtime.agents.verifier_evidence import verifier_files

CLASSES = ("VerifierError", "ValidOutcome", "Review")


def _json(path):
    try:
        value = read_json(path)
        return value, None
    except (OSError, ValueError, UnicodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _hash(path):
    try:
        return sha256(path)
    except OSError:
        return None


def _reward(row):
    raw = row.get("raw_reward")
    value = raw.get("reward") if isinstance(raw, dict) else None
    if not finite_reward(value):
        return value, False
    return value, True


def _sample_key(item):
    return item.get("sample_key") or item.get("key")


def _audit_candidate(root, run_id, row, task):
    run = run_path(root, run_id)
    task_id, rep, attempt_id = row.get("instance_id"), row.get("repetition_id"), row.get("attempt_id")
    relative = Path("rollout/tasks") / str(task_id) / "repetitions" / str(rep) / "attempts" / str(attempt_id)
    if any(Path(str(v)).name != str(v) or str(v) in {".",".."} for v in (task_id,rep,attempt_id)):
        raise ValueError("Unsafe attempt identity")
    attempt = run / relative
    if not attempt.resolve().is_relative_to(run.resolve()):
        raise ValueError("Attempt outside run")
    value, _ = _reward(row)
    result = classify_attempt(attempt, task, reward=value)
    execution = (row.get("serving_assignment") or {}).get("execution_attempt_id")
    started_path = attempt / "attempt-started.json"
    if started_path.exists():
        started = read_json(started_path)
        execution = execution or started.get("execution_attempt_id")
    return {**result, "run_id":run_id, "task_key":row.get("task_key"), "instance_id":task_id,
            "sample_key":_sample_key(row), "repetition_id":rep, "attempt_id":attempt_id,
            "execution_attempt_id":execution, "attempt_path":relative.as_posix(),
            "observed_exit_status":row.get("exit_status"), "observed_disposition":row.get("disposition"),
            "raw_reward":row.get("raw_reward"), "evaluation_status":row.get("evaluation_status"),
            "recorded_audit_classification":(row.get("verifier_zero_audit") or {}).get("classification")}


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def audit_accepted_rewards(root, run_id, output_dir):
    requested = Path(output_dir).expanduser()
    output = requested.resolve()
    root_path = Path(root).resolve()
    loaded = load_plan(root, run_id)
    run = loaded["run"].resolve()
    allowed_roots = [root_path, run]
    if requested.is_symlink() or not any(output.is_relative_to(allowed) for allowed in allowed_roots):
        raise ValueError("Audit output must be a new owned project directory")
    owned_root = next(allowed for allowed in allowed_roots if output.is_relative_to(allowed))
    if any(part in {"rollout", "execution", "subject", "capture", "benchflow"} for part in output.relative_to(owned_root).parts):
        raise ValueError("Audit output must not overlap execution/evidence paths")
    if any(p.is_symlink() for p in [requested, *requested.parents] if p.exists() and p != Path(root)):
        # Project root itself may resolve through the supported /data3 layout.
        if not output.is_relative_to(run):
            raise ValueError("Unsafe symbolic output parent")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    started = datetime.now(timezone.utc).isoformat()
    if loaded["config"].get("execution_context", {}).get("benchmark") != "skillsbench":
        raise ValueError("Run is not a frozen SkillsBench run")
    run = loaded["run"]
    progress_path = run / "progress.json"
    frozen_names = ('plan.json', 'selection.json', 'config.yaml', 'resolved-config.json',
                    'work-items.json', 'tasks.json')
    frozen_before = {name: _hash(run/name) for name in frozen_names}
    progress_before = _hash(progress_path)
    progress = read_json(progress_path) if progress_path.exists() else {}
    rows, projection = _sample_status_projection(loaded, progress)
    projection_before = hash_json(rows)
    task_map = loaded["config"]["selection"]["tasks"]
    candidates, anomalies = [], Counter()
    for row in rows.values():
        if row.get("disposition") != "accepted":
            continue
        value, numeric = _reward(row)
        if not numeric:
            anomalies["missing_or_non_finite_reward"] += 1
        elif value < 1:
            candidates.append(row)
        elif value > 1:
            anomalies["reward_above_one"] += 1
        else:
            anomalies["reward_equal_one"] += 1
    scanned = []
    for row in candidates:
        try:
            scanned.append(_audit_candidate(root, run_id, row, task_map[row["task_key"]]))
        except Exception as exc:
            scanned.append({"run_id": run_id, "task_key": row.get("task_key"), "instance_id": row.get("instance_id"),
                "sample_key": _sample_key(row), "repetition_id": row.get("repetition_id"), "attempt_id": row.get("attempt_id"),
                "attempt_path": None, "observed_exit_status": row.get("exit_status"), "observed_disposition": row.get("disposition"),
                "raw_reward": row.get("raw_reward"), "evaluation_status": row.get("evaluation_status"),
                "recorded_audit_classification": (row.get("verifier_zero_audit") or {}).get("classification"),
                "classification": "Review", "primary_rule": "R6", "reason_code": "candidate_read_failed",
                "reason": f"{type(exc).__name__}: {exc}", "verifier_phase": None, "test_framework": None,
                "test_execution_state": "unknown", "evidence": [], "missing_evidence": [],
                "integrity_errors": [{"error": str(exc)}], "source_changed": False, "truncated": False, "processed": False})
    progress_after = _hash(progress_path)
    frozen_after = {name: _hash(run/name) for name in frozen_names}
    try:
        end_loaded = load_plan(root, run_id)
        end_progress = read_json(progress_path) if progress_path.exists() else {}
        end_rows, _ = _sample_status_projection(end_loaded, end_progress)
        projection_after = hash_json(end_rows)
    except (OSError, ValueError, KeyError):
        projection_after = None
    scope_changed = (progress_before != progress_after or frozen_before != frozen_after
                     or projection_before != projection_after)
    for record in scanned:
        for item in record.get('source_manifest', []):
            if _hash(Path(item['path'])) != item['sha256']:
                record.update(source_changed=True, processed=False, classification='Review', primary_rule='R6',
                              reason_code='source_changed', reason='Consumed evidence changed before report completion.')
        if record.get('attempt_path') and 'verifier_listing' in record:
            attempt_path = run / record['attempt_path']
            listing = [p.relative_to(attempt_path).as_posix() for p in verifier_files(attempt_path)]
            if listing != record['verifier_listing']:
                record.update(source_changed=True, processed=False, classification='Review', primary_rule='R6',
                              reason_code='source_changed', reason='Verifier artifacts changed before report completion.')
    if scope_changed:
        for row in scanned:
            row.update(source_changed=True, processed=False, classification="Review", primary_rule="R6",
                       reason_code="source_changed", reason="Frozen run inputs or current sample projection changed during the audit scan.")
    counts = Counter(row["classification"] for row in scanned)
    task_counts = {}
    rule_counts = Counter()
    for row in scanned:
        entry = task_counts.setdefault(row["instance_id"], {"candidate_count": 0, **{c: 0 for c in CLASSES}})
        entry["candidate_count"] += 1
        entry[row["classification"]] += 1
        rule_counts[row["primary_rule"]] += 1
    completed = datetime.now(timezone.utc).isoformat()
    summary = {"schema_version": 1, "policy": POLICY, "run_id": run_id, "started_at": started,
        "finished_at": completed, "scope": {"benchmark": "skillsbench", "task_count": len(task_map),
        "current_sample_count": len(rows), "candidate_definition": "current disposition=accepted and finite numeric raw_reward.reward < 1",
        "current_sample_projection": "application.execution.status._sample_status_projection"},
        "source_identity": {"plan_sha256": _hash(run/"plan.json"), "selection_sha256": _hash(run/"selection.json"),
        "progress_sha256_before": progress_before, "progress_sha256_after": progress_after,
        "frozen_inputs_before": frozen_before, "frozen_inputs_after": frozen_after,
        "projection_sha256_before": projection_before, "projection_sha256_after": projection_after},
        "scanned_sample_count": len(rows), "non_candidate_accepted_reward_anomalies": dict(anomalies),
        "candidate_count": len(scanned), "processed_count": sum(x["processed"] for x in scanned),
        "unprocessed_count": sum(not x["processed"] for x in scanned), "classification_counts": dict(counts),
        "per_task": dict(sorted(task_counts.items())), "per_rule": dict(sorted(rule_counts.items())),
        "source_changed": scope_changed or any(x["source_changed"] for x in scanned),
        "complete": not scope_changed and all(x["processed"] for x in scanned)}
    if sum(counts[c] for c in CLASSES) != len(scanned) or sum(x["candidate_count"] for x in task_counts.values()) != len(scanned):
        raise AssertionError("Audit count conservation failed")
    output.mkdir(parents=True)
    (output / "samples.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in scanned), encoding="utf-8")
    fields = sorted({key for row in scanned for key in row})
    with (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in scanned)
    write_json(output / "summary.json", summary)
    lines = [f"# SkillsBench accepted reward audit: {run_id}", "", f"- Policy: `{POLICY}`", f"- Tasks: {len(task_map)}; current samples: {len(rows)}; candidates: {len(scanned)}", f"- Results: VerifierError {counts['VerifierError']}, ValidOutcome {counts['ValidOutcome']}, Review {counts['Review']}", f"- Processing gaps: {summary['unprocessed_count']}; complete: {summary['complete']}", "", "## By task", "", "| Task | Candidates | VerifierError | ValidOutcome | Review |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {name} | {item['candidate_count']} | {item['VerifierError']} | {item['ValidOutcome']} | {item['Review']} |" for name,item in sorted(task_counts.items())]
    lines += ["", "## VerifierError and Review", ""]
    lines += [f"- `{row['instance_id']}` `{row['repetition_id']}` `{row['attempt_id']}` {row['classification']} / {row['primary_rule']}: {row['reason']}" for row in scanned if row['classification'] != 'ValidOutcome']
    lines += ["", "Raw runs, attempts and evaluation receipts were read only. No verifier or task was rerun.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"run_id": run_id, "scope": summary["scope"], "candidate_count": len(scanned),
            "classification_counts": dict(counts), "processed_count": summary["processed_count"],
            "unprocessed_count": summary["unprocessed_count"], "complete": summary["complete"],
            "output": str(output)}
