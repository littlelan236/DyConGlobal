"""Read-only, run-scoped audit of accepted SkillsBench rewards below one."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from application.execution.run_plan import load_plan
from application.execution.status import _sample_status_projection
from core.io import read_json, sha256, write_json
from runtime.agents.verifier_audit import (
    _verifier_files,
    classify_verifier_logs,
    frozen_verifier_contract,
)

POLICY = "skillsbench_accepted_reward_audit_v1"
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
    if type(value) not in (int, float) or not math.isfinite(value):
        return value, False
    return value, True


def _sample_key(item):
    return item.get("sample_key") or item.get("key")


def _python_runtime_download_failed(logs):
    recovery_markers = ("successfully installed python", "installed cpython", "pytest session starts", "collected ")
    combined = "\n".join(log.lower() for log in logs)
    has_archive = "python-build-standalone" in combined or "cpython-3.10" in combined
    failed = "failed to download" in combined or "request failed after" in combined
    return has_archive and failed and not any(marker in combined for marker in recovery_markers)


def _audit_candidate(root, run_id, row, task):
    run = Path(root) / "runs" / run_id
    task_id, rep, attempt_id = row.get("instance_id"), row.get("repetition_id"), row.get("attempt_id")
    relative = Path("rollout/tasks") / str(task_id) / "repetitions" / str(rep) / "attempts" / str(attempt_id)
    attempt = run / relative
    record = {
        "run_id": run_id, "task_key": row.get("task_key"), "instance_id": task_id,
        "sample_key": _sample_key(row), "repetition_id": rep,
        "attempt_id": attempt_id, "execution_attempt_id": (row.get("serving_assignment") or {}).get("execution_attempt_id"),
        "attempt_path": relative.as_posix(), "observed_exit_status": row.get("exit_status"),
        "observed_disposition": row.get("disposition"), "raw_reward": row.get("raw_reward"),
        "evaluation_status": row.get("evaluation_status"),
        "recorded_audit_classification": (row.get("verifier_zero_audit") or {}).get("classification"),
        "classification": "Review", "primary_rule": "R6", "reason_code": "insufficient_evidence",
        "reason": "Evidence was not sufficient to classify this accepted below-one reward.",
        "verifier_phase": None, "test_framework": None, "test_execution_state": "unknown",
        "evidence": [], "missing_evidence": [], "integrity_errors": [],
        "source_changed": False, "truncated": False, "processed": False,
    }
    before = {}
    paths = ["attempt.json", "attempt-started.json", "benchflow/result.json", "evaluation-raw.json",
             "evaluation-report.json", "diagnostics/verifier-execution.json",
             "diagnostics/verifier-zero-audit.json", "verifier-environment.json", "subject/subject.json"]
    docs = {}
    for name in paths:
        path = attempt / name
        before[name] = _hash(path)
        if not path.exists():
            record["missing_evidence"].append(name)
            continue
        value, error = _json(path) if name.endswith(".json") else (None, None)
        if error:
            record["integrity_errors"].append({"path": name, "error": error})
        elif name.endswith(".json"):
            docs[name] = value
            record["evidence"].append({"path": name, "sha256": before[name], "field": "document"})
    execution = docs.get("diagnostics/verifier-execution.json")
    if isinstance(execution, dict):
        commands = execution.get("commands") or []
        record["verifier_phase"] = commands[-1].get("phase") if commands and isinstance(commands[-1], dict) else None
        tests = [x for x in commands if isinstance(x, dict) and x.get("phase") in {"task_test", "verifier_test"}]
        record["test_execution_state"] = (tests[-1].get("status", "unknown") if tests else "not_recorded")
        record["test_framework"] = "pytest" if any("pytest" in str(x.get("command_summary", "")).lower() for x in tests) else None
        if tests and tests[-1].get("status") == "completed":
            record["test_execution_state"] = "completed"
    files = _verifier_files(attempt)
    logs, unreadable = [], False
    for path in files:
        rel = path.relative_to(attempt).as_posix()
        if path.name in {"test-stdout.txt", "test-stderr.txt"}:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                logs.append(content)
                record["evidence"].append({"path": rel, "sha256": _hash(path), "excerpt": content[-1200:]})
            except OSError as exc:
                unreadable = True
                record["integrity_errors"].append({"path": rel, "error": str(exc)})
        elif path.suffix == ".json" and "ctrf" in path.name.lower():
            h = _hash(path)
            record["evidence"].append({"path": rel, "sha256": h})
    task_contract = frozen_verifier_contract(task)
    record["task_contract"] = task_contract
    execution_result = classify_verifier_logs(logs, unreadable=unreadable)
    python_archive_failure = _python_runtime_download_failed(logs)
    if python_archive_failure:
        record.update(classification="VerifierError", primary_rule="R2", reason_code="python_runtime_download_failed",
                      reason="The verifier could not download its required CPython runtime after retries; no later test run or recovery evidence was found.")
    elif execution_result[0] == "VerifierError":
        record.update(classification="VerifierError", primary_rule="R2", reason_code="dependency_install_failed",
                      reason=execution_result[1])
    elif not python_archive_failure:
        from runtime.agents.verifier_audit import PYTEST_SUMMARY, inspect_ctrf
        ctrf_executed, ctrf_present = inspect_ctrf(attempt)
        pytest_complete = any(PYTEST_SUMMARY.search(log) for log in logs)
        missing_reference = next((log for log in logs if "missing ground truth: /root/loss.npz" in log.lower()), None)
        raw = docs.get("evaluation-raw.json")
        if isinstance(raw, dict) and (raw.get("completed") is False and raw.get("error")):
            record.update(classification="VerifierError", primary_rule="R1", reason_code="verifier_incomplete",
                          reason="Structured evaluation evidence reports completed=false with an error.")
        elif missing_reference is not None:
            record.update(reason_code="missing_reference_input", reason="Verifier output reports a missing ground-truth input; input ownership is not proven by the attempt evidence, so this remains Review.")
        elif ctrf_executed or pytest_complete:
            record.update(classification="ValidOutcome", primary_rule="R5", reason_code="tests_completed",
                          reason="CTRF or pytest summary proves tests ran and completed; the sub-one reward is from the recorded task evaluation.")
            record["test_execution_state"] = "completed"
            record["test_framework"] = "pytest"
        elif task_contract.get("ctrf_expected") is None or not task_contract.get("verified"):
            record.update(reason_code="frozen_verifier_contract_unknown",
                          reason="Frozen verifier path/hash could not be verified and test completion is unproven.")
        elif task_contract.get("ctrf_expected") and not ctrf_present:
            record.update(reason_code="ctrf_missing", reason="Frozen verifier expects CTRF, but no readable report proves completed tests.")
        elif unreadable or record["integrity_errors"]:
            record.update(reason_code="evidence_read_failed", reason="Required verifier evidence could not be read reliably.")
    value, numeric = _reward(row)
    if numeric and value < 0:
        record.update(classification="Review", primary_rule="R6", reason_code="invalid_reward_range",
                      reason="Reward is below zero; it is a candidate but outside the valid reward range.")
    record["processed"] = not bool(record["integrity_errors"])
    for name, old in before.items():
        path = attempt / name
        if old != _hash(path):
            record["source_changed"] = True
    if record["source_changed"]:
        record.update(classification="Review", primary_rule="R6", reason_code="source_changed",
                      reason="Attempt evidence changed during the audit scan.")
        record["processed"] = False
    return record


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def audit_accepted_rewards(root, run_id, output_dir):
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    started = datetime.now(timezone.utc).isoformat()
    loaded = load_plan(root, run_id)
    if loaded["config"].get("execution_context", {}).get("benchmark") != "skillsbench":
        raise ValueError("Run is not a frozen SkillsBench run")
    run = loaded["run"]
    progress_path = run / "progress.json"
    progress_before = _hash(progress_path)
    progress = read_json(progress_path) if progress_path.exists() else {}
    rows, projection = _sample_status_projection(loaded, progress)
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
    if progress_before != progress_after:
        for row in scanned:
            row.update(source_changed=True, processed=False, classification="Review", primary_rule="R6",
                       reason_code="source_changed", reason="Run progress changed during the audit scan.")
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
        "progress_sha256_before": progress_before, "progress_sha256_after": progress_after},
        "scanned_sample_count": len(rows), "non_candidate_accepted_reward_anomalies": dict(anomalies),
        "candidate_count": len(scanned), "processed_count": sum(x["processed"] for x in scanned),
        "unprocessed_count": sum(not x["processed"] for x in scanned), "classification_counts": dict(counts),
        "per_task": dict(sorted(task_counts.items())), "per_rule": dict(sorted(rule_counts.items())),
        "source_changed": progress_before != progress_after or any(x["source_changed"] for x in scanned),
        "complete": progress_before == progress_after and all(x["processed"] for x in scanned)}
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
