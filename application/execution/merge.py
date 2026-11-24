"""Merge finished rollout sample artifacts with explicit repetition precedence."""

from __future__ import annotations

import copy
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from core.io import artifact_reference, hash_json, identifier, now, read_json, run_path, sha256, write_json
from core.locking import file_lock
from .evidence import RunEvidence
from .run_plan import load_plan
from rollout.retry import disposition as rollout_disposition, sample_accepted


PENDING_STATUSES = {"Pending", "Running"}
PENDING_DISPOSITIONS = {"pending", "retry_pending", "infrastructure_retry", "interrupted"}
TERMINAL_RUN_STAGES = {"complete", "failed", "cancelled", "validated"}


def sample_state(cfg, value):
    """Classify a sample for merge precedence without treating AgentCompleted as failure."""
    if not value or value.get("exit_status") in PENDING_STATUSES or value.get("disposition") in PENDING_DISPOSITIONS:
        return "pending"
    if value.get("disposition") in {"failed", "blocked"}:
        return "failed"
    if sample_accepted(cfg, value):
        return "accepted"
    return "failed"


def choose_sample(destination, source, destination_cfg, source_cfg):
    """Apply source non-pending over pending and source accepted over failed."""
    dst_state = sample_state(destination_cfg, destination)
    src_state = sample_state(source_cfg, source)
    if (dst_state == "pending" and src_state != "pending") or (dst_state == "failed" and src_state == "accepted"):
        return "source"
    return "destination"


def _copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise ValueError(f"Missing source artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Conflicting merge artifact: {destination}")
        return
    shutil.copyfile(source, destination)


def _copy_tree(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise ValueError(f"Missing source artifact directory: {source}")
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            _copy_file(item, target)
        else:
            raise ValueError(f"Unsupported merge artifact: {item}")


def _destination_sample(item, source_result, destination_run, attempt_id):
    value = copy.deepcopy(source_result)
    for key in ("run_id", "task_key", "instance_id", "repetition_id", "repetition_index",
                "sample_key", "sample_seed", "seed_policy"):
        value[key] = item[key]
    source_attempt = source_result.get("attempt_id")
    value["attempt_id"] = attempt_id
    if value.get("last_attempt_id") == source_attempt:
        value["last_attempt_id"] = attempt_id
    value["source_run_id"] = source_result.get("run_id")
    value["source_sample_key"] = source_result.get("sample_key")
    value["source_attempt_id"] = source_attempt
    value["source_repetition_id"] = source_result.get("repetition_id")
    value["source_sample_seed"] = source_result.get("sample_seed")
    value["merged_from"] = {"run_id": source_result.get("run_id"), "attempt_id": source_attempt}
    return value


def _copy_attempt(source_work, destination_work, source_attempt, destination_attempt, item, destination_run):
    source_dir = source_work / "attempts" / source_attempt
    # Runs migrated from schema 1 retain their legacy layout without the
    # repetitions/ directory. Copy those immutable attempt bytes into the
    # destination schema-2 repetitions layout.
    if not source_dir.is_dir():
        legacy_dir = source_work.parent / "attempts" / source_attempt
        if legacy_dir.is_dir():
            source_dir = legacy_dir
    if not source_dir.is_dir():
        return
    destination_dir = destination_work / "attempts" / destination_attempt
    _copy_tree(source_dir, destination_dir)
    # Attempt metadata is state, not immutable model output, so bind it to B.
    for name in ("attempt-context.json", "agent-outcome.json", "attempt.json", "execution-state.json"):
        path = destination_dir / name
        if not path.exists():
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        source_binding = {k: value.get(k) for k in ("run_id", "sample_key", "repetition_id", "sample_seed", "attempt_id")}
        for key in ("run_id", "task_key", "sample_key", "repetition_id", "repetition_index", "sample_seed", "seed_policy"):
            if key in item:
                value[key] = item[key]
        if "attempt_id" in value:
            value["attempt_id"] = destination_attempt
        value["source_run_id"] = value.get("source_run_id") or source_binding["run_id"]
        value["source_attempt_id"] = source_attempt
        value["source_binding"] = source_binding
        write_json(path, value)


def _copy_rollout_artifacts(root, source_run, destination_run, source_row, destination_row):
    """Copy canonical trajectory/patch/subject bytes under B's run-owned paths."""
    source_root = run_path(root, source_run)
    destination_root = run_path(root, destination_run)
    for kind in ("trajectory", "patch"):
        field = kind + "_path"
        if field not in source_row:
            continue
        source = root / source_row[field]
        destination = destination_root / "rollout" / ("trajectories" if kind == "trajectory" else "patches") / destination_row["rollout_id"] / source.name
        _copy_file(source, destination)
        destination_row[field] = artifact_reference(root, destination)
        destination_row[kind + "_sha256"] = sha256(destination)
    if "subject_path" in source_row:
        source = root / source_row["subject_path"]
        destination = destination_root / "rollout" / "submissions" / destination_row["rollout_id"] / source.name
        _copy_tree(source.parent, destination.parent)
        destination_row["subject_path"] = artifact_reference(root, destination)
        destination_row["subject_sha256"] = sha256(destination)


def _copy_sample_attempts(source_work, destination_work, source_attempt, destination_attempt,
                          item, destination_run, mode):
    _copy_attempt(source_work, destination_work, source_attempt, destination_attempt, item, destination_run)
    if mode == "append-executed":
        for history in sorted((source_work / "attempts").glob("*")):
            if history.is_dir() and history.name != source_attempt:
                historical_id = destination_attempt + "-history-" + hash_json(history.name)[:12]
                _copy_attempt(source_work, destination_work, history.name, historical_id, item, destination_run)


def _append_event(run, event):
    events = run / "events"
    events.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in events.glob("*.json"):
        try:
            numbers.append(int(path.stem))
        except ValueError:
            continue
    path = events / f"{max(numbers, default=0) + 1:06d}.json"
    write_json(path, event)
    return path


def _rebuild_progress(destination_run, destination_cfg, work_items, samples):
    accepted = sum(sample_state(destination_cfg, value) == "accepted" for value in samples.values())
    failed = sum(value.get("disposition") in {"failed", "blocked"} for value in samples.values())
    progress_path = destination_run / "progress.json"
    previous = read_json(progress_path) if progress_path.exists() else {}
    progress = {
        "schema_version": 2,
        "updated_at": now(),
        "samples": samples,
        "planned_samples": len(work_items),
        "accepted_samples": accepted,
        "failed_samples": failed,
        "pending_samples": len(work_items) - accepted - failed,
        "physical_attempts": sum(value.get("physical_attempts", 0) for value in samples.values()),
        "resample_retry_count": sum(value.get("resample_retry_count", 0) for value in samples.values()),
        "infrastructure_retry_count": sum(value.get("infrastructure_retry_count", 0) for value in samples.values()),
        "limits_exceeded_count": previous.get("limits_exceeded_count", 0),
        "reasoning_loop_count": previous.get("reasoning_loop_count", 0),
        "resample_trigger_count": previous.get("resample_trigger_count", 0),
        "ready_samples": len(work_items) - accepted - failed,
        "running_attempts": 0,
        "active_generation_samples": 0,
        "finalizing_attempts": 0,
        "admission_waiting_samples": 0,
        "counts_path": previous.get("counts_path", str(destination_run / "rollout/limits-exceeded-counts.json")),
        "observability_errors": previous.get("observability_errors", []),
        "environment_dependency_failure_count": previous.get("environment_dependency_failure_count", 0),
        "phase_projection_path": str(destination_run / "rollout/phases.json"),
    }
    write_json(progress_path, progress)
    write_json(destination_run / "rollout" / "phases.json", {
        "updated_at": progress["updated_at"], "ready_samples": progress["ready_samples"],
        "running_attempts": 0, "active_generation_samples": 0,
        "finalizing_attempts": 0, "admission_waiting_samples": 0,
    })
    return progress


def _legacy_samples(run_id, loaded, progress, evidence, cfg):
    """Reconstruct merge samples for migrated schema-1 progress."""
    samples = progress.get("samples")
    if isinstance(samples, dict) and samples:
        return samples
    items = loaded["work_items"]["items"]
    by_identity = {(item["task_key"], item.get("repetition_id", "rep-000")): item for item in items}
    by_task = {item["task_key"]: item for item in items}
    task_rows = {row["task_key"]: row for row in loaded["tasks"]["tasks"]}
    rebuilt = {}
    for rollout_id, row in evidence.get("rollouts", {}).items():
        task_key = row.get("task_key")
        repetition_id = row.get("repetition_id", "rep-000")
        item = by_identity.get((task_key, repetition_id)) or by_task.get(task_key)
        if item is None:
            continue
        status = row.get("termination_reason") or row.get("exit_status") or "Submitted"
        result = {
            "run_id": run_id,
            "task_key": task_key,
            "instance_id": item.get("instance_id") or task_rows.get(task_key, {}).get("instance_id"),
            "repetition_id": repetition_id,
            "repetition_index": item.get("repetition_index", 0),
            "sample_key": item["sample_key"],
            "sample_seed": item.get("sample_seed"),
            "seed_policy": item.get("seed_policy", "legacy_unrecorded"),
            "attempt_id": row.get("attempt_id") or "attempt-000",
            "rollout_id": rollout_id,
            "exit_status": status,
            "agent_exit_status": row.get("agent_exit_status", status),
            "disposition": rollout_disposition(cfg, status),
        }
        previous = rebuilt.get(item["sample_key"])
        if previous is None or str(result["attempt_id"]) >= str(previous["attempt_id"]):
            rebuilt[item["sample_key"]] = result
    return rebuilt


def _owned_pid_active(pid, run_id):
    """Return true only when a recorded PID is live and still names this run."""
    if not pid:
        return False
    try:
        pid = int(pid)
        os.kill(pid, 0)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\\0", b" ")
        return run_id.encode() in command
    except (OSError, ValueError):
        return False


def _legacy_rollout_is_stale(loaded, progress):
    """Allow a migrated legacy rollout only when no owned process/lock remains."""
    run = loaded["run"]
    if read_json(run / "status.json").get("stage") != "rollout":
        return False
    if isinstance(progress.get("samples"), dict) and progress.get("samples"):
        return False
    latest = run / "execution/latest.json"
    if not latest.exists():
        return False
    try:
        attempt_id = read_json(latest)["attempt"]
        attempt = run / "execution/attempts" / attempt_id
        for name in ("supervisor.json", "process.json"):
            path = attempt / name
            if path.exists() and _owned_pid_active(read_json(path).get("pid"), run.name):
                return False
        with file_lock(run / "execution/supervisor.lock", nonblocking=True):
            with file_lock(run / "execution/active-supervisor.lock", nonblocking=True):
                return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _executed(run, value):
    """An admission/queue cursor is not evidence of physical execution."""
    if not value:
        return False
    if value.get("merged_from") or value.get("rollout_id"):
        return True
    if not value.get("attempt_id") or not value.get("instance_id") or not value.get("repetition_id"):
        return False
    attempt = (run / "rollout/tasks" / value["instance_id"] / "repetitions" /
               value["repetition_id"] / "attempts" / value["attempt_id"])
    return (attempt / "attempt-started.json").is_file()


def _append_pairs(source_samples, destination_samples, items, source_run, destination_run):
    """Keep destination executed repetitions, then fill vacant frozen task slots."""
    used = {(v.get("source_run_id"), v.get("source_sample_key"), v.get("source_attempt_id"))
            for v in destination_samples.values() if v.get("merged_from")}
    used_rollouts = {v.get("rollout_id") for v in destination_samples.values()
                     if v.get("rollout_id")}
    pairs, summary = [], Counter()
    by_task = {}
    for item in items:
        by_task.setdefault(item["task_key"], []).append(item)
    for task_key, task_items in by_task.items():
        slots = [i for i in sorted(task_items, key=lambda i: i["repetition_index"])
                 if not _executed(destination_run, destination_samples.get(i["sample_key"]))]
        candidates = sorted((v for v in source_samples.values() if v.get("task_key") == task_key
                             and _executed(source_run, v)),
                            key=lambda v: (v.get("repetition_index", 0), v.get("sample_key", "")))
        available = []
        for value in candidates:
            rid = value.get("rollout_id")
            if (value.get("run_id"), value.get("sample_key"), value.get("attempt_id")) in used or (rid and rid in used_rollouts):
                summary["already_imported"] += 1
                continue
            available.append(value)
            if rid:
                used_rollouts.add(rid)
        summary["over_target"] += max(0, len(available) - len(slots))
        pairs.extend((i["sample_key"], v) for i, v in zip(slots, available))
    return pairs, summary


def _executed_sample_projection(loaded, progress):
    """Use the final actually started attempt when a later cursor only queued."""
    from .status import _sample_status_projection
    rows, _ = _sample_status_projection(loaded, progress)
    for item in loaded["work_items"]["items"]:
        key = item["sample_key"]
        if _executed(loaded["run"], rows.get(key)):
            continue
        work = loaded["run"] / "rollout/tasks" / item["instance_id"] / "repetitions" / item["repetition_id"]
        started = [p for p in (work / "attempts").glob("*")
                   if (p / "attempt-started.json").is_file() and p.name.startswith(item["repetition_id"] + "-attempt-")]
        if not started:
            continue
        last = max(started, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        final = last / "attempt.json"
        if final.exists():
            rows[key] = {**item, **read_json(final)}
        else:
            rows[key] = {**item, "attempt_id": last.name, "exit_status": "Interrupted", "disposition": "infrastructure_retry"}
    return rows


def _require_clean_exit(loaded):
    run = loaded["run"]
    latest = run / "execution/latest.json"
    if not latest.exists():
        return
    attempt = run / "execution/attempts" / read_json(latest)["attempt"]
    if not (attempt / "exit.json").exists():
        raise ValueError(f"Run {run.name} has no supervisor exit receipt")
    termination = attempt / "termination.json"
    if not termination.exists() or not read_json(termination).get("cleanup_complete"):
        raise ValueError(f"Run {run.name} resource cleanup is incomplete")


def merge_runs(root, source_run_id, destination_run_id, *, mode="by-repetition", dry_run=False):
    """Merge source results into a finished destination under the selected policy."""
    root = Path(root).resolve()
    source_run_id, destination_run_id = identifier(source_run_id), identifier(destination_run_id)
    if source_run_id == destination_run_id:
        raise ValueError("Source and destination runs must differ")
    if mode not in {"by-repetition", "append-executed"}:
        raise ValueError("Unknown merge mode")
    lock_ids = sorted((source_run_id, destination_run_id))
    with file_lock(root / "state/run-locks" / (lock_ids[0] + ".lock"), nonblocking=True):
        with file_lock(root / "state/run-locks" / (lock_ids[1] + ".lock"), nonblocking=True):
            source = load_plan(root, source_run_id)
            destination = load_plan(root, destination_run_id)
            if source["plan"]["operation"] != "rollout" or destination["plan"]["operation"] != "rollout":
                raise ValueError("run merge requires rollout runs")
            source_progress = read_json(source["run"] / "progress.json")
            destination_progress = read_json(destination["run"] / "progress.json")
            for loaded, label, progress in ((source, "source", source_progress),
                                             (destination, "destination", destination_progress)):
                stage = read_json(loaded["run"] / "status.json").get("stage")
                if not dry_run and stage not in TERMINAL_RUN_STAGES and not (label == "source" and _legacy_rollout_is_stale(loaded, progress)):
                    raise ValueError(f"{label} run is not finished: {stage}")
                if mode == "append-executed" and not dry_run:
                    _require_clean_exit(loaded)
            if mode == "by-repetition" and source["plan"]["repeat_count"] != destination["plan"]["repeat_count"]:
                raise ValueError("Runs must have the same repeat_count")
            source_tasks = {row["task_key"]: row["task"] for row in source["tasks"]["tasks"]}
            destination_tasks = {row["task_key"]: row["task"] for row in destination["tasks"]["tasks"]}
            if source_tasks != destination_tasks:
                raise ValueError("Runs must have identical frozen task definitions")
            source_cfg, destination_cfg = source["config"], destination["config"]
            source_evidence = RunEvidence(root, source_run_id).read()
            destination_evidence = RunEvidence(root, destination_run_id)
            source_samples = _legacy_samples(source_run_id, source, source_progress, source_evidence, source_cfg)
            destination_samples = destination_progress.get("samples", {})
            if mode == "append-executed":
                source_samples = _executed_sample_projection(source, source_progress)
                destination_samples = _executed_sample_projection(destination, destination_progress)
            source_by_identity = {(v.get("task_key"), v.get("repetition_id")): v for v in source_samples.values()}
            destination_by_identity = {(v.get("task_key"), v.get("repetition_id")): (k, v) for k, v in destination_samples.items()}
            source_rows = source_evidence["rollouts"]
            changed = []
            stats = Counter()
            items_by_key = {i["sample_key"]: i for i in destination["work_items"]["items"]}
            if mode == "append-executed":
                pairs, append_stats = _append_pairs(source_samples, destination_samples,
                                                   list(items_by_key.values()), source["run"], destination["run"])
                stats.update(append_stats)
                from rollout.retry import verify_final
                source_items = {i["sample_key"]: i for i in source["work_items"]["items"]}
                for _, result in pairs:
                    row = source_rows.get(result.get("rollout_id"), {})
                    if row.get("provenance", {}).get("quarantined"):
                        raise ValueError("Cannot append quarantined source rollout")
                    if row.get("evaluation_subject", {}).get("kind") in {"live_state", "web_trace"}:
                        raise ValueError("Append currently supports patch/workspace subjects only")
                    if sample_state(source_cfg, result) == "accepted":
                        verify_final(root, source_run_id, source_items[result["sample_key"]], result, source_evidence)
                    elif row:
                        from .subject import verify_subject
                        verify_subject(root, row)
            else:
                pairs = [(destination_by_identity[identity][0], result)
                         for identity, result in source_by_identity.items()
                         if identity in destination_by_identity
                         and choose_sample(destination_by_identity[identity][1], result, destination_cfg, source_cfg) == "source"]
            if dry_run:
                return {"source_run_id": source_run_id, "destination_run_id": destination_run_id,
                        "mode": mode, "dry_run": True, "changed": len(pairs), "stats": dict(stats),
                        "mapping": [{"source_sample_key": r.get("sample_key"),
                                     "source_repetition_id": r.get("repetition_id"),
                                     "destination_sample_key": k,
                                     "destination_repetition_id": items_by_key[k]["repetition_id"],
                                     "instance_id": items_by_key[k]["instance_id"]} for k, r in pairs]}
            with destination_evidence.transaction() as evidence:
                for destination_key, source_result in pairs:
                    destination_result = destination_samples.get(destination_key, {})
                    source_rid = source_result.get("rollout_id")
                    source_row = source_rows.get(source_rid)
                    if not source_row and sample_state(source_cfg, source_result) == "accepted":
                        raise ValueError(f"Accepted source sample lacks evidence: {source_rid}")
                    source_attempt = source_result.get("attempt_id") or "unknown"
                    merge_attempt = f"merge-{source_run_id[:24]}-{source_attempt}"
                    if mode == "append-executed":
                        merge_attempt = "merge-" + hash_json({"source": source_run_id, "sample": source_result["sample_key"],
                                                             "attempt": source_attempt,
                                                             "destination": destination_key})[:24]
                    destination_item = items_by_key[destination_key]
                    if mode == "append-executed":
                        old_result = destination["run"] / "rollout/tasks" / destination_item["instance_id"] / "repetitions" / destination_item["repetition_id"] / "result.json"
                        if old_result.exists():
                            backup = destination["run"] / "execution/merge-backups" / merge_attempt / "result.json"
                            _copy_file(old_result, backup)
                    if not source_row:
                        source_work = source["run"] / "rollout/tasks" / source_result["instance_id"] / "repetitions" / source_result["repetition_id"]
                        destination_work = destination["run"] / "rollout/tasks" / destination_item["instance_id"] / "repetitions" / destination_item["repetition_id"]
                        _copy_sample_attempts(source_work, destination_work, source_attempt, merge_attempt,
                                              destination_item, destination_run_id, mode)
                        destination_result = _destination_sample(destination_item, source_result, destination_run_id, merge_attempt)
                        write_json(destination_work / "result.json", destination_result)
                        destination_samples[destination_key] = destination_result
                        changed.append({"task_key": destination_item["task_key"], "repetition_id": destination_item["repetition_id"], "sample_key": destination_key, "rollout_id": None, "source_attempt_id": source_attempt, "destination_attempt_id": merge_attempt,
                                        "source_sample_key": source_result.get("sample_key"), "source_repetition_id": source_result.get("repetition_id")})
                        stats["source_overrides"] += 1
                        continue
                    destination_row = copy.deepcopy(source_row)
                    destination_row["run_id"] = destination_run_id
                    destination_row["rollout_id"] = source_rid
                    destination_row["sample_key"] = destination_key
                    destination_row["attempt_id"] = merge_attempt
                    if mode == "append-executed":
                        for key in ("repetition_id", "repetition_index", "sample_seed", "seed_policy"):
                            destination_row[key] = destination_item[key]
                        if destination_row.get("evaluation_subject", {}).get("kind") == "workspace":
                            evidence["schema_version"] = max(2, evidence["schema_version"])
                    destination_row["provenance"] = {
                        **copy.deepcopy(source_row.get("provenance", {})),
                        "source_run_id": source_run_id,
                        "source_rollout_id": source_rid,
                        "source_attempt_id": source_attempt,
                        "source_sample_key": source_result.get("sample_key"),
                        "source_repetition_id": source_result.get("repetition_id"),
                        "source_sample_seed": source_result.get("sample_seed"),
                        "merged_into_run": destination_run_id,
                    }
                    if source_rid in evidence["rollouts"]:
                        existing = evidence["rollouts"][source_rid]
                        if existing.get("task_key") != destination_row.get("task_key"):
                            raise ValueError(f"Conflicting rollout evidence: {source_rid}")
                    else:
                        _copy_rollout_artifacts(root, source_run_id, destination_run_id, source_row, destination_row)
                        evidence["rollouts"][source_rid] = destination_row
                    evidence["tasks"][destination_row["task_key"]] = copy.deepcopy(destination_tasks[destination_row["task_key"]])
                    # Preserve evaluation evidence under a destination-owned report path.
                    source_eval_id = source_row.get("selected_evaluation_id")
                    if source_eval_id and source_eval_id in source_evidence["evaluations"]:
                        source_eval = source_evidence["evaluations"][source_eval_id]
                        merged_eval_id = identifier("merge-" + hash_json({"run": source_run_id, "evaluation": source_eval_id,
                                                                        "rollout": source_rid})[:24])
                        report_source = root / source_eval["report_path"]
                        report_target = destination["run"] / "reports" / (merged_eval_id + ".json")
                        _copy_file(report_source, report_target)
                        outcome = copy.deepcopy(source_eval["outcomes"].get(source_rid, {}))
                        outcome["subject_sha256"] = destination_row.get("subject_sha256", outcome.get("subject_sha256"))
                        evidence["evaluations"][merged_eval_id] = {
                            "report_path": artifact_reference(root, report_target),
                            "report_sha256": sha256(report_target),
                            "outcomes": {source_rid: outcome},
                            "provenance": {"merged_from_run": source_run_id, "merged_from_evaluation": source_eval_id},
                        }
                        destination_row["selected_evaluation_id"] = merged_eval_id
                        destination_row["evaluation_status"] = source_row.get("evaluation_status")
                        destination_row["resolved"] = source_row.get("resolved")
                    source_work = source["run"] / "rollout/tasks" / source_result["instance_id"] / "repetitions" / source_result["repetition_id"]
                    destination_work = destination["run"] / "rollout/tasks" / destination_item["instance_id"] / "repetitions" / destination_item["repetition_id"]
                    _copy_sample_attempts(source_work, destination_work, source_attempt, merge_attempt,
                                          destination_item, destination_run_id, mode)
                    destination_result = _destination_sample(destination_item, source_result, destination_run_id, merge_attempt)
                    destination_result["rollout_id"] = source_rid
                    destination_result["source_rollout_id"] = source_rid
                    destination_result["source_repetition_id"] = source_result.get("repetition_id")
                    destination_result["source_sample_seed"] = source_result.get("sample_seed")
                    if source_eval_id and source_eval_id in source_evidence["evaluations"]:
                        destination_result["evaluation_id"] = merged_eval_id
                    write_json(destination_work / "result.json", destination_result)
                    destination_samples[destination_key] = destination_result
                    changed.append({"task_key": destination_item["task_key"], "repetition_id": destination_item["repetition_id"],
                                    "sample_key": destination_key, "rollout_id": source_rid,
                                    "source_attempt_id": source_attempt, "destination_attempt_id": merge_attempt,
                                    "source_sample_key": source_result.get("sample_key"),
                                    "source_repetition_id": source_result.get("repetition_id")})
                    stats["source_overrides"] += 1
            from rollout.retry import rebuild_counts
            counts = rebuild_counts(destination["run"], destination["work_items"])
            progress = _rebuild_progress(destination["run"], destination_cfg, destination["work_items"]["items"], destination_samples)
            for field in ("limits_exceeded_count", "reasoning_loop_count", "resample_trigger_count"):
                progress[field] = counts[field]
            write_json(destination["run"] / "progress.json", progress)
            receipt = {
                "schema_version": 1, "source_run_id": source_run_id, "destination_run_id": destination_run_id,
                "created_at": now(), "mode": mode,
                "rule": ("destination executed repetitions first; append source executed repetitions including failures to remaining frozen task slots; cap at destination repeat_count"
                         if mode == "append-executed" else "source non-pending overrides destination pending; source accepted overrides destination failed; destination accepted wins"),
                "changed": changed, "stats": dict(stats), "destination_progress_sha256": hash_json(progress),
            }
            receipt_id = "merge-" + hash_json({"source": source_run_id, "destination": destination_run_id, "changed": changed})[:24]
            receipt_path = destination["run"] / "execution" / "merges" / (receipt_id + ".json")
            write_json(receipt_path, receipt)
            _append_event(destination["run"], {"event": "run.merge", "run_id": destination_run_id,
                                               "source_run_id": source_run_id, "receipt": str(receipt_path),
                                               "changed": len(changed), "at": receipt["created_at"]})
            from .status import refresh_run_status_summary
            refresh_run_status_summary(root, destination_run_id)
            from .artifact import refresh
            refresh(root, destination_run_id)
            return {"source_run_id": source_run_id, "destination_run_id": destination_run_id,
                    "mode": mode, "receipt": str(receipt_path), "changed": len(changed), "stats": dict(stats),
                    "progress": {k: progress[k] for k in ("planned_samples", "accepted_samples", "failed_samples", "pending_samples")}}
