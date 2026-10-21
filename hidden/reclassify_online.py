"""Reclassify finalized capture sessions using their immutable native evidence."""

import os
import select
import subprocess
import sys
from pathlib import Path

from core.config import load_snapshot
from core.io import identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from hidden.artifacts import read_branch_index, read_request
from hidden.branch_coverage import classify_close, summarize_branches
from runtime.serving.spool import publish_files


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Classification revision conflict: {path}")
    else:
        write_json(path, value)


def classify_session(root, session_id):
    """Create a new revision; original branch index/coverage/receipt stay intact."""
    base = Path(root) / "sessions" / identifier(session_id)
    view = read_branch_index(root, session_id)
    revision_path = base / "parallel/branch-classification-v2.json"
    if revision_path.exists():
        return read_json(revision_path)
    missing = {(r["decision_id"], r["branch_id"]) for r in view["coverage"]["missing"]}
    files = {str(p.relative_to(root)): sha256(p) for p in [
        base / "parallel/branch-receipt.json", base / "coverage.json", base / "finalization.json"]}
    rows = []
    decisions = {}
    for row in view["samples"]:
        key = (row["decision_id"], row["branch_id"])
        state = row.get("close_capture", {"status": "complete", "reason": None, "availability": []})
        if key in missing:
            did = row["decision_id"]
            if did not in decisions:
                path = base / "parallel/decisions" / f"{identifier(did)}.json"
                decisions[did] = read_json(path)
                files[str(path.relative_to(root))] = sha256(path)
            candidate = next(c for c in decisions[did]["candidates"] if c["branch_id"] == row["branch_id"])
            receipt = read_request(root, candidate) if candidate.get("ledger_ref") else None
            if receipt:
                if receipt["session_id"] != session_id or receipt["request_id"] != row["native_request_id"]:
                    raise ValueError("Classification ledger belongs to another branch/session")
                files[candidate["ledger_ref"]] = candidate["ledger_sha256"]
                if receipt.get("inventory_ref"):
                    files[receipt["inventory_ref"]] = sha256(owned_path(root, receipt["inventory_ref"]))
            state = classify_close(receipt, row["close_events"])
        rows.append({"decision_id": row["decision_id"], "branch_id": row["branch_id"], "close_capture": state})
    revision = {"schema_version": 2, "session_id": session_id, "rule": "no_generated_close_requires_no_hidden",
                "source_files": files, "coverage": summarize_branches(rows), "branches": rows}
    immutable_json(revision_path, revision)
    immutable_json(base / "parallel/branch-classification-v2-receipt.json", {
        "schema_version": 2, "files": {**files, str(revision_path.relative_to(root)): sha256(revision_path)}})
    return revision


def sync_task_status(project, run_id, session_id, finalization, status_path, *, validate_only=False):
    """Update mutable summaries; native trajectory bytes and exit status stay bound."""
    run = project / "runs" / run_id
    selection = read_json(run / "selection.json")
    matching = [(key, task) for key, task in selection["tasks"].items()
                if session_id.startswith(f"cap-{run_id}-{task['instance_id']}-attempt-")]
    if len(matching) != 1:
        raise ValueError("Session does not identify exactly one selected task")
    key, task = matching[0]
    work = run / "tasks" / task["instance_id"]
    attempt_id = session_id.removeprefix(f"cap-{run_id}-{task['instance_id']}-")
    with file_lock(work / "execution.lock", nonblocking=True):
        records = []
        for path in [work / "attempts" / identifier(attempt_id) / "attempt.json", work / "result.json"]:
            value = read_json(path)
            if value["rollout_id"] != finalization["rollout_id"] or value["attempt_id"] != attempt_id:
                raise ValueError("Finalized capture does not match task result")
            if any(e != "RuntimeError('Capture is partial')" for e in value.get("capture_errors", [])):
                raise ValueError("Additional capture failures prevent complete reclassification")
            records.append((path, value))
        if validate_only:
            return
        for path, value in records:
            value.update(capture_status="complete", capture_errors=[], capture_classification=str(status_path))
            write_json(path, value)
        errors = work / "attempts" / attempt_id / "capture-errors.json"
        if errors.exists():
            write_json(errors, [])
    progress_path = run / "progress.json"
    with file_lock(run / "progress-classification.lock"):
        progress = read_json(progress_path)
        if key in progress and progress[key].get("rollout_id") == finalization["rollout_id"]:
            progress[key] = read_json(work / "result.json")
            write_json(progress_path, progress)


def wait_run_exit(project, run_id):
    process = read_json(project / "runs" / run_id / "process.json")
    pid = process["supervisor_pid"]
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        argv_path = Path(f"/proc/{pid}/cmdline")
        argv = argv_path.read_bytes().split(b"\0") if argv_path.exists() else []
        if argv and (b"runtime.scripts.supervise_rollout" not in argv or run_id.encode() not in argv):
            raise ValueError("Run supervisor PID was reused")
        select.select([fd], [], [])
    finally:
        os.close(fd)


def reclassify_online(cfg, run_id, session_ids, *, wait_for_run=False):
    project = Path(cfg["project_root"])
    run_id = identifier(run_id)
    session_ids = list(dict.fromkeys(identifier(sid) for sid in session_ids))
    frozen = load_snapshot(project, run_id)
    state = project / "state/capture-reclassifications" / run_id
    state.mkdir(parents=True, exist_ok=True)
    if wait_for_run:
        wait_run_exit(project, run_id)
    export = Path(frozen["paths"]["nas_root"]) / frozen["capture"]["export"]["root"] / run_id
    source = project / frozen["capture"]["export"]["spool_root"] / run_id
    result = {}
    for sid in session_ids:
        with file_lock(state / (sid + ".lock"), nonblocking=True):
            base = export / "sessions" / sid
            finalization = read_json(base / "finalization.json")
            from application.execution.evidence import verified_trajectory
            trajectory = verified_trajectory(project, {**finalization, "producer_run_id": run_id})
            if sha256(trajectory) != finalization["trajectory_sha256"]:
                raise ValueError("Finalized trajectory changed")
            if read_json(base / "coverage.json")["missing"]:
                raise ValueError("Selected trajectory still has missing hidden")
            receipt = read_json(base / "receipt.json")
            if receipt["export"]["failed"] or receipt["metrics"].get("fallbacks"):
                raise ValueError("Capture/export failures prevent complete reclassification")
            sync_task_status(project, run_id, sid, finalization, base / "capture-status-v2.json", validate_only=True)
            # Work from retained spool bytes; publish only new revision names.
            revision = classify_session(source, sid)
            if revision["coverage"]["missing"]:
                raise ValueError("Generated close tokens still have missing hidden")
            relative = f"sessions/{sid}/parallel/branch-classification-v2.json"
            proof = f"sessions/{sid}/parallel/branch-classification-v2-receipt.json"
            files = {name: sha256(source / name) for name in [relative, proof]}
            publish_files(source, export, files, frozen["capture"]["export"]["upload_max_elapsed_seconds"])
            read_branch_index(export, sid)  # Verify the published revision and its native evidence.
            status_relative = f"sessions/{sid}/capture-status-v2.json"
            status_path = export / status_relative
            status = {"schema_version": 2, "session_id": sid, "capture_status": "complete",
                      "published": True, "classification": relative, "classification_sha256": files[relative],
                      "previous_capture_status": "partial", "trajectory_sha256": finalization["trajectory_sha256"]}
            immutable_json(source / status_relative, status)
            publish_files(source, export, {status_relative: sha256(source / status_relative)}, 120)
            with file_lock(project / "state/product-manifests.lock"):
                for owner in ["hidden", "enrichment"]:
                    path = project / owner / "manifest.json"
                    manifest = read_json(path)
                    entry = manifest["online_captures"][sid]
                    if entry["run_id"] != run_id:
                        raise ValueError("Capture manifest belongs to another run")
                    entry.update(status="complete", published=True,
                                 eligible_for_training=entry.get("dataset_role") != "diagnostic",
                                 classification_path=str(status_path), classification_sha256=sha256(status_path))
                    write_json(path, manifest)
            sync_task_status(project, run_id, sid, finalization, status_path)
            result[sid] = {"capture_status": "complete", "coverage": revision["coverage"],
                           "classification_path": str(status_path)}
    write_json(state / "latest.json", {"updated_at": now(), "sessions": result})
    run_hidden = project / "runs" / run_id / "hidden"
    run_hidden.mkdir(parents=True, exist_ok=True)
    write_json(run_hidden / "publication.json", {
        "schema_version": 1, "run_id": run_id, "sessions": result,
        "nas_root": frozen["paths"]["nas_root"], "updated_at": now(),
    })
    # A live pre-fix runner keeps old outcomes in memory. Reconcile again after
    # its supervisor exits, so its final progress write cannot undo this repair.
    process = read_json(project / "runs" / run_id / "process.json")
    sync_file = state / "status-sync-process.json"
    sync_exists = sync_file.exists() and Path(f"/proc/{read_json(sync_file)['pid']}").exists()
    if not wait_for_run and not sync_exists and Path(f"/proc/{process['pid']}").exists():
        command = [sys.executable, "-m", "core.cli", "--config", str(project / "config.yaml"),
                   "capture-reclassify", run_id, "--wait-for-run"]
        for sid in session_ids:
            command.extend(["--session-id", sid])
        with (state / "status-sync.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(state / "status-sync-process.json", {"pid": child.pid, "command": command, "started_at": now()})
    return {"run_id": run_id, "sessions": result}
