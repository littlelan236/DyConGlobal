"""Launch one frozen task selection through the official no-poll supervisor."""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
from pathlib import Path

from core.config import load_snapshot, resolve
from core.io import hash_json, identifier, now, read_json, write_json
from core.locking import file_lock


def task_ids(selection):
    keys = selection["task_keys"]
    ids = [identifier(selection["tasks"][key]["instance_id"]) for key in keys]
    if not ids or len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
        raise ValueError("Selection must contain distinct tasks")
    return ids


def wait_predecessor(root, run_id):
    """Await the supervisor exit event, which follows CLI cleanup and its receipt."""
    run = root / "runs" / identifier(run_id)
    frozen = load_snapshot(root, run_id)
    receipt = root / "state/supervisors" / run_id / "exit.json"
    if not receipt.exists():
        pid = read_json(run / "process.json")["supervisor_pid"]
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                try:
                    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                except FileNotFoundError:
                    argv = []
                if argv and (b"runtime.scripts.supervise_rollout" not in argv
                             or run_id.encode() not in argv):
                    raise RuntimeError("Predecessor supervisor PID belongs to another process")
                select.select([fd], [], [])
            finally:
                os.close(fd)
    # A killed/missing supervisor cannot silently release an incomplete dependency.
    exited = read_json(receipt)
    status = read_json(run / "status.json")
    progress = read_json(run / "progress.json")
    if exited["exit_code"] != 0 or status["stage"] not in {"complete", "failed"}:
        raise RuntimeError("Predecessor did not finish its rollout and cleanup normally")
    if set(progress) != set(frozen["selection"]["task_keys"]):
        raise RuntimeError("Predecessor has unfinished tasks")
    return {"run_id": run_id, "exit": exited, "status": status, "tasks": len(progress)}


def supervise_command(run_id, selection):
    command = [sys.executable, "-m", "runtime.scripts.supervise_rollout", run_id]
    for task_id in task_ids(selection):
        command.extend(["--task-id", task_id])
    return command


def run_queued(root, run_id, selection, after_run):
    run_id, after_run = identifier(run_id), identifier(after_run)
    if run_id == after_run:
        raise ValueError("A rollout cannot depend on itself")
    ids = task_ids(selection)
    if (root / "runs" / run_id).exists() or (root / "state/supervisors" / run_id).exists():
        raise FileExistsError(run_id)
    load_snapshot(root, after_run)
    cfg = resolve(root / "config.yaml")
    folder = root / "state/rollout-queue" / run_id
    with file_lock(folder.with_suffix(".lock"), nonblocking=True):
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "selection.json", selection)
        record = {
            "run_id": run_id, "after_run": after_run, "stage": "waiting",
            "pid": os.getpid(), "queued_at": now(), "tasks": len(ids),
            "config_hash": cfg["config_hash"], "selection_sha256": hash_json(selection),
            "command": supervise_command(run_id, selection),
        }
        write_json(folder / "status.json", record)
        try:
            record["predecessor"] = wait_predecessor(root, after_run)
            if resolve(root / "config.yaml")["config_hash"] != record["config_hash"]:
                raise RuntimeError("config.yaml changed while queued; refusing different rollout settings")
            frozen_selection = read_json(folder / "selection.json")
            if hash_json(frozen_selection) != record["selection_sha256"]:
                raise RuntimeError("Queued task selection was modified")
            record.update(stage="launching", started_at=now())
            write_json(folder / "status.json", record)
            child = subprocess.Popen(supervise_command(run_id, frozen_selection), cwd=root)
            record.update(stage="running", supervisor_pid=child.pid)
            write_json(folder / "status.json", record)
            code = child.wait()
            record.update(stage="finished" if code == 0 else "failed", exit_code=code, finished_at=now())
            write_json(folder / "status.json", record)
            return code
        except BaseException as exc:
            record.update(stage="failed", error=repr(exc), finished_at=now())
            write_json(folder / "status.json", record)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--after-run", help="Wait for this supervised rollout to finish before launching")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if args.after_run:
        raise SystemExit(run_queued(Path.cwd(), args.run_id, selection, args.after_run))
    raise SystemExit(subprocess.run(supervise_command(identifier(args.run_id), selection), check=False).returncode)


if __name__ == "__main__":
    main()
