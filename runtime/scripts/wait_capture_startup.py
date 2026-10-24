"""Wait on filesystem/process events for the first real, aligned capture rollout."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import struct
import time
from pathlib import Path

from core.io import now, read_json, write_json


def evidence(root, run_id):
    run = root / "runs" / run_id
    gate_path = run / "startup/receipt.json"
    if not gate_path.exists():
        return None
    gate = read_json(gate_path)
    if not gate.get("passed"):
        return None
    cfg = read_json(run / "resolved-config.json")
    if (
        cfg["capture"]["enabled"]
        and cfg["capture"]["export"]["enabled"]
        and gate.get("capture_status") != "complete"
    ):
        raise ValueError("Production startup gate has not published capture artifacts")
    for path in sorted((run / "rollout/tasks").glob("*/repetitions/*/attempts/*/first-tool.json")):
        tool = read_json(path)
        # Successful startup means a genuine shell observation, even when the
        # repository's first diagnostic command has a nonzero exit status.
        return {
            "at": now(),
            "run_id": run_id,
            "stage": "healthy",
            "service_ready": True,
            "real_tool_observation": True,
            "first_tool_receipt": str(path),
            "tool": tool,
            "startup_gate": gate,
            "workers": cfg["execution"]["rollout_workers"],
            "max_num_seqs": cfg["serving"]["max_num_seqs"],
            "gpus": cfg["serving"]["gpus"],
            "dtype": cfg["serving"]["dtype"],
        }
    return None


def wait(root, run_id, timeout):
    run = root / "runs" / run_id
    from core.locking import directory_changes

    deadline = time.monotonic() + timeout
    while not (run / "process.json").exists():
        watched = run if run.exists() else root / "runs"
        with directory_changes(watched) as changed:
            if not (run / "process.json").exists():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Official runner did not create its process receipt")
                changed(left)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    watched = set()

    def add_tree(path):
        for folder in [path, *(p for p in path.rglob("*") if p.is_dir())]:
            name = str(folder)
            if name not in watched:
                if libc.inotify_add_watch(fd, os.fsencode(name), 0x100 | 0x80 | 0x8) >= 0:
                    watched.add(name)

    add_tree(run)
    pid = read_json(run / "process.json")["pid"]
    pidfd = os.pidfd_open(pid)
    try:
        while True:
            value = evidence(root, run_id)
            if value:
                write_json(run / "startup-verified.json", value)
                print(json.dumps(value), flush=True)
                return
            ready, _, _ = select.select([fd, pidfd], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                raise TimeoutError("Startup evidence deadline exceeded")
            if pidfd in ready:
                raise RuntimeError("Rollout supervisor exited before healthy startup")
            events = os.read(fd, 1024 * 1024)
            offset, directory_created = 0, False
            while offset < len(events):
                _, mask, _, length = struct.unpack_from("iIII", events, offset)
                offset += 16 + length
                directory_created |= bool(mask & 0x40000000)
            if directory_created:
                add_tree(run)
    finally:
        os.close(fd)
        os.close(pidfd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=float, default=2400)
    args = parser.parse_args()
    wait(args.root, args.run_id, args.timeout)
