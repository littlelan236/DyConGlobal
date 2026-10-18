"""Durably await the official CLI and record its exit; no status polling."""

import argparse
import subprocess
import sys
from pathlib import Path

from core.io import identifier, now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--unrolled", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    run_id = identifier(args.run_id)
    command = [sys.executable, "-m", "core.cli", "--config", str(root / "config.yaml"), "rollout", run_id]
    if args.resume:
        command.append("--resume")
    if args.unrolled:
        command.append("--unrolled")
    for task_id in args.task_id:
        command += ["--task-id", task_id]
    state = root / "state/supervisors" / run_id
    state.mkdir(parents=True, exist_ok=False)
    with (state / "rollout.log").open("ab") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write_json(state / "process.json", {"pid": child.pid, "args": command, "started_at": now()})
        code = child.wait()
    write_json(state / "exit.json", {"exit_code": code, "finished_at": now()})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
