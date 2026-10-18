"""Launch one frozen task selection through the official no-poll supervisor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    task_ids = [selection["tasks"][key]["instance_id"] for key in selection["task_keys"]]
    command = [sys.executable, "-m", "runtime.scripts.supervise_rollout", args.run_id]
    for task_id in task_ids:
        command.extend(["--task-id", task_id])
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
