"""Formal detached replay lifecycle, with exact process identity for cancellation."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from core.io import read_json, write_json
from core.locking import file_lock
from ..artifacts import directory
from ..config import load


def _start_ticks(pid):
    try: return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError): return None


def manage(cfg, file, *, detach=False, cancel=False, status=False, recover=False):
    if sum((detach, cancel, status)) != 1: raise ValueError("Choose one replay lifecycle operation")
    config, source = load(file, cfg["project_root"], replay=True)
    state_root = Path(cfg["project_root"]) / "state/rl-replay"
    state_root.mkdir(parents=True, exist_ok=True)
    record_path = state_root / (config.replay_id + ".json")
    target = directory(config.storage.output_root, "rl-replay", config.replay_id)
    stage = target.with_name("." + config.replay_id + ".staging")
    with file_lock(state_root / (config.replay_id + ".lock"), nonblocking=True):
        record = read_json(record_path) if record_path.exists() else {}
        alive = bool(record.get("pid") and _start_ticks(record["pid"]) == record.get("start_ticks"))
        if detach:
            if alive: raise RuntimeError("Replay job already running")
            if target.exists(): raise FileExistsError("Replay ID is already committed")
            if stage.exists() and not recover: raise FileExistsError("Replay staging exists; use recover explicitly")
            command = [sys.executable, "-m", "core.cli", "rl-dataset", "replay-logprobs", "--replay-config", str(source)]
            if recover: command.append("--recover")
            log = state_root / (config.replay_id + ".log")
            with log.open("ab") as stream:
                child = subprocess.Popen(command, cwd=cfg["project_root"], stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record = {"pid": child.pid, "start_ticks": _start_ticks(child.pid), "command": command,
                      "replay_id": config.replay_id, "config": str(source), "log": str(log)}
            if record["start_ticks"] is None: raise RuntimeError("Replay exited during detached admission; inspect lifecycle log")
            write_json(record_path, record)
            return {"status": "started", **record}
        if cancel and alive:
            command = Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b"\0")
            if b"core.cli" not in command or str(source).encode() not in command: raise RuntimeError("Replay process identity differs; refusing cancellation")
            os.kill(record["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and _start_ticks(record["pid"]) == record["start_ticks"]:
                state = read_json(stage / "state.json") if (stage / "state.json").exists() else {}
                if state.get("state") in ("interrupted", "failed") and (stage / "cleanup.json").exists(): break
                time.sleep(.1)
            else:
                if _start_ticks(record["pid"]) == record["start_ticks"]: raise RuntimeError("Cancellation cleanup deadline exceeded; job identity retained")
        alive = bool(record.get("pid") and _start_ticks(record["pid"]) == record.get("start_ticks"))
        actual = target if target.exists() else stage
        state = read_json(actual / "state.json") if (actual / "state.json").exists() else {}
        cleanup = read_json(actual / "cleanup.json") if (actual / "cleanup.json").exists() else None
        progress = {}
        if (actual / 'journal.json').exists():
            journal = read_json(actual / 'journal.json')
            progress['committed_branches'] = journal.get('branch_count', 0)
        workers = []
        for folder in sorted((actual / 'workers').glob('gpu-*')):
            entry = {'worker': folder.name}
            if (folder / 'journal.json').exists():
                journal = read_json(folder / 'journal.json')
                entry.update(state=journal.get('state'), committed_branches=journal.get('branch_count', 0))
            if (folder / 'execution.json').exists():
                execution = read_json(folder / 'execution.json')
                entry.update(prefix_cache_effective=execution.get('prefix_cache_effective'),
                             cudagraph_effective=execution.get('cudagraph_effective'))
            workers.append(entry)
        if workers: progress['workers'] = workers
        return {"status": state.get("state", "running" if alive else "not_running"),
                "replay_id": config.replay_id, "process_alive": alive, "cleanup": cleanup,
                "receipt": (target / "receipt.json").exists(), "log": record.get("log"),
                'progress': progress}
