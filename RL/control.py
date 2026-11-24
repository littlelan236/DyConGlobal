"""Stop an RL supervisor through the public run cancel interface."""
import os
import signal
import time
from pathlib import Path
from core.io import identifier, now, read_json, write_json


def cancel_if_rl(cfg, attempt_id, reason):
    identifier(attempt_id)
    directory = Path(cfg["project_root"]) / "RL/checkpoints" / attempt_id
    if not (directory / "attempt.json").is_file():
        return None
    from RL.observability import matches, process, training_process
    attempt = read_json(directory / "attempt.json")
    record = training_process(directory, attempt, {})
    if record is None:
        return {"attempt_id": attempt_id, "cancel_requested": False, "already_terminal": True}
    # The resolver binds exact config path/attempt ID/output root and creation
    # time. Recheck starttime and command immediately before signalling.
    command = record["command"]
    if not {"rl-train", "lora-train"}.intersection(command):
        raise ValueError("Refusing to signal an unbound RL supervisor")
    tracked = {record["pid"]}
    processes = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                fields = (path/"stat").read_text().rsplit(")", 1)[1].split()
                processes[int(path.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                continue
    for _ in range(len(processes)):
        children = {pid for pid, parent in processes.items() if parent in tracked}
        if children <= tracked:
            break
        tracked |= children
    records = [p for pid in tracked if (p := process(pid)) is not None]
    request = {"attempt_id": attempt_id, "reason": reason, "requested_at": now(),
               "supervisor": record, "semantics": "stop_and_resume_committed_checkpoint"}
    write_json(directory/"cancel-request.json", request)
    if not matches(record):
        raise RuntimeError("RL supervisor identity changed before cancellation")
    os.kill(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 35
    while any(matches(item) for item in records):
        if time.monotonic() >= deadline:
            raise RuntimeError("RL cancellation cleanup not confirmed within 35 seconds")
        time.sleep(.2)
    ack = {**request, "exited_at": now(), "cleanup_confirmed": True,
           "checkpoints": sorted(p.name for p in (directory/"checkpoints").glob("step-*")
                                 if (p/"receipt.json").is_file())}
    write_json(directory/"cancel-receipt.json", ack)
    return {"attempt_id": attempt_id, "cancel_requested": True,
            "cleanup_confirmed": True, "receipt": str(directory/"cancel-receipt.json"),
            "checkpoints": ack["checkpoints"]}
