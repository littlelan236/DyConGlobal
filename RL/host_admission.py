"""Node cooperative host reservations; no lock is held during allocation/copy.

Only participating allocators are governed. MemAvailable already includes charged
resident memory: only pending requests and the new request are subtracted again.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

from RL.resources import check_host_capacity

_current = None


def current():
    return _current


def configure(root, settings, host_settings, *, attempt, rank):
    global _current
    if _current is not None:
        raise RuntimeError("Host admission already configured")
    _current = NodeAdmission(Path(root) / "state/rl-host-admission", settings,
                             host_settings, attempt=attempt, rank=rank)
    return _current


def reset():
    global _current
    if _current is not None:
        _current.close()
    _current = None


def process_start(pid):
    try:
        # comm may contain spaces/parentheses; starttime is field 22.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class NodeAdmission:
    def __init__(self, path, settings, host_settings, *, attempt, rank):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.settings, self.host_settings = settings, host_settings
        self.owner = uuid.uuid4().hex
        self.pid, self.start = os.getpid(), process_start(os.getpid())
        self.attempt, self.rank = attempt, rank
        self.tokens = set()
        self.closed = False

    @contextmanager
    def ledger(self):
        import fcntl
        with (self.path / "lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            target = self.path / "ledger.json"
            entries = json.loads(target.read_text()) if target.exists() else {}
            starts = {pid: process_start(pid) for pid in {v["pid"] for v in entries.values()}}
            entries = {k: v for k, v in entries.items() if starts[v["pid"]] == v["start"]}
            try:
                yield entries
            finally:
                temporary = target.with_suffix(f".{self.pid}.tmp")
                temporary.write_text(json.dumps(entries, sort_keys=True))
                os.replace(temporary, target)
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, size, kind):
        if self.closed or size < 0 or kind not in {"adam", "pinned"}:
            raise ValueError("Invalid host reservation")
        token = uuid.uuid4().hex
        with self.ledger() as entries:
            adam = [v for v in entries.values() if v["kind"] == "adam"]
            node_limit = int(self.settings.node_state_budget_gib * 2**30)
            if any(v["node_limit"] != node_limit for v in entries.values()):
                raise RuntimeError("Conflicting cooperative node Adam budgets")
            if kind == "adam":
                if sum(v["size"] for v in adam) + size > node_limit:
                    raise RuntimeError("Node Adam state budget rejected before allocation")
                if sum(v["size"] for v in adam if v["owner"] == self.owner) + size > int(self.settings.host_state_budget_gib * 2**30):
                    raise RuntimeError("Rank Adam state budget rejected before allocation")
            pending = sum(v["size"] for v in entries.values() if v["status"] == "pending")
            check_host_capacity(self.host_settings, requested_bytes=pending + size)
            entries[token] = {"pid": self.pid, "start": self.start, "owner": self.owner,
                "attempt": self.attempt, "rank": self.rank, "size": size, "kind": kind,
                "status": "pending", "node_limit": node_limit}
        self.tokens.add(token)
        return token

    def commit(self, token):
        with self.ledger() as entries:
            if token not in entries or entries[token]["owner"] != self.owner:
                raise RuntimeError("Lost host reservation")
            entries[token]["status"] = "charged"

    def release(self, token):
        with self.ledger() as entries:
            entry = entries.get(token)
            if entry is not None and entry["owner"] != self.owner:
                raise RuntimeError("Foreign host reservation")
            entries.pop(token, None)
        self.tokens.discard(token)

    def metrics(self):
        with self.ledger() as entries:
            return {"node_adam_bytes": sum(v["size"] for v in entries.values() if v["kind"] == "adam"),
                "node_pending_bytes": sum(v["size"] for v in entries.values() if v["status"] == "pending"),
                "owner_charged_bytes": sum(v["size"] for v in entries.values()
                                           if v["owner"] == self.owner and v["status"] == "charged"),
                "cooperative_only": True}

    def close(self):
        if self.tokens:
            raise RuntimeError("Host admission closed with live allocations")
        self.closed = True
