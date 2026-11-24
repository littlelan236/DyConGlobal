"""Node cooperative host reservations; no lock is held during allocation/copy.

Only participating allocators are governed. MemAvailable already includes charged
resident memory: only pending requests and the new request are subtracted again.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

from RL.resources import check_host_capacity, host_memory

TRACE_WAIT_SECONDS = 4200
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


class TraceAdmission(NodeAdmission):
    """Reserve a complete trace before any of its activations are allocated."""

    def __init__(self, path, *, budget, reserve, attempt, rank):
        super().__init__(path, None, None, attempt=attempt, rank=rank)
        self.budget, self.reserve_bytes = budget, reserve

    def try_acquire(self, size):
        if not 0 < size <= self.budget:
            raise ValueError(f"Trace reservation {size} exceeds shared host budget {self.budget}")
        token = uuid.uuid4().hex
        with self.ledger() as entries:
            used = sum(v["size"] for v in entries.values())
            node_limit = min([self.budget, *(v["node_limit"] for v in entries.values())])
            host = host_memory()
            resident = sum(v.get("resident", 0) for v in entries.values())
            pending = used - resident
            available = host.get("MemAvailable", 0)
            cgroup = host.get("cgroup_path")
            remaining = host.get("cgroup_remaining_bytes")
            status = {"shared_reserved_bytes": used, "host_available_bytes": available,
                      "trace_resident_bytes": resident, "trace_pending_bytes": pending,
                      "cgroup_remaining_bytes": remaining}

            if remaining is not None:
                # Only credit resident blocks known to be charged to this cgroup.
                # Workers under the same leaf also share its ancestor limits.
                cgroup_resident = sum(v.get("resident", 0) for v in entries.values()
                                      if cgroup is not None and v.get("cgroup") == cgroup)
                cgroup_pending = used - cgroup_resident
                status["cgroup_trace_pending_bytes"] = cgroup_pending
                if cgroup_pending + size + self.reserve_bytes > remaining:
                    return None, status
            if used + size > node_limit or pending + size + self.reserve_bytes > available:
                return None, status
            entries[token] = {"pid": self.pid, "start": self.start, "owner": self.owner,
                              "size": size, "resident": 0, "cgroup": cgroup, "node_limit": self.budget,
                              "attempt": self.attempt, "rank": self.rank}
        self.tokens.add(token)
        status.update(shared_reserved_bytes=used + size, trace_pending_bytes=pending + size)
        if remaining is not None:
            status["cgroup_trace_pending_bytes"] += size
        return token, status

    def account_overrun(self, token, size):
        """Charge an active trace without delaying its remaining allocations."""
        self._account(token, projected=size)

    def account_residency(self, token, delta):
        self._account(token, resident_delta=delta)

    def _account(self, token, *, projected=0, resident_delta=0):
        with self.ledger() as entries:
            entry = entries.get(token)
            if entry is None or entry["owner"] != self.owner:
                raise RuntimeError("Lost trace host reservation")
            resident = entry.get("resident", 0) + resident_delta
            if resident < 0:
                raise RuntimeError("Trace host reservation released more bytes than allocated")
            entry.update(size=max(entry["size"], projected, resident), resident=resident)

    def acquire(self, size, on_wait, *, timeout=TRACE_WAIT_SECONDS):
        started = time.monotonic()
        next_report = started
        while True:
            token, status = self.try_acquire(size)
            now = time.monotonic()
            if token is not None:
                return token, status, now - started
            if now >= started + timeout:
                raise RuntimeError(f"Timed out waiting to reserve {size} host bytes for a complete trace: {status}")
            if now >= next_report:
                on_wait(wait_seconds=now - started, **status)
                next_report = now + 30
            time.sleep(0.25)
