"""One cross-process reservation ledger for image admission and capture spool."""

import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from core.io import read_json
from core.locking import file_lock


class DiskBudget:
    def __init__(self, root, path, limit, reserve=0):
        self.root, self.path = Path(root), Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.limit, self.reserve = int(limit), int(reserve)
        self.state = self.root / "state/disk-budget.sqlite3"
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.state.with_suffix(".lock")
        self.condition = threading.Condition()
        with file_lock(self.lock), closing(sqlite3.connect(self.state, timeout=30)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS reservations (key TEXT PRIMARY KEY, bytes INTEGER, kind TEXT, device INTEGER, path TEXT, pid INTEGER)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS totals (key TEXT PRIMARY KEY, bytes INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            if not db.execute("SELECT 1 FROM metadata WHERE key='initialized'").fetchone():
                legacy = self.root / "state/disk-budget.json"
                for key, value in (read_json(legacy) if legacy.exists() else {}).items():
                    self._put(db, key, value)
                db.execute("INSERT INTO metadata VALUES ('initialized', '1')")
            # Reclaim only transient reservations of exited processes. Durable
            # artifact reservations survive process exit and require NAS ACK.
            for key, pid in db.execute(
                "SELECT key,pid FROM reservations WHERE key LIKE 'image:%' OR key LIKE 'queue:%' OR key LIKE 'handoff:%'"
            ).fetchall():
                if not Path(f"/proc/{pid}").exists():
                    self._drop(db, key)

    @staticmethod
    def _change(db, key, amount):
        db.execute(
            "INSERT INTO totals VALUES (?,?) ON CONFLICT(key) DO UPDATE SET bytes=bytes+excluded.bytes",
            (key, amount),
        )

    @staticmethod
    def _total(db, key):
        row = db.execute("SELECT bytes FROM totals WHERE key=?", (key,)).fetchone()
        return row[0] if row else 0

    def _drop(self, db, key):
        row = db.execute("SELECT bytes,kind,device FROM reservations WHERE key=?", (key,)).fetchone()
        if row:
            size, kind, device = row
            if kind == "capture":
                self._change(db, "capture", -size)
            self._change(db, "device:" + str(device), -size)
            db.execute("DELETE FROM reservations WHERE key=?", (key,))

    def _put(self, db, key, value):
        self._drop(db, key)
        db.execute(
            "INSERT INTO reservations VALUES (?,?,?,?,?,?)",
            (key, value["bytes"], value["kind"], value["device"], value["path"], value["pid"]),
        )
        if value["kind"] == "capture":
            self._change(db, "capture", value["bytes"])
        self._change(db, "device:" + str(value["device"]), value["bytes"])

    def allocations(self):
        with closing(sqlite3.connect(self.state, timeout=30)) as db:
            return {
                key: {"bytes": size, "kind": kind, "device": dev, "path": path, "pid": pid}
                for key, size, kind, dev, path, pid in db.execute("SELECT * FROM reservations")
            }

    def usage(self):
        with closing(sqlite3.connect(self.state, timeout=30)) as db:
            return self._total(db, "capture")

    def acquire(self, key, size, *, kind="capture"):
        size = int(size)
        if size < 0:
            raise ValueError("Reservation bytes cannot be negative")
        with file_lock(self.lock), closing(sqlite3.connect(self.state, timeout=30)) as db, db:
            old_row = db.execute("SELECT bytes FROM reservations WHERE key=?", (key,)).fetchone()
            old = old_row[0] if old_row else 0
            cap = self._total(db, "capture")
            dev = self.path.stat().st_dev
            total = self._total(db, "device:" + str(dev))
            if kind == "capture" and cap - old + size > self.limit:
                raise BufferError("capture project spool byte limit reached")
            if size > old and shutil.disk_usage(self.path).free < self.reserve + total - old + size:
                raise BufferError("shared local disk reserve would be exceeded")
            self._put(
                db,
                key,
                {"bytes": size, "kind": kind, "device": dev, "path": str(self.path), "pid": os.getpid()},
            )

    def release(self, key):
        with file_lock(self.lock), closing(sqlite3.connect(self.state, timeout=30)) as db, db:
            self._drop(db, key)
        with self.condition:
            self.condition.notify_all()

    @contextmanager
    def reservation(self, size, kind="capture"):
        key = kind + ":" + uuid.uuid4().hex
        self.acquire(key, size, kind=kind)
        try:
            yield key
        finally:
            self.release(key)

    def admit(self, high, low, timeout):
        import time

        deadline = time.monotonic() + timeout
        if self.usage() < self.limit * high:
            return
        from core.locking import directory_changes

        with directory_changes(self.state.parent) as changed:
            while self.usage() > self.limit * low:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("capture spool admission timed out; recover pending uploads")
                changed(left)
