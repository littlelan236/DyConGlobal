"""Bounded local spool and recoverable asynchronous NAS shard publication.

Tensor encoding is owned by AsyncExportQueue. This module only archives and
copies existing bytes. Independent upload processes isolate hard NFS mounts.
"""

import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import Future
from pathlib import Path

from core.disk_budget import DiskBudget
from core.io import hash_json, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock


class UploadProcesses:
    """Fixed daemon supervisors of independent children, without executor atexit joins.

    A hard-mounted NFS worker may remain in kernel I/O after SIGTERM. Its slot
    stays occupied; no replacement is spawned and parent shutdown stays bounded.
    """

    def __init__(self, workers):
        self.workers = workers
        self.queue = queue.Queue()
        self.children = set()
        self.lock = threading.Lock()
        self.closed = False
        for index in range(workers):
            threading.Thread(target=self._run, name=f"dycon-nas-{index}", daemon=True).start()

    def submit(self, source, destination, job, deadline):
        future = Future()
        if self.closed:
            raise RuntimeError("NAS upload pool is closed")
        self.queue.put((future, source, destination, job, deadline))
        return future

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            future, source, destination, job, deadline = item
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "runtime.serving.spool",
                    "--upload-job",
                    str(job),
                    "--source",
                    str(source),
                    "--destination",
                    str(destination),
                    "--deadline",
                    str(deadline),
                ]
                with Path(str(job) + ".log").open("wb") as log:
                    child = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    write_json(str(job) + ".pid.json", {"pid": child.pid, "job": str(job)})
                    with self.lock:
                        self.children.add(child)
                    code = child.wait()  # no status poller
                    with self.lock:
                        self.children.discard(child)
                if code:
                    raise RuntimeError(f"NAS uploader exited {code}; see {job}.log")
                future.set_result({"files": read_json(job)["files"], "verified_at": now()})
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self.queue.task_done()

    def shutdown(self):
        self.closed = True
        with self.lock:
            for child in self.children:
                child.terminate()
        while True:
            try:
                future, *_ = self.queue.get_nowait()
            except queue.Empty:
                break
            future.cancel()
            self.queue.task_done()
        for _ in range(self.workers):
            self.queue.put(None)


def publish_files(source, destination, files, deadline_seconds):
    """Idempotent copy/verify/rename. Never overwrites conflicting content."""
    start = time.monotonic()
    for relative, digest in files.items():
        src, dst = owned_path(source, relative), owned_path(destination, relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(str(dst) + ".lock"):
            if dst.exists():
                if sha256(dst) != digest:
                    raise ValueError(f"NAS artifact conflict: {relative}")
                continue
            if time.monotonic() - start > deadline_seconds:
                raise TimeoutError("NAS upload deadline exceeded; local bytes retained")
            part = dst.with_name(dst.name + ".part")
            if not part.exists() or part.stat().st_size != src.stat().st_size or sha256(part) != digest:
                shutil.copyfile(src, part)
                with part.open("rb") as f:
                    os.fsync(f.fileno())
            if part.stat().st_size != src.stat().st_size or sha256(part) != digest:
                raise ValueError("NAS upload checksum mismatch")
            os.replace(part, dst)
            # Directory entry is durable before the caller records an ACK.
            fd = os.open(dst.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return {"files": files, "verified_at": now()}


def local_filesystem(path):
    """Reject NFS/CIFS spool, including symlinked directories."""
    path = Path(path).resolve()
    if Path("/proc/mounts").exists():
        matches = []
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            mount = Path(parts[1].replace("\\040", " "))
            if path.is_relative_to(mount):
                matches.append((len(str(mount)), parts[2]))
        if matches and max(matches)[1] in {"nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"}:
            raise ValueError("capture.export.spool_root must be local storage, not NAS")


class CaptureSpool:
    def __init__(self, project, options, run_id, nas_root, reserve=0):
        self.project, self.options = Path(project), options
        self.base = self.project / options["spool_root"]
        local_filesystem(self.base)
        self.root = self.base / run_id
        if self.root.is_symlink() and (self.project / "runs" / run_id / "hidden/capture-archives").is_dir():
            raise ValueError("Capture spool is archived or redirected; restore locally before capture")
        self.root.mkdir(parents=True, exist_ok=True)
        self.nas = Path(nas_root) / options["root"] / run_id
        self.budget = DiskBudget(project, self.base, options["spool_max_bytes"], reserve)
        self.pool = UploadProcesses(options["upload_workers"])
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.pending, self.failures, self.members = {}, [], {}
        self.closed = False

    def key(self, relative):
        return "capture:" + str(self.root / relative)

    def write_json(self, relative, value, *, immutable=False):
        path = owned_path(self.root, relative)
        if immutable and path.exists():
            if read_json(path) != value:
                raise ValueError("Immutable capture record changed")
            return path
        # Account for the temporary atomic write while an old file still exists.
        import json

        estimate = len(json.dumps(value, ensure_ascii=False, indent=2).encode()) + 2
        self.budget.acquire(self.key(relative), estimate * 2)
        write_json(path, value)
        self.budget.acquire(self.key(relative), path.stat().st_size)
        return path

    def add(self, row):
        with self.lock:
            sid = row["session_id"]
            self.members.setdefault(sid, []).append(row)
            size = sum((self.root / r["tensor"]).stat().st_size for r in self.members[sid])
            if size >= self.options["shard_max_bytes"]:
                self.seal(sid)

    def seal(self, session):
        with self.lock:
            rows = self.members.get(session, [])
            if not rows:
                return
            digest = hash_json([r["sample_id"] for r in rows])
            relative = f"shards/{digest}.tar"
            path = self.root / relative
            path.parent.mkdir(exist_ok=True)
            estimate = sum((self.root / r["tensor"]).stat().st_size + 2048 for r in rows) + 10240
            self.budget.acquire(self.key(relative), estimate)
            with tarfile.open(path.with_suffix(".part"), "w:") as archive:
                for row in rows:
                    archive.add(self.root / row["tensor"], arcname=row["sample_id"] + ".npz", recursive=False)
            os.replace(path.with_suffix(".part"), path)
            self.budget.acquire(self.key(relative), path.stat().st_size)
            shard_hash = sha256(path)
            files = {relative: shard_hash}
            for row in rows:
                row["tensor_locator"] = {
                    "shard": relative,
                    "member": row["sample_id"] + ".npz",
                    "member_sha256": row["sha256"],
                    "shard_sha256": shard_hash,
                }
                row["artifact_state"] = "sealed"
                event = "index/" + row["sample_id"] + ".json"
                self.write_json(event, row)
            event = "index/shard-" + digest + ".json"
            self.write_json(event, rows)
            files[event] = sha256(self.root / event)
            job = {
                "key": digest,
                "session_id": session,
                "files": files,
                "release": [r["tensor"] for r in rows] + [relative],
                "state": "sealed",
            }
            self.write_json(f"uploads/{digest}.json", job)
            self.members[session] = []
            self._submit(job)

    def _submit(self, job):
        path = self.root / f"uploads/{job['key']}.json"
        pid_path = Path(str(path) + ".pid.json")
        if pid_path.exists():
            previous = read_json(pid_path)
            process = Path(f"/proc/{previous['pid']}/cmdline")
            if process.exists():
                command = process.read_bytes()
                if str(path).encode() in command and b"runtime.serving.spool" in command:
                    raise RuntimeError("Original NAS uploader is still active/stalled; refusing replacement")
        job["state"] = "uploading"
        path = self.write_json(f"uploads/{job['key']}.json", job)
        self.budget.acquire(self.key(f"uploads/{job['key']}.json.log"), 65536)
        future = self.pool.submit(self.root, self.nas, path, self.options["upload_max_elapsed_seconds"])
        self.pending[future] = job
        future.add_done_callback(self._ack)

    def _ack(self, future):
        with self.condition:
            job = self.pending.pop(future)
            try:
                result = future.result()
                job.update(state="nas_verified", acknowledgment=result)
                self.write_json(f"uploads/{job['key']}.json", job)
                # ACK is durable before local data is removed. Metadata stays
                # small and available for final alignment without tensor I/O.
                for relative in job.get("release", []):
                    (self.root / relative).unlink(missing_ok=True)
                    self.budget.release(self.key(relative))
                job["state"] = "published"
                self.write_json(f"uploads/{job['key']}.json", job)
            except Exception as exc:
                job.update(state="pending_upload", error=repr(exc))
                self.write_json(f"uploads/{job['key']}.json", job)
                self.failures.append(job)
            self.condition.notify_all()
            with self.budget.condition:
                self.budget.condition.notify_all()

    def publish_session(self, session, timeout):
        self.seal(session)
        deadline = time.monotonic() + timeout
        with self.condition:
            if not self.condition.wait_for(
                lambda: not any(j["session_id"] == session for j in self.pending.values()), timeout
            ):
                return False
            if any(j["session_id"] == session for j in self.failures):
                return False
            files = {}
            for pattern in (f"sessions/{session}/**/*.json", "requests/*.json", "index/*.json"):
                for path in self.root.glob(pattern):
                    value = read_json(path)
                    if pattern == "index/*.json":
                        if isinstance(value, list) and value and value[0]["session_id"] == session:
                            files[str(path.relative_to(self.root))] = sha256(path)
                        continue
                    if isinstance(value, list) or value.get("session_id", session) == session:
                        files[str(path.relative_to(self.root))] = sha256(path)
            session_path = self.root / "sessions" / session / "session.json"
            if session_path.exists():
                session_info = read_json(session_path)
                config_ref = session_info.get("capture_config_ref", "capture-config.json")
                config_path = owned_path(self.root, config_ref)
                if config_path.exists():
                    expected_config = session_info.get("capture_config_sha256")
                    if expected_config and hash_json(read_json(config_path)) != expected_config:
                        raise ValueError("Session capture configuration hash mismatch")
                    files[config_ref] = sha256(config_path)
                elif config_ref != "capture-config.json":
                    raise FileNotFoundError(config_path)
            for name in ("inventory.json", "capture-config.json"):
                if (self.root / name).exists():
                    files[name] = sha256(self.root / name)
            key = hash_json({"session": session, "files": files})
            job = {"key": key, "session_id": session, "files": files, "release": [], "state": "sealed"}
            self.write_json(f"uploads/{key}.json", job)
            self._submit(job)
            return self.condition.wait_for(
                lambda: not any(j["session_id"] == session for j in self.pending.values()),
                max(0, deadline - time.monotonic()),
            ) and not any(j["session_id"] == session for j in self.failures)

    def recover(self):
        """Re-submit incomplete jobs; a rename without ACK is safe to retry."""
        jobs = []
        with self.lock:
            active = {j["key"] for j in self.pending.values()}
            self.failures.clear()
            for path in sorted((self.root / "uploads").glob("*.json")):
                if path.name.endswith(".pid.json"):
                    continue
                job = read_json(path)
                if job["state"] != "published" and job["key"] not in active:
                    self._submit(job)
                    jobs.append(job["key"])
            staged = {}
            for path in sorted((self.root / "index").glob("*.json")):
                row = read_json(path)
                if (
                    isinstance(row, dict)
                    and row.get("artifact_state") == "queued"
                    and (self.root / row["tensor"]).exists()
                ):
                    import hashlib

                    import numpy as np

                    with np.load(self.root / row["tensor"], allow_pickle=False) as tensors:
                        content_hash = hash_json(
                            {
                                name: hashlib.sha256(tensors[name].tobytes()).hexdigest()
                                for name in tensors.files
                            }
                        )
                    if content_hash != row["tensor_content_sha256"]:
                        raise ValueError("Crash recovery tensor differs from queued metadata")
                    row.update(
                        artifact_state="local_persisted",
                        status="persisted",
                        sha256=sha256(self.root / row["tensor"]),
                    )
                    self.write_json(str(path.relative_to(self.root)), row)
                if (
                    isinstance(row, dict)
                    and row.get("artifact_state") == "local_persisted"
                    and (self.root / row["tensor"]).exists()
                ):
                    staged.setdefault(row["session_id"], []).append(row)
            for session, rows in staged.items():
                self.members[session] = rows
                self.seal(session)
        return jobs

    def release_session_metadata(self, session):
        """Publication ACK already verified these exact bytes; keep small jobs."""
        with self.lock:
            paths = list((self.root / "sessions" / session).rglob("*.json"))
            for pattern in ("requests/*.json", "index/*.json"):
                for path in self.root.glob(pattern):
                    value = read_json(path)
                    sid = (
                        value[0]["session_id"]
                        if isinstance(value, list) and value
                        else value.get("session_id")
                        if isinstance(value, dict)
                        else None
                    )
                    if sid == session:
                        paths.append(path)
            for path in paths:
                path.unlink(missing_ok=True)
                self.budget.release(self.key(str(path.relative_to(self.root))))

    def close(self, timeout):
        try:
            with self.lock:
                for session in list(self.members):
                    self.seal(session)
            with self.condition:
                complete = self.condition.wait_for(lambda: not self.pending, timeout)
            if not complete:
                for job in list(self.pending.values()):
                    self.write_json(
                        f"uploads/{job['key']}.json",
                        {**job, "state": "pending_upload", "reason": "upload_timeout"},
                    )
                # Bound process count even when hard NFS I/O cannot be interrupted.
            return complete and not self.failures
        finally:
            self.pool.shutdown()
            self.closed = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-job", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--deadline", type=int, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.deadline
    delay = 0.25
    while True:
        try:
            publish_files(
                args.source,
                args.destination,
                read_json(args.upload_job)["files"],
                max(0, deadline - time.monotonic()),
            )
            job = read_json(args.upload_job)
            receipt = Path(args.destination) / "publications" / (job["key"] + ".json")
            value = {"schema_version": 2, "key": job["key"], "files": job["files"]}
            if receipt.exists() and read_json(receipt) != value:
                raise ValueError("Conflicting NAS publication receipt")
            if not receipt.exists():
                write_json(receipt, value)
            break
        except (OSError, TimeoutError):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(5, delay * 2)
