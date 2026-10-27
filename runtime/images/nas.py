"""Streaming NAS OCI imports with four worker slots and crash-safe leases."""

from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from core.io import hash_json, read_json, sha256, write_json
from core.locking import file_lock


class NasImageStore:
    def __init__(self, cfg, reference):
        self.cfg = cfg
        self.reference = reference
        nas = Path(cfg["paths"]["nas_root"]).resolve()
        if not any(p != Path("/") and os.path.ismount(p) for p in [nas, *nas.parents]):
            raise ValueError("Configured NAS namespace is not on a mounted filesystem")
        self.root = (nas / reference["store"]).resolve()
        path = (self.root / reference["manifest"]).resolve()
        if not self.root.is_relative_to(nas) or not path.is_relative_to(self.root):
            raise ValueError("NAS image path escapes configured namespace")
        if sha256(path) != reference["manifest_sha256"]:
            raise ValueError("NAS manifest checksum mismatch")
        self.record = read_json(path)
        self.image_id = self.record["manifest_digest"]
        self.blob(self.image_id)
        if reference.get("image_id", self.image_id) != self.image_id:
            raise ValueError("NAS image digest mismatch")

    def identity(self):
        return {
            "image_id": self.image_id,
            "requested": self.record.get("requested"),
            "repo_digests": self.record.get("repo_digests", []),
            "nas_manifest_sha256": self.reference["manifest_sha256"],
        }

    def blob(self, digest):
        if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
            raise ValueError("Only SHA256 OCI descriptors are supported")
        int(digest[7:], 16)
        path = (self.root / "blobs/sha256" / digest[7:]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("OCI blob escapes store")
        return path

    def verify(self, deep=False):
        from runtime.images.oci import descriptor_graph

        manifest = read_json(self.blob(self.image_id))
        root = {
            "digest": self.image_id,
            "size": self.record["manifest_size"],
            "mediaType": manifest["mediaType"],
        }
        return descriptor_graph(self.root / "blobs/sha256", root, deep=deep)

    def write_archive(self, stream):
        descriptors = self.verify(deep=True)
        manifest = read_json(self.blob(self.image_id))
        descriptor = {
            "mediaType": manifest["mediaType"],
            "digest": self.image_id,
            "size": self.record["manifest_size"],
        }
        entries = {
            "oci-layout": {"imageLayoutVersion": "1.0.0"},
            "index.json": {"schemaVersion": 2, "manifests": [descriptor]},
        }
        with tarfile.open(fileobj=stream, mode="w|") as tar:
            for name, value in entries.items():
                data = json.dumps(value).encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            seen = set()
            for d in descriptors:
                if d["digest"] in seen:
                    continue
                seen.add(d["digest"])
                info = tarfile.TarInfo("blobs/sha256/" + d["digest"][7:])
                info.size = d["size"]
                with self.blob(d["digest"]).open("rb") as f:
                    tar.addfile(info, f)


class ImageLeaseManager:
    def __init__(self, cfg, client):
        self.cfg, self.client = cfg, client
        info = client.info()
        from runtime.docker_storage import assert_data_root

        assert_data_root(cfg, info)
        self.docker_root = Path(info["DockerRootDir"])
        self.state = Path(cfg["project_root"]) / "state/images"
        self.state.mkdir(parents=True, exist_ok=True)
        self.options = cfg["images"]
        self.daemon_key = str(info.get("ID", "unknown"))[:24]
        self.daemon_key = hash_json({"id": info.get("ID"), "root": str(self.docker_root)})[:24]
        self.state = self.state / self.daemon_key
        self.state.mkdir(exist_ok=True)

    def lifecycle_event(self, event, image_id, **extra):
        path = self.cfg.get("image_events_path")
        if path:
            from core.io import now

            with file_lock(Path(path).with_suffix(".lock")):
                with Path(path).open("a") as out:
                    out.write(json.dumps({"at": now(), "event": event, "image_id": image_id, **extra}) + "\n")

    def _read(self):
        p = self.state / "journal.json"
        return read_json(p) if p.exists() else {"images": {}, "reservations": {}}

    def _save(self, value):
        self.state.mkdir(parents=True, exist_ok=True)
        write_json(self.state / "journal.json", value)

    @contextmanager
    def _slot(self):
        deadline = time.monotonic() + self.options["admission_seconds"]
        while True:
            for i in range(self.options["load_workers"]):
                f = (self.state / f"slot-{i}.lock").open("a+")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                try:
                    yield i
                finally:
                    f.close()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("NAS load slots are busy")
            time.sleep(0.1)

    @contextmanager
    def _reservation(self, estimate):
        from core.disk_budget import DiskBudget

        disk = DiskBudget(
            self.cfg["project_root"], self.docker_root, 2**63 - 1, self.options["reserve_bytes"]
        )
        shared_token = "image:" + uuid.uuid4().hex
        token = uuid.uuid4().hex
        deadline = time.monotonic() + self.options["admission_seconds"]
        while True:
            with file_lock(self.state / "journal.lock"):
                state = self._read()
                for key, r in list(state["reservations"].items()):
                    if _owner(r["pid"]) != r.get("owner"):
                        state["reservations"].pop(key)
                reserved = sum(x["bytes"] for x in state["reservations"].values())
                free = shutil.disk_usage(self.docker_root).free
                if (
                    reserved + estimate <= self.options["max_reserved_bytes"]
                    and free >= self.options["reserve_bytes"] + reserved + estimate
                ):
                    try:
                        disk.acquire(shared_token, estimate, kind="image")
                    except BufferError:
                        self._save(state)
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Insufficient shared disk space for NAS image") from None
                        continue
                    state["reservations"][token] = {
                        "bytes": estimate,
                        "pid": os.getpid(),
                        "owner": _owner(os.getpid()),
                    }
                    self._save(state)
                    break
                self._save(state)
            if time.monotonic() >= deadline:
                raise TimeoutError("Insufficient local space for NAS image")
            time.sleep(0.2)
        try:
            yield
        finally:
            disk.release(shared_token)
            with file_lock(self.state / "journal.lock"):
                state = self._read()
                state["reservations"].pop(token, None)
                self._save(state)

    def _exists(self, image_id):
        import docker

        try:
            self.client.images.get(image_id)
            return True
        except docker.errors.ImageNotFound:
            return False

    def _load(self, store):
        log = self.state / (store.image_id[7:] + ".load.log")
        started = time.perf_counter()
        with log.open("wb") as out:
            proc = subprocess.Popen(
                [
                    self.cfg["environment"]["docker_executable"],
                    "--host",
                    self.cfg["environment"]["docker_host"],
                    "load",
                ],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
            timer = threading.Timer(self.options["load_seconds"], proc.kill)
            timer.daemon = True
            timer.start()
            try:
                store.write_archive(proc.stdin)
                proc.stdin.close()
                code = proc.wait()
                write_json(
                    log.with_suffix(".load.timing.json"),
                    {
                        "image_id": store.image_id,
                        "seconds": time.perf_counter() - started,
                        "returncode": code,
                        "compressed_bytes": store.record.get("compressed_bytes"),
                        "log": str(log),
                    },
                )
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            finally:
                timer.cancel()
        self.lifecycle_event(
            "loaded",
            store.image_id,
            seconds=time.perf_counter() - started,
            returncode=code,
            compressed_bytes=store.record.get("compressed_bytes"),
        )
        if code or not self._exists(store.image_id):
            raise RuntimeError(f"docker load failed; see {log}")

    @contextmanager
    def lease(self, store):
        image_id = store.image_id
        name = image_id.replace(":", "-")
        try:
            from runtime.docker_storage import storage_lease, assert_data_root

            with storage_lease(self.cfg), (self.state / (name + ".lease")).open("a+") as lifetime:
                assert_data_root(self.cfg, self.client.info())
                fcntl.flock(lifetime, fcntl.LOCK_SH)
                with file_lock(self.state / (name + ".load.lock")):
                    present = self._exists(image_id)
                    self.lifecycle_event("lease", image_id, present_before=present)
                    if present and self.cfg.get("require_cold_images"):
                        raise RuntimeError(f"Cold-start audit found an existing image: {image_id}")
                    if not present:
                        estimate = max(
                            int(store.record["compressed_bytes"] * self.options["expansion_factor"]),
                            self.options["minimum_load_bytes"],
                        )
                        with self._slot(), self._reservation(estimate):
                            # Persist import intent before load so a crash is recoverable.
                            with file_lock(self.state / "journal.lock"):
                                state = self._read()
                                state["images"][image_id] = {"owned": True}
                                self._save(state)
                            self._load(store)
                    # Pre-existing images are intentionally not enrolled as owned.
                yield image_id
        finally:
            if self.options["cleanup"]:
                self.cleanup(image_id)

    def cleanup(self, only=None):
        import docker

        with file_lock(self.state / "journal.lock"):
            ids = list(self._read()["images"])
        results = {}
        for image_id in ids:
            if only and image_id != only:
                continue
            try:
                with (self.state / (image_id.replace(":", "-") + ".lease")).open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if self.client.containers.list(all=True, filters={"ancestor": image_id}):
                        results[image_id] = "in_use"
                        continue
                    try:
                        self.client.images.remove(image_id, force=False)
                    except docker.errors.ImageNotFound:
                        pass
                    with file_lock(self.state / "journal.lock"):
                        state = self._read()
                        state["images"].pop(image_id, None)
                        self._save(state)
                    results[image_id] = "removed"
            except BlockingIOError:
                results[image_id] = "leased"
            except (docker.errors.APIError, RuntimeError, OSError) as exc:
                results[image_id] = f"deferred: {exc}"
        for image_id, result in results.items():
            self.lifecycle_event("cleanup", image_id, result=result)
        return results


def _owner(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None
