"""Run NAS leases, isolated worker processes and exact Harbor-owned cleanup."""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from core.cancellation import Cancelled
from core.deadline import Deadline
from core.io import hash_json, read_json, write_json


def cpu_quota_supported(info, docker_host):
    if info.get("CpuCfsQuota") is False or info.get("CpuCfsPeriod") is False:
        return False
    if any("No cpu cfs quota support" in str(w) for w in (info.get("Warnings") or [])):
        return False
    # Rootless Docker can advertise host CPU support even when systemd has
    # delegated only memory/pids to this user. Check the local delegation too.
    if (str(docker_host).startswith("unix://") and str(info.get("CgroupVersion")) == "2"
            and "name=rootless" in (info.get("SecurityOptions") or [])):
        uid = os.getuid()
        controllers = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/cgroup.controllers")
        if controllers.is_file() and "cpu" not in controllers.read_text().split():
            return False
    return True


class HarborFactory:
    def __init__(self, cfg, run_id):
        import docker

        from runtime.images.nas import ImageLeaseManager
        self.cfg, self.run_id = cfg, run_id
        self.owner_id = uuid.uuid4().hex
        self.deadline = Deadline(cfg["execution"]["pipeline_wall_seconds"])
        self.client = docker.DockerClient(base_url=cfg["environment"]["docker_host"])
        self.leases = ImageLeaseManager(cfg, self.client)
        self.stack, self.images = contextlib.ExitStack(), {}
        self.processes, self.lock = {}, threading.Lock()
        info = self.client.info()
        self.cpu_supported = cpu_quota_supported(info, cfg["environment"]["docker_host"])
        self.started = False

    def lock_images(self, tasks):
        if self.started:
            return
        from runtime.images.nas import NasImageStore
        refs = self.cfg["execution_context"]["image_references"]
        try:
            for image in sorted({v for t in tasks for v in t["images"].values()}):
                store = NasImageStore(self.cfg, refs[image])
                self.stack.enter_context(self.leases.lease(store))
                tag = "dycon-harbor:" + hash_json(image)[:32]
                self.client.api.tag(store.image_id, tag)
                inspected = self.client.images.get(tag)
                self.images[image] = {"tag": tag, "identity": store.identity(),
                                      "reference": refs[image], "docker_id": inspected.id}
        except BaseException:
            self.stack.close()
            raise
        self.started = True

    def cleanup_attempt(self, attempt_id, sample_key, output=None):
        labels = ["dycon.run=" + self.run_id, "dycon.sample=" + sample_key,
                  "dycon.attempt=" + attempt_id, "dycon.backend=harbor"]
        removed = []
        for container in self.client.containers.list(all=True, filters={"label": labels}):
            removed.append(container.id)
            container.remove(force=True, v=True)
        for network in self.client.networks.list(filters={"label": labels}):
            network.remove()
        residual = [c.id for c in self.client.containers.list(all=True, filters={"label": labels})]
        receipt = {"run_id": self.run_id, "attempt_id": attempt_id, "removed": removed,
                   "residual": residual, "complete": not residual}
        if output is not None:
            write_json(Path(output) / "cleanup.json", receipt)
        if residual:
            raise RuntimeError("Harbor owned containers remain after cleanup")
        return receipt

    def execute_worker(self, request, *, secret=None, cancellation=None, seconds=3600):
        output = Path(request["output"])
        output.mkdir(parents=True, exist_ok=True)
        request["worker_protocol"] = 1
        request["response_path"] = str(output / "worker-response.json")
        path = output / "worker-request.json"
        write_json(path, request)
        env = {**os.environ, "DOCKER_HOST": self.cfg["environment"]["docker_host"],
               "PYTHONDONTWRITEBYTECODE": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "true"}
        env.pop("DOCKER_CONTEXT", None)
        if secret is not None:
            env["DYCON_MODEL_CHANNEL_SECRET"] = secret
        python = self.cfg["execution_context"]["worker_python"]
        process = subprocess.Popen([python, "-m", "runtime.backends.harbor.worker", str(path)],
            cwd=self.cfg["project_root"], env=env, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with self.lock:
            self.processes[(request["binding"]["sample_key"], request["attempt_id"])] = process

        def stop():
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)

        remove = cancellation.add_callback(stop) if cancellation is not None else lambda: None
        deadline = time.monotonic() + seconds
        log = b""
        try:
            while True:
                try:
                    log, _ = process.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    self.deadline.check()
                    if cancellation is not None and cancellation.event.is_set():
                        stop()
                        raise Cancelled("Harbor attempt cancelled")
                    if time.monotonic() >= deadline:
                        stop()
                        raise TimeoutError("Harbor physical attempt deadline exceeded")
            if cancellation is not None:
                cancellation.check()
            if process.returncode:
                raise RuntimeError("Harbor worker failed; see worker.log")
            return read_json(output / "worker-response.json")
        finally:
            remove()
            if process.poll() is None:
                stop()
                try:
                    log, _ = process.communicate(timeout=self.cfg["execution"].get("cancellation", {}).get("term_grace_seconds", 30))
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    log, _ = process.communicate()
            text = log.decode("utf-8", "replace")
            (output / "worker.log").write_text(text.replace(secret, "<redacted>") if secret else text)
            with self.lock:
                self.processes.pop((request["binding"]["sample_key"], request["attempt_id"]), None)
            self.cleanup_attempt(request["attempt_id"], request["binding"]["sample_key"], output)

    def cleanup_owned(self, run_id):
        if run_id != self.run_id:
            raise ValueError("Harbor factory cannot clean another run")
        with self.lock:
            active = list(self.processes.values())
        for process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        try:
            labels = ["dycon.run=" + run_id, "dycon.backend=harbor", "dycon.owner=" + self.owner_id]
            for container in self.client.containers.list(all=True, filters={"label": labels}):
                container.remove(force=True, v=True)
            for network in self.client.networks.list(filters={"label": labels}):
                network.remove()
        finally:
            self.stack.close()
            self.client.close()

    cleanup_cancelled = cleanup_owned
