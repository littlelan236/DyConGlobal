"""Pinned Harbor environment extension for NAS images and exact ownership.

Imported only in the Harbor worker. Network switching uses Docker's own network
attachments, so offline model transport needs no egress sidecar or implicit build.
"""
from __future__ import annotations

import asyncio
import math
import json
import os
import re
import shlex
import uuid
from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode

from core.io import hash_json, write_json
from runtime.backends.harbor.task_dependencies import apt_mirror_setup_command, task_dependency_overlay, resource_cache_setup_command


class DyConDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, dycon, task_env_config, extra_docker_compose=(), **kwargs):
        self.dycon = dycon
        self._saved_networks = {}
        self._saved_dns = None
        self._affinity_cpus = None
        preparation = dycon.get("task_preparation") or {}
        if preparation.get("cpu_affinity"):
            allowed = sorted(os.sched_getaffinity(0))
            count = max(1, math.ceil(dycon["resources"]["requested_cpus"]))
            if count > len(allowed):
                raise ValueError("Insufficient host CPUs for requested task affinity")
            offset = int(hash_json(dycon["ownership"])[:8], 16) % len(allowed)
            selected = sorted(allowed[(offset + i) % len(allowed)] for i in range(count))
            self._affinity_cpus = ",".join(map(str, selected))
        original_image = task_env_config.docker_image
        if original_image not in dycon["images"]:
            raise ValueError("Harbor image was not admitted through the run NAS lease")
        self.image = dycon["images"][original_image]
        task_env_config = task_env_config.model_copy(update={"docker_image": self.image["tag"]})
        output = Path(dycon["output"])
        self.environment_receipt = output / ("environment-" + uuid.uuid4().hex + ".json")
        overlay = output / ("compose-owner-" + uuid.uuid4().hex + ".json")
        labels = {"dycon.managed": "true", "dycon.run": dycon["run_id"],
                  "dycon.sample": dycon["sample_key"], "dycon.attempt": dycon["attempt_id"],
                  "dycon.backend": "harbor"}
        labels["dycon.owner"] = dycon["owner_id"]
        mounts = [{"type": "bind", "source": dycon["dependency_snapshot"],
                   "target": "/opt/dycon-wheels", "read_only": True}]
        task_mounts, task_environment = task_dependency_overlay(dycon, task_env_config.network_mode)
        self._task_dependency_environment = task_environment
        mounts.extend(task_mounts)
        if dycon.get("channel_dir"):
            mounts.append({"type": "bind", "source": dycon["channel_dir"], "target": "/run/dycon-model",
                           "read_only": True})
        main = {"image": self.image["tag"], "pull_policy": "never", "labels": labels, "volumes": mounts}
        if task_environment:
            main["environment"] = task_environment
        if dycon.get("benchmark") == "swebench_pro_v2":
            main["network_mode"] = "none"
        write_json(overlay, {"services": {"main": main}, "networks": {"default": {"labels": labels}}})
        super().__init__(*args, task_env_config=task_env_config,
                         extra_docker_compose=[*extra_docker_compose, overlay], **kwargs)

    @classmethod
    def _requires_egress_control(cls, **kwargs):
        return False

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"disable_internet": True, "dynamic_network_policy": True})

    def validate_network_policy_support(self, network_policy=None):
        BaseEnvironment.validate_network_policy_support(self, network_policy)

    async def _docker(self, *args):
        process = await asyncio.create_subprocess_exec("docker", *args, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
        if process.returncode:
            raise RuntimeError("Owned Docker operation failed: " + stderr.decode()[-2000:])
        return stdout.decode()

    async def container_id(self):
        result = await super()._run_docker_compose_command(["ps", "-q", "main"])
        cid = (result.stdout or "").strip()
        if len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
            raise ValueError("Harbor did not return exactly one owned main container")
        return cid

    async def _run_docker_compose_command(self, command, *args, **kwargs):
        if command[0] in {"build", "pull"}:
            raise ValueError("Harbor implicit image build/pull is disabled")
        if command[0] == "up":
            command = [*command, "--pull", "never", "--no-build"]
        if "--rmi" in command:
            command = list(command)
            index = command.index("--rmi")
            del command[index:index + 2]  # Only the NAS lease manager releases images.
        return await super()._run_docker_compose_command(command, *args, **kwargs)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        # Pinned Harbor discards this return code when parsing the reward file.
        # Capture only its official verifier invocation, never agent commands/env.
        verifier = re.fullmatch(
            r"\(/tests/test\.sh\)\s*>\s*/logs/verifier/test-stdout\.txt\s+2>&1", command.strip())
        if self._affinity_cpus:
            command = "exec /usr/bin/taskset -c " + self._affinity_cpus + " /bin/sh -c " + shlex.quote(command)
        if not verifier or self.dycon.get("benchmark") != "terminalbench_2_1":
            return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        receipt = {"schema_version": 1, "attempt_id": self.dycon["attempt_id"],
                   "return_code": None}
        try:
            result = await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
            receipt["return_code"] = result.return_code
            return result
        except BaseException as exc:
            receipt["exception_type"] = type(exc).__name__
            raise
        finally:
            write_json(Path(self.dycon["output"]) / "verifier-execution.json", receipt)
            for name, diagnostic in (self.dycon.get("task_preparation") or {}).get("post_verifier_diagnostics", {}).items():
                if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
                    raise ValueError("Invalid diagnostic artifact name")
                try:
                    result = await self.exec(diagnostic, user="root", timeout_sec=60)
                    write_json(Path(self.dycon["output"]) / ("diagnostic-" + name + ".json"), {"scope": "post-verifier diagnostic; not scoring evidence", "return_code": result.return_code, "stdout": result.stdout, "stderr": result.stderr})
                except Exception as exc:
                    write_json(Path(self.dycon["output"]) / ("diagnostic-" + name + ".json"), {"error": type(exc).__name__})
            try:
                result = await self.exec("for f in memory.max memory.peak memory.events cpu.max cpuset.cpus.effective; do if [ -r /sys/fs/cgroup/$f ]; then printf '%s\\n' \"$f\"; cat /sys/fs/cgroup/$f; fi; done; grep Cpus_allowed_list /proc/self/status; nproc", user="root", timeout_sec=10)
                write_json(Path(self.dycon["output"]) / "resource-final.json", {"return_code": result.return_code, "output": result.stdout[-4000:]})
            except Exception as exc:
                write_json(Path(self.dycon["output"]) / "resource-final.json", {"error": type(exc).__name__})

    async def start(self, force_build=False):
        if force_build:
            raise ValueError("Harbor force_build is disabled for admitted NAS images")
        await self._docker("image", "inspect", self.image["tag"])
        await super().start(False)
        cid = await self.container_id()
        inspected = json.loads(await self._docker("inspect", cid))[0]
        config_digest = self.image["docker_id"]
        if inspected["Image"] != config_digest or inspected["Config"]["Labels"].get("dycon.attempt") != self.dycon["attempt_id"]:
            raise ValueError("Harbor container image/ownership identity changed")
        self._saved_networks = dict(inspected["NetworkSettings"]["Networks"])
        constraints = (self.dycon.get("task_preparation") or {}).get("python_constraints", [])
        if constraints:
            if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9_.+!-]+", item) for item in constraints):
                raise ValueError("Invalid frozen Python dependency constraint")
            result = await self.exec("printf %s " + shlex.quote("\n".join(constraints) + "\n") + " > /opt/dycon-task-constraints.txt", user="root", timeout_sec=30)
            if result.return_code:
                raise RuntimeError("Task Python constraint preparation failed")
        cache = self.dycon.get("task_resource_cache")
        if cache and self.network_policy.network_mode == NetworkMode.PUBLIC:
            result = await self.exec(resource_cache_setup_command(cache), user="root", timeout_sec=30)
            if result.return_code:
                raise RuntimeError("Resource cache trust setup failed: " + (result.stderr or result.stdout)[-1000:])
        apt_mirror_receipt = {"status": "not_configured"}
        if self.dycon.get("task_apt_mirror"):
            if self.network_policy.network_mode == NetworkMode.PUBLIC:
                warmup = bool(self.dycon.get("task_apt_mirror_warmup"))
                result = await self.exec(apt_mirror_setup_command(
                    self.dycon["task_apt_mirror"], warmup=warmup), user="root", timeout_sec=180)
                if result.return_code:
                    raise RuntimeError("Ubuntu APT mirror setup failed: " + (result.stderr or result.stdout)[-2000:])
                apt_mirror_receipt = {"status": "applied" if "applied:" in result.stdout else "skipped",
                    "requested": self.dycon["task_apt_mirror"], "warmup": warmup,
                    "output": result.stdout[-1000:]}
            else:
                apt_mirror_receipt = {"status": "skipped_no_network",
                                      "requested": self.dycon["task_apt_mirror"]}
        preparation = self.dycon.get("task_preparation") or {}
        preparation_receipt = {"requested": preparation, "cpu_affinity": self._affinity_cpus}
        if self._affinity_cpus:
            # Include the container init, so children created outside exec inherit
            # its mask. Unified exec also explicitly sets the same kernel mask.
            result = await self.exec("taskset -apc " + self._affinity_cpus + " 1; nproc; grep Cpus_allowed_list /proc/self/status", user="root", timeout_sec=30)
            if result.return_code:
                raise RuntimeError("Task CPU affinity setup failed: " + (result.stderr or result.stdout)[-1000:])
            preparation_receipt["affinity_probe"] = result.stdout[-1500:]
        if preparation.get("sshd_runtime_directory"):
            result = await self.exec("set -eu; for p in / /run /var/run /run/sshd; do stat -Lc '%n %u:%g %a' \"$p\" 2>/dev/null || true; done; if [ -e /run/sshd ] && { [ -L /run/sshd ] || [ ! -d /run/sshd ]; }; then exit 74; fi; install -d -o 0 -g 0 -m 0755 /run/sshd; stat -Lc '%n %u:%g %a' /run/sshd", user="root", timeout_sec=30)
            if result.return_code:
                raise RuntimeError("sshd runtime directory preparation failed")
            preparation_receipt["runtime_directory"] = result.stdout[-1500:]
        if preparation.get("system_packages"):
            packages = preparation["system_packages"]
            if any(not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(?:=[A-Za-z0-9+:~.\-]+)?", p) for p in packages):
                raise ValueError("Invalid frozen system package")
            result = await self.exec("set -eu; apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends " + " ".join(shlex.quote(p) for p in packages) + "; dpkg-query -W " + " ".join(shlex.quote(p.split('=')[0]) for p in packages), user="root", timeout_sec=600)
            if result.return_code:
                raise RuntimeError("Task system dependency preparation failed: " + (result.stderr or result.stdout)[-2000:])
            preparation_receipt["system_packages"] = result.stdout[-1500:]
        write_json(self.environment_receipt, {"preparation": preparation_receipt, "container_id": cid, "ownership": self.dycon["ownership"],
                   "image": self.image, "network_phase": "setup", "implicit_pull": False,
                   "implicit_build": False, "resources": self.dycon["resources"],
                   "terminal_policy": self.dycon.get("terminal_policy"),
                   "task_dependency_sources": {
                       "python_index": self._task_dependency_environment.get("UV_DEFAULT_INDEX"),
                       "wheelhouse_manifest_sha256": self.dycon.get("task_wheelhouse_manifest_sha256")
                           if self.dycon.get("task_wheelhouse") else None},
                   "resource_cache": {k: cache[k] for k in ("mode", "ca_sha256", "seal_sha256") if k in cache} if cache else None,
                   "apt_mirror": apt_mirror_receipt,
                   "actual_resources": {k: inspected["HostConfig"].get(k) for k in
                       ("NanoCpus", "CpuQuota", "CpuPeriod", "Memory", "StorageOpt")},
                   "storage_hard_quota_enforced": False})
        if self.network_policy.network_mode == NetworkMode.NO_NETWORK:
            await self._apply_network_policy(self.network_policy)

    async def _apply_network_policy(self, network_policy):
        mode = network_policy.network_mode
        # Pro V2's agent environment is permanently offline. Harbor may ask
        # this environment to restore PUBLIC after agent setup because the
        # verifier has its own separate environment; allowing that request
        # would silently violate the locked protocol.
        if self.dycon.get("benchmark") == "swebench_pro_v2" and mode == NetworkMode.PUBLIC:
            mode = NetworkMode.NO_NETWORK
        if mode not in {NetworkMode.PUBLIC, NetworkMode.NO_NETWORK}:
            raise ValueError("Harbor allowlists require separate qualification; model uses a dedicated Unix channel")
        cid = await self.container_id()
        current = json.loads(await self._docker("inspect", cid))[0]["NetworkSettings"]["Networks"]
        if mode == NetworkMode.NO_NETWORK:
            if self._saved_dns is None:
                dns = await self.exec("cat /etc/resolv.conf", user="root", timeout_sec=10)
                self._saved_dns = dns.stdout
            changed = await self.exec("printf 'nameserver 127.0.0.1\\noptions timeout:1 attempts:1\\n' > /etc/resolv.conf",
                                      user="root", timeout_sec=10)
            if changed.return_code:
                raise ValueError("Could not disable external DNS in offline agent phase")
            for name in current:
                await self._docker("network", "disconnect", name, cid)
        else:
            for name, settings in self._saved_networks.items():
                if name not in current:
                    await self._docker("network", "connect", settings["NetworkID"], cid)
            if self._saved_dns is not None:
                await self.exec("printf %s " + shlex.quote(self._saved_dns) + " > /etc/resolv.conf",
                                user="root", timeout_sec=10)
        actual = json.loads(await self._docker("inspect", cid))[0]["NetworkSettings"]["Networks"]
        if mode == NetworkMode.NO_NETWORK and actual:
            raise ValueError("Harbor failed to disconnect agent networking")
        record = json.loads(self.environment_receipt.read_text())
        record.setdefault("phase_networks", []).append({"mode": mode.value, "networks": sorted(actual)})
        write_json(self.environment_receipt, record)

    async def terminal_identity(self):
        cid = await self.container_id()
        inspected = json.loads(await self._docker("inspect", cid))[0]
        cwd = self.task_env_config.workdir or inspected["Config"].get("WorkingDir") or "/"
        # Bypass login-shell PATH rewrites when resolving the official image.
        value = await self._docker("exec", "--user", "root", "--workdir", cwd,
            "--env", "HOME=/root", cid, "/opt/dycon-openhands/bin/python", "-c",
            "import json,os; print(json.dumps({'cwd':os.getcwd(),'path':os.environ['PATH'],"
            "'uid':os.getuid(),'home':os.environ['HOME']}))")
        identity = json.loads(value)
        if identity["uid"] != 0 or identity["home"] != "/root":
            raise ValueError("Terminal execution identity differs from frozen policy")
        return identity

    async def acp_command(self, environment):
        cid = await self.container_id()
        policy = self.dycon.get("terminal_policy")
        user = policy["user"] if policy else "agent"
        args = ["exec", "--interactive", "--user", user, "--workdir", environment["DYCON_AGENT_CWD"]]
        for name in environment:
            args.extend(["--env", name])
        args.extend([cid, "/opt/dycon-openhands/bin/python", "/opt/dycon-model-socket.py"])
        return "docker", args, {**os.environ, **environment}
