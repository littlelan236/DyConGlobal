"""Pinned Harbor environment extension for NAS images and exact ownership.

Imported only in the Harbor worker. Network switching uses Docker's own network
attachments, so offline model transport needs no egress sidecar or implicit build.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import uuid
from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode

from core.io import write_json


class DyConDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, dycon, task_env_config, extra_docker_compose=(), **kwargs):
        self.dycon = dycon
        self._saved_networks = {}
        self._saved_dns = None
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
        if dycon.get("channel_dir"):
            mounts.append({"type": "bind", "source": dycon["channel_dir"], "target": "/run/dycon-model",
                           "read_only": True})
        main = {"image": self.image["tag"], "pull_policy": "never", "labels": labels, "volumes": mounts}
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
        write_json(self.environment_receipt, {"container_id": cid, "ownership": self.dycon["ownership"],
                   "image": self.image, "network_phase": "setup", "implicit_pull": False,
                   "implicit_build": False, "resources": self.dycon["resources"]})
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

    async def acp_command(self, environment):
        cid = await self.container_id()
        args = ["exec", "--interactive", "--user", "agent", "--workdir", environment["DYCON_AGENT_CWD"]]
        for name in environment:
            args.extend(["--env", name])
        args.extend([cid, "/opt/dycon-openhands/bin/python", "/opt/dycon-model-socket.py"])
        return "docker", args, {**os.environ, **environment}
