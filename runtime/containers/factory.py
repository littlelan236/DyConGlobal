from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
import requests

from core.io import hash_json, now, write_json
from runtime.containers.policy import EnvironmentPolicy, parse_probe, probe_command


class EnvironmentSetupFailed(RuntimeError):
    pass


class ContainerCorrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskContext:
    run_id: str
    instance_id: str
    attempt_id: str
    role: str
    output_dir: Path


def image_name(instance: dict) -> str:
    return (
        instance.get("image_name")
        or instance.get("docker_image")
        or (
            "swebench/sweb.eval.x86_64." + instance["instance_id"].replace("__", "_1776_") + ":latest"
        ).lower()
    )


class ManagedContainer:
    """Docker SDK-compatible container with bootstrap at its start boundary."""

    def __init__(self, factory, container, context: TaskContext, instance: dict, spec: dict):
        self.factory, self.raw, self.context, self.instance, self.spec = (
            factory,
            container,
            context,
            instance,
            spec,
        )
        self.started = False
        self.policy = factory.policy_for(instance)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def start(self):
        if self.started:
            return
        self.raw.start()
        self.started = True
        try:
            bootstrap = self.policy.bootstrap(self.instance["repo"])
            result = self.factory.execute(self, bootstrap, self.factory.cfg["timeouts"]["bootstrap_seconds"])
            (self.context.output_dir / "bootstrap.log").write_text(result["output"])
            if result["returncode"] != 0:
                raise EnvironmentSetupFailed(f"Bootstrap failed: {result['output'][-2000:]}")
            self.factory.probe(self, "environment.json")
            self.factory.event(self.context, "ready", container_id=self.id)
        except Exception:
            self.factory.close(self)
            raise

    def exec_run(self, cmd, **kwargs):
        # Official harness copy/patch helpers call this API directly.
        env = self.policy.variables()
        supplied = kwargs.pop("environment", None) or {}
        if not isinstance(supplied, dict):
            supplied = dict(entry.split("=", 1) for entry in supplied)
        kwargs["environment"] = {**supplied, **env}
        kwargs.setdefault("workdir", self.factory.cfg["environment"]["cwd"])
        return self.raw.exec_run(cmd, **kwargs)

    def remove(self, **kwargs):
        try:
            return self.raw.remove(**kwargs)
        finally:
            self.factory.event(self.context, "removed", container_id=self.id)


class ContainerFactory:
    def __init__(self, cfg: dict, images: dict | None = None, client=None, deadline=None):
        self.cfg = cfg
        self.client = client or docker.DockerClient(
            base_url=cfg["environment"]["docker_host"], timeout=cfg["timeouts"]["container_start_seconds"]
        )
        from core.deadline import Deadline

        self.deadline = deadline or Deadline(cfg["execution"]["pipeline_wall_seconds"])
        self.policy = EnvironmentPolicy(cfg)
        self._task_policies = {}
        self.images = dict(images or {})
        self._lock = threading.Lock()
        self._leases = {}
        self._image_manager = None

    @property
    def image_manager(self):
        if self._image_manager is None:
            from runtime.images.nas import ImageLeaseManager

            with self._lock:
                if self._image_manager is None:
                    self._image_manager = ImageLeaseManager(self.cfg, self.client)
        return self._image_manager

    def policy_for(self, instance: dict) -> EnvironmentPolicy:
        from runtime.containers.repairs import environment_overrides

        overrides = environment_overrides(self.cfg, instance)
        if not overrides:
            return self.policy
        from core.config import Environment

        iid = instance["instance_id"]
        with self._lock:
            if iid not in self._task_policies:
                environment = dict(self.cfg["environment"])
                environment.update(overrides)
                validated = Environment.model_validate(environment).model_dump()
                self._task_policies[iid] = EnvironmentPolicy({**self.cfg, "environment": validated})
            return self._task_policies[iid]

    def event(self, context: TaskContext, event: str, **extra):
        context.output_dir.mkdir(parents=True, exist_ok=True)
        line = (
            json.dumps(
                {
                    "at": now(),
                    "event": event,
                    "role": context.role,
                    "run_id": context.run_id,
                    "instance_id": context.instance_id,
                    "attempt_id": context.attempt_id,
                    **extra,
                }
            )
            + "\n"
        )
        with self._lock, (context.output_dir / "container-events.jsonl").open("a") as f:
            f.write(line)

    def lock_images(self, instances: list[dict]) -> dict:
        for row in instances:
            iid = row["instance_id"]
            if row.get("nas_image"):
                from runtime.images.nas import NasImageStore

                identity = NasImageStore(self.cfg, row["nas_image"]).identity()
                if iid in self.images and self.images[iid]["image_id"] != identity["image_id"]:
                    raise ValueError(f"NAS image identity changed for {iid}")
                # Lock immutable OCI identities without loading the whole cohort.
                self.images[iid] = identity
                continue
            raise ValueError(
                "Task images must have immutable NAS references; local-only fallback is disabled"
            )
        return dict(self.images)

    def spec(self, instance: dict) -> dict:
        policy = self.policy_for(instance)
        e = policy.environment
        self.lock_images([instance])
        return {
            "image_id": self.images[instance["instance_id"]]["image_id"],
            "user": e["user"],
            "platform": e["platform"],
            "network_mode": e.get("network_mode", "none"),
            "cwd": e["cwd"],
            "environment": policy.variables(),
            "volumes": {e["wheelhouse"]: {"bind": "/opt/dycon-wheels", "mode": "ro"}}
            if e["wheelhouse"]
            else {},
            "command": ["sleep", str(self.cfg["timeouts"]["container_lifetime_seconds"])],
            "policy_hash": policy.identity(instance["repo"])["policy_hash"],
        }

    def create(self, instance: dict, context: TaskContext) -> ManagedContainer:
        self.deadline.check()
        if not instance.get("nas_image"):
            raise ValueError("Missing NAS image reference")
        from runtime.images.nas import NasImageStore

        store = NasImageStore(self.cfg, instance["nas_image"])
        lease = self.image_manager.lease(store)
        lease.__enter__()
        try:
            # Docker rootless may reset the API connection while concurrently
            # creating containers (especially immediately after image loads).
            # All callers share this entry point, so retry creation here rather
            # than adding divergent retry logic to rollout or harness paths.
            attempts = 1 + self.cfg["execution"].get("infrastructure_retry_limit", 0)
            for attempt in range(attempts):
                try:
                    container = self._create(instance, context)
                    with self._lock:
                        self._leases[container.id] = lease
                    return container
                except (requests.exceptions.ConnectionError, docker.errors.APIError) as exc:
                    message = str(exc).lower()
                    transient = isinstance(exc, requests.exceptions.ConnectionError) or any(
                        marker in message
                        for marker in ["connection reset", "broken pipe", "internal server error"]
                    )
                    if not transient or attempt + 1 >= attempts:
                        raise
                    self.event(context, "create_retry", attempt=attempt + 1, error=repr(exc))
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
        except BaseException:
            lease.__exit__(*__import__("sys").exc_info())
            raise

    def _create(self, instance: dict, context: TaskContext) -> ManagedContainer:
        spec = self.spec(instance)
        context.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(context.output_dir / "container-spec.json", {**spec, "spec_hash": hash_json(spec)})
        write_json(
            context.output_dir / "environment-policy.json",
            self.policy_for(instance).identity(instance["repo"]),
        )
        container = self.client.containers.create(
            image=spec["image_id"],
            user=spec["user"],
            platform=spec["platform"],
            network_mode=spec["network_mode"],
            working_dir=spec["cwd"],
            environment=spec["environment"],
            volumes=spec["volumes"],
            command=spec["command"],
            detach=True,
            name="dycon-" + context.role + "-" + uuid.uuid4().hex[:16],
            labels={
                "dycon.project": str(self.cfg["project_root"]),
                "dycon.run": context.run_id,
                "dycon.task": context.instance_id,
                "dycon.attempt": context.attempt_id,
                "dycon.role": context.role,
            },
        )
        self.event(context, "created", container_id=container.id, spec_hash=hash_json(spec))
        return ManagedContainer(self, container, context, instance, spec)

    def execute(
        self, container: ManagedContainer, command: str, seconds: int, cwd: str | None = None
    ) -> dict:
        seconds = self.deadline.remaining(seconds)
        if container.started:
            container.raw.reload()
            if container.raw.status != "running":
                raise ContainerCorrupted(f"Task container {container.id} is {container.raw.status}")
        result = container.exec_run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=5",
                str(seconds),
                "bash",
                "-lc",
                "source /opt/miniconda3/bin/activate && conda activate testbed && " + command,
            ],
            workdir=cwd or self.cfg["environment"]["cwd"],
        )
        code = result.exit_code
        return {
            "output": result.output.decode("utf-8", errors="replace"),
            "returncode": code,
            "exception_info": "CommandTimeout" if code in [124, 137] else "",
        }

    def probe(self, container: ManagedContainer, filename: str) -> dict:
        out = self.execute(container, probe_command(), self.cfg["timeouts"]["bootstrap_seconds"])
        if out["returncode"] != 0:
            raise EnvironmentSetupFailed(out["output"][-2000:])
        data = parse_probe(out["output"])
        data["spec_hash"] = hash_json(container.spec)
        data["policy_hash"] = container.spec["policy_hash"]
        data["packages_hash"] = hash_json(data["packages"])
        write_json(container.context.output_dir / filename, data)
        return data

    def close(self, container: ManagedContainer):
        removed = False
        try:
            container.remove(force=True)
            removed = True
        except docker.errors.NotFound:
            removed = True
        finally:
            if removed:
                with self._lock:
                    lease = self._leases.pop(container.id, None)
                if lease:
                    lease.__exit__(None, None, None)

    def cleanup_owned(self, run_id: str):
        for container in self.client.containers.list(
            all=True, filters={"label": ["dycon.project=" + self.cfg["project_root"], "dycon.run=" + run_id]}
        ):
            self.close(container)
        if self._image_manager:
            self._image_manager.cleanup()
