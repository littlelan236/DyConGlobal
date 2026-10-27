from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
import requests

from core.io import hash_json, now, sha256, write_json
from runtime.containers.policy import EnvironmentPolicy, parse_probe, probe_command
from runtime.containers.dependencies import setup_error


class EnvironmentSetupFailed(RuntimeError):
    def __init__(self, message, *, dependency_failure=None):
        super().__init__(message)
        self.dependency_failure = dependency_failure


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
            receipt = {"schema_version": 1, "instance_id": self.instance["instance_id"],
                       "attempt_id": self.context.attempt_id, "container_id": self.id,
                       "policy_hash": self.spec["policy_hash"],
                       "requirements": self.policy.specs(self.instance["repo"]),
                       "constraints": self.policy.specs(self.instance["repo"]),
                       "wheel_hashes": self.policy.wheel_hashes,
                       "offline": not self.policy.environment["network_fallback"],
                       "bootstrap_log_sha256": sha256(self.context.output_dir / "bootstrap.log"),
                       "returncode": result["returncode"], "status": "failed", "at": now()}
            write_json(self.context.output_dir / "bootstrap-receipt.json", receipt)
            if result["returncode"] != 0:
                error = setup_error(self.factory.cfg, self.context, result['output'],
                    stage='bootstrap', returncode=result['returncode'],
                    requirements=self.policy.specs(self.instance['repo']),
                    wheelhouse=self.policy.environment.get('wheelhouse'),
                    log_path=self.context.output_dir / 'bootstrap.log')
                if error.dependency_failure:
                    write_json(self.context.output_dir / 'bootstrap-receipt.json', {
                        **receipt, 'dependency_failure': error.dependency_failure})
                raise error
            probe = self.factory.probe(self, "environment.json")
            write_json(self.context.output_dir / "bootstrap-receipt.json", {
                **receipt, "status": "ready", "packages_hash": probe["packages_hash"]})
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
        if cfg["environment"].get("docker_data_root"):
            from runtime.docker_storage import assert_data_root

            assert_data_root(cfg, self.client.info())
        try:
            self.policy = EnvironmentPolicy(cfg)
        except ValueError as exc:
            run_id = cfg.get('run_id') or cfg.get('evaluation_run_id')
            if not run_id:
                raise
            context = TaskContext(run_id, '__run__', 'preflight', 'environment-preflight',
                Path(cfg['project_root']) / 'state/environment-failures' / run_id / 'preflight')
            raise setup_error(cfg, context, str(exc), stage='preflight') from exc
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
        shared_wheelhouse = self.cfg["environment"].get("wheelhouse")
        if shared_wheelhouse:
            # A configured root wheelhouse is the common mount. Preserve task
            # pins and every other override; legacy snapshots with null retain
            # their task-specific mount behavior.
            overrides["wheelhouse"] = shared_wheelhouse
        if not overrides:
            return self.policy
        from core.config import Environment

        iid = instance["instance_id"]
        with self._lock:
            if iid not in self._task_policies:
                environment = dict(self.cfg["environment"])
                environment.update(overrides)
                validated = Environment.model_validate(environment).model_dump()
                hashes = self.policy.wheel_hashes if shared_wheelhouse else None
                self._task_policies[iid] = EnvironmentPolicy(
                    {**self.cfg, "environment": validated}, wheel_hashes=hashes)
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
        # Validate offline inputs before loading a large image or acquiring its lease.
        try:
            self.policy_for(instance).identity(instance["repo"])
        except ValueError as exc:
            write_json(context.output_dir / "environment-preflight.json", {
                "status": "failed", "instance_id": instance["instance_id"],
                "attempt_id": context.attempt_id, "reason": str(exc), "at": now()})
            raise setup_error(self.cfg, context, str(exc), stage='preflight',
                wheelhouse=self.cfg['environment'].get('wheelhouse') or
                    (instance.get('environment_overrides') or {}).get('wheelhouse'),
                log_path=context.output_dir / 'environment-preflight.json') from exc
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
                **({"dycon.execution": os.environ["DYCON_EXECUTION_ATTEMPT"]}
                   if os.environ.get("DYCON_EXECUTION_ATTEMPT") else {}),
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
            reason = None
            failed = []
            try:
                failed = [c for c in parse_probe(out['output']).get('constraints', []) if not c.get('satisfied')]
                if failed:
                    reason = 'installed_version_mismatch'
            except ValueError:
                pass
            log = container.context.output_dir / 'probe-error.log'
            log.write_text(out['output'], encoding='utf-8')
            raise setup_error(self.cfg, container.context, out['output'], stage='probe',
                returncode=out['returncode'], reason=reason,
                requirements=container.policy.specs(container.instance['repo']),
                wheelhouse=container.policy.environment.get('wheelhouse'), log_path=log, version_checks=failed)
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

    def cleanup_cancelled(self, run_id: str):
        return self.cleanup_owned(run_id, cleanup_images=False)

    def cleanup_owned(self, run_id: str, *, cleanup_images=True):
        for container in self.client.containers.list(
            all=True, filters={"label": ["dycon.project=" + self.cfg["project_root"], "dycon.run=" + run_id]}
        ):
            self.close(container)
        if self._image_manager and cleanup_images:
            self._image_manager.cleanup()


def cleanup_execution_containers(cfg, run_id, execution_attempt_id, client=None):
    """Cancellation cleanup never prunes images or visits another execution."""
    from core.cancellation import policy
    from core.io import identifier
    run_id, execution_attempt_id = identifier(run_id), identifier(execution_attempt_id)
    own_client = client is None
    client = client or docker.DockerClient(base_url=cfg['environment']['docker_host'],
                                         timeout=min(3, policy(cfg)['cleanup_seconds']))
    labels = ['dycon.project=' + cfg['project_root'], 'dycon.run=' + run_id,
              'dycon.execution=' + execution_attempt_id]
    removed, errors, remaining = [], [], []
    try:
        for container in client.containers.list(all=True, filters={'label': labels}):
            if (container.labels.get('dycon.project') != cfg['project_root'] or
                    container.labels.get('dycon.run') != run_id or
                    container.labels.get('dycon.execution') != execution_attempt_id):
                raise RuntimeError('Docker returned a container outside the requested execution')
            try:
                container.remove(force=True)
                removed.append(container.id)
            except docker.errors.NotFound:
                removed.append(container.id)
            except Exception as exc:
                errors.append({'container_id': container.id, 'error': repr(exc)})
        remaining = [c.id for c in client.containers.list(all=True, filters={'label': labels})]
    except Exception as exc:
        errors.append({'error': repr(exc)})
    finally:
        if own_client:
            client.close()
    return {'complete': not errors and not remaining, 'removed': removed,
            'remaining': remaining, 'errors': errors}
