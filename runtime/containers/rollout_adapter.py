from __future__ import annotations

from minisweagent.environments.docker import DockerEnvironmentConfig

from runtime.containers.factory import ContainerFactory, TaskContext


class RolloutEnvironment:
    def __init__(self, factory: ContainerFactory, instance: dict, context: TaskContext):
        self.factory, self.context = factory, context
        cfg = factory.cfg
        try:
            spec = factory.spec(instance)
        except ValueError as exc:
            from runtime.containers.dependencies import setup_error
            raise setup_error(cfg, context, str(exc), stage='preflight') from exc
        self.config = DockerEnvironmentConfig(
            image=spec["image_id"],
            cwd=cfg["environment"]["cwd"],
            env=factory.policy_for(instance).variables(),
            timeout=cfg["timeouts"]["tool_command_seconds"],
            container_timeout=str(cfg["timeouts"]["container_lifetime_seconds"]),
            executable=cfg["environment"]["docker_executable"],
            pull_timeout=cfg["timeouts"]["container_start_seconds"],
        )
        self.container = factory.create(instance, context)
        self.container.start()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
        # Keep mini-swe-agent's native submission-marker detection.
        from minisweagent.environments.docker import DockerEnvironment

        result = self.factory.execute(
            self.container, action.get("command", ""), timeout or self.config.timeout, cwd or self.config.cwd
        )
        from core.io import hash_json, now, write_json

        receipt = self.context.output_dir / "first-tool.json"
        if not receipt.exists():
            write_json(
                receipt,
                {
                    "at": now(),
                    "run_id": self.context.run_id,
                    "task_id": self.context.instance_id,
                    "attempt_id": self.context.attempt_id,
                    "action_sha256": hash_json(action),
                    "returncode": result["returncode"],
                    "observation_sha256": hash_json(result["output"]),
                    "container_id": self.container.id,
                },
            )
        DockerEnvironment._check_finished(self, result)
        return result

    def get_template_vars(self, **kwargs):
        import platform

        return {**self.config.model_dump(), **platform.uname()._asdict(), **kwargs}

    def serialize(self):
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": "runtime.containers.RolloutEnvironment",
                }
            }
        }

    def close(self):
        self.factory.close(self.container)
