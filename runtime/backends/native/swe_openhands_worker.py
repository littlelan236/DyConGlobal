"""Worker seam for running ACP inside a native SWE task container."""
from __future__ import annotations


def execute(cfg, context, task, attempt, factory, capture, bridge):
    """Run the pinned ACP driver in the task container.

    Deployments provide the concrete Docker exec transport through the factory.
    Keeping this seam explicit prevents a host-side OpenHands process from
    accidentally editing a different workspace.
    """
    runner = getattr(factory, "openhands_executor", None)
    if runner is not None:
        return runner(cfg, context, task, attempt, capture, bridge)
    import asyncio
    import base64
    import os
    from pathlib import Path
    from urllib.parse import urlsplit

    from runtime.agents.openhands_acp import run_session
    from runtime.agents.swebench_openhands import patch_command
    from core.deadline import Deadline
    from core.io import hash_json
    from runtime.agents.model_socket import UnixModelChannel
    from runtime.backends.native.swe_patch import validate_diff
    from runtime.containers.factory import TaskContext
    from runtime.containers.rollout_adapter import RolloutEnvironment

    output = Path(attempt.output)
    task_context = TaskContext(context.run_id, task["instance_id"], context.context["attempt_id"]
                               if hasattr(context, "context") else attempt.context["attempt_id"],
                               "swe-openhands", output)
    profile = cfg["execution_context"]["profile"]
    channel_dir = (Path(cfg["project_root"]) / "temp/model-channels" /
                   hash_json({"binding": attempt.context, "output": str(output)})[:16])
    socket_path = channel_dir / "model.sock"
    endpoint = urlsplit(bridge.base_url)
    with UnixModelChannel(socket_path, endpoint.hostname, endpoint.port):
        env = RolloutEnvironment(factory, task["instance"], task_context, socket_path=socket_path)
        try:
            script = Path(__file__).parents[2] / "agents" / "model_socket.py"
            encoded = base64.b64encode(script.read_bytes()).decode()
            bootstrap = factory.execute(
                env.container,
                "set -e; id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; "
                "mkdir -p /home/agent /run/dycon-model; chown -R agent:agent /home/agent; "
                "echo " + encoded +
                " | base64 -d > /tmp/dycon-model-socket.py && chmod 755 /tmp/dycon-model-socket.py",
                cfg["timeouts"]["bootstrap_seconds"],
            )
            if bootstrap["returncode"]:
                raise RuntimeError("SWE OpenHands agent bootstrap failed: " + bootstrap["output"][-2000:])
            from runtime.agents.dependencies import install_command
            install = factory.execute(env.container, install_command("/opt/dycon-agent-wheels"),
                                      cfg["timeouts"]["bootstrap_seconds"])
            if install["returncode"]:
                raise RuntimeError("SWE OpenHands runtime install failed: " + install["output"][-3000:])
            from runtime.agents.openhands import driver_environment
            agent_env = driver_environment(cfg, bridge.base_url, bridge.secret,
                                           Deadline(cfg["timeouts"]["model_seconds"]))
            agent_env.update({
                "HOME": "/home/agent",
                "LLM_API_KEY": bridge.secret,
                "DYCON_AGENT_CWD": profile["task_cwd"],
                "PATH": "/opt/dycon-openhands/bin:/usr/local/bin:/usr/bin:/bin",
            })
            docker_exec = cfg["environment"]["docker_executable"]
            args = [docker_exec, "exec", "--interactive", "--user", profile["sandbox_user"],
                    "--workdir", profile["task_cwd"]]
            for key, value in agent_env.items():
                args.extend(["--env", key + "=" + value])
            args.extend([env.container.id, "/opt/dycon-openhands/bin/python", "/tmp/dycon-model-socket.py"])
            instruction = task["instance"]["problem_statement"]
            result = asyncio.run(run_session(docker_exec, args[1:], os.environ.copy(),
                                             profile["task_cwd"], instruction,
                                             seconds=cfg["timeouts"]["model_seconds"], record={}))
            patch_result = factory.execute(env.container, patch_command(profile),
                                           cfg["timeouts"]["tool_command_seconds"], profile["task_cwd"])
            if patch_result["returncode"]:
                raise RuntimeError("SWE workspace patch export failed: " + patch_result["output"][-2000:])
            validate_diff(patch_result["output"], max_bytes=profile["patch_max_bytes"])
            return {"exit_status": "AgentCompleted", "patch": patch_result["output"],
                    "records": result.get("records", []), "events": result.get("events", []),
                    "cleanup": {"complete": True}}
        finally:
            env.close()

