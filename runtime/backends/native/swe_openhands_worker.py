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

    from runtime.agents.openhands_acp import run_session
    from runtime.agents.swebench_openhands import patch_command
    from runtime.backends.native.swe_patch import validate_diff
    from runtime.containers.factory import TaskContext
    from runtime.containers.rollout_adapter import RolloutEnvironment

    output = Path(attempt.output)
    task_context = TaskContext(context.run_id, task["instance_id"], context.context["attempt_id"]
                               if hasattr(context, "context") else attempt.context["attempt_id"],
                               "swe-openhands", output)
    env = RolloutEnvironment(factory, task["instance"], task_context)
    try:
        profile = cfg["execution_context"]["profile"]
        script = Path(__file__).parents[2] / "agents" / "model_socket.py"
        encoded = base64.b64encode(script.read_bytes()).decode()
        factory.execute(env.container, "mkdir -p /run/dycon-model && echo " + encoded +
                        " | base64 -d > /tmp/dycon-model-socket.py && chmod 755 /tmp/dycon-model-socket.py",
                        cfg["timeouts"]["bootstrap_seconds"])
        agent_env = {
            "LLM_MODEL": "openai/" + str(cfg["model"]["served_name"]),
            "LLM_API_KEY": bridge.secret,
            "DYCON_AGENT_CWD": profile["task_cwd"],
            "PATH": "/opt/dycon-openhands/bin:/usr/local/bin:/usr/bin:/bin",
        }
        docker_exec = cfg["environment"]["docker_executable"]
        args = [docker_exec, "exec", "--interactive", "--user", profile["sandbox_user"],
                "--workdir", profile["task_cwd"]]
        for key, value in agent_env.items():
            args.extend(["--env", key + "=" + value])
        args.extend([env.container.id, "/opt/dycon-openhands/bin/python", "/tmp/dycon-model-socket.py"])
        instruction = task["instance"]["problem_statement"]
        result = asyncio.run(run_session(docker_exec, args[1:], os.environ.copy(), None, instruction,
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
