"""Harbor BaseAgent extension for the fixed DyCon OpenHands ACP driver."""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

from harbor.agents.base import BaseAgent

from core.io import write_json
from runtime.agents.dependencies import install_runtime, probe_runtime
from runtime.agents.openhands_acp import run_session


class DyConOpenHandsACP(BaseAgent):
    def __init__(self, *args, dycon, **kwargs):
        super().__init__(*args, **kwargs)
        self.dycon = dycon

    @staticmethod
    def name():
        return "dycon-openhands-acp"

    def version(self):
        return "1.22.1"

    async def setup(self, environment):
        root = Path(self.dycon["output"])
        cfg, owner = self.dycon["cfg"], self.dycon["ownership"]
        from runtime.backends.harbor.policy import execution_policy
        self.policy = execution_policy(cfg)
        await install_runtime(environment, cfg, root, owner)
        if self.policy:
            await self._setup_terminal(environment, cfg, root, owner)
            return
        result = await environment.exec(
            "set -e; id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; "
            "mkdir -p /home/agent; chown agent:agent /home/agent; "
            "chmod o+x /opt/dycon-openhands; "
            "test -x /opt/dycon-wheels/python/bin/python3.12",
            user="root", timeout_sec=30)
        if result.return_code:
            raise ValueError("Could not create the non-root OpenHands user: " +
                             str(result.stdout) + str(getattr(result, "stderr", "")))
        if cfg["benchmark"]["backend"] == "swebench_pro_v2":
            result = await environment.exec("git -C /app rev-parse --show-toplevel 2>/dev/null || "
                                            "git -C /testbed rev-parse --show-toplevel", user="root", timeout_sec=20)
        else:
            result = await environment.exec("pwd", user="root", timeout_sec=20)
        self.cwd = result.stdout.strip()
        if result.return_code or not self.cwd.startswith("/") or self.cwd in {"/", "/opt", "/usr", "/home"}:
            raise ValueError("Could not resolve the official task working directory")
        # Agent access is limited to its actual workspace, home and agent logs.
        result = await environment.exec("chown -R agent:agent " + shlex.quote(self.cwd) +
            " /home/agent /logs/agent && chmod 700 /logs/verifier; "
            "test ! -e /tests && test ! -e /solution", user="root", timeout_sec=120)
        if result.return_code:
            raise ValueError("Agent workspace/evaluator separation failed")
        from runtime.agents import model_socket
        await environment.upload_file(Path(model_socket.__file__), "/opt/dycon-model-socket.py")
        await probe_runtime(environment, cfg, root, owner, self.cwd)
        if self.dycon.get("probe") and cfg["benchmark"]["backend"] == "swebench_pro_v2":
            probe = await environment.exec(
                "python3 - <<'PY'\n"
                "import json,socket\n"
                "checks={}\n"
                "for label,host,port in [('public_internet','1.1.1.1',443),('host_ssh',%r,22),('host_proxy',%r,17890),('external_dns_tcp','8.8.8.8',53)]:\n"
                " try:\n  s=socket.create_connection((host,port),1);s.close();checks[label+'_denied']=False\n"
                " except OSError: checks[label+'_denied']=True\n"
                "import subprocess\n"
                "checks['external_dns_resolution_denied']=subprocess.run(['timeout','2','getent','hosts','example.com']).returncode != 0\n"
                "print(json.dumps(checks)); assert all(checks.values()) and len(checks)==5\n"
                "PY" % (self.dycon["probe_host"], self.dycon["probe_host"]),
                user="agent", timeout_sec=30)
            write_json(root / "network-probe.json", {"stdout": probe.stdout, "stderr": getattr(probe, "stderr", ""),
                       "return_code": probe.return_code, "verified": probe.return_code == 0})
            if probe.return_code:
                raise ValueError("Harbor offline agent network policy probe failed")

    async def _setup_terminal(self, environment, cfg, root, owner):
        identity = await environment.terminal_identity()
        self.cwd, self.task_path = identity["cwd"], identity["path"]
        if not self.cwd.startswith("/") or not self.task_path:
            raise ValueError("Invalid official Terminal workspace/PATH")
        separated = await environment.exec("set -e; mkdir -p /root; chmod 700 /logs/verifier; test ! -e /tests && test ! -e /solution",
                                           user="root", timeout_sec=20)
        if separated.return_code:
            raise ValueError("Terminal evaluator separation failed")
        from runtime.agents import model_socket
        await environment.upload_file(Path(model_socket.__file__), "/opt/dycon-model-socket.py")
        await probe_runtime(environment, cfg, root, owner, self.cwd,
                            expected_user=self.policy["user"], expected_home=self.policy["home"], task_path=self.task_path)
        write_json(root / "terminal-identity.json", {"policy": self.policy, "cwd": self.cwd,
                   "path": self.task_path, "workspace_permissions_preserved": True})

    async def run(self, instruction, environment, context):
        import os
        cfg = self.dycon["cfg"]
        env = {**self.dycon["agent_environment"], "HOME": "/home/agent",
               "LLM_API_KEY": os.environ["DYCON_MODEL_CHANNEL_SECRET"], "DYCON_AGENT_CWD": self.cwd,
               "PATH": "/opt/dycon-openhands/bin:/usr/local/bin:/usr/bin:/bin"}
        if self.dycon.get("probe") and cfg["benchmark"]["backend"] == "swebench_pro_v2":
            env.update(DYCON_PROBE_NETWORK="1", DYCON_PROBE_HOST=self.dycon["probe_host"])
        if self.policy:
            env.update(HOME=self.policy["home"], PATH=self.task_path,
                       DYCON_PROCESS_RECEIPT="/tmp/dycon-acp-processes.json")
            if self.dycon.get("probe"):
                env["DYCON_PROBE_CHANNEL"] = "1"
        command, args, process_env = await environment.acp_command(env)
        result = {}
        try:
            await run_session(command, args, process_env, None, instruction,
                seconds=self.dycon["agent_seconds"], probe=self.dycon.get("probe", False), record=result,
                session_cwd=self.cwd if self.policy else None)
        finally:
            write_json(Path(self.dycon["output"]) / "openhands/acp-events.json", result)
            primary = sys.exception()
            try:
                await self._finish_agent(environment, cfg)
            except Exception as cleanup_exc:
                write_json(Path(self.dycon["output"]) / "acp-cleanup-error.json",
                           {"error_type": type(cleanup_exc).__name__, "error": str(cleanup_exc)[-1000:]})
                if primary is None:
                    raise

    async def _finish_agent(self, environment, cfg):
        # Terminal services are part of live scoring; only stop owned ACP/relay.
        # Legacy sandboxes retain their user-scoped cleanup before Pro capture.
        if self.policy:
            stopped = await environment.exec(
                "/opt/dycon-openhands/bin/python /opt/dycon-model-socket.py --stop /tmp/dycon-acp-processes.json",
                user="root", timeout_sec=30)
            write_json(Path(self.dycon["output"]) / "acp-cleanup.json",
                       {"return_code": stopped.return_code, "stdout": stopped.stdout})
            if stopped.return_code:
                raise ValueError("Terminal ACP/relay cleanup was not confirmed")
            if self.dycon.get("probe"):
                await environment.download_file("/root/dycon-model-channel-probe.json",
                    Path(self.dycon["output"]) / "model-channel-probe.json")
        else:
            await environment.exec("pkill -TERM -u agent 2>/dev/null || true", user="root", timeout_sec=10)
        if cfg["benchmark"]["backend"] == "swebench_pro_v2" and not self.dycon.get("probe"):
            capture = await environment.exec("cd " + shlex.quote(self.cwd) +
                " && git add -A && git diff --cached --binary > /tmp/dycon-model.patch && git reset -q",
                user="root", timeout_sec=60)
            if capture.return_code:
                raise ValueError("Pro model.patch capture failed")
            await environment.download_file("/tmp/dycon-model.patch", Path(self.dycon["output"]) / "model.patch")
