"""A local fake model endpoint exercises the real agent and LiteLLM adapter."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("minisweagent")


def test_real_agent_request_and_submission_protocol(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from minisweagent.environments.docker import DockerEnvironment

    from baseline.rollout.agent import DurableAgent
    from baseline.rollout.model import make_model
    from core.deadline import Deadline

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(data)
            body = {
                "id": "chat-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "I can submit the fix.",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps(
                                            {
                                                "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    class Environment:
        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

        def execute(self, action):
            assert action["command"] == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
            DockerEnvironment._check_finished(
                self, {"returncode": 0, "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfixture-patch"}
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg["serving"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        agent = DurableAgent(
            make_model(cfg, Deadline()),
            Environment(),
            system_template=cfg["agent"]["system_template"],
            instance_template=cfg["agent"]["instance_template"],
            step_limit=2,
            cost_limit=0,
            output_path=tmp_path / "trace.json",
        )
        result = agent.run("Fix fixture")
        assert result["exit_status"] == "Submitted"
        assert result["submission"] == "fixture-patch"
        assert seen[0]["steering"] is False
        assert seen[0]["min_p"] == 0.0
        assert seen[0]["presence_penalty"] == 0.0
        assert seen[0]["repetition_penalty"] == 1.0
        assert "<pr_description>" in seen[0]["messages"][1]["content"]
        assert "git diff -- path/to/file1 path/to/file2 > patch.txt" in seen[0]["messages"][1]["content"]
        assistant = next(m for m in agent.messages if m["role"] == "assistant")
        assert assistant["extra"]["request"]["model_fingerprint"] == "fixture-model"
        assert assistant["extra"]["response"]["id"] == "chat-fixture"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
