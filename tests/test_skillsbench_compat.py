import json

import pytest

from catalog.skillsbench import package_manifest, validate_payload
from core.backends import capabilities, execution_context
from core.io import hash_json, sha256, write_json
from rollout.adapters.openhands import OpenHandsProtocolError, parse_response, request_identity, request_role
from rollout.adapters.openhands_bridge import buffered_sse
from runtime.agents.openhands import resource_override


def task_payload(tmp_path):
    package = tmp_path / "official-task"
    package.mkdir()
    (package / "task.md").write_text("official task", encoding="utf-8")
    files = package_manifest(package)
    return {"instance_id": "official-task", "task_path": str(package),
            "package_sha256": hash_json(files), "files": files,
            "metadata": {"metadata": {"interface": ["terminal"]}}, "images": []}


def test_skillsbench_openhands_startup_disables_tmux_for_rootless_images():
    from runtime.agents.skillsbench import SKILLSBENCH_AGENT_ENVIRONMENT
    assert SKILLSBENCH_AGENT_ENVIRONMENT["DYCON_OPENHANDS_TERMINAL"] == "subprocess"


def test_skillsbench_payload_keeps_benchmark_identity_without_swe_fields(tmp_path):
    payload = task_payload(tmp_path)
    validate_payload(payload, "official-task", "a" * 40)
    from catalog.store import Task
    task = Task.model_validate({"benchmark": "skillsbench", "revision": "a" * 40,
        "instance_id": "official-task", "partition": "train", "instance": payload})
    row = task.model_dump()
    assert "repo" not in row and "base_commit" not in row
    row["instance"]["package_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        Task.model_validate(row)


def test_backend_matrix_allows_swebench_openhands_but_rejects_action(cfg):
    skills = dict(cfg, benchmark={"backend": "skillsbench"}, agent={"backend": "openhands"})
    assert capabilities(skills)["subject"] == "workspace"
    swe_openhands = {"benchmark": {"backend": "swebench"}, "agent": {"backend": "openhands"}}
    assert capabilities(swe_openhands)["subject"] == "patch"
    with pytest.raises(ValueError, match="action analysis"):
        capabilities(skills, operation="action")


def test_generation_budget_can_be_frozen_per_run_but_seed_remains_root_only():
    from core.cli import main
    with pytest.raises(FileNotFoundError):
        main(["--config", "/missing/config.yaml", "--set", "generation.max_tokens=1024",
              "run", "create", "example", "--operation", "rollout", "--task-id", "task"])
    with pytest.raises(SystemExit):
        main(["--config", "/missing/config.yaml", "--set", "generation.seed=43",
              "run", "create", "example", "--operation", "rollout", "--task-id", "task"])


def test_profile_and_execution_context_bind_snapshot_and_task(cfg, tmp_path):
    from runtime.agents.openhands import OpenHandsProfile, validate_profile
    payload = task_payload(tmp_path)
    from catalog.store import Task
    task = Task.model_validate({"benchmark": "skillsbench", "revision": "a" * 40,
        "instance_id": "official-task", "partition": "train", "instance": payload}).model_dump()
    task_key = hash_json({"benchmark": "skillsbench", "revision": "a" * 40, "instance_id": "official-task"})
    task["instance"]["images"] = []
    tasks = {task_key: task}
    ref = tmp_path / "profile.json"
    profile = OpenHandsProfile(bridge_listen="127.0.0.1", bridge_advertise="127.0.0.1",
        output_paths={"official-task": ["/workspace/result.json"]}).model_dump()
    write_json(ref, profile)
    snapshot = tmp_path / "snapshot.json"
    write_json(snapshot, {"schema_version": 1, "benchmark": "skillsbench", "revision": "a" * 40,
                          "tasks": tasks})
    cfg = {"project_root": str(tmp_path), "benchmark": {"backend": "skillsbench", "snapshot_ref": "snapshot.json"},
           "agent": {"backend": "openhands", "profile_ref": "profile.json"},
           "generation": {"seed": 42}, "parallel": {"enabled": False}}
    context = execution_context(cfg, tasks)
    assert context["profile_sha256"] == sha256(ref)
    assert context["snapshot_sha256"] == sha256(snapshot)
    assert context["context_sha256"] == hash_json({k: v for k, v in context.items() if k != "context_sha256"})
    with pytest.raises(ValueError, match="does not belong"):
        execution_context(cfg, {"wrong": task})
    with pytest.raises(ValueError, match="reachable"):
        validate_profile({**profile, "bridge_advertise": "10.0.0.1"})
    offline = validate_profile({**profile, "network": "none", "cpu_quota": "unenforced_with_receipt"})
    assert offline["cpu_quota"] == "unenforced_with_receipt"
    with pytest.raises(ValueError, match="Host proxy"):
        validate_profile({**offline, "use_host_proxy": True})


def test_openhands_bridges_use_attempt_local_ephemeral_ports(cfg, tmp_path):
    from core.deadline import Deadline
    from rollout.adapters.openhands_bridge import OpenHandsBridge
    profile = {"bridge_listen": "127.0.0.1", "bridge_advertise": "127.0.0.1",
               "request_roles": "unqualified"}
    first = OpenHandsBridge(cfg, {"attempt_id": "a1"}, profile, tmp_path / "a", Deadline(5))
    second = OpenHandsBridge(cfg, {"attempt_id": "a2"}, profile, tmp_path / "b", Deadline(5))
    with first, second:
        assert first.server.server_port != second.server.server_port


def request():
    return {"model": "frozen", "messages": [{"role": "user", "content": "work"}],
            "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}],
            "tool_choice": "auto"}


def response():
    return {"id": "resp-1", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "thinking", "reasoning_content": "reason",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {
            "name": "terminal", "arguments": "{\"command\":\"true\"}"}},
            {"id": "call-2", "type": "function", "function": {
                "name": "terminal", "arguments": "{\"command\":\"echo ok\"}"}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}}


def test_openhands_role_and_request_hash_cover_tools():
    request_a = request()
    request_b = {**request_a, "tools": [{**request_a["tools"][0], "function": {"name": "other"}}]}
    assert request_identity(request_a) != request_identity(request_b)
    assert request_role(request_a, {"request_roles": "unqualified"}) == "unqualified"
    assert request_role(request_a, {"request_roles": "openhands-1.22.1"}) == "agent_decision"
    assert request_role({**request_a, "tools": []}, {"request_roles": "openhands-1.22.1"}) == "unqualified"


def test_multi_tool_candidate_is_complete_and_unknown_finish_rejected():
    parsed = parse_response(response(), request())
    assert len(parsed["tool_calls"]) == 2
    assert parsed["extra"]["request"] == request()
    assert parsed["extra"]["response"]["usage"]["completion_tokens"] == 4
    invalid = json.loads(json.dumps(response()))
    invalid["choices"][0]["finish_reason"] = "unknown-finish"
    with pytest.raises(OpenHandsProtocolError, match="finish reason"):
        parse_response(invalid, request())
    invalid = json.loads(json.dumps(response()))
    invalid["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"] = "{}"
    invalid["choices"][0]["message"]["tool_calls"][1]["id"] = "call-1"
    with pytest.raises(OpenHandsProtocolError, match="duplicate"):
        parse_response(invalid, request())


def test_buffered_stream_has_one_winner_and_final_finish_only():
    data = buffered_sse(response()).decode()
    chunks = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: {")]
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["finish_reason"] is None
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][1]["id"] == "call-2"
    assert data.endswith("data: [DONE]\n\n")


def test_compose_cpu_exception_has_explicit_receipt_and_preserves_memory():
    value = resource_override(1, 4096, cfs_supported=False, policy="unenforced_with_receipt")
    assert value["compose_override"]["environment"] == {"CPUS": "0"}
    assert value["receipt"]["requested_cpu_quota"] == 1
    assert value["receipt"]["memory_limit_preserved"] is True
    with pytest.raises(ValueError, match="explicit"):
        resource_override(1, 4096, cfs_supported=False, policy="strict")
