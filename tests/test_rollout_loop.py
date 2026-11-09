"""Exercise loop limits through the native agent flow without GPUs or Docker."""

# Optional runner imports follow importorskip, as in the existing rollout tests.
# ruff: noqa: E402

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

pytest.importorskip("minisweagent")

from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

from core.config import Agent, load_snapshot, resolve, snapshot
from core.deadline import Deadline
from core.io import hash_json, read_json
from rollout.agent import DurableAgent, DurableAgentConfig, _step_fingerprint


def message(label="A", commands=None):
    commands = commands if commands is not None else [f"echo {label}"]
    return {
        "role": "assistant", "reasoning_content": f"reason {label}", "content": f"thought {label}",
        "tool_calls": [{"id": f"call-{i}", "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": command})}}
                       for i, command in enumerate(commands)],
        "extra": {"actions": [{"command": command, "tool_call_id": f"call-{i}"}
                              for i, command in enumerate(commands)]},
    }


def submitted():
    return Submitted({"role": "exit", "content": "done",
                      "extra": {"exit_status": "Submitted", "submission": "patch"}})


class ScriptedModel:
    def __init__(self, messages):
        self.script = iter(messages)
        self.calls = 0
        self.phases = []

    def query(self, messages):
        self.calls += 1
        item = next(self.script, None)
        if item is None:
            raise submitted()
        if isinstance(item, Exception):
            raise item
        item = copy.deepcopy(item)
        # Real servers assign new identities for semantically identical output.
        for i, action in enumerate(item["extra"]["actions"]):
            action["tool_call_id"] = f"call-{self.calls}-{i}"
            item["tool_calls"][i]["id"] = action["tool_call_id"]
        item["extra"].update(timestamp=self.calls, cost=0.01, request={"query_id": str(self.calls)})
        return item

    def record_action_phase(self, message, phase, details=None):
        self.phases.append((message["extra"]["request"]["query_id"], phase))

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars):
        return format_toolcall_observation_messages(
            actions=message["extra"]["actions"], outputs=outputs,
            observation_template="{{output.output}} {{output.exception_info | default('')}}",
            template_vars=template_vars,
        )

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}


class Environment:
    def __init__(self, *args):
        self.commands = []
        self.closed = False

    def execute(self, action):
        self.commands.append(action["command"])
        if action["command"] == "submit":
            raise submitted()
        return {"returncode": 0, "output": f"observation-{len(self.commands)}"}

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {}

    def close(self):
        self.closed = True


def make_agent(messages, tmp_path, **kwargs):
    model, env = ScriptedModel(messages), Environment()
    agent = DurableAgent(model, env, system_template="system", instance_template="{{task}}",
                         cost_limit=0, loop_repeat_limit=8, output_path=tmp_path / "raw.traj.json", **kwargs)
    return agent, model, env


@pytest.mark.parametrize("period", range(1, 8))
@pytest.mark.parametrize("prefix", [0, 3])
def test_exact_cycles_stop_on_eighth_complete_period(period, prefix, tmp_path):
    cycle = [message(str(i)) for i in range(period)]
    agent, model, env = make_agent([message(f"prefix-{i}") for i in range(prefix)] + cycle * 9, tmp_path)
    result = agent.run("task")
    assert result["exit_status"] == "LimitsExceeded"
    assert result["submission"] == ""
    assert result["limit_reason"] == "repeated_reasoning_action_loop"
    assert model.calls == prefix + 8 * period
    assert len(env.commands) == model.calls - 1
    assert agent.cost == pytest.approx(0.01 * model.calls)
    saved = read_json(tmp_path / "raw.traj.json")
    evidence = saved["info"]["loop_detection"]
    assert evidence["period"] == period
    assert evidence["repeats"] == 8
    assert evidence["start_step"] == prefix
    assert evidence["end_step"] == model.calls - 1
    assert evidence["action_executed"] is False
    assert saved["info"]["exit_status"] == "LimitsExceeded"
    assert saved["messages"][-2]["extra"]["returncode"] == -1
    assert saved["messages"][-2]["extra"]["exception_info"] == "action was not executed"
    assert saved["messages"][-3]["extra"]["action_executed"] is False
    assert len(agent._loop_history) <= 56


@pytest.mark.parametrize("period", range(1, 8))
def test_seven_cycles_do_not_stop(period, tmp_path):
    agent, _, env = make_agent([message(str(i)) for i in range(period)] * 7, tmp_path)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert len(env.commands) == period * 7
    assert "loop_detection" not in agent.serialize()["info"]


@pytest.mark.parametrize("change", ["reasoning", "content", "command", "whitespace", "order"])
def test_changed_output_breaks_period(change, tmp_path):
    original = message(commands=["echo A", "echo B"])
    changed = copy.deepcopy(original)
    if change == "reasoning":
        changed["reasoning_content"] += "!"
    elif change == "content":
        changed["content"] += "!"
    elif change == "order":
        changed["extra"]["actions"].reverse()
    else:
        changed["extra"]["actions"][0]["command"] += " " if change == "whitespace" else "!"
    agent, _, env = make_agent([original] * 7 + [changed] + [original] * 7, tmp_path)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert len(env.commands) == 30


def test_period_eight_is_outside_limit(tmp_path):
    agent, _, env = make_agent([message(str(i)) for i in range(8)] * 9, tmp_path)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert len(env.commands) == 72


def test_multi_action_winner_counts_as_one_step(tmp_path):
    winner = message(commands=["echo A", "echo B"])
    winner["extra"]["dycon_parallel"] = {"selected_branch_id": 2, "branch_count": 4}
    agent, model, env = make_agent([winner] * 8, tmp_path)
    assert agent.run("task")["exit_status"] == "LimitsExceeded"
    assert model.calls == 8
    assert len(env.commands) == 14
    assert len(model.phases) == 14
    assert not any(call == "8" for call, _ in model.phases)
    skipped = [m for m in agent.messages if m["role"] == "tool" and m["extra"]["returncode"] == -1]
    assert len(skipped) == 2
    assert {m["tool_call_id"] for m in skipped} == {"call-8-0", "call-8-1"}


def test_reasoning_fallback_and_transient_metadata():
    original = message()
    altered = copy.deepcopy(original)
    reasoning = altered.pop("reasoning_content")
    altered["extra"]["response"] = {"id": "another-id", "choices": [{"message": {
        "reasoning_content": reasoning}}]}
    altered["extra"]["actions"][0]["tool_call_id"] = "another-call"
    altered["extra"]["dycon_parallel"] = {"selected_branch_id": 3}
    assert _step_fingerprint(original) == _step_fingerprint(altered)
    altered["extra"]["response"]["choices"][0]["message"]["reasoning_content"] += "!"
    assert _step_fingerprint(original) != _step_fingerprint(altered)


def test_format_error_interrupts_consecutive_cycles(tmp_path):
    error = FormatError({"role": "user", "content": "fix format", "extra": {"interrupt_type": "FormatError"}})
    agent, _, env = make_agent([message()] * 7 + [error] + [message()] * 7, tmp_path)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert len(env.commands) == 14
    repeated, _, _ = make_agent([error] * 3, tmp_path, max_consecutive_format_errors=3)
    assert repeated.run("task")["exit_status"] == "RepeatedFormatError"


def test_state_is_reset_when_agent_is_reused(tmp_path):
    agent, model, _ = make_agent([message()] * 8, tmp_path)
    assert agent.run("first")["exit_status"] == "LimitsExceeded"
    model.script = iter([message()] * 7)
    assert agent.run("second")["exit_status"] == "Submitted"
    assert "loop_detection" not in agent.serialize()["info"]


@pytest.mark.parametrize("options,expected,executions", [
    ({"loop_repeat_limit": None}, "Submitted", 9),
    ({"step_limit": 3}, "LimitsExceeded", 3),
    ({"cost_limit": 0.025}, "LimitsExceeded", 3),
    ({"loop_repeat_limit": 3, "loop_max_period": 1}, "LimitsExceeded", 2),
])
def test_other_limits_and_disabled_detection(options, expected, executions, tmp_path):
    model, env = ScriptedModel([message()] * 9), Environment()
    settings = {"loop_repeat_limit": 8, "cost_limit": 0, **options}
    agent = DurableAgent(model, env, system_template="s", instance_template="{{task}}", **settings)
    result = agent.run("task")
    assert result["exit_status"] == expected
    assert len(env.commands) == executions


def test_normal_action_can_submit(tmp_path):
    agent, _, env = make_agent([message(commands=["submit"])], tmp_path)
    assert agent.run("task") == {"exit_status": "Submitted", "submission": "patch"}
    assert env.commands == ["submit"]


@pytest.mark.parametrize("field,value", [
    ("loop_repeat_limit", 1), ("loop_repeat_limit", 0), ("loop_repeat_limit", True),
    ("loop_repeat_limit", "8"), ("loop_max_period", 0), ("loop_max_period", 8),
    ("loop_max_period", 1.5), ("loop_max_period", True),
])
def test_loop_config_rejects_invalid_values(cfg, field, value):
    with pytest.raises(ValidationError):
        Agent.model_validate({**cfg["agent"], field: value})
    with pytest.raises(ValidationError):
        DurableAgentConfig(system_template="s", instance_template="t", **{field: value})


def test_legacy_yaml_identity_and_snapshot_remain_unchanged(cfg):
    path = Path(cfg["config_source"])
    raw = yaml.safe_load(path.read_text())
    for field in ("loop_repeat_limit", "loop_max_period"):
        raw["agent"].pop(field, None)
    path.write_text(yaml.safe_dump(raw))
    legacy = resolve(path)
    assert "loop_repeat_limit" not in legacy["agent"]
    assert "loop_max_period" not in legacy["agent"]
    assert legacy["config_hash"] == hash_json({k: v for k, v in legacy.items() if k != "config_hash"})
    run = snapshot(legacy, "old-loop-snapshot", {"task_keys": []})
    before = (run / "resolved-config.json").read_bytes()
    raw["agent"].update(loop_repeat_limit=8, loop_max_period=7)
    path.write_text(yaml.safe_dump(raw))
    assert resolve(path)["agent"]["loop_repeat_limit"] == 8
    restored = load_snapshot(cfg["project_root"], "old-loop-snapshot")
    assert restored == read_json(run / "resolved-config.json")
    assert "loop_repeat_limit" not in restored["agent"]
    assert (run / "resolved-config.json").read_bytes() == before


@pytest.mark.parametrize("capture,fail_capture", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("terminal_status", ["LimitsExceeded", "reasoning_loop"])
def test_runner_persists_loop_and_resumes_without_retry(cfg, monkeypatch, capture, fail_capture, terminal_status):
    from conftest import task

    from catalog.store import Catalog
    from hidden.backends.vllm import online
    from rollout import model
    from rollout.runner import run_task
    from runtime.containers import rollout_adapter

    pytest.importorskip("docker")
    cfg["execution"].pop("limits_retry", None)  # Explicit legacy no-resampling contract.
    cfg["agent"].update(loop_repeat_limit=8, loop_max_period=7)
    cfg["capture"].update(enabled=capture, failure_policy="block_rollout")
    cfg["execution"]["infrastructure_retry_limit"] = 2
    catalog = Catalog(cfg["project_root"])
    catalog.register_tasks([task("loop-task")])
    task_key, row = next(iter(catalog.read()["tasks"].items()))
    scripted = ScriptedModel([message()] * 8)
    if terminal_status == "reasoning_loop":
        from minisweagent.exceptions import LimitsExceeded

        def all_branches_loop(*args, **kwargs):
            scripted.calls += 1
            raise LimitsExceeded({"role": "exit", "content": "All branches looped",
                                  "extra": {"exit_status": "reasoning_loop", "submission": "",
                                            "limit_reason": "all_parallel_branches_reasoning_loop"}})

        scripted.query = all_branches_loop
    env = Environment()
    calls = []

    class Capture:
        def __init__(self, *args, **kwargs):
            pass

        def _request(self, *args, **kwargs):
            return {}  # The capture flush now uses the shared control client.

        def start(self, *args, **kwargs):
            calls.append("start")

        def stop(self, *args):
            calls.append("stop")

        def prepare(self, session_id, path):
            calls.append("prepare")
            assert read_json(path)["info"]["exit_status"] == terminal_status

        def finalize(self, *args, **kwargs):
            calls.append("finalize")
            assert Path(kwargs["trajectory_path"]).exists()
            if fail_capture:
                raise RuntimeError("capture export failed")
            return {"capture_status": "complete"}

    monkeypatch.setattr(model, "make_model", lambda *args: scripted)
    monkeypatch.setattr(rollout_adapter, "RolloutEnvironment", lambda *args: env)
    monkeypatch.setattr(online, "CaptureClient", Capture)
    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {}))
    runtime = {"capture_token_inventory": {"newline_token_ids": [1], "close_token_ids": [2],
               "tokenizer_fingerprint": "fixture", "capture_select": {}}, "capture_control_url": "fixture"}
    factory = SimpleNamespace(deadline=Deadline())
    result = run_task(cfg, "loop-run", task_key, row, factory, runtime)
    assert result["agent_exit_status"] == terminal_status
    assert result["exit_status"] == ("CaptureError" if fail_capture else terminal_status)
    assert env.closed
    assert len(env.commands) == (7 if terminal_status == "LimitsExceeded" else 0)
    assert calls == (["start", "stop", "prepare", "finalize"] if capture else [])
    from application.execution.evidence import RunEvidence
    saved_row = RunEvidence(cfg["project_root"], "loop-run").read()["rollouts"][result["rollout_id"]]
    assert saved_row["termination_reason"] == terminal_status
    assert saved_row["submission_status"] == "unsubmitted"
    assert saved_row["resolved"] is None
    trajectory = read_json(Path(cfg["project_root"]) / saved_row["trajectory_path"])
    if terminal_status == "LimitsExceeded":
        assert trajectory["info"]["loop_detection"]["period"] == 1
    else:
        assert trajectory["info"]["exit_status"] == "reasoning_loop"
        assert trajectory["messages"][-1]["extra"]["limit_reason"] == "all_parallel_branches_reasoning_loop"
    work = Path(cfg["project_root"]) / "runs/loop-run/rollout/tasks/loop-task/repetitions/rep-000"
    assert len(list((work / "attempts").iterdir())) == 1
    assert read_json(work / "result.json") == result
    assert RunEvidence(cfg["project_root"], "loop-run").verify()["rollouts"] == 1
    if not fail_capture:
        assert run_task(cfg, "loop-run", task_key, row, factory, runtime) == result
        assert scripted.calls == (8 if terminal_status == "LimitsExceeded" else 1)


def test_legacy_agent_settings_disable_loop_detection(tmp_path):
    model, env = ScriptedModel([message()] * 9), Environment()
    agent = DurableAgent(model, env, system_template="s", instance_template="t", cost_limit=0)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert len(env.commands) == 9
