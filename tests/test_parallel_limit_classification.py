"""Exercise native branch-limit exceptions through the coordinator and agent."""

from types import SimpleNamespace

import pytest

from rollout.parallel import ParallelReasoningCoordinator, ParallelSelectionError
from rollout.runner import TERMINAL


@pytest.fixture
def native_errors():
    pytest.importorskip("minisweagent")
    from litellm.exceptions import ContextWindowExceededError
    from minisweagent.exceptions import FormatError, LimitsExceeded

    def make(kind, branch):
        if kind == "context":
            return ContextWindowExceededError(f"context limit branch {branch}", "fixture", "openai")
        if kind == "limits":
            return LimitsExceeded({"role": "exit", "content": f"step limit branch {branch}",
                                   "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        if kind == "format":
            return FormatError({"role": "user", "content": f"invalid tools branch {branch}"})
        return TimeoutError(f"transport timeout branch {branch}")

    return make


def coordinator(errors):
    evidence, calls = [], []

    class Client:
        def call(self, action, payload):
            calls.append(action)
            if action == "results":
                return {"results": [], "estimator_sha256": "sha", "target": "target", "endpoint": "submission"}
            return {}

    def record(value, timeout):
        evidence.append(value)
        return value

    cfg = {"parallel": {"branch_count": len(errors), "score_timeout_ms": 1, "fallback_policy": "stop"},
           "generation": {"seed": 42}, "capture": {"control_seconds": 1}}
    value = ParallelReasoningCoordinator(cfg, SimpleNamespace(record_decision=record), Client())

    def generate(messages, metadata, seed):
        raise errors[metadata["branch_id"]]

    def forbidden(*args):
        pytest.fail("Failed API calls cannot be parsed or execute actions")

    args = ([], {"session_id": "session", "step_id": 0, "query_id": "query"}, generate, forbidden, forbidden)
    return value, args, evidence, calls


@pytest.mark.parametrize("kinds,representative", [
    (["context"] * 4, 0),
    (["limits"] * 4, 0),
    (["limits", "limits", "context", "context"], 2),
    (["context", "limits", "context", "limits"], 0),
    (["context"] * 2, 0),
    (["limits"] * 2, 0),
])
def test_all_limit_errors_keep_native_terminal_type_and_evidence(native_errors, kinds, representative):
    errors = [native_errors(kind, b) for b, kind in enumerate(kinds)]
    value, args, evidence, calls = coordinator(errors)
    with pytest.raises(type(errors[representative])) as raised:
        value.run(*args)
    assert raised.value is errors[representative]
    assert type(raised.value).__name__ in TERMINAL
    assert len(evidence) == 1
    assert evidence[0]["selected_branch_id"] is None
    assert evidence[0]["fallback_reason"] is None
    assert [c["reason"] for c in evidence[0]["candidates"]] == [type(error).__name__ for error in errors]
    assert calls == ["reserve", "results", "release"]


@pytest.mark.parametrize("kinds", [
    ["context", "context", "context", "timeout"],
    ["limits", "limits", "limits", "timeout"],
    ["context", "limits", "context", "timeout"],
    ["timeout"] * 4,
])
def test_transport_errors_are_not_mislabeled_as_limit_errors(native_errors, kinds):
    errors = [native_errors(kind, b) for b, kind in enumerate(kinds)]
    value, args, evidence, calls = coordinator(errors)
    with pytest.raises(ParallelSelectionError, match="All parallel API requests failed"):
        value.run(*args)
    assert len(evidence) == 1
    assert evidence[0]["selected_branch_id"] is None
    assert calls[-1] == "release"


def test_format_error_still_reaches_agent_correction_path(native_errors):
    errors = [native_errors("format", b) for b in range(4)]
    value, args, evidence, calls = coordinator(errors)
    with pytest.raises(type(errors[0])) as raised:
        value.run(*args)
    assert raised.value is errors[0]
    assert evidence[0]["fallback_reason"] == "all_branches_format_error"
    assert calls[-1] == "release"


@pytest.mark.parametrize("kind,expected", [("context", "ContextWindowExceededError"), ("limits", "LimitsExceeded")])
def test_native_agent_run_reports_limit_exit_status(native_errors, kind, expected, tmp_path):
    from minisweagent.agents.default import DefaultAgent

    errors = [native_errors(kind, b) for b in range(4)]
    value, args, evidence, calls = coordinator(errors)
    model = SimpleNamespace(
        query=lambda messages: value.run(*args),
        get_template_vars=lambda: {},
        get_cost=lambda: 0,
        serialize=lambda: {},
        format_message=lambda **kwargs: kwargs,
    )
    env = SimpleNamespace(get_template_vars=lambda: {}, serialize=lambda: {})
    agent = DefaultAgent(model, env, system_template="system", instance_template="{{task}}",
                         output_path=tmp_path / "trajectory.json")
    if kind == "context":
        # ContextWindowExceededError reaches the runner through the native
        # uncaught-exception path; LimitsExceeded uses InterruptAgentFlow.
        with pytest.raises(type(errors[0])) as raised:
            agent.run("task")
        assert raised.value is errors[0]
        assert agent.serialize()["info"]["exit_status"] == expected
    else:
        info = agent.run("task")
        assert info["exit_status"] == expected
    assert len(evidence) == 1
    assert calls[-1] == "release"
