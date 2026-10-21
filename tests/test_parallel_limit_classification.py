"""Exercise native branch-limit exceptions through the coordinator and agent."""

from types import SimpleNamespace

import pytest

from rollout.parallel import ParallelReasoningCoordinator, ParallelSelectionError
from rollout.runner import TERMINAL


@pytest.mark.parametrize("other,expected", [("normal", "winner"), ("loop", "all_parallel_branches_reasoning_loop"),
    ("context", "ContextWindowExceededError"), ("format", "format"), ("timeout", "ParallelSelectionError")])
@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("client_finish", ["abort", "stop"])
def test_loop_branch_is_excluded_from_all_fallbacks(native_errors, other, expected, diagnostic, client_finish):
    from minisweagent.exceptions import FormatError, LimitsExceeded

    evidence, terminations, parses = [], [], []

    class Client:
        def call(self, action, payload):
            if action == "results":
                return {"results": terminations, "estimator_sha256": "sha", "target": "target", "endpoint": "submission"}
            return {}

    class Response:
        def __init__(self, finish, branch):
            self.choices = [SimpleNamespace(finish_reason=finish)]
            self.branch = branch

        def model_dump(self, **kwargs):
            return {"choices": [{"finish_reason": self.choices[0].finish_reason}], "usage": {"completion_tokens": 2048}}

    cfg = {"parallel": {"branch_count": 2, "score_timeout_ms": 1, "fallback_policy": "first_valid"},
           "generation": {"seed": 42}, "capture": {"control_seconds": 1}}

    def record(value, timeout):
        evidence.append(value)
        return value

    coordinator = ParallelReasoningCoordinator(cfg, SimpleNamespace(record_decision=record), Client(),
        diagnostic_mode="first_valid_unscored" if diagnostic else None)

    def generate(messages, md, seed):
        branch = md["branch_id"]
        if branch == 0 or other == "loop":
            loop = {**{k: md[k] for k in ("session_id", "decision_id", "branch_id", "api_attempt_id")},
                    "native_request_id": f"native-{branch}", "abort_completed": True}
            terminations.append({"branch_id": branch, "api_attempt_id": md["api_attempt_id"],
                "native_request_id": f"native-{branch}", "reason": "reasoning_loop", "status": "failed",
                "reasoning_loop": loop})
            return Response(client_finish, branch)
        if other != "normal":
            raise native_errors(other, branch)
        return Response("tool_calls", branch)

    def parse(response):
        parses.append(response.branch)
        return {"role": "assistant", "extra": {"actions": [{"command": "good"}]}}

    args = ([], {"session_id": "s", "step_id": 0, "query_id": "q"}, generate, parse, lambda response: {"cost": 0})
    if expected == "winner":
        assert coordinator.run(*args)["extra"]["actions"] == [{"command": "good"}]
        assert evidence[0]["selected_branch_id"] == 1
    elif expected == "format":
        with pytest.raises(FormatError):
            coordinator.run(*args)
    elif expected == "all_parallel_branches_reasoning_loop":
        with pytest.raises(LimitsExceeded) as raised:
            coordinator.run(*args)
        assert raised.value.messages[0]["extra"]["limit_reason"] == expected
    else:
        with pytest.raises(Exception) as raised:
            coordinator.run(*args)
        assert type(raised.value).__name__ == expected
    assert 0 not in parses
    assert evidence[0]["candidates"][0]["reason"] == "reasoning_loop"
    assert evidence[0]["candidates"][0]["status"] == "ineligible"


def test_unproven_native_abort_is_not_mislabeled_as_loop():
    pytest.importorskip("minisweagent")
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="abort")],
                               model_dump=lambda **kwargs: {"choices": [{"finish_reason": "abort"}]})
    records = []

    class Client:
        def call(self, action, payload):
            return {"results": [], "estimator_sha256": "sha", "target": "t", "endpoint": "submission"}

    def record(value, timeout):
        records.append(value)
        return value

    cfg = {"parallel": {"branch_count": 2, "score_timeout_ms": 1, "fallback_policy": "first_valid"},
           "generation": {"seed": 42}, "capture": {"control_seconds": 1}}
    coordinator = ParallelReasoningCoordinator(cfg, SimpleNamespace(record_decision=record), Client())
    with pytest.raises(ParallelSelectionError):
        coordinator.run([], {"session_id": "s", "step_id": 0, "query_id": "q"},
                        lambda *args: response, lambda response: pytest.fail("Abort cannot be parsed"),
                        lambda response: {"cost": 0})
    assert all(row["reason"] == "ParallelSelectionError" for row in records[0]["candidates"])


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
