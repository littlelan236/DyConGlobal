"""Canonical observations survive a cancelled next-generation request."""
import copy

import pytest

from rollout.adapters.openhands import canonical_trajectory


def decision(call_id="edit", prefix=None):
    request = {"messages": prefix or [{"role": "user", "content": "Fix ars.R"}],
               "tools": [{"type": "function", "function": {"name": "file_editor"}}]}
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {
            "name": "file_editor", "arguments": '{"path":"/app/ars.R"}'}}]}
    return {"role": "agent_decision", "request": request, "request_id": call_id,
            "selected_response": {"choices": [{"finish_reason": "tool_calls", "message": message}]}}


def feedback(call_id="edit", content=None):
    return {"role": "tool", "tool_call_id": call_id,
            "content": content if content is not None else [{"type": "text", "text": "File edited."}]}


def followup(previous, *observations, status="cancelled"):
    prefix = previous["request"]["messages"] + [
        previous["selected_response"]["choices"][0]["message"], *observations]
    return {"role": "agent_decision", "request_id": "next", "request": {"messages": prefix},
            "status": status}


def project(records, events=None):
    return canonical_trajectory("task", {"attempt_id": "attempt"}, records, events or [], "OpenHandsError")


@pytest.mark.parametrize("status", ["cancelled", "failed", "running"])
def test_trailing_observation_does_not_require_a_selected_next_response(status):
    first = decision()
    observation = feedback(content=[{"type": "text", "text": "Line 12 changed.\n"}])
    pending = followup(first, observation, status=status)
    records = [first, pending]
    original = copy.deepcopy(records)
    events = [{"session_id": "s", "update": {"sessionUpdate": "tool_call_update"}}]
    result = project(records, events)
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool"]
    assert result["messages"][-1] == observation
    assert result["info"]["model_stats"]["api_calls"] == 1
    assert result["raw_acp_events"] == events
    assert records == original
    result["messages"][-1]["content"][0]["text"] = "changed"
    assert records == original


def test_retried_prefix_does_not_duplicate_completed_tool_result():
    first = decision()
    pending = followup(first, feedback())
    second = decision("next-edit", pending["request"]["messages"])
    result = project([first, pending, copy.deepcopy(pending), second])
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert result["messages"][2] == feedback()
    assert result["info"]["model_stats"]["api_calls"] == 2


def test_only_selected_call_results_are_projected_in_prefix_order():
    first = decision()
    first["selected_response"]["choices"][0]["message"]["tool_calls"].append(
        decision("edit-2")["selected_response"]["choices"][0]["message"]["tool_calls"][0])
    pending = followup(first, feedback("edit-2"), feedback("unselected"), feedback())
    result = project([first, pending])
    assert [m["tool_call_id"] for m in result["messages"] if m["role"] == "tool"] == ["edit-2", "edit"]


def test_effective_prefix_is_used_and_auxiliary_requests_are_excluded():
    first = decision()
    pending = followup(first, feedback(content="actual output"))
    pending["effective_request"] = copy.deepcopy(pending["request"])
    pending["request"]["messages"][-1] = feedback(content="different original prefix")
    auxiliary = followup(first, feedback(content="condenser text"))
    auxiliary["role"] = "condensation"
    result = project([first, auxiliary, pending])
    assert result["messages"][-1] == feedback(content="actual output")


def test_no_completion_is_fabricated_from_tool_call_or_acp_progress():
    first = decision()
    events = [{"session_id": "s", "update": {"sessionUpdate": "tool_call",
               "toolCallId": "edit", "status": "in_progress"}}]
    assert [m["role"] for m in project([first], events)["messages"]] == ["user", "assistant"]


def test_completed_next_response_preserves_existing_message_content():
    first = decision()
    second = decision("edit-2", followup(first, feedback())["request"]["messages"])
    result = project([first, second])
    assert result["messages"][2] == feedback()
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool", "assistant"]
