import json
from pathlib import Path

import pytest

pytest.importorskip("minisweagent")

from core.io import read_json, sha256
from runtime.agents.webarena import WebArenaAgent, finalize_subject


class FakeModel:
    def get_template_vars(self):
        return {}

    def format_observation_messages(self, message, outputs, template_vars):
        return [{"role": "tool", "content": json.dumps(outputs)}]

    def format_message(self, **kwargs):
        return kwargs

    def serialize(self):
        return {}


class FakeEnv:
    def __init__(self, out):
        self.out = Path(out)

    def execute(self, action):
        return {"returncode": 0, "output": "ok"}

    def get_template_vars(self):
        return {}


def agent():
    return WebArenaAgent(FakeModel(), FakeEnv("."), system_template="system", instance_template="user",
                         step_limit=4, cost_limit=0, max_consecutive_format_errors=2)


def test_web_agent_rejects_multiple_actions_before_execution():
    value = agent()
    message = {"role": "assistant", "extra": {"actions": [
        {"command": "browser click [1]"}, {"command": "browser click [2]"}]}}
    with pytest.raises(Exception) as exc:
        value.execute_actions(message)
    assert type(exc.value).__name__ == "FormatError"
    assert message["extra"]["action_executed"] is False


def test_web_agent_executes_only_browser_protocol():
    value = agent()
    output = value.execute_actions({"role": "assistant", "extra": {"actions": [{"command": "browser scroll [down]"}]}})
    assert output[0]["role"] == "tool"
    with pytest.raises(Exception) as exc:
        value.execute_actions({"role": "assistant", "extra": {"actions": [{"command": "cat /etc/passwd"}]}})
    assert type(exc.value).__name__ == "FormatError"


def test_subject_requires_flushed_har_and_valid_answer(tmp_path):
    out = tmp_path / "attempt"
    web = out / "web-subject"
    web.mkdir(parents=True)
    har = web / "network.har"
    actions = web / "actions.jsonl"
    har.write_text('{"log":{"entries":[]}}')
    actions.write_text('{"index":1,"command":"browser scroll [down]"}\n')

    class Env:
        def finish(self):
            return {"har_path": str(har), "actions_path": str(actions), "har_flushed": True,
                    "runtime_identity": {"environments": {"reddit": {"urls": ["http://reddit.test"]}}}}

    task = {"source_task_id": 108, "intent": "Retrieve", "task_revision": 1}
    context = {"run_id": "run", "attempt_id": "attempt"}
    path, errors = finalize_subject(task, out, context, Env(), json.dumps({
        "task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": ["actual result"],
        "error_details": None}), "Submitted")
    subject = read_json(path)
    assert not errors
    assert subject["complete"] is True
    assert subject["har_flushed"] is True
    assert {row["path"] for row in subject["outputs"]} == {
        "network.har", "actions.jsonl", "agent_response.json"}
    assert all(sha256(path.parent / row["path"]) == row["sha256"] for row in subject["outputs"])


def test_invalid_answer_and_missing_har_produce_incomplete_subject(tmp_path):
    path, errors = finalize_subject({"source_task_id": 1}, tmp_path, {"attempt_id": "a"}, None,
                                    '{"task_type":"RETRIEVE","status":"SUCCESS"}', "Submitted")
    subject = read_json(path)
    assert errors
    assert subject["complete"] is False
    assert subject["har_flushed"] is False


def test_native_agent_first_request_contains_initial_ax_and_stop_has_tool_result(tmp_path):
    from minisweagent.exceptions import Submitted

    answer = {"task_type": "NAVIGATE", "status": "SUCCESS", "retrieved_data": None, "error_details": None}
    calls = []

    class Model(FakeModel):
        def query(self, messages):
            calls.append(list(messages))
            return {"role": "assistant", "content": "Finished", "extra": {"cost": 0, "actions": [
                {"command": "browser stop [" + json.dumps(answer) + "]", "tool_call_id": "tool-stop"}]}}

    class Env(FakeEnv):
        def get_template_vars(self):
            return {"initial_observation": "URL: http://reddit\n[7] link 'Observed article'"}

        def execute(self, action):
            raise Submitted({"role": "exit", "content": json.dumps(answer),
                             "extra": {"exit_status": "Submitted", "submission": json.dumps(answer)}})

        def serialize(self):
            return {}

    value = WebArenaAgent(Model(), Env(tmp_path), system_template="Browser only",
                         instance_template="{{ task }}\n{{ initial_observation }}", step_limit=3,
                         output_path=tmp_path / "raw.json")
    result = value.run("Navigate to the observed article")
    assert result["exit_status"] == "Submitted"
    assert len(calls) == 1
    assert "Observed article" in calls[0][1]["content"]
    assert [message["role"] for message in value.messages] == ["system", "user", "assistant", "tool", "exit"]
    assert "Browser task submitted" in value.messages[-2]["content"]
    assert json.loads(result["submission"]) == answer


def test_official_statistics_keep_pending_ungraded_and_infra_samples_in_denominator(tmp_path, monkeypatch):
    from application.execution.evidence import RunEvidence
    from application.execution.report import webarena_statistics
    from core.io import write_json

    rows = {
        "r-success": {"submission_status": "submitted", "evaluation_status": "completed", "resolved": True},
        "r-failed": {"submission_status": "submitted", "evaluation_status": "completed", "resolved": False},
        "r-ungraded": {"submission_status": "submitted", "evaluation_status": "not_run", "resolved": None},
        "r-infra": {"submission_status": "submitted", "evaluation_status": "infra_error", "resolved": None},
        "r-limits": {"submission_status": "unsubmitted", "evaluation_status": "not_run", "resolved": None},
    }
    monkeypatch.setattr(RunEvidence, 'read', lambda self: {'rollouts': rows})
    items = []
    for index, suffix in enumerate(('success', 'failed', 'ungraded', 'infra', 'limits', 'pending')):
        item = {'instance_id': f'wav-{index:03d}', 'repetition_id': 'rep-000', 'sample_key': suffix}
        items.append(item)
        if suffix != 'pending':
            path = tmp_path / 'rollout/tasks' / item['instance_id'] / 'repetitions/rep-000/result.json'
            write_json(path, {'sample_key': suffix, 'rollout_id': 'r-' + suffix,
                              'exit_status': 'LimitsExceeded' if suffix == 'limits' else 'Submitted'})
    cfg = {'project_root': str(tmp_path), 'execution_context': {'revision': 'fixed'}}
    result = webarena_statistics({'run': tmp_path, 'config': cfg, 'plan': {'run_id': 'smoke'},
                                 'work_items': {'items': items}})
    assert result['success_rate_denominator'] == 6
    assert result['success_rate'] == 1 / 6
    assert all(result[key] == 1 for key in ('resolved', 'unresolved', 'infra_error', 'ungraded', 'pending', 'unsubmitted'))
