"""Cross-step escape through the shared coordinator and native agent (no GPUs)."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.config import Parallel, load_snapshot, resolve, snapshot
from core.io import hash_json, read_json
from rollout.loop_escape import (
    LoopEscapeHistory,
    apply_escape,
    fingerprints,
    prepare_escape,
    validate_loop_escape_evidence,
)
from rollout.parallel import ParallelReasoningCoordinator, baseline_selection_seed, random_branch_id

MODES = ["parallel", "parallel_baseline", "parallel_consensus"]


class Response:
    def __init__(self, branch, command, reasoning="reason", finish="tool_calls"):
        self.branch = branch
        self.choices = [SimpleNamespace(finish_reason=finish)]
        self.value = {"id": f"response-{branch}", "choices": [{"finish_reason": finish, "message": {
            "role": "assistant", "reasoning_content": reasoning,
            "tool_calls": [{"id": f"call-{branch}", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": command})}}]}}],
            "usage": {"completion_tokens": 3}}

    def model_dump(self, **kwargs):
        return copy.deepcopy(self.value)


def parse(response):
    message = copy.deepcopy(response.value["choices"][0]["message"])
    message["extra"] = {"actions": [{
        **json.loads(call["function"]["arguments"]), "tool_call_id": call["id"]}
        for call in message["tool_calls"]], "response": response.model_dump()}
    return message


class Harness:
    def __init__(self, mode="parallel", count=4, enabled=True, fallback="first_valid"):
        self.records, self.calls = [], []
        self.score_error = False
        self.scores = {i: float(i) for i in range(count)}
        self.scoring_status = {}
        self.cfg = {"parallel": {"mode": mode, "branch_count": count, "score_timeout_ms": 200,
                                "fallback_policy": "stop" if mode == "parallel_consensus" else fallback,
                                "loop_escape": {"enabled": enabled}, "dynamic_branch_admission": True,
                                "consensus": {"retention_ratio": 1., "min_retained_branches": 2,
                                              "rule": "exact_action", "tie_policy": "estimator_score"}},
                    "generation": {"seed": 42}, "capture": {"control_seconds": 1}}
        self.history = LoopEscapeHistory()

    def call(self, action, payload):
        self.calls.append((action, copy.deepcopy(payload)))
        if action == "branch-acquire":
            return {"acquired": True}
        if action != "results":
            return {}
        if self.score_error:
            import requests

            raise requests.ConnectionError("unreachable")
        return {"results": [{"branch_id": b, "status": self.scoring_status.get(b, "scored"), "score": score}
                            for b, score in self.scores.items()],
                "estimator_sha256": "sha", "target": "target", "endpoint": "submission"}

    def record(self, decision, timeout):
        validate_loop_escape_evidence(decision, self.records)
        self.records.append(copy.deepcopy(decision))
        return decision

    def run(self, commands, reasons=None, step=None, failed=(), diagnostic=None):
        step = len(self.records) if step is None else step
        metadata = {"session_id": "session", "task_id": "task", "step_id": step, "query_id": f"query-{step}"}
        reasons = reasons if reasons is not None else ["reason"] * len(commands)

        def generate(messages, md, seed):
            b = md["branch_id"]
            assert seed == 42 + b
            if b in failed:
                raise RuntimeError("transport failed")
            return Response(b, commands[b], reasons[b])

        # The production wrapper creates a coordinator each query: history must
        # belong to the model instead of that short-lived coordinator.
        coordinator = ParallelReasoningCoordinator(
            self.cfg, SimpleNamespace(record_decision=self.record), self,
            loop_escape_history=self.history, diagnostic_mode=diagnostic)
        return coordinator.run([], metadata, generate, parse, lambda _: {"cost": .01})


def test_canonical_fingerprints_ignore_transport_and_json_key_order():
    a = Response(0, "echo x").model_dump()
    b = copy.deepcopy(a)
    b.update(id="transport", usage={"completion_tokens": 100})
    call = b["choices"][0]["message"]["tool_calls"][0]
    call["id"] = "another"
    call["function"]["arguments"] = '{ "command" : "echo x" }'
    assert fingerprints(a) == fingerprints(b)
    call["function"]["arguments"] = '{"command":"echo x "}'
    assert fingerprints(a)["action_fingerprint"] != fingerprints(b)["action_fingerprint"]
    c = copy.deepcopy(a)
    fn = c["choices"][0]["message"]["tool_calls"][0]["function"]
    fn["arguments"] = '{"command":"x","timeout":3}'
    d = copy.deepcopy(c)
    d["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"timeout":3,"command":"x"}'
    assert fingerprints(c) == fingerprints(d)
    c["choices"][0]["message"]["tool_calls"].append(a["choices"][0]["message"]["tool_calls"][0])
    d = copy.deepcopy(c)
    d["choices"][0]["message"]["tool_calls"].reverse()
    assert fingerprints(c)["action_fingerprint"] != fingerprints(d)["action_fingerprint"]


@pytest.mark.parametrize("reason", [None, "", 42])
def test_missing_reasoning_is_unknown(reason):
    assert fingerprints(Response(0, "x", reason).model_dump())["reasoning_fingerprint"] is None


@pytest.mark.parametrize("arguments", ['{"command":"x","command":"y"}', '[]', 'bad'])
def test_invalid_action_is_unknown(arguments):
    value = Response(0, "x").model_dump()
    value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    assert fingerprints(value)["action_fingerprint"] is None


@pytest.mark.parametrize("mode", MODES)
def test_initial_then_action_filter_preserves_mode_policy(mode):
    h = Harness(mode)
    h.run(["A", "B", "C", "D"])
    first = h.records[0]
    assert first["loop_escape"]["stage"] == "initial"
    assert not any(r["loop_escape"]["excluded"] for r in first["candidates"])
    winner = h.run(["A", "X", "Y", "D"], step=8)
    decision = h.records[-1]
    assert decision["loop_escape"]["decision_index"] == 1
    assert decision["loop_escape"]["stage"] == "action"
    assert decision["loop_escape"]["candidate_ids_after"] == [1, 2]
    assert decision["selected_branch_id"] in [1, 2]
    for i in (0, 3):
        row = decision["candidates"][i]
        assert row["loop_escape"]["action_comparison"] == "repeated"
        assert row["status"] == "loop_excluded"
        assert row["loop_escape"]["exclusion_reason"] == "repeated_action"
        assert row["loop_escape"]["action_matched_decision_id"] == first["decision_id"]
        assert row.get("exclusion_reason") != "estimator_filtered"
    if mode == "parallel_baseline":
        seed = baseline_selection_seed(42, "task", 8, decision["parent_prefix_sha256"])
        assert decision["selected_branch_id"] == random_branch_id(seed, [1, 2])
        assert decision["selection_candidate_ids"] == [1, 2]
    else:
        assert decision["selected_branch_id"] == 1
    if mode == "parallel_consensus":
        from rollout.consensus import validate_consensus_evidence

        validate_consensus_evidence(decision)
        assert decision["consensus"]["retained_branch_ids"] == [1, 2]
    from runtime.serving.parallel_validation import validate_selection

    assert validate_selection(decision, 42)["selected_branch_id"] == decision["selected_branch_id"]
    assert validate_selection(winner["extra"]["dycon_parallel"], 42)
    validate_loop_escape_evidence(decision, h.records[:-1])
    assert h.calls[-1][0] == "release"
    assert len([a for a, _ in h.calls if a == "branch-release"]) == 8


@pytest.mark.parametrize("mode", MODES)
def test_reasoning_only_when_all_actions_repeated(mode):
    h = Harness(mode)
    h.run(["A", "B", "C", "D"])
    h.run(["A", "B", "C", "D"], ["reason", "new1", "new2", "reason"])
    assert h.records[-1]["loop_escape"]["stage"] == "reasoning"
    assert h.records[-1]["loop_escape"]["candidate_ids_after"] == [1, 2]
    h.run(["A", "X", "Y", "D"], ["fresh", "new1", "new2", "fresh"])
    assert h.records[-1]["loop_escape"]["stage"] == "action"
    assert h.records[-1]["loop_escape"]["candidate_ids_after"] == [1, 2]


def test_same_slot_only_and_all_prior_history_including_unselected():
    h = Harness()
    h.run(["A", "B", "C", "D"])
    h.run(["B", "A", "D", "C"])
    assert h.records[-1]["loop_escape"]["candidate_ids_after"] == [0, 1, 2, 3]
    assert all(r["loop_escape"]["action_comparison"] == "new" for r in h.records[-1]["candidates"])
    h.run(["A", "B", "C", "D"])
    decision = h.records[-1]
    assert all(r["loop_escape"]["action_comparison"] == "repeated" for r in decision["candidates"])
    assert decision["loop_escape"]["failure_reason"] == "all_comparable_outputs_repeated"
    assert decision["loop_escape"]["proposed_candidate_ids"] == []
    assert decision["selected_branch_id"] == 0


def test_unknown_is_not_novelty_and_is_kept_conservatively():
    h = Harness()
    h.run(["A", "B", "C", "D"], [None] * 4)
    h.run(["A", "B", "C", "D"], [None] * 4)
    assert h.records[-1]["loop_escape"]["stage"] == "reasoning"
    assert h.records[-1]["loop_escape"]["failure_reason"] is None
    assert all(r["loop_escape"]["reasoning_comparison"] == "unknown" for r in h.records[-1]["candidates"])
    rows = [{"branch_id": 0, "status": "eligible", "reason": None,
             "response": {"choices": [{"message": {"reasoning_content": "r"}}]}}]
    state = LoopEscapeHistory()
    meta = prepare_escape(rows, {0}, state)
    apply_escape(rows, meta)
    state.remember({"decision_id": "first", "candidates": rows})
    meta = prepare_escape(rows, {0}, state)
    assert rows[0]["loop_escape"]["action_comparison"] == "unknown"
    assert meta["stage"] == "unknown"
    assert meta["proposed_candidate_ids"] == [0]


@pytest.mark.parametrize("mode", MODES)
def test_disabled_shape_and_new_attempt_isolation(mode):
    h = Harness(mode, enabled=False)
    for _ in range(2):
        h.run(["A", "B", "C", "D"])
    assert h.history.decisions == 0
    assert all("loop_escape" not in d and all("loop_escape" not in r for r in d["candidates"]) for d in h.records)
    h.cfg["parallel"]["loop_escape"]["enabled"] = True
    h.run(["A", "B", "C", "D"])
    assert h.records[-1]["loop_escape"]["stage"] == "initial"
    another = Harness(mode)
    another.run(["A", "B", "C", "D"])
    assert another.records[0]["loop_escape"]["stage"] == "initial"


def test_consensus_single_survivor_preserves_minimum_and_records_failure():
    h = Harness("parallel_consensus")
    h.run(["A", "B", "C", "D"])
    h.run(["A", "X", "C", "D"])
    decision = h.records[-1]
    assert decision["loop_escape"]["proposed_candidate_ids"] == [1]
    assert decision["loop_escape"]["failure_reason"] == "selection_unavailable_after_filter"
    assert decision["loop_escape"]["candidate_ids_after"] == [0, 1, 2, 3]
    assert len(decision["consensus"]["retained_branch_ids"]) == 4
    assert not any(r["loop_escape"]["excluded"] for r in decision["candidates"])


def test_score_fallback_selects_real_unscored_survivor():
    h = Harness()
    h.run(["A", "B", "C", "D"])
    h.scoring_status[1] = "failed"
    h.run(["A", "X", "C", "D"])
    decision = h.records[-1]
    assert decision["selected_branch_id"] == 1
    assert decision["fallback_reason"] == "no_eligible_scored_branch"
    from runtime.serving.parallel_validation import validate_selection

    assert validate_selection(decision, 42)


def test_stop_policy_restores_real_candidates_when_survivor_is_unscored():
    h = Harness(fallback="stop")
    h.run(["A", "B", "C", "D"])
    h.scoring_status[1] = "failed"
    h.run(["A", "X", "C", "D"])
    assert h.records[-1]["selected_branch_id"] == 0
    assert h.records[-1]["loop_escape"]["failure_reason"] == "selection_unavailable_after_filter"


def test_failed_branch_does_not_establish_history():
    h = Harness()
    h.run(["A", "B", "C", "D"], failed=[1])
    h.run(["A", "X", "Y", "D"])
    row = h.records[-1]["candidates"][1]["loop_escape"]
    assert row["history_count"] == 0
    assert row["action_comparison"] == "unknown"
    assert not row["excluded"]


def test_baseline_scoring_failure_remains_fatal_and_releases_reservation():
    from rollout.parallel import ParallelSelectionError

    h = Harness("parallel_baseline")
    h.run(["A", "B", "C", "D"])
    h.score_error = True
    with pytest.raises(ParallelSelectionError, match="Cannot verify baseline"):
        h.run(["A", "B", "C", "D"])
    assert h.records[-1]["selected_branch_id"] is None
    assert h.calls[-1][0] == "release"


@pytest.mark.parametrize("enabled", [False, True])
def test_diagnostic_selection_does_not_enable_history(enabled):
    h = Harness(enabled=enabled)
    h.run(["A", "B", "C", "D"], diagnostic="first_valid_unscored")
    assert "loop_escape" not in h.records[-1]
    assert h.history.decisions == 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("enabled", [False, True])
def test_native_agent_all_repeat_stops_at_eight_not_two(tmp_path, mode, enabled):
    from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

    from rollout.agent import DurableAgent

    h = Harness(mode, enabled=enabled)

    class Model:
        def query(self, messages):
            return h.run(["A"] * 4)

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, template_vars):
            return format_toolcall_observation_messages(actions=message["extra"]["actions"], outputs=outputs,
                observation_template="{{output.output}} {{output.exception_info | default('')}}",
                template_vars=template_vars)

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    executed = []

    class Environment:
        def execute(self, action):
            executed.append(action)
            return {"returncode": 0, "output": "ok"}

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    agent = DurableAgent(Model(), Environment(), system_template="s", instance_template="{{task}}",
                         loop_repeat_limit=8, cost_limit=0, output_path=tmp_path / "raw.traj.json")
    result = agent.run("task")
    assert result["exit_status"] == "LimitsExceeded"
    assert result["limit_reason"] == "repeated_reasoning_action_loop"
    assert len(h.records) == 8
    assert len(executed) == 7
    assert read_json(tmp_path / "raw.traj.json")["info"]["loop_detection"]["repeats"] == 8
    if enabled:
        assert all(d["loop_escape"]["failure_reason"] == "all_comparable_outputs_repeated" for d in h.records[1:])
        assert all(d["selected_branch_id"] is not None for d in h.records)


@pytest.mark.parametrize("field,value", [("enabled", "true"), ("enabled", 1), ("enabled", None),
                                         ("typo", True)])
def test_strict_config_rejects_bad_escape(field, value):
    with pytest.raises(ValueError):
        Parallel(loop_escape={field: value})


def test_config_defaults_legacy_identity_and_frozen_resume(cfg):
    assert Parallel().loop_escape.enabled is False
    path = Path(cfg["config_source"])
    raw = yaml.safe_load(path.read_text())
    raw["parallel"] = {"enabled": False}
    path.write_text(yaml.safe_dump(raw))
    legacy = resolve(path)
    assert "loop_escape" not in legacy["parallel"]
    old_run = snapshot(legacy, "legacy-escape", {"task_keys": []})
    before = (old_run / "resolved-config.json").read_bytes()
    raw["parallel"]["loop_escape"] = {"enabled": True}
    path.write_text(yaml.safe_dump(raw))
    fresh = resolve(path)
    assert fresh["parallel"]["loop_escape"]["enabled"] is True
    assert legacy["config_hash"] != fresh["config_hash"]
    run = snapshot(fresh, "new-escape", {"task_keys": []})
    fresh_bytes = (run / "resolved-config.json").read_bytes()
    raw["parallel"]["loop_escape"]["enabled"] = False
    path.write_text(yaml.safe_dump(raw))
    assert load_snapshot(cfg["project_root"], "new-escape")["parallel"]["loop_escape"]["enabled"] is True
    assert "loop_escape" not in load_snapshot(cfg["project_root"], "legacy-escape")["parallel"]
    assert (old_run / "resolved-config.json").read_bytes() == before
    assert (run / "resolved-config.json").read_bytes() == fresh_bytes
    assert hash_json({k: v for k, v in legacy.items() if k != "config_hash"}) == legacy["config_hash"]


@pytest.mark.parametrize("change", ["comparison", "response", "excluded", "stage", "winner", "index", "status", "fallback"])
def test_evidence_verifier_rejects_tampering(change):
    h = Harness()
    h.run(["A", "B", "C", "D"])
    h.run(["A", "X", "Y", "D"])
    value = copy.deepcopy(h.records[-1])
    row = value["candidates"][0]
    if change == "comparison":
        row["loop_escape"]["action_comparison"] = "new"
    elif change == "response":
        row["response"]["choices"][0]["message"]["reasoning_content"] = "tampered"
    elif change == "excluded":
        row["loop_escape"]["excluded"] = False
    elif change == "stage":
        value["loop_escape"]["stage"] = "reasoning"
    elif change == "winner":
        value["selected_branch_id"] = 0
    elif change == "index":
        value["loop_escape"]["decision_index"] = 99
    elif change == "status":
        row["status"] = "eligible"
    else:
        value["loop_escape"].update(failure_reason="selection_unavailable_after_filter", applied=False,
                                  candidate_ids_after=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="Loop escape"):
        validate_loop_escape_evidence(value, h.records[:-1])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", ["every_step", "every_k_steps", "first_step"])
def test_recording_model_keeps_attempt_history_and_real_receipt_replay(cfg, tmp_path, monkeypatch, mode, policy):
    import numpy as np
    from litellm import ModelResponse
    from minisweagent.models.litellm_model import LitellmModel

    from hidden.backends.vllm.online import capture_request_metadata
    from rollout.model import make_model
    from runtime.serving.parallel import ScoringBroker
    from tests.test_parallel_rollout import controller, provider

    options = provider(tmp_path, monkeypatch)
    options.update(mode=mode, branch_count=4, max_inflight_branches=4, policy=policy,
                   loop_escape={"enabled": True}, consensus={"retention_ratio": 1., "min_retained_branches": 2,
                                                           "rule": "exact_action", "tie_policy": "estimator_score"})
    if mode == "parallel_consensus":
        options["fallback_policy"] = "stop"
    if policy == "every_k_steps":
        options["every_k_steps"] = 3
    cfg = copy.deepcopy(cfg)
    cfg["parallel"] = options
    cfg["capture"].update(enabled=True, anchors=["think_close"])
    broker = ScoringBroker(options, "model", 2)
    capture = controller(tmp_path)
    md = capture_request_metadata("session", "task", "run", "pending", 0, ["think_close"], close_token_ids=[11])
    capture.start(md)

    class Client:
        def call(self, action, payload):
            if action == "reserve":
                return broker.reserve(payload["decision_id"], payload["branches"])
            if action == "results":
                return broker.results(payload["decision_id"], payload["timeout"])
            return broker.release(payload["decision_id"])

    def query(self, messages, **kwargs):
        metadata = copy.deepcopy(kwargs.get("extra_body", self.config.model_kwargs["extra_body"])["dycon_capture"])
        b = metadata.get("branch_id", 0)
        step = metadata["step_id"]
        command = f"old-{b}" if step == 0 or b in [0, 3] else f"new-{b}"
        response = Response(b, command).model_dump()
        if metadata.get("decision_id"):
            rid = f"native-{step}-{b}"
            capture.bind(rid, {**metadata, "prompt_token_ids": [10, 198]})
            capture.record_output({"request_id": rid, "prompt_token_ids": [10, 198],
                                   "output_token_ids": [20, 11, 30]})
            arrays = {"layer_0": np.array([b + 1, 1], dtype=np.float32)}
            capture.record(rid, arrays, anchor="think_close", position=3, token_id=11, absolute_position=3)
            broker.submit(metadata, rid, 3, 11, arrays)
        return ModelResponse(model="fixture", choices=response["choices"])

    monkeypatch.setattr(LitellmModel, "_query", query)
    try:
        model = make_model(cfg, SimpleNamespace(remaining=lambda n: n), md,
                           {"capture_controller": capture, "parallel_client": Client()})
        queries = 4 if policy == "every_k_steps" else 2
        for _ in range(queries):
            message = model.query([{"role": "user", "content": "task"}])
        records = sorted(capture.parallel_decisions.values(), key=lambda d: d["step_id"])
        assert len(records) == (1 if policy == "first_step" else 2)
        assert model._dycon_loop_escape.decisions == len(records)
        if policy != "first_step":
            assert records[-1]["loop_escape"]["candidate_ids_after"] == [1, 2]
            assert records[-1]["loop_escape"]["decision_index"] == 1
            validate_loop_escape_evidence(records[-1], records[:-1])
            assert message["extra"]["dycon_parallel"]["selected_branch_id"] in [1, 2]
        paths = list((capture.export_root / "sessions/session/parallel/decisions").glob("*.json"))
        assert len(paths) == len(records)
        model.reset_loop_escape()
        assert model._dycon_loop_escape.decisions == 0
        assert not model._dycon_loop_escape.slots
        other = make_model(cfg, SimpleNamespace(remaining=lambda n: n), md,
                           {"capture_controller": capture, "parallel_client": Client()})
        assert other._dycon_loop_escape is not model._dycon_loop_escape
    finally:
        capture.close()
        broker.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("enabled", [False, True])
def test_native_token_loop_keeps_original_termination(mode, enabled):
    from minisweagent.exceptions import LimitsExceeded

    h = Harness(mode, count=2, enabled=enabled)
    metadata = {}
    original_call = h.call

    def call(action, payload):
        if action != "results":
            return original_call(action, payload)
        rows = []
        for branch, md in metadata.items():
            native = f"native-{branch}"
            loop = {**{k: md[k] for k in ("session_id", "decision_id", "branch_id", "api_attempt_id")},
                    "native_request_id": native, "abort_completed": True}
            rows.append({"branch_id": branch, "status": "failed", "reason": "reasoning_loop",
                         "native_request_id": native, "api_attempt_id": md["api_attempt_id"],
                         "reasoning_loop": loop})
        return {"results": rows, "estimator_sha256": "sha", "target": None, "endpoint": None}

    def generate(messages, md, seed):
        metadata[md["branch_id"]] = copy.deepcopy(md)
        return Response(md["branch_id"], "partial", finish="abort")

    parsed = []
    client = SimpleNamespace(call=call)
    c = ParallelReasoningCoordinator(h.cfg, SimpleNamespace(record_decision=h.record), client,
                                     loop_escape_history=h.history)
    with pytest.raises(LimitsExceeded) as exc:
        c.run([], {"session_id": "session", "task_id": "task", "step_id": 0, "query_id": "query"},
              generate, lambda response: parsed.append(response), lambda _: {"cost": 0.})
    assert not parsed
    extra = exc.value.messages[-1]["extra"]
    assert extra["limit_reason"] == "all_parallel_branches_reasoning_loop"
    assert extra["exit_status"] == ("reasoning_loop" if mode == "parallel_baseline" else "LimitsExceeded")
    assert h.records[-1]["selected_branch_id"] is None
    assert all(r["reason"] == "reasoning_loop" for r in h.records[-1]["candidates"])
    assert h.calls[-1][0] == "release"


@pytest.mark.parametrize("mode", MODES)
def test_active_escape_continues_native_agent_past_old_repetition_point(mode):
    from minisweagent.exceptions import Submitted
    from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

    from rollout.agent import DurableAgent

    h = Harness(mode)
    executed = []

    class Model:
        def query(self, messages):
            step = len(h.records)
            return h.run(["stuck", f"progress-{step}", f"progress-{step}", "stuck"])

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, template_vars):
            return format_toolcall_observation_messages(actions=message["extra"]["actions"], outputs=outputs,
                observation_template="{{output.output}}", template_vars=template_vars)

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    class Environment:
        def execute(self, action):
            executed.append(action["command"])
            if len(executed) == 10:
                raise Submitted({"role": "exit", "content": "done", "extra": {
                    "exit_status": "Submitted", "submission": "patch"}})
            return {"returncode": 0, "output": "ok"}

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    agent = DurableAgent(Model(), Environment(), system_template="s", instance_template="{{task}}",
                         loop_repeat_limit=8, cost_limit=0)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert len(executed) == 10
    assert all(command.startswith("progress-") for command in executed[1:])
    assert all(d["loop_escape"]["candidate_ids_after"] == [1, 2] for d in h.records[1:])


def test_create_run_binds_new_config_without_modifying_old_run(cfg):
    from application.execution.run_plan import create_run, load_plan
    from catalog.store import Catalog
    from tests.conftest import task

    path = Path(cfg["config_source"])
    raw = yaml.safe_load(path.read_text())
    raw["parallel"] = {"enabled": False, "loop_escape": {"enabled": True}}
    path.write_text(yaml.safe_dump(raw))
    configured = resolve(path)
    Catalog(configured["project_root"]).register_tasks([task("escape-config-task")])
    run = create_run(configured, "escape-config-run", operation="rollout", task_ids=["escape-config-task"])
    before = (run / "resolved-config.json").read_bytes()
    raw["parallel"]["loop_escape"]["enabled"] = False
    path.write_text(yaml.safe_dump(raw))
    plan = load_plan(configured["project_root"], "escape-config-run")
    assert plan["config"]["parallel"]["loop_escape"]["enabled"] is True
    assert (run / "code-identity.json").exists()
    assert (run / "resolved-config.json").read_bytes() == before
