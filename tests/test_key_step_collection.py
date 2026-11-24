"""Recovery must fail closed; selection and scheduling must preserve groups."""

import copy
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from core.io import hash_json, read_json, write_json
from rollout.key_steps.config import KeyStepConfig, work_items
from rollout.key_steps.peaks import detect, signal
from rollout.key_steps.schedule import MixedQueue
from rollout.key_steps.state import ReplayMismatch, append, replay


def test_full_branch_std_includes_excluded_candidates_and_refuses_gaps():
    candidates = [{"branch_id": i, "score": float(i), "status": "loop_excluded"} for i in range(8)]
    assert signal(0, candidates)["std"] == pytest.approx(np.std(range(8)))
    assert not signal(0, candidates[:-1])["complete"]
    assert not signal(0, [*candidates[:-1], candidates[0]])["complete"]
    candidates[3]["score"] = float("nan")
    assert not signal(0, candidates)["complete"]


def test_adaptive_gaps_endpoints_and_no_forced_peaks():
    def make(values):
        return [{"step": i, "complete": v is not None, "std": v} for i, v in enumerate(values)]

    assert detect(make([0.2] * 12))[0] == []
    peaks, _ = detect(make([0.7, 0.1, 0.1, 0.1, None, 0.8, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
    assert 0 in [p["step"] for p in peaks]
    assert 5 not in [p["step"] for p in peaks]  # missing-score edge is not an endpoint
    assert 8 in [p["step"] for p in peaks]


def test_mixed_queue_task_service_caps_and_retry_without_dropping_items(tmp_path):
    settings = KeyStepConfig(
        mode="source", network_default="local-only", task_services={"a__1": ["httpbin"], "b__2": ["httpbin"]}
    ).model_dump()
    items = [
        {"sample_key": str(i), "instance_id": t}
        for i, t in enumerate(["a__1", "a__1", "b__1", "b__2", "c__1"])
    ]
    pending = deque(items)
    journal = tmp_path / "schedule.jsonl"
    queue = MixedQueue(items, settings, 4, journal)
    started = []
    while True:
        item = queue.pop(pending, clock=100)
        if item is None:
            break
        started.append(item)
    assert len({x["instance_id"] for x in started}) == len(started)
    assert sum("httpbin" in queue.services(x) for x in started) <= 1
    first = started[0]
    queue.release(first, retry=True, clock=101)
    pending.append(first)
    assert queue.not_before[first["sample_key"]] == 131
    restored = MixedQueue(items, settings, 4, journal)
    assert restored.rank == queue.rank
    assert restored.not_before[first["sample_key"]] == 131
    assert len(pending) + len(queue.active) == len(items)


def test_work_items_keep_child_identity_and_group_multiplicity():
    children = [
        {
            "child_id": str(i),
            "task_key": "t",
            "instance_id": "repo__1",
            "seed": i,
            "parent_id": "p",
            "branch": i,
        }
        for i in range(8)
    ]
    cfg = {"key_steps": {"mode": "continuation", "inventory": {"children": children}}}
    value = work_items(cfg, {"task_keys": ["t"], "tasks": {"t": {"instance_id": "repo__1"}}}, "new-run", 1)
    assert len({x["sample_key"] for x in value["items"]}) == 8
    assert [x["continuation"]["branch"] for x in value["items"]] == list(range(8))
    assert len({x["repetition_id"] for x in value["items"]}) == 8
    with pytest.raises(ValueError):
        KeyStepConfig(mode="source", branch_count=32)


def test_replay_checks_each_action_and_never_reaches_candidate_on_mismatch(tmp_path, monkeypatch):
    from rollout.key_steps import state

    initial = {"files": {"/testbed/a": {"bytes": 1}}, "git": {}, "errors": []}
    final = {"files": {"/testbed/a": {"bytes": 2}}, "git": {}, "errors": []}
    action = {"command": "edit"}
    expected = {"output": "ok", "returncode": 0, "exception_info": ""}
    write_json(tmp_path / "replay-contract.json", {"image": "fixed"})
    append(
        tmp_path / "replay-actions.jsonl",
        {
            "action": action,
            "cwd": "",
            "timeout": None,
            "status": "returned",
            "result_sha256": hash_json(expected),
        },
    )
    for t, s in enumerate([initial, final]):
        append(
            tmp_path / "replay-boundaries.jsonl",
            {
                "step": t,
                "action_count": t,
                "state": {"changed": s["files"], "removed": [], "git": {}},
                "state_sha256": hash_json(s),
            },
        )
    states = iter([initial, final])
    calls = []
    env = SimpleNamespace(execute=lambda *a, **k: (calls.append(a), expected)[1])
    monkeypatch.setattr(state, "contract", lambda e: {"image": "fixed"})
    monkeypatch.setattr(state, "fingerprint", lambda *a: next(states))
    result = replay(env, tmp_path, {}, 1, tmp_path / "receipt.json")
    assert result["step"] == 1 and len(calls) == 1 and read_json(tmp_path / "receipt.json")["passed"]
    states = iter([initial])
    monkeypatch.setattr(state, "fingerprint", lambda *a: next(states))
    env.execute = lambda *a, **k: {**expected, "output": "HTTP 429"}
    with pytest.raises(ReplayMismatch, match="observation mismatch"):
        replay(env, tmp_path, {}, 1, tmp_path / "failed.json")
    assert not read_json(tmp_path / "failed.json")["passed"]


def test_entropy_is_distribution_entropy_and_leaves_rng_and_logits_unchanged():
    torch = pytest.importorskip("torch")
    from runtime.serving.generation_entropy import entropy

    logits = torch.tensor([[0.0, 0.0, -float("inf")], [0.0, -float("inf"), -float("inf")]])
    original = logits.clone()
    rng = torch.random.get_rng_state().clone()
    values = entropy(logits)
    assert values.tolist() == pytest.approx([math.log(2), 0])
    assert torch.equal(original, logits) and torch.equal(rng, torch.random.get_rng_state())
    assert entropy(logits, torch.tensor([True, False])).tolist() == [0, 0]


def test_saved_response_uses_native_parser_and_spends_one_call(cfg, tmp_path):
    pytest.importorskip("minisweagent")
    from core.deadline import Deadline
    from rollout.agent import DurableAgent
    from rollout.model import make_model

    model = make_model(cfg, Deadline())
    response = {
        "id": "source-response",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "```bash\necho candidate\n```"},
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
    }
    # The exact parser output is the public contract; the test does not assume
    # that a free-text bash block is valid for every installed native protocol.
    from litellm import ModelResponse

    try:
        expected = model._parse_actions(ModelResponse(**response))
    except Exception as error:
        assert type(error).__name__ == "FormatError"
        response["choices"][0]["message"] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"echo candidate"}'},
                }
            ],
        }
        expected = model._parse_actions(ModelResponse(**response))
    model.inject_saved_response(response, 0.0, {"native_request_id": "original"})
    env = SimpleNamespace(get_template_vars=lambda **kw: {}, serialize=lambda: {})
    agent = DurableAgent(
        model,
        env,
        system_template="system",
        instance_template="{{task}}",
        step_limit=3,
        output_path=tmp_path / "trajectory.json",
    )
    agent.messages = [{"role": "system", "content": "prefix"}, {"role": "user", "content": "task"}]
    agent.n_calls = 2
    message = agent.query()
    assert agent.n_calls == 3 and len(agent.messages) == 3
    assert message["extra"]["actions"] == expected
    assert message["extra"]["continuation_source"]["native_request_id"] == "original"
    from minisweagent.exceptions import LimitsExceeded

    with pytest.raises(LimitsExceeded):
        agent.query()
    assert agent.n_calls == 3


def test_source_and_scan_run_plans_roundtrip_without_mutating_source(cfg, monkeypatch, tmp_path):
    import yaml

    from application.execution.run_plan import create_run, load_plan
    from catalog.store import Catalog
    from core.config import Parallel
    from tests.conftest import task

    config = copy.deepcopy(cfg)
    manifest = tmp_path / "estimator.json"
    manifest.write_text("{}")
    config["parallel"] = Parallel(
        enabled=True,
        mode="parallel_consensus",
        branch_count=4,
        estimator_ref=str(manifest),
        decision_ttl_seconds=config["timeouts"]["model_seconds"] + 60,
        save_sites=[3],
        fallback_policy="stop",
    ).model_dump()
    config["capture"].update(enabled=True, anchors=["think_close"], layers=[3])
    config["capture"]["export"]["enabled"] = True
    config["serving"].update(management="managed", max_num_seqs=8)
    config["config_hash"] = hash_json({k: v for k, v in config.items() if k != "config_hash"})
    monkeypatch.setattr("runtime.serving.parallel.artifact_contract", lambda *a, **kw: {"sites": [3]})
    Catalog(cfg["project_root"]).register_tasks([task("key-task")])
    file = tmp_path / "source.yaml"
    file.write_text("schema_version: 1\nmode: source\n")
    source = create_run(config, "mother", operation="rollout", task_ids=["key-task"], key_step_config=file)
    loaded = load_plan(cfg["project_root"], "mother")
    assert loaded["config"]["parallel"]["branch_count"] == 8
    assert loaded["config"]["key_steps"]["mode"] == "source"
    assert loaded["config"]["capture"]["logprobs"]["enabled"]
    source_bytes = (source / "resolved-config.json").read_bytes()
    first = loaded["work_items"]["items"][0]
    inventory = {
        "schema": "key-step-scan-v1",
        "branch_count": 8,
        "source_run": "mother",
        "source_config_hash": loaded["config"]["config_hash"],
        "children": [
            {
                "child_id": "scan-1",
                "task_key": first["task_key"],
                "instance_id": "key-task",
                "seed": 123,
                "source": {},
            }
        ],
    }
    inventory["sha256"] = hash_json(inventory)
    scan = tmp_path / "scan.yaml"
    scan.write_text(yaml.safe_dump({"mode": "scan", "inventory": inventory}))
    create_run(config, "scan", operation="rollout", from_run="mother", key_step_config=scan)
    restored = load_plan(cfg["project_root"], "scan")
    assert restored["config"]["key_steps"]["mode"] == "scan"
    assert not restored["config"]["capture"]["enabled"]
    assert not restored["config"]["parallel"]["enabled"]
    assert restored["work_items"]["items"][0]["continuation"]["child_id"] == "scan-1"
    assert (source / "resolved-config.json").read_bytes() == source_bytes


def test_capture_finalizes_child_suffix_without_requiring_injection_in_new_session(tmp_path):
    from core.io import sha256
    from hidden.backends.vllm.online import capture_request_metadata
    from runtime.serving.capture import OnlineCaptureController

    controller = OnlineCaptureController(tmp_path, {"export": {"root": "online"}, "finalize_seconds": 1})
    md = capture_request_metadata("child-session", "task", "child-run", "pending", 4, ["think_close"])
    md["query_id"] = "new-query"
    md["message_index"] = 12
    controller.start(md)
    try:
        controller.bind("new-native", md)
        controller.record_output(
            {
                "request_id": "new-native",
                "metadata": md,
                "external_response_id": "new-native",
                "prompt_token_ids": [1, 2],
                "output_token_ids": [3, 4],
                "finish_reason": "stop",
            }
        )
        path = tmp_path / "child.traj.json"
        write_json(
            path,
            {
                "continuation": {"prefix_message_count": 10},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "original candidate",
                        "extra": {
                            "continuation_source": {"native_request_id": "old-native"},
                            "response": {"id": "old-native"},
                        },
                    },
                    {"role": "tool", "content": "actual candidate result"},
                    {
                        "role": "assistant",
                        "content": "suffix",
                        "extra": {
                            "request": {"dycon_capture": md},
                            "response": {"id": "new-native", "choices": []},
                        },
                    },
                ],
            },
        )
        controller.prepare_trajectory("child-session", str(path))
        source = read_json(path)
        assert source["messages"][0]["extra"]["response"]["id"] == "old-native"
        assert source["messages"][2]["extra"]["request"]["message_index"] == 12
        # New session accepts only its generated suffix, while the injection is
        # explicitly bound to the old native receipt in continuation_source.
        result = controller.finalize(
            "child-session",
            rollout_id="r-child",
            trajectory_path=str(path),
            trajectory_sha256=sha256(path),
            model_fingerprint="fixture",
        )
        assert result["accepted_native_request_ids"] == ["new-native"]
    finally:
        controller.close()
