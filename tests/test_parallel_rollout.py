"""Branch identity, action isolation and real spool round trips (no GPU)."""

import copy
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.config import Parallel, resolve
from core.io import read_json, sha256, write_json
from enrichment.online_trace import finalize_token_events
from hidden.artifacts import read_arrays, read_branch_index
from hidden.backends.vllm.online import capture_request_metadata
from rollout.parallel import ParallelReasoningCoordinator, active
from runtime.serving.capture import OnlineCaptureController
from runtime.serving.parallel import ScoringBroker, artifact_contract


def provider(tmp_path, monkeypatch, predict=None, sites=None):
    module = SimpleNamespace()
    module.load = lambda root, contract: SimpleNamespace(
        predict=predict or (lambda x: x[:, 0, 0]))
    monkeypatch.setitem(sys.modules, "parallel_fixture_provider", module)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"fixture")
    contract = {"schema_version": 1, "loader": "parallel_fixture_provider:load",
                "model_fingerprint": "model", "sites": sites or [0], "hidden_size": 2,
                "target": "fixture_remaining", "endpoint": "submission",
                "files": {"weights.bin": sha256(weights)}}
    path = tmp_path / "estimator.json"
    write_json(path, contract)
    return {"enabled": True, "branch_count": 2, "policy": "every_step",
            "estimator_ref": str(path), "estimator_sha256": sha256(path),
            "max_inflight_branches": 2, "score_timeout_ms": 500,
            "fallback_policy": "first_valid", "decision_ttl_seconds": 60}


def controller(tmp_path, spool=False):
    export = {"root": "online", "queue_size": 8}
    if spool:
        export.update(spool_root="spool", spool_max_bytes=10 * 1024**2,
                      shard_max_bytes=32768, upload_workers=1,
                      upload_max_elapsed_seconds=10)
    nas_base = os.environ.get("DYCON_PARALLEL_TEST_NAS_ROOT")
    nas_root = Path(nas_base) / tmp_path.name if nas_base else tmp_path / "nas"
    return OnlineCaptureController(tmp_path, {"export": export, "run_id": "run",
        "nas_root": str(nas_root), "finalize_seconds": 2,
        "open_token_ids": [10], "close_token_ids": [11], "layers": [0]})


class Completion:
    def __init__(self, branch):
        self.branch = branch
        self.choices = [SimpleNamespace(finish_reason="tool_calls")]

    def model_dump(self, **kwargs):
        return {"id": f"external-{self.branch}", "choices": [{"index": 0,
                "finish_reason": "tool_calls", "message": {"role": "assistant",
                "tool_calls": [{"id": f"tool-{self.branch}"}]}}],
                "usage": {"completion_tokens": 4}}


class BrokerClient:
    def __init__(self, broker):
        self.broker = broker

    def call(self, action, payload):
        if action == "reserve":
            return self.broker.reserve(payload["decision_id"], payload["branches"])
        if action == "results":
            return self.broker.results(payload["decision_id"], payload["timeout"])
        return self.broker.release(payload["decision_id"])


def test_loop_terminal_result_survives_late_predictor_and_release(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def predict(batch):
        entered.set()
        release.wait(2)
        return np.zeros(len(batch))

    broker = ScoringBroker(provider(tmp_path, monkeypatch, predict), "model", 2)
    md = {"session_id": "s", "decision_id": "d", "branch_id": 0, "api_attempt_id": "a"}
    try:
        broker.reserve("d", [0])
        broker.submit(md, "r", 1, 11, {"layer_0": np.ones(2)})
        assert entered.wait(1)
        broker.mark_terminated(md, "r", {"abort_completed": True})
        release.set()
        broker.jobs.join()
        row = broker.results("d", 0)["results"][0]
        assert row["reason"] == "reasoning_loop"
        assert row["status"] == "failed"
        broker.submit(md, "r", 2, 11, {"layer_0": np.ones(2)})
        assert broker.results("d", 0)["results"][0] == row
        broker.release("d")
        broker.mark_terminated(md, "r", {"abort_completed": True})
        assert "d" not in broker.groups
    finally:
        release.set()
        broker.close()


@pytest.mark.parametrize("offline", [False, True])
def test_winner_one_all_hidden_nas_and_canonical_zero_choice(tmp_path, monkeypatch, offline):
    options = provider(tmp_path, monkeypatch)
    broker = ScoringBroker(options, "model", 2)
    capture = controller(tmp_path, spool=True)
    original_submit = capture.spool.pool.submit
    if offline:
        from concurrent.futures import Future

        def disconnected(*args):
            future = Future()
            future.set_exception(OSError("NAS unavailable during selection"))
            return future

        monkeypatch.setattr(capture.spool.pool, "submit", disconnected)
    metadata = capture_request_metadata("session", "task", "run", "pending", 0,
                                        ["think_close"], close_token_ids=[11])
    metadata.update(query_id="query")
    capture.start(metadata)

    def generate(messages, md, seed):
        branch = md["branch_id"]
        rid = f"native-{branch}"
        prompt, output = [1, 10], [20, 11, 30, 31]
        capture.bind(rid, {**md, "prompt_token_ids": prompt})
        capture.record_output({"request_id": rid, "prompt_token_ids": prompt,
                               "output_token_ids": output})
        arrays = {"layer_0": np.array([6 - branch * 4, 1], dtype=np.float32)}
        capture.record(rid, arrays, anchor="think_close", position=3, token_id=11,
                       absolute_position=3, graph_replays=1, batch_kind="decode")
        broker.submit(md, rid, 3, 11, arrays)
        return Completion(branch)

    def parse(response):
        return {"role": "assistant", "content": f"branch-{response.branch}",
                "extra": {"actions": [{"branch": response.branch}],
                          "response": response.model_dump()}}

    cfg = {"parallel": options, "generation": {"seed": 42},
           "capture": {"control_seconds": 1}, "timeouts": {"metadata_request_seconds": 1}}
    coordinator = ParallelReasoningCoordinator(cfg, capture, BrokerClient(broker))
    try:
        winner = coordinator.run([], metadata, generate, parse, lambda _: {"cost": 1.0})
        assert winner["content"] == "branch-1"
        assert winner["extra"]["actions"] == [{"branch": 1}]
        assert winner["extra"]["cost"] == 2
        assert winner["extra"]["dycon_parallel"]["selected_branch_id"] == 1
        assert len(capture.sessions["session"].requests) == 2
        winner["extra"]["request"] = {"dycon_capture": dict(metadata)}
        trajectory = tmp_path / "trajectory.json"
        write_json(trajectory, {"messages": [winner]})
        capture.prepare_trajectory("session", trajectory)
        canonical = read_json(trajectory)
        assert canonical["messages"][0]["extra"]["response"]["dycon_native_request_id"] == "native-1"
        finalized = capture.finalize("session", rollout_id="canonical", trajectory_path=str(trajectory),
                                     trajectory_sha256=sha256(trajectory), model_fingerprint="model")
        assert finalized["capture_status"] == ("pending_upload" if offline else "complete")
        assert finalized["accepted_native_request_ids"] == ["native-1"]
        assert not capture.parallel_decisions  # Completed sessions do not accumulate branch histories.
        if offline:
            decision_file = next((capture.export_root / "sessions/session/parallel/decisions").glob("*.json"))
            original_hash = sha256(decision_file)
            monkeypatch.setattr(capture.spool.pool, "submit", original_submit)
            assert capture.spool.recover()
            assert capture.spool.publish_session("session", 10)
            assert sha256(decision_file) == original_hash
        nas = capture.spool.nas
        branch_root = nas / "sessions/session/parallel"
        index = read_json(branch_root / "branch-index.json")
        assert read_branch_index(nas, "session")["samples"] == index
        assert [r["selected"] for r in index] == [False, True]
        assert read_json(branch_root / "branch-coverage.json")["missing"] == []
        for row in index:
            assert row["labels"]["remaining_steps"] is None
            arrays = read_arrays(nas, row["close_events"][0])
            assert arrays["layer_0"][0] == 6 - row["branch_id"] * 4
        proof = read_json(branch_root / "branch-receipt.json")
        for relative, digest in proof["files"].items():
            assert sha256(nas / relative) == digest
        finalized_events = tmp_path / "enriched"
        finalize_token_events(nas, finalized_events, open_ids=[10], close_ids=[11])
        events = read_json(finalized_events / "index.json")
        assert len(events) == 1 and events[0]["request_id"] == "native-1"
        assert "native-0" not in json.dumps(events)
        assert broker.groups == {}
    finally:
        capture.close()
        broker.close()


def test_broker_bounded_hung_provider_release_and_no_late_recreation(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def predict(batch):
        entered.set()
        release.wait(2)
        return np.zeros(len(batch))

    options = provider(tmp_path, monkeypatch, predict)
    broker = ScoringBroker(options, "model", 2)
    try:
        broker.reserve("d", [0, 1])
        for b in range(2):
            broker.submit({"decision_id": "d", "branch_id": b, "session_id": "s"},
                          f"r{b}", 5, 11, {"layer_0": np.ones(2)})
        assert entered.wait(1)
        assert broker.results("d", 0)["results"][0]["status"] == "pending"
        broker.release("d")
        broker.reserve("next", [0])
        broker.submit({"decision_id": "next", "branch_id": 0, "session_id": "s"},
                      "r2", 5, 11, {"layer_0": np.ones(2)})
        assert broker.results("next", 0)["results"][0]["reason"] == "scorer_busy"
        broker.release("next")
    finally:
        broker.close()
        release.set()
    assert not broker.groups


@pytest.mark.parametrize("sites,native", [([0], "layer_0"), ([1], "layer_2")])
def test_public_final_norm_mapping_and_predictor_contract(tmp_path, monkeypatch, sites, native):
    opts = provider(tmp_path, monkeypatch, sites=sites)
    broker = ScoringBroker(opts, "model", 2)
    try:
        assert broker.layer_names == [native]
        broker.reserve("d", [0])
        broker.submit({"decision_id": "d", "branch_id": 0, "session_id": "s"},
                      "r", 4, 11, {native: np.array([2, 3])})
        assert broker.results("d", 1)["results"][0]["score"] == 2
    finally:
        broker.close()


@pytest.mark.parametrize("value", [np.array([np.nan]), np.array([1, 2]), np.array([[1]])])
def test_invalid_predictor_outputs_are_not_scores(tmp_path, monkeypatch, value):
    broker = ScoringBroker(provider(tmp_path, monkeypatch, lambda _: value), "model", 2)
    try:
        broker.reserve("d", [0])
        broker.submit({"decision_id": "d", "branch_id": 0, "session_id": "s"},
                      "r", 4, 11, {"layer_0": np.ones(2)})
        row = broker.results("d", 1)["results"][0]
        assert row["status"] == "failed" and "score" not in row
    finally:
        broker.close()


def test_artifact_changes_rejected_and_no_provider_loaded(tmp_path, monkeypatch):
    opts = provider(tmp_path, monkeypatch)
    assert artifact_contract(opts["estimator_ref"], opts["estimator_sha256"], "model")
    (tmp_path / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        artifact_contract(opts["estimator_ref"], opts["estimator_sha256"], "model")


def test_decision_immutable_and_branch_identity_validated(tmp_path):
    c = controller(tmp_path)
    md = capture_request_metadata("s", "t", "r", "rid", 0, ["think_close"])
    c.start(md)
    try:
        with pytest.raises(ValueError, match="identity"):
            c.bind("r", {**md, "decision_id": "d", "branch_id": 2, "branch_count": 2})
        c._write("sessions/s/parallel/example.json", {"selected": 0}, immutable=True)
        with pytest.raises(ValueError, match="Immutable"):
            c._write("sessions/s/parallel/example.json", {"selected": 1}, immutable=True)
    finally:
        c.close()


def test_parallel_policies_and_config_hash_compatibility(cfg):
    assert "parallel" not in cfg
    assert not active(cfg, 0)
    sample = {"parallel": {"enabled": True, "branch_count": 2, "policy": "first_step"}}
    assert active(sample, 0) and not active(sample, 1)
    sample["parallel"].update(policy="every_k_steps", every_k_steps=3)
    assert active(sample, 3) and not active(sample, 2)
    sample["parallel"]["branch_count"] = 1
    assert not active(sample, 0)
    with pytest.raises(ValueError, match="every_k_steps"):
        Parallel(policy="every_k_steps")
    import yaml

    path = Path(cfg["config_source"])
    original = resolve(path)
    assert original["config_hash"] == cfg["config_hash"]
    raw = yaml.safe_load(path.read_text())
    raw["parallel"] = {"enabled": True, "estimator_ref": "absent.json"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="online capture"):
        resolve(path)


def test_recording_model_executes_one_branch_and_clears_next_query(cfg, tmp_path, monkeypatch):
    pytest.importorskip("minisweagent")
    from litellm import ModelResponse
    from minisweagent.models.litellm_model import LitellmModel

    from rollout.model import make_model

    options = provider(tmp_path, monkeypatch)
    options["policy"] = "first_step"
    cfg = copy.deepcopy(cfg)
    cfg["parallel"] = options
    cfg["capture"].update(enabled=True, anchors=["think_close"])
    broker = ScoringBroker(options, "model", 2)
    c = controller(tmp_path)
    md = capture_request_metadata("s", "t", "run", "pending", 0, ["think_close"], close_token_ids=[11])
    c.start(md)
    physical_calls = []
    lock = threading.Lock()

    def query(self, messages, **kwargs):
        metadata = copy.deepcopy(kwargs.get("extra_body", self.config.model_kwargs["extra_body"])["dycon_capture"])
        branch = metadata.get("branch_id", 0)
        with lock:
            physical_calls.append(metadata)
        if metadata.get("decision_id"):
            rid = "native-" + str(branch)
            c.bind(rid, {**metadata, "prompt_token_ids": [10, 198]})
            c.record_output({"request_id": rid, "prompt_token_ids": [10, 198], "output_token_ids": [20, 11, 30]})
            arrays = {"layer_0": np.array([8 - branch * 6, 1])}
            c.record(rid, arrays, anchor="think_close", position=3, token_id=11, absolute_position=3)
            broker.submit(metadata, rid, 3, 11, arrays)
        return ModelResponse(model="fixture", choices=[{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "reasoning_content": "analysis",
                        "tool_calls": [{"id": "call-" + str(branch), "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"command": "echo " + str(branch)})}}]}}])

    monkeypatch.setattr(LitellmModel, "_query", query)
    model = make_model(cfg, SimpleNamespace(remaining=lambda n: n), md,
                       {"capture_controller": c, "parallel_client": BrokerClient(broker)})
    try:
        winner = model.query([{"role": "user", "content": "task"}])
        assert winner["extra"]["dycon_parallel"]["selected_branch_id"] == 1
        assert len(winner["extra"]["actions"]) == 1
        assert "1" in json.dumps(winner["extra"]["actions"])
        from rollout.agent import DurableAgent

        executed = []

        class Environment:
            def execute(self, action):
                executed.append(action)
                return {"output": "ok", "returncode": 0, "exception_info": ""}

            def get_template_vars(self):
                return {}

        agent = DurableAgent(model, Environment(), system_template="system", instance_template="{{ task }}")
        agent.execute_actions(winner)
        assert executed == winner["extra"]["actions"]
        assert len(executed) == 1
        assert len(physical_calls) == 2
        model.query([{"role": "user", "content": "next"}])
        assert len(physical_calls) == 3
        assert "decision_id" not in physical_calls[-1]
        assert "branch_id" not in physical_calls[-1]
    finally:
        c.close()
        broker.close()


@pytest.mark.parametrize("bad_branch", [0, 1, "all", "none"])
def test_candidate_format_isolation_ties_and_usage(tmp_path, bad_branch):
    pytest.importorskip("minisweagent")
    from minisweagent.exceptions import FormatError

    decisions = []

    class Client:
        def call(self, action, payload):
            if action != "results":
                return {}
            return {"results": [{"branch_id": b, "score": 1.0, "status": "scored"} for b in (0, 1)],
                    "estimator_sha256": "sha", "target": "target", "endpoint": "submission"}

    def record(value, timeout):
        decisions.append(value)
        return value

    options = {"branch_count": 2, "score_timeout_ms": 200, "fallback_policy": "first_valid"}
    cfg = {"parallel": options, "generation": {"seed": 1}, "capture": {"control_seconds": 1}}
    coordinator = ParallelReasoningCoordinator(cfg, SimpleNamespace(record_decision=record), Client())

    def parse(response):
        if bad_branch in (response.branch, "all"):
            raise FormatError({"role": "user", "content": "invalid", "extra": {"response": response.model_dump()}})
        return {"role": "assistant", "extra": {"actions": [{"branch": response.branch}]}}

    md = {"session_id": "s", "step_id": 0, "query_id": "q"}
    if bad_branch == "all":
        with pytest.raises(FormatError):
            coordinator.run([], md, lambda m, d, s: Completion(d["branch_id"]), parse, lambda _: {"cost": 2})
    else:
        winner = coordinator.run([], md, lambda m, d, s: Completion(d["branch_id"]), parse, lambda _: {"cost": 2})
        assert winner["extra"]["cost"] == 4
        expected = 1 if bad_branch == 0 else 0
        assert decisions[0]["selected_branch_id"] == expected
    assert len(decisions) == 1


def test_missing_rejected_close_reports_partial_without_breaking_winner(tmp_path):
    c = controller(tmp_path)
    base = capture_request_metadata("s", "t", "r", "pending", 0, ["think_close"], close_token_ids=[11])
    c.start(base)
    candidates = []
    try:
        for b in range(2):
            md = {**base, "decision_id": "d", "branch_id": b, "branch_count": 2, "api_attempt_id": str(b)}
            c.bind(f"r{b}", {**md, "prompt_token_ids": [10, 198]})
            c.record_output({"request_id": f"r{b}", "prompt_token_ids": [10, 198], "output_token_ids": [20, 11, 30]})
            candidates.append({"branch_id": b, "api_attempt_id": str(b), "response": {}, "score": None})
        c.record("r0", {"layer_0": np.ones(2)}, anchor="think_close", position=3, token_id=11, absolute_position=3)
        c.record_decision({"session_id": "s", "decision_id": "d", "branch_count": 2,
                           "selected_branch_id": 0, "candidates": candidates})
        path = tmp_path / "trajectory.json"
        write_json(path, {"messages": [{"role": "assistant", "extra": {"response": {
            "dycon_native_request_id": "r0", "prompt_token_ids": [10, 198], "output_token_ids": [20, 11, 30]}}}]})
        result = c.finalize("s", rollout_id="rid", trajectory_path=str(path),
                            trajectory_sha256=sha256(path), model_fingerprint="model")
        assert result["capture_status"] == "partial"
        missing = read_json(c.export_root / "sessions/s/parallel/branch-coverage.json")["missing"]
        assert len(missing) == 1 and missing[0]["branch_id"] == 1
    finally:
        c.close()


@pytest.mark.parametrize("fallback", ["first_valid", "stop"])
def test_score_transport_failure_fallback_or_persist_stop(fallback):
    import requests

    from rollout.parallel import ParallelSelectionError

    evidence = []

    class Client:
        def call(self, action, payload):
            if action == "results":
                raise requests.Timeout("score service timeout")
            return {}

    def record(value, timeout):
        evidence.append(value)
        return value

    cfg = {"parallel": {"branch_count": 2, "score_timeout_ms": 1, "fallback_policy": fallback},
           "generation": {"seed": 42}, "capture": {"control_seconds": 1}}
    coordinator = ParallelReasoningCoordinator(cfg, SimpleNamespace(record_decision=record), Client())
    arguments = ([], {"session_id": "s", "step_id": 0, "query_id": "q"},
                 lambda m, md, seed: Completion(md["branch_id"]),
                 lambda response: {"role": "assistant", "extra": {"actions": [response.branch]}},
                 lambda _: {"cost": 0})
    if fallback == "stop":
        with pytest.raises(ParallelSelectionError):
            coordinator.run(*arguments)
        assert evidence[0]["selected_branch_id"] is None
    else:
        assert coordinator.run(*arguments)["extra"]["actions"] == [0]
        assert evidence[0]["fallback_reason"] == "no_eligible_scored_branch"
    assert "Timeout" in evidence[0]["score_error"]


def test_parallel_plugin_http_scores_only_and_lifecycle(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from runtime.serving.vllm_plugin import DyConCapturePlugin

    broker = ScoringBroker(provider(tmp_path, monkeypatch), "model", 2)
    monkeypatch.setenv("DYCON_CAPTURE_EXPORT_OPTIONS", "{}")
    plugin = DyConCapturePlugin()
    plugin.scoring = broker
    async def collective_rpc(*args, **kwargs):
        return []

    plugin.engine = SimpleNamespace(collective_rpc=collective_rpc)
    app = fastapi.FastAPI()
    plugin.attach_router(app)
    try:
        with TestClient(app) as client:
            assert client.post("/dycon/parallel/reserve", json={"decision_id": "d", "branches": [0, 1]}).status_code == 200
            for b in (0, 1):
                broker.submit({"decision_id": "d", "branch_id": b, "session_id": "s"}, f"r{b}",
                              4, 11, {"layer_0": np.array([b, 1])})
            response = client.post("/dycon/parallel/results", json={"decision_id": "d", "timeout": 1})
            assert response.status_code == 200
            assert [row["score"] for row in response.json()["results"]] == [0, 1]
            assert "layer_0" not in response.text
            assert client.post("/dycon/parallel/release", json={"decision_id": "d"}).status_code == 200
            assert client.post("/dycon/parallel/results", json={"decision_id": "d", "timeout": 0}).status_code == 400
    finally:
        broker.close()


def test_b1_config_uses_old_path_without_artifact(cfg):
    import yaml

    path = Path(cfg["config_source"])
    raw = yaml.safe_load(path.read_text())
    raw["parallel"] = {"enabled": True, "branch_count": 1}
    path.write_text(yaml.safe_dump(raw))
    resolved = resolve(path)
    assert not active(resolved, 0)


def test_real_parallel_config_hashes_and_preserves_full_arguments(cfg, tmp_path, monkeypatch):
    import yaml

    from core.config import assert_frozen
    from runtime.serving.manager import ServingManager

    opts = provider(tmp_path, monkeypatch)
    path = Path(cfg["config_source"])
    raw = yaml.safe_load(path.read_text())
    raw["model"].update(path=str(tmp_path), fingerprint="model")
    raw["capture"].update(enabled=True, anchors=["think_close"], layers=[0, 1])
    raw["serving"]["management"] = "managed"
    raw["parallel"] = {"enabled": True, "branch_count": 2, "estimator_ref": opts["estimator_ref"],
                       "save_sites": [0], "max_inflight_branches": 2}
    path.write_text(yaml.safe_dump(raw))
    resolved = resolve(path)
    assert_frozen(resolved)
    assert resolved["parallel"]["estimator_sha256"] == opts["estimator_sha256"]
    before = copy.deepcopy(resolved)
    before.pop("parallel")
    a = ServingManager(before, tmp_path / "before").build_args("python", "127.0.0.1", 8101)
    b = ServingManager(resolved, tmp_path / "after").build_args("python", "127.0.0.1", 8101)
    assert a == b
    assert "FULL_AND_PIECEWISE" in " ".join(b)
    assert "--enforce-eager" not in b
    (tmp_path / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        resolve(path)


def test_parallel_action_receipts_keep_selection_separate(tmp_path, cfg):
    pytest.importorskip("minisweagent")
    from minisweagent.agents.default import DefaultAgent

    from rollout.agent import DurableAgent
    from rollout.model import make_model

    c = controller(tmp_path)
    md = capture_request_metadata("s", "t", "run", "pending", 0, ["think_close"])
    c.start(md)
    c.parallel_decisions["d"] = {"session_id": "s"}
    model = make_model(cfg, SimpleNamespace(remaining=lambda n: n), None, {"capture_controller": c})
    message = {"extra": {"actions": [1], "dycon_parallel": {
        "session_id": "s", "decision_id": "d", "selected_branch_id": 1}}}
    monkey = pytest.MonkeyPatch()
    monkey.setattr(DefaultAgent, "execute_actions", lambda self, msg: [{"role": "tool", "content": "ok"}])
    try:
        agent = object.__new__(DurableAgent)
        agent.model = model
        assert agent.execute_actions(message)[0]["content"] == "ok"
        base = c.export_root / "sessions/s/parallel/actions"
        assert read_json(base / "d-started.json")["selected_branch_id"] == 1
        assert read_json(base / "d-completed.json")["details"]["observations"] == 1
    finally:
        monkey.undo()
        c.close()


def test_existing_ridge_api_bridge_preserves_prediction(tmp_path):
    from difficulty.models.ridge_grid import predict_model, save_model
    from runtime.serving.ridge_provider import load

    rng = np.random.default_rng(42)
    model = {"coef": rng.normal(size=8), "intercept": np.array(2.0),
             "feature_mean": rng.normal(size=8), "feature_scale": np.ones(8) * 2}
    save_model(tmp_path / "model.npz", model, {})
    bridge = load(tmp_path, {"model_file": "model.npz", "files": {"model.npz": "hash"},
                             "sites": [31], "hidden_size": 8})
    batch = rng.normal(size=(4, 1, 8))
    np.testing.assert_array_equal(bridge.predict(batch), predict_model(model, batch[:, 0, :]))


def test_diagnostic_no_score_consumes_same_hidden_without_predictor(tmp_path, monkeypatch):
    def forbidden(batch):
        raise AssertionError("Unscored validation must not call predictor")

    broker = ScoringBroker(provider(tmp_path, monkeypatch, forbidden), "model", 2)
    try:
        broker.reserve("d", [0, 1])
        for branch in (0, 1):
            broker.submit({"session_id": "s", "decision_id": "d", "branch_id": branch,
                           "diagnostic_score_disabled": True}, f"r{branch}", 3, 11,
                          {"layer_0": np.ones(2)})
        rows = broker.results("d", 1)["results"]
        assert [r["status"] for r in rows] == ["unscored", "unscored"]
        assert all(r["sample_id"] for r in rows)
        assert broker.outstanding == 0
    finally:
        broker.close()


def test_admission_waits_for_whole_group_without_partial_slot_deadlock(tmp_path, monkeypatch):
    opts = provider(tmp_path, monkeypatch)
    opts["admission_timeout_seconds"] = 1
    broker = ScoringBroker(opts, "model", 2)
    entered, done = threading.Event(), threading.Event()
    try:
        broker.reserve("a", [0, 1])

        def reserve():
            entered.set()
            broker.reserve("b", [0, 1])
            done.set()

        thread = threading.Thread(target=reserve)
        thread.start()
        assert entered.wait(1)
        assert not done.is_set()
        assert "b" not in broker.groups
        broker.release("a")
        assert done.wait(1)
        thread.join(1)
        assert set(broker.groups) == {"b"}
    finally:
        broker.close()


def test_late_score_cannot_modify_reused_decision_identity(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def predict(batch):
        started.set()
        release.wait(2)
        return np.ones(len(batch)) * 7

    broker = ScoringBroker(provider(tmp_path, monkeypatch, predict), "model", 2)
    try:
        broker.reserve("d", [0])
        broker.submit({"session_id": "s", "decision_id": "d", "branch_id": 0}, "old", 3, 11,
                      {"layer_0": np.ones(2)})
        assert started.wait(1)
        broker.release("d")
        broker.reserve("d", [0])
        release.set()
        # Drain the finite worker queue before inspecting the new generation.
        broker.jobs.join()
        assert broker.results("d", 0)["results"] == [{"branch_id": 0, "status": "missing"}]
    finally:
        release.set()
        broker.close()
