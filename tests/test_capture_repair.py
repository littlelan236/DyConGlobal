"""Protocol, recovery and resource correctness; deliberately no speed tests."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from core.config import assert_frozen, load_snapshot, snapshot
from core.disk_budget import DiskBudget
from core.io import read_json, sha256, write_json
from hidden.artifacts import event_records, read_arrays, read_request, response_records
from hidden.backends.vllm.online import capture_request_metadata, tokenizer_newline_inventory
from runtime.serving.capture import OnlineCaptureController, _ExportItem
from runtime.serving.manager import ServingManager
from runtime.serving.spool import CaptureSpool, publish_files


def options(tmp_path):
    return dict(
        enabled=True,
        root="online",
        spool_root="spool",
        spool_max_bytes=1024**2 * 10,
        spool_high_watermark=0.8,
        spool_low_watermark=0.6,
        shard_max_bytes=1024 * 32,
        upload_workers=2,
        upload_max_elapsed_seconds=10,
        queue_size=32,
    )


def controller(tmp_path, **kwargs):
    config = dict(
        export=options(tmp_path),
        run_id="run",
        nas_root=str(tmp_path / "nas"),
        finalize_seconds=1,
    )
    config.update(kwargs)
    return OnlineCaptureController(
        tmp_path,
        config,
    )


def test_full_queue_spills_and_drains_without_loss(tmp_path):
    c = controller(tmp_path, export={**options(tmp_path), "queue_size": 1})
    started = threading.Event()
    release = threading.Event()
    original = c.queue._persist

    def blocked(item):
        started.set()
        assert release.wait(5)
        return original(item)

    c.queue._persist = blocked
    try:
        first = _ExportItem("first", {"layer_0": np.ones(8, dtype=np.float32)}, {"session_id": "s"})
        second = _ExportItem("second", {"layer_0": np.ones(8, dtype=np.float32)}, {"session_id": "s"})
        third = _ExportItem("third", {"layer_0": np.ones(8, dtype=np.float32)}, {"session_id": "s"})
        assert c.queue.submit(first)
        assert started.wait(5)
        assert c.queue.submit(second)
        assert c.queue.submit(third)
        assert list(c.queue.spill.glob("*.json"))
        release.set()
        c.queue.drain_session("s", 10)
        assert c.queue.status()["failed"] == []
        assert c.queue.status()["spill_files"] == 0
        status = c.queue.status()
        assert status["submitted"] == 3 and status["persisted"] == 3
        assert status["spilled_total"] >= 1 and status["pending_total"] == 0
        assert status["queue_high_water"] >= 1 and status["spill_high_water"] >= 1
        assert (c.queue.root / "raw" / "first.npz").exists()
        assert (c.queue.root / "raw" / "second.npz").exists()
        assert (c.queue.root / "raw" / "third.npz").exists()
    finally:
        release.set()
        c.close()


def test_plugin_reasoning_filter_excludes_prompt_and_answer_events():
    from runtime.serving.capture import accept_reasoning_event

    state = {"prompt_length": 10, "last_position": -1, "close_seen": False}
    stats = {"prompt": 0, "after_close": 0, "duplicate": 0}
    assert not accept_reasoning_event(state, 9, "reasoning_newline", stats)
    assert accept_reasoning_event(state, 10, "reasoning_newline", stats)
    assert accept_reasoning_event(state, 11, "think_close", stats)
    assert not accept_reasoning_event(state, 12, "reasoning_newline", stats)
    assert stats["prompt"] == 1
    assert stats["after_close"] == 1


def test_frozen_config_and_resume(cfg):
    original = copy.deepcopy(cfg)
    snapshot(cfg, "r", {"task_keys": []})
    cfg["serving"]["tool_parser"] = "changed"
    with pytest.raises(ValueError, match="modified"):
        assert_frozen(cfg)
    assert load_snapshot(cfg["project_root"], "r")["serving"] == original["serving"]


def test_official_rollout_rejects_overrides():
    from core.cli import main

    with pytest.raises(SystemExit):
        main(["--set", "serving.tool_parser=hermes", "rollout", "r"])


def test_four_modes_share_public_args(cfg, tmp_path):
    payload = tmp_path / "payload.json"
    write_json(
        payload, {"model_fingerprint": "fixture-model", "steering": {"vectors": [{"algorithm": "direct"}]}}
    )
    common = None
    for cap, steer in [(False, False), (True, False), (False, True), (True, True)]:
        cfg["capture"]["enabled"] = cap
        cfg["steering"].update(enabled=steer, payload=str(payload) if steer else None)
        args = ServingManager(cfg, tmp_path).build_args("python", "localhost", 8100)
        assert args.count("--enable-steer-vector") == int(cap or steer)
        filtered = []
        iterator = iter(args)
        for arg in iterator:
            if arg in {
                "--compilation-config",
                "--worker-extension-cls",
                "--steer-algorithms",
                "--steer-graph-mode",
                "--steer-vector-dtype",
            }:
                next(iterator)
            elif arg != "--enable-steer-vector":
                filtered.append(arg)
        if common is None:
            common = filtered
        assert common == filtered


@pytest.mark.parametrize("policy,expected", [("all", [1, 2, 3]), ("double_only", [2])])
def test_inventory_policy(policy, expected):
    class Tokenizer:
        vocab_size = 4

        def decode(self, ids, **kwargs):
            return ["x", "\n", "\n\n", "x\n"][ids[0]]

    assert tokenizer_newline_inventory(Tokenizer(), policy=policy)[0] == expected


def test_anchor_subset_selector():
    metadata = capture_request_metadata(
        "s", "t", "r", "rid", 0, ["think_close"], newline_token_ids=[198, 271], close_token_ids=[11]
    )
    assert metadata["capture_select"] == {"tact_states": {"generation_tokens": [11]}}


def test_32k_ledger_dedup_tar_and_nas(tmp_path):
    c = controller(tmp_path)
    try:
        prompt = [10] * 32768
        md = capture_request_metadata(
            "s", "t", "run", "pending", 0, ["reasoning_newline"], newline_token_ids=[271]
        )
        c.start(md)
        c.bind("native", {**md, "prompt_token_ids": prompt})
        c.record_output(
            {"request_id": "native", "prompt_token_ids": prompt, "output_token_ids": [271] * 8 + [11]}
        )
        for i in range(8):
            c.record(
                "native",
                {"layer_0": np.arange(32, dtype=np.float32)},
                anchor="reasoning_newline",
                token_id=271,
                absolute_position=len(prompt) + i,
                position=len(prompt) + i,
            )
        c.queue.queue.join()
        c.spool.seal("s")
        assert c.spool.publish_session("s", 10)
        files = list((c.spool.nas / "requests").glob("*.json"))
        assert len(files) == 1
        ledger = read_json(files[0])
        assert ledger["prompt_token_ids"] == prompt
        assert len(list((c.spool.nas / "shards").glob("*.tar"))) == 1
        for event, row in event_records(c.spool.nas):
            assert len(json.dumps(row)) < 4000
            assert "prompt_token_ids" not in row and "capture_select" not in row
            assert np.array_equal(read_arrays(c.spool.nas, row)["layer_0"], np.arange(32))
        assert not list((c.export_root / "raw").glob("*.npz"))
    finally:
        c.close()


@pytest.mark.parametrize("disabled", [True, False])
def test_export_toggle(tmp_path, disabled):
    opts = options(tmp_path)
    opts["enabled"] = not disabled
    c = OnlineCaptureController(tmp_path, {"export": opts, "nas_root": str(tmp_path / "nas"), "run_id": "r"})
    try:
        md = capture_request_metadata("s", "t", "r", "rid", 0, ["reasoning_newline"])
        c.start(md)
        c.bind("n", md)
        c.record("n", {"layer_0": np.ones(2)}, anchor="reasoning_newline", token_id=271, position=3)
        c.queue.queue.join()
        assert bool(list(c.export_root.rglob("*.npz"))) != disabled
        if disabled:
            assert not (tmp_path / "hidden/manifest.json").exists()
    finally:
        c.close()


def test_upload_idempotence_ack_and_conflict(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write_json(src / "a.json", {"x": 1})
    files = {"a.json": sha256(src / "a.json")}
    publish_files(src, dst, files, 10)
    publish_files(src, dst, files, 10)  # rename succeeded but caller lost ACK
    assert sha256(dst / "a.json") == files["a.json"]
    write_json(dst / "a.json", {"x": 2})
    with pytest.raises(ValueError, match="conflict"):
        publish_files(src, dst, files, 10)


def test_interrupted_part_and_bad_hash_keep_source(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write_json(src / "a.json", {"x": 1})
    write_json(dst / "a.json.part", {"x": "truncated"})
    files = {"a.json": sha256(src / "a.json")}
    publish_files(src, dst, files, 10)
    assert sha256(dst / "a.json") == files["a.json"]
    assert (src / "a.json").exists()


def test_multi_run_byte_bound_and_watermark(tmp_path):
    a = DiskBudget(tmp_path, tmp_path / "spool", 100)
    b = DiskBudget(tmp_path, tmp_path / "spool", 100)

    def acquire(i):
        try:
            [a, b][i % 2].acquire(str(i), 10)
            return True
        except BufferError:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(acquire, range(20))) == 10
    assert a.usage() == 100
    released = threading.Event()

    def admit():
        a.admit(0.8, 0.6, 2)
        released.set()

    thread = threading.Thread(target=admit)
    thread.start()
    for key in list(a.allocations())[:5]:
        a.release(key)
    thread.join(2)
    assert released.is_set()


def test_disk_budget_does_not_apply_shared_reserve_to_capture(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    space = SimpleNamespace(free=300)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: space)
    budget = DiskBudget(tmp_path, tmp_path / "spool", 1000)
    budget.acquire("image", 150, kind="image")
    space.free = 100
    budget.acquire("capture", 60)
    assert budget.usage() == 60
    with pytest.raises(BufferError, match="capture project spool byte limit reached"):
        budget.acquire("capture-over-limit", 950)


def test_format_error_response_join_and_ledger_refs(tmp_path):
    c = controller(tmp_path)
    try:
        md = capture_request_metadata("s", "t", "run", "pending", 0, ["reasoning_newline"])
        c.start(md)
        messages = []
        for i, role in enumerate(["assistant", "user", "assistant"]):
            metadata = {**md, "step_id": i, "query_id": str(i), "api_attempt_id": "a" + str(i)}
            c.bind("native-" + str(i), {**metadata, "prompt_token_ids": [10, 198]})
            c.record_output(
                {
                    "request_id": "native-" + str(i),
                    "prompt_token_ids": [10, 198],
                    "output_token_ids": [271, 11, 30],
                }
            )
            messages.append(
                {
                    "role": role,
                    "extra": {"request": {"dycon_capture": metadata}, "response": {"id": "native-" + str(i)}},
                }
            )
        path = tmp_path / "trajectory.json"
        write_json(path, {"messages": messages})
        assert c.prepare_trajectory("s", path)["responses"] == 3
        source = read_json(path)
        for _, _, response in response_records(source):
            assert "prompt_token_ids" not in response
            assert read_request(c.export_root, response)["output_token_ids"] == [271, 11, 30]
        assert (
            c.finalize(
                "s",
                rollout_id="rid",
                trajectory_path=str(path),
                trajectory_sha256=sha256(path),
                model_fingerprint="m",
            )["capture_status"]
            == "complete"
        )
    finally:
        c.close()


@pytest.mark.parametrize("policy", ["record_and_continue", "block_rollout"])
def test_missing_receipt_policy(tmp_path, policy):
    c = controller(tmp_path, failure_policy=policy)
    c.config["finalize_seconds"] = 0.01
    try:
        md = capture_request_metadata("s", "t", "run", "pending", 0, ["think_close"])
        c.start(md)
        path = tmp_path / "trajectory.json"
        write_json(
            path,
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": ["keep-original"],
                        "extra": {
                            "request": {"dycon_capture": {"query_id": "missing"}},
                            "response": {"id": "missing"},
                        },
                    }
                ]
            },
        )
        if policy == "block_rollout":
            with pytest.raises(RuntimeError, match="CaptureError"):
                c.prepare_trajectory("s", path)
        else:
            assert c.prepare_trajectory("s", path)["capture_failures"]
        assert read_json(path)["messages"][0]["tool_calls"] == ["keep-original"]
    finally:
        c.close()


def test_empty_canonical_capture_reports_auditable_partial_reason(tmp_path):
    c = controller(tmp_path)
    try:
        c.start(capture_request_metadata('s', 't', 'run', 'pending', 0, ['think_close']))
        path = tmp_path / 'empty-trajectory.json'
        write_json(path, {'messages': []})
        result = c.finalize('s', rollout_id='rid', trajectory_path=str(path),
                            trajectory_sha256=sha256(path), model_fingerprint='m')
        assert result['capture_status'] == 'partial'
        assert result['capture_diagnostics']['reason_codes'] == ['no_canonical_model_responses']
    finally:
        c.close()


def test_query_does_not_wait_for_capture_and_keeps_format_error(cfg, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("minisweagent")
    import litellm
    from minisweagent.exceptions import FormatError
    from minisweagent.models.litellm_model import LitellmModel

    from core.deadline import Deadline
    from rollout.model import make_model

    cfg["capture"]["enabled"] = True
    md = capture_request_metadata("s", "t", "run", "pending", 0, ["think_close"], close_token_ids=[11])
    responses = iter([False, True])

    def query(*args, **kwargs):
        valid = next(responses)
        message = {"role": "assistant", "content": "saved-raw-response"}
        if valid:
            message["tool_calls"] = [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ]
        return litellm.ModelResponse(
            id="response-id",
            choices=[{"index": 0, "finish_reason": "stop", "message": message}],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    monkeypatch.setattr(LitellmModel, "_query", query)
    # No capture URL exists: a synchronous receipt query would fail this test.
    model = make_model(cfg, Deadline(), md)
    with pytest.raises(FormatError) as error:
        model.query([{"role": "user", "content": "task"}])
    extra = error.value.messages[0]["extra"]
    assert extra["response"]["choices"][0]["message"]["content"] == "saved-raw-response"
    first = extra["request"]["dycon_capture"]
    second = model.query([{"role": "user", "content": "task"}])["extra"]["request"]["dycon_capture"]
    assert first["query_id"] != second["query_id"]
    assert first["api_attempt_id"] != second["api_attempt_id"]


def test_recovery_from_raw_before_seal(tmp_path):
    spool = CaptureSpool(tmp_path, options(tmp_path), "run", tmp_path / "nas")
    raw = spool.root / "raw/event.npz"
    raw.parent.mkdir()
    np.savez_compressed(raw, layer_0=np.ones(3))
    row = {
        "sample_id": "event",
        "session_id": "s",
        "tensor": "raw/event.npz",
        "sha256": sha256(raw),
        "artifact_state": "local_persisted",
    }
    spool.write_json("index/event.json", row)
    try:
        spool.recover()
        assert spool.publish_session("s", 10)
        events = list(event_records(spool.nas))
        assert len(events) == 1
        assert np.array_equal(read_arrays(spool.nas, events[0][1])["layer_0"], np.ones(3))
        assert not raw.exists()
    finally:
        spool.close(10)


def test_failed_upload_preserves_local_then_recovers_without_model(tmp_path, monkeypatch):
    from concurrent.futures import Future

    spool = CaptureSpool(tmp_path, options(tmp_path), "run", tmp_path / "nas")
    original = spool.pool.submit

    def offline(*args):
        future = Future()
        future.set_exception(OSError("NAS temporarily unavailable"))
        return future

    monkeypatch.setattr(spool.pool, "submit", offline)
    raw = spool.root / "raw/event.npz"
    raw.parent.mkdir()
    np.savez_compressed(raw, layer_0=np.ones(3))
    row = {
        "sample_id": "event",
        "session_id": "s",
        "tensor": "raw/event.npz",
        "sha256": sha256(raw),
        "artifact_state": "local_persisted",
    }
    spool.write_json("index/event.json", row)
    try:
        spool.add(row)
        spool.seal("s")
        assert not spool.publish_session("s", 0.1)
        assert raw.exists()
        assert not list((spool.nas / "shards").glob("*.tar"))
        monkeypatch.setattr(spool.pool, "submit", original)
        spool.recover()
        assert spool.publish_session("s", 10)
        assert not raw.exists()
        assert read_arrays(spool.nas, next(event_records(spool.nas))[1])["layer_0"].sum() == 3
    finally:
        spool.close(10)


def test_cross_process_admission_wakes_on_release(tmp_path):
    import subprocess
    import sys

    budget = DiskBudget(tmp_path, tmp_path / "spool", 100)
    budget.acquire("persisted", 90)
    command = "from core.disk_budget import DiskBudget;import sys;DiskBudget(sys.argv[1],sys.argv[2],100).release('persisted')"
    child = subprocess.Popen([sys.executable, "-c", command, str(tmp_path), str(tmp_path / "spool")])
    budget.admit(0.8, 0.6, 5)
    assert child.wait(timeout=5) == 0
    assert budget.usage() == 0


@pytest.mark.parametrize("policy", ["record_and_continue", "block_rollout"])
def test_plugin_receipt_failure_does_not_become_api_500(tmp_path, monkeypatch, policy):
    pytest.importorskip("starlette")
    import asyncio
    from types import SimpleNamespace

    from runtime.serving.vllm_plugin import NATIVE_CONTEXT, REQUEST_CONTEXT, DyConCapturePlugin

    plugin = DyConCapturePlugin()
    plugin.staging = tmp_path / "staging"
    plugin.staging.mkdir()
    plugin.failure_policy = policy

    def disconnected(*args):
        raise ConnectionError("injected control-plane failure")

    monkeypatch.setattr(plugin, "_post", disconnected)
    output = SimpleNamespace(
        request_id="native",
        prompt_token_ids=[10, 198],
        outputs=[SimpleNamespace(token_ids=[20, 11, 30], finish_reason="stop")],
        num_cached_tokens=0,
        finished=True,
    )

    async def generate(*args, **kwargs):
        await plugin.engine._add_request(SimpleNamespace(request_id="native", prompt_token_ids=[10, 198]))
        yield output

    called = []

    async def add(*args, **kwargs):
        called.append(True)

    plugin.engine = SimpleNamespace(
        input_processor=SimpleNamespace(process_inputs=lambda: None), _add_request=add, generate=generate
    )
    plugin._attach_admission()

    async def run():
        token = REQUEST_CONTEXT.set({"session_id": "s", "anchors": ["think_close"]})
        native_token = NATIVE_CONTEXT.set({})
        try:
            return [
                row
                async for row in plugin.engine.generate(
                    {}, SimpleNamespace(output_kind=SimpleNamespace(name="FINAL_ONLY"))
                )
            ]
        finally:
            NATIVE_CONTEXT.reset(native_token)
            REQUEST_CONTEXT.reset(token)

    assert asyncio.run(run()) == [output]
    assert called == [True]
    assert len(plugin.failures) == 2


def test_block_policy_saves_response_without_api_resampling(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("minisweagent")
    from types import SimpleNamespace

    import litellm
    from minisweagent.models.litellm_model import LitellmModel

    from core.deadline import Deadline
    from rollout.model import make_model
    from runtime.serving.errors import CaptureError

    cfg["capture"].update(enabled=True, failure_policy="block_rollout")
    failures, calls = [], []
    controller = SimpleNamespace(
        sessions={"s": SimpleNamespace(options={"failures": failures})},
        export_root=tmp_path,
        queue=SimpleNamespace(status=lambda: {"failed": []}),
    )

    def query(*args, **kwargs):
        calls.append(True)
        failures.append("receipt failure after native completion")
        return litellm.ModelResponse(
            id="r",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "preserve me",
                        "tool_calls": [
                            {
                                "id": "c",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                            }
                        ],
                    },
                }
            ],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    monkeypatch.setattr(LitellmModel, "_query", query)
    md = capture_request_metadata("s", "t", "run", "rid", 0, ["think_close"], close_token_ids=[11])
    model = make_model(cfg, Deadline(), md, {"capture_controller": controller})
    with pytest.raises(CaptureError) as error:
        model.query([{"role": "user", "content": "task"}])
    assert len(calls) == 1
    assert error.value.response_message["content"] == "preserve me"


def test_training_reader_reuses_published_tar_and_task_partition(cfg, tmp_path):
    from difficulty.training.train import load_training_data

    c = controller(tmp_path)
    try:
        md = capture_request_metadata("s", "t", "run", "rid", 0, ["reasoning_newline"])
        c.start(md)
        c.bind("n", md)
        c.record(
            "n", {"layer_0": np.array([1.0, 2.0, 3.0])}, anchor="reasoning_newline", token_id=271, position=3
        )
        c.queue.queue.join()
        assert c.spool.publish_session("s", 10)
        event = next(event_records(c.spool.nas))[1]
        index = Path(cfg["project_root"]) / "difficulty/datasets/published/index.json"
        write_json(
            index,
            {
                "schema_version": 3,
                "hidden_root": str(c.spool.nas),
                "samples": [
                    {
                        **event,
                        "partition": "val",
                        "labels": {"remaining_steps": 4, "remaining_reasoning_tokens": 20},
                    }
                ],
            },
        )
        features, targets, ids = load_training_data(cfg, "published", "remaining_steps_linear", "val")
        assert np.array_equal(features, np.array([[[1.0, 2.0, 3.0]]]))
        assert targets.tolist() == [4]
        assert ids == [event["sample_id"]]
        with pytest.raises(ValueError, match="requested task partition"):
            load_training_data(cfg, "published", "remaining_steps_linear", "train")
    finally:
        c.close()


def test_disk_budget_transaction_rollback_and_legacy_import(tmp_path, monkeypatch):
    import os

    spool = tmp_path / "spool"
    spool.mkdir()
    write_json(
        tmp_path / "state/disk-budget.json",
        {
            "capture:old": {
                "bytes": 10,
                "kind": "capture",
                "device": spool.stat().st_dev,
                "path": str(spool),
                "pid": os.getpid(),
            }
        },
    )
    budget = DiskBudget(tmp_path, spool, 100)
    assert budget.usage() == 10
    assert DiskBudget(tmp_path, spool, 100).usage() == 10
    original = budget._put

    def interrupted(*args):
        original(*args)
        raise OSError("injected transaction interruption")

    monkeypatch.setattr(budget, "_put", interrupted)
    with pytest.raises(OSError, match="transaction interruption"):
        budget.acquire("new", 20)
    assert budget.usage() == 10
    assert set(budget.allocations()) == {"capture:old"}
