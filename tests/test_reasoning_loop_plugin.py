"""Native request abort and ledger behavior through the serving plugin hook."""

import asyncio
import copy
import sys
from enum import Enum
from types import SimpleNamespace

import pytest

from core.config import ParallelReasoningLoop


class Kind(Enum):
    FINAL_ONLY = 1
    CUMULATIVE = 2
    DELTA = 3


class Params:
    n = 1
    output_kind = Kind.FINAL_ONLY
    stream_interval = None

    def clone(self):
        return copy.copy(self)


class Engine:
    def __init__(self, *, error=False, finished_at=256, with_close=False, active=True):
        self.input_processor = SimpleNamespace(process_inputs=lambda *a, **kw: None)
        self.aborts, self.params, self.closed = [], {}, []
        self.error, self.finished_at = error, finished_at
        self.with_close, self.active = with_close, active

    async def _add_request(self, request, *args, **kwargs):
        pass

    async def abort(self, native_id, internal=False):
        assert internal
        if self.error:
            raise RuntimeError("abort failed")
        self.aborts.append(native_id)

    async def generate(self, prompt, sampling_params, request_id):
        native = request_id + "-native"
        await self._add_request(SimpleNamespace(request_id=native, prompt_token_ids=[248068, 198] if self.active else [3]))
        self.params[request_id] = sampling_params
        ids = []
        try:
            while True:
                await asyncio.sleep(0)
                aborted = native in self.aborts
                if not aborted:
                    ids += [1] * 8 if request_id != "healthy" else list(range(len(ids), len(ids) + 8))
                    if self.with_close and len(ids) == 32:
                        ids[-1] = 248069
                finished = aborted or len(ids) >= self.finished_at
                if sampling_params.output_kind == Kind.FINAL_ONLY and not finished:
                    continue
                output_ids = ids[-8:] if sampling_params.output_kind == Kind.DELTA and not aborted else (
                    [] if sampling_params.output_kind == Kind.DELTA else list(ids))
                yield SimpleNamespace(request_id=request_id, prompt_token_ids=[248068, 198], finished=finished,
                    outputs=[SimpleNamespace(token_ids=output_ids, finish_reason="abort" if aborted else (
                        "length" if finished else None))])
                if finished:
                    return
        finally:
            self.closed.append(native)


def plugin_fixture(monkeypatch, **engine_kwargs):
    pytest.importorskip("starlette")
    from runtime.serving.vllm_plugin import DyConCapturePlugin

    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(RequestOutputKind=Kind))
    monkeypatch.setenv("DYCON_CAPTURE_EXPORT_OPTIONS", "{}")
    p = DyConCapturePlugin()
    p.parallel_options = {"reasoning_loop": ParallelReasoningLoop(enabled=True, min_generated_tokens=64,
        repeat_count=4, min_repeated_tokens=32, max_period_tokens=8, check_interval_tokens=8).model_dump()}
    p.engine = Engine(**engine_kwargs)
    posts, terminal = [], []
    p._safe_post = lambda path, payload, metadata: posts.append((path, payload))
    p.scoring = SimpleNamespace(mark_terminated=lambda *args: terminal.append(args))
    p._attach_admission()
    return p, posts, terminal


async def collect(p, name="loop", metadata=None, params=None, keywords=False):
    from runtime.serving.vllm_plugin import LOOP_CONTEXT, REQUEST_CONTEXT

    md = {"session_id": "s", "decision_id": "d", "branch_id": 0,
          "api_attempt_id": name, "anchors": ["think_close"]} if metadata is None else metadata
    token = REQUEST_CONTEXT.set(md)
    try:
        stream = p.engine.generate(prompt={}, sampling_params=params or Params(), request_id=name) if keywords else (
            p.engine.generate({}, params or Params(), name))
        result = [output async for output in stream]
        assert LOOP_CONTEXT.get() is None
        return result
    finally:
        REQUEST_CONTEXT.reset(token)


@pytest.mark.parametrize("keywords", [False, True])
@pytest.mark.parametrize("kind", [Kind.FINAL_ONLY, Kind.DELTA])
def test_internal_incremental_output_aborts_only_bound_branch(monkeypatch, keywords, kind):
    p, posts, terminal = plugin_fixture(monkeypatch)
    params = Params()
    params.output_kind = kind

    async def run():
        return await asyncio.gather(collect(p, params=params, keywords=keywords), collect(p, "healthy"))

    loop, healthy = asyncio.run(run())
    assert p.engine.aborts == ["loop-native"]
    assert loop[-1].outputs[0].finish_reason == "abort"
    assert healthy[-1].outputs[0].finish_reason == "length"
    assert params.output_kind == kind and params.stream_interval is None
    assert p.engine.params["loop"].output_kind == (Kind.CUMULATIVE if kind == Kind.FINAL_ONLY else kind)
    assert sorted(p.engine.closed) == ["healthy-native", "loop-native"]
    ledgers = [data for path, data in posts if path.endswith("/output")]
    stopped = next(data for data in ledgers if data["request_id"] == "loop")
    assert stopped["output_token_ids"] == [1] * 64
    assert stopped["reasoning_loop"]["abort_completed"] is True
    assert stopped["reasoning_loop"]["native_request_id"] == "loop-native"
    assert len(terminal) == 1
    assert "reasoning_loop" not in next(data for data in ledgers if data["request_id"] == "healthy")


@pytest.mark.parametrize("mode", ["disabled", "ordinary", "replay", "finished"])
def test_unguarded_and_already_finished_requests_are_not_aborted(monkeypatch, mode):
    p, posts, terminal = plugin_fixture(monkeypatch, finished_at=64 if mode == "finished" else 256)
    metadata = {"session_id": "s", "decision_id": "d", "branch_id": 0, "api_attempt_id": "a"}
    if mode == "disabled":
        p.parallel_options = {}
    if mode == "ordinary":
        metadata.pop("decision_id")
    if mode == "replay":
        metadata["replay"] = True
    asyncio.run(collect(p, metadata=metadata))
    assert p.engine.aborts == []
    assert terminal == []
    assert "reasoning_loop" not in posts[-1][1]


def test_abort_failure_cleans_up_and_is_not_reported_as_completed(monkeypatch):
    p, posts, terminal = plugin_fixture(monkeypatch, error=True)
    with pytest.raises(RuntimeError, match="abort failed"):
        asyncio.run(collect(p))
    assert p.engine.closed == ["loop-native"]
    assert not terminal
    assert not any(path.endswith("/output") for path, _ in posts)


@pytest.mark.parametrize("kwargs", [{"with_close": True}, {"active": False}])
def test_action_phase_and_closed_prompt_are_outside_reasoning_guard(monkeypatch, kwargs):
    p, _, terminal = plugin_fixture(monkeypatch, **kwargs)
    asyncio.run(collect(p))
    assert not p.engine.aborts
    assert not terminal


def test_disconnect_closes_native_generator_and_request_context(monkeypatch):
    p, _, _ = plugin_fixture(monkeypatch)

    async def run():
        from runtime.serving.vllm_plugin import LOOP_CONTEXT, REQUEST_CONTEXT

        token = REQUEST_CONTEXT.set({"session_id": "s", "decision_id": "d", "branch_id": 0,
                                     "api_attempt_id": "a"})
        try:
            stream = p.engine.generate({}, Params(), "loop")
            await anext(stream)
            await stream.aclose()
            assert LOOP_CONTEXT.get() is None
        finally:
            REQUEST_CONTEXT.reset(token)

    asyncio.run(run())
    assert p.engine.closed == ["loop-native"]
