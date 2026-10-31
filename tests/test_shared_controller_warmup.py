import asyncio
import sys
from types import ModuleType, SimpleNamespace

from runtime.serving.vllm_plugin import DyConCapturePlugin, REQUEST_CONTEXT


def test_replica_warmups_share_controller_without_reusing_session(monkeypatch):
    vllm = ModuleType('vllm')
    vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    params = ModuleType('vllm.sampling_params')
    params.RequestOutputKind = SimpleNamespace(FINAL_ONLY='final')
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', params)
    sessions, stopped, generated = {}, set(), set()

    def post(route, value):
        if route.endswith('/start'):
            metadata = value['metadata']
            sid = metadata['session_id']
            assert sid not in sessions, 'Shared controller rejects duplicate sessions'
            sessions[sid] = metadata
        else:
            sid = route.split('/')[-2]
            assert sid in sessions
            stopped.add(sid)

    async def lane_run(function, *args):
        return function(*args)

    async def generate(prompt, params, request_id):
        metadata = REQUEST_CONTEXT.get()
        assert metadata['session_id'] == request_id
        assert metadata['task_id'] == 'engine-warmup'
        generated.add(request_id)
        await asyncio.sleep(0)
        yield None

    async def drain():
        return None

    def plugin():
        item = DyConCapturePlugin.__new__(DyConCapturePlugin)
        item.engine = SimpleNamespace(generate=generate)
        item.receipt_lane = SimpleNamespace(run=lane_run)
        item._post = post
        item.newline_ids, item.close_ids = {1}, {2}
        item.drain_once = drain
        return item

    async def exercise():
        first, second = plugin(), plugin()
        await asyncio.gather(first._warmup_capture_graphs(), second._warmup_capture_graphs())
        await first._warmup_capture_graphs()  # A restarted engine is distinct too.

    asyncio.run(exercise())
    assert len(sessions) == 3
    assert set(sessions) == stopped == generated
    assert REQUEST_CONTEXT.get() is None
