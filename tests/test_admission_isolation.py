"""Real ASGI routes and broker under 4x32 / capacity64 pressure (CPU only)."""
import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from runtime.serving.parallel import ScoringBroker


def broker_fixture(monkeypatch, *, dynamic=True):
    import runtime.serving.parallel as module
    contract = {'sites': [0], 'loader': 'isolation_fixture:load', 'hidden_size': 2,
                'target': 'fixture', 'endpoint': 'submission'}
    monkeypatch.setattr(module, 'artifact_contract', lambda *args: contract)
    monkeypatch.setitem(sys.modules, 'isolation_fixture', SimpleNamespace(
        load=lambda *args: SimpleNamespace(predict=lambda batch: np.zeros(len(batch)))))
    options = {'estimator_ref': 'fixture/manifest.json', 'estimator_sha256': 'fixture',
               'branch_count': 32, 'max_inflight_branches': 64, 'dynamic_branch_admission': dynamic,
               'admission_timeout_seconds': 10, 'decision_ttl_seconds': 20}
    return ScoringBroker(options, 'fixture', 2)


@pytest.mark.parametrize('workers', [1, 2, 8, 32])
def test_real_routes_complete_ten_oversubscribed_rounds(monkeypatch, workers):
    fastapi, httpx = pytest.importorskip('fastapi'), pytest.importorskip('httpx')
    from runtime.serving.vllm_plugin import DyConCapturePlugin
    monkeypatch.setenv('DYCON_PARALLEL_OPTIONS', json.dumps({'max_inflight_branches': 64}))
    broker = broker_fixture(monkeypatch)
    plugin = DyConCapturePlugin()
    plugin.scoring = broker
    app = fastapi.FastAPI()
    plugin.attach_router(app)

    async def run():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=workers))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture') as client:
            async def post(action, payload):
                result = await client.post('/dycon/parallel/' + action, json=payload)
                assert result.status_code == 200, result.text
                return result.json()
            for iteration in range(10):
                names = [f'{iteration}-{i}' for i in range(4)]
                for name in names:
                    await post('reserve', {'decision_id': name, 'branches': list(range(32))})
                for name in names[:2]:
                    for branch in range(32):
                        await post('branch-acquire', {'decision_id': name, 'branch_id': branch})
                waiting = [asyncio.create_task(post('branch-acquire', {'decision_id': name, 'branch_id': branch}))
                           for name in names[2:] for branch in range(32)]
                async with asyncio.timeout(5):
                    while plugin.wait_lane.status()['outstanding'] < 64:
                        await asyncio.sleep(0)
                    assert sum(len(g['active']) for g in broker.groups.values()) == 64
                    assert sum(len(g['pending']) for g in broker.groups.values()) == 64
                    # Receipt callback and HTTP release cannot depend on admission workers.
                    assert await plugin.receipt_lane.run(lambda: 'receipt') == 'receipt'
                    await asyncio.gather(*(post('branch-release', {'decision_id': name, 'branch_id': branch})
                                           for name in names[:2] for branch in range(32)))
                    values = await asyncio.gather(*waiting)
                    assert all(value['acquired'] for value in values)
                    for name in names[2:]:
                        for branch in range(32):
                            await post('branch-release', {'decision_id': name, 'branch_id': branch})
                    for name in names:
                        await post('release', {'decision_id': name})
                    assert not broker.groups
            await plugin.shutdown()
    try:
        asyncio.run(run())
    finally:
        broker.close()
        for lane in (plugin.wait_lane, plugin.receipt_lane, plugin.publish_lane, plugin.control_lane):
            lane.close()


@pytest.mark.parametrize('operation', ['acquire', 'reserve', 'results'])
def test_cancel_wakes_actual_broker_wait_without_releasing_native(monkeypatch, operation):
    from core.cancellation import Cancelled
    broker = broker_fixture(monkeypatch, dynamic=operation != 'reserve')
    broker.cfg['max_inflight_branches'] = 1
    broker.reserve('occupied', [0])
    if operation != 'reserve':
        broker.acquire_branch('occupied', 0)
    if operation == 'acquire':
        broker.reserve('waiting', [0])
    started = threading.Event()
    def wait():
        started.set()
        if operation == 'acquire':
            return broker.acquire_branch('waiting', 0)
        if operation == 'reserve':
            return broker.reserve('waiting', [0])
        return broker.results('occupied', 10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wait)
        assert started.wait(1)
        broker.cancel('occupied' if operation == 'results' else 'waiting')
        with pytest.raises(Cancelled):
            future.result(timeout=1)
        assert len(broker.groups['occupied']['active']) == 1
    broker.close()


def test_lane_cancellation_keeps_permit_until_real_thread_exits():
    from contextvars import ContextVar

    from runtime.serving.execution import ExecutionLane
    lane = ExecutionLane('bounded-test', 1, 1)
    entered, release = threading.Event(), threading.Event()
    variable = ContextVar('test_context', default='missing')
    def blocking():
        assert variable.get() == 'copied'
        entered.set()
        release.wait(2)
    async def run():
        variable.set('copied')
        task = asyncio.create_task(lane.run(blocking))
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lane.status()['outstanding'] == 1
        with pytest.raises(RuntimeError, match='queue full'):
            await lane.run(lambda: None)
        release.set()
        async with asyncio.timeout(2):
            while lane.status()['outstanding']:
                await asyncio.sleep(0)
        lane.close()
    try:
        asyncio.run(run())
    finally:
        release.set()
        lane.close()


def test_results_close_wakes_waiter(monkeypatch):
    broker = broker_fixture(monkeypatch)
    broker.reserve('d', [0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(broker.results, 'd', 10)
        broker.close()
        with pytest.raises((RuntimeError, KeyError)):
            future.result(timeout=1)
