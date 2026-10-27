"""Actual plugin hooks: user cancellation differs from loop abort and complete output."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('starlette')
from runtime.serving.vllm_plugin import DyConCapturePlugin
from tests.test_reasoning_loop_plugin import Engine, collect


def plugin(monkeypatch):
    monkeypatch.setenv('DYCON_CAPTURE_EXPORT_OPTIONS', '{}')
    p = DyConCapturePlugin()
    p.engine = Engine(finished_at=256)
    posts, releases = [], []
    p._safe_post = lambda route, value, metadata: posts.append((route, value))
    p.scoring = SimpleNamespace(release_branch=lambda *args: releases.append(args), cancel=lambda *args: None)
    return p, posts, releases


def close_lanes(p):
    for lane in (p.wait_lane, p.receipt_lane, p.publish_lane, p.control_lane):
        lane.close()


@pytest.mark.parametrize('phase', ['binding', 'submitting', 'running'])
def test_cancel_race_around_native_add(monkeypatch, phase):
    p, posts, releases = plugin(monkeypatch)
    entered, proceed = threading.Event(), threading.Event()
    if phase == 'binding':
        original_post = p._safe_post
        def slow_bind(route, value, md):
            if route.endswith('/bind'):
                entered.set()
                assert proceed.wait(3)
            return original_post(route, value, md)
        p._safe_post = slow_bind
    elif phase == 'submitting':
        async def add(request, *args, **kwargs):
            entered.set()
            while not proceed.is_set():
                await asyncio.sleep(0)
        p.engine._add_request = add
    p._attach_admission()
    async def run():
        task = asyncio.create_task(collect(p))
        async with asyncio.timeout(3):
            while not entered.is_set() if phase != 'running' else not p.native_requests:
                await asyncio.sleep(0)
            if phase == 'running':
                while not any(s['phase'] == 'running' for s in p.native_requests.values()):
                    await asyncio.sleep(0)
            report = await p.cancel_native('d')
            assert report['cancel_requested']
            assert report['abort_completed'] is False
            proceed.set()
            if phase == 'binding':
                from core.cancellation import Cancelled
                with pytest.raises(Cancelled):
                    await task
                assert not p.engine.aborts
                assert not any(path.endswith('/output') for path, _ in posts)
            else:
                outputs = await task
                assert outputs[-1].outputs[0].finish_reason == 'abort'
                assert 'loop-native' in p.engine.aborts
                output = next(value for path, value in posts if path.endswith('/output'))
                assert 'reasoning_loop' not in output
            assert not p.native_requests
            assert len(releases) == 1
    try:
        asyncio.run(run())
    finally:
        proceed.set()
        close_lanes(p)


def test_cancel_endpoint_requires_execution_and_control_token(monkeypatch):
    fastapi, httpx = pytest.importorskip('fastapi'), pytest.importorskip('httpx')
    p, _, _ = plugin(monkeypatch)
    p.cancel_token, p.owner_run, p.owner_attempt = 'test-secret', 'owned-run', '003'
    app = fastapi.FastAPI()
    p.attach_router(app)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture') as client:
            body = {'run_id': 'owned-run', 'execution_attempt_id': '003'}
            assert (await client.post('/dycon/capture/engine/cancel', json=body)).status_code == 403
            assert (await client.post('/dycon/capture/engine/cancel', json={**body, 'run_id': 'other'},
                                     headers={'X-DyCon-Cancel-Token': 'test-secret'})).status_code == 403
            result = await client.post('/dycon/capture/engine/cancel', json=body,
                                       headers={'X-DyCon-Cancel-Token': 'test-secret'})
            assert result.status_code == 200 and p.cancelled_run
    try:
        asyncio.run(run())
    finally:
        close_lanes(p)


def test_receipt_deadline_retains_real_native_completion(monkeypatch, tmp_path):
    p, _, releases = plugin(monkeypatch)
    p.control_seconds = .03
    p.staging = tmp_path / 'staging'
    release = threading.Event()
    p._safe_post = lambda *args: release.wait(2)
    p._attach_admission()
    try:
        outputs = asyncio.run(collect(p))
        assert outputs[-1].finished and outputs[-1].outputs[0].finish_reason == 'length'
        assert len(releases) == 1
        assert p.failures and all('total deadline' in f['reason'] for f in p.failures)
        assert list((tmp_path / 'failures').glob('*.json'))
    finally:
        release.set()
        close_lanes(p)
