import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from core.io import hash_json, run_path, write_json


def test_deep_snapshot_verification_keeps_event_loop_responsive(tmp_path, monkeypatch):
    from runtime.agents import dependencies
    release = threading.Event()
    owner_thread = threading.get_ident()
    (tmp_path / 'manifest.json').write_text('{}')

    def verify(path):
        assert threading.get_ident() != owner_thread
        assert release.wait(timeout=2), 'event loop was blocked by snapshot hashing'
        return {'verified': True}

    monkeypatch.setattr(dependencies, 'verify_snapshot', verify)

    async def execute():
        async def concurrent_io():
            await asyncio.sleep(0.01)
            release.set()
        result, _ = await asyncio.gather(dependencies.verify_runtime_snapshot(
            {'dependency_snapshot': str(tmp_path)}), concurrent_io())
        return result

    assert asyncio.run(execute()) == {'verified': True}


def test_async_snapshot_check_retains_frozen_manifest_binding(tmp_path, monkeypatch):
    from runtime.agents import dependencies
    (tmp_path / 'manifest.json').write_text('{}')
    monkeypatch.setattr(dependencies, 'verify_snapshot', lambda path: {})
    with pytest.raises(ValueError, match='frozen execution context'):
        asyncio.run(dependencies.verify_runtime_snapshot({'dependency_snapshot': str(tmp_path),
                                                         'dependency_snapshot_sha256': '0' * 64}))


@pytest.mark.parametrize('cached_false_positive', [False, True])
def test_post_install_failure_cannot_pass_or_reuse_success_receipt(cfg, tmp_path, monkeypatch, cached_false_positive):
    from runtime.agents import skillsbench
    contract = {'policy': 'test-environment-integrity'}
    monkeypatch.setattr(skillsbench, 'environment_contract', lambda *args: contract)
    cfg['execution_context'] = {'profile': {}}
    task = {'instance_id': 'audit-fixture', 'instance': {'task_path': str(tmp_path)}}
    output = None

    class Factory:
        def __init__(self, *args):
            self.images = {('audit-fixture', 'main'): {'tag': 'fixture', 'identity': {}}}
            self.cpu_quota_supported = True
            self.client = SimpleNamespace(containers=SimpleNamespace(list=lambda **kwargs: []))
        def lock_images(self, *args): pass
        def cleanup_owned(self, *args): pass

    class Planes:
        def __init__(self, inner, cfg, run, sample, attempt, image, path):
            nonlocal output
            output = Path(path)
            self.ownership = {'epoch': 'test-epoch'}

    class Environment:
        async def exec(self, *args, **kwargs):
            raise RuntimeError('post-install inventory failed')

    class Rollout:
        _env = Environment()
        @staticmethod
        async def create(config): return Rollout()
        async def setup(self): pass
        async def start(self): pass
        async def install_agent(self):
            for name in ('dependency-install.json', 'skills-visibility.json', 'agent-environment.json', 'compose-image-verification.json'):
                write_json(output / name, {'verified': True, 'return_code': 0})
        async def cleanup(self):
            write_json(output / 'sandbox-cleanup.json', {'complete': True})

    monkeypatch.setattr(skillsbench, 'SkillsBenchFactory', Factory)
    monkeypatch.setattr(skillsbench, 'OwnedPlanes', Planes)
    monkeypatch.setattr(skillsbench, 'agent_rollout_config', lambda *args, **kwargs: None)
    monkeypatch.setattr(skillsbench, 'Rollout', Rollout)
    base = run_path(cfg, 'audit-integrity') / 'validation/environments/audit-fixture' / hash_json(contract)
    if cached_false_positive:
        write_json(base / 'result.json', {'contract': contract, 'verified': True,
                                         'error': 'previous post-install failure', 'evidence_sha256': {}})
    result = asyncio.run(skillsbench._audit_task_environment(cfg, 'audit-integrity', 'fixture', task))
    assert result['verified'] is False
    assert result['error'] == "RuntimeError('post-install inventory failed')"
    assert result['remaining_containers'] == []
    assert result['attempt_id'].startswith('environment-')


def test_concurrent_snapshot_consumers_share_only_the_running_full_check(tmp_path, monkeypatch):
    from runtime.agents import dependencies
    (tmp_path / 'manifest.json').write_text('{}')
    payload = tmp_path / 'payload'
    payload.write_text('original')
    started, release = threading.Event(), threading.Event()
    calls = []

    def verify(path):
        calls.append(path)
        started.set()
        assert release.wait(timeout=2)
        if payload.read_text() != 'original':
            raise ValueError('changed payload')
        return {'verified': True}

    monkeypatch.setattr(dependencies, 'verify_snapshot', verify)
    profile = {'dependency_snapshot': str(tmp_path)}

    async def execute():
        checks = [asyncio.create_task(dependencies.verify_runtime_snapshot(profile)) for _ in range(4)]
        assert await asyncio.to_thread(started.wait, 2)
        await asyncio.sleep(0)
        assert len(calls) == 1
        release.set()
        assert await asyncio.gather(*checks) == [{'verified': True}] * 4
        payload.write_text('changed')
        with pytest.raises(ValueError, match='changed payload'):
            await dependencies.verify_runtime_snapshot(profile)
        assert len(calls) == 2

    asyncio.run(execute())


def test_cancelling_one_snapshot_waiter_does_not_cancel_another(tmp_path, monkeypatch):
    from runtime.agents import dependencies
    (tmp_path / 'manifest.json').write_text('{}')
    started, release = threading.Event(), threading.Event()
    calls = []

    def verify(path):
        calls.append(path)
        started.set()
        assert release.wait(timeout=2)
        return {'verified': True}

    monkeypatch.setattr(dependencies, 'verify_snapshot', verify)

    async def execute():
        profile = {'dependency_snapshot': str(tmp_path)}
        first, second = [asyncio.create_task(dependencies.verify_runtime_snapshot(profile)) for _ in range(2)]
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second == {'verified': True}
        assert len(calls) == 1

    asyncio.run(execute())
