import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Environment, resolve
from core.io import read_json
from core.locking import file_lock
from runtime import docker_storage as storage


def test_optional_setting_preserves_legacy_config_shape(cfg):
    assert 'docker_data_root' not in cfg['environment']


@pytest.mark.parametrize('path', ['relative/docker', '/', '/data1/../docker', ''])
def test_rejects_unsafe_config_paths(cfg, path):
    with pytest.raises(ValueError, match='absolute non-root'):
        Environment.model_validate({**cfg['environment'], 'docker_data_root': path})


def test_root_mismatch_fails_before_any_image_load(cfg):
    from runtime.images.nas import ImageLeaseManager

    cfg = copy.deepcopy(cfg)
    cfg['environment']['docker_data_root'] = '/data1/docker'
    client = SimpleNamespace(info=lambda: {'DockerRootDir': '/old/docker'})
    with pytest.raises(RuntimeError, match='docker-storage apply'):
        ImageLeaseManager(cfg, client)


def test_storage_switch_cannot_take_lock_during_container_lease(cfg):
    with storage.storage_lease(cfg):
        with pytest.raises(BlockingIOError):
            with file_lock(Path(cfg['project_root']) / 'state/docker-storage.lock', nonblocking=True):
                pytest.fail('Storage switching must not interrupt an active image lease')


def test_new_container_lease_refused_during_maintenance(cfg):
    with file_lock(Path(cfg['project_root']) / 'state/docker-storage.lock'):
        with pytest.raises(RuntimeError, match='maintenance'):
            with storage.storage_lease(cfg):
                pytest.fail('A new lease must not enter during maintenance')


@pytest.fixture
def engine(cfg, monkeypatch):
    cfg = copy.deepcopy(cfg)
    root = Path(cfg['project_root'])
    old = root / 'old-docker'
    old.mkdir()
    (old / 'keep.txt').write_text('old Docker data')
    target = root / 'new-docker'
    cfg['environment'].update(docker_data_root=str(target),
                              docker_host=f'unix:///run/user/{os.getuid()}/docker.sock')
    config = root / 'user-config/docker/daemon.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'log-driver': 'json-file'}))
    state = {'root': str(old), 'calls': [], 'containers': [], 'volumes': [], 'fail_start': False}
    def info():
        return {'DockerRootDir': state['root'], 'Driver': 'overlayfs',
                'Containers': len(state['containers']), 'ContainersRunning': 0,
                'SecurityOptions': ['name=rootless']}
    client = SimpleNamespace(info=info, close=lambda: None,
                             containers=SimpleNamespace(list=lambda **kw: state['containers']),
                             volumes=SimpleNamespace(list=lambda: state['volumes']))
    def systemctl(action):
        state['calls'].append(action)
        if action == 'start':
            if state['fail_start']:
                state['fail_start'] = False
                raise RuntimeError('simulated startup failure')
            state['root'] = read_json(config).get('data-root', str(old)) if config.exists() else str(old)
    monkeypatch.setattr(storage, '_daemon_config', lambda: config)
    monkeypatch.setattr(storage, '_client', lambda cfg: client)
    monkeypatch.setattr(storage, '_systemctl', systemctl)
    monkeypatch.setattr(storage.subprocess, 'run', lambda *a, **kw: None)
    return cfg, state, config, old, target


def test_apply_preserves_old_data_merges_config_and_is_idempotent(engine):
    cfg, state, config, old, target = engine
    result = storage.apply(cfg)
    assert result['changed'] and result['matches']
    assert read_json(config) == {'log-driver': 'json-file', 'data-root': str(target)}
    assert (old / 'keep.txt').read_text() == 'old Docker data'
    assert state['calls'] == ['is-active', 'stop', 'start']
    assert not storage.apply(cfg)['changed']
    assert state['calls'] == ['is-active', 'stop', 'start']


def test_apply_rolls_back_config_and_service_on_start_failure(engine):
    cfg, state, config, old, target = engine
    before = config.read_bytes()
    state['fail_start'] = True
    with pytest.raises(RuntimeError, match='simulated startup'):
        storage.apply(cfg)
    assert config.read_bytes() == before
    assert state['root'] == str(old)
    assert (old / 'keep.txt').exists()
    receipts = list((Path(cfg['project_root']) / 'state/docker-storage').glob('*/rollback.json'))
    assert len(receipts) == 1 and read_json(receipts[0])['restored']


@pytest.mark.parametrize('resource', ['containers', 'volumes'])
def test_apply_refuses_existing_resources_without_service_changes(engine, resource):
    cfg, state, config, old, target = engine
    state[resource] = [object()]
    with pytest.raises(RuntimeError, match='exist'):
        storage.apply(cfg)
    assert state['calls'] == [] and not target.exists()


def test_apply_refuses_active_run_before_service_changes(engine):
    cfg, state, config, old, target = engine
    with file_lock(Path(cfg['project_root']) / 'state/run-locks/active.lock'):
        with pytest.raises(RuntimeError, match='Active project run'):
            storage.apply(cfg)
    assert state['calls'] == []


def test_apply_refuses_populated_destination(engine):
    cfg, state, config, old, target = engine
    target.mkdir()
    (target / 'existing').write_text('preserve')
    with pytest.raises(ValueError, match='must be empty'):
        storage.apply(cfg)
    assert 'stop' not in state['calls']
    assert (target / 'existing').read_text() == 'preserve'


def test_apply_removes_generated_config_on_rollback_when_original_absent(engine):
    cfg, state, config, old, target = engine
    config.unlink()
    state['fail_start'] = True
    with pytest.raises(RuntimeError, match='simulated startup'):
        storage.apply(cfg)
    assert not config.exists() and state['root'] == str(old)
