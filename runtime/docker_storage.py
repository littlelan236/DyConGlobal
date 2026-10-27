"""Rootless Docker storage maintenance driven by the editable project config."""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from core.io import now, write_json
from core.locking import file_lock


def assert_data_root(cfg, info):
    expected = cfg['environment'].get('docker_data_root')
    if expected is not None and Path(info['DockerRootDir']).resolve() != Path(expected).resolve():
        raise RuntimeError(
            f"Docker data-root mismatch: configured={expected}, actual={info['DockerRootDir']}. "
            "Stop active tasks, then run dycon docker-storage apply from the editable root config."
        )


@contextmanager
def storage_lease(cfg):
    """Block storage switching while any project image/container lease is active."""
    import fcntl

    path = Path(cfg['project_root']) / 'state/docker-storage.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Docker storage maintenance is active; retry after it completes') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _client(cfg):
    import docker

    return docker.DockerClient(base_url=cfg['environment']['docker_host'], timeout=30)


def _daemon_config():
    return Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'docker/daemon.json'


def status(cfg):
    client = _client(cfg)
    try:
        info = client.info()
        expected = cfg['environment'].get('docker_data_root')
        return {
            'configured_data_root': expected,
            'actual_data_root': info['DockerRootDir'],
            'matches': expected is None or Path(expected).resolve() == Path(info['DockerRootDir']).resolve(),
            'daemon_config': str(_daemon_config()),
            'docker_host': cfg['environment']['docker_host'],
            'storage_driver': info['Driver'],
            'containers': info['Containers'],
            'running': info['ContainersRunning'],
        }
    finally:
        client.close()


def _systemctl(action):
    return subprocess.run(['systemctl', '--user', action, 'docker.service'],
                          check=True, capture_output=True, text=True, timeout=120)


def _assert_idle(cfg, client, stack):
    if client.containers.list(all=True):
        raise RuntimeError('Docker containers exist; finish tasks and explicitly remove/preserve them before switching')
    if client.volumes.list():
        raise RuntimeError('Docker volumes exist; preserve their data before switching to an empty data-root')
    for path in sorted((Path(cfg['project_root']) / 'state/run-locks').glob('*.lock')):
        try:
            stack.enter_context(file_lock(path, nonblocking=True))
        except BlockingIOError:
            raise RuntimeError(f'Active project run prevents Docker storage switching: {path.stem}') from None


def apply(cfg):
    """Apply data-root, restart the user service and roll back on failure; never copy/delete Docker data."""
    expected = cfg['environment'].get('docker_data_root')
    if not expected:
        raise ValueError('Set environment.docker_data_root in the root config.yaml first')
    target = Path(expected)
    if not target.is_absolute() or target.resolve() == Path('/') or '..' in target.parts:
        raise ValueError('Docker data-root must be an absolute non-root path')
    expected_socket = f'unix:///run/user/{os.getuid()}/docker.sock'
    if cfg['environment']['docker_host'] != expected_socket:
        raise ValueError(f'Automatic apply supports only this user rootless socket: {expected_socket}')
    path = _daemon_config()
    lock = Path(cfg['project_root']) / 'state/docker-storage.lock'
    with file_lock(lock, nonblocking=True), ExitStack() as stack:
        client = _client(cfg)
        try:
            info = client.info()
            if 'name=rootless' not in info.get('SecurityOptions', []):
                raise ValueError('Automatic apply requires rootless Docker')
            if Path(info['DockerRootDir']).resolve() == target.resolve():
                result = status(cfg)
                return {**result, 'changed': False}
            _assert_idle(cfg, client, stack)
            _systemctl('is-active')
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.stat().st_uid != os.getuid():
                raise ValueError('Docker data-root must belong to the current rootless user')
            if any(target.iterdir()):
                raise ValueError('New Docker data-root must be empty; no existing data is overwritten')
            original = path.read_bytes() if path.exists() else None
            merged = json.loads(original) if original is not None else {}
            merged['data-root'] = str(target)
            audit = Path(cfg['project_root']) / 'state/docker-storage' / uuid.uuid4().hex
            audit.mkdir(parents=True)
            candidate = audit / 'daemon.proposed.json'
            write_json(candidate, merged)
            if original is not None:
                (audit / 'daemon.before.json').write_bytes(original)
            write_json(audit / 'before.json', {
                'at': now(), 'original_config_exists': original is not None,
                'old_data_root': info['DockerRootDir'], 'new_data_root': str(target),
                'config_source': cfg.get('config_source'), 'config_hash': cfg.get('config_hash'),
            })
            dockerd = str(Path(cfg['environment']['docker_executable']).with_name('dockerd'))
            subprocess.run([dockerd, '--validate', '--config-file', str(candidate)],
                           check=True, capture_output=True, text=True, timeout=30)
        finally:
            client.close()
        # No workloads are restarted implicitly. The original directory remains intact.
        _systemctl('stop')
        try:
            write_json(path, merged)
            _systemctl('start')
            result = status(cfg)
            if not result['matches']:
                raise RuntimeError(f'Docker ignored requested data-root: {result}')
        except BaseException as exc:
            try:
                _systemctl('stop')
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(original)
                _systemctl('start')
                write_json(audit / 'rollback.json', {'at': now(), 'error': repr(exc), 'restored': True})
            except BaseException as rollback_exc:
                raise RuntimeError(f'Docker storage apply failed ({exc!r}); rollback failed ({rollback_exc!r}); backup={audit}') from exc
            raise
        result.update(changed=True, audit=str(audit), previous_data_root=info['DockerRootDir'])
        write_json(audit / 'result.json', {**result, 'at': now()})
        return result
