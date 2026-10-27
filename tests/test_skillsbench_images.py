import shutil
import socket
import socketserver
import threading
from pathlib import Path

import pytest
import yaml

from core.io import sha256, write_json
from runtime.images.registry import Registry
from runtime.images.skillsbench import (
    _command,
    _context_hash,
    build_proxy,
    effective_dockerfile,
    inventory,
    source_images,
    verified_wheel_context,
)


def fixture_source(tmp_path, monkeypatch, *, compose=False):
    root = tmp_path / 'benchmark'
    env = root / 'tasks' / 'document-task' / 'environment'
    env.mkdir(parents=True)
    (env / 'Dockerfile').write_text('FROM ubuntu:24.04\nCOPY input.txt /input.txt\n')
    (env / 'input.txt').write_text('frozen input')
    (env.parent / 'task.md').write_text('---\nschema_version: "1.3"\nsandbox:\n  cpus: 1\n---\nGenerate an output.')
    if compose:
        (env / 'Dockerfile.api').write_text('FROM node:20-slim\n')
        (env / 'docker-compose.yaml').write_text(yaml.safe_dump({'services': {
            'main': {'build': {'context': '${CONTEXT_DIR}'}},
            'api': {'build': {'context': '.', 'dockerfile': 'Dockerfile.api'}}}}))
    excluded = root / 'tasks-extra' / 'excluded' / 'environment'
    excluded.mkdir(parents=True)
    (excluded / 'Dockerfile').write_text('FROM scratch\n')
    monkeypatch.setattr('runtime.images.skillsbench.subprocess.check_output',
                        lambda args, **kwargs: '' if 'status' in args else 'a' * 40)
    return {'project_root': str(tmp_path), 'skillsbench_images': {
        'source_root': 'benchmark', 'revision': 'a' * 40}}


def test_inventory_excludes_extra_and_binds_inputs(tmp_path, monkeypatch):
    cfg = fixture_source(tmp_path, monkeypatch)
    plan = inventory(cfg)
    assert len(plan['tasks']) == len(plan['units']) == 1
    assert plan['units'][0]['task_id'] == 'document-task'
    before = plan['units'][0]['context_sha256']
    (Path(plan['source_root']) / 'tasks/document-task/environment/input.txt').write_text('changed')
    assert inventory(cfg)['units'][0]['context_sha256'] != before


def test_compose_sidecar_is_an_independent_image(tmp_path, monkeypatch):
    cfg = fixture_source(tmp_path, monkeypatch, compose=True)
    plan = inventory(cfg)
    assert len(plan['tasks']) == 1 and len(plan['units']) == 2
    assert {x['service'] for x in plan['units']} == {'main', 'api'}


def test_revision_mismatch_is_rejected(tmp_path, monkeypatch):
    cfg = fixture_source(tmp_path, monkeypatch)
    cfg['skillsbench_images']['revision'] = 'b' * 40
    with pytest.raises(ValueError, match='pinned revision'):
        inventory(cfg)


def test_dirty_tasks_are_rejected(tmp_path, monkeypatch):
    cfg = fixture_source(tmp_path, monkeypatch)
    monkeypatch.setattr('runtime.images.skillsbench.subprocess.check_output',
                        lambda args, **kwargs: ' M tasks/document-task/task.md' if 'status' in args else 'a' * 40)
    with pytest.raises(ValueError, match='local modifications'):
        inventory(cfg)


def test_compose_context_cannot_escape_package(tmp_path, monkeypatch):
    cfg = fixture_source(tmp_path, monkeypatch, compose=True)
    env = tmp_path / 'benchmark/tasks/document-task/environment'
    (env / 'docker-compose.yaml').write_text('services:\n  main:\n    build: ../../../../\n')
    with pytest.raises(ValueError, match='escapes'):
        inventory(cfg)


def test_context_hash_tracks_symlink_targets(tmp_path):
    (tmp_path / 'data').write_text('content')
    (tmp_path / 'link').symlink_to('data')
    first = _context_hash(tmp_path)
    (tmp_path / 'link').unlink()
    (tmp_path / 'link').symlink_to('other')
    assert first != _context_hash(tmp_path)


def test_buildx_client_auth_uses_host_proxy(monkeypatch):
    observed = {}
    monkeypatch.setenv('ALL_PROXY', 'socks5://unrelated.invalid:1234')
    def run(command, **kwargs):
        observed.update(kwargs)
        return 'builder\n'
    monkeypatch.setattr('runtime.images.skillsbench.subprocess.check_output', run)
    cfg = {'environment': {'docker_executable': '/usr/bin/docker', 'docker_host': 'unix:///test.sock'},
           'skillsbench_images': {'host_proxy': 'http://127.0.0.1:7890',
                                 'build_proxy': 'http://192.0.2.1:7897'}}
    assert _command(cfg, ['buildx', 'ls']) == 'builder'
    assert observed['env']['HTTPS_PROXY'] == 'http://127.0.0.1:7890'
    assert observed['env']['http_proxy'] == 'http://127.0.0.1:7890'
    assert 'ALL_PROXY' not in observed['env']


def test_source_images_resolves_defaults_and_ignores_stage_aliases():
    text = '# syntax=docker/dockerfile:1.4\nARG tag=pin\nFROM bugswarm/cached-images:${tag} AS base\nFROM base AS final\n'
    assert source_images(text) == ['docker/dockerfile:1.4', 'bugswarm/cached-images:pin']
    assert source_images(text, {'tag': 'override'})[-1] == 'bugswarm/cached-images:override'


def test_download_profile_keeps_original_instructions_and_valid_shell():
    text = 'FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y python3\n'
    result = effective_dockerfile(text, {'ubuntu_mirror': 'http://mirrors.example/ubuntu'})
    assert result.endswith('RUN apt-get update && apt-get install -y python3\n')
    assert result.count('FROM ubuntu:24.04') == 1
    assert "'(' -name '*.list'" in result
    assert 'Acquire::Retries 5;' in result
    assert effective_dockerfile(text, {}) == text


def test_public_registry_bearer_challenge_and_range():
    class Response:
        def __init__(self, status=200, headers=None, body=None):
            self.status_code, self.headers, self.body = status, headers or {}, body
        def close(self):
            pass
        def raise_for_status(self):
            assert self.status_code == 200
        def json(self):
            return self.body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    responses = [Response(401, {'WWW-Authenticate': 'Bearer realm="https://gcr.io/v2/token",service="gcr.io"'}),
                 Response(body={'token': 'anonymous-public-token'}), Response()]
    calls = []
    class Session:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return responses.pop(0)
        def close(self):
            pass
    registry = Registry(registry='gcr.io')
    registry.session.close()
    registry.session = Session()
    registry.get('oss-fuzz-base/base-builder-python', 'blobs/sha256:test', stream=True, offset=123)
    assert calls[-1][1]['headers']['Authorization'] == 'Bearer anonymous-public-token'
    assert calls[-1][1]['headers']['Range'] == 'bytes=123-'
    assert calls[1][1]['params']['scope'] == 'repository:oss-fuzz-base/base-builder-python:pull'


def test_python_heredoc_import_is_not_a_docker_from_instruction():
    text = "FROM python:3.12-slim\nRUN python - <<'PY'\nfrom transformers import AutoModel\nfrom typing import Any\nPY\nRUN pip install x\n"
    assert source_images(text) == ['python:3.12-slim']
    staged = effective_dockerfile(text, {'debian_mirror': 'http://mirrors.example/debian',
                                       'python_index': 'https://mirrors.example/simple'})
    assert staged.count('RUN if [ $(id -u)') == 1
    assert 'from transformers import AutoModel\nfrom typing import Any\nPY\n' in staged


@pytest.mark.skipif(shutil.which('socat') is None, reason='socat is not installed')
def test_owned_proxy_forwards_bytes_and_releases_its_port(tmp_path):
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            data = self.request.recv(1024)
            if data:
                self.request.sendall(data)
    upstream = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Echo)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    cfg = {'skillsbench_images': {'manage_build_proxy': True,
           'build_proxy': f'http://127.0.0.1:{port}',
           'host_proxy': f'http://127.0.0.1:{upstream.server_address[1]}'}}
    try:
        with build_proxy(cfg, tmp_path):
            with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                client.sendall(b'opaque TLS-like payload')
                assert client.recv(1024) == b'opaque TLS-like payload'
        with pytest.raises(OSError):
            socket.create_connection(('127.0.0.1', port), timeout=0.2)
        # A previous connection's TIME_WAIT must not block a resumable job.
        with build_proxy(cfg, tmp_path):
            with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                client.sendall(b'resumed')
                assert client.recv(1024) == b'resumed'
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_verified_wheel_context_rejects_corruption(tmp_path):
    nas = tmp_path / 'nas'
    downloads = nas / 'store/downloads/wheels'
    downloads.mkdir(parents=True)
    wheel = downloads / 'torch-2.14.1-py3-none-any.whl'
    wheel.write_bytes(b'checksum-checked fixture')
    item = {'package': 'torch', 'version': '2.14.1', 'filename': wheel.name,
            'path': str(wheel.relative_to(nas)), 'sha256': sha256(wheel)}
    write_json(wheel.with_name(wheel.name + '.source.json'),
               {**item, 'url': 'https://files.pythonhosted.org/official-wheel'})
    cfg = {'paths': {'nas_root': str(nas)},
           'skillsbench_images': {'verified_wheels': {'target': [item]}}}
    unit = {'task_id': 'target', 'key': 'target--main'}
    context, receipts = verified_wheel_context(cfg, nas / 'store', unit)
    assert receipts[0]['sha256'] == item['sha256']
    assert (context / 'constraints.txt').read_text() == (
        'torch @ file:///opt/dycon-verified-wheels/' + wheel.name + '\n')
    wheel.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verified_wheel_context(cfg, nas / 'store', unit)


def test_verified_wheel_mount_does_not_enter_heredoc_body():
    text = "FROM python:3.12-slim\nRUN pip install torch\nRUN python - <<'PY'\nfrom typing import Any\nPY\n"
    staged = effective_dockerfile(text, {'debian_mirror': 'http://mirror.example/debian',
        'python_index': 'https://pypi.org/simple', 'mount_verified_wheels': True})
    assert 'ARG PIP_CONSTRAINT' in staged
    assert staged.count('from=dycon_verified_wheels') == 2
    assert 'from typing import Any\nPY\n' in staged


def test_bootstrap_precedes_original_install_without_changing_its_requirements():
    text = 'FROM python:3.12-slim\nRUN pip install seisbench==0.10.2\n'
    staged = effective_dockerfile(text, {'python_index': 'https://pypi.org/simple',
        'compatibility': {'python_bootstrap': ['setuptools==80.9.0']}})
    assert staged.index('python3 -m pip install --no-cache-dir setuptools==80.9.0') < staged.index('RUN pip install seisbench==0.10.2')
    assert 'RUN pip install seisbench==0.10.2\n' in staged


def test_tar_ownership_adjustment_is_an_exact_task_scoped_replacement():
    text = 'FROM python:3.12-slim\nRUN tar -xzf /root/data.tar.gz -C /root/traces\n'
    profile = {'replacements': [{'before': 'tar -xzf /root/data.tar.gz',
                                'after': 'tar --no-same-owner -xzf /root/data.tar.gz'}]}
    assert 'tar --no-same-owner -xzf' in effective_dockerfile(text, {'compatibility': profile})
    with pytest.raises(ValueError, match='exactly once'):
        effective_dockerfile(text, {'compatibility': {'replacements': [{'before': 'different command', 'after': 'changed'}]}})


def test_coursier_proxy_flags_follow_the_single_configured_proxy():
    text = 'FROM ubuntu:24.04\nRUN ./cs setup --yes\n'
    staged = effective_dockerfile(text, {'build_proxy': 'http://192.0.2.4:17899',
        'compatibility': {'replacements': [{'before': './cs setup --yes',
            'after': './cs {build_proxy_jvm_args} setup --yes'}]}})
    assert '-J-Dhttps.proxyHost=192.0.2.4' in staged
    assert '-J-Dhttps.proxyPort=17899' in staged
    assert '{build_proxy_jvm_args}' not in staged
