import json
import zipfile
from pathlib import Path

import pytest

from runtime.containers.factory import ContainerFactory
from runtime.containers.policy import EnvironmentPolicy
from runtime.wheelhouse import build, verify


def make_wheel(directory, version='1.0', extra=''):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'demo-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'demo-{version}.dist-info/METADATA',
                         f'Name: demo\nVersion: {version}\nRequires-Python: >=3.6\n\n{extra}')
    return path


def test_union_preserves_versions_and_deduplicates_identical_content(cfg, tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    first = make_wheel(a)
    b.mkdir()
    (b / first.name).write_bytes(first.read_bytes())
    make_wheel(b, '2.0')
    result = build(cfg, 'union-fixture', [a, b])
    assert result['wheel_count'] == 2
    assert verify(result['path'])['status'] == 'verified'
    manifest = json.loads((Path(result['path']) / 'manifest.json').read_text())
    assert len(manifest['wheels'][first.name]['sources']) == 2
    with pytest.raises(FileExistsError):
        build(cfg, 'union-fixture', [a])


def test_conflicting_same_filename_is_rejected_without_publication(cfg, tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    make_wheel(a)
    make_wheel(b, extra='different content')
    with pytest.raises(ValueError, match='conflicting content'):
        build(cfg, 'conflict-fixture', [a, b])
    assert not (tmp_path / 'runtime/cache/conflict-fixture').exists()


def test_verify_rejects_changed_content(cfg, tmp_path):
    source = tmp_path / 'source'
    make_wheel(source)
    result = build(cfg, 'verify-fixture', [source])
    path = next(Path(result['path']).glob('*.whl'))
    path.chmod(0o644)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        verify(result['path'])


def test_root_mount_wins_without_discarding_task_pins(cfg, tmp_path, monkeypatch):
    shared = tmp_path / 'shared'
    make_wheel(shared)
    cfg['environment']['wheelhouse'] = str(shared)
    factory = ContainerFactory(cfg, client=object())
    row = {'instance_id': 'task', 'repo': 'org/repo', 'environment_overrides': {
        'wheelhouse': str(tmp_path / 'legacy'), 'repo_pins': {'org/repo': ['demo==1.0']},
        'network_mode': 'bridge'}}
    policy = factory.policy_for(row)
    assert policy.environment['wheelhouse'] == str(shared)
    assert policy.environment['network_mode'] == 'bridge'
    assert policy.specs('org/repo') == ['demo==1.0']
    assert policy.wheel_hashes == factory.policy.wheel_hashes
    assert policy.variables()['PIP_NO_INDEX'] == '1'
    factory.images = {'task': {'image_id': 'sha256:fixture'}}
    monkeypatch.setattr(factory, 'lock_images', lambda rows: factory.images)
    assert factory.spec(row)['volumes'] == {str(shared): {'bind': '/opt/dycon-wheels', 'mode': 'ro'}}


def test_legacy_null_root_keeps_task_mount(cfg, tmp_path):
    source = tmp_path / 'legacy'
    make_wheel(source)
    factory = ContainerFactory(cfg, client=object())
    row = {'instance_id': 'task', 'repo': 'org/repo',
           'environment_overrides': {'wheelhouse': str(source)}}
    assert factory.policy_for(row).environment['wheelhouse'] == str(source)


def test_configured_empty_directory_is_rejected_before_container_load(cfg, tmp_path):
    cfg['environment']['wheelhouse'] = str(tmp_path / 'empty')
    with pytest.raises(ValueError, match='contains no wheels'):
        EnvironmentPolicy(cfg)


def test_offline_export_reaches_official_script_and_build_subprocesses(cfg, tmp_path):
    source = tmp_path / 'wheels'
    make_wheel(source)
    cfg['environment']['wheelhouse'] = str(source)
    policy = EnvironmentPolicy(cfg)
    script = policy.wrap_eval('#!/bin/bash\npython -m pip install -e .\n')
    assert 'export PIP_NO_INDEX=1' in script
    assert script.index('export PIP_NO_INDEX=1') < script.index('python -m pip install')


def test_explicit_network_fallback_is_not_forced_offline(cfg, tmp_path):
    source = tmp_path / 'wheels'
    make_wheel(source)
    cfg['environment'].update(wheelhouse=str(source), network_fallback=True)
    assert 'PIP_NO_INDEX' not in EnvironmentPolicy(cfg).variables()


def test_build_downloads_must_match_published_sha256(cfg, tmp_path, monkeypatch):
    from runtime import wheelhouse
    import io

    cfg['environment']['wheelhouse'] = None
    monkeypatch.setattr(wheelhouse, 'BUILD_TOOLS', (('demo', '1.0'),))
    responses = [io.BytesIO(json.dumps({'urls': [{'packagetype': 'bdist_wheel',
        'filename': 'demo-1.0-py3-none-any.whl', 'url': 'https://example.invalid/demo',
        'size': len(b'not a wheel'), 'digests': {'sha256': '0'*64}}]}).encode()), io.BytesIO(b'not a wheel')]
    monkeypatch.setattr(wheelhouse.urllib.request, 'urlopen', lambda *a, **k: responses.pop(0))
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        build(cfg, 'bad-download', download=True)
    assert not (tmp_path / 'runtime/cache/bad-download').exists()


def test_vendor_metadata_is_not_the_wheel_distribution_metadata(cfg, tmp_path):
    source = tmp_path / 'source'
    path = make_wheel(source)
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('demo/_vendor/other-1.0.dist-info/METADATA', 'Name: other\nVersion: 1.0\n')
    assert build(cfg, 'vendor-fixture', [source])['wheel_count'] == 1


def test_declared_build_backends_include_sphinx_editable_backend():
    from runtime.wheelhouse import BUILD_TOOLS
    assert ('flit_core', '3.9.0') in BUILD_TOOLS
