"""Frozen startup switches, shallow checks, and physical attempt boundaries."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.config import Config, assert_frozen, resolve
from core.io import hash_json, read_json, sha256, write_json
from core.model_identity import register, verify_model
from core.preflight import SKIP_REASON, long_checks_enabled


def skills_run(cfg, monkeypatch, *, long_checks=True):
    from application.execution.run_plan import create_run, load_plan
    from catalog.skillsbench import package_manifest
    from catalog.store import Catalog
    from runtime.agents.openhands import OpenHandsProfile

    cfg = copy.deepcopy(cfg)
    root = Path(cfg['project_root'])
    package = root / 'official/tasks/task'
    package.mkdir(parents=True)
    (package / 'task.md').write_text('frozen task')
    files = package_manifest(package)
    task = {'benchmark': 'skillsbench', 'revision': 'a' * 40, 'instance_id': 'task',
            'partition': 'train', 'instance': {'instance_id': 'task', 'task_path': str(package),
            'package_sha256': hash_json(files), 'files': files, 'metadata': {}, 'images': []}}
    from catalog.store import Task
    task = Task.model_validate(task).model_dump()
    key = hash_json({k: task[k] for k in ('benchmark', 'revision', 'instance_id')})
    cfg['benchmark'].update(backend='skillsbench', snapshot_ref='catalog/snapshot.json')
    cfg['agent'].update(backend='openhands', profile_ref='runtime/profile.json')
    cfg['skillsbench_images']['source_root'] = 'official'
    cfg['execution']['preflight'] = {'long_checks': long_checks}
    cfg['model']['verify_identity'] = False
    profile = OpenHandsProfile(bridge_listen='127.0.0.1', bridge_advertise='127.0.0.1',
                              output_paths={'task': ['/workspace/result']}).model_dump()
    write_json(root / 'runtime/profile.json', profile)
    write_json(root / 'catalog/snapshot.json', {'schema_version': 1, 'benchmark': 'skillsbench',
               'revision': 'a' * 40, 'tasks': {key: task}})
    Catalog(root).register_tasks([task])
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    run = create_run(cfg, 'switch-test', operation='rollout', task_ids=['task'])
    frozen = load_plan(root, run.name)['config']
    return run, frozen, key


def test_optional_fields_preserve_legacy_resolution(cfg):
    path = Path(cfg['config_source'])
    raw = yaml.safe_load(path.read_text())
    raw['execution'].pop('preflight', None)
    raw['serving'].pop('startup_mode', None)
    path.write_text(yaml.safe_dump(raw))
    first = resolve(path)
    assert 'preflight' not in first['execution']
    assert 'startup_mode' not in first['serving']
    assert long_checks_enabled(first)
    raw['execution']['preflight'] = {'long_checks': False}
    raw['serving']['startup_mode'] = 'parallel'
    path.write_text(yaml.safe_dump(raw))
    second = resolve(path)
    assert second['config_hash'] != first['config_hash']
    assert second['execution']['preflight'] == {'long_checks': False}
    assert second['serving']['startup_mode'] == 'parallel'
    assert_frozen(second)
    # A false flag never affects another backend.
    assert long_checks_enabled(second)
    second['benchmark']['backend'] = 'skillsbench'
    second['agent']['backend'] = 'openhands'
    assert not long_checks_enabled(second)
    for value in ('false', 0, None):
        bad = copy.deepcopy(raw)
        bad['execution']['preflight']['long_checks'] = value
        with pytest.raises(ValueError):
            Config.model_validate(bad)


def test_run_freezes_switches_despite_mutable_root(cfg, monkeypatch):
    from application.execution.run_plan import load_plan
    cfg['serving']['startup_mode'] = 'parallel'
    run, frozen, _ = skills_run(cfg, monkeypatch, long_checks=False)
    root_cfg = Path(cfg['config_source'])
    raw = yaml.safe_load(root_cfg.read_text())
    raw['execution']['preflight'] = {'long_checks': True}
    raw['serving']['startup_mode'] = 'serial'
    root_cfg.write_text(yaml.safe_dump(raw))
    loaded = load_plan(cfg['project_root'], run.name)['config']
    assert loaded == frozen
    assert loaded['execution']['preflight']['long_checks'] is False
    assert loaded['serving']['startup_mode'] == 'parallel'
    assert_frozen(loaded)


@pytest.mark.parametrize('long_checks', [True, False])
def test_preflight_records_skipped_checks_and_keeps_capabilities(cfg, monkeypatch, long_checks):
    from application.execution.run_plan import load_plan
    from runtime.agents import skillsbench
    run, frozen, key = skills_run(cfg, monkeypatch, long_checks=long_checks)
    client = SimpleNamespace(info=lambda: {'Warnings': []}, close=lambda: None)
    monkeypatch.setattr(skillsbench.docker, 'DockerClient', lambda **kwargs: client)
    monkeypatch.setattr('runtime.agents.openhands.check_version', lambda: None)
    # A native package's unsupported capability must still block fast startup.
    result = skillsbench.preflight(cfg['project_root'], run.name)
    assert any('multi-service' in x for x in result['blockers'])
    assert result['checks']['task_package_sha']['status'] == ('verified' if long_checks else 'skipped')
    if not long_checks:
        assert all(row['status'] == 'skipped' for row in result['checks'].values())
        assert result['environment_audit'] == {'status': 'skipped', 'verified': False, 'reason': SKIP_REASON}
    assert load_plan(cfg['project_root'], run.name)['config'] == frozen
    assert result['tasks']['task']['supported'] is False


def test_model_light_check_skips_bytes_but_retains_identity_and_size(cfg, tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    weight = checkpoint / 'model.safetensors'
    weight.write_bytes(b'old!')
    cfg['model'].update(path=str(checkpoint), fingerprint=register(cfg['project_root'], checkpoint))
    weight.write_bytes(b'new!')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        verify_model(cfg)
    monkeypatch.setattr('core.model_identity.sha256', lambda path: pytest.fail('light model check hashed a file'))
    verify_model(cfg, deep=False)
    weight.write_bytes(b'longer!')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        verify_model(cfg, deep=False)


def test_package_light_check_and_default_attempt_depth(tmp_path, monkeypatch):
    from catalog.skillsbench import verify_package
    from tests.test_skillsbench_compat import task_payload
    payload = task_payload(tmp_path)
    task = {'instance_id': 'official-task', 'revision': 'a' * 40, 'instance': payload}
    source = Path(payload['task_path']) / 'task.md'
    source.write_text('modified task')
    with pytest.raises(ValueError, match='package changed'):
        verify_package(task)
    monkeypatch.setattr('catalog.skillsbench.sha256', lambda path: pytest.fail('light package check hashed a file'))
    verify_package(task, deep=False)
    source.unlink()
    with pytest.raises(ValueError, match='path changed'):
        verify_package(task, deep=False)


def test_dependency_light_check_retains_receipt_inventory_and_sizes(tmp_path, monkeypatch):
    from runtime.agents.dependencies import verify_snapshot
    from tests.test_skillsbench_dependency_snapshot import publication
    manifest = publication(tmp_path)
    member = tmp_path / 'requirements.txt'
    member.write_text('example==2\n')
    with pytest.raises(ValueError, match='Changed dependency'):
        verify_snapshot(tmp_path)
    monkeypatch.setattr('runtime.agents.dependencies.sha256', lambda path: pytest.fail('light dependency check hashed a file'))
    assert verify_snapshot(tmp_path, deep=False) == manifest
    member.write_text('wrong size')
    with pytest.raises(ValueError, match='Changed dependency'):
        verify_snapshot(tmp_path, deep=False)


def test_oci_light_check_retains_graph_and_sizes(tmp_path, monkeypatch):
    from runtime.images.oci import descriptor_graph
    layer = tmp_path / ('1' * 64)
    layer.write_bytes(b'abc')
    config = tmp_path / ('2' * 64)
    config.write_text('{}')
    document = {'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.manifest.v1+json',
                'config': {'digest': 'sha256:' + config.name, 'size': 2},
                'layers': [{'digest': 'sha256:' + layer.name, 'size': 3}]}
    manifest = tmp_path / ('3' * 64)
    manifest.write_text(json.dumps(document))
    descriptor = {'digest': 'sha256:' + manifest.name, 'size': manifest.stat().st_size}
    with pytest.raises(ValueError, match='Corrupt OCI'):
        descriptor_graph(tmp_path, descriptor, deep=True)
    monkeypatch.setattr('runtime.images.oci.sha256', lambda path: pytest.fail('light OCI check hashed a blob'))
    assert len(descriptor_graph(tmp_path, descriptor, deep=False)) == 3
    layer.write_bytes(b'wrong size')
    with pytest.raises(FileNotFoundError):
        descriptor_graph(tmp_path, descriptor, deep=False)


def test_benchflow_admission_receipt_uses_frozen_policy(cfg, monkeypatch, tmp_path):
    from runtime.agents import skillsbench
    from runtime.backends.benchflow.adapter import BenchFlowAdapter
    cfg['benchmark']['backend'] = 'skillsbench'
    cfg['agent']['backend'] = 'openhands'
    cfg['execution']['preflight'] = {'long_checks': False}
    calls = []
    ctx = SimpleNamespace(cfg=cfg, run_id='run', factory=SimpleNamespace(
        lock_images=lambda tasks, **kwargs: calls.append(kwargs)))
    monkeypatch.setattr(skillsbench, 'preflight', lambda *args, **kwargs: {'blockers': []})
    result = BenchFlowAdapter(None).preflight(ctx, {})
    assert result.passed and calls == [{'deep': False}]
    receipt = read_json(tmp_path / 'runs/run/validation/startup-preflight.json')
    assert receipt['image_admission_checks']['image_admission_oci_sha']['status'] == 'skipped'

@pytest.mark.parametrize('long_checks', [True, False])
def test_shared_preflight_passes_depth_to_all_checks(cfg, monkeypatch, long_checks):
    from application.execution.run_plan import create_run
    from catalog.store import Catalog
    from runtime.agents import skillsbench
    from runtime.agents.openhands import OpenHandsProfile
    root = Path(cfg['project_root'])
    package = root / 'official/tasks/task'
    package.mkdir(parents=True)
    (package / 'task.md').write_text('frozen task')
    from catalog.skillsbench import package_manifest
    files = package_manifest(package)
    from catalog.store import Task
    task = Task.model_validate({'benchmark': 'skillsbench', 'revision': 'a' * 40,
        'instance_id': 'task', 'partition': 'train', 'instance': {
        'instance_id': 'task', 'task_path': str(package), 'package_sha256': hash_json(files),
        'files': files, 'metadata': {}, 'images': [{'key': 'unit', 'service': 'main'}]}}).model_dump()
    key = hash_json({k: task[k] for k in ('benchmark', 'revision', 'instance_id')})
    profile = OpenHandsProfile(bridge_listen='127.0.0.1', bridge_advertise='127.0.0.1',
                              output_paths={'task': ['/workspace/result']}).model_dump()
    # Creation freezes valid manifest identity; runtime file checks run below.
    dependency = root / 'dependency'
    dependency.mkdir()
    write_json(dependency / 'manifest.json', {})
    profile.update(dependency_snapshot=str(dependency), dependency_snapshot_sha256=sha256(dependency / 'manifest.json'))
    write_json(root / 'profile.json', profile)
    write_json(root / 'snapshot.json', {'schema_version': 1, 'benchmark': 'skillsbench',
        'revision': 'a' * 40, 'tasks': {key: task}})
    cfg['benchmark'].update(backend='skillsbench', snapshot_ref='snapshot.json')
    cfg['agent'].update(backend='openhands', profile_ref='profile.json')
    cfg['skillsbench_images']['source_root'] = 'official'
    cfg['execution']['preflight'] = {'long_checks': long_checks}
    cfg['model'].update(path=str(root), verify_identity=True)
    manifest = {'receipt_sha256': 'receipt', 'agent_runtime': {'source': 'frozen'},
        'source_adaptations': [{'policy': p} for p in (
            'dycon_bounded_request_subprocess_terminal_v1', 'preserve_explicit_terminal_backend_on_reset_v1')]}
    monkeypatch.setattr('runtime.agents.dependencies.verify_snapshot', lambda directory: manifest)
    Catalog(root).register_tasks([task])
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    run = create_run(cfg, 'all-checks', operation='rollout', task_ids=['task'])
    depths, audits = [], []
    monkeypatch.setattr('runtime.agents.dependencies.verify_snapshot',
                        lambda directory, *, deep=True: depths.append(('dependency', deep)) or manifest)
    monkeypatch.setattr('core.model_identity.verify_model',
                        lambda cfg, *, deep=True: depths.append(('model', deep)))
    image = SimpleNamespace(identity=lambda: {'image_id': 'frozen'})
    monkeypatch.setattr(skillsbench, 'image_reference',
                        lambda cfg, unit, *, deep=True: depths.append(('oci', deep)) or ({}, image))
    monkeypatch.setattr(skillsbench, 'audit_task_environments',
                        lambda *args, **kwargs: audits.append(True) or {'verified': True})
    monkeypatch.setattr(skillsbench.docker, 'DockerClient', lambda **kwargs:
                        SimpleNamespace(info=lambda: {}, close=lambda: None))
    monkeypatch.setattr('runtime.agents.openhands.check_version', lambda: None)
    result = skillsbench.preflight(root, run.name)
    assert result['ready'] and not result['blockers']
    assert depths == [('model', long_checks), ('oci', long_checks), ('dependency', long_checks)]
    assert bool(audits) is long_checks
    assert all(row['status'] == ('verified' if long_checks else 'skipped') for row in result['checks'].values())


def test_physical_attempt_still_checks_package_when_startup_checks_disabled(cfg, tmp_path):
    import asyncio

    from runtime.agents.skillsbench import _execute_attempt
    from tests.test_skillsbench_compat import task_payload
    payload = task_payload(tmp_path)
    (Path(payload['task_path']) / 'task.md').write_text('changed after import')
    task = {'instance_id': 'official-task', 'revision': 'a' * 40, 'instance': payload}
    cfg['execution']['preflight'] = {'long_checks': False}
    cfg['execution_context'] = {}
    with pytest.raises(ValueError, match='package changed'):
        asyncio.run(_execute_attempt(cfg, 'attempt-test', task, {}, object(),
                    {'attempt_id': 'attempt'}, tmp_path / 'attempt', object()))


def test_clone_can_freeze_startup_mode_without_changing_source(cfg):
    from application.execution.run_plan import create_run, load_plan
    from tests.test_limits_retry import prepare
    source, before = prepare(cfg, run_id='serial-source')
    clone = create_run(cfg, 'parallel-clone', operation='rollout', from_run=source.name,
                       clone_overrides=('serving.startup_mode=parallel',))
    after = load_plan(cfg['project_root'], clone.name)['config']
    assert after['serving']['startup_mode'] == 'parallel'
    assert load_plan(cfg['project_root'], source.name) == before
    assert_frozen(after)
