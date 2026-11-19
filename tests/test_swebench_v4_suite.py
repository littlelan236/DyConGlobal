import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.io import read_json, write_json


def test_acceptance_records_unattempted_tasks_and_never_invents_pairs(tmp_path, monkeypatch):
    from benchmark_evaluation import swebench_acceptance as module
    run = tmp_path / 'runs/source'
    selection = {'rollout_ids': ['r1', 'r2'],
                 'rollouts': {'r1': {'task_key': 'a'}, 'r2': {'task_key': 'b'}},
                 'tasks': {key: {'instance': {'instance_id': key}} for key in ['a', 'b']}}
    write_json(run / 'harness/selection.json', selection)
    monkeypatch.setattr(module, 'load_plan', lambda *_: {'run': run, 'config': {}})
    result = module.collect(tmp_path, 'source', tmp_path / 'acceptance')
    assert result['requested'] == 2
    assert result['gold_valid_count'] == result['accepted_pair_count'] == 0
    assert result['execution_finished'] is False
    assert read_json(tmp_path / 'acceptance/tasks/a.json')['reward'] is None


def test_acceptance_audit_incomplete_and_cleanup_unavailable_still_publish(tmp_path, monkeypatch):
    from benchmark_evaluation import swebench_acceptance as module
    import docker
    run = tmp_path / 'runs/source'
    write_json(run / 'harness/selection.json', {'rollout_ids': ['r1'],
        'rollouts': {'r1': {'task_key': 'a'}}, 'tasks': {'a': {'instance': {'instance_id': 'a'}}}})
    write_json(run / 'harness/summary.json', {'evaluation_id': 'eval', 'outcomes': {}})
    audit = tmp_path / 'audit'
    write_json(audit / 'summary.json', {'complete': False})
    monkeypatch.setattr(module, 'load_plan', lambda *_: {'run': run, 'config': {'environment': {'docker_host': 'x'}}})
    monkeypatch.setattr(docker, 'DockerClient', lambda **_: (_ for _ in ()).throw(RuntimeError('offline')))
    result = module.collect(tmp_path, 'source', tmp_path / 'ledger', audit_dir=audit)
    assert result['audit_complete'] is False
    assert result['cleanup']['checked'] is False
    assert result['gold_valid_count'] == 0
    assert (tmp_path / 'ledger/tasks/a.json').exists()


def test_harness_clone_task_overlay_and_patch_rejection(cfg):
    from application.execution.run_plan import create_run, load_plan
    from application.execution.harness import prepare
    from application.execution.evidence import RunEvidence
    from catalog.store import Catalog
    from conftest import import_fixture
    rid = import_fixture(cfg)
    selected = RunEvidence(cfg['project_root'], 'source').select([rid])
    iid = next(iter(selected['tasks'].values()))['instance_id']
    source = create_run(cfg, 'harness-source', operation='harness', task_ids=[iid])
    prepare(cfg['project_root'], source.name, [rid], source_run='source')
    with Catalog(cfg['project_root']).transaction() as catalog:
        for task in catalog['tasks'].values():
            if task['instance_id'] == iid:
                task['instance']['environment_overrides'] = {'network_mode': 'bridge'}
    clone = create_run(cfg, 'harness-clone', operation='harness', from_run=source.name,
        task_ids=[iid], clone_overrides=['execution.infrastructure_retry_limit=0'])
    prepare(cfg['project_root'], clone.name, [rid], source_run='source')
    result = read_json(clone / 'harness/selection.json')
    assert next(iter(result['tasks'].values()))['instance']['environment_overrides']['network_mode'] == 'bridge'
    assert load_plan(cfg['project_root'], clone.name)['config']['execution']['infrastructure_retry_limit'] == 0
    with Catalog(cfg['project_root']).transaction() as catalog:
        for task in catalog['tasks'].values():
            if task['instance_id'] == iid:
                task['instance']['test_patch'] = task['instance'].get('test_patch', '') + '\nchanged'
    bad = create_run(cfg, 'harness-changed-patch', operation='harness', from_run=source.name, task_ids=[iid])
    with pytest.raises(ValueError, match='benchmark payload changed'):
        prepare(cfg['project_root'], bad.name, [rid], source_run='source')


def test_adapter_exception_collects_sidecars_before_cleanup(cfg, tmp_path, monkeypatch):
    from benchmark_evaluation.harness import install_harness_adapter
    from benchmark_evaluation.swebench_receipts import instrument
    from swebench.harness.test_spec.test_spec import TestSpec
    monkeypatch.setattr(TestSpec, 'eval_script', property(lambda _: ": '>>>>> Start Test Output'\npytest -q\n: '>>>>> End Test Output'\n"))
    collected = []
    factory = SimpleNamespace(
        policy_for=lambda _: SimpleNamespace(wrap_eval=lambda script: instrument(script)[0]),
        execute=lambda *_: (_ for _ in ()).throw(ConnectionError('transport lost')),
        collect_eval_receipts=lambda *_: collected.append('collected') or [])
    adapter = install_harness_adapter(cfg, factory, [{'instance_id': 'a'}], tmp_path)
    try:
        TestSpec.eval_script.fget(SimpleNamespace(instance_id='a'))
        container = SimpleNamespace(instance={'instance_id': 'a'}, id='container',
            context=SimpleNamespace(output_dir=tmp_path, run_id='run', attempt_id='attempt'))
        with pytest.raises(ConnectionError, match='transport lost'):
            adapter.exec_run_with_timeout(container, 'bash /eval.sh', 10)
        receipt = read_json(tmp_path / 'eval-execution.json')
        assert collected == ['collected']
        assert receipt['returncode'] is None
        assert receipt['receipt_protocol'] == 2
        assert receipt['execution_error'] == 'transport lost'
    finally:
        adapter.dycon_restore()


def test_exact_wheel_download_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    import io
    import json
    from runtime import wheelhouse
    payload = {'urls': [{'packagetype': 'bdist_wheel', 'filename': 'thing-1.0-py3-none-any.whl',
        'size': 3, 'url': 'https://files.pythonhosted.org/thing.whl', 'digests': {'sha256': '0' * 64}}]}
    monkeypatch.setattr(wheelhouse.urllib.request, 'urlopen',
        lambda url, **_: io.BytesIO(json.dumps(payload).encode() if url.endswith('/json') else b'bad'))
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        wheelhouse.download_releases(tmp_path, [('thing', '1.0')], ['cp39'])
