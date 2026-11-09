from types import SimpleNamespace

import pytest

from core.io import read_json, write_json
from runtime.agents.validation_cleanup import reconcile_validation


def scenario(tmp_path, monkeypatch, *, status='exited', foreign=False):
    source = tmp_path / 'validation/oracle/task'
    labels = {'dycon.project': str(tmp_path.resolve()), 'dycon.run': 'run',
        'dycon.task': 'task', 'dycon.role': 'skillsbench-environment',
        'dycon.epoch': 'epoch', 'dycon.attempt': 'attempt', 'dycon.sample': 'sample'}
    write_json(source / 'ownership-compose.json', {'services': {'main': {'labels': labels}}})
    write_json(source / 'result.json', {'valid': False, 'cleanup_confirmed': False})
    frozen = {'benchmark': {'backend': 'skillsbench'}, 'environment': {'docker_host': 'fake'},
              'selection': {'tasks': {'key': {'instance_id': 'task'}}}}
    monkeypatch.setattr('application.execution.run_plan.load_plan',
                        lambda *args: {'config': frozen, 'run': tmp_path})
    events = []
    actual = dict(labels)
    if foreign:
        actual['dycon.run'] = 'other-run'
    container = SimpleNamespace(id='container', status=status, labels=actual,
        attrs={'State': {'Status': status}, 'Image': 'image'}, reload=lambda: None)
    def remove(**kw):
        assert kw == {'force': False, 'v': False}
        events.append('removed')
    container.remove = remove
    def containers(**kw):
        assert 'dycon.run=run' in kw['filters']['label']
        assert 'dycon.epoch=epoch' in kw['filters']['label']
        return [] if 'removed' in events else [container]
    client = SimpleNamespace(containers=SimpleNamespace(list=containers),
        networks=SimpleNamespace(list=lambda **kw: []), close=lambda: events.append('closed'))
    monkeypatch.setattr('runtime.agents.validation_cleanup.docker.DockerClient', lambda **kw: client)
    return source, events


def test_cleanup_removes_only_stopped_owned_sandbox_and_preserves_failed_receipt(tmp_path, monkeypatch):
    source, events = scenario(tmp_path, monkeypatch)
    original = (source / 'result.json').read_bytes()
    result = reconcile_validation({'project_root': str(tmp_path)}, 'run')
    assert result['cleanup_confirmed'] and result['removed_containers'] == 1
    assert events == ['removed', 'closed']
    assert (source / 'result.json').read_bytes() == original
    receipt = read_json(result['receipt'])
    assert receipt['attempts'][0]['source_sha256']
    again = reconcile_validation({'project_root': str(tmp_path)}, 'run')
    assert again['cleanup_confirmed'] and again['removed_containers'] == 0


@pytest.mark.parametrize('status,foreign', [('running', False), ('exited', True)])
def test_cleanup_refuses_live_or_foreign_resources(tmp_path, monkeypatch, status, foreign):
    _, events = scenario(tmp_path, monkeypatch, status=status, foreign=foreign)
    with pytest.raises(ValueError, match='active or differently owned'):
        reconcile_validation({'project_root': str(tmp_path)}, 'run')
    assert events == ['closed']


def test_environment_cleanup_finds_nested_attempt_without_a_completed_result(tmp_path, monkeypatch):
    source, events = scenario(tmp_path, monkeypatch)
    target = tmp_path / 'validation/environments/task/contract/attempts/environment-id'
    target.parent.mkdir(parents=True)
    source.rename(target)
    (target / 'result.json').unlink()
    result = reconcile_validation({'project_root': str(tmp_path)}, 'run', section='environments')
    assert result['cleanup_confirmed'] and result['removed_containers'] == 1
    assert events == ['removed', 'closed']
    assert not (target / 'result.json').exists()


def test_environment_cleanup_refuses_live_auditor_before_any_docker_mutation(tmp_path, monkeypatch):
    _, events = scenario(tmp_path, monkeypatch)
    monkeypatch.setattr('runtime.agents.validation_cleanup._environment_auditor_running', lambda *a: True)
    with pytest.raises(ValueError, match='auditor is active'):
        reconcile_validation({'project_root': str(tmp_path)}, 'run', section='environments')
    assert events == []
