import copy
import pytest
from core.io import write_json
from core.backends import execution_context

def context_fixture(tmp_path, monkeypatch):
    import runtime.agents.openhands as openhands
    monkeypatch.setattr(openhands, 'validate_profile', lambda value: value)
    write_json(tmp_path / 'profile.json', {})
    task = {'benchmark': 'skillsbench', 'revision': 'revision', 'instance_id': 'task',
            'partition': 'train', 'exclusion_reason': None,
            'instance': {'package_sha256': 'package', 'images': [{'image_id': 'image'}]}}
    write_json(tmp_path / 'snapshot.json', {'schema_version': 1, 'benchmark': 'skillsbench',
        'revision': 'revision', 'tasks': {'key': task}})
    cfg = {'project_root': str(tmp_path), 'benchmark': {'backend': 'skillsbench',
        'snapshot_ref': 'snapshot.json'}, 'agent': {'backend': 'openhands',
        'profile_ref': 'profile.json'}, 'generation': {}}
    return cfg, copy.deepcopy(task)

def test_partition_metadata_change(tmp_path, monkeypatch):
    cfg, task = context_fixture(tmp_path, monkeypatch)
    task['partition'] = 'test'
    assert execution_context(cfg, {'key': task})['task_keys'] == ['key']

@pytest.mark.parametrize('field', ['revision', 'package', 'image', 'exclusion'])
def test_definition_change_still_rejected(tmp_path, monkeypatch, field):
    cfg, task = context_fixture(tmp_path, monkeypatch)
    task['partition'] = 'test'
    if field == 'revision': task['revision'] = 'changed'
    if field == 'package': task['instance']['package_sha256'] = 'changed'
    if field == 'image': task['instance']['images'][0]['image_id'] = 'changed'
    if field == 'exclusion': task['exclusion_reason'] = 'changed'
    with pytest.raises(ValueError, match='frozen benchmark snapshot'):
        execution_context(cfg, {'key': task})
