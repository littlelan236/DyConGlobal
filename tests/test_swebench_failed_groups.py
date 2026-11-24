import copy

import pytest

from catalog import swebench_sync as subject
from catalog.store import Catalog, empty_manifest
from core.backends import benchmark_family
from core.io import hash_json, read_json, write_json


@pytest.mark.parametrize('group', ['dev', 'verified', 'nonverified'])
def test_failed_group_survives_sync_and_remains_selectable(tmp_path, monkeypatch, group):
    iid = 'repo__task-1'
    benchmark = group + '-gold-patch-failed'
    row = {'benchmark': benchmark, 'revision': 'original', 'instance_id': iid,
           'repo': 'org/repo', 'base_commit': 'abc', 'partition': 'train',
           'exclusion_reason': None, 'instance': {'instance_id': iid, 'repo': 'org/repo',
           'base_commit': 'abc', 'nas_image': {'store': 'unchanged'}, 'patch': 'original-gold',
           'environment_overrides': {'constraint': 'original'}}}
    write_json(tmp_path / 'catalog/manifest.json', empty_manifest())
    catalog = Catalog(tmp_path)
    catalog.register_tasks([row])
    before = catalog.read()
    frozen = tmp_path / 'runs/old/selection.json'
    write_json(frozen, before)
    frozen_bytes = frozen.read_bytes()
    sets = {'verified': {iid} if group == 'verified' else set(),
            'dev': {iid} if group == 'dev' else set(),
            'test': {iid} if group != 'dev' else set()}
    monkeypatch.setattr(subject, '_reference', lambda *a, **k: None)
    class Image:
        def __init__(self, *a, **k):
            pass
        def verify(self, **k):
            pass
    monkeypatch.setattr(subject, 'NasImageStore', Image)
    cfg = {'project_root': str(tmp_path)}
    source = {'sources': {'verified': {'revision': 'v1'}, 'dev': {'revision': 'd1'}}}
    monkeypatch.setattr(subject, 'load_sources', lambda p: (source, sets, {}))
    monkeypatch.setattr(subject, '_plan_references', lambda *a: {})
    write_json(tmp_path / 'source.json', {})
    result = subject.sync(cfg, source_path=tmp_path / 'source.json', image_plans=[],
                          output=tmp_path / 'receipt', apply=True)
    assert not result['changed']
    assert catalog.read() == before
    assert not any(result['missing_counts'].values())
    assert frozen.read_bytes() == frozen_bytes
    assert benchmark_family(benchmark) == 'swebench'
    assert catalog.select_tasks(benchmark='swebench', task_ids=[iid])['tasks'] == before['tasks']
    export = tmp_path / 'failed.json'
    assert subject.list_tasks(cfg, group=benchmark, output=export)['selected'] == 1
    assert read_json(export)['tasks'] == [iid]
    assert subject.list_tasks(cfg, group=group)['selected'] == 0
