import copy
from pathlib import Path

import pytest

from core.io import hash_json, read_json, write_json


def test_variant_selection_and_new_profile_are_frozen_without_touching_old_run(cfg, monkeypatch):
    from application.execution.run_plan import create_run, load_plan
    from catalog.store import Catalog
    from runtime.agents.openhands import validate_profile
    root = Path(cfg['project_root'])
    profile = validate_profile({'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1'})
    write_json(root / 'profile.json', profile)
    task = {'benchmark': 'skillsbench', 'revision': 'a' * 40, 'instance_id': 'freeze-task',
            'partition': 'train', 'instance': {'instance_id': 'freeze-task',
            'task_path': '/source/freeze-task', 'files': [], 'package_sha256': hash_json([]),
            'metadata': {}, 'images': []}}
    Catalog(root).register_tasks([task])
    key = hash_json({k: task[k] for k in ('benchmark', 'revision', 'instance_id')})
    write_json(root / 'snapshot.json', {'schema_version': 1, 'benchmark': 'skillsbench',
                                      'revision': 'a' * 40, 'tasks': {key: task}})
    cfg['benchmark'].update(backend='skillsbench', snapshot_ref='snapshot.json')
    cfg['agent'].update(backend='openhands', profile_ref='profile.json')
    create_run(cfg, 'legacy-run', operation='rollout', task_ids=['freeze-task'])
    legacy = load_plan(root, 'legacy-run')
    original_files = {name: (legacy['run'] / name).read_bytes() for name in
                      ('resolved-config.json', 'selection.json', 'plan.json', 'tasks.json')}
    updated = {**profile, 'verifier_python_index': 'https://mirror.test/simple/'}
    write_json(root / 'profile.json', updated)
    create_run(cfg, 'new-run', operation='rollout', task_ids=['freeze-task'])
    fresh = load_plan(root, 'new-run')
    assert fresh['config']['execution_context']['profile']['verifier_python_index'] == updated['verifier_python_index']
    assert 'verifier_python_index' not in load_plan(root, 'legacy-run')['config']['execution_context']['profile']
    for name, original in original_files.items():
        assert (legacy['run'] / name).read_bytes() == original
    write_json(root / 'profile.json', {**updated, 'verifier_python_index': 'https://later.test/simple/'})
    assert load_plan(root, 'new-run')['config']['execution_context'] == fresh['config']['execution_context']
