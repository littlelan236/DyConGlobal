import copy

import pytest

from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.io import hash_json, read_json, sha256
from tests.conftest import task


def source_run(cfg):
    Catalog(cfg['project_root']).register_tasks([task('a'), task('b')])
    source_cfg = copy.deepcopy(cfg)
    source_cfg['benchmark']['backend'] = 'swebench'
    source_cfg['agent']['backend'] = 'minisweagent'
    source_cfg['config_hash'] = hash_json({k: v for k, v in source_cfg.items() if k != 'config_hash'})
    return create_run(source_cfg, 'source-frozen', operation='rollout', task_ids=['a', 'b'])


def test_clone_preserves_every_effective_setting_and_source_bytes(cfg):
    source = source_run(cfg)
    old = load_plan(cfg['project_root'], source.name)
    protected = {n: sha256(source/n) for n in ['resolved-config.json', 'selection.json', 'tasks.json', 'plan.json']}
    # Changing current catalog and caller defaults must not change the source selection/settings.
    Catalog(cfg['project_root']).register_tasks([task('unrelated')])
    caller = copy.deepcopy(cfg)
    caller['generation']['seed'] = 999
    new = create_run(caller, 'new-clone', operation='rollout', from_run=source.name)
    loaded = load_plan(cfg['project_root'], new.name)
    ignore = {'run_id', 'config_hash', 'config_source', 'config_source_sha256', 'overrides'}
    assert {k:v for k,v in loaded['config'].items() if k not in ignore} == {k:v for k,v in old['config'].items() if k not in ignore}
    assert loaded['config']['selection'] == old['config']['selection']
    assert all(sha256(source/n) == digest for n,digest in protected.items())
    assert read_json(new/'cloned-from.json')['source_run_id'] == source.name
    assert all(item['sample_seed'] == 42 for item in loaded['work_items']['items'])


def test_clone_can_reverse_frozen_task_order_without_changing_source(cfg):
    source = source_run(cfg)
    source_plan = load_plan(cfg['project_root'], source.name)
    new = create_run(cfg, 'reverse-clone', operation='rollout',
                     from_run=source.name, reverse_task_order=True)
    loaded = load_plan(cfg['project_root'], new.name)
    assert loaded['config']['selection']['task_keys'] == list(
        reversed(source_plan['config']['selection']['task_keys']))
    assert [item['instance_id'] for item in loaded['work_items']['items']] == [item['instance_id'] for item in source_plan['work_items']['items']][::-1]
    assert load_plan(cfg['project_root'], source.name)['config']['selection'] == source_plan['config']['selection']
    assert read_json(new / 'cloned-from.json')['task_order'] == 'reversed'


def test_clone_allows_explicit_reasoning_close_budget(cfg):
    source = source_run(cfg)
    new = create_run(cfg, 'think-close-clone', operation='rollout', from_run=source.name,
        clone_overrides=['generation.thinking_token_budget=81920',
                         'generation.require_reasoning_close=true',
                         'generation.max_tokens=98304'])
    generation = load_plan(cfg['project_root'], new.name)['config']['generation']
    assert generation['thinking_token_budget'] == 81920
    assert generation['require_reasoning_close'] is True
    assert generation['max_tokens'] == 98304


def test_clone_applies_only_explicit_gpu_and_endpoint_changes(cfg):
    source = source_run(cfg)
    new = create_run(cfg, 'resource-clone', operation='rollout', from_run=source.name, clone_overrides=[
        'serving.gpus=[0,1,2,3]', 'serving.tensor_parallel_size=4',
        'serving.base_url=http://127.0.0.1:8104/v1'])
    loaded = load_plan(cfg['project_root'], new.name)
    assert loaded['config']['serving']['gpus'] == [0,1,2,3]
    assert loaded['config']['serving']['tensor_parallel_size'] == 4
    assert loaded['config']['serving']['base_url'] == 'http://127.0.0.1:8104/v1'
    assert len(loaded['work_items']['items']) == 2


def test_clone_can_select_a_catalog_subset_without_changing_source(cfg):
    source = source_run(cfg)
    new = create_run(cfg, 'subset-clone', operation='rollout', from_run=source.name, task_ids=['a'])
    loaded = load_plan(cfg['project_root'], new.name)
    assert [item['instance_id'] for item in loaded['work_items']['items']] == ['a']
    assert {item['instance_id'] for item in load_plan(cfg['project_root'], source.name)['work_items']['items']} == {'a', 'b'}


@pytest.mark.parametrize('kwargs,message', [
    ({'repeat_count':2}, 'retains source repeat_count'),
    ({'clone_overrides':['generation.seed=43']}, 'Unsupported frozen-clone override'),
    ({'clone_overrides':['serving.gpus=[0,1,2,3]']}, 'count must equal'),
])
def test_clone_rejects_unrequested_semantic_changes(cfg, kwargs, message):
    source = source_run(cfg)
    with pytest.raises(ValueError, match=message):
        create_run(cfg, 'invalid-clone', operation='rollout', from_run=source.name, **kwargs)


def test_cli_clone_uses_frozen_seed_and_full_selection(cfg, capsys):
    from pathlib import Path
    from core.cli import main
    source = source_run(cfg)
    main(['--config', str(Path(cfg['project_root'])/'config.yaml'),
          '--set', 'serving.gpus=[0,1,2,3]', '--set', 'serving.tensor_parallel_size=4',
          'run', 'create', 'cli-clone', '--operation', 'rollout', '--from-run', source.name])
    loaded = load_plan(cfg['project_root'], 'cli-clone')
    assert len(loaded['work_items']['items']) == 2
    assert loaded['config']['generation']['seed'] == 42


def test_clone_to_external_storage_preserves_source_and_selection(cfg):
    import json
    from pathlib import Path
    source = source_run(cfg)
    protected = {name: sha256(source / name) for name in ['config.yaml', 'resolved-config.json', 'tasks.json', 'plan.json']}
    physical = Path(cfg['project_root']).parent / 'external-run-products'
    run = create_run(cfg, 'external-clone', operation='rollout', from_run=source.name,
                     clone_overrides=['paths.artifact_roots=' + json.dumps({'run': str(physical)})])
    assert run == physical / 'runs' / 'external-clone'
    loaded = load_plan(cfg['project_root'], run.name)
    assert loaded['config']['run_storage_root'] == str(physical)
    assert loaded['config']['selection'] == load_plan(cfg['project_root'], source.name)['config']['selection']
    assert loaded['config']['generation'] == load_plan(cfg['project_root'], source.name)['config']['generation']
    assert all(sha256(source / name) == digest for name, digest in protected.items())
    assert read_json(run / 'cloned-from.json')['overrides'][0]['path'] == 'paths.artifact_roots'
    alias = Path(cfg['project_root']) / 'runs' / run.name
    assert alias.is_symlink() and alias.resolve() == run
    from runtime.serving.capture import OnlineCaptureController
    from core.io import write_json
    trajectory = run / 'rollout' / 'external.traj.json'
    write_json(trajectory, {'messages': []})
    controller = OnlineCaptureController.__new__(OnlineCaptureController)
    controller.project_root = Path(cfg['project_root'])
    controller.request_outputs = {}
    import threading
    controller._output_condition = threading.Condition()
    controller.config = {"finalize_seconds": 1}
    from types import SimpleNamespace
    controller.sessions = {"external": SimpleNamespace(options={})}
    controller.prepare_trajectory('external', str(trajectory))
    # Reaching the checksum check proves finalize accepted the registered external path.
    import pytest
    with pytest.raises(ValueError, match='checksum mismatch'):
        controller.finalize('external', rollout_id='r-test', trajectory_path=str(trajectory),
                            trajectory_sha256='different', model_fingerprint='fixture')
