import copy
import os

import pytest

from application.execution.config_revision import revise_config
from application.execution.run_plan import load_plan
from core.io import hash_json, read_json, sha256, write_json
from core.processes import identity
from tests.test_capture_budget_revision import make_run as fixture_run


def make_run(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['serving']['management'] = 'managed'
    cfg['config_hash'] = hash_json({k:v for k,v in cfg.items() if k != 'config_hash'})
    return fixture_run(cfg)


def revise(cfg, **overrides):
    args = dict(revision_id='tp1', gpus=[0, 1], base_url='http://127.0.0.1:18200/v1',
                trajectory_sharding='per_gpu_tp1',
                replica_base_urls=['http://127.0.0.1:18200/v1', 'http://127.0.0.1:18201/v1'],
                max_cpu_workers_per_gpu=2, gpu_memory_utilization=.4,
                reason='User authorized topology conversion preserving progress')
    args.update(overrides)
    reason, revision_id = args.pop('reason'), args.pop('revision_id')
    updates = {'serving.' + key: value for key, value in args.items()}
    updates['serving.tensor_parallel_size'] = 1
    return revise_config(cfg['project_root'], 'budget-test', revision_id=revision_id,
                         updates=updates, reason=reason)


def test_conversion_preserves_samples_algorithm_and_audits_before_after(cfg):
    original = copy.deepcopy(cfg)
    original['serving'].update(gpus=[0, 1], tensor_parallel_size=2)
    original['config_hash'] = hash_json({k:v for k,v in original.items() if k != 'config_hash'})
    run = make_run(original)
    old = read_json(run / 'resolved-config.json')
    stable = {name: sha256(run / name) for name in
              ['selection.json', 'tasks.json', 'rollout/work-items.json', 'evidence.json']
              if (run / name).exists()}
    receipt = revise(original)
    new = load_plan(cfg['project_root'], 'budget-test')['config']
    changed = {key.removeprefix('serving.') for key in receipt['changed_config_fields']}
    normalized = copy.deepcopy(new)
    for key in changed:
        if key in old['serving']:
            normalized['serving'][key] = old['serving'][key]
        else:
            normalized['serving'].pop(key, None)
    normalized['config_hash'] = old['config_hash']
    assert normalized == old
    assert new['serving']['tensor_parallel_size'] == 1
    assert new['serving']['trajectory_sharding'] == 'per_gpu_tp1'
    assert all(sha256(run / name) == digest for name, digest in stable.items())
    assert read_json(run / 'config-revisions/tp1/before/resolved-config.json') == old
    assert revise(original) == receipt


@pytest.mark.parametrize('override', [
    {'replica_base_urls': ['http://127.0.0.1:18200/v1']},
    {'replica_base_urls': ['http://127.0.0.1:18200/v1', 'http://localhost:18200/v1']},
    {'max_cpu_workers_per_gpu': 0},
    {'gpu_memory_utilization': 1.1},
    {'gpu_memory_utilization': None},
    {'trajectory_sharding': None},
])
def test_invalid_or_incomplete_profile_cannot_change_frozen_files(cfg, override):
    run = make_run(cfg)
    before = sha256(run / 'resolved-config.json')
    with pytest.raises(ValueError):
        revise(cfg, **override)
    assert sha256(run / 'resolved-config.json') == before


@pytest.mark.parametrize('active', [True, False])
def test_conversion_requires_stopped_and_clean_execution(cfg, active):
    run = make_run(cfg)
    attempt = run / 'execution/attempts/001'
    attempt.mkdir()
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    if active:
        write_json(attempt / 'process.json', identity(os.getpid()))
    else:
        write_json(attempt / 'exit.json', {'exit_code': -15})
        write_json(attempt / 'termination.json', {'cleanup_complete': False})
    before = sha256(run / 'resolved-config.json')
    with pytest.raises(RuntimeError, match='still active|cleanup is incomplete'):
        revise(cfg)
    assert sha256(run / 'resolved-config.json') == before


def test_explicit_kv_bytes_remains_explicit_when_fraction_changes(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['serving']['kv_cache_memory_bytes'] = 1024
    cfg['config_hash'] = hash_json({k:v for k,v in cfg.items() if k != 'config_hash'})
    run = make_run(cfg)
    revise(cfg)
    current = load_plan(cfg['project_root'], run.name)['config']
    assert current['serving']['kv_cache_memory_bytes'] == 1024
