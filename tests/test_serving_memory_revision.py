import copy
import os

import pytest

from application.execution.capture_budget_revision import revise_serving_memory
from application.execution.run_plan import load_plan
from core.io import read_json, sha256, write_json
from core.processes import identity
from tests.test_capture_budget_revision import make_run


def revise(cfg, **kwargs):
    params = dict(revision_id='gpu035', gpu_memory_utilization=.35,
                  reason='User authorized GPU0/1 TP2 restart at .35')
    params.update(kwargs)
    return revise_serving_memory(cfg['project_root'], 'budget-test', **params)


def test_only_memory_fraction_changes_and_revision_is_idempotent(cfg):
    run = make_run(cfg)
    old = read_json(run / 'resolved-config.json')
    stable = {name: sha256(run / name) for name in
              ['tasks.json', 'selection.json', 'rollout/work-items.json']}
    result = revise(cfg)
    new = load_plan(cfg['project_root'], 'budget-test')['config']
    normalized = copy.deepcopy(new)
    normalized['serving']['gpu_memory_utilization'] = old['serving']['gpu_memory_utilization']
    normalized['config_hash'] = old['config_hash']
    assert normalized == old
    assert new['serving']['gpu_memory_utilization'] == .35
    assert result['changed_config_fields'] == ['serving.gpu_memory_utilization']
    assert all(sha256(run / name) == digest for name, digest in stable.items())
    assert read_json(run / 'config-revisions/gpu035/before/resolved-config.json') == old
    assert read_json(run / 'events/config-revision-gpu035.json')['event'] == 'run.serving_memory_revised'
    assert revise(cfg) == result


def test_live_execution_rejected(cfg):
    run = make_run(cfg)
    path = run / 'execution/attempts/001'
    path.mkdir()
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    write_json(path / 'process.json', identity(os.getpid()))
    with pytest.raises(RuntimeError, match='still active'):
        revise(cfg)


def test_incomplete_cleanup_rejected(cfg):
    run = make_run(cfg)
    path = run / 'execution/attempts/001'
    path.mkdir()
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    write_json(path / 'exit.json', {'exit_code': 1})
    write_json(path / 'termination.json', {'cleanup_complete': False})
    with pytest.raises(RuntimeError, match='cleanup is incomplete'):
        revise(cfg)


@pytest.mark.parametrize('fraction', [0, -0.1, 1.1])
def test_invalid_fraction_never_changes_snapshot(cfg, fraction):
    run = make_run(cfg)
    before = sha256(run / 'resolved-config.json')
    with pytest.raises(ValueError):
        revise(cfg, gpu_memory_utilization=fraction)
    assert sha256(run / 'resolved-config.json') == before


def test_reason_required(cfg):
    make_run(cfg)
    with pytest.raises(ValueError, match='authorization/reason'):
        revise(cfg, reason=' ')
