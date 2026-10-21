import copy
import sys
from types import SimpleNamespace

import pytest

from core.io import read_json, write_json
from enrichment.scripts import run_extraction_queue as worker
from enrichment.scripts.launch_extraction import recover_jobs


def bundle_for(cfg):
    return {'models': {'fixture': {'path': '/fixture', 'fingerprint': 'fixture-model'}},
            'configs': {'fixture|think_open': copy.deepcopy(cfg)}}


def test_config_snapshot_survives_live_config_edit(cfg):
    from pathlib import Path
    bundle = bundle_for(cfg)
    Path(cfg['config_source']).write_text('future_schema_key: invalid for old process')
    first = worker.frozen_config(bundle, 'fixture', ['think_open'])
    first['hidden']['layers'].append(123)
    second = worker.frozen_config(bundle, 'fixture', ['think_open'])
    assert second == cfg
    assert bundle['configs']['fixture|think_open'] == cfg


def test_corrupt_snapshot_fails_before_work(cfg):
    bundle = bundle_for(cfg)
    bundle['configs']['fixture|think_open']['hidden']['device'] = 'changed'
    with pytest.raises(ValueError, match='checksum'):
        worker.frozen_config(bundle, 'fixture', ['think_open'])


def test_recovery_preserves_completed_and_data_errors():
    jobs = [
        {'rollout_id': 'done', 'status': 'completed'},
        {'rollout_id': 'schema', 'status': 'failed', 'error': 'core/config.py capture extra_forbidden'},
        {'rollout_id': 'data', 'status': 'failed', 'error': 'Prompt count mismatch: 1 != 2'},
    ]
    before = copy.deepcopy(jobs)
    assert recover_jobs(jobs) == ['schema']
    assert jobs[0] == before[0]
    assert jobs[2] == before[2]
    assert jobs[1]['status'] == 'pending'
    assert jobs[1]['failure_history'][0]['error'] == before[1]['error']


def test_system_failure_stops_before_claiming_second_job(cfg, monkeypatch):
    pytest.skip("GPU worker admission is covered by the isolated vLLM smoke")
    from pathlib import Path
    root = Path(cfg['project_root'])
    snapshot = root / 'frozen.json'
    write_json(snapshot, bundle_for(cfg))
    jobs = [dict(rollout_id=f'r-{i}', status='pending', model='fixture',
                 anchors=['think_open'], task_id=f't-{i}', run_id='test', trajectory_sha256='sha')
            for i in range(2)]
    queue = root / 'state' / worker.BATCH / 'queue.json'
    write_json(queue, jobs)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=object()))
    monkeypatch.setattr(worker, 'update', lambda *args: None)
    def broken_model(cfg):
        raise RuntimeError('checkpoint cannot load')
    monkeypatch.setattr(worker, 'verify_model', broken_model)
    from application.execution.evidence import RunEvidence
    original_select = RunEvidence.select
    with pytest.raises(RuntimeError, match='System error'):
        worker.worker(root, 0, snapshot)
    assert RunEvidence.select is original_select
    after = read_json(queue)
    assert after[0]['status'] == 'pending'
    assert after[0]['failure_history'][0]['scope'] == 'system'
    assert after[1] == jobs[1]


@pytest.mark.parametrize('exc,scope', [
    (ValueError('Prompt count mismatch: 10 != 20'), 'trajectory'),
    (ValueError('Frozen extraction config checksum mismatch'), 'system'),
    (RuntimeError('CUDA out of memory'), 'system'),
    (ImportError('missing model package'), 'system'),
])
def test_failure_classification(exc, scope):
    assert worker.failure_scope(exc) == scope
