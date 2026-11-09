import copy
import os
import threading
import asyncio
import subprocess
import sys

import pytest

from application.execution.config_revision import revise_config
from application.execution.failed_retry import request_failed_retry, retry_index, superseded
from application.execution.run_plan import load_plan
from core.config import Capture
from core.io import read_json, sha256, write_json
from core.processes import identity
from rollout.runner import run_manifest
from rollout.sample_token_budget import SampleTokenBudget
from tests.test_limits_retry import configure, prepare, fake_runtime


def request(cfg, run, **overrides):
    args = dict(request_id='operator-1', exit_statuses=['ValueError'], reason='Explicit test authorization')
    args.update(overrides)
    return request_failed_retry(cfg['project_root'], run.name, **args)


def test_control_revision_preserves_other_settings_and_archives_old_snapshot(cfg):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    old = read_json(run / 'resolved-config.json')
    protected = {p: sha256(run / p) for p in ['tasks.json', 'selection.json', 'rollout/work-items.json']}
    values = dict(revision_id='control-120', updates={'capture.control_seconds': 120,
                  'capture.receipt_workers': 16}, reason='Explicit test authorization')
    receipt = revise_config(cfg['project_root'], run.name, **values)
    new = load_plan(cfg['project_root'], run.name)['config']
    normalized = copy.deepcopy(new)
    normalized['capture'] = old['capture']
    normalized['config_hash'] = old['config_hash']
    assert normalized == old
    assert new['capture']['control_seconds'] == 120
    assert new['capture']['receipt_workers'] == 16
    assert all(sha256(run / p) == h for p, h in protected.items())
    assert read_json(run / 'config-revisions/control-120/before/resolved-config.json') == old
    assert revise_config(cfg['project_root'], run.name, **values) == receipt


@pytest.mark.parametrize('value', [0, 129])
def test_receipt_worker_bounds(value):
    with pytest.raises(ValueError):
        Capture(receipt_workers=value)


def test_retry_failed_executes_new_attempt_only_for_requested_task(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg, tasks=('a', 'b'))
    factory, calls, _, _ = fake_runtime(monkeypatch, {'a': ['ValueError'], 'b': ['Submitted']})
    first = run_manifest(cfg, run.name, resume=True, factory=factory)
    a = next(v for v in first.values() if v['instance_id'] == 'a')
    b = next(v for v in first.values() if v['instance_id'] == 'b')
    work = run / 'rollout/tasks/a/repetitions/rep-000'
    old_attempt = work / 'attempts' / a['attempt_id'] / 'attempt.json'
    old_sha = sha256(old_attempt)
    receipt = request(cfg, run)
    assert receipt['sample_count'] == 1
    assert request(cfg, run) == receipt
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    result = run_manifest(cfg, run.name, resume=True, factory=factory)
    new_a = next(v for v in result.values() if v['instance_id'] == 'a')
    assert [c[0] for c in calls] == ['a']
    assert new_a['attempt_id'] != a['attempt_id']
    assert new_a['effective_seed'] == a['effective_seed']
    assert new_a['infrastructure_retry_count'] == 0
    assert new_a['physical_attempts'] == 2
    assert sha256(old_attempt) == old_sha
    assert next(v for v in result.values() if v['instance_id'] == 'b') == b
    assert read_json(work / 'attempts' / new_a['attempt_id'] / 'attempt-started.json')['retry_kind'] == 'operator'
    calls.clear()
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert calls == []


def test_retry_failed_physical_attempt_gets_fresh_budget_ledger(cfg, monkeypatch):
    from core.io import hash_json
    cfg = configure(cfg)
    cfg['execution']['sample_completion_token_budget'] = 10
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    old = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    first = SampleTokenBudget(cfg, old)
    assert first.reserve('api-old', 'agent_decision', 10) == 10
    first.record('api-old', 'agent_decision', {'completion_tokens': 7})
    old_bytes = first.path.read_bytes()

    request(cfg, run)
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    new = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert new['attempt_id'] != old['attempt_id']
    second = SampleTokenBudget(cfg, new)
    assert second.reserve('api-new', 'agent_decision', 10) == 10
    assert second.snapshot()['completion_tokens'] == 0
    assert first.path.read_bytes() == old_bytes


def test_worker_failure_without_physical_attempt_can_be_retried(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    work = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']
    failure = {**item, 'attempt_id': None, 'exit_status': 'WorkerFailure',
               'disposition': 'failed', 'physical_attempts': 0}
    write_json(work / 'result.json', failure)

    receipt = request(cfg, run, exit_statuses=['WorkerFailure'],
                      sample_keys=[item['sample_key']])
    assert receipt['sample_count'] == 1
    assert receipt['items'][0]['superseded_attempts'][0]['source_kind'] == 'result_without_physical_attempt'
    assert superseded(retry_index(run), item['sample_key'], failure)

    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    outcome = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert len(calls) == 1
    assert outcome['exit_status'] == 'Submitted'
    assert outcome['physical_attempts'] == 1
    assert read_json(work / 'attempts' / outcome['attempt_id'] / 'attempt-started.json')['retry_kind'] == 'operator'


def test_progress_only_worker_failure_can_be_requested_again(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    failure = {**item, 'attempt_id': None, 'exit_status': 'WorkerFailure',
               'disposition': 'failed', 'physical_attempts': 0}
    progress = {'samples': {item['sample_key']: failure}, 'updated_at': 'before'}
    write_json(run / 'progress.json', progress)
    args = dict(exit_statuses=['WorkerFailure'], sample_keys=[item['sample_key']])
    first = request(cfg, run, request_id='first', **args)
    progress['updated_at'] = 'after-unrelated-progress-update'
    write_json(run / 'progress.json', progress)
    second = request(cfg, run, request_id='second', **args)
    old = first['items'][0]['superseded_attempts'][0]
    new = second['items'][0]['superseded_attempts'][0]
    assert old['outcome_sha256'] == new['outcome_sha256']
    assert old['source_sha256'] != new['source_sha256']
    assert superseded(retry_index(run), item['sample_key'], failure)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    outcome = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert len(calls) == 1
    assert outcome['exit_status'] == 'Submitted'
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 1


def test_physical_retry_source_hash_conflict_still_rejected(cfg, monkeypatch):
    from core.io import hash_json
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    run_manifest(cfg, run.name, resume=True, factory=factory)
    receipt = request(cfg, run)
    duplicate = copy.deepcopy(receipt)
    duplicate['request_id'] = 'conflicting'
    duplicate['items'][0]['superseded_attempts'][0]['source_sha256'] = 'different'
    duplicate['request_sha256'] = hash_json({k: v for k, v in duplicate.items()
                                           if k != 'request_sha256'})
    write_json(run / 'rollout/operator-retries/conflicting/request.json', duplicate)
    with pytest.raises(ValueError, match='Conflicting operator retry attempt identity'):
        retry_index(run)


def test_operator_retry_is_once_even_if_new_attempt_fails(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    run_manifest(cfg, run.name, resume=True, factory=factory)
    request(cfg, run)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 1
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 1


def test_operator_retry_keeps_resampled_seed_round(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['LimitsExceeded', 'ValueError'])
    previous = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert previous['seed_round'] == 1
    request(cfg, run)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    result = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert len(calls) == 1
    assert result['seed_round'] == 1
    assert result['effective_seed'] == previous['effective_seed']
    assert result['resample_retry_count'] == 1


def test_success_cannot_be_requested_for_rerun(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    run_manifest(cfg, run.name, resume=True, factory=factory)
    with pytest.raises(ValueError, match='Only failed/blocked'):
        request(cfg, run, exit_statuses=['Submitted'])


def test_retry_source_tampering_is_rejected(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    old = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    request(cfg, run)
    assert superseded(retry_index(run), old['sample_key'], old)
    with pytest.raises(ValueError, match='source outcome changed'):
        superseded(retry_index(run), old['sample_key'], {**old, 'exit_status': 'Submitted'})


def test_explicit_blocked_capture_retry_preserves_old_attempt_and_bypasses_recovery(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    previous = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    work = run / 'rollout/tasks/a/repetitions/rep-000'
    blocked = {**previous, 'exit_status': 'CaptureError', 'disposition': 'blocked',
               'capture_status': 'pending_recovery'}
    source = work / 'attempts' / previous['attempt_id'] / 'attempt.json'
    write_json(source, blocked)
    write_json(work / 'result.json', blocked)
    old_sha = sha256(source)
    receipt = request(cfg, run, exit_statuses=['CaptureError'])
    assert receipt['sample_count'] == 1
    def unexpected_recovery(*args, **kwargs):
        raise AssertionError('Explicitly superseded blocked attempt must not be recovered in place')
    import rollout.runner
    monkeypatch.setattr(rollout.runner, '_recover_attempt', unexpected_recovery)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    outcome = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert len(calls) == 1 and outcome['exit_status'] == 'Submitted'
    assert outcome['attempt_id'] != blocked['attempt_id']
    assert sha256(source) == old_sha


def test_live_execution_rejects_retry_and_control_revision(cfg):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    p = run / 'execution/attempts/000'
    write_json(run / 'execution/latest.json', {'attempt': '000'})
    write_json(p / 'process.json', identity(os.getpid()))
    with pytest.raises(RuntimeError, match='still active'):
        request(cfg, run)
    with pytest.raises(RuntimeError, match='still active'):
        revise_config(cfg['project_root'], run.name, revision_id='live', updates={
            'capture.control_seconds': 120, 'capture.receipt_workers': 16}, reason='Explicit test authorization')


def test_unrelated_capture_run_does_not_block_task_retry(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    run_manifest(cfg, run.name, resume=True, factory=factory)
    env = {**os.environ, 'DYCON_CAPTURE_PROJECT': cfg['project_root'], 'DYCON_CANCEL_OWNER_RUN': 'other-run'}
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], env=env)
    try:
        assert request(cfg, run)['sample_count'] == 1
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_plugin_uses_sixteen_receipt_workers_concurrently(monkeypatch):
    pytest.importorskip('starlette', reason='Native serving plugin belongs to the easysteer environment')
    from runtime.serving.vllm_plugin import DyConCapturePlugin
    monkeypatch.setenv('DYCON_CAPTURE_CONTROL_SECONDS', '120')
    monkeypatch.setenv('DYCON_CAPTURE_RECEIPT_WORKERS', '16')
    monkeypatch.setenv('DYCON_CAPTURE_EXPORT_OPTIONS', '{}')
    plugin = DyConCapturePlugin()
    barrier = threading.Barrier(16, timeout=10)
    async def exercise():
        return await asyncio.gather(*(plugin.receipt_lane.run(barrier.wait) for _ in range(16)))
    try:
        assert plugin.control_seconds == 120
        assert plugin.receipt_workers == 16
        assert sorted(asyncio.run(exercise())) == list(range(16))
    finally:
        for lane in [plugin.wait_lane, plugin.receipt_lane, plugin.publish_lane, plugin.control_lane]:
            lane.close()
