"""Fresh operator attempts for archived WorkerFailure cursors with broken history."""
import pytest

from application.execution.failed_retry import request_failed_retry, retry_index
from core.io import hash_json, read_json, sha256, write_json
from rollout import runner
from rollout.retry import SEED_POLICY
from rollout.sampling import AttemptContext, sample_from_item
from tests.test_limits_retry import configure, prepare, fake_runtime


def broken_attempt(cfg, run, loaded, *, seed_round=0, failure_kind='missing'):
    item = loaded['work_items']['items'][0]
    work = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']
    attempt_id = item['repetition_id'] + '-attempt-000'
    out = work / 'attempts' / attempt_id
    context = AttemptContext(run.name, item['task_key'], item['sample_key'], item['repetition_id'],
                             attempt_id, seed_round, item['sample_seed'] + seed_round,
                             item['sample_seed'], SEED_POLICY).as_dict()
    write_json(out / 'attempt-context.json', context)
    write_json(out / 'attempt-started.json', {'attempt_id': attempt_id,
        'started_at': '2026-01-01T00:00:00+00:00', 'retry_kind': 'initial', 'execution_attempt_id': None})
    write_json(out / 'agent-outcome.json', {'schema_version': 1, **context,
        'agent_exit_status': 'OpenHandsError', 'sample_key': item['sample_key'],
        'repetition_id': item['repetition_id'], 'task_key': item['task_key']})
    write_json(out / 'openhands/canonical.traj.json', {'instance_id': item['instance_id'],
        'messages': [], 'info': {'exit_status': 'OpenHandsError'}})
    if failure_kind == 'mismatch':
        write_json(out / 'raw.traj.json', {'instance_id': item['instance_id'], 'messages': [],
            'info': {'exit_status': 'Submitted', 'attempt_id': attempt_id,
                     'dycon': {'attempt_context': {**context, 'session_id': 'old-session'}}}})
    failure = {**item, 'attempt_id': None, 'exit_status': 'WorkerFailure', 'disposition': 'failed',
               'error': 'Confirmed agent outcome has no recoverable trajectory'}
    write_json(run / 'progress.json', {'samples': {item['sample_key']: failure}})
    return item, work, out, context, failure


def authorize(cfg, run, item, *, request_id='fresh'):
    return request_failed_retry(cfg['project_root'], run.name, request_id=request_id,
        exit_statuses=['WorkerFailure'], sample_keys=[item['sample_key']], reason='Retry failed recovery')


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('failure_kind', ['missing', 'mismatch'])
def test_progress_only_retry_skips_broken_history_once_and_preserves_seed(cfg, monkeypatch, legacy, failure_kind):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item, work, out, context, failure = broken_attempt(cfg, run, loaded, seed_round=1, failure_kind=failure_kind)
    old_bytes = {str(f.relative_to(out)): f.read_bytes() for f in out.rglob('*') if f.is_file()}
    receipt = authorize(cfg, run, item)
    assert len(receipt['items'][0]['superseded_incomplete_attempts']) == 1
    request_path = run / 'rollout/operator-retries/fresh/request.json'
    if legacy:
        receipt['items'][0].pop('superseded_incomplete_attempts')
        receipt['request_sha256'] = hash_json({k: v for k, v in receipt.items() if k != 'request_sha256'})
        write_json(request_path, receipt)
    request_bytes = request_path.read_bytes()
    # The dispatcher rebuilds progress before entering run_task; the archive is
    # the durable authorization even when no result or WorkerFailure cursor remains.
    write_json(run / 'progress.json', {'samples': {item['sample_key']: {**item, 'exit_status': 'Pending'}}})
    factory, calls, _, _ = fake_runtime(monkeypatch, ['ValueError'])
    monkeypatch.setattr(runner, '_recover_attempt', lambda *a, **k: pytest.fail('Old recovery was re-entered'))
    outcome = next(iter(runner.run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert len(calls) == 1
    assert outcome['attempt_id'] == item['repetition_id'] + '-attempt-001'
    assert outcome['seed_round'] == 1
    assert outcome['effective_seed'] == context['effective_seed']
    assert outcome['physical_attempts'] == 2
    assert outcome['infrastructure_retry_count'] == 0
    assert read_json(work / 'attempts' / outcome['attempt_id'] / 'attempt-started.json')['retry_kind'] == 'operator'
    assert not (out / 'attempt.json').exists()
    assert {str(f.relative_to(out)): f.read_bytes() for f in out.rglob('*') if f.is_file()} == old_bytes
    assert request_path.read_bytes() == request_bytes
    consumption = run / 'rollout/operator-retry-consumptions/fresh' / item['sample_key'] / (out.name + '.json')
    assert read_json(consumption)['context_sha256'] == sha256(out / 'attempt-context.json')
    # A failed new attempt does not turn the same authorization into another retry.
    runner.run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 1


def test_no_request_does_not_bypass_recovery(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item, work, out, context, failure = broken_attempt(cfg, run, loaded)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    outcome = next(iter(runner.run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert calls == []
    assert outcome['exit_status'] == 'WorkerFailure'
    assert 'no recoverable trajectory' in outcome['error']


def test_new_incomplete_attempt_is_not_superseded_by_old_request(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item, work, out, context, failure = broken_attempt(cfg, run, loaded)
    authorize(cfg, run, item)
    fresh = work / 'attempts' / (item['repetition_id'] + '-attempt-001')
    write_json(fresh / 'attempt-context.json', {**context, 'attempt_id': fresh.name})
    write_json(fresh / 'attempt-started.json', {'attempt_id': fresh.name,
        'started_at': '2099-01-01T00:00:00+00:00', 'retry_kind': 'operator', 'execution_attempt_id': None})
    write_json(fresh / 'agent-outcome.json', {**read_json(out / 'agent-outcome.json'), 'attempt_id': fresh.name})
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    recovered = []
    original = runner._recover_attempt
    def recover(*args, **kwargs):
        recovered.append(args[5])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, '_recover_attempt', recover)
    outcome = next(iter(runner.run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert calls == [] and outcome['exit_status'] == 'WorkerFailure'
    assert recovered == [fresh]


@pytest.mark.parametrize('artifact', ['archive', 'context', 'started'])
def test_retry_rejects_changed_authorized_evidence(cfg, monkeypatch, artifact):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item, work, out, context, failure = broken_attempt(cfg, run, loaded)
    receipt = authorize(cfg, run, item)
    if artifact == 'archive':
        archived = run / receipt['items'][0]['previous_result_archive']
        write_json(archived, {**failure, 'error': 'changed'})
        with pytest.raises(ValueError, match='archived worker failure changed'):
            retry_index(run)
        return
    path = out / ('attempt-context.json' if artifact == 'context' else 'attempt-started.json')
    write_json(path, {**read_json(path), 'changed': True})
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    with pytest.raises(ValueError, match='Incomplete operator retry source changed'):
        runner.run_task(loaded['config'], run.name, item['task_key'],
                        loaded['config']['selection']['tasks'][item['task_key']], factory,
                        sample=sample_from_item(item))
    assert calls == []


def test_retry_still_validates_reserved_sample_identity(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item, work, out, context, failure = broken_attempt(cfg, run, loaded)
    authorize(cfg, run, item)
    write_json(out / 'attempt-context.json', {**context, 'sample_seed': context['sample_seed'] + 1})
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    with pytest.raises(ValueError, match='Reserved attempt identity changed'):
        runner.run_task(loaded['config'], run.name, item['task_key'],
                        loaded['config']['selection']['tasks'][item['task_key']], factory,
                        sample=sample_from_item(item))
    assert calls == []
