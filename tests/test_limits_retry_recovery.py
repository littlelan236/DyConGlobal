"""Failure injection at durable boundaries, counters, and canonical recovery."""
import concurrent.futures
from pathlib import Path

import pytest

from application.execution.evidence import RunEvidence
from application.execution.report import report
from application.execution.status import run_status
from core.io import hash_json, read_json, write_json
from rollout.retry import rebuild_counts, record_agent_outcome
from rollout.runner import run_manifest
from rollout.sampling import AttemptContext, effective_seed, sample_from_item
from tests.test_limits_retry import configure, fake_runtime, prepare


def attempt_context(item, manifest, n=0, round_id=0):
    sample = sample_from_item(item)
    aid = f'{sample.attempt_prefix}-{n:03d}'
    context = AttemptContext(sample.run_id, sample.task_key, sample.sample_key, sample.repetition_id,
        aid, round_id, effective_seed(sample, manifest, round_id), sample.sample_seed,
        'repetition_round_offset_v1').as_dict()
    return sample, context


def work_path(run, item):
    return run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']


def test_reserved_but_not_started_resumes_same_attempt_and_seed(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    _, context = attempt_context(item, loaded['work_items'])
    out = work_path(run, item) / 'attempts' / context['attempt_id']
    write_json(out / 'attempt-context.json', context)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    value = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert calls[0][1:] == (context['attempt_id'], 42, 'Submitted')
    assert value['physical_attempts'] == 1
    assert value['infrastructure_retry_count'] == 0


def test_unknown_interrupted_attempt_uses_same_seed_and_infra_budget(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    _, context = attempt_context(item, loaded['work_items'])
    out = work_path(run, item) / 'attempts' / context['attempt_id']
    write_json(out / 'attempt-context.json', context)
    write_json(out / 'attempt-started.json', {'attempt_id': context['attempt_id'], 'retry_kind': 'initial'})
    write_json(out / 'raw.traj.json', {'instance_id': item['instance_id'], 'messages': [], 'info': {'exit_status': 'Running'}})
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    value = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert calls[0][1:] == ('rep-000-attempt-001', 42, 'Submitted')
    assert value['physical_attempts'] == 2
    assert value['infrastructure_retry_count'] == 1
    assert value['resample_retry_count'] == 0
    assert read_json(run / 'rollout/limits-exceeded-counts.json')['resample_trigger_count'] == 0


@pytest.mark.parametrize('crash_file', ['agent-outcome.json', 'attempt.json', 'retry-state.json'])
@pytest.mark.parametrize('first_status', ['LimitsExceeded', 'reasoning_loop', 'Submitted', 'ParallelSelectionError'])
def test_failure_at_persisted_boundary_recovers_without_replaying_agent(cfg, monkeypatch, crash_file, first_status):
    import rollout.retry as retry
    import rollout.runner as runner
    cfg = configure(cfg)
    if first_status == 'ParallelSelectionError':
        cfg['execution']['selection_retry'] = {'enabled': True, 'max_retries': 3}
    run, _ = prepare(cfg)
    factory, calls, closed, _ = fake_runtime(monkeypatch, [first_status, 'Submitted'])
    original = runner.write_json
    broken = []
    def fail_once(path, value):
        if Path(path).name == crash_file and not broken:
            broken.append(True)
            raise OSError('injected crash at ' + crash_file)
        return original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'write_json', fail_once)
        patch.setattr(retry, 'write_json', fail_once)
        run_manifest(cfg, run.name, resume=True, factory=factory)
    assert broken
    assert len(calls) == len(closed) == 1
    # Missing/corrupt cursor and projection are explicitly disposable.
    (run / 'rollout/limits-exceeded-counts.json').write_text('{corrupt')
    result = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    expected = 1 if first_status == 'Submitted' else 2
    assert len(calls) == expected
    assert [c[2] for c in calls] == ([42] if expected == 1 else [42, 43])
    assert result['exit_status'] == 'Submitted'
    assert result['physical_attempts'] == expected
    summary = read_json(run / 'rollout/limits-exceeded-counts.json')
    assert summary['resample_trigger_count'] == (first_status in {'LimitsExceeded', 'reasoning_loop'})
    data = RunEvidence(cfg['project_root'], run.name)
    assert data.verify()['rollouts'] == expected
    assert data.select()['rollout_ids'] == [result['rollout_id']]
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == expected


def test_statistics_failure_does_not_skip_capture_or_container_cleanup(cfg, monkeypatch):
    import rollout.retry as retry
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, calls, closed, _ = fake_runtime(monkeypatch, ['LimitsExceeded'])
    original = retry.write_json
    def fail_summary(path, value):
        if Path(path).name == 'limits-exceeded-counts.json' and value['resample_trigger_count']:
            raise OSError('disk full')
        original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(retry, 'write_json', fail_summary)
        with pytest.raises(OSError, match='disk full'):
            run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == len(closed) == 1
    assert len(list((run / 'rollout/tasks').glob('*/repetitions/*/attempts/*/agent-outcome.json'))) == 1
    evidence = RunEvidence(cfg['project_root'], run.name)
    assert len(evidence.read()['rollouts']) == 1
    with pytest.raises(ValueError, match='statistics'):
        evidence.verify()
    from application.execution.run_plan import load_plan
    from rollout.retry import rebuild_counts
    rebuild_counts(run, load_plan(cfg['project_root'], run.name)['work_items'])
    assert evidence.verify()['rollouts'] == 1


def test_outcome_before_projection_crash_rebuilds_once_across_tasks(cfg):
    cfg = configure(cfg)
    run, loaded = prepare(cfg, repetitions=3, tasks=('a', 'b'))
    facts = []
    for i, item in enumerate(loaded['work_items']['items']):
        _, context = attempt_context(item, loaded['work_items'])
        out = work_path(run, item) / 'attempts' / context['attempt_id']
        write_json(out / 'attempt-context.json', context)
        status = 'LimitsExceeded' if i % 2 else 'reasoning_loop'
        facts.append((out, context, status))
    def publish(entry):
        out, context, status = entry
        return record_agent_outcome(run, out, context, status)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(publish, facts * 3))
    path = run / 'rollout/limits-exceeded-counts.json'
    counts = read_json(path)
    assert counts['limits_exceeded_count'] == 3
    assert counts['reasoning_loop_count'] == 3
    assert counts['resample_trigger_count'] == 6
    for field in ('limits_exceeded_count', 'reasoning_loop_count', 'resample_trigger_count'):
        assert counts[field] == sum(row[field] for row in counts['samples'].values())
        assert counts[field] == sum(row[field] for row in counts['tasks'].values())
    # A late lower attempt number must not be skipped by high-water deduplication.
    item = loaded['work_items']['items'][0]
    _, context = attempt_context(item, loaded['work_items'], n=5)
    out = work_path(run, item) / 'attempts' / context['attempt_id']
    write_json(out / 'attempt-context.json', context)
    record_agent_outcome(run, out, context, 'LimitsExceeded')
    _, context2 = attempt_context(item, loaded['work_items'], n=2)
    out2 = work_path(run, item) / 'attempts' / context2['attempt_id']
    write_json(out2 / 'attempt-context.json', context2)
    record_agent_outcome(run, out2, context2, 'reasoning_loop')
    path.unlink()
    for _ in range(2):
        counts = rebuild_counts(run, loaded['work_items'])
        assert counts['resample_trigger_count'] == 8
    assert run_status(cfg['project_root'], run.name)['rollout_counts']['resample_trigger_count'] == 8
    assert report(cfg['project_root'], run.name)['rollout_counts']['resample_trigger_count'] == 8
    with pytest.raises(ValueError, match='Conflicting'):
        record_agent_outcome(run, out, context, 'Submitted')


def test_generation_counter_tracks_transport_and_finalization_separately(cfg):
    from rollout.progress import Progress, generation_phase
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    tracker = Progress(run, cfg, loaded['work_items']['items'])
    tracker.phase(item['sample_key'], 'agent')
    assert read_json(run / 'rollout/phases.json')['active_generation_samples'] == 0
    runtime = {'sample_key': item['sample_key'], 'progress_tracker': tracker}
    with generation_phase(runtime):
        assert read_json(run / 'rollout/phases.json')['active_generation_samples'] == 1
        with generation_phase(runtime):
            assert read_json(run / 'rollout/phases.json')['active_generation_samples'] == 1
    tracker.phase(item['sample_key'], 'finalizing')
    value = read_json(run / 'rollout/phases.json')
    assert value['active_generation_samples'] == 0
    assert value['finalizing_attempts'] == 1


def test_existing_normal_attempt_does_not_scan_statistics_history(cfg, monkeypatch):
    import rollout.retry as retry
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    _, context = attempt_context(item, loaded['work_items'])
    out = work_path(run, item) / 'attempts' / context['attempt_id']
    write_json(out / 'attempt-context.json', context)
    monkeypatch.setattr(Path, 'glob', lambda *args: (_ for _ in ()).throw(AssertionError('normal counter update scanned history')))
    retry.record_agent_outcome(run, out, context, 'reasoning_loop')
    assert read_json(run / 'rollout/limits-exceeded-counts.json')['reasoning_loop_count'] == 1


def test_missing_agent_fact_rebuilds_from_verified_canonical_even_when_settled(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['LimitsExceeded', 'Submitted'])
    run_manifest(cfg, run.name, resume=True, factory=factory)
    for path in (run / 'rollout/tasks').glob('*/repetitions/*/attempts/*/agent-outcome.json'):
        path.unlink()
    (run / 'rollout/limits-exceeded-counts.json').unlink()
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 2
    assert read_json(run / 'rollout/limits-exceeded-counts.json')['limits_exceeded_count'] == 1
    assert len(list((run / 'rollout/tasks').glob('*/repetitions/*/attempts/*/agent-outcome.json'))) == 2


def test_capture_recovery_reuses_original_trajectory_without_generation(cfg, monkeypatch):
    import hidden.recover_online as recovery
    cfg = configure(cfg)
    cfg['capture'].update(enabled=True, failure_policy='block_rollout')
    cfg['capture']['export']['enabled'] = False
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    run, _ = prepare(cfg)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['reasoning_loop', 'Submitted'], capture_failure=True)
    first = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert first['exit_status'] == 'CaptureError'
    row = RunEvidence(cfg['project_root'], run.name).read()['rollouts'][first['rollout_id']]
    before = (Path(cfg['project_root']) / row['trajectory_path']).read_bytes()
    session = first['capture_session_id']
    spool = Path(cfg['project_root']) / cfg['capture']['export']['spool_root'] / run.name
    write_json(spool / 'sessions' / session / 'finalization.json', {
        'rollout_id': first['rollout_id'], 'trajectory_sha256': row['trajectory_sha256']})
    write_json(run / 'hidden/publication.json', {'sessions': {session: 'complete'}})
    monkeypatch.setattr(recovery, 'recover_online', SimpleRecovery())
    # A second finalization may fail, but the first attempt is recovered exactly once.
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 2
    assert [c[2] for c in calls] == [42, 43]
    assert (Path(cfg['project_root']) / row['trajectory_path']).read_bytes() == before
    counts = read_json(run / 'rollout/limits-exceeded-counts.json')
    assert counts['reasoning_loop_count'] == 1


class SimpleRecovery:
    def __init__(self):
        self.__wrapped__ = lambda *args: None


def test_legacy_infrastructure_attempts_do_not_gain_fresh_budget(cfg, monkeypatch):
    cfg = configure(cfg, infra=2, retry=False)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    for number in range(3):
        aid = f'rep-000-attempt-{number:03d}'
        write_json(work_path(run, item) / 'attempts' / aid / 'attempt.json', {
            'attempt_id': aid, 'exit_status': 'APIError'})
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    result = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert not calls
    assert result['exit_status'] == 'infrastructure_exhausted'
    assert result['infrastructure_retry_count'] == 2
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert not calls


def test_gapped_reserved_attempt_number_does_not_use_directory_count(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    item = loaded['work_items']['items'][0]
    _, context = attempt_context(item, loaded['work_items'], n=7)
    write_json(work_path(run, item) / 'attempts' / context['attempt_id'] / 'attempt-context.json', context)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['LimitsExceeded', 'Submitted'])
    value = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert [row[1] for row in calls] == ['rep-000-attempt-007', 'rep-000-attempt-008']
    assert [row[2] for row in calls] == [42, 43]
    assert value['physical_attempts'] == 2
    assert value['infrastructure_retry_count'] == 0


def test_crash_before_capture_prepare_preserves_raw_and_does_not_advance_seed(cfg, monkeypatch):
    from hidden.backends.vllm import online
    cfg = configure(cfg)
    cfg['capture'].update(enabled=True, failure_policy='block_rollout')
    cfg['capture']['export']['enabled'] = False
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    run, _ = prepare(cfg)
    factory, calls, closed, _ = fake_runtime(monkeypatch, ['LimitsExceeded', 'Submitted'])
    def crash(*args):
        raise KeyboardInterrupt('crash before capture prepare')
    monkeypatch.setattr(online.CaptureClient, 'stop', crash)
    with pytest.raises(KeyboardInterrupt):
        run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == len(closed) == 1
    value = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert value['exit_status'] == 'CaptureError'
    assert value['agent_exit_status'] == 'LimitsExceeded'
    assert value['resample_retry_count'] == 0
    assert len(calls) == 1
    assert len(list((run / 'rollout/tasks').glob('*/repetitions/*/attempts/*/raw.traj.json'))) == 1
    assert RunEvidence(cfg['project_root'], run.name).read()['rollouts'] == {}


def test_frozen_seed_stride_cannot_change_without_changing_items_hash(cfg):
    from application.execution.run_plan import load_plan
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    path = run / 'rollout/work-items.json'
    manifest = read_json(path)
    manifest['seed_policy']['branch_stride'] += 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match='seed policy'):
        load_plan(cfg['project_root'], run.name)
