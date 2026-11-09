"""Exercise explicit capture reruns and append-only planning through real resume."""
from pathlib import Path

import pytest

from application.execution.failed_retry import request_failed_retry, retry_index, superseded
from application.execution.repetitions import add_repetitions
from application.execution.run_plan import load_plan
from core.io import read_json, sha256, write_json
from rollout.runner import run_manifest
from rollout.sampling import effective_seed, sample_from_item
from tests.test_limits_retry import configure, fake_runtime, prepare


def add(cfg, run, **kwargs):
    return add_repetitions(cfg['project_root'], run.name, request_id='append-1',
                           reason='Append independent samples', **kwargs)


def retry(cfg, run, **kwargs):
    return request_failed_retry(cfg['project_root'], run.name, request_id='partial-1',
                                reason='Explicit partial capture rerun', **kwargs)


def mark_partial(run, result):
    value = {**result, 'capture_status': 'partial', 'failure_type': 'capture_finalization_error'}
    work = run / 'rollout/tasks' / result['instance_id'] / 'repetitions' / result['repetition_id']
    write_json(work / 'result.json', value)
    write_json(work / 'attempts' / result['attempt_id'] / 'attempt.json', value)
    return value


def test_partial_union_retries_once_preserves_complete_sample_and_sources(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg, tasks=('a', 'b', 'c'))
    factory, _, _, _ = fake_runtime(monkeypatch, {'a': ['Submitted'], 'b': ['TimeoutError'], 'c': ['Submitted']})
    original = run_manifest(cfg, run.name, resume=True, factory=factory)
    a = mark_partial(run, next(v for v in original.values() if v['instance_id'] == 'a'))
    complete = next(v for v in original.values() if v['instance_id'] == 'c')
    source = run / 'rollout/tasks/a/repetitions/rep-000/attempts' / a['attempt_id'] / 'attempt.json'
    digest = sha256(source)
    receipt = retry(cfg, run, exit_statuses=['TimeoutError'], accepted_capture_partial=True)
    assert receipt['sample_count'] == 2
    assert retry(cfg, run, exit_statuses=['TimeoutError'], accepted_capture_partial=True) == receipt
    assert superseded(retry_index(run), a['sample_key'], a)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    result = run_manifest(cfg, run.name, resume=True, factory=factory)
    assert sorted(c[0] for c in calls) == ['a', 'b']
    assert result[complete['sample_key']] == complete
    new = result[a['sample_key']]
    assert new['attempt_id'] != a['attempt_id']
    assert new['effective_seed'] == a['effective_seed']
    assert sha256(source) == digest
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 2


def test_partial_opt_in_and_exact_selection(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    old = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    partial = mark_partial(run, old)
    with pytest.raises(ValueError, match='Only failed/blocked'):
        retry(cfg, run, exit_statuses=['Submitted'])
    with pytest.raises(ValueError, match='current attempt changed'):
        retry(cfg, run, exit_statuses=[], accepted_capture_partial=True,
              sample_keys=[partial['sample_key']], expected_attempts={partial['sample_key']: 'wrong'})
    assert not (run / 'rollout/operator-retries/partial-1/request.json').exists()
    receipt = retry(cfg, run, exit_statuses=[], accepted_capture_partial=True,
                    sample_keys=[partial['sample_key']],
                    expected_attempts={partial['sample_key']: partial['attempt_id']})
    assert receipt['sample_count'] == 1


def test_complete_accepted_still_cannot_be_retried(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    value = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    with pytest.raises(ValueError, match='failure status'):
        retry(cfg, run, exit_statuses=[], accepted_capture_partial=True, sample_keys=[value['sample_key']])
    with pytest.raises(ValueError, match='Only failed/blocked'):
        retry(cfg, run, exit_statuses=['Submitted'], accepted_capture_partial=True)


def test_append_preserves_old_bytes_and_resume_executes_only_new_samples(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg, tasks=('a', 'b'), repetitions=2)
    factory, _, _, _ = fake_runtime(monkeypatch, ['LimitsExceeded', 'Submitted'])
    original = run_manifest(cfg, run.name, resume=True, factory=factory)
    protected = {p: sha256(p) for p in run.rglob('*') if p.is_file()
                 and ('attempts' in p.parts or p.name in {'result.json', 'evidence.json', 'resolved-config.json'})}
    old_manifest = loaded['work_items']
    old_seeds = {i['sample_key']: [effective_seed(sample_from_item(i), old_manifest, n) for n in range(4)]
                 for i in old_manifest['items']}
    preview = add(cfg, run, count=1, dry_run=True)
    assert preview['added_sample_count'] == 2
    assert not (run / 'repetition-extensions').exists()
    receipt = add(cfg, run, count=1, expect_plan_hash=preview['old_plan_sha256'])
    assert add(cfg, run, count=1) == receipt
    assert all(sha256(p) == digest for p, digest in protected.items())
    loaded = load_plan(cfg['project_root'], run.name)
    assert loaded['work_items']['items'][:4] == old_manifest['items']
    for item in old_manifest['items']:
        assert [effective_seed(sample_from_item(item), loaded['work_items'], n) for n in range(4)] == old_seeds[item['sample_key']]
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    result = run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 2 and len(result) == 6
    assert all(result[k] == v for k, v in original.items())
    assert read_json(run / 'status-summary.json')['sample_summary']['sample_count'] == 6
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == 2


def test_multiple_appends_keep_disjoint_seed_blocks(cfg):
    cfg = configure(cfg)
    run, initial = prepare(cfg, repetitions=2)
    add(cfg, run, count=2)
    first = load_plan(cfg['project_root'], run.name)
    add_repetitions(cfg['project_root'], run.name, request_id='append-2', count=1, reason='More samples')
    second = load_plan(cfg['project_root'], run.name)
    assert second['work_items']['items'][:4] == first['work_items']['items']
    seeds = []
    for item in second['work_items']['items']:
        sample = sample_from_item(item)
        seeds += [effective_seed(sample, second['work_items'], n) for n in range(10)]
    assert len(seeds) == len(set(seeds))
    assert second['plan']['repeat_count'] == 5
    sample = sample_from_item(second['work_items']['items'][-1])
    with pytest.raises(ValueError, match='seed_space_exhausted'):
        effective_seed(sample, second['work_items'], 2**32)


@pytest.mark.parametrize('count', [None, 0, -1, True])
def test_bad_append_count_rejected(cfg, count):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    with pytest.raises(ValueError, match='positive integer'):
        add(cfg, run, count=count)


def test_append_stale_plan_and_incomplete_cleanup_rejected(cfg):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    with pytest.raises(ValueError, match='changed since preview'):
        add(cfg, run, count=1, expect_plan_hash='stale')
    write_json(run / 'execution/latest.json', {'attempt': '000'})
    with pytest.raises(RuntimeError, match='exit receipt'):
        add(cfg, run, count=1)
    assert not (run / 'repetition-extensions').exists()


def test_interrupted_append_blocks_execution_and_recovers(cfg, monkeypatch):
    import application.execution.repetitions as module
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    original_write = module.write_json
    def interrupted(path, value):
        if Path(path) == run / 'rollout/work-items.json':
            raise OSError('simulated interruption after plan write')
        return original_write(path, value)
    monkeypatch.setattr(module, 'write_json', interrupted)
    with pytest.raises(OSError, match='simulated interruption'):
        add(cfg, run, count=1)
    with pytest.raises(RuntimeError, match='Incomplete repetition extension'):
        load_plan(cfg['project_root'], run.name)
    monkeypatch.setattr(module, 'write_json', original_write)
    receipt = add(cfg, run, recover=True)
    assert receipt['new_repeat_count'] == 2
    assert load_plan(cfg['project_root'], run.name)['plan']['repeat_count'] == 2
    assert add(cfg, run, recover=True) == receipt


def test_cli_append_and_recovery_work_with_broken_root_config(cfg, monkeypatch, capsys):
    import json

    import application.execution.repetitions as module
    from core.cli import main
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    Path(cfg['config_source']).write_text('invalid: [')
    args = ['--config', cfg['config_source'], 'run', 'add-repetitions', run.name,
            '--request-id', 'cli-append', '--reason', 'CLI append']
    main(args + ['--count', '1', '--dry-run'])
    assert json.loads(capsys.readouterr().out)['added_sample_count'] == 1
    writer = module.write_json
    def interrupted(path, value):
        if Path(path) == run / 'rollout/work-items.json':
            raise OSError('simulated interruption')
        return writer(path, value)
    monkeypatch.setattr(module, 'write_json', interrupted)
    with pytest.raises(SystemExit):
        main(args + ['--count', '1'])
    capsys.readouterr()
    monkeypatch.setattr(module, 'write_json', writer)
    main(args + ['--recover'])
    assert json.loads(capsys.readouterr().out)['new_repeat_count'] == 2


def test_cli_partial_selector_without_exit_status(cfg, monkeypatch, capsys):
    import json

    from core.cli import main
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    value = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    mark_partial(run, value)
    main(['--config', cfg['config_source'], 'run', 'retry-failed', run.name,
          '--request-id', 'cli-partial', '--reason', 'Retry partial',
          '--accepted-capture-partial'])
    assert json.loads(capsys.readouterr().out)['sample_count'] == 1
    summary = read_json(run / 'status-summary.json')['sample_summary']
    assert summary['pending_count'] == 1 and summary['accepted_count'] == 0


def test_append_updates_pending_summary_immediately(cfg):
    cfg = configure(cfg)
    run, _ = prepare(cfg, tasks=('a', 'b'))
    add(cfg, run, count=2)
    summary = read_json(run / 'status-summary.json')['sample_summary']
    assert summary['sample_count'] == 6
    assert summary['pending_count'] == 6


def test_recovery_rejects_unrelated_plan_edits(cfg, monkeypatch):
    import application.execution.repetitions as module
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    writer = module.write_json
    def interrupted(path, value):
        if Path(path) == run / 'rollout/work-items.json':
            raise OSError('interrupted')
        return writer(path, value)
    monkeypatch.setattr(module, 'write_json', interrupted)
    with pytest.raises(OSError):
        add(cfg, run, count=1)
    monkeypatch.setattr(module, 'write_json', writer)
    plan = read_json(run / 'plan.json')
    plan['repeat_count'] = 999
    write_json(run / 'plan.json', plan)
    with pytest.raises(ValueError, match='current file changed'):
        add(cfg, run, recover=True)


def test_append_then_config_revision_and_replan_remain_compatible(cfg):
    from application.execution.config_revision import revise_config
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    add(cfg, run, count=1)
    revise_config(cfg['project_root'], run.name, revision_id='resource',
                  updates={'execution.rollout_workers': 2}, reason='Resource update')
    assert load_plan(cfg['project_root'], run.name)['plan']['repeat_count'] == 2
    revise_config(cfg['project_root'], run.name, revision_id='new-seed',
                  updates={'generation.seed': 100}, replan=True, reason='New seed')
    loaded = load_plan(cfg['project_root'], run.name)
    assert loaded['plan']['repeat_count'] == 2
    assert loaded['work_items']['items'][0]['sample_seed'] == 100


def test_partial_retry_does_not_fall_back_to_older_accepted_attempt(cfg, monkeypatch):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    old = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    first = mark_partial(run, old)
    retry(cfg, run, exit_statuses=[], accepted_capture_partial=True)
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    current = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    current = mark_partial(run, current)
    request_failed_retry(cfg['project_root'], run.name, request_id='partial-2',
                         exit_statuses=[], reason='Retry second partial',
                         accepted_capture_partial=True)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    final = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert len(calls) == 1
    assert final['attempt_id'] not in {first['attempt_id'], current['attempt_id']}


def test_parallel_repetition_seed_blocks_include_all_branches():
    from rollout.sampling import make_work_items
    selection = {'task_keys': ['a'], 'tasks': {'a': {'instance_id': 'task'}}}
    manifest = make_work_items(selection, 'parallel', 4, 42, 3,
                              extension={'original_repeat_count': 2, 'seed_block_size': 2**32})
    seeds = []
    for item in manifest['items']:
        for round_id in range(5):
            seed = effective_seed(sample_from_item(item), manifest, round_id, actual_branches=3)
            seeds += list(range(seed, seed + 3))
    assert len(seeds) == len(set(seeds))
    with pytest.raises(ValueError, match='seed_space_exhausted'):
        make_work_items(selection, 'overflow', 2, -(2**63), 3,
                        extension={'original_repeat_count': 1, 'seed_block_size': 2**32})


def test_append_holds_same_run_lock_as_resume(cfg):
    from core.locking import file_lock
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    with file_lock(Path(cfg['project_root']) / 'state/run-locks' / (run.name + '.lock')):
        with pytest.raises(BlockingIOError):
            add(cfg, run, count=1)
    assert not (run / 'repetition-extensions').exists()
