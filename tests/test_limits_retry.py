"""Contract tests exercise the real manifest scheduler and attempt lifecycle."""
import copy
import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

from application.execution.evidence import RunEvidence
from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.deadline import Deadline
from core.io import hash_json, read_json, write_json
from rollout.retry import disposition, enabled
from rollout.runner import run_manifest
from rollout.sampling import effective_seed, sample_from_item
from tests.conftest import task


def configure(cfg, *, workers=1, infra=2, retry=True):
    cfg = copy.deepcopy(cfg)
    cfg['execution'].update(rollout_workers=workers, infrastructure_retry_limit=infra)
    cfg['execution']['limits_retry'] = {
        'enabled': retry, 'retry_statuses': ['LimitsExceeded', 'reasoning_loop'],
        'seed_policy': 'repetition_round_offset_v1'}
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    return cfg


def fake_runtime(monkeypatch, script, *, on_run=None, capture_failure=False):
    import requests

    import core.model_identity
    import runtime.serving.startup
    from hidden.backends.vllm import online
    from rollout import agent, model
    from runtime.containers import rollout_adapter
    from runtime.serving import manager

    calls, closed = [], []
    indices = defaultdict(int)
    lock = threading.Lock()
    service_count = []

    class Env:
        def __init__(self, factory, instance, context):
            self.context = context
            self.iid = instance['instance_id']
        def close(self):
            closed.append(self.context.attempt_id)

    class Agent:
        def __init__(self, model, env, **kwargs):
            self.model, self.env = model, env
            self.messages = []
        def run(self, problem):
            with lock:
                key = self.env.iid
                n = indices[key]
                indices[key] += 1
                choices = script.get(key, ['Submitted']) if isinstance(script, dict) else script
                status = choices[min(n, len(choices) - 1)]
                calls.append((key, self.env.context.attempt_id, self.model['generation']['seed'], status))
            if on_run:
                on_run(self, status)
            return {'exit_status': status, 'submission': 'patch' if status == 'Submitted' else ''}
        def save(self, path, extra):
            write_json(path, {**extra, 'messages': [{'role': 'assistant', 'content': 'done'}]})

    class Capture:
        def __init__(self, *args, **kwargs): pass
        def start(self, *args, **kwargs): pass
        def stop(self, *args): pass
        def prepare(self, *args): pass
        def finalize(self, *args, **kwargs):
            if capture_failure:
                raise RuntimeError('capture finalization interrupted')
            return {'capture_status': 'complete'}

    class Service:
        def __init__(self, *args):
            service_count.append(1)
            self.runtime = {
                'capture_token_inventory': {'newline_token_ids': [1], 'close_token_ids': [2],
                    'tokenizer_fingerprint': 'fixture', 'capture_select': {}}, 'capture_control_url': 'fixture'}
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(agent, 'DurableAgent', Agent)
    monkeypatch.setattr(model, 'make_model', lambda cfg, *args: cfg)
    monkeypatch.setattr(rollout_adapter, 'RolloutEnvironment', Env)
    monkeypatch.setattr(manager, 'ServingManager', Service)
    monkeypatch.setattr(core.model_identity, 'verify_model', lambda cfg: None)
    monkeypatch.setattr(runtime.serving.startup, 'check_tool_chain', lambda *args: None)
    monkeypatch.setattr(online, 'CaptureClient', Capture)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: {}))
    factory = SimpleNamespace(deadline=Deadline(), lock_images=lambda rows: None, cleanup_owned=lambda run: None)
    return factory, calls, closed, service_count


def prepare(cfg, run_id='retry', *, repetitions=1, tasks=('a',)):
    Catalog(cfg['project_root']).register_tasks([task(iid) for iid in tasks])
    run = create_run(cfg, run_id, operation='rollout', task_ids=tasks, repeat_count=repetitions)
    return run, load_plan(cfg['project_root'], run_id)


@pytest.mark.parametrize('statuses,limit,expected', [
    (['ParallelSelectionError', 'Submitted'], 3, 'Submitted'),
    (['ParallelSelectionError'], 2, 'selection_retry_exhausted'),
    (['ParallelSelectionError'], 0, 'selection_retry_exhausted'),
])
def test_selection_retry_real_scheduler_is_bounded_and_resumable(cfg, monkeypatch, statuses, limit, expected):
    cfg = configure(cfg, workers=2, retry=False)
    cfg['execution']['selection_retry'] = {'enabled': True, 'max_retries': limit}
    run, _ = prepare(cfg, tasks=('a', 'b'))
    factory, calls, _, _ = fake_runtime(monkeypatch, {'a': statuses, 'b': ['Submitted']})
    results = run_manifest(cfg, run.name, resume=True, factory=factory)
    a = next(row for row in results.values() if row['instance_id'] == 'a')
    assert a['exit_status'] == expected
    attempts = 2 if expected == 'Submitted' else limit + 1
    assert [c[2] for c in calls if c[0] == 'a'] == list(range(42, 42 + attempts))
    assert len([c for c in calls if c[0] == 'b']) == 1
    if expected != 'Submitted':
        assert a['disposition'] == 'failed'
        assert a['selection_retry_count'] == limit
    before = list(calls)
    assert run_manifest(cfg, run.name, resume=True, factory=factory) == results
    assert calls == before


@pytest.mark.parametrize('statuses,counts', [
    (['LimitsExceeded', 'LimitsExceeded', 'Submitted'], (2, 0)),
    (['reasoning_loop', 'reasoning_loop', 'Submitted'], (0, 2)),
    (['LimitsExceeded', 'reasoning_loop', 'Submitted'], (1, 1)),
    (['LimitsExceeded', 'ContextWindowExceededError'], (1, 0)),
    (['Submitted'], (0, 0)),
])
def test_real_attempts_resample_and_select_only_final(cfg, monkeypatch, statuses, counts):
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    factory, calls, closed, services = fake_runtime(monkeypatch, statuses)
    result = run_manifest(cfg, run.name, resume=True, factory=factory)
    sample = next(iter(result.values()))
    assert sample['exit_status'] == statuses[-1]
    assert sample['physical_attempts'] == len(statuses)
    assert sample['resample_retry_count'] == len(statuses) - 1
    assert sample['infrastructure_retry_count'] == 0
    assert [c[2] for c in calls] == list(range(42, 42 + len(statuses)))
    assert len(closed) == len(statuses)
    assert len(services) == 1
    data = RunEvidence(cfg['project_root'], run.name)
    assert data.verify()['rollouts'] == len(statuses)
    assert data.select()['rollout_ids'] == [sample['rollout_id']]
    summary = read_json(run / 'rollout/limits-exceeded-counts.json')
    assert (summary['limits_exceeded_count'], summary['reasoning_loop_count']) == counts
    assert summary['resample_trigger_count'] == sum(counts)
    before = len(calls)
    assert run_manifest(cfg, run.name, resume=True, factory=factory) == result
    assert len(calls) == before
    assert len(services) == 1
    assert read_json(run / 'rollout/limits-exceeded-counts.json')['resample_trigger_count'] == sum(counts)


@pytest.mark.parametrize('statuses,expected_seeds,infra', [
    (['LimitsExceeded', 'APIError', 'Submitted'], [42, 43, 43], 1),
    (['LimitsExceeded', 'APIError', 'LimitsExceeded', 'Submitted'], [42, 43, 43, 44], 1),
    (['APIError', 'LimitsExceeded', 'APIError', 'reasoning_loop', 'APIError'], [42, 42, 43, 43, 44], 2),
])
def test_infrastructure_budget_is_cumulative_across_seeds(cfg, monkeypatch, statuses, expected_seeds, infra):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, calls, _, _ = fake_runtime(monkeypatch, statuses)
    result = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert [c[2] for c in calls] == expected_seeds
    assert result['infrastructure_retry_count'] == infra
    if statuses[-1] == 'APIError':
        assert result['exit_status'] == 'infrastructure_exhausted'
        assert read_json(run / 'status.json')['stage'] == 'failed'
        assert RunEvidence(cfg['project_root'], run.name).select()['rollout_ids'] == []
        run_manifest(cfg, run.name, resume=True, factory=factory)
        assert len(calls) == len(statuses)
        assert read_json(run / 'status.json')['stage'] == 'failed'


def test_cross_repetition_retry_joins_tail_without_barrier(cfg, monkeypatch):
    cfg = configure(cfg)
    run, loaded = prepare(cfg, repetitions=2, tasks=('a', 'b'))
    factory, calls, _, services = fake_runtime(monkeypatch, {'a': ['LimitsExceeded', 'Submitted'], 'b': ['Submitted']})
    result = run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(result) == 4
    assert [(iid, aid[:7]) for iid, aid, *_ in calls[:4]] == [(i['instance_id'], i['repetition_id']) for i in loaded['work_items']['items']]
    assert calls[-1][0:3] == ('a', 'rep-000-attempt-001', 44)
    assert len(services) == 1


def test_same_task_repetitions_can_overlap_and_samples_cannot(cfg, monkeypatch):
    cfg = configure(cfg, workers=2)
    run, _ = prepare(cfg, repetitions=2)
    barrier = threading.Barrier(2, timeout=5)
    live, overlap = set(), []
    lock = threading.Lock()
    def on_run(agent, status):
        rep = agent.env.context.attempt_id.split('-attempt')[0]
        with lock:
            assert rep not in live
            live.add(rep)
        barrier.wait()
        with lock:
            overlap.append(len(live))
            live.remove(rep)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'], on_run=on_run)
    result = run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(result) == 2
    assert max(overlap) == 2
    assert {c[2] for c in calls} == {42, 43}


@pytest.mark.parametrize('status', ['LimitsExceeded', 'reasoning_loop'])
def test_blocking_capture_counts_raw_status_and_never_resamples(cfg, monkeypatch, status):
    cfg = configure(cfg)
    cfg['capture'].update(enabled=True, failure_policy='block_rollout')
    cfg['capture']['export']['enabled'] = False
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    run, _ = prepare(cfg)
    factory, calls, closed, _ = fake_runtime(monkeypatch, [status, 'Submitted'], capture_failure=True)
    result = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert result['agent_exit_status'] == status
    assert result['exit_status'] == 'CaptureError'
    assert result['resample_retry_count'] == 0
    assert len(calls) == len(closed) == 1
    counts = read_json(run / 'rollout/limits-exceeded-counts.json')
    assert counts['resample_trigger_count'] == 1
    assert counts['limits_exceeded_count'] == (status == 'LimitsExceeded')


@pytest.mark.parametrize('boundary', ['cancel', 'deadline'])
def test_stop_at_attempt_boundary_counts_but_does_not_retry(cfg, monkeypatch, boundary):
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    def stop(agent, status):
        if boundary == 'cancel':
            write_json(run / 'execution/cancel-request.json', {'reason': 'test'})
        else:
            factory.deadline.end = 0
    factory, calls, closed, _ = fake_runtime(monkeypatch, ['LimitsExceeded'], on_run=stop)
    run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(calls) == len(closed) == 1
    assert read_json(run / 'status.json')['stage'] == ('cancelled' if boundary == 'cancel' else 'failed')
    counts = read_json(run / 'rollout/limits-exceeded-counts.json')
    assert counts['limits_exceeded_count'] == 1
    progress = read_json(run / 'progress.json')
    assert progress['accepted_samples'] == 0
    assert progress['resample_retry_count'] == 0


def test_default_selection_refuses_pending_but_explicit_intermediate_is_available(cfg, monkeypatch):
    from application.execution.harness import prepare as harness_prepare
    from application.execution.selection import rollout_selection
    from rollout.runner import run_task
    cfg = configure(cfg)
    run, loaded = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['LimitsExceeded'])
    item = loaded['work_items']['items'][0]
    row = loaded['config']['selection']['tasks'][item['task_key']]
    result = run_task(loaded['config'], run.name, item['task_key'], row, factory,
                      {'work_items': loaded['work_items']}, sample_from_item(item))
    evidence = RunEvidence(cfg['project_root'], run.name)
    for selector in (lambda: evidence.select(), lambda: harness_prepare(cfg['project_root'], run.name),
                     lambda: rollout_selection(loaded['config'])):
        with pytest.raises(ValueError, match='not finished'):
            selector()
    assert not (run / 'analysis-selection.json').exists()
    assert not (run / 'harness/selection.json').exists()
    selected = evidence.select([result['rollout_id']])
    assert selected['rollouts'][result['rollout_id']]['selection_role'] == 'explicit_intermediate'
    assert selected['rollouts'][result['rollout_id']]['resolved'] is None


def test_seed_ranges_cover_repetitions_rounds_and_branches(cfg):
    cfg = configure(cfg)
    cfg['parallel'] = {'enabled': True, 'branch_count': 4}
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    _, loaded = prepare(cfg, repetitions=5)
    manifest = loaded['work_items']
    seeds = set()
    for item in manifest['items']:
        sample = sample_from_item(item)
        for round_id in range(20):
            seed = effective_seed(sample, manifest, round_id, 4)
            assert seed == 42 + (item['repetition_index'] + 5 * round_id) * 4
            for b in range(4):
                assert seed + b not in seeds
                seeds.add(seed + b)
    sample = sample_from_item(manifest['items'][0])
    with pytest.raises(ValueError, match='seed_space_exhausted'):
        effective_seed(sample, manifest, 2**63, 4)
    with pytest.raises(ValueError, match='stride'):
        effective_seed(sample, manifest, 0, 5)


def test_disabled_policy_records_counts_and_legacy_snapshot_is_unchanged(cfg, monkeypatch):
    cfg = configure(cfg, retry=False)
    run, _ = prepare(cfg)
    factory, calls, _, _ = fake_runtime(monkeypatch, ['LimitsExceeded'])
    result = next(iter(run_manifest(cfg, run.name, resume=True, factory=factory).values()))
    assert result['exit_status'] == 'LimitsExceeded'
    assert len(calls) == 1
    assert read_json(run / 'rollout/limits-exceeded-counts.json')['limits_exceeded_count'] == 1
    legacy = copy.deepcopy(cfg)
    legacy['execution'].pop('limits_retry')
    legacy['config_hash'] = hash_json({k: v for k, v in legacy.items() if k != 'config_hash'})
    legacy_run, _ = prepare(legacy, 'legacy')
    # Model a pre-release snapshot with no statistics file.
    (legacy_run / 'rollout/limits-exceeded-counts.json').unlink()
    (legacy_run / 'rollout/statistics-policy.json').unlink()
    before = (legacy_run / 'resolved-config.json').read_bytes()
    assert not enabled(legacy)
    assert disposition(legacy, 'LimitsExceeded') == 'accepted'
    run_manifest(legacy, legacy_run.name, resume=True, factory=factory)
    assert (legacy_run / 'resolved-config.json').read_bytes() == before
    assert not (legacy_run / 'rollout/limits-exceeded-counts.json').exists()


def test_strict_config_validates_exact_retry_statuses():
    from pydantic import ValidationError

    from core.config import LimitsRetry
    assert LimitsRetry().enabled is False
    for values in ([], ['LimitsExceeded'], ['reasoning_loop'], ['LimitsExceeded', 'LimitsExceeded'], ['APIError', 'reasoning_loop']):
        with pytest.raises(ValidationError):
            LimitsRetry(retry_statuses=values)
    with pytest.raises(ValidationError):
        LimitsRetry(seed_policy='increment_one')


def test_report_usage_includes_superseded_attempts_and_all_parallel_branches(cfg, monkeypatch):
    from application.execution.report import report
    from rollout import agent
    cfg = configure(cfg)
    run, _ = prepare(cfg)
    factory, _, _, _ = fake_runtime(monkeypatch, ['LimitsExceeded', 'reasoning_loop', 'Submitted'])
    original = agent.DurableAgent.save
    def save(self, path, extra):
        original(self, path, extra)
        raw = read_json(path)
        raw['messages'][0]['extra'] = {
            'cost': 4.0, 'dycon_parallel': {'decision_id': 'decision',
                'total_usage': {'prompt_tokens': 28, 'completion_tokens': 20}},
            'response': {'usage': {'prompt_tokens': 7, 'completion_tokens': 5}}}
        write_json(path, raw)
    monkeypatch.setattr(agent.DurableAgent, 'save', save)
    run_manifest(cfg, run.name, resume=True, factory=factory)
    usage = report(cfg['project_root'], run.name)['all_attempt_usage']
    assert usage['physical_attempts'] == 3
    assert usage['cost'] == 12
    assert usage['prompt_tokens'] == 84
    assert usage['completion_tokens'] == 60
    assert usage['usage_complete']


@pytest.mark.parametrize('operation', ['harness', 'analysis', 'hidden', 'enrichment'])
def test_retry_does_not_change_other_operation_selection(cfg, operation):
    cfg = configure(cfg)
    cfg['operation'] = operation
    assert not enabled(cfg)
    assert disposition(cfg, 'LimitsExceeded') == 'accepted'


@pytest.mark.parametrize("status,attempts_per_slot,discarded_count", [
    ("APIError", 2, 2), ("LimitsExceeded", 4, 1)])
def test_online_sampler_settles_exhausted_slot_and_preserves_healthy_slot(
        cfg, monkeypatch, status, attempts_per_slot, discarded_count):
    from application.execution.run_plan import _selection_from_tasks
    from RL.online_runtime import make_online_run_work_items

    cfg = configure(cfg, infra=1)
    cfg['execution']['online_rl_failure_policy'] = {'baseline_failure_reward': -3.0}
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    root = cfg['project_root']
    Catalog(root).register_tasks([task('a'), task('b')])
    selection = _selection_from_tasks(Catalog(root), task_ids=['a', 'b'], cfg=cfg)
    run_id = 'online-local-failure'
    work = make_online_run_work_items(selection, run_id, m=1, base_seed=cfg['generation']['seed'],
                                      branch_count=1)
    run = create_run(cfg, run_id, operation='rollout', task_ids=['a', 'b'], repeat_count=2,
                     online_work_items=work)
    factory, calls, _, _ = fake_runtime(monkeypatch, {'a': [status], 'b': ['Submitted']})
    outcomes = run_manifest(cfg, run_id, resume=True, factory=factory)
    failures = attempts_per_slot * discarded_count
    accepted_a = 2 - discarded_count
    assert [c[3] for c in calls if c[0] == 'a'] == [status] * (failures + accepted_a)
    assert [c[3] for c in calls if c[0] == 'b'] == ['Submitted'] * 2
    summary = read_json(run / 'status.json')
    assert summary['stage'] == 'complete'
    assert len(summary['discarded_samples']) == discarded_count
    assert all(outcomes[key]['instance_id'] == 'a' for key in summary['discarded_samples'])
    assert len(RunEvidence(root, run_id).select()['rollouts']) == 2 + accepted_a
    assert run_manifest(cfg, run_id, resume=True, factory=factory) == outcomes
    assert len(calls) == failures + accepted_a + 2
