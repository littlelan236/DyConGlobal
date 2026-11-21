"""Versioned sample dispositions and durable, idempotent agent outcome counts."""
from __future__ import annotations

from pathlib import Path

from core.io import hash_json, now, read_json, write_json
from core.locking import file_lock

RETRY_STATUSES = frozenset({'LimitsExceeded', 'reasoning_loop'})
SEED_POLICY = 'repetition_round_offset_v1'
TERMINAL = frozenset({'Submitted', 'reasoning_loop', 'LimitsExceeded',
                      'ContextWindowExceededError', 'RepeatedFormatError', 'TimeExceeded', 'AgentCompleted', 'AgentTimeout'})
INFRASTRUCTURE = frozenset({'ContainerCorrupted', 'EnvironmentSetupFailed', 'APIError',
    'Timeout', 'ReadTimeout', 'ConnectError', 'APIConnectionError', 'DockerException',
    'NotFound', 'ConnectionError', 'Interrupted'})
COUNT_FIELDS = ('limits_exceeded_count', 'reasoning_loop_count', 'resample_trigger_count')


def enabled(cfg):
    if cfg.get('operation') not in (None, 'rollout'):
        return False
    policy = cfg.get('execution', {}).get('limits_retry', {})
    if policy.get('enabled', False):
        if (policy.get('seed_policy') != SEED_POLICY or
                set(policy.get('retry_statuses', ())) != RETRY_STATUSES):
            raise ValueError('Unsupported frozen rollout retry policy')
        return True
    return False


def resampling_enabled(cfg):
    return enabled(cfg) or cfg.get('execution', {}).get('selection_retry', {}).get('enabled', False)


def disposition(cfg, status):
    if status == 'Cancelled':
        return 'interrupted'
    if enabled(cfg) and status in RETRY_STATUSES:
        return 'retry_pending'
    if status == 'ParallelSelectionError' and cfg.get('execution', {}).get('selection_retry', {}).get('enabled', False):
        return 'selection_retry'
    if status in TERMINAL:
        return 'accepted'
    if status in INFRASTRUCTURE:
        return 'infrastructure_retry'
    return 'failed'


def sample_accepted(cfg, result):
    from RL.online_failures import baseline_failure_reward
    if (result.get('exit_status') in RETRY_STATUSES
            and baseline_failure_reward(cfg, result.get('online_source'), result) is not None):
        return True
    return disposition(cfg, result.get('exit_status')) == 'accepted'


def sample_settled(cfg, result):
    return sample_accepted(cfg, result) or result.get('disposition') == 'failed' or result.get('exit_status') == 'infrastructure_exhausted'


def sample_work(run, sample):
    return Path(run) / 'rollout/tasks' / sample.instance_id / 'repetitions' / sample.repetition_id


def publish_once(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Conflicting immutable outcome: {path}')
    else:
        write_json(path, value)


def _empty_counts(run_id, items):
    zero = dict.fromkeys(COUNT_FIELDS, 0)
    result = {'schema_version': 1, 'run_id': run_id, 'updated_at': now(),
              **zero, 'tasks': {}, 'samples': {}}
    for item in items:
        result['tasks'].setdefault(item['task_key'], {'instance_id': item['instance_id'], **zero})
        result['samples'][item['sample_key']] = {
            'task_key': item['task_key'], 'repetition_id': item['repetition_id'],
            **zero, 'counted_attempts': {}}
    return result


def initialize_counts(run, manifest):
    path = Path(run) / 'rollout/limits-exceeded-counts.json'
    with file_lock(path.with_suffix('.lock')):
        publish_once(Path(run) / 'rollout/statistics-policy.json', {
            'schema_version': 1, 'run_id': manifest['run_id'], 'counts_file': path.name})
        if not path.exists():
            write_json(path, _empty_counts(manifest['run_id'], manifest['items']))


def tracks_counts(cfg):
    # The schema field is frozen only in runs created after this release.
    return 'limits_retry' in cfg.get('execution', {})


def counts_recorded(run):
    run = Path(run)
    return ((run / 'rollout/statistics-policy.json').exists() or
            (run / 'rollout/limits-exceeded-counts.json').exists())


def _add_outcome(counts, value):
    if value['run_id'] != counts['run_id']:
        raise ValueError('Agent outcome belongs to another run')
    sample = counts['samples'][value['sample_key']]
    if (sample['task_key'], sample['repetition_id']) != (value['task_key'], value['repetition_id']):
        raise ValueError('Agent outcome sample binding changed')
    seen = sample['counted_attempts']
    aid = value['attempt_id']
    digest = hash_json(value)
    if aid in seen:
        if seen[aid] != digest:
            raise ValueError('Conflicting already-counted agent outcome')
        return
    seen[aid] = digest
    status = value['agent_exit_status']
    field = {'LimitsExceeded': 'limits_exceeded_count', 'reasoning_loop': 'reasoning_loop_count'}.get(status)
    if field:
        for row in (counts, counts['tasks'][value['task_key']], sample):
            row[field] += 1
            row['resample_trigger_count'] += 1


def record_agent_outcome(run, out, context, status):
    path = Path(out) / 'agent-outcome.json'
    with file_lock(Path(out) / 'agent-outcome.lock'):
        value = {**context, 'agent_exit_status': status}
        if path.exists():
            old = read_json(path)
            if any(old.get(k) != v for k, v in value.items()):
                raise ValueError('Conflicting agent exit fact')
            value = old
        else:
            value['agent_finished_at'] = now()
            publish_once(path, value)
    counts_path = Path(run) / 'rollout/limits-exceeded-counts.json'
    with file_lock(counts_path.with_suffix('.lock')):
        counts = read_json(counts_path)
        _add_outcome(counts, value)
        counts['updated_at'] = now()
        write_json(counts_path, counts)
    return value


def rebuild_counts(run, manifest):
    """Startup/final reconciliation only; normal updates never scan trajectories."""
    run = Path(run)
    path = run / 'rollout/limits-exceeded-counts.json'
    from application.execution.evidence import RunEvidence
    canonical = RunEvidence(run.parent.parent, manifest['run_id']).read()['rollouts']
    by_attempt = {}
    for rid, row in canonical.items():
        if not row.get('sample_key'):
            continue
        key = (row.get('sample_key'), row.get('attempt_id'))
        if key in by_attempt:
            raise ValueError('Duplicate canonical physical attempt')
        by_attempt[key] = row
    with file_lock(path.with_suffix('.lock')):
        value = _empty_counts(manifest['run_id'], manifest['items'])
        for item in manifest['items']:
            work = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']
            for context_path in sorted((work / 'attempts').glob('*/attempt-context.json')):
                context = read_json(context_path)
                for key in ('run_id', 'task_key', 'sample_key', 'repetition_id', 'sample_seed'):
                    if context.get(key) != item.get(key, manifest['run_id'] if key == 'run_id' else None):
                        raise ValueError('Attempt context does not match frozen sample')
                outcome = context_path.with_name('agent-outcome.json')
                if not outcome.exists():
                    source = context_path.with_name('raw.traj.json')
                    row = by_attempt.get((item['sample_key'], context['attempt_id']))
                    if row is not None:
                        source = run.parent.parent / row['trajectory_path']
                        from core.io import sha256
                        if sha256(source) != row['trajectory_sha256']:
                            raise ValueError('Canonical bytes changed during outcome recovery')
                    if source.exists():
                        raw = read_json(source)
                        info = raw.get('info', {})
                        if (info.get('exit_status') in TERMINAL and
                                info.get('dycon', {}).get('attempt_context') == context and
                                info.get('attempt_id') == context['attempt_id'] and
                                raw.get('instance_id') == item['instance_id']):
                            with file_lock(context_path.with_name('agent-outcome.lock')):
                                if not outcome.exists():
                                    publish_once(outcome, {**context, 'agent_exit_status': info['exit_status'],
                                        'agent_finished_at': now(), 'reconstructed_from_trajectory': True})
                if not outcome.exists():
                    continue
                fact = read_json(outcome)
                context = read_json(outcome.with_name('attempt-context.json'))
                if any(fact.get(k) != v for k, v in context.items()):
                    raise ValueError('Agent outcome differs from reserved attempt context')
                _add_outcome(value, fact)
        write_json(path, value)
    return value


def verify_final(root, run_id, item, result, evidence):
    rid = result.get('rollout_id')
    if not rid or rid not in evidence['rollouts']:
        raise ValueError('Accepted sample has no canonical rollout')
    row = evidence['rollouts'][rid]
    for key in ('sample_key', 'task_key', 'repetition_id'):
        if row.get(key) != item[key]:
            raise ValueError('Accepted canonical sample identity changed')
    if row.get('attempt_id') != result.get('attempt_id'):
        raise ValueError('Accepted canonical attempt pointer changed')
    for key in ('sample_seed', 'effective_seed', 'seed_round'):
        if key in row and row[key] != result.get(key):
            raise ValueError('Accepted canonical execution provenance changed')
    if row.get('termination_reason') != result.get('agent_exit_status', result.get('exit_status')):
        raise ValueError('Accepted canonical terminal status changed')
    from core.io import owned_path, sha256
    from application.execution.subject import artifact_kinds, verify_subject
    verify_subject(root, row)
    for kind in artifact_kinds(row):
        if sha256(owned_path(root, row[kind + '_path'])) != row[kind + '_sha256']:
            raise ValueError('Completed canonical bytes changed')
    return rid


def default_final_ids(root, run_id, data):
    """Return None for legacy semantics; new runs require settled producer inputs."""
    from application.execution.run_plan import load_plan
    from core.config import load_snapshot
    run = Path(root) / 'runs' / run_id
    if not (run / 'resolved-config.json').exists():
        return None
    cfg = load_snapshot(root, run_id)
    if not enabled(cfg):
        return None
    stage = read_json(run / 'status.json').get('stage')
    if stage not in {'complete', 'failed', 'cancelled'}:
        raise ValueError('Source rollout run is not finished; use explicit rollout IDs for partial analysis')
    ids = []
    for item in load_plan(root, run_id)['work_items']['items']:
        path = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id'] / 'result.json'
        if not path.exists():
            if stage == 'complete':
                raise ValueError('Complete source run has pending samples')
            continue
        result = read_json(path)
        if sample_accepted(cfg, result):
            ids.append(verify_final(root, run_id, item, result, data))
        elif stage == 'complete':
            raise ValueError('Complete source run contains an unaccepted sample')
    if len(ids) != len(set(ids)):
        raise ValueError('Multiple samples point to the same canonical rollout')
    return sorted(ids)


def count_projection(run):
    path = Path(run) / 'rollout/limits-exceeded-counts.json'
    if not path.exists():
        if counts_recorded(run):
            return {'recorded': False, 'needs_rebuild': True, 'message': 'statistics missing; resume to recover', 'path': str(path)}
        return {'recorded': False, 'message': 'not recorded (legacy run)', 'path': str(path)}
    value = read_json(path)
    return {'recorded': True, 'path': str(path), **value}


def sampling_notice(cfg):
    return ('Samples are accepted after excluding LimitsExceeded / reasoning_loop terminations; '
            'this is conditional resampling, not unconditional fixed-seed sampling. '
            'Costs and time cover every physical attempt.') if enabled(cfg) else None


def verify_counts(run):
    run = Path(run)
    if not counts_recorded(run):
        return {'recorded': False}
    manifest = read_json(run / 'rollout/work-items.json')
    expected = _empty_counts(manifest['run_id'], manifest['items'])
    for item in manifest['items']:
        work = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']
        for outcome in (work / 'attempts').glob('*/agent-outcome.json'):
            fact = read_json(outcome)
            context = read_json(outcome.with_name('attempt-context.json'))
            if any(fact.get(k) != v for k, v in context.items()):
                raise ValueError('Agent outcome context changed')
            _add_outcome(expected, fact)
    actual = read_json(run / 'rollout/limits-exceeded-counts.json')
    for key in (*COUNT_FIELDS, 'run_id', 'schema_version', 'tasks', 'samples'):
        if actual.get(key) != expected[key]:
            raise ValueError('Agent outcome statistics do not match durable facts: ' + key)
    return {'recorded': True, 'verified': True, **{key: actual[key] for key in COUNT_FIELDS}}


def attempt_usage(run):
    """Report all attempts, never just the selected final canonical trajectories."""
    result = {'physical_attempts': 0, 'agent_seconds': 0.0, 'capture_finalize_seconds': 0.0,
              'cost': 0.0, 'prompt_tokens': 0, 'completion_tokens': 0, 'usage_recorded_attempts': 0}
    result['incomplete_attempts'] = 0
    for directory in (Path(run) / 'rollout/tasks').glob('*/repetitions/*/attempts/*'):
        path = directory / 'attempt.json'
        if not path.exists() and not (directory / 'attempt-started.json').exists():
            continue
        if path.exists():
            record = read_json(path)
        else:
            state = directory / 'execution-state.json'
            record = read_json(state) if state.exists() else {}
            result['incomplete_attempts'] += 1
        result['physical_attempts'] += 1
        for field in ('agent_seconds', 'capture_finalize_seconds'):
            result[field] += record.get('timings', {}).get(field, 0.0)
        usage = record.get('usage')
        if usage and usage.get('complete', True):
            result['usage_recorded_attempts'] += 1
            for key in ('cost', 'prompt_tokens', 'completion_tokens'):
                value = usage.get(key, 0)
                if value is None:
                    result[key] = None
                elif result[key] is not None:
                    result[key] += value
    result['usage_complete'] = result['usage_recorded_attempts'] == result['physical_attempts']
    if not result['usage_complete']:
        result['known_usage'] = {key: result[key] for key in ('cost', 'prompt_tokens', 'completion_tokens')}
        for key in ('cost', 'prompt_tokens', 'completion_tokens'):
            result[key] = None
    return result
