"""Explicit offline requests supersede failed cursors, never historical evidence."""
from contextlib import ExitStack
from pathlib import Path
import shutil

from application.execution.run_plan import load_plan
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from core.processes import matches
from rollout.retry import sample_accepted


def retry_index(run):
    """Index immutable requests by sample and physical attempt identity."""
    result = {}
    for path in sorted((Path(run) / 'rollout/operator-retries').glob('*/request.json')):
        request = read_json(path)
        expected = request.get('request_sha256')
        if request.get('run_id') != Path(run).name or expected != hash_json({k: v for k, v in request.items() if k != 'request_sha256'}):
            raise ValueError('Operator retry request integrity mismatch')
        for item in request['items']:
            attempts = result.setdefault(item['sample_key'], {})
            for old in item['superseded_attempts']:
                previous = attempts.setdefault(old['attempt_id'], old)
                identity = ('attempt_id', 'outcome_sha256', 'source_path', 'source_sha256')
                if any(previous.get(k) != old.get(k) for k in identity):
                    raise ValueError('Conflicting operator retry attempt identity')
    return result


def superseded(index, sample_key, outcome):
    old = index.get(sample_key, {}).get(outcome.get('attempt_id'))
    if old is None:
        return False
    if outcome.get('sample_key') != sample_key or hash_json(outcome) != old['outcome_sha256']:
        raise ValueError('Operator retry source outcome changed')
    return True


def request_failed_retry(root, run_id, *, request_id, exit_statuses, task_ids=(), reason):
    root = Path(root).resolve()
    run_id, request_id = identifier(run_id), identifier(request_id)
    if not reason or not reason.strip() or not exit_statuses:
        raise ValueError('Explicit reason and failure exit statuses are required')
    selectors = {'exit_statuses': sorted(set(exit_statuses)), 'task_ids': sorted(set(task_ids))}
    run = root / 'runs' / run_id
    target = run / 'rollout/operator-retries' / request_id / 'request.json'
    with ExitStack() as locks:
        for path in [root / 'state/run-locks' / (run_id + '.lock'), run / 'execution/supervisor.lock', run / 'execution/active-supervisor.lock']:
            locks.enter_context(file_lock(path, nonblocking=True))
        loaded = load_plan(root, run_id)
        if loaded['plan']['operation'] != 'rollout':
            raise ValueError('Failed retry requires a rollout run')
        latest = run / 'execution/latest.json'
        generation = read_json(latest)['attempt'] if latest.exists() else None
        if generation:
            previous = run / 'execution/attempts' / identifier(generation)
            for name in ['process.json', 'supervisor.json']:
                if (previous / name).exists() and matches(read_json(previous / name)):
                    raise RuntimeError('Execution still active; cancel/wait before retrying')
            if not (previous / 'exit.json').exists() or not (previous / 'termination.json').exists() or not read_json(previous / 'termination.json').get('cleanup_complete'):
                raise RuntimeError('Previous execution exit/cleanup is incomplete')
        # This operation writes only this run's retry audit. The matching process,
        # supervisor locks and cleanup receipt above are its offline boundary.
        # A project-wide spool relocation guard would incorrectly block on
        # unrelated live runs even though no shared capture storage is changed.
        if target.exists():
            old = read_json(target)
            retry_index(run)
            if old['selectors'] != selectors or old['reason'] != reason:
                raise ValueError('Retry request ID binds different selectors/reason')
            return old
        items = []
        for item in loaded['work_items']['items']:
            if task_ids and item['instance_id'] not in task_ids:
                continue
            work = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']
            p = work / 'result.json'
            if not p.exists():
                continue
            outcome = read_json(p)
            if outcome.get('exit_status') not in exit_statuses:
                continue
            if outcome.get('sample_key') != item['sample_key']:
                raise ValueError('Retry source sample identity mismatch')
            if sample_accepted(loaded['config'], outcome) or outcome.get('disposition') not in {'failed', 'blocked'}:
                raise ValueError('Only failed/blocked results may be explicitly retried')
            superseded_attempts = []
            for attempt in sorted((work / 'attempts').glob('*/attempt.json')):
                value = read_json(attempt)
                if value.get('disposition') not in {'failed', 'blocked'}:
                    continue
                if value.get('sample_key') != item['sample_key']:
                    raise ValueError('Retry attempt sample identity mismatch')
                # Freeze each source byte sequence outside the live attempt directory.
                archived = target.parent / 'before' / item['sample_key'] / (value['attempt_id'] + '.json')
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(attempt, archived)
                digest = sha256(attempt)
                if sha256(archived) != digest:
                    raise ValueError('Retry source changed while archiving')
                superseded_attempts.append({'attempt_id': value['attempt_id'], 'outcome_sha256': hash_json(value),
                    'source_path': str(attempt.relative_to(run)), 'source_sha256': digest,
                    'archive_path': str(archived.relative_to(run))})
            if outcome.get('attempt_id') not in {v['attempt_id'] for v in superseded_attempts}:
                raise ValueError('Failed result has no matching durable physical attempt')
            selected = next(v for v in superseded_attempts if v['attempt_id'] == outcome['attempt_id'])
            if selected['outcome_sha256'] != hash_json(outcome):
                raise ValueError('Canonical failed cursor and physical outcome disagree')
            archived_result = target.parent / 'before' / item['sample_key'] / 'result.json'
            shutil.copyfile(p, archived_result)
            if sha256(archived_result) != sha256(p):
                raise ValueError('Retry source result changed while archiving')
            items.append({**item, 'previous_exit_status': outcome['exit_status'], 'previous_attempt_id': outcome['attempt_id'],
                          'previous_result_archive': str(archived_result.relative_to(run)), 'previous_result_sha256': sha256(p),
                          'superseded_attempts': superseded_attempts})
        request = {'schema_version': 1, 'run_id': run_id, 'request_id': request_id,
            'reason': reason, 'selectors': selectors, 'previous_execution': generation,
            'config_hash': loaded['config']['config_hash'], 'selection_sha256': loaded['plan']['selection_sha256'],
            'requested_at': now(), 'items': items, 'sample_count': len(items),
            'policy': 'one_fresh_physical_attempt_same_seed_round_preserve_all_source_evidence'}
        request['request_sha256'] = hash_json(request)
        write_json(target, request)
        write_json(run / 'events' / ('operator-retry-' + request_id + '.json'),
                   {'event': 'run.failed_retry_requested', 'request_id': request_id,
                    'sample_count': len(items), 'request_sha256': request['request_sha256'], 'at': now()})
        return request
