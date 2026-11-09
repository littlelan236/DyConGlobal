"""Explicit offline requests supersede failed cursors, never historical evidence."""
import shutil
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from application.execution.run_plan import load_plan
from application.execution.status import refresh_run_status_summary
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from core.processes import matches
from rollout.retry import publish_once


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
                key = old['attempt_id']
                if key is None:
                    key = 'result:' + old['outcome_sha256']
                previous = attempts.setdefault(key, old)
                identity = ('attempt_id', 'outcome_sha256', 'source_path', 'source_sha256')
                if (old['attempt_id'] is None
                        and old.get('source_kind') == previous.get('source_kind')
                        == 'progress_cursor_without_result'):
                    # Other samples can change the enclosing progress file while
                    # this cursor's outcome and source location stay identical.
                    identity = ('attempt_id', 'outcome_sha256', 'source_path')
                if any(previous.get(k) != old.get(k) for k in identity):
                    raise ValueError('Conflicting operator retry attempt identity')
                if old['attempt_id'] is None:
                    archived = read_json(Path(run) / item['previous_result_archive'])
                    if (archived.get('sample_key') != item['sample_key']
                            or archived.get('attempt_id') is not None
                            or archived.get('exit_status') != 'WorkerFailure'
                            or hash_json(archived) != old['outcome_sha256']):
                        raise ValueError('Operator retry archived worker failure changed')
                    authorization = {
                        'request_id': request['request_id'],
                        'request_sha256': expected,
                        'requested_at': request['requested_at'],
                        'previous_execution': request.get('previous_execution'),
                        'outcome': archived,
                        'incomplete_attempts': item.get('superseded_incomplete_attempts'),
                    }
                    current = previous.get('worker_failure_authorization')
                    if current is None or datetime.fromisoformat(authorization['requested_at']) > datetime.fromisoformat(current['requested_at']):
                        previous['worker_failure_authorization'] = authorization
    return result


def superseded(index, sample_key, outcome):
    key = outcome.get('attempt_id')
    if key is None:
        key = 'result:' + hash_json(outcome)
    old = index.get(sample_key, {}).get(key)
    if old is None:
        return False
    if outcome.get('sample_key') != sample_key or hash_json(outcome) != old['outcome_sha256']:
        raise ValueError('Operator retry source outcome changed')
    return True


def worker_failure_authorization(index, sample_key):
    """Read the authorized cursor from its archive, not mutable progress.json."""
    candidates = [old['worker_failure_authorization']
                  for old in index.get(sample_key, {}).values()
                  if 'worker_failure_authorization' in old]
    return max(candidates, key=lambda value: datetime.fromisoformat(value['requested_at']), default=None)


def supersede_incomplete_attempt(run, sample_key, authorization, out, context):
    """Bind an explicit worker retry to old incomplete evidence, preserving it."""
    if authorization is None:
        return False
    out = Path(out)
    started_path = out / 'attempt-started.json'
    context_path = out / 'attempt-context.json'
    if not started_path.exists():
        return False
    started = read_json(started_path)
    if started.get('attempt_id') != context['attempt_id']:
        raise ValueError('Incomplete retry physical attempt identity changed')
    # A request never authorizes skipping a fresh attempt created after it.
    if datetime.fromisoformat(started['started_at']) > datetime.fromisoformat(authorization['requested_at']):
        return False
    generation = started.get('execution_attempt_id')
    previous = authorization['previous_execution']
    if generation is not None and previous is not None and int(generation) > int(previous):
        return False
    snapshots = authorization['incomplete_attempts']
    if snapshots is not None:
        bound = next((item for item in snapshots if item['attempt_id'] == context['attempt_id']), None)
        if bound is None:
            return False
        if (sha256(context_path) != bound['context_sha256']
                or sha256(started_path) != bound['started_sha256']):
            raise ValueError('Incomplete operator retry source changed')
    # Legacy requests bind the archived WorkerFailure and its timestamp instead
    # of a directory snapshot. Freeze that resolution before starting new work.
    receipt = {
        'schema_version': 1, 'request_id': authorization['request_id'],
        'request_sha256': authorization['request_sha256'], 'sample_key': sample_key,
        'attempt_id': context['attempt_id'], 'context': context,
        'context_sha256': sha256(context_path), 'started': started,
        'started_sha256': sha256(started_path),
        'previous_outcome_sha256': hash_json(authorization['outcome']),
        'policy': 'skip_authorized_incomplete_recovery_preserve_source_bytes',
    }
    publish_once(Path(run) / 'rollout/operator-retry-consumptions'
                 / authorization['request_id'] / sample_key / (context['attempt_id'] + '.json'), receipt)
    return True


def request_failed_retry(root, run_id, *, request_id, exit_statuses, task_ids=(), sample_keys=(), expected_attempts=None, reason, accepted_capture_partial=False):
    root = Path(root).resolve()
    run_id, request_id = identifier(run_id), identifier(request_id)
    if not reason or not reason.strip() or (not exit_statuses and not accepted_capture_partial):
        raise ValueError('Explicit reason and exit statuses or accepted capture partial selector are required')
    selectors = {'exit_statuses': sorted(set(exit_statuses)), 'task_ids': sorted(set(task_ids))}
    if accepted_capture_partial:
        selectors['accepted_capture_partial'] = True

    def partial(outcome):
        return (accepted_capture_partial and outcome.get('disposition') == 'accepted'
                and outcome.get('capture_status') == 'partial')

    def eligible(outcome):
        return outcome.get('disposition') in {'failed', 'blocked'} or partial(outcome)

    expected_attempts = dict(expected_attempts or {})
    if not set(expected_attempts).issubset(set(sample_keys)):
        raise ValueError('Expected attempts must be a subset of selected sample keys')
    if sample_keys:
        selectors['sample_keys'] = sorted(set(sample_keys))
    if expected_attempts:
        selectors['expected_attempts'] = expected_attempts
    run = root / 'runs' / run_id
    target = run / 'rollout/operator-retries' / request_id / 'request.json'
    with ExitStack() as locks:
        for path in [root / 'state/run-locks' / (run_id + '.lock'), run / 'execution/supervisor.lock', run / 'execution/active-supervisor.lock']:
            locks.enter_context(file_lock(path, nonblocking=True))
        loaded = load_plan(root, run_id)
        progress_path = run / 'progress.json'
        progress_samples = (
            read_json(progress_path).get('samples', {}) if progress_path.exists() else {}
        )
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
            refresh_run_status_summary(root, run_id)
            return old
        # Validate all exact selectors before archiving or publishing any request.
        if sample_keys:
            selected = {item['sample_key']: item for item in loaded['work_items']['items']
                        if item['sample_key'] in sample_keys}
            if set(selected) != set(sample_keys):
                raise ValueError('Unknown exact sample key')
            for key, item in selected.items():
                if task_ids and item['instance_id'] not in task_ids:
                    raise ValueError('Exact sample does not match task selector')
                path = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id'] / 'result.json'
                outcome = read_json(path) if path.exists() else progress_samples.get(key)
                if outcome is None:
                    raise ValueError('Exact sample has no terminal outcome')
                if outcome.get('exit_status') not in exit_statuses and not partial(outcome):
                    raise ValueError('Exact sample does not match failure status')
                if key in expected_attempts and outcome.get('attempt_id') != expected_attempts[key]:
                    raise ValueError('Exact sample current attempt changed')
                if not eligible(outcome):
                    raise ValueError('Only failed/blocked results or explicitly selected accepted capture partial results may be retried')
        items = []
        for item in loaded['work_items']['items']:
            if sample_keys and item['sample_key'] not in sample_keys:
                continue
            if task_ids and item['instance_id'] not in task_ids:
                continue
            work = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id']
            p = work / 'result.json'
            outcome = read_json(p) if p.exists() else progress_samples.get(item['sample_key'])
            if outcome is None:
                continue
            if outcome.get('exit_status') not in exit_statuses and not partial(outcome):
                continue
            if outcome.get('sample_key') != item['sample_key']:
                raise ValueError('Retry source sample identity mismatch')
            if not eligible(outcome):
                raise ValueError('Only failed/blocked results or explicitly selected accepted capture partial results may be retried')
            superseded_attempts = []
            for attempt in sorted((work / 'attempts').glob('*/attempt.json')):
                value = read_json(attempt)
                if not eligible(value) and not (partial(outcome) and value.get('disposition') == 'accepted'):
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
            attempt_id = outcome.get('attempt_id')
            if attempt_id is None:
                if outcome.get('exit_status') != 'WorkerFailure':
                    raise ValueError('Only WorkerFailure without a physical attempt may be retried')
                superseded_attempts.append({
                    'attempt_id': None,
                    'outcome_sha256': hash_json(outcome),
                    'source_path': (
                        str(p.relative_to(run)) if p.exists()
                        else 'progress.json#samples/' + item['sample_key']
                    ),
                    'source_sha256': sha256(p) if p.exists() else sha256(progress_path),
                    'source_kind': (
                        'result_without_physical_attempt' if p.exists()
                        else 'progress_cursor_without_result'
                    ),
                })
            else:
                if attempt_id not in {v['attempt_id'] for v in superseded_attempts}:
                    raise ValueError('Failed result has no matching durable physical attempt')
                selected = next(v for v in superseded_attempts if v['attempt_id'] == attempt_id)
                if selected['outcome_sha256'] != hash_json(outcome):
                    raise ValueError('Canonical failed cursor and physical outcome disagree')
            archived_result = target.parent / 'before' / item['sample_key'] / 'result.json'
            archived_result.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                shutil.copyfile(p, archived_result)
                source_result_sha256 = sha256(p)
            else:
                write_json(archived_result, outcome)
                source_result_sha256 = sha256(progress_path)
            if hash_json(read_json(archived_result)) != hash_json(outcome):
                raise ValueError('Retry source result changed while archiving')
            incomplete = []
            if attempt_id is None:
                for context_path in sorted((work / 'attempts').glob('*/attempt-context.json')):
                    started_path = context_path.parent / 'attempt-started.json'
                    if (context_path.parent / 'attempt.json').exists() or not started_path.exists():
                        continue
                    context = read_json(context_path)
                    if context.get('sample_key') != item['sample_key'] or context.get('attempt_id') != context_path.parent.name:
                        raise ValueError('Incomplete retry source sample identity mismatch')
                    incomplete.append({'attempt_id': context['attempt_id'],
                                       'context_sha256': sha256(context_path),
                                       'started_sha256': sha256(started_path)})
            items.append({**item, 'previous_exit_status': outcome['exit_status'], 'previous_attempt_id': outcome.get('attempt_id'),
                          'previous_result_archive': str(archived_result.relative_to(run)), 'previous_result_sha256': source_result_sha256,
                          'superseded_attempts': superseded_attempts,
                          **({'superseded_incomplete_attempts': incomplete} if attempt_id is None else {})})
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
        refresh_run_status_summary(root, run_id)
        return request
