"""Audited, offline append-only expansion of a frozen rollout sample plan."""
import copy
from contextlib import ExitStack
from pathlib import Path

from application.execution.config_revision import _offline
from application.execution.run_plan import load_plan
from core.io import hash_json, identifier, now, read_json, run_path, sha256, write_json
from core.locking import file_lock
from rollout.sampling import make_work_items


def _commit(root, run, audit, request_id):
    journal = read_json(audit / 'progress.json')
    # Recovery accepts only bytes from this transaction, never arbitrary edits.
    for name, hashes in journal['files'].items():
        if sha256(audit / 'before' / name) != hashes['before']:
            raise ValueError('Repetition extension backup changed: ' + name)
        if sha256(audit / 'after' / name) != hashes['after']:
            raise ValueError('Repetition extension proposal changed: ' + name)
        if sha256(run / name) not in {hashes['before'], hashes['after']}:
            raise ValueError('Repetition extension current file changed: ' + name)
    for name, digest in journal['protected'].items():
        if sha256(run / name) != digest:
            raise ValueError('Repetition extension frozen evidence changed: ' + name)
    write_json(audit / 'progress.json', {**journal, 'phase': 'committing'})
    for name in journal['files']:
        write_json(run / name, read_json(audit / 'after' / name))
    loaded = load_plan(root, run.name, allow_incomplete_extension=request_id)
    receipt = {
        'schema_version': 1, 'run_id': run.name, 'request_id': request_id,
        'request': read_json(audit / 'request.json'),
        'previous_execution': journal['previous_execution'],
        'old_plan_sha256': read_json(audit / 'before/plan.json')['plan_sha256'],
        'new_plan_sha256': loaded['plan']['plan_sha256'],
        'old_repeat_count': read_json(audit / 'before/plan.json')['repeat_count'],
        'new_repeat_count': loaded['plan']['repeat_count'],
        'added_sample_count': journal['added_sample_count'],
        'files': journal['files'], 'completed_at': now(),
    }
    write_json(audit / 'receipt.json', receipt)
    write_json(run / 'events' / ('repetitions-added-' + request_id + '.json'),
               {'event': 'run.repetitions_added', **receipt})
    write_json(audit / 'progress.json', {**journal, 'phase': 'complete'})
    from application.execution.status import refresh_run_status_summary
    refresh_run_status_summary(root, run.name)
    return receipt


def add_repetitions(root, run_id, *, request_id, count=None, reason,
                    dry_run=False, recover=False, expect_plan_hash=None):
    """Append COUNT repetitions to every selected task; resume launches them."""
    root = Path(root).resolve()
    run_id, request_id = identifier(run_id), identifier(request_id)
    if not reason or not reason.strip():
        raise ValueError('Explicit reason is required')
    if recover and (dry_run or count is not None or expect_plan_hash is not None):
        raise ValueError('--recover cannot be combined with count, preview or expected hash')
    if not recover and (type(count) is not int or count < 1):
        raise ValueError('count must be a positive integer')
    run = run_path(root, run_id)
    audit = run / 'repetition-extensions' / request_id
    with ExitStack() as locks:
        for path in [root / 'state/run-locks' / (run_id + '.lock'),
                     run / 'execution/supervisor.lock', run / 'execution/active-supervisor.lock']:
            locks.enter_context(file_lock(path, nonblocking=True))
        if recover:
            _offline(run)
            if not (audit / 'progress.json').exists():
                # No journal means preparation never modified the live plan.
                load_plan(root, run_id)
                raise ValueError('No prepared repetition extension to recover; use a new request ID')
            request = read_json(audit / 'request.json')
            if request['reason'] != reason:
                raise ValueError('Recovery reason must match the original request')
            journal = read_json(audit / 'progress.json')
            if journal['phase'] == 'complete':
                load_plan(root, run_id)
                return read_json(audit / 'receipt.json')
            return _commit(root, run, audit, request_id)
        loaded = load_plan(root, run_id)
        if loaded['plan']['operation'] != 'rollout':
            raise ValueError('Adding repetitions requires a rollout run')
        request = {'count': count, 'reason': reason}
        if (audit / 'receipt.json').exists():
            receipt = read_json(audit / 'receipt.json')
            if receipt['request'] != request:
                raise ValueError('Repetition request ID binds different count/reason')
            return receipt
        if audit.exists():
            raise RuntimeError('Incomplete extension preparation; recover it or use a new request ID')
        if expect_plan_hash and expect_plan_hash != loaded['plan']['plan_sha256']:
            raise ValueError('Run plan changed since preview')
        cfg, plan = loaded['config'], copy.deepcopy(loaded['plan'])
        old_count = plan['repeat_count']
        new_count = old_count + count
        extension = plan.get('sampling_extension') or {
            'original_repeat_count': old_count, 'seed_block_size': 2**32}
        work = make_work_items(cfg['selection'], run_id, new_count, cfg['generation']['seed'],
                               cfg.get('parallel', {}).get('branch_count', 1), extension=extension)
        old_items = loaded['work_items']['items']
        if work['items'][:len(old_items)] != old_items:
            raise ValueError('Repetition extension would change existing sample identities')
        work['items_sha256'] = hash_json(work['items'])
        plan.update(repeat_count=new_count, sampling_extension=extension,
                    previous_plan_sha256=plan['plan_sha256'], repetition_extension=request_id)
        plan['plan_sha256'] = hash_json({k: v for k, v in plan.items() if k != 'plan_sha256'})
        preview = {'run_id': run_id, 'request_id': request_id, 'dry_run': dry_run,
                   'old_repeat_count': old_count, 'new_repeat_count': new_count,
                   'added_sample_count': len(work['items']) - len(old_items),
                   'old_plan_sha256': loaded['plan']['plan_sha256'],
                   'new_plan_sha256': plan['plan_sha256'],
                   'new_repetition_ids': sorted({i['repetition_id'] for i in work['items'][len(old_items):]}),
                   'sampling_extension': extension}
        if dry_run:
            return preview
        execution = _offline(run)
        proposed = {'plan.json': plan, 'rollout/work-items.json': work}
        journal = {'phase': 'prepared', 'previous_execution': execution,
                   'added_sample_count': preview['added_sample_count'], 'files': {},
                   'protected': {name: sha256(run / name) for name in
                                 ['resolved-config.json', 'selection.json', 'tasks.json']}}
        for name, value in proposed.items():
            backup = audit / 'before' / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes((run / name).read_bytes())
            write_json(audit / 'after' / name, value)
            journal['files'][name] = {'before': sha256(backup),
                                     'after': sha256(audit / 'after' / name)}
        write_json(audit / 'request.json', request)
        write_json(audit / 'progress.json', journal)
        return _commit(root, run, audit, request_id)
