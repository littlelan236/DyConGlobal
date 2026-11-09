"""Durable supervision, execution-scoped cancellation and bounded resource cleanup."""
from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from core.cancellation import policy, request_for
from core.io import identifier, now, read_json, write_json
from core.locking import directory_changes, file_lock
from core.processes import enable_subreaper, identity, matches, members, signal_group

from .run_plan import load_plan
from .status import update_run_status


def _apply_open_file_limit(cfg):
    """Raise the supervisor soft FD limit and report effective bounds."""
    import resource

    configured = int(cfg.get('execution', {}).get('supervisor_max_open_files', 1024))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if configured > hard:
        raise RuntimeError(
            f'execution.supervisor_max_open_files={configured} exceeds hard RLIMIT_NOFILE={hard}'
        )
    original_soft = soft
    if configured > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (configured, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return {
        'configured_max_open_files': configured,
        'soft_limit': soft,
        'hard_limit': hard,
        'raised': soft > original_soft,
    }


def launch(root, run_id, *, resume=False, detach=False, check_only=False, task_ids=()):
    root = Path(root).resolve()
    run = load_plan(root, run_id)['run']
    attempts = run / 'execution/attempts'
    with file_lock(run / 'execution/supervisor.lock', nonblocking=True):
        latest = run / 'execution/latest.json'
        if latest.exists():
            previous = attempts / read_json(latest)['attempt']
            if not (previous / 'exit.json').exists():
                process = read_json(previous / 'supervisor.json')
                if matches(process):
                    raise RuntimeError('Run already has a live supervisor')
                raise RuntimeError('Previous supervisor has no exit receipt; reconcile its resources first')
            termination = previous / 'termination.json'
            if termination.exists() and not read_json(termination).get('cleanup_complete', False):
                raise RuntimeError('Previous execution resource cleanup is incomplete')
        request = run / 'execution/cancel-request.json'
        if request.exists():
            if not resume:
                raise RuntimeError('Cancelled run requires explicit run resume')
            history = run / 'execution/cancel-history'
            history.mkdir(parents=True, exist_ok=True)
            request.replace(history / (uuid4().hex + '.json'))
        attempts.mkdir(parents=True, exist_ok=True)
        number = max((int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdigit()), default=-1) + 1
        attempt = attempts / f'{number:03d}'
        attempt.mkdir()
        write_json(attempt / 'options.json', {'resume': resume, 'check_only': check_only, 'task_ids': list(task_ids)})
        log = run / 'logs/supervisor.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'application.execution.supervisor', '--root', str(root), run_id, attempt.name]
        with log.open('ab') as stream:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write_json(attempt / 'supervisor.json', {**identity(child.pid), 'argv': command, 'started_at': now()})
        write_json(latest, {'attempt': attempt.name})
    if detach:
        return {'run_id': run_id, 'supervisor_pid': child.pid, 'attempt': attempt.name, 'detached': True}
    child.wait()
    return wait(root, run_id)


def cancellation_status(run, attempt_id=None):
    run = Path(run)
    result = {}
    request = run / 'execution/cancel-request.json'
    if request.exists():
        result['request'] = read_json(request)
    latest = run / 'execution/latest.json'
    if attempt_id is None and latest.exists():
        attempt_id = read_json(latest)['attempt']
    if attempt_id is not None:
        base = run / 'execution/attempts' / identifier(attempt_id)
        for name in ('cancel-supervisor-ack', 'cancel-runner-ack', 'exit', 'termination'):
            path = base / (name + '.json')
            if path.exists():
                result[name.replace('-', '_')] = read_json(path)
    return result


def wait(root, run_id):
    run = load_plan(root, run_id)['run']
    attempt = run / 'execution/attempts' / identifier(read_json(run / 'execution/latest.json')['attempt'])
    receipt = attempt / 'exit.json'
    if not receipt.exists():
        process = read_json(attempt / 'supervisor.json')
        try:
            fd = os.pidfd_open(process['pid'])
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                if not receipt.exists():
                    if not matches(process):
                        raise RuntimeError('Supervisor process identity changed')
                    select.select([fd], [], [])
            finally:
                os.close(fd)
    if not receipt.exists():
        raise RuntimeError('Supervisor exited without a receipt; inspect logs and reconcile')
    return {'run_id': run_id, 'attempt': attempt.name, **read_json(receipt),
            'cancellation': cancellation_status(run, attempt.name)}


def _resources(run, attempt, cancelled):
    result = []
    for path in sorted((attempt / 'resources').glob('*.json')):
        if path.name.endswith('.ack.json'):
            continue
        record = read_json(path)
        if (record.get('run_id') != run.name or record.get('execution_attempt_id') != attempt.name
                or record.get('project_root') != str(run.parent.parent) or record.get('managed') is not True):
            raise RuntimeError('Resource ownership differs from execution')
        if record.get('role') not in {'serving','network-proxy'} or record.get('pgid') != record.get('pid') or record.get('sid') != record.get('pid'):
            raise RuntimeError('Resource is not an isolated managed serving process')
        ack = path.with_name(path.stem + '.ack.json')
        if not ack.exists():
            write_json(ack, {'allowed': not cancelled, 'acknowledged_at': now()})
        result.append(record)
    return result


def monitor_child(child, run, attempt, cfg):
    """Monitor a real child independently from its threads and serving HTTP."""
    run, attempt = Path(run), Path(attempt)
    enable_subreaper()
    settings = policy(cfg)
    write_json(attempt / 'cancellation-policy.json', {'version': 1, **settings})
    child_record = {**(getattr(child, 'identity_record', None) or identity(child.pid)), 'role': 'rollout'}
    write_json(attempt / 'process.json', {**child_record, 'started_at': now()})
    started = request = None
    events = []
    # Capture descendants before poll() can reap a rapidly exited group leader.
    known = {str(child_record['pid']): members(child_record)}
    term_sent = kill_sent = False
    with directory_changes(run / 'execution') as changed:
        try:
            fd = os.pidfd_open(child.pid)
        except ProcessLookupError:
            fd = None
        try:
            while True:
                current = request_for(run, attempt.name)
                if current is not None and started is None:
                    request = current
                    ack = attempt / 'cancel-supervisor-ack.json'
                    prior = read_json(ack) if ack.exists() else {}
                    observed_unix = prior.get('observed_unix', time.time())
                    started = time.monotonic() - max(0, time.time() - observed_unix)
                    if not prior:
                        write_json(ack, {**request, 'observed_unix': observed_unix, 'acknowledged_at': now()})
                    update_run_status(run, 'cancelling', request_id=request.get('request_id'))
                code = child.poll()
                resources = _resources(run, attempt, started is not None or code is not None)
                pending_spawns = [p.stem for p in (attempt / 'spawn-intents').glob('*.json')
                                  if not (attempt / 'resources' / p.name).exists()
                                  and not (attempt / 'spawn-errors' / p.name).exists()]
                records = [child_record, *resources]
                alive = []
                for record in records:
                    key = str(record['pid'])
                    known[key] = members(record, known.get(key, ()))
                    for member in known[key]:
                        if member['pid'] != child.pid and member['state'] == 'Z':
                            try:
                                os.waitpid(member['pid'], os.WNOHANG)
                            except ChildProcessError:
                                pass
                    if any(item['state'] != 'Z' for item in known[key]):
                        alive.append(record)
                code = child.poll()
                elapsed = time.monotonic() - started if started is not None else 0
                if code is not None and started is None and (alive or pending_spawns):
                    started = time.monotonic() - settings['cooperative_grace_seconds']
                    elapsed = settings['cooperative_grace_seconds']
                if started is not None and elapsed >= settings['cooperative_grace_seconds'] and not term_sent:
                    for record in alive:
                        events.append({**signal_group(record, 15, known[str(record['pid'])]), 'at': now(), 'role': record['role']})
                    term_sent = True
                kill_at = settings['cooperative_grace_seconds'] + settings['term_grace_seconds']
                if started is not None and elapsed >= kill_at and not kill_sent:
                    for record in alive:
                        events.append({**signal_group(record, 9, known[str(record['pid'])]), 'at': now(), 'role': record['role']})
                    kill_sent = True
                if code is not None and not alive and not pending_spawns:
                    break
                if started is not None and elapsed >= kill_at + settings['kill_verify_seconds']:
                    break
                deadlines = [1.0]
                if started is not None:
                    next_at = (settings['cooperative_grace_seconds'] if not term_sent else
                               kill_at if not kill_sent else kill_at + settings['kill_verify_seconds'])
                    deadlines.append(max(0.001, next_at - elapsed))
                changed(min(deadlines), (fd,) if code is None and fd is not None else ())
        finally:
            if fd is not None:
                os.close(fd)
    code = child.poll()
    if code is not None:
        child.wait()
    remaining = [item for group in known.values() for item in group if matches(item) and identity(item['pid'])['state'] != 'Z']
    cleanup = _cleanup_bounded(cfg, run, attempt)
    termination = {'request_id': (request or {}).get('request_id'), 'signals': events, 'pending_spawns': pending_spawns,
                   'remaining_processes': remaining, 'processes_stopped': code is not None and not remaining,
                   'container_cleanup': cleanup, 'cleanup_complete': code is not None and not remaining and not pending_spawns and cleanup.get('complete', False),
                   'finished_at': now()}
    write_json(attempt / 'termination.json', termination)
    if request is not None:
        update_run_status(run, 'cancelled' if termination['processes_stopped'] else 'cancelling',
                      cleanup_status='complete' if termination['cleanup_complete'] else 'failed',
                      cleanup_complete=termination['cleanup_complete'], request_id=request.get('request_id'))
    return code, termination


def _cleanup_bounded(cfg, run, attempt):
    command = [sys.executable, '-m', 'application.execution.supervisor', '--root', cfg['project_root'],
               '--cleanup', run.name, attempt.name]
    with (attempt / 'cleanup.log').open('ab') as stream:
        process = subprocess.Popen(command, cwd=cfg['project_root'], stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            process.wait(timeout=policy(cfg)['cleanup_seconds'])
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
            return {'complete': False, 'error': 'container cleanup deadline exceeded'}
    result = attempt / 'container-cleanup.json'
    if result.exists():
        return read_json(result)
    return {'complete': False, 'error': 'cleanup worker exited without receipt', 'exit_code': process.returncode}


class ObservedChild:
    """After a controller crash, exit can be observed but its return code may be lost."""
    def __init__(self, record):
        self.identity_record = record
        self.pid = record['pid']

    def poll(self):
        if matches(self.identity_record) and identity(self.pid)['state'] != 'Z':
            return None
        return 'returncode_unavailable_non_child'

    def wait(self):
        return self.poll()


def reconcile_cancelled(root, run_id):
    """Resume requested cancellation using durable identities and the original deadline."""
    loaded = load_plan(root, run_id)
    run, cfg = loaded['run'], loaded['config']
    latest = run / 'execution/latest.json'
    if not latest.exists():
        return {'reconciled': False, 'reason': 'no_execution'}
    attempt = run / 'execution/attempts' / identifier(read_json(latest)['attempt'])
    if request_for(run, attempt.name) is None:
        return {'reconciled': False, 'reason': 'no_cancel_request'}
    previous = read_json(attempt / 'supervisor.json')
    if matches(previous) and identity(previous['pid'])['state'] != 'Z':
        return {'reconciled': False, 'reason': 'supervisor_still_alive'}
    with file_lock(run / 'execution/supervisor.lock', nonblocking=True), file_lock(run / 'execution/active-supervisor.lock', nonblocking=True):
        previous = read_json(attempt / 'supervisor.json')
        if matches(previous) and identity(previous['pid'])['state'] != 'Z':
            return {'reconciled': False, 'reason': 'supervisor_still_alive'}
        process = read_json(attempt / 'process.json')
        if process.get('pgid') != process['pid'] or process.get('sid') != process['pid']:
            raise RuntimeError('Legacy non-isolated execution requires explicit identity-aware recovery')
        history = attempt / 'supervisor-history' / (uuid4().hex + '.json')
        write_json(history, previous)
        write_json(attempt / 'supervisor.json', {**identity(os.getpid()), 'reconciliation': True, 'started_at': now()})
        observed, termination = monitor_child(ObservedChild(process), run, attempt, cfg)
        exit_path = attempt / 'exit.json'
        if not exit_path.exists():
            write_json(exit_path, {'exit_code': None, 'exit_observed': observed is not None,
                                  'returncode_source': 'unavailable_non_child', 'error': None, 'finished_at': now()})
        return {'reconciled': True, 'termination': termination}


def supervise(root, run_id, attempt_id):
    loaded = load_plan(root, run_id)
    run, cfg = loaded['run'], loaded['config']
    attempt = run / 'execution/attempts' / identifier(attempt_id)
    from .run_plan import _git_identity
    code, error = None, None
    child = None
    try:
        for name in ['resolved-config.json', 'plan.json', 'selection.json', 'tasks.json',
                     'config.yaml', 'rollout/work-items.json', 'parallel-config.yaml']:
            source = run / name
            target = attempt / 'configuration' / name
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        write_json(attempt / 'supervisor-limits.json', _apply_open_file_limit(cfg))
        write_json(attempt / 'code-identity.json', {'git': _git_identity(Path(root)), 'started_at': now(),
                                                 'frozen_config_hash': cfg['config_hash']})
        options = read_json(attempt / 'options.json')
        command = [sys.executable, '-m', 'core.cli', '--config', str(Path(root) / 'config.yaml'), 'rollout', run_id]
        # Every supervisor owns an already-frozen run, including `run start`.
        command.append('--resume')
        if options['check_only']:
            command.append('--check-only')
        for task_id in options['task_ids']:
            command.extend(('--task-id', task_id))
        token = uuid4().hex
        write_json(attempt / 'control.json', {'token': token})
        (attempt / 'control.json').chmod(0o600)
        env = {**os.environ, 'DYCON_SUPERVISED': '1', 'DYCON_EXECUTION_ATTEMPT': attempt.name,
               'DYCON_EXECUTION_RUN': str(run), 'DYCON_CANCEL_TOKEN': token}
        with file_lock(run / 'execution/active-supervisor.lock', nonblocking=True):
            child = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
            code, _ = monitor_child(child, run, attempt, cfg)
    except BaseException as exc:
        error = repr(exc)
        if child is not None and child.poll() is None:
            try:
                signal_group(identity(child.pid), 9)
                code = child.wait(timeout=1)
            except Exception as stop_error:
                error += '; child stop: ' + repr(stop_error)
        cleanup = _cleanup_bounded(cfg, run, attempt)
        write_json(attempt / 'termination.json', {'cleanup_complete': False,
                   'container_cleanup': cleanup, 'error': error, 'finished_at': now()})
    finally:
        write_json(attempt / 'exit.json', {'exit_code': code, 'error': error, 'finished_at': now()})
    return code if code is not None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('run_id')
    parser.add_argument('attempt_id')
    args = parser.parse_args()
    if args.cleanup:
        loaded = load_plan(args.root, args.run_id)
        from runtime.containers.factory import cleanup_execution_containers
        result = cleanup_execution_containers(loaded['config'], args.run_id, args.attempt_id)
        write_json(loaded['run'] / 'execution/attempts' / identifier(args.attempt_id) / 'container-cleanup.json', result)
        raise SystemExit(0 if result['complete'] else 1)
    raise SystemExit(supervise(args.root, args.run_id, args.attempt_id))
