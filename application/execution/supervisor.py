"""Durable supervision and OS process-exit event waits."""
from __future__ import annotations
import argparse
import os
import select
import socket
import subprocess
import sys
from pathlib import Path
from core.io import identifier, now, read_json, write_json
from core.locking import file_lock
from .run_plan import load_plan


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19], 'host': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def launch(root, run_id, *, resume=False, detach=False):
    root = Path(root).resolve()
    run = load_plan(root, run_id)['run']
    attempts = run / 'execution/attempts'
    with file_lock(run / 'execution/supervisor.lock', nonblocking=True):
        latest = run / 'execution/latest.json'
        if latest.exists():
            previous = attempts / read_json(latest)['attempt']
            if not (previous / 'exit.json').exists():
                process = read_json(previous / 'supervisor.json')
                try:
                    actual = identity(process['pid'])
                    alive = actual == {k: process[k] for k in actual}
                except FileNotFoundError:
                    alive = False
                if alive:
                    raise RuntimeError('Run already has a live supervisor')
        attempts.mkdir(parents=True, exist_ok=True)
        number = max((int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdigit()), default=-1) + 1
        attempt = attempts / f'{number:03d}'
        attempt.mkdir()
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


def wait(root, run_id):
    run = load_plan(root, run_id)['run']
    attempt = run / 'execution/attempts' / identifier(read_json(run / 'execution/latest.json')['attempt'])
    receipt = attempt / 'exit.json'
    if not receipt.exists():
        process = read_json(attempt / 'supervisor.json')
        pid = process['pid']
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                if not receipt.exists():
                    actual = identity(pid)
                    if actual != {k: process[k] for k in actual}:
                        raise RuntimeError('Supervisor process identity changed')
                    select.select([fd], [], [])  # No progress polling: wait for kernel exit event.
            finally:
                os.close(fd)
    if not receipt.exists():
        raise RuntimeError('Supervisor exited without a receipt; inspect logs and reconcile')
    return {'run_id': run_id, 'attempt': attempt.name, **read_json(receipt)}


def supervise(root, run_id, attempt_id):
    run = load_plan(root, run_id)['run']
    attempt = run / 'execution/attempts' / identifier(attempt_id)
    command = [sys.executable, '-m', 'core.cli', '--config', str(Path(root)/'config.yaml'),
               'rollout', run_id, '--resume']
    code, error = 1, None
    try:
        child = subprocess.Popen(command, cwd=root)
        write_json(attempt / 'process.json', {**identity(child.pid), 'argv': command, 'started_at': now()})
        code = child.wait()
    except BaseException as exc:
        error = repr(exc)
    finally:
        write_json(attempt / 'exit.json', {'exit_code': code, 'error': error, 'finished_at': now()})
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('run_id')
    parser.add_argument('attempt_id')
    args = parser.parse_args()
    raise SystemExit(supervise(args.root, args.run_id, args.attempt_id))
