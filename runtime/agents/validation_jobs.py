"""Bounded detached execution of the existing SkillsBench CLI, with durable logs."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from core.io import identifier, now, read_json, write_json

ALLOWED = {'dependencies-build', 'dependencies-verify', 'environment-check',
           'preflight', 'oracle', 'offline-check', 'empty-control', 'audit-rewards','verifier-control',
           'regrade-submissions'}


def start(root, job_id, args, seconds=21600):
    root = Path(root).resolve()
    if args[:1] == ['--']:
        args = args[1:]
    if not args or args[0] not in ALLOWED or not 1 <= seconds <= 86400:
        raise ValueError('Expected a supported SkillsBench validation command and bounded timeout')
    target = root / 'temp/skillsbench-jobs' / identifier(job_id)
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'request.json', {'args': args, 'seconds': seconds, 'created_at': now()})
    with (target / 'worker.log').open('wb') as log:
        proc = subprocess.Popen([sys.executable, '-m', __name__, str(root), job_id],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    return {'job_id': job_id, 'worker_pid': proc.pid, 'path': str(target)}


def status(root, job_id):
    target = Path(root) / 'temp/skillsbench-jobs' / identifier(job_id)
    result = read_json(target / 'status.json') if (target / 'status.json').exists() else {'state': 'starting'}
    if result.get('state') == 'running':
        try:
            os.kill(result['worker_pid'], 0)
        except ProcessLookupError:
            result = {**result, 'state': 'interrupted', 'success': False}
    return {**result, 'job_id': job_id, 'log': str(target / 'command.log')}


def worker(root, job_id):
    target = Path(root) / 'temp/skillsbench-jobs' / identifier(job_id)
    req = read_json(target / 'request.json')
    row = {'state': 'running', 'worker_pid': os.getpid(), 'started_at': now(), **req}
    write_json(target / 'status.json', row)
    proc = None
    try:
        with (target / 'command.log').open('wb') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'core.cli', 'skillsbench', *req['args']],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            row['command_pid'] = proc.pid
            write_json(target / 'status.json', row)
            try:
                code = proc.wait(timeout=req['seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
                row['timed_out'] = True
                code = 124
        row.update(state='completed', return_code=code, success=code == 0, finished_at=now())
    except BaseException as exc:
        row.update(state='failed', success=False, error=repr(exc), finished_at=now())
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        raise
    finally:
        write_json(target / 'status.json', row)


if __name__ == '__main__':
    worker(*sys.argv[1:])
