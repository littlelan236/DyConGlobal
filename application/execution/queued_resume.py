import json
"""Durable, one-shot resume after an execution's GPU processes exit."""
import argparse
import os
import select
import subprocess
import sys
from pathlib import Path

from application.execution.run_plan import load_plan
from core.io import identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from core.processes import identity, matches


def wait_processes(records):
    """Observe pinned process lifetimes through pidfds, without periodic polling."""
    pending = {}
    try:
        for record in records:
            try:
                fd = os.pidfd_open(record['pid'])
            except ProcessLookupError:
                continue
            if matches(record) and identity(record['pid'])['state'] != 'Z':
                pending[fd] = record
            else:
                os.close(fd)
        while pending:
            ready, _, _ = select.select(list(pending), [], [])
            for fd in ready:
                os.close(fd)
                pending.pop(fd)
    finally:
        for fd in pending:
            os.close(fd)


def gpu_processes(serving_records):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20)
    groups = {r['pgid'] for r in serving_records}
    sessions = {r['sid'] for r in serving_records}
    rows = []
    for line in sorted(set(result.stdout.splitlines())):
        if not line.strip().isdigit():
            continue
        try:
            record = identity(int(line.strip()))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if record['pgid'] in groups and record['sid'] in sessions:
            rows.append(record)
    return rows


def assert_offline(loaded):
    run = loaded['run']
    latest = run / 'execution/latest.json'
    if not latest.exists():
        return
    attempt = run / 'execution/attempts' / read_json(latest)['attempt']
    for name in ['process.json', 'supervisor.json']:
        if (attempt / name).exists() and matches(read_json(attempt / name)):
            raise RuntimeError('Target execution is still active')
    if not (attempt / 'exit.json').exists() or not read_json(attempt / 'termination.json').get('cleanup_complete'):
        raise RuntimeError('Target exit/cleanup has not been confirmed')


def enqueue(root, run_id, *, after_run, request_id, reason):
    root = Path(root).resolve()
    run_id, after_run, request_id = map(identifier, (run_id, after_run, request_id))
    if run_id == after_run or not reason.strip():
        raise ValueError('Distinct runs and an authorization reason are required')
    folder = root / 'state/run-resume-queue' / request_id
    with file_lock(folder.with_suffix('.lock'), nonblocking=True):
        if (folder / 'request.json').exists():
            prior = read_json(folder / 'request.json')
            if (prior['run_id'], prior['after_run'], prior['reason']) != (run_id, after_run, reason):
                raise ValueError('Queue ID is already bound to different parameters')
            return read_json(folder / 'status.json')
        target = load_plan(root, run_id)
        assert_offline(target)
        predecessor = load_plan(root, after_run)
        if set(target['config']['serving']['gpus']) != set(predecessor['config']['serving']['gpus']):
            raise ValueError('Queued target must use the predecessor GPU placement')
        if target['config']['serving']['base_url'] == predecessor['config']['serving']['base_url']:
            raise ValueError('GPU-exit queue needs a separate serving endpoint')
        source = predecessor['run']
        execution = read_json(source / 'execution/latest.json')['attempt']
        resources = []
        for path in (source / 'execution/attempts' / execution / 'resources').glob('*.json'):
            if path.name.endswith('.ack.json'):
                continue
            record = read_json(path)
            if record.get('role') == 'serving' and record.get('managed') is True:
                if record.get('run_id') != after_run or record.get('execution_attempt_id') != execution:
                    raise ValueError('Predecessor serving ownership differs')
                resources.append(record)
        if not resources:
            raise ValueError('Predecessor has no registered managed serving resource')
        observed = gpu_processes(resources)
        if not observed and any(matches(r) for r in resources):
            raise RuntimeError('Live predecessor serving has no observable GPU processes; inspect before queueing')
        folder.mkdir(parents=True, exist_ok=False)
        request = {'schema_version': 1, 'request_id': request_id, 'run_id': run_id, 'after_run': after_run,
                   'after_execution': execution, 'reason': reason, 'queued_at': now(),
                   'target_config_sha256': sha256(target['run'] / 'resolved-config.json'),
                   'target_plan_sha256': sha256(target['run'] / 'plan.json'),
                   'target_previous_execution': read_json(target['run'] / 'execution/latest.json')['attempt'] if (target['run'] / 'execution/latest.json').exists() else None,
                   'serving_resources': resources, 'gpu_processes': observed,
                   'trigger': 'all_pinned_predecessor_gpu_processes_exited',
                   'unit': 'dycon-resume-' + request_id + '.service'}
        write_json(folder / 'request.json', request)
        write_json(folder / 'status.json', {'stage': 'queued', 'request_id': request_id, 'queued_at': now()})
        command = ['systemd-run', '--user', '--unit=' + request['unit'], '--working-directory=' + str(root),
                   '--property=Restart=no', '--property=RemainAfterExit=yes', '--', sys.executable,
                   '-m', 'application.execution.queued_resume', '--root', str(root), request_id]
        launched = subprocess.run(command, capture_output=True, text=True, timeout=30)
        write_json(folder / 'service-launch.json', {'command': command, 'returncode': launched.returncode,
                                                  'stdout': launched.stdout, 'stderr': launched.stderr, 'at': now()})
        if launched.returncode:
            write_json(folder / 'status.json', {'stage': 'failed', 'error': launched.stderr, 'at': now()})
            raise RuntimeError('Could not start queued resume service: ' + launched.stderr)
        return {'stage': 'queued', **request, 'path': str(folder)}


def execute(root, request_id):
    root = Path(root).resolve()
    folder = root / 'state/run-resume-queue' / identifier(request_id)
    with file_lock(folder / 'worker.lock', nonblocking=True):
        request = read_json(folder / 'request.json')
        status = read_json(folder / 'status.json')
        if status.get('stage') in {'resumed', 'failed'}:
            return status
        def publish(stage, **values):
            record = {'stage': stage, 'request_id': request_id, 'at': now(), **values}
            write_json(folder / 'status.json', record)
            return record
        publish('waiting_gpu_exit', worker=identity(os.getpid()), observed_gpu_processes=request['gpu_processes'])
        try:
            wait_processes(request['gpu_processes'])
            predecessor = load_plan(root, request['after_run'])
            execution = read_json(predecessor['run'] / 'execution/latest.json')['attempt']
            if execution != request['after_execution']:
                raise RuntimeError('Predecessor execution changed while queued')
            if gpu_processes(request['serving_resources']):
                raise RuntimeError('Predecessor has replacement GPU processes; refusing early launch')
            target = load_plan(root, request['run_id'])
            assert_offline(target)
            if (sha256(target['run'] / 'resolved-config.json') != request['target_config_sha256']
                    or sha256(target['run'] / 'plan.json') != request['target_plan_sha256']
                    or ((read_json(target['run'] / 'execution/latest.json')['attempt'] if (target['run'] / 'execution/latest.json').exists() else None) != request['target_previous_execution'])):
                raise RuntimeError('Queued target changed after registration')
            publish('launching', gpu_exit_observed_at=now())
            action = 'resume' if request.get('target_previous_execution') is not None else 'start'
            command = [sys.executable, '-m', 'core.cli', 'run', action, request['run_id'], '--detach']
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90)
            write_json(folder / 'resume-result.json', {'command': command, 'returncode': result.returncode,
                                                     'stdout': result.stdout, 'stderr': result.stderr, 'at': now()})
            if result.returncode:
                raise RuntimeError('Resume CLI failed: ' + result.stderr[-3000:])
            return publish('resumed', resume_dispatched=True, startup_health_verified=False,
                           execution=read_json(target['run'] / 'execution/latest.json')['attempt'])
        except Exception as error:
            publish('failed', error=repr(error))
            raise


def amend(root, request_id, *, reason):
    root = Path(root).resolve(); folder = root / "state/run-resume-queue" / identifier(request_id)
    with file_lock(folder / "amend.lock", nonblocking=True):
        request = read_json(folder / "request.json"); status = read_json(folder / "status.json")
        if status.get("stage") not in {"queued", "waiting_gpu_exit"}: raise RuntimeError("Queue is no longer waiting; refuse amendment")
        target = load_plan(root, request["run_id"]); assert_offline(target)
        before = {"target_config_sha256": request["target_config_sha256"], "target_plan_sha256": request["target_plan_sha256"], "reason": request["reason"]}
        request["target_config_sha256"] = sha256(target["run"] / "resolved-config.json"); request["target_plan_sha256"] = sha256(target["run"] / "plan.json"); request["reason"] = reason
        request.setdefault("amendments", []).append({"before": before, "at": now(), "reason": reason})
        temp = folder / "request.json.amend.part"; write_json(temp, request); temp.replace(folder / "request.json")
        write_json(folder / "amendment.json", {"before": before, "after": {"target_config_sha256": request["target_config_sha256"], "target_plan_sha256": request["target_plan_sha256"], "reason": reason}, "at": now()})
        return {"stage": "amended", "request_id": request_id, "target_config_sha256": request["target_config_sha256"], "target_plan_sha256": request["target_plan_sha256"]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('request_id')
    parser.add_argument('--amend-reason')
    args = parser.parse_args()
    if args.amend_reason:
        print(json.dumps(amend(args.root, args.request_id, reason=args.amend_reason), ensure_ascii=False))
    else:
        execute(args.root, args.request_id)
