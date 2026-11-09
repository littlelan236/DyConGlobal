import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from application.execution.config_revision import revise_config
from application.execution.queued_resume import execute, wait_processes
from application.execution.run_plan import load_plan
from core.io import read_json, sha256, write_json
from core.processes import identity
from tests.test_capture_budget_revision import make_run


def test_placement_revision_preserves_computation_and_completed_evidence(cfg):
    run = make_run(cfg)
    before = read_json(run / 'resolved-config.json')
    gpus = [2] if before['serving']['tensor_parallel_size'] == 1 else list(range(2, 2 + before['serving']['tensor_parallel_size']))
    stable = {name: sha256(run / name) for name in ['selection.json', 'tasks.json', 'rollout/work-items.json', 'evidence.json'] if (run / name).exists()}
    updates = {'serving.gpus': gpus, 'serving.base_url': 'http://127.0.0.1:8107/v1'}
    result = revise_config(cfg['project_root'], run.name, revision_id='move', updates=updates,
                           reason='User authorized GPU placement')
    after = load_plan(cfg['project_root'], run.name)['config']
    normalized = copy.deepcopy(after)
    for key in ['gpus', 'base_url']:
        normalized['serving'][key] = before['serving'][key]
    normalized['config_hash'] = before['config_hash']
    assert normalized == before
    assert all(sha256(run / name) == digest for name, digest in stable.items())
    assert result['changed_config_fields'] == ['serving.base_url', 'serving.gpus']
    assert read_json(run / 'events/config-revision-move.json')['event'] == 'run.config_revised'
    assert revise_config(cfg['project_root'], run.name, revision_id='move', updates=updates,
                         reason='User authorized GPU placement') == result


def test_wait_processes_observes_all_real_child_exits():
    children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.2)']) for _ in range(2)]
    records = [identity(p.pid) for p in children]
    wait_processes(records)
    assert all(p.poll() == 0 for p in children)


def test_gpu_exit_triggers_resume_without_waiting_for_source_supervisor(cfg, monkeypatch):
    from application.execution import queued_resume as module
    root = Path(cfg['project_root'])
    target = make_run(cfg)
    previous = target / 'execution/attempts/007'
    previous.mkdir(parents=True, exist_ok=True)
    write_json(target / 'execution/latest.json', {'attempt': '007'})
    write_json(previous / 'exit.json', {'exit_code': -15})
    write_json(previous / 'termination.json', {'cleanup_complete': True})
    source = root / 'runs/source-running'
    write_json(source / 'execution/latest.json', {'attempt': '005'})
    write_json(source / 'status.json', {'stage': 'rollout'})
    folder = root / 'state/run-resume-queue/test-request'
    request = {'run_id': target.name, 'after_run': source.name, 'after_execution': '005',
               'target_config_sha256': sha256(target / 'resolved-config.json'),
               'target_plan_sha256': sha256(target / 'plan.json'), 'target_previous_execution': '007',
               'gpu_processes': [], 'serving_resources': []}
    write_json(folder / 'request.json', request)
    write_json(folder / 'status.json', {'stage': 'queued'})
    original = module.load_plan
    monkeypatch.setattr(module, 'load_plan', lambda r, name: {'run': source} if name == source.name else original(r, name))
    monkeypatch.setattr(module, 'gpu_processes', lambda _: [])
    commands = []
    def resume(command, **kwargs):
        commands.append(command)
        write_json(target / 'execution/latest.json', {'attempt': '008'})
        return SimpleNamespace(returncode=0, stdout='{}', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', resume)
    result = execute(root, 'test-request')
    assert result['stage'] == 'resumed' and result['execution'] == '008'
    assert commands[0][-4:] == ['run', 'resume', target.name, '--detach']
    assert read_json(source / 'status.json')['stage'] == 'rollout'
    assert execute(root, 'test-request') == result
    assert len(commands) == 1
