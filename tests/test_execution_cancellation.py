"""Cancellation control uses real event notifications and isolated OS children."""
import signal
import subprocess
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from core.cancellation import CancellationContext, Cancelled, CompletionWake, request_for
from core.io import read_json, write_json
from core.processes import identity, matches, signal_group


def test_file_event_wakes_scheduler_with_no_completed_work(tmp_path):
    run = tmp_path / 'runs/r'
    (run / 'execution').mkdir(parents=True)
    context = CancellationContext(run, '003').start()
    wake = CompletionWake(context)
    future = Future()
    result = []
    thread = threading.Thread(target=lambda: result.append(wake.wait([future], context)))
    thread.start()
    try:
        write_json(run / 'execution/cancel-request.json', {'execution_attempt_id': '003', 'request_id': 'request'})
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert result == [set()]
        assert read_json(run / 'execution/attempts/003/cancel-runner-ack.json')['request_id'] == 'request'
        assert not future.done()
        with pytest.raises(Cancelled):
            context.check()
    finally:
        context.cancel()
        thread.join(timeout=2)
        context.close()


def test_request_is_scoped_to_execution_generation(tmp_path):
    write_json(tmp_path / 'execution/cancel-request.json', {'execution_attempt_id': '002'})
    assert request_for(tmp_path, '003') is None
    assert request_for(tmp_path, '002') is not None


@pytest.mark.parametrize('ignore_term,expected', [(False, -15), (True, -9)])
def test_supervisor_escalates_only_its_isolated_child(tmp_path, monkeypatch, ignore_term, expected):
    from application.execution import supervisor
    run = tmp_path / 'runs/r'
    attempt = run / 'execution/attempts/000'
    attempt.mkdir(parents=True)
    monkeypatch.setattr(supervisor, '_cleanup_bounded', lambda *args: {'complete': True})
    settings = {'project_root': str(tmp_path), 'execution': {'cancellation': {
        'cooperative_grace_seconds': .05, 'term_grace_seconds': .05,
        'kill_verify_seconds': 1.0, 'cleanup_seconds': 1.0}}}
    script = ('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore_term else 'import time; ')
    script += "print('ready',flush=True); time.sleep(30)"
    child = subprocess.Popen([sys.executable, '-c', script], start_new_session=True, stdout=subprocess.PIPE, text=True)
    other = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
    other_identity = identity(other.pid)
    assert child.stdout.readline().strip() == 'ready'
    write_json(run / 'execution/cancel-request.json', {'execution_attempt_id': '000', 'request_id': 'test'})
    try:
        code, result = supervisor.monitor_child(child, run, attempt, settings)
        assert code == expected
        assert result['cleanup_complete'] and not result['remaining_processes']
        assert [item['signal'] for item in result['signals'] if item['sent']] == ([15, 9] if ignore_term else [15])
        assert other.poll() is None and matches(other_identity)
        assert read_json(run / 'status.json')['stage'] == 'cancelled'
    finally:
        child.stdout.close()
        for process in (child, other):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_pid_identity_mismatch_never_signals_reused_target():
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
    try:
        record = identity(child.pid)
        record['start_ticks'] = str(int(record['start_ticks']) + 1)
        assert signal_group(record, signal.SIGTERM)['sent'] is False
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_container_cleanup_exact_execution_and_no_image_prune():
    from runtime.containers.factory import cleanup_execution_containers
    removed, calls = [], []
    container = SimpleNamespace(id='owned', labels={'dycon.project': '/p', 'dycon.run': 'r', 'dycon.execution': '004'},
                                remove=lambda force: removed.append('owned'), reload=lambda: None)
    def listed(**kwargs):
        calls.append(kwargs)
        return [] if removed else [container]
    result = cleanup_execution_containers({'project_root': '/p'}, 'r', '004',
                                         client=SimpleNamespace(containers=SimpleNamespace(list=listed),
                                             networks=SimpleNamespace(list=lambda **kw: [])))
    assert result['complete'] and removed == ['owned']
    assert calls[0]['filters']['label'] == ['dycon.project=/p', 'dycon.run=r', 'dycon.execution=004']


def test_container_cleanup_removes_only_owned_empty_networks():
    from runtime.containers.networks import cleanup_owned_resources
    removed = []
    class Resource:
        def __init__(self, kind, labels):
            self.id = kind
            self.labels = labels
            self.attrs = {'Labels': labels, 'Containers': {}}
        def reload(self):
            return None
        def remove(self, **kwargs):
            removed.append(self.id)
    labels = {'dycon.project': '/p', 'dycon.run': 'r', 'dycon.execution': '004'}
    owned = Resource('owned-network', labels)
    foreign = Resource('foreign-network', {'dycon.project': '/p', 'dycon.run': 'other', 'dycon.execution': '004'})
    class Client:
        class containers:
            @staticmethod
            def list(**kwargs):
                return []
        class networks:
            @staticmethod
            def list(**kwargs):
                return ([] if owned.id in removed else [owned]) if kwargs.get('filters') else [owned, foreign]
    result = cleanup_owned_resources({'project_root': '/p', 'environment': {'docker_host': 'fixture'}},
        {'project': '/p', 'run': 'r', 'execution': '004'}, client=Client())
    assert result['complete'] and result['removed_networks'] == ['owned-network']
    assert removed == ['owned-network']


def test_cancelled_never_becomes_a_seed_or_infrastructure_retry():
    from rollout.retry import disposition, sample_accepted, sample_settled
    assert disposition({}, 'Cancelled') == 'interrupted'
    assert not sample_accepted({}, {'exit_status': 'Cancelled'})
    assert not sample_settled({}, {'exit_status': 'Cancelled', 'disposition': 'interrupted'})


def frozen_run(cfg, name='cancel-fixture'):
    from application.execution.run_plan import create_run
    from catalog.store import Catalog
    from tests.conftest import task
    Catalog(cfg['project_root']).register_tasks([task('cancel-task')])
    return create_run(cfg, name, operation='rollout', task_ids=['cancel-task'])


def test_cancel_request_idempotent_and_does_not_change_snapshot(cfg):
    from application.execution.status import cancel_run
    from core.io import sha256
    run = frozen_run(cfg)
    original = sha256(run / 'resolved-config.json')
    first = cancel_run(cfg['project_root'], run.name, 'first')
    second = cancel_run(cfg['project_root'], run.name, 'second')
    assert first['request_id'] == second['request_id']
    assert read_json(run / 'execution/cancel-request.json')['reason'] == 'first'
    assert sha256(run / 'resolved-config.json') == original


def test_explicit_resume_archives_old_request_and_creates_new_generation(cfg, monkeypatch):
    import os

    from application.execution import supervisor
    from core.io import sha256
    run = frozen_run(cfg)
    original = sha256(run / 'resolved-config.json')
    attempt = run / 'execution/attempts/000'
    write_json(run / 'execution/latest.json', {'attempt': '000'})
    write_json(attempt / 'exit.json', {'exit_code': -15})
    write_json(attempt / 'termination.json', {'cleanup_complete': True})
    write_json(run / 'execution/cancel-request.json', {'request_id': 'old', 'execution_attempt_id': '000'})
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(pid=os.getpid()))
    with pytest.raises(RuntimeError, match='explicit run resume'):
        supervisor.launch(cfg['project_root'], run.name, detach=True)
    result = supervisor.launch(cfg['project_root'], run.name, resume=True, detach=True)
    assert result['attempt'] == '001'
    assert not (run / 'execution/cancel-request.json').exists()
    assert len(list((run / 'execution/cancel-history').glob('*.json'))) == 1
    assert sha256(run / 'resolved-config.json') == original


def test_incomplete_cleanup_blocks_new_execution(cfg):
    from application.execution.supervisor import launch
    run = frozen_run(cfg)
    write_json(run / 'execution/latest.json', {'attempt': '000'})
    write_json(run / 'execution/attempts/000/exit.json', {'exit_code': -9})
    write_json(run / 'execution/attempts/000/termination.json', {'cleanup_complete': False})
    with pytest.raises(RuntimeError, match='cleanup is incomplete'):
        launch(cfg['project_root'], run.name, resume=True, detach=True)


def test_capture_wait_cancelled_without_fabricating_native_output(tmp_path):
    from hidden.backends.vllm.online import capture_request_metadata
    from runtime.serving.capture import OnlineCaptureController
    controller = OnlineCaptureController(tmp_path, {'export': {'enabled': False}, 'run_id': 'r'})
    md = capture_request_metadata('session', 'task', 'r', 'pending', 0, ['think_close'], close_token_ids=[11])
    controller.start(md)
    try:
        with __import__('concurrent.futures', fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(controller.wait_output, 'session', 0, timeout=10)
            controller.cancel_run('r')
            with pytest.raises(Cancelled):
                future.result(timeout=1)
        assert not controller.request_outputs
        assert controller.sessions['session'].status == 'active'
    finally:
        controller.close()


def test_released_branch_response_never_calls_model():
    from rollout.parallel import ParallelReasoningCoordinator
    called = []
    class Client:
        def call(self, action, payload):
            return {'acquired': False, 'already_released': True} if action == 'branch-acquire' else {}
    cfg = {'parallel': {'branch_count': 2, 'dynamic_branch_admission': True}, 'generation': {'seed': 1}}
    coordinator = ParallelReasoningCoordinator(cfg, object(), Client())
    with pytest.raises(Cancelled):
        coordinator.run([], {'session_id': 's'}, lambda *args: called.append(1), lambda *args: None, lambda *args: None)
    assert not called


def test_reconcile_after_controller_crash_preserves_unknown_returncode(cfg, monkeypatch):
    import os

    from application.execution import supervisor
    run = frozen_run(cfg)
    attempt = run / 'execution/attempts/000'
    write_json(run / 'execution/latest.json', {'attempt': '000'})
    stale = identity(os.getpid())
    stale['start_ticks'] = 'not-the-current-controller'
    write_json(attempt / 'supervisor.json', stale)
    child = subprocess.Popen([sys.executable, '-c', "import time;print('ready',flush=True);time.sleep(30)"],
                             start_new_session=True, stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == 'ready'
    write_json(attempt / 'process.json', identity(child.pid))
    write_json(run / 'execution/cancel-request.json', {'execution_attempt_id': '000', 'request_id': 'adopt'})
    monkeypatch.setattr(supervisor, '_cleanup_bounded', lambda *a: {'complete': True})
    monkeypatch.setattr(supervisor, 'policy', lambda cfg: {'cooperative_grace_seconds': .01,
        'term_grace_seconds': .05, 'kill_verify_seconds': 1., 'cleanup_seconds': 1.})
    try:
        result = supervisor.reconcile_cancelled(cfg['project_root'], run.name)
        assert result['reconciled'] and result['termination']['cleanup_complete']
        receipt = read_json(attempt / 'exit.json')
        assert receipt['exit_code'] is None and receipt['exit_observed'] is True
        assert receipt['returncode_source'] == 'unavailable_non_child'
        assert child.wait(timeout=1) == -15
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.kill()
        child.wait()


def test_normal_child_exit_still_reaps_orphaned_same_session_worker(tmp_path, monkeypatch):
    from application.execution import supervisor
    run = tmp_path / 'runs/r'
    attempt = run / 'execution/attempts/000'
    attempt.mkdir(parents=True)
    monkeypatch.setattr(supervisor, '_cleanup_bounded', lambda *a: {'complete': True})
    settings = {'project_root': str(tmp_path), 'execution': {'cancellation': {
        'cooperative_grace_seconds': .05, 'term_grace_seconds': .05,
        'kill_verify_seconds': 1.0, 'cleanup_seconds': 1.0}}}
    script = "import os,time; child=os.fork(); print('ready',flush=True) if child else None; time.sleep(.15 if child else 30)"
    child = subprocess.Popen([sys.executable, '-c', script], start_new_session=True, stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == 'ready'
    try:
        code, result = supervisor.monitor_child(child, run, attempt, settings)
        assert code == 0 and result['cleanup_complete']
        assert not result['remaining_processes']
        assert any(item['sent'] for item in result['signals'])
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.kill()
        child.wait()


def test_docker_cleanup_worker_cannot_block_controller_forever(tmp_path, monkeypatch):
    from application.execution import supervisor
    run = tmp_path / 'runs/r'
    attempt = run / 'execution/attempts/000'
    attempt.mkdir(parents=True)
    events = []
    class SlowCleanup:
        returncode = -9
        def wait(self, timeout):
            events.append(('wait', timeout))
            if len(events) == 1:
                raise subprocess.TimeoutExpired('test-cleanup', timeout)
            return -9
        def kill(self):
            events.append(('kill',))
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **kw: SlowCleanup())
    result = supervisor._cleanup_bounded({'project_root': str(tmp_path)}, run, attempt)
    assert not result['complete'] and 'deadline' in result['error']
    assert ('kill',) in events


def test_direct_rollout_cli_uses_same_supervisor(cfg, monkeypatch, capsys):
    from application.execution import supervisor
    from core.cli import main
    run = frozen_run(cfg)
    calls = []
    monkeypatch.delenv('DYCON_SUPERVISED', raising=False)
    monkeypatch.setattr(supervisor, 'launch', lambda *a, **kw: calls.append((a, kw)) or {'exit_code': 0})
    main(['--config', cfg['config_source'], 'rollout', run.name, '--check-only'])
    assert len(calls) == 1 and calls[0][1]['check_only'] is True
    assert 'exit_code' in capsys.readouterr().out


def test_cancel_in_startup_is_not_projected_as_generic_failure(cfg, monkeypatch):
    import runtime.serving.startup
    from core.cancellation import CURRENT
    from rollout.runner import run_manifest
    from tests.test_limits_retry import configure, fake_runtime, prepare
    configured = configure(cfg)
    run, _ = prepare(configured, 'cancel-startup')
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    def stop(*args):
        CURRENT.get().cancel({'reason': 'test cancellation'})
        raise Cancelled('test cancellation')
    monkeypatch.setattr(runtime.serving.startup, 'check_tool_chain', stop)
    assert run_manifest(configured, run.name, resume=True, factory=factory, check_only=True) == {}
    assert read_json(run / 'status.json')['stage'] == 'cancelling'
