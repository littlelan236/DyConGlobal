from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from core.io import read_json, sha256, write_json
from core.locking import file_lock
from hidden import recover_online as recovery
from runtime.serving.spool import CaptureSpool


def config(tmp_path):
    return {'project_root': str(tmp_path), 'capture': {'enabled': True, 'export': {
        'spool_root': 'spool', 'root': 'online', 'spool_max_bytes': 10000000,
        'upload_workers': 1, 'shard_max_bytes': 1000000, 'upload_max_elapsed_seconds': 5}},
        'paths': {'nas_root': str(tmp_path / 'nas')}}


def cancellation():
    return SimpleNamespace(attempt_id='009', event=threading.Event())


@pytest.mark.parametrize('stop_by_cancel', [False, True])
def test_background_does_not_block_and_is_joined_under_run_lock(tmp_path, monkeypatch, stop_by_cancel):
    cfg = config(tmp_path)
    root = tmp_path / 'spool/run'
    write_json(root / 'uploads/old.json', {'session_id': 'old'})
    entered = threading.Event()
    finished = threading.Event()
    lock = tmp_path / 'state/run-locks/run.lock'

    def blocked(config, run_id, *, snapshot, checkpoint):
        assert snapshot['uploads'] == (root / 'uploads/old.json',)
        with pytest.raises(BlockingIOError):
            with file_lock(lock, nonblocking=True):
                pass
        entered.set()
        try:
            while True:
                checkpoint()
                finished.wait(0.01)
        finally:
            finished.set()

    monkeypatch.setattr(recovery, 'recover_online_service', blocked)
    cancel = cancellation()
    with file_lock(lock):
        with recovery.BackgroundCaptureRecovery(cfg, 'run', cancel) as job:
            assert entered.wait(2)
            with pytest.raises(BlockingIOError):
                recovery.recover_online(cfg, 'run')
            # Stand-in for new rollout work: it can execute while upload waits.
            write_json(root / 'uploads/new.json', {'session_id': 'new'})
            assert not finished.is_set()
            if stop_by_cancel:
                cancel.event.set()
        assert finished.is_set() and not job.thread.is_alive()
    status = read_json(tmp_path / 'runs/run/capture-recovery.json')
    assert status['status'] == 'stopped' and status['execution_attempt_id'] == '009'


@pytest.mark.parametrize('failure_phase', ['snapshot', 'worker'])
def test_recovery_error_is_observable_without_aborting_rollout(tmp_path, monkeypatch, failure_phase):
    cfg = config(tmp_path)
    if failure_phase == 'snapshot':
        write_json(tmp_path / 'spool/run/spill/bad.json', {})
    else:
        write_json(tmp_path / 'spool/run/uploads/old.json', {})
        def fail(*a, **kw):
            raise OSError('upload unavailable')
        monkeypatch.setattr(recovery, 'recover_online_service', fail)
    with recovery.BackgroundCaptureRecovery(cfg, 'run', cancellation()):
        rollout_executed = True
    assert rollout_executed
    assert read_json(tmp_path / 'runs/run/capture-recovery.json')['status'] == 'failed'
    assert 'error' in read_json(tmp_path / 'runs/run/capture-recovery-error.json')


def test_scoped_recovery_publishes_old_bytes_and_preserves_new_sessions(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    root = tmp_path / 'spool/run'
    trajectory = tmp_path / 'trajectory.json'
    write_json(trajectory, {'messages': []})
    for sid in ['old', 'new', 'spill-session']:
        tensor = root / 'raw' / (sid + '.npz')
        tensor.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tensor, layer_0=np.ones(4))
        write_json(root / 'index' / (sid + '.json'), {
            'session_id': sid, 'sample_id': sid, 'tensor': 'raw/' + tensor.name,
            'sha256': sha256(tensor), 'artifact_state': 'local_persisted'})
        write_json(root / 'requests' / (sid + '.json'), {'session_id': sid})
        write_json(root / 'sessions' / sid / 'finalization.json', {'trajectory_sha256': sha256(trajectory)})
        write_json(root / 'sessions' / sid / 'coverage.json', {'missing': []})
    for owner in ['hidden', 'enrichment']:
        write_json(tmp_path / owner / 'manifest.json', {'online_captures': {'old': {'status': 'pending_upload'}}})
    snapshot = {'uploads': (), 'index': (root / 'index/old.json', root / 'index/spill-session.json'),
                'requests': (root / 'requests/old.json', root / 'requests/spill-session.json'),
                'finalizations': (root / 'sessions/old/finalization.json', root / 'sessions/spill-session/finalization.json'),
                'deferred_sessions': {'spill-session'}}
    untouched = {p: p.read_bytes() for sid in ['new', 'spill-session']
                 for p in [root / 'raw' / (sid + '.npz'), root / 'index' / (sid + '.json'),
                           root / 'requests' / (sid + '.json'), root / 'sessions' / sid / 'finalization.json']}
    monkeypatch.setattr(recovery, 'load_snapshot', lambda *a: cfg)
    import application.execution.evidence as evidence
    monkeypatch.setattr(evidence, 'verified_trajectory', lambda *a: trajectory)
    result = recovery.recover_online_service(cfg, 'run', snapshot=snapshot, checkpoint=lambda: None)
    assert result['sessions'] == {'old': 'complete'}
    assert not (root / 'raw/old.npz').exists()
    assert not (root / 'requests/old.json').exists()
    nas = tmp_path / 'nas/online/run'
    assert read_json(nas / 'requests/old.json')['session_id'] == 'old'
    assert list((nas / 'shards').glob('*.tar'))
    assert all(p.read_bytes() == content for p, content in untouched.items())
    assert read_json(tmp_path / 'hidden/manifest.json')['online_captures']['old']['published']


def test_publish_wait_is_cooperatively_cancellable():
    spool = CaptureSpool.__new__(CaptureSpool)
    spool.lock = threading.RLock()
    spool.condition = threading.Condition(spool.lock)
    spool.pending = {object(): {'session_id': 'old'}}
    spool.seal = lambda sid: None
    stopped = threading.Event()
    timer = threading.Timer(0.03, stopped.set)
    def check():
        if stopped.is_set():
            raise recovery.RecoveryStopped()
    timer.start()
    stamp = time.monotonic()
    try:
        with pytest.raises(recovery.RecoveryStopped):
            spool.publish_session('old', 30, checkpoint=check)
    finally:
        timer.join()
    assert time.monotonic() - stamp < 2


def test_formal_resume_executes_agent_while_history_recovery_is_waiting(cfg, monkeypatch):
    pytest.importorskip('docker')
    pytest.importorskip('minisweagent')
    import core.model_identity
    import runtime.serving.startup
    from application.execution.run_plan import create_run
    from catalog.store import Catalog
    from core.deadline import Deadline
    from rollout import agent, model
    from rollout.runner import run_manifest
    from runtime.containers import rollout_adapter
    from runtime.serving import manager
    from tests.conftest import task

    Catalog(cfg['project_root']).register_tasks([task('recovery-fixture')])
    run_id = 'parallel-capture-test'
    create_run(cfg, run_id, operation='rollout', repeat_count=1)
    write_json(Path(cfg['project_root']) / 'spool' / run_id / 'uploads/old.json', {})
    entered, agent_called, finished = threading.Event(), threading.Event(), threading.Event()
    original_class = recovery.BackgroundCaptureRecovery
    def background(config, rid, cancel, **kw):
        enabled_cfg = {**config, 'capture': {'enabled': True, 'export': {'spool_root': 'spool'}}}
        return original_class(enabled_cfg, rid, cancel, enabled=True)
    def blocked(*a, checkpoint, **kw):
        entered.set()
        try:
            while True:
                checkpoint()
                finished.wait(0.01)
        finally:
            finished.set()
    class Agent:
        def __init__(self, *a, **kw):
            pass
        def run(self, problem):
            assert entered.wait(2) and not finished.is_set()
            agent_called.set()
            return {'exit_status': 'Submitted', 'submission': 'patch'}
        def save(self, path, extra):
            write_json(path, {**extra, 'messages': [{'role': 'assistant', 'reasoning_content': 'done'}]})
    class Environment:
        def __init__(self, *a):
            pass
        def close(self):
            pass
    class Serving:
        def __init__(self, *a):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
    def finalize(*a, **kw):
        assert finished.is_set()
    import application.execution.capture_publication as publication
    monkeypatch.setattr(publication, 'finalize_capture_publication', finalize)
    monkeypatch.setattr(recovery, 'BackgroundCaptureRecovery', background)
    monkeypatch.setattr(recovery, 'recover_online_service', blocked)
    monkeypatch.setattr(runtime.serving.startup, 'check_tool_chain', lambda *a: None)
    monkeypatch.setattr(core.model_identity, 'verify_model', lambda *a: None)
    monkeypatch.setattr(agent, 'DurableAgent', Agent)
    monkeypatch.setattr(model, 'make_model', lambda *a: None)
    monkeypatch.setattr(rollout_adapter, 'RolloutEnvironment', Environment)
    monkeypatch.setattr(manager, 'ServingManager', Serving)
    factory = SimpleNamespace(deadline=Deadline(), lock_images=lambda rows: None, cleanup_owned=lambda run: None)
    result = run_manifest(cfg, run_id, resume=True, factory=factory)
    assert agent_called.is_set() and finished.is_set()
    assert list(result.values())[0]['exit_status'] == 'Submitted'
