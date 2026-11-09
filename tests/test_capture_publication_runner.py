from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('docker')
pytest.importorskip('minisweagent')

from application.execution.evidence import RunEvidence
from catalog.store import Catalog
from core.deadline import Deadline
from core.io import read_json, write_json
from core.locking import file_lock
from tests.conftest import task


@pytest.mark.parametrize('publish_fails', [False, True])
def test_runner_calls_publication_after_teardown_under_existing_lock(cfg, monkeypatch, publish_fails):
    import application.execution.capture_publication as publication
    import core.model_identity
    import runtime.serving.startup
    from rollout import agent, model
    from rollout.runner import run_manifest
    from runtime.containers import rollout_adapter
    from runtime.serving import manager

    Catalog(cfg['project_root']).register_tasks([task('capture-fixture')])
    events = []
    class Agent:
        def __init__(self, *a, **kw):
            pass
        def run(self, problem):
            return {'exit_status': 'Submitted', 'submission': 'patch'}
        def save(self, path, extra):
            write_json(path, {**extra, 'messages': [{'role': 'assistant', 'reasoning_content': 'done'}]})
    class Environment:
        def __init__(self, *a):
            pass
        def close(self):
            events.append('environment_exit')
    class Serving:
        def __init__(self, *a):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            events.append('serving_exit')
    def finalize(config, run_id, **kw):
        assert 'serving_exit' in events
        assert events[-1] == 'backend_exit'
        assert RunEvidence(config['project_root'], run_id).verify()['rollouts'] == 1
        lock = Path(config['project_root']) / 'state/run-locks' / (run_id + '.lock')
        with pytest.raises(BlockingIOError):
            with file_lock(lock, nonblocking=True):
                pass
        events.append('publication')
        if publish_fails:
            raise OSError('publication storage failed')
    monkeypatch.setattr(publication, 'finalize_capture_publication', finalize)
    monkeypatch.setattr(runtime.serving.startup, 'check_tool_chain', lambda *a: None)
    monkeypatch.setattr(core.model_identity, 'verify_model', lambda cfg: None)
    monkeypatch.setattr(agent, 'DurableAgent', Agent)
    monkeypatch.setattr(model, 'make_model', lambda *a: None)
    monkeypatch.setattr(rollout_adapter, 'RolloutEnvironment', Environment)
    monkeypatch.setattr(manager, 'ServingManager', Serving)
    factory = SimpleNamespace(deadline=Deadline(), lock_images=lambda rows: None,
                              cleanup_owned=lambda run: events.append('backend_exit'))
    result = run_manifest(cfg, 'new-capture-lifecycle-test', factory=factory)
    assert list(result.values())[0]['exit_status'] == 'Submitted'
    assert events[-1] == 'publication'
    assert events.index('serving_exit') < events.index('publication')
    assert read_json(Path(cfg['project_root']) / 'runs/new-capture-lifecycle-test/status.json')['stage'] == 'complete'
