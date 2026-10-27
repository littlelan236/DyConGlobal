import copy
import os

import pytest

from application.execution.capture_budget_revision import revise_capture_budget
from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.io import hash_json, read_json, sha256, write_json
from core.processes import identity
from runtime.serving.capture import OnlineCaptureController
from runtime.serving.spool import CaptureSpool, publish_files
from tests.conftest import task


def make_run(cfg):
    Catalog(cfg['project_root']).register_tasks([task('budget-task')])
    return create_run(cfg, 'budget-test', operation='rollout', task_ids=['budget-task'])


def revise(cfg, **kwargs):
    return revise_capture_budget(cfg['project_root'], 'budget-test', revision_id='r128',
                                 max_bytes=128 * 2**30, high=.95, low=.95,
                                 reason='Explicit test authorization', **kwargs)


def test_revision_preserves_tasks_model_and_previous_snapshot(cfg):
    run = make_run(cfg)
    old = read_json(run / 'resolved-config.json')
    stable = {name: sha256(run / name) for name in ['tasks.json', 'selection.json', 'rollout/work-items.json']}
    result = revise(cfg)
    new = load_plan(cfg['project_root'], 'budget-test')['config']
    normalized = copy.deepcopy(new)
    normalized['capture']['export'] = old['capture']['export']
    normalized['config_hash'] = old['config_hash']
    assert normalized == old
    assert all(sha256(run / name) == digest for name, digest in stable.items())
    assert read_json(run / 'config-revisions/r128/before/resolved-config.json') == old
    assert new['capture']['export']['spool_max_bytes'] == 128 * 2**30
    assert result['new_config_hash'] == new['config_hash']
    assert revise(cfg) == result


def test_live_execution_blocks_revision(cfg):
    run = make_run(cfg)
    path = run / 'execution/attempts/001'
    path.mkdir()
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    write_json(path / 'process.json', identity(os.getpid()))
    old = sha256(run / 'resolved-config.json')
    with pytest.raises(RuntimeError, match='still active'):
        revise(cfg)
    assert sha256(run / 'resolved-config.json') == old


def test_incomplete_cleanup_blocks_revision(cfg):
    run = make_run(cfg)
    path = run / 'execution/attempts/001'
    path.mkdir()
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    write_json(path / 'exit.json', {'exit_code': -15})
    write_json(path / 'termination.json', {'cleanup_complete': False})
    with pytest.raises(RuntimeError, match='cleanup is incomplete'):
        revise(cfg)


def test_config_versions_preserve_legacy_bytes(tmp_path):
    old = {'export': {'spool_max_bytes': 32 * 2**30}}
    new = {'export': {'spool_max_bytes': 128 * 2**30}}
    write_json(tmp_path / 'capture-config.json', old)
    controller = OnlineCaptureController.__new__(OnlineCaptureController)
    controller.export_root = tmp_path
    controller.spool = None
    controller.config = new
    ref = controller.write_configuration()
    assert ref.startswith('capture-config-revisions/')
    assert read_json(tmp_path / ref) == new
    assert read_json(tmp_path / 'capture-config.json') == old
    assert controller.write_configuration() == ref


def test_storage_status_reports_live_budget_not_a_copied_limit(tmp_path):
    from types import SimpleNamespace
    controller = OnlineCaptureController.__new__(OnlineCaptureController)
    controller.config = {'export': {'spool_max_bytes': 99,
                         'spool_high_watermark': .95, 'spool_low_watermark': .95}}
    controller.spool = SimpleNamespace(base=tmp_path,
        budget=SimpleNamespace(limit=128 * 2**30, reserve=8 * 2**30, usage=lambda: 11))
    controller.capture_config_ref = 'capture-config-revisions/test.json'
    value = controller.storage_status()
    assert value['spool_max_bytes'] == 128 * 2**30
    assert value['spool_high_watermark'] == value['spool_low_watermark'] == .95
    assert value['physical_spool_root'] == str(tmp_path)


def test_metadata_publication_includes_version_ref(tmp_path):
    old = {'budget': 32}
    new = {'budget': 128}
    source, destination = tmp_path / 'local', tmp_path / 'nas'
    source.mkdir()
    destination.mkdir()
    ref = 'capture-config-revisions/' + hash_json(new) + '.json'
    write_json(source / 'capture-config.json', old)
    write_json(destination / 'capture-config.json', old)
    write_json(source / ref, new)
    write_json(source / 'sessions/s/session.json', {'session_id': 's',
               'capture_config_ref': ref, 'capture_config_sha256': hash_json(new)})
    class Condition:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def wait_for(self, condition, timeout):
            return condition()
    class Spool:
        root = source
        pending = {}
        failures = []
        condition = Condition()
        submitted = None
        def seal(self, session):
            pass
        def write_json(self, relative, value):
            write_json(self.root / relative, value)
            return self.root / relative
        def _submit(self, job):
            self.submitted = job
    spool = Spool()
    assert CaptureSpool.publish_session(spool, 's', 1)
    assert ref in spool.submitted['files']
    publish_files(source, destination, spool.submitted['files'], 5)
    assert read_json(destination / 'capture-config.json') == old
    assert read_json(destination / ref) == new
