import os

import pytest

from application.execution.config_revision import revise_config
from application.execution.run_plan import load_plan
from core.io import read_json, write_json
from core.processes import identity
from tests.test_capture_budget_revision import make_run


def _blocked(*args):
    raise RuntimeError('unrelated project capture writer is active')


def test_quota_only_increase_does_not_require_other_runs_to_stop(cfg, monkeypatch):
    run = make_run(cfg)
    before = read_json(run / 'resolved-config.json')
    limit = before['capture']['export']['spool_max_bytes'] * 2
    monkeypatch.setattr('hidden.relocate_spool.assert_no_writers', _blocked)
    revise_config(cfg['project_root'], 'budget-test', revision_id='quota-only',
                  updates={'capture.export.spool_max_bytes': limit}, reason='fixture')
    after = load_plan(cfg['project_root'], run.name)['config']
    assert after['capture']['export']['spool_max_bytes'] == limit
    assert after['capture']['export']['spool_root'] == before['capture']['export']['spool_root']


def test_quota_decrease_and_storage_change_keep_global_writer_guard(cfg, monkeypatch):
    run = make_run(cfg)
    before = read_json(run / 'resolved-config.json')
    monkeypatch.setattr('hidden.relocate_spool.assert_no_writers', _blocked)
    for update in [{'capture.export.spool_max_bytes': before['capture']['export']['spool_max_bytes'] // 2},
                   {'capture.export.spool_root': 'another-spool'}]:
        with pytest.raises(RuntimeError, match='unrelated project'):
            revise_config(cfg['project_root'], run.name, revision_id='blocked', updates=update, reason='fixture')


def test_live_target_still_blocks_quota_increase(cfg):
    run = make_run(cfg)
    old = read_json(run / 'resolved-config.json')['capture']['export']['spool_max_bytes']
    write_json(run / 'execution/latest.json', {'attempt': '001'})
    write_json(run / 'execution/attempts/001/process.json', identity(os.getpid()))
    with pytest.raises(RuntimeError, match='still active'):
        revise_config(cfg['project_root'], run.name, revision_id='live-quota',
                      updates={'capture.export.spool_max_bytes': old * 2}, reason='fixture')
