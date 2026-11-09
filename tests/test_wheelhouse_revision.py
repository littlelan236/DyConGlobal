import zipfile

import pytest

from application.execution.config_revision import revise_config
from application.execution.run_plan import load_plan
from core.io import read_json, sha256
from runtime.wheelhouse import build
from tests.test_capture_budget_revision import make_run


def make_wheel(path):
    path.mkdir(parents=True, exist_ok=True)
    wheel = path / 'demo-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('demo-1.0.dist-info/METADATA',
                         'Name: demo\nVersion: 1.0\nRequires-Python: >=3.6\n\n')
    return path


def test_wheelhouse_revision_updates_only_environment_and_preserves_progress(cfg, tmp_path):
    run = make_run(cfg)
    source = make_wheel(tmp_path / 'source')
    published = build(cfg, 'revision-fixture', [source])
    before = read_json(run / 'resolved-config.json')
    protected = {name: sha256(run / name) for name in
                 ['selection.json', 'tasks.json', 'rollout/work-items.json', 'evidence.json']
                 if (run / name).exists()}
    args = dict(revision_id='wheelhouse', updates={'environment.wheelhouse': published['path']},
                reason='User requested verified wheelhouse')

    receipt = revise_config(cfg['project_root'], run.name, **args)
    after = load_plan(cfg['project_root'], run.name)['config']
    normalized = dict(after)
    normalized['environment'] = before['environment']
    normalized['config_hash'] = before['config_hash']

    assert after['environment']['wheelhouse'] == published['path']
    assert normalized == before
    assert all(sha256(run / name) == digest for name, digest in protected.items())
    assert receipt['changed_config_fields'] == ['environment.wheelhouse']
    assert read_json(run / 'events/config-revision-wheelhouse.json')['event'] == 'run.config_revised'
    assert revise_config(cfg['project_root'], run.name, **args) == receipt


def test_wheelhouse_revision_rejects_unverified_path_without_mutation(cfg, tmp_path):
    run = make_run(cfg)
    before = sha256(run / 'resolved-config.json')

    with pytest.raises((FileNotFoundError, ValueError)):
        revise_config(cfg['project_root'], run.name, revision_id='bad-wheelhouse',
                      updates={'environment.wheelhouse': str(tmp_path / 'missing')}, reason='Invalid fixture')

    assert sha256(run / 'resolved-config.json') == before
    assert not (run / 'config-revisions/bad-wheelhouse').exists()
