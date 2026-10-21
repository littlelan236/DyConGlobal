"""Task-only Catalog and isolated run evidence, including migration boundaries."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from application.execution.evidence import RunEvidence
from catalog.store import Catalog
from core.io import hash_json, read_json, sha256, write_json
from difficulty.datasets.selection import select_run
from migration.catalog_to_runs import migrate
from rollout.artifacts import import_rollout
from tests.conftest import import_fixture


def test_rollout_and_evaluation_do_not_change_benchmark_catalog(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg)
    benchmark = (root / 'catalog/manifest.json').read_bytes()
    evidence = RunEvidence(root, 'source')
    row = evidence.read()['rollouts'][rid]
    report = root / 'runs/source/harness/new/summary.json'
    outcomes = {rid: {'status': 'completed', 'resolved': False, 'patch_sha256': row['patch_sha256']}}
    write_json(report, {'outcomes': outcomes})
    evidence.record_evaluation('new', report, outcomes, {})
    assert (root / 'catalog/manifest.json').read_bytes() == benchmark
    assert set(Catalog(root).read()) == {'schema_version', 'catalog_revision', 'tasks'}
    assert evidence.read()['rollouts'][rid]['resolved'] is False


def test_same_trajectory_in_two_runs_remains_independently_addressable(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg, resolved=True)
    source = RunEvidence(root, 'source').read()
    row = source['rollouts'][rid]
    other = import_rollout(root, {'source': root / row['trajectory_path'],
        'task': source['tasks'][row['task_key']], 'run_id': 'other', 'attempt_id': 'a0',
        'model_fingerprint': row['model_fingerprint']})
    assert other == rid
    assert RunEvidence(root, 'other').read()['rollouts'][rid]['resolved'] is None
    assert RunEvidence(root, 'source').read() == source
    assert select_run(root, 'other', [rid])['source_run_id'] == 'other'


def test_run_scope_is_required_and_foreign_ids_are_rejected(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg)
    with pytest.raises(ValueError, match='source_run'):
        select_run(root, None, [rid])
    with pytest.raises(ValueError, match='Unknown rollout IDs'):
        select_run(root, 'unrelated', [rid])
    with pytest.raises(ValueError, match='Duplicate'):
        select_run(root, 'source', [rid, rid])
    listed = root / 'ids.json'
    write_json(listed, [])
    with pytest.raises(ValueError, match='nonempty'):
        select_run(root, 'source', rollout_list=listed)
    write_json(listed, [{'rollout_id': rid, 'trajectory_sha256': 'changed'}])
    with pytest.raises(ValueError, match='SHA mismatch'):
        select_run(root, 'source', rollout_list=listed)


def test_run_evidence_survives_removal_of_global_catalog(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg, resolved=True)
    (root / 'catalog/manifest.json').unlink()
    assert select_run(root, 'source', [rid])['rollouts'][rid]['resolved'] is True
    assert RunEvidence(root, 'source').verify()['verified']


def test_invalid_evaluation_and_failed_transaction_leave_run_unchanged(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg, resolved=False)
    evidence = RunEvidence(root, 'source')
    original = evidence.path.read_bytes()
    with pytest.raises(ValueError, match='disagrees'):
        with evidence.transaction() as data:
            data['rollouts'][rid]['resolved'] = True
    assert evidence.path.read_bytes() == original


def test_migration_retains_only_explicit_run_and_is_repeatable(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg, resolved=True)
    evidence = RunEvidence(root, 'source')
    original = evidence.read()
    other = copy.deepcopy(original['rollouts'][rid])
    other['run_id'] = 'discarded'
    old = {'schema_version': 1, 'catalog_revision': 7,
           'tasks': Catalog(root).read()['tasks'],
           'rollouts': {rid: original['rollouts'][rid], 'discarded-id': other},
           'evaluations': original['evaluations'], 'baseline_selections': {}}
    write_json(root / 'catalog/manifest.json', old)
    evidence.path.unlink()
    before = (root / 'catalog/manifest.json').read_bytes()
    planned = migrate(root, ['source'])
    assert planned['retained_rollouts'] == 1 and planned['discarded_registrations'] == 1
    assert (root / 'catalog/manifest.json').read_bytes() == before
    assert not evidence.path.exists()
    assert migrate(root, ['source'], apply=True)['state'] == 'complete'
    assert evidence.read()['rollouts'][rid] == original['rollouts'][rid]
    assert not RunEvidence(root, 'discarded').path.exists()
    assert set(Catalog(root).read()) == {'schema_version', 'catalog_revision', 'tasks'}
    assert migrate(root, ['source'], apply=True)['state'] == 'already_migrated'


def test_concurrent_run_evidence_updates_preserve_all_rows(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg)
    evidence = RunEvidence(root, 'source')
    original = evidence.read()['rollouts'][rid]
    task = evidence.read()['tasks'][original['task_key']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: evidence.register_rollout('attempt-' + str(i), copy.deepcopy(original), task), range(8)))
    assert len(evidence.read()['rollouts']) == 9


@pytest.mark.parametrize("builder", ["index", "token_join"])
def test_offline_joins_require_source_run_before_reading_inputs(cfg, builder):
    if builder == "index":
        from difficulty.datasets.index import build_index as build
    else:
        from difficulty.datasets.token_join import build_token_hidden_index as build
    with pytest.raises(ValueError, match="explicit source run"):
        build(cfg, "unread", "unread", "output")
