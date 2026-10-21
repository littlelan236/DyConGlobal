from application.execution.evidence import RunEvidence
from pathlib import Path

import pytest

from application.execution.run_plan import create_run
from catalog.store import Catalog
from core.io import read_json, sha256, write_json
from migration.import_capture import migrate
from tests.conftest import import_fixture


def capture_fixture(cfg):
    root = Path(cfg['project_root'])
    rid = import_fixture(cfg)
    row = RunEvidence(root, 'source').read()['rollouts'][rid]
    nas = root / 'nas/source'
    session = nas / 'sessions/cap'
    write_json(session / 'session.json', {'session_id': 'cap'})
    write_json(session / 'receipt.json', {'status': 'stopped'})
    write_json(session / 'coverage.json', {'expected': 1, 'observed': 1, 'missing': []})
    final = {'rollout_id': rid, 'trajectory_sha256': row['trajectory_sha256'],
             'trajectory_path': '/obsolete/baseline/raw.traj.json', 'model_fingerprint': 'fixture-model',
             'accepted_native_request_ids': ['request-1']}
    write_json(session / 'finalization.json', final)
    write_json(root / 'hidden/manifest.json', {'online_captures': {'cap': {
        'run_id': 'source', 'session_id': 'cap', 'rollout_id': rid,
        'status': 'complete', 'published': True, 'trajectory_sha256': row['trajectory_sha256'],
        'finalization_path': str(session / 'finalization.json'), 'export_root': str(nas),
        'sample_count': 1}}})
    selection = root / 'chosen.json'
    write_json(selection, {'rollouts': [{'rollout_id': rid, 'trajectory_sha256': row['trajectory_sha256'],
                                       'patch_sha256': row['patch_sha256']}]})
    create_run(cfg, 'migrated', operation='harness', task_ids=['t-1'])
    return root, rid, row, selection, session


def test_migration_preserves_identity_payload_and_nas_sources(cfg):
    root, rid, old, selection, session = capture_fixture(cfg)
    source_hash = sha256(session / 'finalization.json')
    receipt = migrate(root, 'source', 'migrated', selection)
    new = RunEvidence(root, 'migrated').read()['rollouts'][rid]
    assert new['run_id'] == 'migrated'
    assert new['trajectory_sha256'] == old['trajectory_sha256']
    assert new['patch_sha256'] == old['patch_sha256']
    assert new['provenance']['source_run_id'] == 'source'
    assert new['resolved'] is None and new['evaluation_status'] == 'not_run'
    assert sha256(session / 'finalization.json') == source_hash
    assert (root / old['trajectory_path']).is_file()
    final = read_json(root / 'runs/migrated/hidden/sessions/cap/finalization.json')
    assert final['trajectory_path'] == str(root / new['trajectory_path'])
    assert receipt['rollouts'] == 1
    assert migrate(root, 'source', 'migrated', selection) == receipt


def test_migration_rejects_partial_capture_before_catalog_mutation(cfg):
    root, rid, old, selection, session = capture_fixture(cfg)
    write_json(session / 'coverage.json', {'missing': ['request-1']})
    with pytest.raises(ValueError, match='missing evidence'):
        migrate(root, 'source', 'migrated', selection)
    assert RunEvidence(root, 'source').read()['rollouts'][rid] == old


def test_migration_rejects_selection_content_mismatch(cfg):
    root, rid, old, selection, session = capture_fixture(cfg)
    value = read_json(selection)
    value['rollouts'][0]['patch_sha256'] = 'bad'
    write_json(selection, value)
    with pytest.raises(ValueError, match='Selection hash'):
        migrate(root, 'source', 'migrated', selection)
    assert RunEvidence(root, 'source').read()['rollouts'][rid] == old


def test_bounded_finalize_uses_explicit_current_locator_and_ledger_hash(tmp_path):
    import numpy as np
    from core.io import hash_json
    from enrichment.online_trace import finalize_token_events

    nas = tmp_path / 'nas'
    trajectory = tmp_path / 'current.traj.json'
    write_json(trajectory, {'messages': []})
    ledger = nas / 'requests/one.json'
    write_json(ledger, {'schema_version': 2, 'request_id': 'native-1', 'session_id': 'cap',
        'task_id': 'task', 'step_id': 0, 'prompt_token_ids': [1, 10],
        'output_token_ids': [20, 11, 30], 'request_sha256': hash_json([1, 10]),
        'anchors': ['think_close']})
    write_json(nas / 'requests/unselected.json', {'intentionally': 'not a ledger'})
    final = nas / 'sessions/cap/finalization.json'
    write_json(final, {'accepted_native_request_ids': ['native-1'], 'rollout_id': 'rollout',
        'trajectory_path': '/obsolete/old.traj.json', 'trajectory_sha256': sha256(trajectory),
        'model_fingerprint': 'model'})
    original_sha = sha256(final)
    (nas / 'raw').mkdir()
    np.savez(nas / 'raw/event.npz', layer_0=np.ones(4))
    event = {'sample_id': 'sample', 'request_id': 'native-1', 'session_id': 'cap',
        'task_id': 'task', 'rollout_id': 'pending', 'step_id': 0, 'absolute_position': 3,
        'token_id': 11, 'anchor': 'think_close', 'tensor': 'raw/event.npz',
        'sha256': sha256(nas / 'raw/event.npz')}
    index = nas / 'index/shard.json'
    write_json(index, [event])
    kwargs = dict(open_ids=[10], close_ids=[11],
        event_rows=[(index, event)], trajectory_paths={'rollout': str(trajectory)})
    with pytest.raises(ValueError, match='Canonical request ledger checksum'):
        finalize_token_events(nas, tmp_path / 'bad', request_evidence=[{
            'ledger_ref': 'requests/one.json', 'ledger_sha256': 'bad'}], **kwargs)
    output = finalize_token_events(nas, tmp_path / 'good', request_evidence=[{
        'ledger_ref': 'requests/one.json', 'ledger_sha256': sha256(ledger)}], **kwargs)
    rows = read_json(output / 'index.json')
    assert len(rows) == 1 and rows[0]['rollout_id'] == 'rollout'
    assert rows[0]['alignment_status'] == 'exact'
    assert rows[0]['remaining_steps'] == 1
    assert sha256(final) == original_sha


@pytest.mark.parametrize('finite', [True, False])
def test_tensor_audit_checks_payload_finiteness_not_just_hashes(tmp_path, finite):
    import hashlib
    import io
    import tarfile
    import numpy as np
    from migration.verify_capture import verify_shard

    data = io.BytesIO()
    np.savez(data, layer_0=np.array([1., 2. if finite else np.nan], dtype=np.float32))
    content = data.getvalue()
    shard = tmp_path / 'one.tar'
    with tarfile.open(shard, 'w:') as archive:
        member = tarfile.TarInfo('one.npz')
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    digest = hashlib.sha256(content).hexdigest()
    row = {'tensor_locator': {'shard': 'one.tar', 'member': 'one.npz',
            'member_sha256': digest, 'shard_sha256': sha256(shard)},
           'sha256': digest, 'storage_shapes': {'layer_0': [2]},
           'storage_dtypes': {'layer_0': 'float32'}}
    if finite:
        assert verify_shard((tmp_path, 'one.tar', [row]))['samples'] == 1
    else:
        with pytest.raises(ValueError, match='finite-value'):
            verify_shard((tmp_path, 'one.tar', [row]))


def test_online_index_uses_migrated_publication_root_and_null_success_labels(cfg):
    from application.execution.run_plan import load_plan
    from difficulty.datasets.online import build_published_index

    root, rid, old, selection, session = capture_fixture(cfg)
    migrate(root, 'source', 'migrated', selection)
    run = root / 'runs/migrated'
    trace = run / 'enrichment/online'
    write_json(trace / 'index.json', [{'rollout_id': rid,
        'trajectory_sha256': old['trajectory_sha256'], 'censored': False,
        'remaining_steps': 2, 'remaining_reasoning_tokens_after': 3}])
    publication = read_json(run / 'hidden/publication.json')
    write_json(trace / 'receipt.json', {'origin': 'migrated',
        'files': {'index.json': sha256(trace / 'index.json')},
        'source_publication_sha256': sha256(run / 'hidden/publication.json'),
        'source_root': publication['nas_locator']})
    frozen = load_plan(root, 'migrated')['config']
    out = build_published_index(frozen, 'migrated', 'view')
    index = read_json(out / 'index.json')
    assert index['hidden_root'] == str(root / 'nas/source')
    assert index['samples'][0]['labels'] == {
        'remaining_steps': None, 'remaining_reasoning_tokens': None}
    publication['nas_locator'] = str(root / 'wrong')
    write_json(run / 'hidden/publication.json', publication)
    with pytest.raises(ValueError, match='publication changed'):
        build_published_index(frozen, 'migrated', 'tampered')
