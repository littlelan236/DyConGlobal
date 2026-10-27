from types import SimpleNamespace
import pytest

from core.io import read_json, write_json, sha256
from hidden.repair_points import recover_points
from hidden.publish_online import complete_native, publish_online
from test_repair_points import fixture


def prepare(tmp_path, monkeypatch, partial=False):
    import hidden.publish_online as publisher
    cfg, sid, spool, event = fixture(tmp_path, monkeypatch, partial=partial)
    reqpath = spool / 'requests/native.json'
    req = read_json(reqpath)
    req.update(open_token_ids=[10], step_id=0, rollout_id='pending')
    write_json(reqpath, req)
    event.update(step_id=0, rollout_id='pending')
    write_json(spool / 'index/sample.json', event)
    trajectory = tmp_path / 'runs/run/trajectory.json'
    value = read_json(trajectory)
    value['info'] = {'exit_status': 'Submitted'}
    write_json(trajectory, value)
    row = {'run_id': 'run', 'sample_key': 's-a', 'attempt_id': 'attempt', 'task_key': 'task-key',
           'trajectory_path': 'runs/run/trajectory.json', 'trajectory_sha256': sha256(trajectory),
           'model_fingerprint': 'model', 'provenance': {}, 'submission_status': 'submitted',
           'evaluation_status': 'completed', 'resolved': True, 'selected_evaluation_id': 'eval',
           'patch_sha256': 'patch'}
    report = tmp_path / 'runs/run/report.json'
    write_json(report, {'outcomes': {'rollout': {'status': 'completed', 'resolved': True, 'patch_sha256': 'patch'}}})
    evidence = {'rollouts': {'rollout': row}, 'tasks': {'task-key': {'instance_id': 'task'}},
                'evaluations': {'eval': {'report_path': 'runs/run/report.json', 'report_sha256': sha256(report)}}}
    import hidden.repair_points as points
    monkeypatch.setattr(points, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
    monkeypatch.setattr(publisher, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
    monkeypatch.setattr(publisher, 'load_snapshot', lambda *a: cfg)
    recover_points(cfg, 'run', session_ids=[sid], apply=True, repair_id='repair')
    return cfg, sid, evidence


def test_complete_repaired_publication_and_idempotence(tmp_path, monkeypatch):
    cfg, sid, evidence = prepare(tmp_path, monkeypatch)
    original = tmp_path / 'nas/online/run/capture-repairs/repair/receipt.json'
    digest = sha256(original)
    result = publish_online(cfg, 'run', repair_id='repair', publication_id='pub', resolved_only=True)
    assert result['published_rollouts'] == 1 and result['published_points'] == 1
    out = tmp_path / 'runs/run/enrichment/online'
    sample = read_json(out / 'index.json')[0]
    assert sample['remaining_steps'] == 1
    assert sample['remaining_reasoning_tokens_after'] == 0
    assert sample['rollout_id'] == 'rollout'
    assert sample['alignment_status'] == 'exact'
    assert not (tmp_path / 'nas/online/run/sessions' / sid / 'finalization.json').exists()
    assert sha256(original) == digest
    assert publish_online(cfg, 'run', repair_id='repair', publication_id='pub', resolved_only=True) == result
    (out / 'index.json').write_text('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        publish_online(cfg, 'run', repair_id='repair', publication_id='pub', resolved_only=True)


def test_partial_native_session_never_promoted(tmp_path, monkeypatch):
    cfg, sid, evidence = prepare(tmp_path, monkeypatch, partial=True)
    with pytest.raises(ValueError, match='nonempty'):
        publish_online(cfg, 'run', repair_id='repair', publication_id='pub', resolved_only=True)
    assert not (tmp_path / 'runs/run/enrichment/online').exists()


def test_unresolved_trajectory_never_promoted(tmp_path, monkeypatch):
    cfg, sid, evidence = prepare(tmp_path, monkeypatch)
    evidence['rollouts']['rollout']['resolved'] = False
    with pytest.raises(ValueError, match='nonempty'):
        publish_online(cfg, 'run', repair_id='repair', publication_id='pub', resolved_only=True)


def test_coverage_requires_all_native_conditions():
    coverage = {'status': 'complete', 'expected_count_complete': True}
    assert complete_native(coverage)
    assert not complete_native({**coverage, 'unbound_responses': [{}]})
    assert not complete_native({**coverage, 'missing_known': [1]})
    assert not complete_native({**coverage, 'expected_count_complete': False})


def test_sharded_repair_uses_original_published_index(tmp_path, monkeypatch):
    import tarfile
    cfg, sid, evidence = prepare(tmp_path, monkeypatch)
    nas = tmp_path / 'nas/online/run'
    shard = nas / 'shards/key.tar'
    shard.parent.mkdir()
    with tarfile.open(shard, 'w') as archive:
        archive.add(nas / 'raw/sample.npz', arcname='sample.npz')
    point_path = nas / 'capture-repairs/repair/sessions' / sid / 'point-index.json'
    points = read_json(point_path)
    event = points['points'][0]
    event['tensor_locator'] = {'shard': 'shards/key.tar', 'member': 'sample.npz',
                               'member_sha256': event['sha256'], 'shard_sha256': sha256(shard)}
    original_index = nas / 'index/shard-key.json'
    write_json(original_index, [event])
    write_json(nas / 'publications/key.json', {'files': {
        'shards/key.tar': sha256(shard), 'index/shard-key.json': sha256(original_index)}})
    write_json(point_path, points)
    session_receipt = point_path.parent / 'receipt.json'
    receipt = read_json(session_receipt)
    receipt['files'][str(point_path.relative_to(nas))] = sha256(point_path)
    receipt['dependencies']['shards/key.tar'] = sha256(shard)
    write_json(session_receipt, receipt)
    top_path = nas / 'capture-repairs/repair/receipt.json'
    manifest = read_json(top_path)
    manifest['session_receipts'][str(session_receipt.relative_to(nas))] = sha256(session_receipt)
    write_json(top_path, manifest)
    result = publish_online(cfg, 'run', repair_id='repair', publication_id='pub', resolved_only=True)
    assert result['published_points'] == 1
    trace = read_json(tmp_path / 'runs/run/enrichment/online/receipt.json')
    assert 'index/shard-key.json' in trace['source_files']
