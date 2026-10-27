from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.io import hash_json, read_json, sha256, write_json
from hidden.publish_online import native_complete_suffix, publish_online
from hidden.repair_points import recover_points


def sequence(tmp_path, monkeypatch, gap):
    import hidden.publish_online as publisher
    import hidden.repair_points as points
    sid, run_id = 'cap-s-a-attempt', 'run'
    spool = tmp_path / 'spool/run'
    cfg = {'project_root': str(tmp_path), 'parallel': {'enabled': False},
           'capture': {'anchors': ['think_close'], 'layers': [0],
                       'export': {'spool_root': 'spool', 'root': 'online', 'upload_max_elapsed_seconds': 30}},
           'paths': {'nas_root': str(tmp_path / 'nas')}}
    messages = []
    for i, length in enumerate([2, 3, 1]):
        md = {'session_id': sid, 'run_id': run_id, 'task_id': 'task',
              'query_id': 'q' + str(i), 'api_attempt_id': 'api' + str(i)}
        reqid = 'native-' + str(i)
        output = [5] * length + [11, 20]
        req = {**md, 'request_id': reqid, 'anchors': ['think_close'], 'prompt_token_ids': [10],
               'output_token_ids': output, 'request_sha256': hash_json([10]),
               'output_sha256': hash_json(output), 'open_token_ids': [10], 'close_token_ids': [11],
               'step_id': i, 'rollout_id': 'pending'}
        messages.append({'role': 'assistant', 'extra': {'request': {'dycon_capture': md},
                                                       'response': {'id': reqid}}})
        if not (i == 1 and gap == 'native'):
            write_json(spool / 'requests' / (reqid + '.json'), req)
        if not (i == 1 and gap == 'hidden'):
            tensor = spool / 'raw' / (str(i) + '.npz')
            tensor.parent.mkdir(parents=True, exist_ok=True)
            np.savez(tensor, layer_0=np.ones(4, dtype=np.float32))
            event = {**md, 'request_id': reqid, 'sample_id': 'sample-' + str(i),
                     'step_id': i, 'rollout_id': 'pending', 'anchor': 'think_close',
                     'absolute_position': 1 + length, 'token_id': 11,
                     'request_sha256': req['request_sha256'], 'model_fingerprint': 'model',
                     'tensor': 'raw/' + tensor.name, 'sha256': sha256(tensor), 'status': 'persisted',
                     'storage_shapes': {'layer_0': [4]}, 'storage_dtypes': {'layer_0': 'float32'}}
            write_json(spool / 'index' / (str(i) + '.json'), event)
    run = tmp_path / 'runs/run'
    trajectory = run / 'trajectory.json'
    write_json(trajectory, {'messages': messages, 'info': {'exit_status': 'Submitted'}})
    patch = run / 'patch.diff'
    patch.write_text('patch', encoding='utf-8')
    report = run / 'report.json'
    write_json(report, {'outcomes': {'rollout': {'status': 'completed', 'resolved': True, 'patch_sha256': sha256(patch)}}})
    row = {'run_id': 'run', 'sample_key': 's-a', 'attempt_id': 'attempt', 'task_key': 'task-key',
           'trajectory_path': 'runs/run/trajectory.json', 'trajectory_sha256': sha256(trajectory),
           'patch_path': 'runs/run/patch.diff', 'patch_sha256': sha256(patch),
           'model_fingerprint': 'model', 'provenance': {}, 'submission_status': 'submitted',
           'evaluation_status': 'completed', 'resolved': True, 'selected_evaluation_id': 'eval'}
    evidence = {'revision': 1, 'rollouts': {'rollout': row},
                'tasks': {'task-key': {'instance_id': 'task', 'partition': 'train', 'repo': 'repo', 'base_commit': 'commit'}},
                'evaluations': {'eval': {'report_path': 'runs/run/report.json', 'report_sha256': sha256(report),
                                        'outcomes': {'rollout': {'status': 'completed', 'resolved': True, 'patch_sha256': sha256(patch)}}}}}
    monkeypatch.setattr(points, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
    monkeypatch.setattr(publisher, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
    monkeypatch.setattr(publisher, 'load_snapshot', lambda *a: cfg)
    write_json(run / 'resolved-config.json', cfg)
    recover_points(cfg, 'run', session_ids=[sid], apply=True, repair_id='repair')
    return cfg, evidence


def publish(cfg):
    return publish_online(cfg, 'run', repair_id='repair', publication_id='available',
                          resolved_only=True, coverage_policy='available_native_suffix_v1')


def test_missing_hidden_does_not_reduce_remaining_token_label(tmp_path, monkeypatch):
    cfg, evidence = sequence(tmp_path, monkeypatch, 'hidden')
    legacy = tmp_path / 'runs/run/enrichment/online/sentinel'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('unchanged')
    result = publish(cfg)
    assert result['published_rollouts'] == 1 and result['published_partial_rollouts'] == 1
    assert result['published_points'] == 2
    out = Path(result['path'])
    rows = read_json(out / 'index.json')
    assert [r['remaining_reasoning_tokens_after'] for r in rows] == [4, 0]
    assert [r['remaining_steps'] for r in rows] == [3, 1]
    coverage = read_json(out / 'coverage.json')
    assert len(coverage['missing']) == 1 and not coverage['unexpected']
    assert coverage['policy'] == 'available_points'
    assert result['known_missing_hidden_in_qualified_suffixes'] == 1
    assert legacy.read_text() == 'unchanged'
    assert publish(cfg) == result
    (out / 'index.json').write_text('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        publish(cfg)


def test_unknown_native_response_excludes_earlier_points_but_keeps_suffix(tmp_path, monkeypatch):
    cfg, evidence = sequence(tmp_path, monkeypatch, 'native')
    result = publish(cfg)
    rows = read_json(Path(result['path']) / 'index.json')
    assert [r['sample_id'] for r in rows] == ['sample-2']
    assert rows[0]['remaining_reasoning_tokens_after'] == 0
    assert rows[0]['remaining_steps'] == 1
    assert result['point_exclusion_counts'] == {'unknown_future_reasoning_tokens': 1, 'not_native_exact': 1}
    assert result['sessions'][0]['unknown_response_indices'] == [1]


def test_named_publication_flows_through_index_and_resolved_dataset(tmp_path, monkeypatch):
    import core.config
    import difficulty.datasets.online as online
    import difficulty.datasets.resolved as resolved
    cfg, evidence = sequence(tmp_path, monkeypatch, 'hidden')
    publish(cfg)
    monkeypatch.setattr(core.config, 'load_snapshot', lambda *a: cfg)
    for module in [online, resolved]:
        monkeypatch.setattr(module, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
        monkeypatch.setattr(module, 'select_run', lambda *a: {'rollout_ids': ['rollout'], 'evidence_sha256': 'evidence'})
    online.build_published_index(cfg, 'run', 'index', publication_id='available')
    result = resolved.build_resolved_selection(tmp_path, 'run', 'index', 'dataset')
    assert result['samples'] == 2 and result['rollouts'] == 1
    indexpath = tmp_path / 'difficulty/datasets/index/index.json'
    index = read_json(indexpath)
    index['label_root'] = 'runs/other/enrichment/online'
    write_json(indexpath, index)
    with pytest.raises(ValueError, match='label root'):
        resolved.build_resolved_selection(tmp_path, 'run', 'index', 'wrong')


def test_missing_action_response_also_blocks_earlier_suffix():
    bound = {'a': {'message_index': 0}, 'c': {'message_index': 4}}
    kept, cutoff, unknown = native_complete_suffix(bound, [(0, {}, {}), (4, {}, {})], [0, 2, 4])
    assert list(kept) == ['c'] and cutoff == 2 and unknown == [2]


def test_unknown_error_response_invalidates_earlier_reasoning():
    bound = {'a': {'message_index': 0}, 'c': {'message_index': 3}}
    kept, cutoff, unknown = native_complete_suffix(bound, [(0, {}, {}), (1, {}, {}), (3, {}, {})], [0, 3])
    assert list(kept) == ['c'] and cutoff == 1
