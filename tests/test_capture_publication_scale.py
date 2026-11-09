"""Synthetic receipt-bound acceptance and scale fixtures, with no GPU/model run."""
from pathlib import Path
import io
import json
import os
import tarfile
import time
import hashlib
import builtins

import numpy as np
import pytest

from application.execution.capture_publication import finalize_capture_publication
from core.io import hash_json, read_json, sha256, write_json
from tests.conftest import task


def cohort(tmp_path, monkeypatch, sessions, expected, observed, task_count):
    import hidden.publish_online as publisher
    import core.config
    run_id = 'synthetic-capture-partial'
    root = tmp_path
    run = root / 'runs' / run_id
    nas = root / 'nas/online' / run_id
    cfg = {'project_root': str(root), 'parallel': {'enabled': False},
           'capture': {'enabled': True, 'publication_seconds': 120, 'anchors': ['think_close'], 'layers': [0],
                       'export': {'enabled': True, 'spool_root': 'spool', 'root': 'online', 'upload_max_elapsed_seconds': 30}},
           'paths': {'nas_root': str(root / 'nas')}}
    write_json(run / 'resolved-config.json', cfg)
    monkeypatch.setattr(publisher, 'load_snapshot', lambda *a: cfg)
    monkeypatch.setattr(core.config, 'load_snapshot', lambda *a: cfg)
    evidence = {'schema_version': 1, 'revision': 1, 'run_id': run_id, 'rollouts': {}, 'tasks': {}, 'evaluations': {}}
    patch = run / 'patch.diff'
    patch.write_text('patch', encoding='utf-8')
    outcomes, items = {}, []
    buf = io.BytesIO()
    np.savez(buf, layer_0=np.ones(4, dtype=np.float32))
    tensor = buf.getvalue()
    tensor_hash = hashlib.sha256(tensor).hexdigest()
    for i in range(sessions):
        task_id, repetition_id = f'task-{i % task_count:03}', f'rep-{i // task_count:03}'
        taskrow = task(task_id)
        task_key = hash_json({k: taskrow[k] for k in ('benchmark', 'revision', 'instance_id')})
        sample_key, attempt_id, rid = f's-{i:03}', repetition_id + '-attempt-000', f'rollout-{i:03}'
        sid = f'cap-{sample_key}-{attempt_id}'
        item = {'sample_key': sample_key, 'instance_id': task_id, 'task_key': task_key, 'repetition_id': repetition_id}
        n, m = expected[i], observed[i]
        events, messages = [], []
        archive_path = nas / 'shards' / (sid + '.tar')
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, 'w') as archive:
            for j in range(n):
                reqid, sampleid = f'native-{i:03}-{j:03}', f'point-{i:03}-{j:03}'
                md = {'session_id': sid, 'run_id': run_id, 'task_id': task_id,
                      'query_id': f'q-{i}-{j}', 'api_attempt_id': f'api-{i}-{j}'}
                request = {**md, 'request_id': reqid, 'anchors': ['think_close'], 'prompt_token_ids': [10],
                           'output_token_ids': [5, 11, 20], 'request_sha256': hash_json([10]),
                           'output_sha256': hash_json([5, 11, 20]), 'open_token_ids': [10], 'close_token_ids': [11],
                           'step_id': j, 'rollout_id': 'pending'}
                write_json(nas / 'requests' / (reqid + '.json'), request)
                messages.append({'role': 'assistant', 'extra': {'request': {'dycon_capture': md}, 'response': {'id': reqid}}})
                if j >= m:
                    continue
                member = sampleid + '.npz'
                info = tarfile.TarInfo(member)
                info.size = len(tensor)
                archive.addfile(info, io.BytesIO(tensor))
                events.append({**md, 'request_id': reqid, 'sample_id': sampleid, 'step_id': j,
                               'rollout_id': 'pending', 'anchor': 'think_close', 'absolute_position': 2,
                               'token_id': 11, 'request_sha256': request['request_sha256'], 'model_fingerprint': 'model',
                               'tensor': 'raw/' + member, 'sha256': tensor_hash, 'status': 'persisted',
                               'storage_shapes': {'layer_0': [4]}, 'storage_dtypes': {'layer_0': 'float32'},
                               'tensor_locator': {'shard': str(archive_path.relative_to(nas)), 'member': member,
                                                  'member_sha256': tensor_hash}})
        shard_digest = sha256(archive_path)
        for event in events:
            event['tensor_locator']['shard_sha256'] = shard_digest
        index_path = nas / 'index' / ('shard-' + sid + '.json')
        write_json(index_path, events)
        write_json(nas / 'publications' / (sid + '.json'), {'files': {
            str(archive_path.relative_to(nas)): shard_digest, str(index_path.relative_to(nas)): sha256(index_path)}})
        trajectory = run / 'trajectories' / (rid + '.json')
        write_json(trajectory, {'messages': messages, 'info': {'exit_status': 'Submitted'}})
        resolved = i < 6 if sessions == 13 else i % 2 == 0
        row = {'run_id': run_id, 'sample_key': sample_key, 'attempt_id': attempt_id, 'task_key': task_key,
               'repetition_id': repetition_id, 'termination_reason': 'Submitted', 'model_fingerprint': 'model',
               'trajectory_path': str(trajectory.relative_to(root)), 'trajectory_sha256': sha256(trajectory),
               'patch_path': str(patch.relative_to(root)), 'patch_sha256': sha256(patch), 'provenance': {},
               'submission_status': 'submitted', 'evaluation_status': 'completed', 'resolved': resolved,
               'selected_evaluation_id': 'eval'}
        evidence['rollouts'][rid] = row
        evidence['tasks'][task_key] = taskrow
        outcomes[rid] = {'resolved': resolved, 'status': 'completed', 'patch_sha256': sha256(patch)}
        write_json(run / 'rollout/tasks' / task_id / 'repetitions' / repetition_id / 'result.json',
                   {**item, 'attempt_id': attempt_id, 'rollout_id': rid, 'exit_status': 'Submitted',
                    'disposition': 'accepted', 'capture_status': 'complete' if n == m else 'partial'})
        items.append(item)
    report = run / 'report.json'
    write_json(report, {'outcomes': outcomes})
    evidence['evaluations']['eval'] = {'outcomes': outcomes, 'report_path': str(report.relative_to(root)),
                                       'report_sha256': sha256(report), 'provenance': {}}
    write_json(run / 'evidence.json', evidence)
    return cfg, run_id, items


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


class CountedFile:
    """Count requested file bytes; this is logical I/O, not physical NFS traffic."""
    def __init__(self, stream, counts):
        self.stream, self.counts = stream, counts
    def __getattr__(self, name):
        return getattr(self.stream, name)
    def __enter__(self):
        self.stream.__enter__()
        return self
    def __exit__(self, *a):
        return self.stream.__exit__(*a)
    def read(self, *a):
        value = self.stream.read(*a)
        self.counts['read_bytes'] += len(value.encode('utf-8') if isinstance(value, str) else value)
        return value
    def write(self, value):
        self.counts['write_bytes'] += len(value.encode('utf-8') if isinstance(value, str) else value)
        return self.stream.write(value)
    def __iter__(self):
        return self
    def __next__(self):
        value = next(self.stream)
        self.counts['read_bytes'] += len(value.encode('utf-8') if isinstance(value, str) else value)
        return value


@pytest.mark.parametrize('sessions', [13, 375])
def test_original_coverage_and_75_by_5_scale(tmp_path, monkeypatch, sessions):
    import hidden.artifacts as artifacts
    import hidden.repair_points as repair
    from difficulty.datasets.online import build_published_index
    from difficulty.datasets.resolved import build_resolved_selection
    expected = [30] * 12 + [29] if sessions == 13 else [32] * 276 + [31] * 99
    observed = [23] * 12 + [21] if sessions == 13 else [n - (7 if i < 12 else 8 if i == 12 else 0) for i, n in enumerate(expected)]
    cfg, run_id, items = cohort(tmp_path, monkeypatch, sessions, expected, observed, 13 if sessions == 13 else 75)
    nas = tmp_path / 'nas/online' / run_id
    originals = {str(p): sha256(p) for p in nas.rglob('*') if p.is_file()}
    unpack_count = 0
    original_bytes = repair.artifact_bytes
    def counted(*a, **kw):
        nonlocal unpack_count
        unpack_count += 1
        return original_bytes(*a, **kw)
    monkeypatch.setattr(repair, 'artifact_bytes', counted)
    before_bytes = tree_bytes(tmp_path)
    before_nas_bytes = tree_bytes(nas)
    nas_io = {'read_bytes': 0, 'write_bytes': 0}
    def count_opener(opener):
        def open_counted(file, *a, **kw):
            stream = opener(file, *a, **kw)
            return CountedFile(stream, nas_io) if isinstance(file, (str, Path)) and Path(file).is_relative_to(nas) else stream
        return open_counted
    stamp = time.monotonic()
    with monkeypatch.context() as io_patch:
        io_patch.setattr(builtins, 'open', count_opener(builtins.open))
        io_patch.setattr(io, 'open', count_opener(io.open))
        summary = finalize_capture_publication(cfg, run_id, items=items)
    elapsed = time.monotonic() - stamp
    finalizer_additional_bytes = tree_bytes(tmp_path) - before_bytes
    assert summary['status'] == 'partial', summary
    assert (summary['expected'], summary['observed'], summary['missing']) == (sum(expected), sum(observed), 92)
    assert summary['published_points'] == sum(observed)
    assert unpack_count == sum(observed)  # One integrity extraction per observed point.
    for path, digest in originals.items():
        assert sha256(path) == digest
    # Exercise standard downstream implementations with their formal selection
    # API, selecting the exact frozen cohort before terminal plan projection.
    ids = [f'rollout-{i:03}' for i in range(sessions)]
    index = build_published_index(cfg, run_id, summary['index_id'], rollout_ids=ids,
                                  publication_id=summary['publication_id'])
    built = read_json(index / 'index.json')
    assert len(built['samples']) == sum(observed)
    result = build_resolved_selection(tmp_path, run_id, summary['index_id'], 'resolved', rollout_ids=ids)
    assert result['rollouts'] == (6 if sessions == 13 else 188)
    assert result['samples'] == sum(observed[i] for i in range(sessions) if (i < 6 if sessions == 13 else i % 2 == 0))
    metrics = {'sessions': sessions, 'expected': sum(expected), 'observed': sum(observed), 'missing': 92,
               'finalization_seconds': elapsed, 'repair_tensor_extractions': unpack_count,
               'source_bytes': sum(Path(p).stat().st_size for p in originals),
               'finalizer_additional_bytes': finalizer_additional_bytes,
               'nas_additional_bytes': tree_bytes(nas) - before_nas_bytes,
               'nas_logical_io': nas_io,
               'additional_bytes_with_index_and_dataset': tree_bytes(tmp_path) - before_bytes,
               'stage_seconds': summary['stage_seconds']}
    print('CAPTURE_SCALE_METRICS=' + json.dumps(metrics, sort_keys=True))
    output_root = os.environ.get('DYCON_CAPTURE_TEST_REPORT_ROOT')
    if output_root:
        report_root = Path(output_root)
        write_json(report_root / f'scale-{sessions}.json', metrics)
        write_json(report_root / f'partial-summary-{sessions}.json', summary)
        publication = tmp_path / 'runs' / run_id / 'enrichment/online-publications' / summary['publication_id']
        write_json(report_root / f'partial-publication-{sessions}.json', read_json(publication / 'publication.json'))
