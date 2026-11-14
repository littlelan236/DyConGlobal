"""Independent-engine topology, committed-prefix migration and publication."""
import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.io import hash_json, read_json, sha256, write_json
from RL.dataset.config import ReplayConfig, Runtime
from RL.dataset.loader import iter_groups
from RL.dataset.materialize import materialize, read_array
from RL.dataset.replay.artifacts import ReplayWriter, check_replay
from RL.dataset.replay.inputs import Prefetch, selected_groups
from RL.dataset.replay.parallel import CommittedReader, partition_groups, run
from RL.dataset.replay.service import replay_logprobs
from test_rl_dataset_replay import ROOT, build, token_source, replay_config


def three_groups(tmp_path):
    """Partition synthetic branch inventory into three complete two-member groups."""
    import json
    from RL.dataset.artifacts import seal
    ds = build(token_source(tmp_path)); source = next(iter(iter_groups(ds, with_tokens=False)))
    groups = []; branches = []
    for i in range(3):
        members = copy.deepcopy(source['members'][i*2:i*2+2])
        gid = source['group']['group_id'] + '-' + str(i)
        for row in members: row['group_id'] = gid
        group = {**source['group'], 'group_id': gid, 'expected_branch_count': 2,
                 'available_branch_count': 2, 'branch_ids': [r['branch_sample_id'] for r in members]}
        groups.append(group); branches.extend(members)
    (ds/'groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
    (ds/'branches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in branches))
    seal(ds, 'rl-dataset', ds.name)
    return ds


def test_independent_workers_require_explicit_tp1_gpu_mapping():
    r = Runtime(gpus=[0, 1, 2], tensor_parallel_size=1, parallel_workers=3)
    assert r.parallel_workers == 3
    assert Runtime(gpus=[0, 1], tensor_parallel_size=2).parallel_workers == 1
    for options in [dict(gpus=[0, 1], parallel_workers=3),
                    dict(gpus=[0, 0], parallel_workers=2),
                    dict(gpus=[0, 1, 2, 3], parallel_workers=2, tensor_parallel_size=2)]:
        with pytest.raises(ValueError): Runtime(**options)


def test_partition_covers_whole_groups_once_and_skips_completed(tmp_path):
    ds = three_groups(tmp_path); value = replay_config(ds, tmp_path)
    value['runtime'].update(gpus=[0, 1], parallel_workers=2)
    config = ReplayConfig.model_validate(value)
    groups = list(iter_groups(ds, with_tokens=False))
    done = {r['branch_sample_id'] for r in groups[0]['members']}
    partition = partition_groups(ds, config, done)
    ids = [g for worker in partition['group_ids'] for g in worker]
    assert len(ids) == len(set(ids)) == len(groups) - 1
    assert groups[0]['group']['group_id'] not in ids
    assert partition == partition_groups(ds, config, done)


def test_prefetch_partition_and_completed_filter_preserve_cache_order(tmp_path):
    ds = three_groups(tmp_path); cache = Path(materialize(ds, tmp_path/'cache')['path'])
    config = ReplayConfig.model_validate(replay_config(ds, tmp_path))
    groups = list(iter_groups(ds)); chosen = groups[-1]
    done = {chosen['members'][0]['branch_sample_id']}
    producer = Prefetch(ds, cache, config, group_ids=[chosen['group']['group_id']], completed_ids=done)
    received = []
    try:
        while True:
            group = producer.get()
            if group is producer.END: break
            if group is not None:
                received.extend(group['members']); producer.release(group)
    finally: producer.close()
    assert [r['branch_sample_id'] for r in received] == [r['branch_sample_id'] for r in chosen['members'][1:]]
    assert received[0]['completion_token_ids'] == chosen['members'][1]['completion_token_ids']


def fake_children(monkeypatch, *, fail_rank=None):
    from RL.dataset.replay import parallel
    events = []
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(events); self.code = 0
            job = read_json(command[-1]); config = ReplayConfig.model_validate(job['replay_config'])
            rank = config.runtime.gpus[0]
            assert config.runtime.tensor_parallel_size == config.runtime.parallel_workers == 1
            assert kwargs.get('start_new_session', False) is False
            events.append(rank)
            writer = ReplayWriter(job['stage'], job['dataset'], job['identity'], 1024,
                                  recover=job['recover'], commit_every_branches=1)
            completed = set(job['completed_branch_ids'])
            for group in selected_groups(job['dataset'], config):
                if group['group']['group_id'] not in job['assigned_group_ids']: continue
                for row in group['members']:
                    if row['branch_sample_id'] in completed or row['branch_sample_id'] in writer.finished: continue
                    # Fake scoring is test data; production uses the existing
                    # native worker and raw trace validation without alteration.
                    from RL.dataset.loader import native_tokens
                    row = dict(row); _, row['completion_token_ids'] = native_tokens(row, audit=False)
                    writer.put(row, {'token_ids': row['completion_token_ids']}, [-.125]*row['completion_count'],
                        policy_sha256=next(iter(job['plan']['policies'])), model_sha256=next(iter(job['plan']['models'])))
            writer.checkpoint('complete')
            write_json(Path(job['stage'])/'execution.json', {
                'logprobs_mode': 'raw_logprobs', 'enable_trace_replay': True,
                'model_inventory_sha256': next(iter(job['plan']['models'].values()))['model']['fingerprint'],
                'replay_implementation': {'test': 'fake-worker'}, 'prefix_cache_effective': True})
            if rank == fail_rank: self.code = 1
        def poll(self): return self.code
        def wait(self, **kwargs): return self.code
        def terminate(self): self.code = -15
        def kill(self): self.code = -9
    monkeypatch.setattr(parallel.subprocess, 'Popen', Child)
    return events


def run_config(tmp_path, monkeypatch, *, resume=False, fail_rank=None):
    ds = three_groups(tmp_path); value = replay_config(ds, tmp_path)
    value['runtime'].update(gpus=[0, 1], parallel_workers=2)
    file = tmp_path/'parallel.yaml'; file.write_text(yaml.safe_dump(value))
    from RL.dataset.replay import service
    monkeypatch.setattr('hidden.backends.vllm.engine.gpu_reservation', lambda *a, **k: nullcontext())
    monkeypatch.setattr(service, 'dispatch', lambda python, job, *args: run(job))
    events = fake_children(monkeypatch, fail_rank=fail_rank)
    return ds, file, events


def test_parallel_formal_dispatch_publishes_one_complete_checked_artifact(tmp_path, monkeypatch):
    ds, file, events = run_config(tmp_path, monkeypatch)
    result = replay_logprobs({**ROOT, 'project_root': str(tmp_path)}, file)
    assert set(events) == {0, 1}
    assert check_replay(result['path'], dataset_path=ds, with_values=True)['branches'] == 6
    runtime = read_json(Path(result['path'])/'models.json')['runtime']
    assert runtime['execution_topology']['parallel_workers'] == 2
    assert len(runtime['parallel_workers']) == 2
    rows = [r for g in iter_groups(ds) for r in g['members']]
    index = [__import__('json').loads(s) for s in (Path(result['path'])/'branches.jsonl').read_text().splitlines()]
    assert {r['branch_sample_id'] for r in index} == {r['branch_sample_id'] for r in rows}
    assert all(r['execution_source'].startswith('workers/') for r in index)


def test_failed_worker_never_publishes_then_exact_recovery_reuses_results(tmp_path, monkeypatch):
    ds, file, _ = run_config(tmp_path, monkeypatch, fail_rank=1)
    cfg = {**ROOT, 'project_root': str(tmp_path)}
    with pytest.raises(RuntimeError, match='worker exited'): replay_logprobs(cfg, file)
    value = yaml.safe_load(file.read_text())
    dest = Path(value['storage']['output_root'])/'RL/replays'/value['replay_id']
    assert not dest.exists()
    fake_children(monkeypatch)
    result = replay_logprobs(cfg, file, recover=True)
    assert check_replay(result['path'], dataset_path=ds, with_values=True)['branches'] == 6


def test_committed_reader_ignores_uncommitted_tail_and_detects_corruption(tmp_path):
    import hashlib,json
    raw = (json.dumps({'branch_sample_id': 'one'})+'\n').encode()
    (tmp_path/'progress-branches.jsonl').write_bytes(raw+b'incomplete')
    write_json(tmp_path/'journal.json', {'committed_bytes': len(raw), 'branch_count': 1,
                                       'index_sha256': hashlib.sha256(raw).hexdigest()})
    reader = CommittedReader(tmp_path)
    assert reader.read() == [{'branch_sample_id': 'one'}]
    assert reader.read() == []
    (tmp_path/'progress-branches.jsonl').write_bytes(b'x'+raw[1:])
    with pytest.raises(ValueError, match='checksum'): CommittedReader(tmp_path).read()


@pytest.mark.parametrize('damage', ['binding', 'policy', 'origin', 'array'])
def test_import_rejects_corruption_without_committing_a_branch(tmp_path, damage):
    ds = build(token_source(tmp_path)); row = next(iter(iter_groups(ds)))['members'][0]
    source = tmp_path/'source'; source.mkdir()
    identity = {'raw_contract_sha256': hash_json(__import__('RL.dataset.replay.validation', fromlist=['raw_contract']).raw_contract())}
    original = ReplayWriter(source, ds, identity, 1024, commit_every_branches=1)
    original.put(row, {'token_ids': row['completion_token_ids']}, [-.25]*3,
                 policy_sha256='policy', model_sha256='model')
    record = copy.deepcopy(original.records[0])
    if damage == 'binding': record['prompt_sha256'] = 'wrong'
    if damage == 'policy': record['policy_sha256'] = 'wrong'
    if damage == 'origin': record['origin'] = 'processed'
    if damage == 'array': (source/record['values_ref']['shard']).write_bytes(b'xxxx'*3)
    target = tmp_path/'target'; target.mkdir(); writer = ReplayWriter(target, ds, identity, 1024)
    with pytest.raises(ValueError):
        writer.import_record(source, record, row, policy_sha256='policy', model_sha256='model', origin='source')
    assert not writer.finished and read_json(target/'journal.json')['branch_count'] == 0


def test_migration_rejects_live_source(tmp_path):
    from RL.dataset.replay.parallel import source_is_stopped
    from RL.dataset.config import DatasetBinding
    write_json(tmp_path/'state.json', {'state': 'running'})
    with pytest.raises(ValueError, match='must be stopped'):
        source_is_stopped(tmp_path, tmp_path, DatasetBinding(id='source', root=str(tmp_path)))


def test_stopped_raw_prefix_migrates_to_new_topology_without_rescoring(tmp_path, monkeypatch):
    from RL.dataset.replay import service
    ds = three_groups(tmp_path); old_value = replay_config(ds, tmp_path)
    old_value['replay_id'] = 'old-raw'
    old_file = tmp_path/'old.yaml'; old_file.write_text(yaml.safe_dump(old_value))
    cfg = {**ROOT, 'project_root': str(tmp_path)}
    monkeypatch.setattr('hidden.backends.vllm.engine.gpu_reservation', lambda *a, **k: nullcontext())
    def old_dispatch(python, file, *args):
        job = read_json(file); stage = Path(job['stage'])
        writer = ReplayWriter(stage, ds, job['identity'], 1024, commit_every_branches=1)
        for row in next(iter(iter_groups(ds)))['members'][:2]:
            writer.put(row, {'token_ids': row['completion_token_ids']}, [-.75]*row['completion_count'],
                policy_sha256=next(iter(job['plan']['policies'])), model_sha256=next(iter(job['plan']['models'])))
        runtime = {'logprobs_mode': 'raw_logprobs', 'enable_trace_replay': True,
                   'model_inventory_sha256': 'unit-model', 'replay_implementation': {'original': 'implementation'}}
        write_json(stage/'runtime-identity.json', runtime); write_json(stage/'execution.json', runtime)
        write_json(stage/'cleanup.json', {'process_group_empty': True})
        raise RuntimeError('test interruption')
    monkeypatch.setattr(service, 'dispatch', old_dispatch)
    with pytest.raises(RuntimeError, match='test interruption'): replay_logprobs(cfg, old_file)
    value = copy.deepcopy(old_value); value['replay_id'] = 'new-parallel'
    value['resume_from'] = {'id': 'old-raw', 'root': old_value['storage']['output_root']}
    value['runtime'].update(gpus=[0, 1], parallel_workers=2)
    file = tmp_path/'new.yaml'; file.write_text(yaml.safe_dump(value))
    assert replay_logprobs(cfg, file, dry=True)['resume_from']['committed_branches'] == 2
    monkeypatch.setattr(service, 'dispatch', lambda python, job, *args: run(job))
    fake_children(monkeypatch)
    result = replay_logprobs(cfg, file)
    path = Path(result['path'])
    assert check_replay(path, dataset_path=ds, with_values=True)['branches'] == 6
    index = [__import__('json').loads(s) for s in (path/'branches.jsonl').read_text().splitlines()]
    imported = [r for r in index if r['execution_source'] == 'resume-source']
    assert len(imported) == 2
    for row in imported: np.testing.assert_array_equal(read_array(path, row['values_ref']), np.array([-.75]*3, dtype=np.float32))
    assert (path/'resume-source.json').exists()
    assert (Path(value['resume_from']['root'])/'RL/replays/.old-raw.staging').exists()
