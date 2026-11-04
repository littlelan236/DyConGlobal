from pathlib import Path
import numpy as np
import pytest

from core.io import read_json, sha256, write_json
from difficulty.datasets.submission import feature_contract, public_features, prepare_cache
from difficulty.training.splits import make_manifest_split
from tests.test_ridge_probe import fixture_project


def test_native_contract_5120_and_old_final_norm():
    shape = {f'layer_{i}': [5120] for i in range(32)}
    contract = feature_contract([{'storage_shapes': shape}])
    assert contract['width'] == 5120 and contract['native_sites'][-1] == 31
    sample = {'_native_sites': contract['native_sites'], '_hidden_width': contract['width']}
    native = {f'layer_{i}': np.full(5120, i) for i in range(32)}
    assert public_features(native, sample)[31, 0] == 31
    old = {f'layer_{i}': np.full(4096, i) for i in [*range(31), 32]}
    assert public_features(old, {})[31, 0] == 32
    with pytest.raises(ValueError, match='contract'):
        public_features(old, sample)
    with pytest.raises(ValueError, match='mixes'):
        feature_contract([{'storage_shapes': shape}, {}])
    native['layer_5'] = np.full(5120, np.nan)
    with pytest.raises(ValueError, match='finite'):
        public_features(native, sample)


def test_bound_task_holdout_rejects_unknown_and_keeps_repetitions_together():
    rows, samples, assignments = [], [], []
    for task in range(6):
        key, group = f'k{task}', f'skillsbench::t{task}'
        assignments.append({'task_key': key, 'instance_id': f't{task}',
                            'partition': 'train' if task < 4 else 'test'})
        for rep in range(2):
            rid = f'r{task}-{rep}'
            rows.append({'benchmark': 'skillsbench', 'revision': 'rev', 'task_key': key,
                         'instance_id': f't{task}', 'task_group': group,
                         'rollout_id': rid, 'trajectory_sha256': rid})
            samples.append({'rollout_id': rid, 'sample_id': rid, 'task_group': group})
    manifest = {'task_assignments': assignments}
    split = make_manifest_split(rows, samples, manifest, 42, .25)
    assert split['task_counts'] == {'train': 3, 'validation': 1, 'test': 2}
    assert split == make_manifest_split(list(reversed(rows)), samples, manifest, 42, .25) or split['sample_ids'] == make_manifest_split(list(reversed(rows)), samples, manifest, 42, .25)['sample_ids']
    assert set(split['sample_ids']['test']) == {'r4-0', 'r4-1', 'r5-0', 'r5-1'}
    train_only = make_manifest_split(rows[:8], samples[:8], manifest, 42, .25)
    assert set(train_only['sample_ids']) == {'train', 'validation'}
    assert train_only['external_test_tasks'] == ['t4', 't5']
    with pytest.raises(ValueError, match='absent'):
        make_manifest_split(rows, samples, {'task_assignments': assignments[:-1]}, 42, .25)


def test_feature_cache_roundtrip_5120(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / 'hidden.npz'
    np.savez(path, hidden=np.arange(2 * 32 * 5120, dtype=np.float32).reshape(2, 32, 5120), step=np.array([0, 1]))
    samples = [{'sample_id': str(i), 'hidden_root': str(tmp_path), 'hidden_path': path.name,
                'hidden_sha256': sha256(path), 'array_row': i, 'agent_step': i,
                'storage_shapes': {f'layer_{k}': [5120] for k in range(32)}} for i in range(2)]
    cfg = {'project_root': str(tmp_path)}
    settings = SimpleNamespace(dataset_receipt_sha256='a' * 64)
    cache = prepare_cache(cfg, settings, samples)
    assert read_json(cache / 'cache.json')['feature_contract']['width'] == 5120
    matrix = np.load(cache / 'layer-31.f32.npy')
    assert matrix.shape == (2, 5120) and matrix[1, 0] == 63 * 5120
    assert prepare_cache(cfg, settings, samples) == cache


def test_real_ridge_5120_grid_refits_all_train_cohort(fixture_project):
    import json
    from difficulty.training.ridge_probe import run_probe
    from core.io import hash_json
    cfg, settings, dataset = fixture_project
    rows = read_json(dataset / 'rollouts.json')['rollouts']
    samples = [json.loads(line) for line in (dataset / 'samples.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(11)
    for row in rows:
        path = Path(next(iter(row['source_files'])))
        np.savez(path, hidden=rng.normal(size=(3, 32, 5120)).astype(np.float32), step=np.arange(3))
        row['source_files'] = {str(path): sha256(path)}
        for sample in samples:
            if sample['rollout_id'] == row['rollout_id']:
                sample['hidden_sha256'] = sha256(path)
                sample['storage_shapes'] = {f'layer_{i}': [5120] for i in range(32)}
    write_json(dataset / 'rollouts.json', {'rollouts': rows})
    (dataset / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    receipt = read_json(dataset / 'receipt.json')
    receipt['files'] = {name: sha256(dataset / name) for name in receipt['files']}
    write_json(dataset / 'receipt.json', receipt)
    from difficulty.training.splits import create_split
    manifest_path = dataset.parent / 'fit-partitions.json'
    write_json(manifest_path, {'task_assignments': [
        {'task_key': row['task_key'], 'instance_id': row['instance_id'], 'partition': 'train'}
        for row in rows]})
    create_split(cfg['project_root'], 'fixture', 'fit-split',
                 strategy='task_manifest_v1', task_partitions=manifest_path,
                 validation_fraction=.25)
    settings = settings.model_copy(update={'dataset_receipt_sha256': sha256(dataset / 'receipt.json'),
                                           'refit_train_validation': True, 'split_id': 'fit-split',
                                           'split_strategy': 'task_manifest_v1',
                                           'validation_tasks': 2})
    cfg['difficulty']['ridge_probe'] = settings.model_dump()
    cfg.pop('config_hash', None)
    cfg['config_hash'] = hash_json(cfg)
    result = run_probe(cfg, 'fit5120')
    output = Path(result['path'])
    attempt = read_json(output / 'attempt.json')
    assert attempt['model_contract']['input_width'] == 5120
    for metric in read_json(output / 'metrics.json').values():
        assert metric['refit_train_validation']['count'] == 24
        assert 'test' not in metric
        metadata = read_json((output / metric['refit_model']).with_suffix('.json'))['metadata']
        assert metadata['input_width'] == 5120
        assert metadata['native_site'] in (0, 31)
