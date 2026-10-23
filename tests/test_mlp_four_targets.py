"""Four-target MLP configuration, transform and official CLI regressions."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from core.io import hash_json, read_json, sha256, write_json
from difficulty.probe_config import LABELS, MLP_DEFAULT_TARGETS, MLP_TARGETS, MLPProbe
from difficulty.models.estimators import FFNMLEstimator, inverse_target, transform_target
from difficulty.models.mlp_checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint
from tests.test_mlp_probe import _metadata
from tests.test_ridge_probe import fixture_project


def test_four_targets_keep_original_defaults_and_accept_nonempty_subsets():
    assert tuple(MLPProbe().targets) == MLP_DEFAULT_TARGETS
    for targets in [[target] for target in MLP_TARGETS] + [list(MLP_TARGETS)]:
        assert MLPProbe(targets=targets).targets == targets
    for targets in [[], [MLP_TARGETS[0]] * 2, ['unknown'], [123]]:
        with pytest.raises(ValueError):
            MLPProbe(targets=targets)


@pytest.mark.parametrize('target', MLP_TARGETS)
def test_each_target_transform_and_checkpoint_roundtrip(tmp_path, target):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    x = np.random.default_rng(17).normal(size=(16, 3)).astype('float32')
    raw = np.arange(16, dtype=float)
    y = transform_target(raw, target)
    assert np.allclose(inverse_target(y, target), raw)
    assert np.allclose(y, np.log1p(raw) if target.endswith('_log1p') else raw)
    model = FFNMLEstimator(hidden_dim=4, epochs=2, batch_size=4, patience=2)
    model.fit(x[:12], y[:12], validation_data=(x[12:], y[12:]))
    metadata = _metadata(model, target)
    metadata.update(target_label=LABELS[target],
                    target_transform='log1p' if target.endswith('_log1p') else 'linear')
    path = tmp_path / 'best.npz'
    save_checkpoint(path, model, metadata)
    assert validate_checkpoint(path)[0]['target_transform'] == metadata['target_transform']
    assert np.allclose(model.predict(x), load_checkpoint(path).predict(x), atol=1e-6)
    invalid = dict(metadata, target_transform='linear' if target.endswith('_log1p') else 'log1p')
    with pytest.raises(ValueError, match='target contract'):
        save_checkpoint(tmp_path / 'invalid.npz', model, invalid)


def test_four_target_official_cli_fit_verify_predictions(fixture_project, monkeypatch):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from core.cli import main
    from difficulty.artifacts import verify
    from difficulty.training.mlp_probe import code_evidence
    from difficulty.training.splits import create_split
    cfg, _, dataset = fixture_project
    root = Path(cfg['project_root']); source = Path(__file__).resolve().parents[1]
    for name in code_evidence(source):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((source / name).read_bytes())
    rollouts = read_json(dataset / 'rollouts.json')
    for row in rollouts['rollouts']:
        row.update(resolved=True, evaluation_status='completed')
    write_json(dataset / 'rollouts.json', rollouts)
    receipt = read_json(dataset / 'receipt.json')
    receipt['files']['rollouts.json'] = sha256(dataset / 'rollouts.json')
    write_json(dataset / 'receipt.json', receipt)
    create_split(root, 'fixture', 'event-four', strategy='event_random_v1')
    training = {'mlp_probe': {
        'dataset_id': 'fixture', 'dataset_receipt_sha256': sha256(dataset / 'receipt.json'),
        'split_id': 'event-four', 'expected_tasks': 8, 'expected_rollouts': 8,
        'expected_samples': 24, 'public_site': 3, 'targets': list(MLP_TARGETS),
        'hidden_dim': 4, 'dropout': 0.1, 'epochs': 2, 'patience': 2, 'torch_threads': 1}}
    (root / 'difficulty/training/config.yaml').write_text(yaml.safe_dump(training))
    cfg['model'] = {'fingerprint': 'model'}
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    monkeypatch.setattr('core.cli.resolve', lambda *a: cfg)
    main(['difficulty-mlp', 'fit-four', '--mode', 'fit', '--dataset-id', 'fixture'])
    output = root / 'difficulty/checkpoints/fit-four'
    attempt = read_json(output / 'attempt.json'); metrics = read_json(output / 'metrics.json')
    assert set(metrics) == set(MLP_TARGETS)
    assert verify(root, 'difficulty-attempt', 'fit-four')['verified']
    samples = [json.loads(line) for line in (dataset / 'samples.jsonl').read_text().splitlines()]
    by_id = {sample['sample_id']: sample for sample in samples}
    for target in MLP_TARGETS:
        expected = 'log1p' if target.endswith('_log1p') else 'linear'
        assert attempt['model_contract']['target_transforms'][target] == expected
        path = output / f'models/{target}/best.npz'
        assert validate_checkpoint(path)[0]['target_transform'] == expected
        assert read_json(path.parent / 'config.json')['effective_settings'] == attempt['settings']
        history = [json.loads(line) for line in (output / f'history/{target}.jsonl').read_text().splitlines()]
        assert len(history) == 2
        assert all(f'{role}_loss' in history[0] for role in ['train','validation','test'])
        for role in ['train','validation','test']:
            rows = [json.loads(line) for line in (output / f'predictions/{role}/{target}.jsonl').read_text().splitlines()]
            assert len(rows) == attempt['sample_counts'][role]
            assert np.allclose([row['target_raw'] for row in rows],
                               [by_id[row['sample_id']][LABELS[target]] for row in rows])
            assert np.allclose([row['target_transformed'] for row in rows],
                               transform_target([row['target_raw'] for row in rows], target))
            assert np.allclose([row['prediction_raw'] for row in rows],
                               inverse_target([row['prediction_transformed'] for row in rows], target))
