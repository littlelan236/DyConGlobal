"""Submission target transforms retain labels and held-out model semantics."""
import json
from pathlib import Path

import numpy as np
import pytest

from core.io import hash_json, read_json, sha256
from difficulty.models.estimators import inverse_target, transform_target
from difficulty.models.ridge_grid import load_model, predict_model
from difficulty.probe_config import RidgeProbe
from difficulty.training.ridge_probe import report, run_probe
from tests.test_ridge_probe import fixture_project  # noqa: F401


TARGETS = ['submission_remaining_steps_log1p', 'submission_remaining_reasoning_tokens_linear']


def test_new_targets_are_explicit_and_old_defaults_are_preserved():
    assert RidgeProbe().targets == ['submission_remaining_reasoning_tokens_log1p', 'submission_remaining_steps_linear']
    assert RidgeProbe(targets=TARGETS).targets == TARGETS
    for targets in [TARGETS + [TARGETS[0]], ['submission_remaining_tokens_linear']]:
        with pytest.raises(ValueError, match='Unknown or duplicate'):
            RidgeProbe(targets=targets)


@pytest.mark.parametrize('target,expected', [
    (TARGETS[0], np.log1p([0.0, 1.0, 99.0])),
    (TARGETS[1], np.array([0.0, 1.0, 99.0])),
])
def test_submission_transform_zero_and_inverse(target, expected):
    values = np.array([0.0, 1.0, 99.0])
    transformed = transform_target(values, target)
    assert np.allclose(transformed, expected)
    assert np.allclose(inverse_target(transformed, target), values)


def test_new_target_threeway_roundtrip_and_raw_metrics(fixture_project):  # noqa: F811
    pytest.importorskip('matplotlib')
    cfg, settings, dataset = fixture_project
    cfg['difficulty']['ridge_probe'] = {
        **settings.model_dump(), 'targets': TARGETS, 'layers': [0],
        'split_strategy': 'event_random_v1', 'alphas': [1.0, 100.0],
    }
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    result = run_probe(cfg, 'new-targets-threeway')
    output = Path(result['path'])
    assert result['models'] == 4
    receipt = read_json(output / 'receipt.json')
    assert receipt['state'] == 'complete'
    assert all(sha256(output / name) == digest for name, digest in receipt['files'].items())
    assert len(list((output / 'plots').glob('*.png'))) == 6
    samples = [json.loads(line) for line in (dataset / 'samples.jsonl').read_text().splitlines()]
    by_id = {sample['sample_id']: sample for sample in samples}
    attempt = read_json(output / 'attempt.json')
    split = read_json(dataset / 'splits' / attempt['split_id'] / 'split.json')
    test_ids = set(split['sample_ids']['test'])
    fit_indices = [i for i, s in enumerate(samples) if s['sample_id'] not in test_ids]
    test_indices = [i for i, s in enumerate(samples) if s['sample_id'] in test_ids]
    metrics = read_json(output / 'metrics.json')
    predictions = [json.loads(line) for line in (output / 'test-predictions.jsonl').read_text().splitlines()]
    matrix = np.load(Path(read_json(output / 'data-check.json')['cache']) / 'layer-00.f32.npy')
    for target in TARGETS:
        rows = [row for row in predictions if row['target'] == target]
        raw = np.array([by_id[row['sample_id']]['remaining_steps_to_submission' if target == TARGETS[0]
                       else 'remaining_reasoning_tokens_to_submission'] for row in rows], dtype=float)
        expected = np.log1p(raw) if target == TARGETS[0] else raw
        assert np.array_equal(raw, [row['target_raw'] for row in rows])
        assert np.allclose(expected, [row['target_transformed'] for row in rows])
        model = load_model(output / metrics[target]['refit_model'])
        with np.load(output / metrics[target]['refit_model'], allow_pickle=False) as archive:
            assert str(archive['target']) == target
            assert str(archive['fit_scope']) == 'train_validation'
        assert np.allclose(model['feature_mean'], matrix[fit_indices].astype(float).mean(0))
        transformed_prediction = predict_model(model, matrix[test_indices].astype(float))
        assert np.allclose(transformed_prediction, [row['prediction_transformed'] for row in rows])
        raw_prediction = np.expm1(transformed_prediction) if target == TARGETS[0] else transformed_prediction
        assert np.allclose(raw_prediction, [row['prediction_raw'] for row in rows])
        raw_report = report(raw, raw_prediction)
        for key in ['r2', 'mae', 'rmse']:
            assert np.isclose(raw_report[key], metrics[target]['test_raw_units'][key])
