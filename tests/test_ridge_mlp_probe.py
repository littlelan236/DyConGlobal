"""Frozen Ridge provenance and shared MLP lifecycle regression tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.io import hash_json, read_json, sha256, write_json
from difficulty.models.estimators import FrozenRidgePlusFFNEstimator
from difficulty.models.mlp_checkpoint import (RIDGE_MLP_ARCHITECTURE, RIDGE_MLP_KIND,
    load_checkpoint, save_checkpoint, validate_checkpoint)
from difficulty.probe_config import MLP_DEFAULT_TARGETS as MLP_TARGETS, RidgeMLPProbe
from difficulty.training.ridge_source import load_sources, read_publication
from tests.test_mlp_probe import _metadata
from tests.test_ridge_probe import fixture_project  # noqa: F401


def _publication(root, target, split, *, scope='train', width=3, site=2):
    from difficulty.models.ridge_grid import save_model
    directory = root / 'runtime/parallel-artifacts' / target
    directory.mkdir(parents=True)
    write_json(directory / 'source-split.json', split)
    arrays = {'coef': np.full(width, 0.05), 'intercept': np.asarray(0.5),
              'feature_mean': np.zeros(width), 'feature_scale': np.ones(width)}
    save_model(directory / 'model.npz', arrays, {
        'target': target, 'target_transform': 'log1p', 'layer': site, 'input_width': width,
        'model_kind': 'ridge', 'model_fingerprint': 'model', 'fit_scope': scope,
        'split_sha256': sha256(directory / 'source-split.json')})
    write_json(directory / 'manifest.json', {
        'schema_version': 1, 'loader': 'runtime.serving.ridge_provider:load',
        'target': target, 'model_fingerprint': 'model', 'endpoint': 'submission',
        'sites': [site], 'hidden_size': width, 'model_file': 'model.npz',
        'provenance': {'fit_scope': scope},
        'files': {name: sha256(directory / name) for name in ['model.npz', 'source-split.json']}})
    return directory / 'manifest.json', arrays


def test_frozen_residual_initialization_scheduler_and_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    target = MLP_TARGETS[0]
    split = {'sample_ids': {'train': ['a'], 'validation': ['b'], 'test': ['c']}}
    path, ridge = _publication(tmp_path, target, split)
    _, _, source, _ = read_publication(path, target=target, site=2, width=3, fingerprint='model')
    source['current_split_training_overlap'] = {'train': 1, 'validation': 0, 'test': 0}
    x = np.random.default_rng(4).normal(size=(20, 3)).astype('float32')
    model = FrozenRidgePlusFFNEstimator(ridge, hidden_dim=4, epochs=4, batch_size=4,
        scheduler={'enabled': True, 'kind': 'reduce_on_plateau', 'factor': 0.5,
                   'patience': 0, 'threshold': 0.0, 'min_lr': 1e-6})
    before = {k: v.copy() for k, v in model.ridge_model.items()}
    model.input_dim = 3
    model._module = model._make_module()
    assert np.array_equal(model.predict(x), model.predict_base(x))
    # A controlled plateau proves scheduler integration and epoch LR semantics.
    model._validation_loss = lambda *args: 1.0
    model.fit(x[:12], np.ones(12), validation_data=(x[12:], np.ones(8)))
    assert [row['learning_rate'] for row in model.history_] == [1e-3, 1e-3, 5e-4, 2.5e-4]
    assert model.best_epoch_ == 1
    assert all(np.array_equal(before[k], model.ridge_model[k]) for k in before)
    metadata = _metadata(model)
    metadata.update(model_kind=RIDGE_MLP_KIND, architecture_id=RIDGE_MLP_ARCHITECTURE,
                    model_fingerprint='model', ridge_source=source)
    saved = tmp_path / 'best.npz'
    save_checkpoint(saved, model, metadata)
    loaded = load_checkpoint(saved)
    assert np.allclose(model.predict(x), loaded.predict(x), atol=1e-7)
    assert validate_checkpoint(saved)[0]['ridge_source'] == source
    # The combined NPZ can predict without the original publication directory.
    path.rename(path.with_name('old-manifest.json'))
    assert np.allclose(loaded.predict(x), model.predict(x), atol=1e-7)


def test_publication_rejects_target_mismatch_tamper_and_heldout_overlap(tmp_path):
    from difficulty.training.ridge_source import snapshot_publication, verify_source_snapshot
    split = {'sample_ids': {'train': ['a'], 'validation': ['b'], 'test': ['c']}}
    paths = {t: _publication(tmp_path, t, split, width=4096, site=3)[0] for t in MLP_TARGETS}
    settings = RidgeMLPProbe(public_site=3, ridge_publications={k: str(v) for k, v in paths.items()})
    models, sources = load_sources(tmp_path, settings, 'model', split)
    target = MLP_TARGETS[0]
    copied = tmp_path / 'attempt'
    snapshot_publication(sources[target], copied / f'models/{target}/ridge-publication')
    verify_source_snapshot(copied, target, sources[target], models[target], split)
    with pytest.raises(ValueError, match='contract mismatch'):
        read_publication(paths[target], target=MLP_TARGETS[1], site=3, width=4096, fingerprint='model')
    swapped = {'sample_ids': {'train': ['b'], 'validation': ['a'], 'test': ['c']}}
    with pytest.raises(ValueError, match='held-out'):
        load_sources(tmp_path, settings, 'model', swapped)
    payload = paths[target].parent / 'model.npz'
    payload.write_bytes(payload.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='file changed'):
        load_sources(tmp_path, settings, 'model', split)


def test_ridge_mlp_requires_both_publications():
    with pytest.raises(ValueError, match='publication manifest'):
        RidgeMLPProbe(ridge_publications={MLP_TARGETS[0]: 'manifest.json'})


@pytest.mark.parametrize('kind,command', [('ffn_mlp', 'difficulty-mlp'),
                                        (RIDGE_MLP_KIND, 'difficulty-ridge-mlp')])
def test_official_cli_full_fit_verify_and_reload(fixture_project, monkeypatch, capsys, kind, command):
    pytest.importorskip('torch')
    from core.cli import main
    from difficulty.artifacts import verify
    from difficulty.training.mlp_probe import code_evidence
    from difficulty.training.splits import create_split
    cfg, _, dataset = fixture_project
    root = Path(cfg['project_root'])
    source_root = Path(__file__).resolve().parents[1]
    for name in code_evidence(source_root):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / name).read_bytes())
    # Fixture already binds canonical trajectory/hidden; mark its eligibility explicitly.
    rollouts = read_json(dataset / 'rollouts.json')
    for row in rollouts['rollouts']:
        row.update(resolved=True, evaluation_status='completed')
    write_json(dataset / 'rollouts.json', rollouts)
    receipt = read_json(dataset / 'receipt.json')
    receipt['files']['rollouts.json'] = sha256(dataset / 'rollouts.json')
    write_json(dataset / 'receipt.json', receipt)
    create_split(root, 'fixture', 'event', strategy='event_random_v1')
    split = read_json(dataset / 'splits/event/split.json')
    references = {t: str(_publication(root, t, split, width=4096, site=3)[0]) for t in MLP_TARGETS}
    training = {'mlp_probe': {
        'dataset_id': 'fixture', 'dataset_receipt_sha256': sha256(dataset / 'receipt.json'),
        'expected_tasks': 8, 'expected_rollouts': 8, 'expected_samples': 24, 'public_site': 3,
        'hidden_dim': 4, 'epochs': 2, 'patience': 2, 'torch_threads': 1,
        'scheduler': {'enabled': True, 'patience': 0}},
        'ridge_mlp_probe': {'ridge_publications': references}}
    (root / 'difficulty/training/config.yaml').write_text(yaml.safe_dump(training))
    # No root mlp_probe settings: training must use the training-owned config.
    cfg['model'] = {'fingerprint': 'model'}
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    monkeypatch.setattr('core.cli.resolve', lambda *a: cfg)
    before = {t: sha256(Path(p).parent / 'model.npz') for t, p in references.items()}
    main([command, 'fit', '--mode', 'fit', '--dataset-id', 'fixture'])
    output = root / 'difficulty/checkpoints/fit'
    attempt = read_json(output / 'attempt.json')
    assert attempt['split_strategy'] == 'event_random_v1'
    assert attempt['model_contract']['model_kind'] == kind
    assert attempt['sample_counts'] == {'train': 14, 'validation': 5, 'test': 5}
    assert verify(root, 'difficulty-attempt', 'fit')['verified']
    for target in MLP_TARGETS:
        directory = output / f'models/{target}'
        model = load_checkpoint(directory / 'best.npz')
        history = [json.loads(s) for s in (output / f'history/{target}.jsonl').read_text().splitlines()]
        assert len(history) == 2
        assert all(f'{role}_r2' in history[0] and f'{role}_loss' in history[0]
                   for role in ['train', 'validation', 'test'])
        assert 'learning_rate' in history[0]
        assert read_json(directory / 'config.json')['effective_settings'] == attempt['settings']
        assert sha256(Path(references[target]).parent / 'model.npz') == before[target]
        cache = Path(read_json(output / 'data-check.json')['cache'])
        x = np.load(cache / 'layer-03.f32.npy', allow_pickle=False)
        samples = [json.loads(s) for s in (dataset / 'samples.jsonl').read_text().splitlines()]
        rows = [json.loads(s) for s in (output / f'predictions/test/{target}.jsonl').read_text().splitlines()]
        indices = [{s['sample_id']: i for i, s in enumerate(samples)}[row['sample_id']] for row in rows]
        assert np.allclose(model.predict(x[indices]), [r['prediction_transformed'] for r in rows], atol=1e-6)
    main([command, 'dry', '--mode', 'dry-run'])
    assert verify(root, 'difficulty-attempt', 'dry')['verified']
    main([command, 'check', '--mode', 'check-data'])
    assert verify(root, 'difficulty-attempt', 'check')['verified']
    with pytest.raises(FileExistsError):
        main([command, 'fit'])
