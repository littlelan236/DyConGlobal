"""Versioned numeric MLP checkpoints; structural verification never imports torch."""

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from core.io import identifier, owned_path, read_json, sha256, write_json
from difficulty.probe_config import MLP_ALLOWED_TARGETS, target_transform

MODEL_KIND = 'ffn_mlp'
ARCHITECTURE_ID = 'layernorm-ffn-2block-v1'
RIDGE_MLP_KIND = 'ridge_plus_ffn'
RIDGE_MLP_ARCHITECTURE = 'frozen-ridge-layernorm-ffn-2block-v1'
FEATURE_SCHEMA = 'public-hidden-last-token-v1'
IDENTITY_KEYS = ('attempt_id', 'dataset_id', 'split_id', 'dataset_receipt_sha256',
                 'split_receipt_sha256', 'split_sha256', 'sample_order_sha256',
                 'train_sample_ids_sha256', 'model_fingerprint')


def parameter_shapes(width, hidden):
    return {'0.weight': [width], '0.bias': [width],
            '1.weight': [hidden, width], '1.bias': [hidden],
            '4.weight': [hidden, hidden], '4.bias': [hidden],
            '7.weight': [1, hidden], '7.bias': [1]}


def _validate(metadata, state, expected=None):
    architecture_id = {MODEL_KIND: ARCHITECTURE_ID, RIDGE_MLP_KIND: RIDGE_MLP_ARCHITECTURE}.get(
        metadata.get('model_kind'))
    if (metadata.get('schema_version') != 1 or architecture_id is None
            or metadata.get('architecture_id') != architecture_id
            or metadata.get('feature_schema') != FEATURE_SCHEMA
            or metadata.get('target') not in MLP_ALLOWED_TARGETS
            or metadata.get('target_transform') not in {
                target_transform(metadata.get('target', '')),
                'linear' if metadata.get('target', '').endswith('_fraction') else target_transform(metadata.get('target', '')),
            }
            or metadata.get('fit_scope') != 'train'
            or metadata.get('normalization') != 'layernorm'
            or metadata.get('endpoint') != 'submission'):
        raise ValueError('Invalid MLP checkpoint model/target contract')
    for key in IDENTITY_KEYS:
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f'Missing MLP checkpoint identity: {key}')
        if key.endswith('sha256') and not re.fullmatch('[0-9a-f]{64}', metadata[key]):
            raise ValueError(f'Invalid MLP checkpoint SHA: {key}')
    for key in ('attempt_id', 'dataset_id', 'split_id'):
        identifier(metadata[key])
    architecture = metadata['architecture']
    width, hidden = architecture['input_width'], architecture['hidden_dim']
    if (type(width) is not int or type(hidden) is not int or width <= 0 or hidden <= 0
            or architecture.get('dtype') != 'float32'
            or architecture.get('layernorm_eps') != 1e-5 or architecture.get('activation') != 'gelu'
            or not 0 <= architecture['dropout'] < 1):
        raise ValueError('Invalid MLP architecture')
    spec = metadata['input_spec']
    sites = spec.get('public_sites')
    if (spec.get('width') != width or spec.get('dtype') != 'float32'
            or spec.get('feature_transform') != 'single-site-vector'
            or not isinstance(sites, list) or len(sites) != 1
            or type(sites[0]) is not int or not 0 <= sites[0] < 32):
        raise ValueError('Invalid MLP input spec')
    if (type(metadata.get('best_epoch')) is not int or metadata['best_epoch'] < 1
            or type(metadata.get('epochs_trained')) is not int
            or metadata['epochs_trained'] < metadata['best_epoch']):
        raise ValueError('Invalid MLP best epoch')
    shapes = parameter_shapes(width, hidden)
    if metadata['model_kind'] == RIDGE_MLP_KIND:
        source = metadata.get('ridge_source', {})
        if (source.get('target') != metadata['target'] or source.get('public_site') != sites[0]
                or source.get('width') != width or source.get('model_fingerprint') != metadata['model_fingerprint']
                or source.get('fit_scope') not in {'train', 'train_validation'}
                or any(not re.fullmatch('[0-9a-f]{64}', source.get(k, '')) for k in
                       ('manifest_sha256', 'checkpoint_sha256', 'trained_sample_ids_sha256'))):
            raise ValueError('Invalid Ridge+MLP base source metadata')
        shapes.update({'ridge.coef': [width], 'ridge.intercept': [],
                       'ridge.feature_mean': [width], 'ridge.feature_scale': [width]})
    if set(state) != set(shapes):
        raise ValueError('MLP checkpoint parameter names mismatch')
    for name, shape in shapes.items():
        value = state[name]
        dtype = 'float64' if name.startswith('ridge.') else 'float32'
        if list(value.shape) != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
            raise ValueError(f'MLP checkpoint parameter shape/dtype/finite mismatch: {name}')
    if metadata['model_kind'] == RIDGE_MLP_KIND and (state['ridge.feature_scale'] <= 0).any():
        raise ValueError('Invalid Ridge+MLP feature scale')
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f'MLP checkpoint identity disagrees with attempt: {key}')


def save_checkpoint(path, estimator, metadata):
    path = Path(path)
    state = {key: value.detach().cpu().numpy().copy()
             for key, value in estimator._module.state_dict().items()}
    if metadata['model_kind'] == RIDGE_MLP_KIND:
        state.update({'ridge.' + k: v.copy() for k, v in estimator.ridge_model.items()})
    _validate(metadata, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.write-', suffix='.npz', delete=False) as f:
            temporary = Path(f.name)
            np.savez(f, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
                     **{'state.' + key: value for key, value in state.items()})
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(path.with_suffix('.json'), {
        'schema_version': 1, 'checkpoint': path.name, 'checkpoint_sha256': sha256(path),
        'metadata': metadata, 'parameter_shapes': {k: list(v.shape) for k, v in state.items()},
    })
    return sha256(path)


def validate_checkpoint(path, expected=None):
    path = Path(path)
    sidecar = read_json(path.with_suffix('.json'))
    if (sidecar.get('schema_version') != 1 or sidecar.get('checkpoint') != path.name
            or sidecar.get('checkpoint_sha256') != sha256(path)):
        raise ValueError('MLP checkpoint sidecar SHA/identity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_json'].item())
        state = {key.removeprefix('state.'): archive[key].copy() for key in archive.files
                 if key.startswith('state.')}
        if set(archive.files) != {'metadata_json', *('state.' + k for k in state)}:
            raise ValueError('Unexpected MLP checkpoint arrays')
    _validate(metadata, state, expected)
    if (sidecar['metadata'] != metadata
            or sidecar['parameter_shapes'] != {k: list(v.shape) for k, v in state.items()}):
        raise ValueError('MLP checkpoint sidecar metadata mismatch')
    return metadata, state


def load_checkpoint(path, device='cpu'):
    metadata, state = validate_checkpoint(path)
    import torch
    from difficulty.models.estimators import FFNMLEstimator, FrozenRidgePlusFFNEstimator, _mlp, resolve_device

    architecture = metadata['architecture']
    kwargs = dict(hidden_dim=architecture['hidden_dim'], dropout=architecture['dropout'],
                  seed=metadata['seed'], device=device)
    if metadata['model_kind'] == RIDGE_MLP_KIND:
        ridge = {k.removeprefix('ridge.'): state.pop(k) for k in list(state) if k.startswith('ridge.')}
        estimator = FrozenRidgePlusFFNEstimator(ridge, **kwargs)
    else:
        estimator = FFNMLEstimator(**kwargs)
    estimator.input_dim = architecture['input_width']
    estimator.device_ = str(resolve_device(device))
    estimator._module = _mlp(estimator.input_dim, estimator.hidden_dim, estimator.dropout, estimator.seed)
    estimator._module.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    estimator._module.to(estimator.device_).eval()
    estimator.best_epoch_ = metadata['best_epoch']
    estimator.epochs_trained_ = metadata['epochs_trained']
    return estimator


def verify_attempt_models(directory, attempt, receipt):
    """Validate required MLP outputs and cross-check selection against epoch history."""
    contract = attempt['model_contract']
    selected = read_json(directory / 'selected.json')
    targets = contract['targets']
    if (not targets or len(targets) != len(set(targets)) or not set(targets) <= set(MLP_ALLOWED_TARGETS)
            or selected.get('selection_metric') != 'validation_mse'
            or selected.get('selection_split') != 'validation'
            or set(selected['targets']) != set(targets)):
        raise ValueError('Invalid MLP checkpoint selection')
    progress = read_json(directory / 'progress.json')
    if progress.get('state') != 'complete' or set(progress['completed_targets']) != set(targets):
        raise ValueError('Incomplete MLP training progress')
    metrics = read_json(directory / 'metrics.json')
    ridge_plus_mlp = contract['model_kind'] == RIDGE_MLP_KIND
    for target in targets:
        winner = selected['targets'][target]
        relative = f'models/{target}/best.npz'
        required = {relative, relative.removesuffix('.npz') + '.json', f'history/{target}.jsonl',
                    'selected.json', 'metrics.json', 'progress.json', 'report.md',
                    f'plots/{target}-loss.png'}
        if attempt.get('training_config_sha256'):
            required.add(relative.replace('/best.npz', '/config.json'))
        if ridge_plus_mlp:
            required.add(f'models/{target}/ridge-publication/manifest.json')
        required.update(f'predictions/{role}/{target}.jsonl' for role in ('train', 'validation', 'test'))
        if not required <= set(receipt['files']) or winner['model'] != relative:
            raise ValueError('Missing required MLP model/evaluation artifact')
        path = owned_path(directory, relative)
        if sha256(path) != winner['model_sha256']:
            raise ValueError('Selected MLP checkpoint SHA changed')
        if attempt.get('training_config_sha256'):
            snapshot = read_json(directory / relative.replace('/best.npz', '/config.json'))
            if (snapshot.get('attempt_id') != attempt['attempt_id'] or snapshot.get('target') != target
                    or snapshot.get('training_config_sha256') != attempt.get('training_config_sha256')
                    or snapshot.get('effective_settings') != attempt.get('settings')
                    or snapshot.get('training_config') != attempt.get('training_config')):
                raise ValueError('MLP training config snapshot disagrees with attempt')
        expected = {key: attempt[key] for key in IDENTITY_KEYS}
        expected.update(model_kind=contract['model_kind'], architecture_id=contract['architecture_id'], feature_schema=FEATURE_SCHEMA,
                        target=target, target_transform=target_transform(target), fit_scope='train',
                        architecture=contract['architecture'], input_spec=contract['input_spec'],
                        best_epoch=winner['best_epoch'], epochs_trained=winner['epochs_trained'])
        if ridge_plus_mlp:
            expected['ridge_source'] = attempt['ridge_sources'][target]
        metadata, state = validate_checkpoint(path, expected)
        if attempt.get('training_config_sha256') and snapshot.get('model_metadata') != metadata:
            raise ValueError('MLP model metadata snapshot disagrees with checkpoint')
        if ridge_plus_mlp:
            from difficulty.training.ridge_source import verify_source_snapshot
            root = directory.parents[2]
            split = read_json(root / 'difficulty/datasets' / attempt['dataset_id'] /
                              'splits' / attempt['split_id'] / 'split.json')
            verify_source_snapshot(directory, target, attempt['ridge_sources'][target],
                {k.removeprefix('ridge.'): v for k, v in state.items() if k.startswith('ridge.')}, split)
            manifest = read_json(directory / f'models/{target}/ridge-publication/manifest.json')
            if not {f'models/{target}/ridge-publication/{name}' for name in manifest['files']} <= set(receipt['files']):
                raise ValueError('Unregistered Ridge publication snapshot files')
        history = [json.loads(line) for line in (directory / f'history/{target}.jsonl').read_text().splitlines()]
        if (not history or [row['epoch'] for row in history] != list(range(1, len(history) + 1))
                or len(history) != winner['epochs_trained']
                or not np.isfinite([row[key] for row in history for key in ('train_loss', 'eval_loss')]).all()):
            raise ValueError('Invalid MLP epoch history')
        best = min(history, key=lambda row: (row['eval_loss'], row['epoch']))
        if (winner['best_epoch'] != best['epoch'] or winner['best_eval_loss'] != best['eval_loss']
                or metrics[target]['model_sha256'] != winner['model_sha256']):
            raise ValueError('MLP checkpoint selection disagrees with history/metrics')
