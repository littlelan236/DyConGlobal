"""Official single-site, two-target MLP attempts over frozen dataset-owned splits."""

import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from core.config import assert_frozen
from core.io import artifact_dir, hash_json, identifier, now, read_json, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.datasets.submission import load_selection, prepare_cache
from difficulty.evaluation.metrics import regression_metrics
from difficulty.models.estimators import FFNMLEstimator, FrozenRidgePlusFFNEstimator, inverse_target, resolve_device, transform_target
from difficulty.models.mlp_checkpoint import (ARCHITECTURE_ID, FEATURE_SCHEMA, MODEL_KIND,
    RIDGE_MLP_ARCHITECTURE, RIDGE_MLP_KIND, FLEX_ARCHITECTURE, FLEX_RIDGE_ARCHITECTURE, save_checkpoint)
from difficulty.probe_config import LABELS, MLPProbe, RidgeMLPProbe, target_transform
from difficulty.receipts import verify_receipt
from difficulty.training.artifacts import finish_attempt
from difficulty.training.splits import create_split, load_split


def code_evidence(root, *, frozen_training=False):
    names = ['core/config.py', 'core/cli.py', 'core/io.py', 'core/locking.py',
             'difficulty/probe_config.py', 'difficulty/receipts.py', 'difficulty/artifacts.py',
             'difficulty/datasets/submission.py', 'difficulty/training/split.py',
             'difficulty/training/splits.py', 'difficulty/training/artifacts.py',
             'difficulty/training/mlp_probe.py', 'difficulty/training/mlp_visualize.py',
             'difficulty/training/config.yaml',
             'difficulty/training/ridge_source.py', 'difficulty/models/ridge_grid.py',
             'runtime/serving/parallel.py', 'runtime/serving/ridge_provider.py',
             'difficulty/models/estimators.py',
             'difficulty/models/mlp_checkpoint.py', 'difficulty/evaluation/metrics.py']
    names += ['difficulty/training/config.py', 'difficulty/training/entrypoint.py', 'difficulty/training/visualization.py']
    if frozen_training:
        names.remove('difficulty/training/config.yaml')
    return {name: sha256(root / name) for name in names if (root / name).exists()}


def _report(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    nonfinite = int((~np.isfinite(prediction)).sum())
    if nonfinite or not np.isfinite(y).all():
        return {'count': len(y), 'mae': None, 'rmse': None, 'r2': None,
                'nonfinite_predictions': nonfinite, 'reason': 'nonfinite_inverse_transform'}
    with np.errstate(over='ignore', invalid='ignore'):
        result = regression_metrics(y, prediction)
    for key in ('mae', 'rmse', 'r2'):
        if result[key] is not None and not np.isfinite(result[key]):
            result[key] = None
            result['reason'] = 'metric_overflow'
    if result['r2'] is None:
        result.setdefault('reason', 'zero_target_variance_or_insufficient_count')
    result['negative_predictions'] = int((prediction < 0).sum())
    return result


def _loss_plot(path, history, target):
    from difficulty.training.mlp_visualize import plot_history

    plot_history(history, path, target)


@exclusive_artifact('difficulty-mlp', 'attempt_id')
def run_probe(cfg, attempt_id, *, mode='fit', dataset_id=None, split_id=None, kind=MODEL_KIND,
              settings_override=None, training_config_path=None, training_config_raw=None, visualization=None):
    assert_frozen(cfg)
    if mode not in {'dry-run', 'check-data', 'fit'}:
        raise ValueError('Unknown MLP mode')
    if kind not in {MODEL_KIND, RIDGE_MLP_KIND}:
        raise ValueError('Unknown MLP estimator kind')
    ridge_plus_mlp = kind == RIDGE_MLP_KIND
    architecture_id = RIDGE_MLP_ARCHITECTURE if ridge_plus_mlp else ARCHITECTURE_ID
    root = Path(cfg['project_root'])
    supplied_settings = settings_override is not None
    if supplied_settings:
        training_config_path = Path(training_config_path)
        training_config_raw = dict(training_config_raw)
        settings = settings_override
        training_config_sha256 = sha256(training_config_path)
    else:
        training_config_path = Path(training_config_path or root / 'difficulty/training/config.yaml')
        if not training_config_path.is_file():
            raise FileNotFoundError(training_config_path)
        training_config_raw = yaml.safe_load(training_config_path.read_text(encoding='utf-8')) or {}
        if training_config_raw.get('schema_version') == 2:
            from difficulty.training.config import TrainingConfig, resolve_settings
            import copy
            adapted = copy.deepcopy(training_config_raw)
            adapted['model']['kind'] = kind
            if dataset_id is not None:
                adapted['data']['dataset_id'] = dataset_id
                adapted['data']['expected_receipt_sha256'] = None
            if split_id is not None:
                adapted['data']['split'] = {'id': split_id}
            settings_override = resolve_settings(root, TrainingConfig.model_validate(adapted))
            raw_settings = settings_override.model_dump()
        else:
            raw_settings = dict(training_config_raw.get('mlp_probe') or {})
            if not raw_settings:
                raise ValueError('Training config must define mlp_probe')
            if ridge_plus_mlp:
                overrides = training_config_raw.get('ridge_mlp_probe')
                if not overrides:
                    raise ValueError('Training config must define ridge_mlp_probe')
                raw_settings.update(overrides)
        training_config_sha256 = sha256(training_config_path)
        if dataset_id is not None:
            dataset_id = identifier(dataset_id)
            source = root / 'difficulty/datasets' / dataset_id
            rows = read_json(source / 'rollouts.json')['rollouts']
            raw_settings.update(dataset_id=dataset_id, dataset_receipt_sha256=sha256(source / 'receipt.json'),
                                expected_tasks=len({(r['repo'], r['instance_id']) for r in rows}),
                                expected_rollouts=len(rows),
                                expected_samples=sum(bool(s.strip()) for s in (source / 'samples.jsonl').read_text().splitlines()))
        if split_id is not None:
            raw_settings['split_id'] = identifier(split_id)
        settings = (RidgeMLPProbe if ridge_plus_mlp else MLPProbe).model_validate(raw_settings)
    if settings.split_id is None:
        split_identity = {
            'dataset_id': settings.dataset_id,
            'dataset_receipt_sha256': settings.dataset_receipt_sha256,
            'strategy': settings.split_strategy,
            'seed': settings.split_seed,
            'validation_fraction': settings.validation_fraction,
            'test_fraction': settings.test_fraction,
        }
        split_id = 'auto-mlp-' + hash_json(split_identity)[:16]
        split_path = root / 'difficulty/datasets' / settings.dataset_id / 'splits' / split_id
        if not split_path.exists():
            create_split(root, settings.dataset_id, split_id, strategy=settings.split_strategy,
                         seed=settings.split_seed, validation_fraction=settings.validation_fraction,
                         test_fraction=settings.test_fraction)
        settings = settings.model_copy(update={'split_id': split_id})
    output = artifact_dir(cfg, 'difficulty', attempt_id, 'checkpoints')
    if output.exists():
        raise FileExistsError(f'Attempt ID already exists (including partial): {output}')
    before_code = code_evidence(root, frozen_training=supplied_settings)
    rollouts, samples, evidence = load_selection(cfg, settings)
    if not all(r.get('resolved') is True and r.get('evaluation_status') == 'completed' for r in rollouts):
        raise ValueError('MLP requires an explicitly resolved-only training selection')
    fingerprints = {r['model_fingerprint'] for r in rollouts}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        raise ValueError('MLP requires a single explicit hidden model fingerprint')
    source = root / 'difficulty/datasets' / settings.dataset_id
    split_path = source / 'splits' / settings.split_id
    split, _ = load_split(root, settings.dataset_id, settings.split_id, settings.dataset_receipt_sha256, samples)
    split_receipt_sha = sha256(split_path / 'receipt.json')
    if settings.split_receipt_sha256 and settings.split_receipt_sha256 != split_receipt_sha:
        raise ValueError('Frozen MLP split receipt SHA mismatch')
    if mode == 'fit' and set(split['sample_ids']) != {'train', 'validation', 'test'}:
        raise ValueError('MLP held-out fit requires train, validation and test partitions')
    ridge_models, ridge_sources = {}, {}
    if ridge_plus_mlp:
        from difficulty.training.ridge_source import load_sources
        ridge_models, ridge_sources = load_sources(root, settings, next(iter(fingerprints)), split)
    ids = {s['sample_id']: i for i, s in enumerate(samples)}
    indices = {role: np.asarray([ids[sid] for sid in assigned], dtype=np.int64)
               for role, assigned in split['sample_ids'].items()}
    tasks = {role: {samples[i]['task_group'] for i in idx} for role, idx in indices.items()}
    task_overlap = {a + '/' + b: len(tasks[a] & tasks[b]) for a in tasks for b in tasks if a < b}
    spec = {'public_sites': [settings.public_site], 'width': 4096,
            'feature_transform': 'single-site-vector', 'dtype': 'float32'}
    architecture = {'input_width': 4096, 'hidden_dim': settings.hidden_dim, 'dropout': settings.dropout,
                    'layernorm_eps': 1e-5, 'activation': 'gelu', 'dtype': 'float32'}
    if settings.hidden_dims is not None:
        architecture_id = FLEX_RIDGE_ARCHITECTURE if ridge_plus_mlp else FLEX_ARCHITECTURE
        architecture.update(hidden_dims=settings.hidden_dims, activation=settings.activation,
                            normalization=settings.normalization)
    environment = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}
    actual_device = None
    if mode == 'fit':
        import torch
        torch.set_num_threads(settings.torch_threads)
        actual_device = str(resolve_device(settings.device))
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        torch.backends.cudnn.benchmark = False
        environment.update(torch=torch.__version__, cuda=torch.version.cuda, device=actual_device,
                           torch_threads=torch.get_num_threads(),
                           device_name=torch.cuda.get_device_name(actual_device) if actual_device.startswith('cuda') else 'CPU')
    attempt = {
        'schema_version': 3, 'artifact_layout_version': 2, 'mode': mode, 'created_at': now(),
        'attempt_id': attempt_id, 'dataset_id': settings.dataset_id, 'split_id': settings.split_id,
        'dataset_receipt_sha256': settings.dataset_receipt_sha256, 'split_receipt_sha256': split_receipt_sha,
        'split_sha256': sha256(split_path / 'split.json'), 'sample_order_sha256': evidence['sample_order_sha256'],
        'train_sample_ids_sha256': hash_json(split['sample_ids']['train']),
        'model_fingerprint': next(iter(fingerprints)), 'fit_scope': 'train',
        'training_config_path': str(training_config_path),
        'training_config_sha256': training_config_sha256,
        'training_config': training_config_raw,
        'training_config_frozen': supplied_settings,
        'model_contract': {'model_kind': kind, 'architecture_id': architecture_id,
                           'feature_schema': FEATURE_SCHEMA, 'input_spec': spec, 'architecture': architecture,
                           'targets': settings.targets, 'target_transforms': {
                               target: (target_transform(target))
                               for target in settings.targets}},
        'config': cfg, 'settings': settings.model_dump(), 'input': evidence, 'code': before_code,
        'environment': environment, 'actual_device': actual_device,
        'split_strategy': split['strategy'], 'task_counts': split['task_counts'],
        'sample_counts': split['sample_counts'], 'task_overlap': task_overlap,
        'selection_metric': 'validation_mse', 'selection_split': 'validation',
    }
    if ridge_plus_mlp:
        attempt['ridge_sources'] = ridge_sources
        attempt['model_contract']['combination'] = 'ridge(x) + mlp_residual(x), in the configured target space'
        attempt['model_contract']['ridge_frozen'] = True
    output.mkdir(parents=True)
    write_json(output / 'training-config.json', training_config_raw)
    write_json(output / 'attempt.json', attempt)
    write_json(output / 'experiment.json', attempt)
    completed = []

    def progress(state, **fields):
        write_json(output / 'progress.json', {'state': state, 'mode': mode, 'updated_at': now(),
                   'completed_targets': list(completed), **fields})

    def finish():
        if not supplied_settings and sha256(training_config_path) != training_config_sha256:
            raise ValueError('MLP training config changed during attempt')
        load_selection(cfg, settings)
        load_split(root, settings.dataset_id, settings.split_id, settings.dataset_receipt_sha256, samples)
        verify_receipt(root, source, expected_sha256=settings.dataset_receipt_sha256)
        if ridge_plus_mlp:
            from difficulty.training.ridge_source import load_sources
            _, current_sources = load_sources(root, settings, next(iter(fingerprints)), split)
            if current_sources != ridge_sources:
                raise ValueError('Ridge publication changed during attempt')
        if visualization is not None:
            visualization.close_writers()
        progress('complete')
        finish_attempt(output, attempt, root, before_code, parents=[
            {'path': str(source.relative_to(root)), 'receipt_sha256': settings.dataset_receipt_sha256},
            {'path': str(split_path.relative_to(root)), 'receipt_sha256': split_receipt_sha},
        ])

    progress('running')
    try:
        if mode == 'dry-run':
            finish()
            return {'path': str(output), 'mode': mode, 'task_counts': split['task_counts'],
                    'sample_counts': split['sample_counts'], 'task_overlap': task_overlap}
        cache = prepare_cache(cfg, settings, samples)
        matrix = np.load(cache / f'layer-{settings.public_site:02d}.f32.npy', mmap_mode='r', allow_pickle=False)
        if matrix.shape != (len(samples), 4096) or not np.isfinite(matrix).all():
            raise ValueError('MLP feature cache shape/finite mismatch')
        x = np.asarray(matrix, dtype=np.float32)
        labels = {target: np.asarray([s[LABELS[target]] for s in samples], dtype=float) for target in settings.targets}
        y = {target: transform_target(value, target) for target, value in labels.items()}
        write_json(output / 'data-check.json', {**evidence, 'cache': str(cache),
                   'cache_manifest_sha256': sha256(cache / 'cache.json'),
                   'verified_cache_sites': list(range(32)), 'selected_input_spec': spec,
                   'selected_shape': list(x.shape), 'target_counts': {t: len(v) for t, v in y.items()},
                   'task_overlap': task_overlap, 'checked_at': now()})
        if mode == 'check-data':
            finish()
            return {'path': str(output), 'mode': mode, 'samples': len(samples), 'input_spec': spec}
        if visualization is not None:
            visualization.start(output)
        trained, selected = {}, {}
        train, validation, test = indices['train'], indices['validation'], indices['test']
        for target in settings.targets:
            history_path = output / f'history/{target}.jsonl'
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open('w', encoding='utf-8') as stream:
                def record(row):
                    row = {'target': target, 'train_optimization_loss': row['train_loss'], **row}
                    # Regression has no classification accuracy. Record MSE/R2/MAE/RMSE
                    # for all partitions; test remains diagnostic only.
                    for role, idx in (('train', train), ('validation', validation), ('test', test)):
                        prediction = model.predict(x[idx])
                        quality = regression_metrics(y[target][idx], prediction)
                        row[f'{role}_loss'] = quality['rmse'] ** 2
                        row[f'{role}_r2'] = quality['r2']
                        row[f'{role}_mae'] = quality['mae']
                        row[f'{role}_rmse'] = quality['rmse']
                    if visualization is not None:
                        visualization.write(target, row['epoch'], {
                            'loss/train': row['train_loss'], 'loss/val': row['eval_loss'],
                            'loss/train_optimization': row['train_optimization_loss'],
                            'optimizer/learning_rate': row['learning_rate'],
                            'selection/best_val_loss': row['best_eval_loss'],
                            'selection/best_epoch': row['best_epoch']})
                    def display(value):
                        return 'null' if value is None else f'{value:.5g}'
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    progress('running', target=target, **{k: v for k, v in row.items() if k != 'target'})
                    print(f"mlp {target} epoch={row['epoch']}/{settings.epochs} "
                          f"train_loss={row['train_loss']:.7g} val_loss={row['eval_loss']:.7g} "
                          f"test_loss={row['test_loss']:.7g} train_r2={display(row['train_r2'])} "
                          f"val_r2={display(row['validation_r2'])} test_r2={display(row['test_r2'])} "
                          f"best_epoch={row['best_epoch']} best_eval_loss={row['best_eval_loss']:.7g} "
                          f"device={row['device']} seconds={row['epoch_seconds']:.3f}", flush=True)

                kwargs = dict(hidden_dim=settings.hidden_dim, dropout=settings.dropout, epochs=settings.epochs,
                    learning_rate=settings.learning_rate, weight_decay=settings.weight_decay, seed=settings.seed,
                    device=actual_device, batch_size=settings.batch_size, patience=settings.patience,
                    min_delta=settings.min_delta, scheduler=settings.scheduler.model_dump(),
                    hidden_dims=settings.hidden_dims, activation=settings.activation,
                    normalization=settings.normalization, optimizer=settings.optimizer)
                model = (FrozenRidgePlusFFNEstimator(ridge_models[target], **kwargs) if ridge_plus_mlp
                         else FFNMLEstimator(**kwargs))
                model.fit(x[train], y[target][train], validation_data=(x[validation], y[target][validation]), epoch_callback=record)
            metadata = {key: attempt[key] for key in ('attempt_id', 'dataset_id', 'split_id', 'dataset_receipt_sha256',
                        'split_receipt_sha256', 'split_sha256', 'sample_order_sha256', 'train_sample_ids_sha256', 'model_fingerprint')}
            metadata.update(schema_version=1, model_kind=kind, architecture_id=architecture_id,
                            feature_schema=FEATURE_SCHEMA, input_spec=spec, architecture=architecture,
                            target=target, target_label=LABELS[target], target_transform=(
                                target_transform(target)), endpoint='submission',
                            normalization=settings.normalization, fit_scope='train', seed=settings.seed,
                            best_epoch=model.best_epoch_, epochs_trained=model.epochs_trained_,
                            scheduler=settings.scheduler.model_dump(),
                            training_config_sha256=training_config_sha256)
            if ridge_plus_mlp:
                from difficulty.training.ridge_source import snapshot_publication
                metadata['ridge_source'] = ridge_sources[target]
                snapshot_publication(ridge_sources[target], output / f'models/{target}/ridge-publication')
            relative = f'models/{target}/best.npz'
            digest = save_checkpoint(output / relative, model, metadata)
            write_json(output / relative.replace('/best.npz', '/config.json'), {
                'schema_version': 1, 'attempt_id': attempt_id, 'target': target,
                'training_config_sha256': training_config_sha256,
                'training_config': training_config_raw,
                'effective_settings': settings.model_dump(),
                'model_metadata': metadata,
            })
            selected[target] = {'model': relative, 'model_sha256': digest, 'best_epoch': model.best_epoch_,
                                'best_eval_loss': model.best_eval_loss_, 'epochs_trained': model.epochs_trained_,
                                'device': model.device_, 'finite_gradients': model.finite_gradients_}
            _loss_plot(output / f'plots/{target}-loss.png', history_path, target)
            trained[target] = model
            completed.append(target)
        write_json(output / 'selected.json', {'selection_metric': 'validation_mse',
                   'selection_split': 'validation', 'fit_scope': 'train', 'targets': selected})
        # Final evaluation uses the validation-best weights restored by fit.
        metrics = {}
        for target, model in trained.items():
            metrics[target] = {'model_sha256': selected[target]['model_sha256'], 'partitions': {}}
            mean = float(y[target][train].mean())
            for role, idx in indices.items():
                prediction = model.predict(x[idx])
                with np.errstate(over='ignore', invalid='ignore'):
                    raw_prediction = inverse_target(prediction, target)
                    baseline_raw = inverse_target(np.full(len(idx), mean), target)
                metrics[target]['partitions'][role] = {
                    'transformed': _report(y[target][idx], prediction), 'raw_units': _report(labels[target][idx], raw_prediction),
                    'train_mean_baseline_transformed': _report(y[target][idx], np.full(len(idx), mean)),
                    'train_mean_baseline_raw_units': _report(labels[target][idx], baseline_raw),
                }
                if ridge_plus_mlp:
                    base = model.predict_base(x[idx])
                    with np.errstate(over='ignore', invalid='ignore'):
                        base_raw = inverse_target(base, target)
                    metrics[target]['partitions'][role]['ridge_baseline_transformed'] = _report(y[target][idx], base)
                    metrics[target]['partitions'][role]['ridge_baseline_raw_units'] = _report(labels[target][idx], base_raw)
                path = output / f'predictions/{role}/{target}.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('w', encoding='utf-8') as stream:
                    for j, sample_idx in enumerate(idx):
                        sample = samples[sample_idx]
                        row = {key: sample[key] for key in ('sample_id', 'rollout_id', 'task_group')}
                        finite = bool(np.isfinite(raw_prediction[j]))
                        row.update(target=target, partition=role, checkpoint_sha256=selected[target]['model_sha256'],
                                   target_raw=float(labels[target][sample_idx]), target_transformed=float(y[target][sample_idx]),
                                   prediction_transformed=float(prediction[j]),
                                   prediction_raw=float(raw_prediction[j]) if finite else None,
                                   raw_prediction_reason=None if finite else 'inverse_transform_overflow')
                        if ridge_plus_mlp:
                            row.update(ridge_prediction_transformed=float(base[j]),
                                       residual_prediction_transformed=float(prediction[j] - base[j]))
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
        write_json(output / 'metrics.json', metrics)
        report = [f'# {"Ridge+MLP" if ridge_plus_mlp else "MLP"} submission remainder', '', f'Dataset: {settings.dataset_id}; split: {settings.split_id}.',
                  f'Strategy: {split["strategy"]}; tasks: {split["task_counts"]}; samples: {split["sample_counts"]}.',
                  f'Task overlaps: {task_overlap}. Site {settings.public_site}; device {actual_device}.',
                  'Each target uses its validation-MSE best epoch, trained on train only; no train+validation refit.',
                  'Per-epoch test curves are diagnostic only; test does not select checkpoints or stop training.',
                  'Metrics describe the resolved trajectory subset.', '',
                  '| Target | Best epoch | Validation MSE | Test R2 (log1p) | Test R2 (raw) |',
                  '|---|---:|---:|---:|---:|']
        if ridge_plus_mlp:
            report[2:2] = ['Frozen published Ridge + zero-initialized FFN residual in the configured target space.',
                'Ridge publication snapshots and hashes are saved per target; Ridge is never refitted.',
                'Published Ridge fitted samples do not overlap current validation/test samples.', '']
        for target in settings.targets:
            report.append(f"| {target} | {selected[target]['best_epoch']} | {selected[target]['best_eval_loss']} | "
                          f"{metrics[target]['partitions']['test']['transformed']['r2']} | "
                          f"{metrics[target]['partitions']['test']['raw_units']['r2']} |")
        (output / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        finish()
        return {'path': str(output), 'mode': mode, 'models': len(trained), 'device': actual_device,
                'task_counts': split['task_counts'], 'sample_counts': split['sample_counts'], 'selected': selected}
    except BaseException as exc:
        progress('failed', error_type=type(exc).__name__, error=str(exc))
        raise
