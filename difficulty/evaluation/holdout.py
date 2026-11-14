from difficulty.storage import difficulty_path
"""Formal dataset task-holdout workflow over the shared Ridge training service."""
from pathlib import Path
from contextlib import contextmanager
import os

import yaml

from difficulty.storage import scoped_storage
from core.config import assert_frozen
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.evaluation.holdout_config import read_config
from difficulty.training.config import TrainingConfig, resolve_settings


def child_id(evaluation_id, kind):
    return identifier(evaluation_id[:100] + '-' + kind + '-' + hash_json([evaluation_id, kind])[:16])


@scoped_storage
@exclusive_artifact('difficulty-evaluation', 'evaluation_id')
def run(cfg, evaluation_id, *, workflow_config, reuse_attempt_id=None):
    """Python and CLI share this service; every invocation requires a fresh ID."""
    assert_frozen(cfg)
    identifier(evaluation_id)
    root = Path(cfg['project_root'])
    config_path = (root / workflow_config).resolve()
    config = read_config(config_path)
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = str(config.model.ridge.blas_threads)
    # Set thread controls before importing NumPy/estimators; threadpoolctl also
    # covers embedded Python callers that have already imported BLAS.
    from threadpoolctl import threadpool_limits
    from difficulty.datasets.submission import load_selection
    from difficulty.probe_config import LABELS, RidgeProbe
    from difficulty.training.split import make_task_holdout_split
    from difficulty.training.splits import create_split, load_split
    from difficulty.artifacts import verify
    from difficulty.training.entrypoint import run_config
    from difficulty.evaluation.holdout_results import build_cohort, evaluate, draw_test_trajectories

    output = difficulty_path(root, 'evaluations', evaluation_id)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    workflow = {'schema_version': 1, 'evaluation_id': evaluation_id, 'mode': config.run.mode,
                'status': 'running', 'started_at': now(), 'stages': []}

    @contextmanager
    def stage(name):
        item = {'name': name, 'status': 'running', 'started_at': now()}
        workflow['stages'].append(item)
        write_json(output / 'workflow.json', workflow)
        try:
            yield
        except Exception as error:
            item.update(status='failed', completed_at=now(), error=f'{type(error).__name__}: {error}')
            raise
        else:
            item.update(status='complete', completed_at=now())
        finally:
            write_json(output / 'workflow.json', workflow)

    try:
        from difficulty.training.ridge_probe import code_evidence
        code = code_evidence(root)
        with stage('preflight'):
            dataset_id = config.data.dataset_id
            dataset_dir = difficulty_path(root, 'datasets', dataset_id)
            digest = sha256(dataset_dir / 'receipt.json')
            if config.data.expected_receipt_sha256 and digest != config.data.expected_receipt_sha256:
                raise ValueError('Expected dataset receipt SHA mismatch')
            # Reuse the standalone resolver for dataset counts, target and model validation.
            provisional = TrainingConfig.model_validate({'data': {'dataset_id': dataset_id,
                'expected_receipt_sha256': digest, 'targets': config.data.targets},
                'model': {'kind': 'ridge', 'ridge': {'layers': [0], 'alphas': config.model.ridge.alphas}}})
            settings = resolve_settings(root, provisional)
            rollouts, samples, evidence = load_selection(cfg, settings)
            contract = evidence['feature_contract']
            legal = contract['public_sites']
            layers = legal if config.model.ridge.layers == 'auto' else config.model.ridge.layers
            if not set(layers) <= set(legal):
                raise ValueError('Requested layers are outside dataset feature contract')
            # Existing RidgeProbe validates the fully expanded search grid.
            RidgeProbe.model_validate({**settings.model_dump(), 'layers': layers, 'alphas': config.model.ridge.alphas})
            import math
            for target in config.data.targets:
                if any(type(s.get(LABELS[target])) not in (int, float) or not math.isfinite(s[LABELS[target]])
                       for s in samples):
                    raise ValueError(f'Dataset lacks finite labels for {target}')
            manifest_path = ((root / config.outer_split.task_partitions).resolve()
                             if config.outer_split.task_partitions else None)
            manifest = read_json(manifest_path) if manifest_path else None
            expected_split = make_task_holdout_split(rollouts, samples, config.outer_split.test_fraction,
                config.outer_split.seed, config.inner_split.validation_fraction, config.inner_split.seed, manifest)

        with stage('split'):
            attempt_id = identifier(reuse_attempt_id) if reuse_attempt_id else child_id(evaluation_id, 'ridge')
            attempt_dir = difficulty_path(root, 'checkpoints', attempt_id)
            if reuse_attempt_id:
                if config.run.mode != 'fit':
                    raise ValueError('reuse-attempt-id requires fit mode')
                verify(root, 'difficulty-attempt', attempt_id)
                previous = read_json(attempt_dir / 'attempt.json')
                if previous['mode'] != 'fit' or previous['dataset_id'] != dataset_id:
                    raise ValueError('Reuse attempt mode/dataset mismatch')
                split_id = previous['split_id']
                split, _ = load_split(root, dataset_id, split_id, digest, samples)
                if any(split.get(k) != v for k, v in expected_split.items()):
                    raise ValueError('Reuse attempt split/config mismatch')
            else:
                split_id = child_id(evaluation_id, 'split')
                create_split(root, dataset_id, split_id, strategy='task_holdout_event_v1',
                    seed=config.outer_split.seed, test_fraction=config.outer_split.test_fraction,
                    validation_fraction=config.inner_split.validation_fraction,
                    inner_seed=config.inner_split.seed, task_partitions=manifest_path)
                split, _ = load_split(root, dataset_id, split_id, digest, samples)
            split_dir = dataset_dir / 'splits' / split_id
            workflow.update(dataset_id=dataset_id, split_id=split_id, attempt_id=attempt_id)
            ridge = config.model.ridge.model_dump()
            ridge.update(layers=layers, plots_enabled=config.visualization.enabled)
            training = TrainingConfig.model_validate({'schema_version': 2,
                'run': {'attempt_id': attempt_id, 'mode': config.run.mode},
                'data': {'dataset_id': dataset_id, 'expected_receipt_sha256': digest,
                         'split': {'id': split_id}, 'targets': config.data.targets},
                'model': {'kind': 'ridge', 'ridge': ridge}, 'tensorboard': {'enabled': False}})
            training_raw = training.model_dump()
            snapshot = output / 'training-config.yaml'
            snapshot.write_text(yaml.safe_dump(training_raw, sort_keys=False), encoding='utf-8')
            resolved = {'workflow': config.model_dump(), 'dataset_receipt_sha256': digest,
                        'feature_contract': contract, 'layers': layers, 'alphas': config.model.ridge.alphas,
                        'split_id': split_id, 'split_receipt_sha256': sha256(split_dir / 'receipt.json'),
                        'training_config': training_raw,
                        'workflow_config_sha256': sha256(config_path), 'code': code}
            write_json(output / 'resolved-config.json', resolved)
            cohort = build_cohort(rollouts, samples, split)
            write_json(output / 'cohort.json', cohort)
            if reuse_attempt_id:
                previous_training = previous.get('training_config')
                if previous_training != training_raw:
                    raise ValueError('Reuse attempt full effective training configuration mismatch')
                if previous['input']['feature_contract'] != contract:
                    raise ValueError('Reuse attempt feature contract mismatch')

        with stage('reuse_attempt' if reuse_attempt_id else config.run.mode):
            if not reuse_attempt_id:
                with threadpool_limits(limits=config.model.ridge.blas_threads):
                    result = run_config(cfg, training, raw=training_raw, path=snapshot)
            else:
                result = {'path': str(attempt_dir), 'status': 'complete'}

        summary = {'evaluation_id': evaluation_id, 'dataset_id': dataset_id, 'split_id': split_id,
                   'attempt_id': attempt_id, 'mode': config.run.mode,
                   'test_rollouts': len(cohort['rollouts']), 'test_events': len(cohort['sample_ids']),
                   'visualization': 'skipped_by_mode' if config.run.mode != 'fit' else
                                    'enabled' if config.visualization.enabled else 'skipped_by_config',
                   'report': str(output / 'report.md')}
        complete = True
        if config.run.mode == 'fit':
            with stage('evaluate'):
                metrics, coverage, indexed = evaluate(output, attempt_dir, config, cohort, samples, split)
                complete = coverage['complete'] and result.get('status', 'complete') == 'complete'
                chosen = read_json(attempt_dir / 'selected.json')['targets']
                summary['targets'] = {t: {'layer': chosen[t]['layer'],
                    'native_site': contract['native_sites'][chosen[t]['layer']], 'width': contract['width'],
                    'alpha': chosen[t]['alpha'], **{k: metrics[t]['test_raw_units']['pooled'][k]
                         for k in ('count', 'r2', 'nmse_energy', 'nmse_variance')}} for t in config.data.targets}
                summary['successful_events_by_target'] = {t: a['successful'] for t, a in coverage['targets'].items()}
            if config.visualization.enabled:
                with stage('visualize'):
                    draw_test_trajectories(output, config, cohort, samples, indexed, metrics)
                    summary['html'] = str(output / 'index.html')

        summary['status'] = 'complete' if complete else 'partial'
        lines = ['# Dataset Ridge task holdout', '', f"Mode: {config.run.mode}; status: {summary['status']}.",
                 'Inner validation events may share tasks and trajectories with inner train. '
                 'Outer test tasks, trajectories and samples are isolated. Cross-task generalization uses outer test only.',
                 '', 'Primary metrics: raw units; finite paired events only. NMSE_energy=SSE/sum(y²); '
                 'NMSE_variance=SSE/sum((y-mean(y))²); R²=1-NMSE_variance. '
                 'Undefined metrics are null with reasons. Negative predictions are preserved.', '',
                 '```json', __import__('json').dumps(summary, indent=2, ensure_ascii=False), '```']
        (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        workflow.update(status=summary['status'], completed_at=now(), summary=summary)
        write_json(output / 'workflow.json', workflow)
        if complete:
            with stage('verify'):
                verify(root, 'difficulty-attempt', attempt_id)
                verify(root, 'difficulty-split', split_id, dataset_id=dataset_id)
                if {name: sha256(root / name) for name in code} != code:
                    raise ValueError('Implementation changed during workflow')
            parents = [dataset_dir, split_dir, attempt_dir]
            write_json(output / 'receipt.json', {'schema_version': 1, 'kind': 'ridge-task-holdout-v1',
                'state': 'complete', 'mode': config.run.mode, 'evaluation_id': evaluation_id,
                'dataset_id': dataset_id, 'split_id': split_id, 'attempt_id': attempt_id,
                'completed_at': now(), 'code': code,
                'parents': [{'path': str(p.relative_to(root)) if p.is_relative_to(root) else str(p), 'receipt_sha256': sha256(p / 'receipt.json')} for p in parents],
                'files': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            verify(root, 'difficulty-evaluation', evaluation_id)
        return summary
    except Exception as error:
        # Successful attempt receipts remain sealed if later plots fail.
        (output / 'receipt.json').unlink(missing_ok=True)
        workflow.update(status='failed', completed_at=now(), error=f'{type(error).__name__}: {error}')
        write_json(output / 'workflow.json', workflow)
        if not (output / 'report.md').exists():
            (output / 'report.md').write_text('# Failed Ridge holdout workflow\n\n' + workflow['error'] +
                '\n\nSee workflow.json for stages; preserved coverage/plot artifacts describe partial results.\n', encoding='utf-8')
        raise
