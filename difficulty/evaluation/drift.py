"""Receipt-bound frozen-predictor drift evaluation; no model fitting or replay.

Reuse publication indexing, submission selection, dataset composition, grouped
tensor readers, the exported predictor loader, and shared regression metrics.
Cross-model use is explicit and recorded here; online serving remains strict.
"""
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from core.config import load_snapshot
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.datasets.compose import compose_datasets
from difficulty.datasets.online import build_published_index
from difficulty.datasets.resolved import build_resolved_selection
from difficulty.evaluation.metrics import paired_regression_metrics
from difficulty.receipts import verify_receipt
from difficulty.storage import difficulty_path, scoped_storage
from hidden.artifacts import iter_arrays_grouped
from runtime.serving.parallel import artifact_contract


def seal(directory, *, sources=None, parents=(), **metadata):
    receipt = {'schema_version': 1, 'state': 'complete', 'created_at': now(),
               'source_files': sources or {}, 'parents': list(parents), **metadata,
               'files': {str(p.relative_to(directory)): sha256(p)
                         for p in sorted(directory.rglob('*'))
                         if p.is_file() and p.name != 'receipt.json'}}
    write_json(directory / 'receipt.json', receipt)


@scoped_storage
@exclusive_artifact('difficulty-evaluation', 'evaluation_id')
def run(cfg, evaluation_id, *, workflow_config):
    root = Path(cfg['project_root']).resolve()
    evaluation_id = identifier(evaluation_id)
    workflow_path = Path(workflow_config).resolve()
    workflow = yaml.safe_load(workflow_path.read_text())
    if set(workflow) != {'schema_version', 'estimator_run', 'allow_hidden_model_shift', 'groups', 'skipped_groups'}:
        raise ValueError('Unexpected drift workflow fields')
    if workflow['schema_version'] != 1 or type(workflow['allow_hidden_model_shift']) is not bool:
        raise ValueError('Invalid drift workflow schema')
    output = difficulty_path(root, 'evaluations', evaluation_id)
    if output.exists():
        raise FileExistsError(output)
    # The original run snapshot owns the exact exported predictor identity.
    estimator_run = identifier(workflow['estimator_run'])
    original = load_snapshot(root, estimator_run)
    estimator = Path(original['parallel']['estimator_ref'])
    if not estimator.is_absolute():
        estimator = owned_path(root, estimator)
    contract = artifact_contract(estimator, original['parallel']['estimator_sha256'])
    if (contract['target'] != 'submission_remaining_reasoning_tokens_log1p'
            or contract['endpoint'] != 'submission' or len(contract['sites']) != 1):
        raise ValueError('Drift evaluation requires the exported single-site submission reasoning Ridge')
    module, name = contract['loader'].split(':')
    predictor = getattr(importlib.import_module(module), name)(estimator.parent, contract)
    source_proofs = {str(workflow_path): sha256(workflow_path),
                     str(root / 'runs' / estimator_run / 'resolved-config.json'):
                     sha256(root / 'runs' / estimator_run / 'resolved-config.json'),
                     str(estimator): sha256(estimator)}
    source_proofs.update({str(owned_path(estimator.parent, f)): digest
                          for f, digest in contract['files'].items()})
    output.mkdir(parents=True)
    write_json(output / 'workflow.json', workflow)
    write_json(output / 'estimator.json', {'run_id': estimator_run, 'path': str(estimator),
                                         'sha256': sha256(estimator), 'contract': contract})
    reports = []
    parents = []

    def progress(phase, **kwargs):
        write_json(output / 'progress.json', {'updated_at': now(), 'phase': phase,
                                            'completed_groups': len(reports), **kwargs})

    for group in workflow['groups']:
        if set(group) != {'name', 'dataset_id', 'sources'}:
            raise ValueError('Invalid group fields')
        group_name = identifier(group['name'])
        dataset_id = identifier(group['dataset_id'])
        progress('building_source_datasets', group=group_name)
        source_datasets = []
        for source in group['sources']:
            if set(source) != {'run_id', 'publication_id', 'publication_receipt_sha256', 'index_id', 'dataset_id'}:
                raise ValueError('Invalid source fields')
            run_id = identifier(source['run_id'])
            publication_id = identifier(source['publication_id'])
            trace = root / 'runs' / run_id / 'enrichment/online-publications' / publication_id
            if sha256(trace / 'receipt.json') != source['publication_receipt_sha256']:
                raise ValueError('Frozen source publication receipt changed')
            frozen = load_snapshot(root, run_id)
            selection = read_json(root / 'runs' / run_id / 'hidden/capture-repairs' /
                                  read_json(trace / 'publication.json')['repair_id'] / 'publication-selection.json')
            selected = selection['selection']
            if selected['run_id'] != run_id:
                raise ValueError('Selection belongs to another run')
            rollouts = [r['rollout_id'] for r in selected['rollouts']]
            if not rollouts or len(set(rollouts)) != len(rollouts):
                raise ValueError('Empty/duplicate canonical submission selection')
            index = build_published_index(cfg, run_id, source['index_id'],
                                          rollout_ids=rollouts, publication_id=publication_id)
            source_dataset = difficulty_path(root, 'datasets', source['dataset_id'])
            if source_dataset.exists():
                verify_receipt(root, source_dataset)
                prior = read_json(source_dataset / 'source-evidence.json')
                if prior['source_index_sha256'] != sha256(index / 'index.json'):
                    raise ValueError('Existing source dataset binds another index')
            else:
                build_resolved_selection(root, run_id, source['index_id'], source['dataset_id'],
                                         rollout_ids=rollouts, eligibility='submitted')
            source_datasets.append(source['dataset_id'])
            print('drift source dataset:', group_name, run_id, flush=True)
        dataset = difficulty_path(root, 'datasets', dataset_id)
        if dataset.exists():
            verify_receipt(root, dataset)
            if read_json(dataset / 'dataset.json')['source_datasets'] != [
                    {'dataset_id': sid, 'receipt_sha256': sha256(difficulty_path(root, 'datasets', sid) / 'receipt.json'),
                     'sample_order_sha256': hash_json([json.loads(line)['sample_id'] for line in
                         (difficulty_path(root, 'datasets', sid) / 'samples.jsonl').read_text().splitlines() if line]),
                     'source_run_ids': [source['run_id']]}
                    for sid, source in zip(source_datasets, group['sources'], strict=True)]:
                raise ValueError('Existing composed dataset has another cohort')
        else:
            compose_datasets(root, dataset_id, source_datasets)
        verify_receipt(root, dataset)
        rollouts = read_json(dataset / 'rollouts.json')['rollouts']
        samples = [json.loads(line) for line in (dataset / 'samples.jsonl').read_text().splitlines() if line]
        lookup = {r['rollout_id']: r for r in rollouts}
        fingerprints = sorted({r['model_fingerprint'] for r in rollouts})
        if len(fingerprints) != 1:
            raise ValueError('Each drift group must have one recorded model fingerprint')
        shifted = fingerprints[0] != contract['model_fingerprint']
        if shifted and not workflow['allow_hidden_model_shift']:
            raise ValueError('Cross-model hidden scoring requires explicit drift authorization')
        if len({s['sample_id'] for s in samples}) != len(samples):
            raise ValueError('Duplicate sample ID')
        for sample in samples:
            row = lookup[sample['rollout_id']]
            if (sample['hidden_root'] != row['hidden_root'] or sample['trajectory_sha256'] != row['trajectory_sha256']
                    or sample['anchor'] != 'think_close' or sample.get('censored')
                    or sample.get('alignment_status') != 'exact'):
                raise ValueError('Invalid exact submission sample binding')
        truth = np.asarray([s['remaining_reasoning_tokens_to_submission'] for s in samples], dtype=float)
        if not np.isfinite(truth).all() or (truth < 0).any():
            raise ValueError('Invalid submission labels')
        scores = np.full(len(samples), np.nan)
        site = 'layer_' + str(contract['sites'][0])
        batch, indices = [], []

        def predict_batch():
            if not batch:
                return
            result = np.asarray(predictor.predict(np.stack(batch)), dtype=float)
            if result.shape != (len(indices),) or not np.isfinite(result).all():
                raise ValueError('Invalid frozen estimator prediction')
            scores[indices] = result
            batch.clear()
            indices.clear()

        progress('reading_hidden_and_predicting', group=group_name, expected_samples=len(samples))
        for index, arrays in iter_arrays_grouped((i, s['hidden_root'], s) for i, s in enumerate(samples)):
            if site not in arrays or arrays[site].shape != (contract['hidden_size'],) or not np.isfinite(arrays[site]).all():
                raise ValueError('Estimator site/width/finite feature mismatch')
            batch.append(np.asarray(arrays[site], dtype=float)[None, :])
            indices.append(index)
            if len(batch) == 256:
                predict_batch()
        predict_batch()
        if not np.isfinite(scores).all():
            raise ValueError('Incomplete predicted sample coverage')
        with np.errstate(over='raise', invalid='raise'):
            native = np.expm1(scores)
        logtruth = np.log1p(truth)
        metrics = {'log1p': paired_regression_metrics(logtruth, scores),
                   'tokens': paired_regression_metrics(truth, native)}
        if any(m['excluded_nonfinite_count'] for m in metrics.values()):
            raise ValueError('Nonfinite scoring pairs')
        summary = {'group': group_name, 'dataset_id': dataset_id, 'run_count': len(group['sources']),
                   'rollouts': len(rollouts), 'tasks': len({r['task_group'] for r in rollouts}),
                   'samples': len(samples), 'hidden_model_fingerprint': fingerprints[0],
                   'estimator_model_fingerprint': contract['model_fingerprint'],
                   'cross_model_evaluation': shifted, 'metrics': metrics,
                   'prediction_clipping': False, 'endpoint': 'observed_submission',
                   'success_labels_used': False, 'negative_native_predictions': int((native < 0).sum())}
        groupout = output / 'groups' / group_name
        groupout.mkdir(parents=True)
        by_rollout = defaultdict(list)
        with (groupout / 'predictions.jsonl').open('w') as stream:
            for i, sample in enumerate(samples):
                by_rollout[sample['rollout_id']].append(i)
                stream.write(json.dumps({'sample_id': sample['sample_id'], 'rollout_id': sample['rollout_id'],
                    'target_tokens': float(truth[i]), 'target_log1p': float(logtruth[i]),
                    'prediction_tokens': float(native[i]), 'prediction_log1p': float(scores[i])}, allow_nan=False) + '\n')
        write_json(groupout / 'by-rollout.json', [{'rollout_id': rid,
            'log1p': paired_regression_metrics(logtruth[ids], scores[ids]),
            'tokens': paired_regression_metrics(truth[ids], native[ids])} for rid, ids in by_rollout.items()])
        write_json(groupout / 'summary.json', summary)
        reports.append(summary)
        parents.append({'path': str(dataset), 'receipt_sha256': sha256(dataset / 'receipt.json')})
        write_json(output / 'summary.json', {'evaluation_id': evaluation_id, 'groups': reports,
            'skipped_groups': workflow['skipped_groups'], 'estimator_sha256': sha256(estimator),
            'primary_metric': 'pooled R2 of log1p remaining reasoning tokens to submission',
            'is_task_holdout': False, 'no_fit': True})
        print('drift result:', json.dumps(summary), flush=True)
    progress('complete')
    seal(output, sources=source_proofs, parents=parents, kind='frozen-estimator-drift-v1',
         evaluation_id=evaluation_id, no_fit=True)
    verify_receipt(root, output)
    return read_json(output / 'summary.json')
