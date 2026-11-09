"""Evaluate and plot the authoritative predictions without further inference."""
import base64
import csv
import html
import json
from collections import defaultdict

import numpy as np

from core.io import hash_json, owned_path, read_json, sha256, write_json
from difficulty.evaluation.metrics import paired_regression_metrics, macro_metrics
from difficulty.models.estimators import transform_target
from difficulty.probe_config import LABELS


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def build_cohort(rollouts, samples, split):
    expected = set(split['outer_test_sample_ids'])
    rows = [s for s in samples if s['sample_id'] in expected]
    tasks = {g for g, role in split['outer']['task_assignments'].items() if role == 'test'}
    selected = [r for r in rollouts if any(s['rollout_id'] == r['rollout_id'] for s in rows)]
    if {s['task_group'] for s in rows} != tasks:
        raise ValueError('Outer test cohort does not cover assigned tasks')
    by_rollout = defaultdict(list)
    for row in rows:
        by_rollout[row['rollout_id']].append(row['sample_id'])
    return {'scope': 'frozen dataset outer_test only; every published repetition/close event',
            'task_groups': sorted(tasks), 'sample_ids': [s['sample_id'] for s in rows],
            'sample_order_sha256': hash_json([s['sample_id'] for s in rows]),
            'rollouts': [{**r, 'sample_ids': by_rollout[r['rollout_id']]} for r in selected]}


def evaluate(output, attempt_dir, config, cohort, samples, split):
    """Write exact prediction copy, finite paired metrics, and exhaustive coverage."""
    source = attempt_dir / 'test-predictions.jsonl'
    rows = read_jsonl(source)
    output.joinpath('predictions.jsonl').write_bytes(source.read_bytes())
    if sha256(output / 'predictions.jsonl') != sha256(source):
        raise ValueError('Authoritative prediction copy differs')
    lookup = {s['sample_id']: s for s in samples}
    expected_ids = set(cohort['sample_ids'])
    targets = config.data.targets
    training_metrics = read_json(attempt_dir / 'metrics.json')
    checkpoints = {t: training_metrics[t]['refit_model'] for t in targets}
    model_hashes = {t: sha256(owned_path(attempt_dir, p)) for t, p in checkpoints.items()}
    indexed = {}
    for row in rows:
        key = (row['target'], row['sample_id'])
        if key in indexed or key[0] not in targets or key[1] not in expected_ids:
            raise ValueError('Duplicate or unexpected test prediction')
        sample = lookup[key[1]]
        if row['task_group'] != sample['task_group'] or row['rollout_id'] != sample['rollout_id']:
            raise ValueError('Prediction identity mismatch')
        if row['split'] != 'test':
            raise ValueError('Prediction is not outer test')
        expected_truth = sample[LABELS[key[0]]]
        expected_transformed = float(transform_target(np.array([expected_truth], float), key[0])[0])
        if row['target_raw'] != expected_truth or not np.isclose(row['target_transformed'], expected_transformed):
            raise ValueError('Prediction truth differs from frozen dataset')
        model = checkpoints[key[0]]
        if row['model'] != model or row['model_sha256'] != model_hashes[key[0]]:
            raise ValueError('Prediction does not bind selected-refit checkpoint')
        indexed[key] = row
    fit_ids = set(split['sample_ids']['train'] + split['sample_ids']['validation'])
    fit_rows = [s for s in samples if s['sample_id'] in fit_ids]
    metrics, audits, task_csv, rollout_csv = {}, {}, [], []
    for target in targets:
        ordered, statuses = [], []
        for sid in cohort['sample_ids']:
            row = indexed.get((target, sid))
            complete = bool(row and row.get('status') == 'complete' and all(
                row.get(k) is not None and np.isfinite(row[k]) for k in ('prediction_raw', 'prediction_transformed')))
            statuses.append({'sample_id': sid, 'rollout_id': lookup[sid]['rollout_id'],
                             'status': 'complete' if complete else 'failed',
                             'reason': None if complete else row.get('reason', 'invalid_prediction') if row else 'missing_prediction'})
            ordered.append(row or {'target_raw': lookup[sid][LABELS[target]],
                                   'target_transformed': float(transform_target(np.array([lookup[sid][LABELS[target]]]), target)[0]),
                                   'prediction_raw': None, 'prediction_transformed': None})
        per_rollout = []
        for rollout in cohort['rollouts']:
            points = [r for r in statuses if r['rollout_id'] == rollout['rollout_id']]
            count = sum(p['status'] == 'complete' for p in points)
            per_rollout.append({'rollout_id': rollout['rollout_id'], 'expected': len(points),
                                'successful': count, 'failed': len(points)-count})
        successful = sum(r['status'] == 'complete' for r in statuses)
        audits[target] = {'expected': len(statuses), 'successful': successful,
                          'failed': len(statuses)-successful, 'events': statuses, 'by_rollout': per_rollout}
        target_metrics = {}
        for space, truth_key, pred_key in (('test_raw_units', 'target_raw', 'prediction_raw'),
                                           ('test_transformed', 'target_transformed', 'prediction_transformed')):
            y = np.array([r[truth_key] for r in ordered], float)
            p = np.array([r[pred_key] if r.get(pred_key) is not None else np.nan for r in ordered], float)
            pooled = paired_regression_metrics(y, p)
            grouped = {}
            for field, name, csv_rows in (('task_group', 'by_task', task_csv), ('rollout_id', 'by_rollout', rollout_csv)):
                groups = defaultdict(list)
                for i, sid in enumerate(cohort['sample_ids']):
                    groups[lookup[sid][field]].append(i)
                grouped[name] = {key: paired_regression_metrics(y[idx], p[idx]) for key, idx in sorted(groups.items())}
                grouped['macro_task' if field == 'task_group' else 'macro_rollout'] = macro_metrics(list(grouped[name].values()))
                for key, report in grouped[name].items():
                    csv_rows.append({'target': target, 'space': space, field: key,
                                     **{k: v for k, v in report.items() if k not in {'formulas', 'reasons'}},
                                     'undefined_reasons': json.dumps(report['reasons'], sort_keys=True)})
            fit_y = np.array([s[LABELS[target]] for s in fit_rows], float)
            if space == 'test_transformed':
                fit_y = transform_target(fit_y, target)
            mean = float(fit_y.mean())
            target_metrics[space] = {'pooled': pooled, **grouped,
                                     'outer_train_mean_baseline': {'mean': mean, 'fit_sample_count': len(fit_rows),
                                         **paired_regression_metrics(y, np.full(len(y), mean))}}
        target_metrics['primary_space'] = 'test_raw_units'
        target_metrics['primary_nmse'] = 'nmse_energy' if config.evaluation.nmse_normalization == 'target_energy' else 'nmse_variance'
        metrics[target] = target_metrics
    coverage = {'scope': 'outer_test', 'expected_rollout_ids': sorted(r['rollout_id'] for r in cohort['rollouts']),
                'targets': audits, 'complete': all(a['failed'] == 0 for a in audits.values())}
    write_json(output / 'coverage.json', coverage)
    write_json(output / 'metrics.json', metrics)
    for filename, values in (('by-task.csv', task_csv), ('by-rollout.csv', rollout_csv)):
        fields = list(dict.fromkeys(k for row in values for k in row))
        with (output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    write_json(output / 'prediction-source.json', {'source': str(source), 'source_sha256': sha256(source),
                                                  'records': len(rows), 'copy_policy': 'byte identical; no new inference'})
    return metrics, coverage, indexed


def draw_test_trajectories(output, config, cohort, samples, indexed, metrics):
    # Importing this module, or disabling visualization, never loads matplotlib.
    from difficulty.evaluation.visualize import plot_trajectory
    lookup = {s['sample_id']: s for s in samples}
    manifests, cards = [], []
    for rollout in cohort['rollouts']:
        rid = rollout['rollout_id']
        for target in config.data.targets:
            ordered = sorted((lookup[sid] for sid in rollout['sample_ids']), key=lambda r: r['agent_step'])
            points, model = [], None
            for sample in ordered:
                pred = indexed.get((target, sample['sample_id']), {})
                model = pred or model
                points.append({'step_id': sample['agent_step'], 'target': sample[LABELS[target]],
                               'prediction': pred.get('prediction_raw'), 'prediction_log1p': pred.get('prediction_transformed')})
            info = {'rollout_id': rid, 'target': target, 'task_group': ordered[0]['task_group'],
                    'source_run_id': rollout.get('source_run_id', rollout.get('run_id')),
                    'repetition_id': rollout.get('repetition_id'), 'status': 'complete', 'plots': []}
            group_metrics = metrics[target]['test_raw_units']['by_rollout'][rid]
            info['metrics'] = group_metrics
            if model:
                info.update(layer=model['layer'], alpha=model['alpha'], checkpoint_sha256=model['model_sha256'],
                            native_site=model.get('native_site'), width=model.get('width'))
            info.update(expected_events=len(points), successful_events=sum(p['prediction'] is not None for p in points))
            title = f"{rollout['instance_id']} | rep={info['repetition_id']}\nrun={info['source_run_id']}\n{rid}\n{target}"
            number = lambda v: f'{v:.5g}' if v is not None else 'undefined'
            details = (f"public={model['layer']} native={model.get('native_site')} width={model.get('width')} alpha={model['alpha']:g} "
                       f"NMSE_energy={number(group_metrics['nmse_energy'])} R2={number(group_metrics['r2'])}\n"
                       f"checkpoint SHA256={model['model_sha256']}") if model else 'Missing predictions'
            label = LABELS[target].replace('_to_submission', '').replace('_to_success', '').replace('_', ' ')
            key = hash_json([rid, target])[:24]
            try:
                for extension in config.visualization.export_formats:
                    relative = f'plots/{key}.{extension}'
                    plot_trajectory(points, output / relative, title=title, ylabel=label,
                                    comparison=target.endswith('_log1p'), note=details,
                                    figsize=(11, 8 if target.endswith('_log1p') else 5.5), bottom_margin=.08)
                    info['plots'].append(relative)
            except Exception as error:
                info.update(status='failed', reason=f'{type(error).__name__}: {error}')
            manifests.append(info)
            title_html = html.escape(f"{info['task_group']} | {rid} | {target}")
            if info['status'] == 'complete':
                path = output / info['plots'][-1]
                mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/png'
                encoded = base64.b64encode(path.read_bytes()).decode('ascii')
                body = f'<img loading="lazy" alt="{title_html}" src="data:{mime};base64,{encoded}">'
            else:
                body = '<p>' + html.escape(info['reason']) + '</p>'
            cards.append(f'<details data-rollout="{html.escape(rid, quote=True)}" data-target="{target}" data-search="{title_html.lower()}"><summary>{title_html} | coverage {info["successful_events"]}/{info["expected_events"]} | {info["status"]}</summary>{body}</details>')
    write_json(output / 'plot-manifest.json', {'trajectories': manifests})
    content = ('<!doctype html><html><head><meta charset="utf-8"><title>Outer test Ridge trajectories</title>'
               '<style>body{font:15px system-ui;margin:2em}img{max-width:100%}details{margin:1em 0}input{padding:.6em;width:90%}</style>'
               '</head><body><h1>Outer test Ridge trajectories</h1><p>Frozen dataset cohort. Inner validation events can share trajectories; '
               'cross-task generalization is measured only on outer test. Predictions preserve negatives and gaps.</p>'
               '<input id="filter" placeholder="Filter by task, rollout or target">'
               '<details><summary>Overall metrics (raw units and transformed)</summary><pre>' +
               html.escape(json.dumps(metrics, ensure_ascii=False, indent=2)) + '</pre></details>' + ''.join(cards) +
               '<script>document.getElementById("filter").addEventListener("input",e=>{let q=e.target.value.toLowerCase();'
               'document.querySelectorAll("details[data-search]").forEach(d=>d.hidden=!d.dataset.search.includes(q))})</script></body></html>')
    (output / 'index.html').write_text(content, encoding='utf-8')
    if any(m['status'] != 'complete' for m in manifests):
        raise RuntimeError('One or more trajectory plots failed; see plot-manifest.json/index.html')
    return manifests
