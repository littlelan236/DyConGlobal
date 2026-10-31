"""Resolved repetition counts and receipt-bound Ridge/truth trajectory views."""
from collections import Counter, defaultdict
import base64
import csv
import html
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from action.datasets.trajectories import read_actions
from application.execution.evidence import RunEvidence
from application.execution.run_plan import load_plan
from application.execution.subject import artifact_kinds, verify_subject
from core.io import artifact_dir, identifier, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.datasets.submission import load_selection, prepare_cache
from difficulty.datasets.selection import select_run
from difficulty.models.ridge_grid import load_model, predict_model
from difficulty.receipts import verify_receipt
from difficulty.evaluation.visualize import plot_trajectory
from difficulty.evaluation.trajectory_sources import predict_repaired, publication_truth, extend_original_publications

TARGETS = {
    'submission_remaining_reasoning_tokens_log1p': ('remaining_reasoning_tokens_to_submission', True),
    'submission_remaining_reasoning_tokens_linear': ('remaining_reasoning_tokens_to_submission', False),
    'submission_remaining_steps_log1p': ('remaining_steps_to_submission', True),
    'submission_remaining_steps_linear': ('remaining_steps_to_submission', False),
}


def final_selection(root, run_id, catalog):
    """Use frozen work items and the existing canonical final-result verifier."""
    from rollout.retry import verify_final
    plan = load_plan(root, run_id)
    items = plan['work_items']['items']
    ids, sources, pending = [], {}, []
    for item in items:
        path = Path(root) / 'runs' / run_id / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id'] / 'result.json'
        if not path.exists():
            pending.append(item)
            continue
        result = read_json(path)
        if result.get('disposition', 'accepted') != 'accepted':
            pending.append(item)
            continue
        ids.append(verify_final(root, run_id, item, result, catalog))
        sources[str(path)] = sha256(path)
    selected = RunEvidence(root, run_id).select(ids)
    if len(selected['rollout_ids']) + len(pending) != len(items):
        raise ValueError('Final selection does not cover all frozen work items')
    return selected, sources, items, pending


def summarize_repetitions(tasks, rollouts):
    """Count each canonical task/repetition once; never count retries twice."""
    by_task = {t['instance_id']: [] for t in tasks.values()}
    seen = set()
    for rid, row in rollouts.items():
        task = tasks[row['task_key']]['instance_id']
        repetition = row.get('repetition_id') or row.get('sample_key') or rid
        identity = (task, repetition)
        if identity in seen:
            raise ValueError('Multiple selected rollouts for one task/repetition: ' + str(identity))
        seen.add(identity)
        by_task[task].append((rid, row))
    result = []
    for task, entries in sorted(by_task.items()):
        statuses = Counter('resolved' if r['evaluation_status'] == 'completed' and r['resolved'] is True
                           else 'unresolved' if r['evaluation_status'] == 'completed' and r['resolved'] is False
                           else 'ungraded' for _, r in entries)
        result.append(dict(task_id=task, selected_rollouts=len(entries),
                           resolved=statuses['resolved'], unresolved=statuses['unresolved'],
                           ungraded=statuses['ungraded']))
    return result


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def build_html(task_rows, trajectories, output, summary):
    esc = lambda value: html.escape(str(value), quote=True)
    by_task = defaultdict(list)
    for row in trajectories:
        by_task[row['task_id']].append(row)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Repeat rollout: predictions and truth</title>',
             '<style>body{font:15px system-ui;margin:24px;color:#172033;background:#f5f7fb}'
             'header,section{background:white;padding:18px;margin-bottom:18px;border-radius:10px}'
             '.strip{display:flex;flex-wrap:nowrap;overflow-x:auto;gap:14px;align-items:flex-start}'
             'figure{flex:0 0 540px;margin:0}img{width:100%;height:auto}figcaption{font-size:12px;overflow-wrap:anywhere}'
             'table{border-collapse:collapse}th,td{padding:5px 14px;border-bottom:1px solid #ddd;text-align:left}'
             '.warning{color:#9a3412}input{padding:8px;width:360px}h2{font-size:17px}</style>',
             '<header><h1>Repeat rollout：预测与真实剩余量</h1>',
             '<p>来源 run：' + esc(summary['run_id']) + '<br>模型：' + esc(summary['attempt_id']) +
             '<br>目标：' + esc(summary['target']) + '</p>',
             '<p>' + esc(summary['selected_rollouts']) + ' 条有最终结果的独立 repetition；' +
             esc(summary['resolved_rollouts']) + ' 条 resolved；绘制 ' + esc(summary.get('plotted_rollouts', summary['resolved_rollouts'])) + ' 条；' + esc(summary['plotted_with_predictions']) +
             ' 条有可用预测，' + esc(summary['without_predictions']) + ' 条无可用预测。</p>',
             '<p class="warning">缺少最终结果的 repetition：' + esc(summary.get('pending_repetitions', 0)) +
             '。未评测/基础设施故障不伪造失败评测。historical_attempt 为历史尝试；最终成功次数与全部成功轨迹分别统计。</p>',
             '<p class="warning">诊断视图：' + esc(summary.get('interpretation', '训练重叠情况见 CSV')) + '。'
             '样本 split 见 CSV。预测与真值分别筛选：有 hidden 可预测，缺真值不删除预测。'
             'identity-only 点单列为预测证据，不冒充 native exact token 标签。真实 reasoning tokens 仅采用已验证 native 标签；缺失值保留断线。'
             '横轴为原始 0-based assistant step；close 时当前 action 尚未执行，剩余 steps=T−s。'
             '同 task 共用坐标范围；每条轨迹单独绘图，不平均、不平滑、不截断负预测。</p>',
             '<label>筛选 task：<input id="filter" placeholder="输入 task 名称"></label>',
             '<p>同一 task 的所选轨迹位于同一行，可横向滚动。显式轨迹视图的绿线是到本次 agent 结束的实际剩余量，'
             '不代表到成功解题的剩余量；评测状态独立显示。此 HTML 已嵌入全部图像，可单独复制打开。</p>',
             '<details><summary>全部 task 的 resolved 次数</summary><table><thead><tr>'
             '<th>Task</th><th>最终 resolved</th><th>Unresolved</th><th>未评测/故障</th><th>已完成</th>'
             '<th>计划次数</th><th>缺最终结果</th><th>全部成功轨迹</th>'
             '</tr></thead><tbody>']
    for row in task_rows:
        parts.append('<tr>' + ''.join('<td>' + esc(row[k]) + '</td>' for k in
                     ('task_id', 'resolved', 'unresolved', 'ungraded', 'selected_rollouts')) +
                     ''.join('<td>' + esc(row.get(k, '')) + '</td>' for k in
                             ('expected_rollouts', 'pending', 'all_resolved_attempts')) + '</tr>')
    parts.append('</tbody></table></details></header>')
    for task in task_rows:
        parts.append('<section data-task="' + esc(task['task_id'].lower()) + '"><h2>' +
                     esc(task['task_id']) + ' — resolved ' + str(task['resolved']) + '/' +
                     str(task.get('expected_rollouts', task['selected_rollouts'])) + '</h2><div class="strip">')
        for row in by_task[task['task_id']]:
            data = base64.b64encode((output / row['plot']).read_bytes()).decode('ascii')
            parts.append('<figure><img loading="lazy" alt="' + esc(row['rollout_id']) +
                         '" src="data:image/svg+xml;base64,' + data + '"><figcaption>' +
                         esc(row['repetition_id']) + ' | ' + esc(row.get('selection_role', 'final')) + ' | ' + esc(row['rollout_id']) + '<br>' +
                         'resolved=' + esc(row.get('resolved', 'unknown')) + ' | ' + esc(row.get('termination_reason', '')) + '<br>' +
                         str(row['predicted_points']) + '/' + str(row['total_steps']) +
                         ' steps 有预测；真值 ' + str(row.get('true_points', 0)) + '/' + str(row['total_steps']) + '；' + esc(row['coverage_status']) + '</figcaption></figure>')
        if not by_task[task['task_id']]:
            parts.append('<p>本 task 无符合本次选择的轨迹。</p>')
        parts.append('</div></section>')
    parts.append('<script>document.getElementById("filter").addEventListener("input",function(){'
                 'const q=this.value.toLowerCase();document.querySelectorAll("section[data-task]")'
                 '.forEach(s=>s.hidden=!s.dataset.task.includes(q));});</script></html>')
    (output / 'report.html').write_text('\n'.join(parts), encoding='utf-8')


@exclusive_artifact('difficulty-evaluation', 'plot_id')
def visualize_trajectories(cfg, run_id, plot_id, *, attempt_id, dataset_id=None, target, repair_id=None, reuse_predictions_id=None,
                           rollout_ids=(), rollout_list=None):
    """Reuse strict dataset/cache/model readers; publish one immutable report."""
    if target not in TARGETS:
        raise ValueError('Unsupported trajectory target')
    root = Path(cfg['project_root'])
    explicit = bool(rollout_ids) or rollout_list is not None
    if not explicit and not dataset_id:
        raise ValueError('Default resolved view requires --dataset-id')
    if explicit and (dataset_id or repair_id or reuse_predictions_id):
        raise ValueError('Explicit trajectory view uses original native publications; omit dataset/repair/cache options')
    for value in (run_id, plot_id, attempt_id):
        identifier(value)
    if dataset_id:
        identifier(dataset_id)
    out = artifact_dir(cfg, 'difficulty', plot_id, 'evaluations')
    if out.exists():
        raise FileExistsError(out)
    attempt_dir = artifact_dir(cfg, 'difficulty', attempt_id, 'checkpoints')
    dataset_dir = artifact_dir(cfg, 'difficulty', dataset_id, 'datasets') if dataset_id else None
    verify_receipt(root, attempt_dir)
    attempt = read_json(attempt_dir / 'attempt.json')
    if attempt.get('mode') != 'fit':
        raise ValueError('Trajectory prediction requires a completed fit attempt')
    selected = read_json(attempt_dir / 'selected.json')['targets'][target]
    metrics = read_json(attempt_dir / 'metrics.json')[target]
    model_relative = metrics.get('refit_model') or selected['model']
    model_path = owned_path(attempt_dir, model_relative)
    metadata_path = model_path.with_suffix('.json')
    sidecar = read_json(metadata_path)
    metadata = sidecar['metadata']
    if sidecar['checkpoint_sha256'] != sha256(model_path):
        raise ValueError('Checkpoint SHA mismatch')
    if metadata.get('target') != target:
        raise ValueError('Checkpoint target mismatch')
    if (metadata.get('model_kind') != 'ridge'
            or metadata.get('model_fingerprint') != cfg['model']['fingerprint']
            or metadata.get('layer') != selected['layer']):
        raise ValueError('Checkpoint hidden/model contract mismatch')
    evidence = RunEvidence(root, run_id)
    catalog = evidence.read()
    physical_rollout_count = len(catalog['rollouts'])
    selection, final_sources, work_items, pending = final_selection(root, run_id, catalog)
    requested = select_run(root, run_id, rollout_ids, rollout_list) if explicit else None
    if requested:
        # Freeze only the requested cohort for native loading, while retaining
        # the independent full-run final-result counts above.
        catalog['rollouts'] = requested['rollouts']
    tasks = summarize_repetitions(catalog['tasks'], requested['rollouts'] if requested else selection['rollouts'])
    if requested:
        tasks = [task for task in tasks if task['selected_rollouts']]
    for task in tasks:
        task['expected_rollouts'] = sum(i['instance_id'] == task['task_id'] for i in work_items)
        task['pending'] = sum(i['instance_id'] == task['task_id'] for i in pending)
    samples, dataset_receipt = [], None
    if dataset_dir:
        dataset_receipt = sha256(dataset_dir / 'receipt.json')
        dataset_rollouts = read_json(dataset_dir / 'rollouts.json')['rollouts']
        sample_count = sum(bool(line.strip()) for line in (dataset_dir / 'samples.jsonl').read_text().splitlines())
        settings = SimpleNamespace(dataset_id=dataset_id, dataset_receipt_sha256=dataset_receipt,
            expected_tasks=len({(r['repo'], r['instance_id']) for r in dataset_rollouts}),
            expected_rollouts=len(dataset_rollouts), expected_samples=sample_count)
        _, samples, _ = load_selection(cfg, settings)
    model = load_model(model_path)
    layer = selected['layer']
    field, logarithmic = TARGETS[target]
    true_map = {(s['rollout_id'], s['agent_step']): s[field] for s in samples}
    training_by_step = {(s['rollout_id'], s['agent_step']): s for s in samples}
    coverage_sources, coverage_audit = {}, {}
    if explicit:
        action_map = {rid: read_actions(root, dict(row, rollout_id=rid)) for rid, row in catalog['rollouts'].items()}
        original, observed, coverage_audit = extend_original_publications(
            cfg, run_id, catalog, {}, action_map, {}, model, layer, coverage_sources)
        if field == 'remaining_reasoning_tokens_to_submission':
            true_map.update(observed)
        scored = list(original.values())
    elif repair_id:
        repaired, coverage_sources, coverage_audit, action_map = predict_repaired(
            cfg, run_id, repair_id, catalog, model, layer, cache_id=reuse_predictions_id, model_path=model_path, target=target)
        missing_training = set(training_by_step) - set(repaired)
        if missing_training:
            raise ValueError('Requested repair omits existing training prediction points: ' + str(len(missing_training)))
        # GT uses its own eligibility: exact native suffix labels can be drawn
        # even when this step has no captured hidden tensor.
        more_truth, publication_id = publication_truth(root, run_id, dataset_dir, catalog, action_map, coverage_sources)
        if field == 'remaining_reasoning_tokens_to_submission':
            for key, value in more_truth.items():
                if key in true_map and true_map[key] != value:
                    raise ValueError('Independent native truth conflicts with training labels')
                true_map[key] = value
        repaired, native_truth, original_audit = extend_original_publications(
            cfg, run_id, catalog, repaired, action_map, true_map, model, layer, coverage_sources)
        if field == 'remaining_reasoning_tokens_to_submission':
            for key, value in native_truth.items():
                if key in true_map and true_map[key] != value:
                    raise ValueError('Original native GT conflicts with existing exact labels')
                true_map[key] = value
        coverage_audit.update(original_publications=original_audit,
            binding_counts=dict(Counter(s.get('prediction_binding') for s, _ in repaired.values())))
        coverage_audit['truth_publication_id'] = publication_id
        scored = [(sample, score) for sample, score in repaired.values()]
    else:
        cache = prepare_cache(cfg, settings, samples)
        matrix = np.load(cache / f'layer-{layer:02d}.f32.npy', mmap_mode='r', allow_pickle=False)
        scores = np.concatenate([predict_model(model, np.asarray(matrix[i:i+512], dtype=float))
                                 for i in range(0, len(samples), 512)])
        scored = list(zip(samples, scores, strict=True))
    sample_map = {}
    for sample, score in scored:
        if sample.get('run_id', sample.get('source_run_id')) != run_id:
            raise ValueError('Dataset contains samples from a different run')
        key = (sample['rollout_id'], sample['agent_step'])
        if key in sample_map:
            raise ValueError('Duplicate sample for one trajectory step')
        with np.errstate(over='raise', invalid='raise'):
            prediction = float(np.expm1(score)) if logarithmic else float(score)
        if not np.isfinite(score) or not np.isfinite(prediction):
            raise ValueError('Nonfinite predictions')
        if key in training_by_step and sample['sample_id'] != training_by_step[key]['sample_id']:
            if sample['hidden_sha256'] != training_by_step[key]['hidden_sha256']:
                raise ValueError('Repair and training hidden disagree at the same canonical close')
        sample_map[key] = (sample, float(score), prediction)
    coverage_audit.update(baseline_training_points=len(samples), predicted_points=len(sample_map),
                          added_prediction_points=len(sample_map)-len(samples),
                          independent_truth_points=len(true_map))
    roles = {}
    if dataset_dir and dataset_id == attempt['dataset_id']:
        split_path = dataset_dir / 'splits' / attempt['split_id'] / 'split.json'
        split = read_json(split_path)
        if sha256(split_path) != attempt['split_sha256']:
            raise ValueError('Training split changed')
        roles = {sid: role for role, ids in split['sample_ids'].items() for sid in ids}
    # Keep every published resolved trajectory; mark historical retries explicitly.
    resolved = {rid: row for rid, row in catalog['rollouts'].items()
                if row['evaluation_status'] == 'completed' and row['resolved'] is True
                and not row.get('provenance', {}).get('quarantined')}
    plotted = catalog['rollouts'] if explicit else resolved
    for task in tasks:
        task['all_resolved_attempts'] = sum(catalog['tasks'][r['task_key']]['instance_id'] == task['task_id']
                                             for r in resolved.values())
    grouped, curves, manifests = defaultdict(list), [], []
    source_files = {str(attempt_dir / 'receipt.json'): sha256(attempt_dir / 'receipt.json'),
                    str(model_path): sha256(model_path), str(metadata_path): sha256(metadata_path),
                    **final_sources, **coverage_sources}
    if dataset_dir:
        source_files[str(dataset_dir / 'receipt.json')] = dataset_receipt
    if rollout_list:
        source_files[str(Path(rollout_list).resolve())] = sha256(rollout_list)
    snapshots = {'run_evidence': catalog, 'selection': selection, 'pending_work_items': pending,
                 'explicit_selection': requested}
    for rid, row in sorted(plotted.items(), key=lambda item: (
            catalog['tasks'][item[1]['task_key']]['instance_id'], item[1].get('repetition_id', ''), item[0])):
        task = catalog['tasks'][row['task_key']]['instance_id']
        source = {**row, 'rollout_id': rid}
        actions = read_actions(root, source)
        verify_subject(root, row)
        eid = row.get('selected_evaluation_id')
        if eid:
            report = catalog['evaluations'][eid]
            if sha256(owned_path(root, report['report_path'])) != report['report_sha256']:
                raise ValueError('Selected evaluation report changed')
            source_files[str(owned_path(root, report['report_path']))] = report['report_sha256']
        points = []
        for step in range(len(actions)):
            found = sample_map.get((rid, step))
            sample, score, prediction = found if found else ({}, None, None)
            if found and (sample['trajectory_sha256'] != row['trajectory_sha256']
                          or sample['message_index'] != actions[step]['message_index']):
                raise ValueError('Prediction does not bind canonical step')
            truth = true_map.get((rid, step))
            if field == 'remaining_steps_to_submission':
                truth = len(actions) - step
            score_log = score if logarithmic else (float(np.log1p(prediction)) if prediction is not None and prediction > -1 else None)
            points.append(dict(task_id=task, rollout_id=rid, step_id=step, sample_id=sample.get('sample_id'),
                               resolved=row.get('resolved'), evaluation_status=row.get('evaluation_status'),
                               truth_endpoint='observed_agent_terminal' if explicit else 'successful_submission',
                               target=truth, prediction=prediction, prediction_transformed=score,
                               prediction_log1p=score_log,
                               capture_source=sample.get('capture_source', 'pointwise_repair') if found else None,
                               prediction_binding=sample.get('prediction_binding', 'training_native_exact') if found else None,
                               truth_available=truth is not None,
                               hidden_available=found is not None,
                               used_in_fit=None if explicit else (roles.get(sample.get('sample_id')) in
                                            ({'train', 'validation'} if metadata.get('fit_scope') == 'train_validation' else {'train'})) if found and roles else None,
                               split='not_audited' if explicit else roles.get(sample.get('sample_id'), 'unavailable' if not found else 'not_in_training_dataset'),
                               remaining_steps_true=len(actions)-step))
        curves.extend(points)
        manifest = dict(task_id=task, rollout_id=rid, repetition_id=row.get('repetition_id', ''),
                        resolved=row.get('resolved'), evaluation_status=row.get('evaluation_status'),
                        termination_reason=row.get('termination_reason'),
                        truth_endpoint='observed_agent_terminal' if explicit else 'successful_submission',
                        selection_role='final' if rid in selection['rollouts'] else 'historical_attempt',
                        trajectory_sha256=row['trajectory_sha256'], selected_evaluation_id=eid,
                        total_steps=len(actions), predicted_points=sum(p['prediction'] is not None for p in points),
                        true_points=sum(p['target'] is not None for p in points),
                        prediction_only_points=sum(p['prediction'] is not None and p['target'] is None for p in points),
                        truth_only_points=sum(p['prediction'] is None and p['target'] is not None for p in points),
                        plot=f'plots/{rid}.svg')
        manifest['coverage_status'] = ('complete' if manifest['predicted_points'] == len(actions) else
                                       'partial: missing points are gaps' if manifest['predicted_points'] else
                                       'unavailable: no verifiable canonical close hidden')
        manifests.append(manifest)
        grouped[task].append((manifest, points))
        source_files[str(owned_path(root, row['trajectory_path']))] = row['trajectory_sha256']
        for kind in artifact_kinds(row):
            subject_path = owned_path(root, row[kind + '_path'])
            source_files[str(subject_path)] = row[kind + '_sha256']
            if kind == 'subject':
                for output in read_json(subject_path)['outputs']:
                    source_files[str(owned_path(subject_path.parent, output['path']))] = output['sha256']
    summary = dict(run_id=run_id, attempt_id=attempt_id, dataset_id=dataset_id, target=target,
                   model=str(model_path), fit_scope=metadata.get('fit_scope'), tasks=len(tasks),
                   selected_rollouts=len(plotted) if explicit else len(selection['rollouts']), physical_rollouts=physical_rollout_count,
                   resolved_rollouts=len(resolved), plotted_with_predictions=sum(m['predicted_points'] > 0 for m in manifests),
                   plotted_rollouts=len(plotted), selection_mode='explicit' if explicit else 'resolved',
                   truth_endpoint='observed_agent_terminal' if explicit else 'successful_submission',
                   final_resolved_rollouts=sum(t['resolved'] for t in tasks),
                   expected_repetitions=len(work_items), pending_repetitions=len(pending),
                   without_predictions=sum(m['predicted_points'] == 0 for m in manifests),
                   complete_curves=sum(m['predicted_points'] == m['total_steps'] for m in manifests),
                   prediction_points=sum(m['predicted_points'] for m in manifests),
                   true_points=sum(m['true_points'] for m in manifests),
                   prediction_only_points=sum(m['prediction_only_points'] for m in manifests),
                   truth_only_points=sum(m['truth_only_points'] for m in manifests),
                   repair_id=repair_id, reuse_predictions_id=reuse_predictions_id, coverage_policy='independent_prediction_and_truth',
                   added_prediction_points=len(sample_map)-len(samples),
                   success_histogram=dict(sorted(Counter(t['resolved'] for t in tasks).items())),
                   interpretation=('Explicit native trajectories; observed termination is not successful completion; '
                                   'training task overlap not audited' if explicit else
                                   'diagnostic; includes training samples, not independent generalization'),
                   negative_predictions_preserved=True, missing_values='null; no interpolation')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.' + plot_id + '-', dir=out.parent) as staging:
        staged = Path(staging)
        for task, trajectories in grouped.items():
            xmax = max(m['total_steps'] for m, _ in trajectories) - 1
            values = [p[k] for _, ps in trajectories for p in ps for k in ('target', 'prediction') if p[k] is not None]
            limits = (min([0.0] + values), max([1.0] + values))
            logvalues = [float(np.log1p(p['target'])) for _, ps in trajectories for p in ps if p['target'] is not None]
            logvalues += [p['prediction_log1p'] for _, ps in trajectories for p in ps if p['prediction_log1p'] is not None]
            loglimits = (min([0.0] + logvalues), max([1.0] + logvalues))
            for manifest, points in trajectories:
                plot_trajectory(points, staged / manifest['plot'], title=task + '\n' + manifest['repetition_id'],
                                ylabel=target.replace('submission_remaining_', '').replace('_log1p', '').replace('_linear', ''),
                                comparison=True, xlim=(-0.5, max(0.5, xmax+0.5)), ylim=limits, logylim=loglimits,
                                note=manifest['coverage_status'] + ' | resolved=' + str(manifest['resolved']),
                                truth_label='Observed to agent termination' if explicit else 'True remaining')
            print('plotted ' + task, flush=True)
        write_json(staged / 'summary.json', summary)
        write_json(staged / 'coverage-audit.json', coverage_audit)
        write_json(staged / 'selection.json', snapshots)
        write_json(staged / 'trajectories.json', manifests)
        write_csv(staged / 'task-resolved-counts.csv', tasks, list(tasks[0]))
        write_csv(staged / 'curves.csv', curves, ['task_id','rollout_id','step_id','sample_id','resolved','evaluation_status','truth_endpoint','target',
                  'prediction','prediction_transformed','prediction_log1p','split','used_in_fit','remaining_steps_true','prediction_binding','truth_available','hidden_available','capture_source'])
        write_csv(staged / 'trajectories.csv', manifests, list(manifests[0]) if manifests else ['task_id','rollout_id'])
        build_html(tasks, manifests, staged, summary)
        for name in ('difficulty/evaluation/trajectories.py', 'difficulty/evaluation/visualize.py', 'difficulty/evaluation/trajectory_sources.py', 'enrichment/online_trace.py', 'core/cli.py'):
            path = root / name
            destination = staged / 'code' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        for path, digest in source_files.items():
            if sha256(path) != digest:
                raise ValueError('Bound source changed during plotting: ' + path)
        write_json(staged / 'receipt.json', dict(schema_version=1, state='complete',
                   source_files=source_files, files={str(p.relative_to(staged)):sha256(p)
                       for p in sorted(staged.rglob('*')) if p.is_file()}))
        staged.rename(out)
    return dict(output=str(out), html=str(out / 'report.html'), **summary)
