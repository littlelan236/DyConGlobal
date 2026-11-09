"""Semantic verification of the holdout receipt, cohort and model/prediction chain."""
from html.parser import HTMLParser

from core.io import owned_path, read_json, sha256
from difficulty.evaluation.holdout_results import read_jsonl, build_cohort
from difficulty.evaluation.metrics import paired_regression_metrics


def verify_evaluation(root, directory, receipt):
    from difficulty.artifacts import verify
    from difficulty.datasets.submission import load_selection
    from difficulty.probe_config import RidgeProbe
    from difficulty.training.splits import load_split
    workflow = read_json(directory / 'workflow.json')
    resolved = read_json(directory / 'resolved-config.json')
    if (workflow['status'] != 'complete' or workflow['evaluation_id'] != directory.name
            or receipt['evaluation_id'] != directory.name or workflow['mode'] != receipt['mode']):
        raise ValueError('Holdout workflow identity/status mismatch')
    attempt_id, dataset_id, split_id = (receipt[k] for k in ('attempt_id', 'dataset_id', 'split_id'))
    if any(workflow[k] != receipt[k] for k in ('attempt_id', 'dataset_id', 'split_id')):
        raise ValueError('Holdout workflow child IDs mismatch')
    verify(root, 'difficulty-attempt', attempt_id)
    attempt_dir = owned_path(root, 'difficulty/checkpoints/' + attempt_id)
    attempt = read_json(attempt_dir / 'attempt.json')
    if (attempt['mode'] != receipt['mode'] or attempt['dataset_id'] != dataset_id or attempt['split_id'] != split_id
            or attempt['training_config'] != resolved['training_config']):
        raise ValueError('Holdout attempt identity/config mismatch')
    settings = RidgeProbe.model_validate(attempt['settings'])
    rollouts, samples, _ = load_selection(attempt['config'], settings)
    split, _ = load_split(root, dataset_id, split_id, attempt['dataset_receipt_sha256'], samples)
    if split['strategy'] != 'task_holdout_event_v1':
        raise ValueError('Holdout evaluation requires nested task/event split')
    cohort = read_json(directory / 'cohort.json')
    if cohort != build_cohort(rollouts, samples, split):
        raise ValueError('Holdout cohort differs from frozen dataset/split')
    if receipt['mode'] != 'fit':
        if any((directory / name).exists() for name in ('predictions.jsonl', 'metrics.json', 'index.html', 'plots')):
            raise ValueError('Non-fit holdout contains fit artifacts')
        return
    provenance = read_json(directory / 'prediction-source.json')
    if not (sha256(directory / 'predictions.jsonl') == provenance['source_sha256'] == sha256(attempt_dir / 'test-predictions.jsonl')):
        raise ValueError('Holdout authoritative predictions mismatch')
    rows = read_jsonl(directory / 'predictions.jsonl')
    expected = {(target, sid) for target in settings.targets for sid in cohort['sample_ids']}
    actual = {(r['target'], r['sample_id']) for r in rows}
    if actual != expected or len(actual) != len(rows) or any(r.get('status') != 'complete' for r in rows):
        raise ValueError('Holdout test prediction coverage is incomplete')
    models = read_json(attempt_dir / 'metrics.json')
    for target in settings.targets:
        path = models[target]['refit_model']
        digest = sha256(attempt_dir / path)
        if any(r['model'] != path or r['model_sha256'] != digest for r in rows if r['target'] == target):
            raise ValueError('Holdout predictions use a different selected-refit checkpoint')
    metrics, coverage = read_json(directory / 'metrics.json'), read_json(directory / 'coverage.json')
    if not coverage['complete'] or coverage['expected_rollout_ids'] != sorted(r['rollout_id'] for r in cohort['rollouts']):
        raise ValueError('Holdout rollout coverage mismatch')
    lookup = {(r['target'], r['sample_id']): r for r in rows}
    for target in settings.targets:
        ordered = [lookup[(target, sid)] for sid in cohort['sample_ids']]
        audit = coverage['targets'][target]
        if (audit['expected'] != len(ordered) or audit['successful'] != len(ordered) or audit['failed'] != 0
                or {r['sample_id'] for r in audit['events']} != set(cohort['sample_ids'])
                or len(audit['events']) != len(ordered)
                or any(r['status'] != 'complete' for r in audit['events'])):
            raise ValueError('Holdout event coverage audit mismatch')
        for space, truth, prediction in (('test_raw_units', 'target_raw', 'prediction_raw'),
                                         ('test_transformed', 'target_transformed', 'prediction_transformed')):
            if metrics[target][space]['pooled'] != paired_regression_metrics(
                    [r[truth] for r in ordered], [r[prediction] for r in ordered]):
                raise ValueError('Holdout pooled metrics differ from authoritative predictions')
    enabled = resolved['workflow']['visualization']['enabled']
    if not enabled:
        if any((directory / name).exists() for name in ('index.html', 'plots', 'plot-manifest.json')):
            raise ValueError('Visualization disabled but plot artifacts exist')
        return
    expected_plots = {(r['rollout_id'], t) for r in cohort['rollouts'] for t in settings.targets}
    manifest = read_json(directory / 'plot-manifest.json')['trajectories']
    if len(manifest) != len(expected_plots) or {(r['rollout_id'], r['target']) for r in manifest} != expected_plots:
        raise ValueError('Holdout plot manifest misses test trajectories')
    for row in manifest:
        model = models[row['target']]['refit_model']
        if row['status'] != 'complete' or row['checkpoint_sha256'] != sha256(attempt_dir / model):
            raise ValueError('Holdout plot checkpoint/status mismatch')
        if not row['plots'] or any(name not in receipt['files'] for name in row['plots']):
            raise ValueError('Holdout plot missing from receipt')

    class Index(HTMLParser):
        def __init__(self):
            super().__init__()
            self.keys = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'details' and 'data-rollout' in attrs:
                self.keys.append((attrs['data-rollout'], attrs['data-target']))

    index = Index()
    index.feed((directory / 'index.html').read_text(encoding='utf-8'))
    if len(index.keys) != len(expected_plots) or set(index.keys) != expected_plots:
        raise ValueError('Holdout HTML index misses test trajectories')
