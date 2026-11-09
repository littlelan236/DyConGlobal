import builtins
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.io import read_json, sha256, write_json
from difficulty.artifacts import verify
from difficulty.evaluation.holdout import run, child_id
from difficulty.evaluation.holdout_config import HoldoutConfig
from difficulty.evaluation.metrics import paired_regression_metrics, macro_metrics
from difficulty.models.ridge_grid import load_model, predict_model
from difficulty.training.split import make_task_holdout_split, make_event_twoway_split
from difficulty.training.splits import create_split, load_split
from tests.test_ridge_probe import fixture_project  # noqa: F401


def config_file(cfg, name, *, enabled=False, mode='fit', layers=None):
    root = Path(cfg['project_root'])
    config = {'schema_version': 1, 'run': {'mode': mode}, 'data': {'dataset_id': 'fixture'},
              'model': {'kind': 'ridge', 'ridge': {'layers': layers or [0, 31], 'alphas': [1.0, 100.0]}},
              'visualization': {'enabled': enabled}}
    path = root / (name+'.yaml')
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return path


def test_nested_split_multiple_runs_repetitions_and_manifest():
    rollouts, samples = [], []
    for task in range(6):
        for rep in range(2):
            rid = f'r{task}-{rep}'
            row = {'benchmark': 'skillsbench', 'revision': 'v1', 'instance_id': str(task),
                   'task_key': f'skillsbench:v1:{task}', 'rollout_id': rid, 'repetition_id': str(rep),
                   'source_run_id': f'run{rep}', 'source_kind': 'online', 'trajectory_sha256': rid}
            rollouts.append(row)
            samples.extend({'sample_id': f'{rid}:{s}', 'rollout_id': rid,
                            'task_group': f'skillsbench@v1::{task}', 'source_kind': 'online'} for s in range(5))
    split = make_task_holdout_split(rollouts, samples)
    assert split == make_task_holdout_split(list(reversed(rollouts)), samples)
    assert set(split['sample_ids']) == {'train', 'validation'}
    inner = [sid for ids in split['sample_ids'].values() for sid in ids]
    outer = split['outer_test_sample_ids']
    assert not set(inner) & set(outer)
    assert len(set(inner + outer)) == len(samples)
    assert {s['task_group'] for s in samples if s['sample_id'] in inner}.isdisjoint(
        {s['task_group'] for s in samples if s['sample_id'] in outer})
    train_rollouts = {s['rollout_id'] for s in samples if s['sample_id'] in split['sample_ids']['train']}
    val_rollouts = {s['rollout_id'] for s in samples if s['sample_id'] in split['sample_ids']['validation']}
    assert train_rollouts & val_rollouts  # allowed and expected event correlation
    manifest = {'task_assignments': [{'task_key': r['task_key'], 'instance_id': r['instance_id'],
                                      'partition': 'test' if r['instance_id'] == '0' else 'train'}
                                     for r in rollouts[::2]]}
    explicit = make_task_holdout_split(rollouts, samples, manifest=manifest)
    assert explicit['outer']['task_counts'] == {'test': 1, 'train': 5}
    for bad in (manifest['task_assignments'][:-1], manifest['task_assignments'] + [manifest['task_assignments'][0]],
                manifest['task_assignments'] + [{'task_key': 'unknown', 'instance_id': 'no', 'partition': 'test'}]):
        with pytest.raises(ValueError, match='manifest'):
            make_task_holdout_split(rollouts, samples, manifest={'task_assignments': bad})
    rollouts[0]['benchmark'] = 'webarena_verified'
    with pytest.raises(ValueError, match='WebArena.*template'):
        make_task_holdout_split(rollouts, samples)


def test_small_stratum_rules_and_degenerate_rejection():
    rows = [{'sample_id': str(i), 'source_kind': 'a' if i == 0 else 'b'} for i in range(4)]
    split = make_event_twoway_split(rows)
    assert '0' in split['sample_ids']['train']
    assert split['source_allocations']['a'] == {'train': 1, 'validation': 0}
    with pytest.raises(ValueError, match='empty'):
        make_event_twoway_split(rows[:1])
    with pytest.raises(ValueError, match='at least two'):
        make_task_holdout_split([{'repo': 'x', 'instance_id': 'one'}], [])


def test_metrics_manual_undefined_and_macro():
    m = paired_regression_metrics([1, 2, 3], [2, 2, 2])
    assert m['sse'] == 2 and m['nmse_energy'] == pytest.approx(1/7)
    assert m['nmse_variance'] == 1 and m['r2'] == 0
    assert m['nmse_variance'] == 1-m['r2']
    assert paired_regression_metrics([1, 2], [9, 9])['r2'] < 0
    zero = paired_regression_metrics([0, 0], [1, 1])
    assert zero['nmse_energy'] is None and zero['reasons']['nmse_energy'] == 'zero_energy'
    assert zero['r2'] is None
    one = paired_regression_metrics([2], [3])
    assert one['r2'] is None and one['reasons']['r2'] == 'insufficient_count'
    assert paired_regression_metrics([], [])['reason'] == 'empty'
    pairs = paired_regression_metrics([1, 2, 3], [1, float('inf'), 4])
    assert pairs['count'] == 2 and pairs['excluded_nonfinite_count'] == 1
    overflow = paired_regression_metrics([1e308, 1e308], [-1e308, -1e308])
    assert overflow['mse'] is None and overflow['reasons']['nmse_energy'] == 'arithmetic_overflow'
    json.dumps(overflow, allow_nan=False)
    a, b = paired_regression_metrics([1, 2], [1, 2]), paired_regression_metrics([10, 20, 30], [0, 0, 0])
    macro = macro_metrics([a, b, zero])
    assert macro['r2']['defined_groups'] == 2 and macro['r2']['undefined_groups'] == 1
    pooled = paired_regression_metrics([1, 2, 10, 20, 30], [1, 2, 0, 0, 0])
    assert macro['r2']['value'] != pooled['r2']


@pytest.mark.parametrize('overrides', [
    {'visualization': {'enabled': 'false'}}, {'outer_split': {'test_fraction': 0.0}},
    {'inner_split': {'strategy': 'task_random_v1'}},
    {'outer_split': {'strategy': 'task_manifest_v1', 'task_partitions': 'x', 'test_fraction': 0.2}},
    {'data': {'dataset_id': 'fixture', 'split': {'id': 'bad'}}}, {'unknown': True},
])
def test_strict_workflow_config(overrides):
    with pytest.raises(ValueError):
        HoldoutConfig.model_validate({'data': {'dataset_id': 'fixture'}, **overrides})


def test_nested_artifact_roundtrip_and_tampered_protocol(fixture_project):
    cfg, settings, dataset = fixture_project
    root = Path(cfg['project_root'])
    create_split(root, 'fixture', 'nested', strategy='task_holdout_event_v1')
    samples = [json.loads(line) for line in (dataset/'samples.jsonl').read_text().splitlines()]
    split, _ = load_split(root, 'fixture', 'nested', settings.dataset_receipt_sha256, samples)
    assert verify(root, 'difficulty-split', 'nested', dataset_id='fixture')['verified']
    path = dataset / 'splits/nested'
    split['outer_test_sample_ids'][0] = split['sample_ids']['train'][0]
    write_json(path/'split.json', split)
    proof = read_json(path/'receipt.json')
    proof['split_sha256'] = proof['files']['split.json'] = sha256(path/'split.json')
    write_json(path/'receipt.json', proof)
    with pytest.raises(ValueError, match='protocol mismatch'):
        load_split(root, 'fixture', 'nested', settings.dataset_receipt_sha256, samples)


def test_formal_cli_modes_and_visualization_parity(fixture_project, monkeypatch, capsys):
    from core.cli import main
    cfg, _, _ = fixture_project
    root = Path(cfg['project_root'])
    monkeypatch.setattr('core.cli.resolve', lambda *args, **kwargs: cfg)
    original_import = builtins.__import__

    def no_plots(name, *args, **kwargs):
        if name == 'matplotlib' or name.startswith('matplotlib.'):
            raise AssertionError('Visualization disabled must not import matplotlib')
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, '__import__', no_plots)
        for mode in ('dry-run', 'check-data', 'fit'):
            path = config_file(cfg, mode, mode=mode)
            main(['difficulty-ridge-holdout', mode, '--workflow-config', str(path)])
            capsys.readouterr()
            assert verify(root, 'difficulty-evaluation', mode)['verified']
            output = root / 'difficulty/evaluations' / mode
            assert read_json(output/'workflow.json')['mode'] == mode
            assert not (output/'plots').exists() and not (output/'index.html').exists()
            assert (output/'predictions.jsonl').exists() == (mode == 'fit')
            attempt = root / 'difficulty/checkpoints' / child_id(mode, 'ridge')
            assert (attempt/'models').exists() == (mode == 'fit')
    path = config_file(cfg, 'plot', enabled=True)
    run(cfg, 'plot', workflow_config=path)
    on, off = (root / 'difficulty/evaluations' / name for name in ('plot', 'fit'))
    assert read_json(on/'metrics.json') == read_json(off/'metrics.json')
    assert read_json(on/'coverage.json') == read_json(off/'coverage.json')
    coverage = read_json(on/'coverage.json')
    assert len(list((on/'plots').glob('*.svg'))) == len(coverage['expected_rollout_ids'])*2
    assert verify(root, 'difficulty-evaluation', 'plot')['verified']
    for target in coverage['targets']:
        before = read_jsonl(off/'predictions.jsonl', target)
        after = read_jsonl(on/'predictions.jsonl', target)
        assert [(r['sample_id'], r['prediction_raw']) for r in before] == [(r['sample_id'], r['prediction_raw']) for r in after]
        attempt_dir = root/'difficulty/checkpoints'/child_id('fit', 'ridge')
        attempt = read_json(attempt_dir/'attempt.json')
        split = read_json(root/'difficulty/datasets/fixture/splits'/attempt['split_id']/'split.json')
        samples = [json.loads(line) for line in (root/'difficulty/datasets/fixture/samples.jsonl').read_text().splitlines()]
        positions = {s['sample_id']: i for i, s in enumerate(samples)}
        matrix = np.load(Path(read_json(attempt_dir/'data-check.json')['cache'])/f"layer-{before[0]['layer']:02d}.f32.npy").astype(float)
        candidate = load_model(attempt_dir/read_json(attempt_dir/'selected.json')['targets'][target]['model'])
        refit = load_model(attempt_dir/before[0]['model'])
        assert np.allclose(candidate['feature_mean'], matrix[[positions[s] for s in split['sample_ids']['train']]].mean(0))
        fit_ids = split['sample_ids']['train']+split['sample_ids']['validation']
        assert np.allclose(refit['feature_mean'], matrix[[positions[s] for s in fit_ids]].mean(0))
        replay = predict_model(refit, matrix[[positions[r['sample_id']] for r in before]])
        assert np.allclose(replay, [r['prediction_transformed'] for r in before], rtol=1e-12, atol=1e-12)
    with pytest.raises(FileExistsError):
        run(cfg, 'plot', workflow_config=path)


def read_jsonl(path, target):
    return [r for r in (json.loads(line) for line in path.read_text().splitlines()) if r['target'] == target]


def test_test_labels_do_not_select_or_fit_models(fixture_project):
    cfg, _, dataset = fixture_project
    path = config_file(cfg, 'before')
    run(cfg, 'before', workflow_config=path)
    root = Path(cfg['project_root'])
    output = root/'difficulty/evaluations/before'
    ids = set(read_json(output/'cohort.json')['sample_ids'])
    rows = [json.loads(line) for line in (dataset/'samples.jsonl').read_text().splitlines()]
    for row in rows:
        if row['sample_id'] in ids:
            row['remaining_reasoning_tokens_to_submission'] += 100000
    (dataset/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    receipt = read_json(dataset/'receipt.json')
    receipt['files']['samples.jsonl'] = sha256(dataset/'samples.jsonl')
    write_json(dataset/'receipt.json', receipt)
    run(cfg, 'after', workflow_config=path)
    a, b = (root/'difficulty/checkpoints'/child_id(name, 'ridge') for name in ('before', 'after'))
    sa, sb = (read_json(p/'selected.json')['targets'] for p in (a, b))
    assert {t: (r['layer'], r['alpha']) for t, r in sa.items()} == {t: (r['layer'], r['alpha']) for t, r in sb.items()}
    for first in (a/'models').rglob('*.npz'):
        second = b/first.relative_to(a)
        ma, mb = load_model(first), load_model(second)
        assert all(np.array_equal(ma[k], mb[k]) for k in ma)


def test_plot_failure_recovery_and_configuration_rejection(fixture_project, monkeypatch):
    cfg, _, _ = fixture_project
    root = Path(cfg['project_root'])
    path = config_file(cfg, 'recover', enabled=True)
    with monkeypatch.context() as context:
        def fail(*args, **kwargs):
            raise RuntimeError('injected plot failure')
        # Leave training plots working; fail only after sealed training.
        context.setattr('difficulty.evaluation.visualize.plot_trajectory', fail)
        with pytest.raises(RuntimeError, match='trajectory plots failed'):
            run(cfg, 'failed-plots', workflow_config=path)
    failed = root/'difficulty/evaluations/failed-plots'
    assert not (failed/'receipt.json').exists()
    assert (failed/'plot-manifest.json').exists() and (failed/'index.html').exists()
    attempt = child_id('failed-plots', 'ridge')
    assert verify(root, 'difficulty-attempt', attempt)['verified']
    result = run(cfg, 'recovered', workflow_config=path, reuse_attempt_id=attempt)
    assert result['status'] == 'complete'
    assert verify(root, 'difficulty-evaluation', 'recovered')['verified']
    changed = config_file(cfg, 'changed', enabled=True, layers=[1])
    with pytest.raises(ValueError, match='effective training configuration mismatch'):
        run(cfg, 'bad-reuse', workflow_config=changed, reuse_attempt_id=attempt)
    assert not (root/'difficulty/evaluations/bad-reuse/receipt.json').exists()


def test_inverse_overflow_keeps_diagnostics_without_success_receipts(fixture_project, monkeypatch):
    cfg, _, _ = fixture_project
    import difficulty.training.ridge_probe as probe
    original = probe.inverse_target
    def overflow(values, target):
        if target.endswith('_log1p'):
            return np.full_like(values, np.inf)
        return original(values, target)
    monkeypatch.setattr(probe, 'inverse_target', overflow)
    path = config_file(cfg, 'overflow')
    result = run(cfg, 'overflow', workflow_config=path)
    root = Path(cfg['project_root'])
    assert result['status'] == 'partial'
    assert not (root/'difficulty/evaluations/overflow/receipt.json').exists()
    assert not (root/'difficulty/checkpoints'/child_id('overflow', 'ridge')/'receipt.json').exists()
    audit = read_json(root/'difficulty/evaluations/overflow/coverage.json')
    assert not audit['complete']
    assert audit['targets']['submission_remaining_reasoning_tokens_log1p']['failed'] > 0


def test_composed_dataset_manifest_fraction_plots_preserve_both_runs(fixture_project):
    from difficulty.datasets.compose import compose_datasets
    cfg, _, dataset = fixture_project
    root = Path(cfg['project_root'])
    rollouts = read_json(dataset/'rollouts.json')['rollouts']
    samples = [json.loads(line) for line in (dataset/'samples.jsonl').read_text().splitlines()]
    for part in (0, 1):
        directory = root/'difficulty/datasets'/f'batch{part}'
        directory.mkdir()
        chosen = [r for i, r in enumerate(rollouts) if i % 2 == part]
        for row in chosen:
            row.update(run_id=f'run{part}', source_run_id=f'run{part}', total_reasoning_tokens=100,
                       repetition_id='rep-000', benchmark='swebench')
        ids = {r['rollout_id'] for r in chosen}
        selected = [r for r in samples if r['rollout_id'] in ids]
        for row in selected:
            row.update(benchmark='swebench', remaining_steps_fraction=row['remaining_steps_to_submission']/3,
                       remaining_reasoning_tokens_fraction=row['remaining_reasoning_tokens_to_submission']/100)
        write_json(directory/'rollouts.json', {'rollouts': chosen})
        write_json(directory/'dataset.json', {'label_normalization': 'trajectory_fraction'})
        (directory/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
        write_json(directory/'receipt.json', {'state': 'complete', 'source_run_id': f'run{part}',
            'files': {name: sha256(directory/name) for name in ('rollouts.json', 'samples.jsonl', 'dataset.json')}})
    compose_datasets(root, 'combined', ['batch0', 'batch1'])
    manifest = root/'tasks.json'
    write_json(manifest, {'task_assignments': [{'task_key': r['task_key'], 'instance_id': r['instance_id'],
        'partition': 'test' if r['rollout_id'] in {'r0', 'r1'} else 'train'} for r in rollouts]})
    path = root/'composed.yaml'
    path.write_text(yaml.safe_dump({'schema_version': 1, 'data': {'dataset_id': 'combined',
        'targets': ['submission_remaining_steps_fraction', 'submission_remaining_reasoning_tokens_fraction']},
        'outer_split': {'strategy': 'task_manifest_v1', 'task_partitions': str(manifest)},
        'model': {'kind': 'ridge', 'ridge': {'layers': [0], 'alphas': [1.0]}},
        'visualization': {'enabled': True, 'export_formats': ['svg']}}))
    result = run(cfg, 'composed', workflow_config=path)
    assert result['status'] == 'complete'
    output = root/'difficulty/evaluations/composed'
    cohort = read_json(output/'cohort.json')
    assert {r['source_run_id'] for r in cohort['rollouts']} == {'run0', 'run1'}
    assert {r['rollout_id'] for r in cohort['rollouts']} == {'r0', 'r1'}
    assert len(list((output/'plots').glob('*.svg'))) == 4
    assert verify(root, 'difficulty-evaluation', 'composed')['verified']


def test_bad_expected_receipt_fails_with_diagnostic(fixture_project):
    cfg, _, _ = fixture_project
    path = config_file(cfg, 'bad-source', mode='dry-run')
    value = yaml.safe_load(path.read_text())
    value['data']['expected_receipt_sha256'] = '0'*64
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='receipt SHA mismatch'):
        run(cfg, 'bad-source', workflow_config=path)
    output = Path(cfg['project_root'])/'difficulty/evaluations/bad-source'
    assert (output/'report.md').exists() and not (output/'receipt.json').exists()
