"""Numeric step coordinates, sparse fixed-group plotting and JSON-only reuse."""
import argparse
import copy
from pathlib import Path

import pytest

from core.io import hash_json, read_json, sha256, write_json
from RL.probe_plot import plot_data, publish, run
from RL.probe_schedule import positions


@pytest.fixture
def prepared(tmp_path):
    pytest.importorskip('torch')
    from tests.test_rl_probe import prepared as fixture
    return fixture.__wrapped__(tmp_path)


def sample():
    ids = [f'g{i}' for i in range(16)]
    contract = {'input': {'eligible_group_ids': ids, 'group_batch_size': 2, 'max_epochs': 1}}
    selected = ['g14', 'g0', 'g6', 'g1']
    schedule = positions(contract, {'global_step': 4, 'group_cursor': 8}, selected)
    identity = {'groups': [{'group_id': gid} for gid in selected], 'protocol': {}}
    report = {'schema_version': 1, 'kind': 'fixed_grpo_preference', 'group_ids': selected,
              'checkpoint': {'global_step': 4}, 'probe_identity': identity,
              'probe_sha256': hash_json(identity), 'groups': [
                  {'group_id': gid, 'metrics': {'positive_logprob_change': .02,
                                              'negative_logprob_change': -.01,
                                              'preference_gap_change': .03}}
                  for gid in selected], 'training_schedule': schedule}
    return report, schedule


def test_sparse_groups_use_numeric_step_spacing_and_preserve_same_step():
    report, schedule = sample()
    data = plot_data(report, schedule)
    assert [g['group_id'] for g in data['groups']] == ['g0', 'g1', 'g6']
    assert [g['x'] for g in data['groups']] == [1, 1, 4]
    assert data['omitted_groups'] == [{'group_id': 'g14', 'status_at_checkpoint': 'not_yet_trained'}]
    complete = plot_data(report, schedule, include_untrained=True)
    assert [g['x'] for g in complete['groups']] == [1, 1, 4, 8]
    assert complete['groups'][-1]['first_training_step'] is None


def test_segment_tails_use_actual_windows_not_global_ordinal_division():
    ids = [f'g{i}' for i in range(7)]
    contract = {'input': {'eligible_group_ids': ids, 'group_batch_size': 2, 'max_epochs': 1}}
    segments = [{'group_ids': ids[:3], 'group_batch_size': 2},
                {'group_ids': ids[3:], 'group_batch_size': 3}]
    result = positions(contract, {'global_step': 3, 'group_cursor': 6}, ids, segments=segments)
    assert [g['scheduled_steps'] for g in result['groups']] == [[1], [1], [2], [3], [3], [3], [4]]
    with pytest.raises(ValueError, match='cursor'):
        positions(contract, {'global_step': 3, 'group_cursor': 5}, ids, segments=segments)


def test_two_pass_schedule_distinguishes_partial_from_fully_trained(tmp_path):
    pytest.importorskip('torch')
    from tests.test_grpo_execution import cfg32
    from RL.grpo.streaming import create_plan
    from RL.grpo.execution import cursor
    plan = create_plan(cfg32(tmp_path, n=9, streaming=True))
    execution = plan['proof']['execution_plan']
    boundary = len(execution['windows']) // 2
    state = {**cursor(execution, execution['windows'][boundary - 1]['end']), 'global_step': boundary}
    ids = plan['proof']['eligible_group_ids']
    coords = positions({'input': plan['proof']}, state, ids, execution_plan=execution)
    assert all(g['status_at_checkpoint'] == 'partially_trained' for g in coords['groups'])
    assert all(len(g['scheduled_steps']) == 2 for g in coords['groups'])
    last = {**cursor(execution, len(execution['chunks'])), 'global_step': len(execution['windows'])}
    coords = positions({'input': plan['proof']}, last, ids, execution_plan=execution)
    assert all(g['status_at_checkpoint'] == 'trained' for g in coords['groups'])


def test_chart_is_headless_numeric_and_no_interpolation(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    from RL.probe_plot import render
    import matplotlib.figure
    report, schedule = sample()
    data = plot_data(report, schedule, include_untrained=True)
    observed = {}
    original = matplotlib.figure.Figure.savefig
    def capture(self, *args, **kwargs):
        observed['axes'] = self.axes
        return original(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    payload = render(data, 'png')
    assert payload.startswith(b'\x89PNG\r\n\x1a\n')
    axes = observed['axes']
    assert len(axes) == 10
    assert all(ax.get_xscale() == 'linear' for ax in axes[:5])
    assert axes[0].collections[0].get_offsets()[:, 0].tolist() == [1, 1, 4]
    assert not any(len(line.get_xdata()) > 2 for ax in axes for line in ax.lines)
    assert 'Per-group results' in axes[5].get_title(loc='left')
    assert len(axes[5].tables) == 1
    assert 'Cohort preference summary' in axes[6].get_title(loc='left')
    assert 'not by the historical model' in axes[-1].texts[0].get_text()


def test_replot_uses_embedded_metadata_without_checkpoint_or_model(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    report, _ = sample()
    write_json(tmp_path / 'source.json', report)
    monkeypatch.setattr('RL.probe._checkpoint_context', lambda *a: pytest.fail('must not load checkpoint'))
    before = sha256(tmp_path / 'source.json')
    result = run({'project_root': str(tmp_path)}, report='source.json', output='chart.svg')
    assert result['group_count'] == 3 and result['omitted_group_count'] == 1
    assert '<svg' in (tmp_path / 'chart.svg').read_text()
    assert [g['x'] for g in read_json(tmp_path / 'chart.data.json')['groups']] == [1, 1, 4]
    assert before == sha256(tmp_path / 'source.json')
    with pytest.raises(FileExistsError):
        run({'project_root': str(tmp_path)}, report='source.json', output='chart.svg')


def test_historical_report_recovers_positions_from_verified_checkpoint(prepared, tmp_path, monkeypatch):
    from RL.probe_schedule import checkpoint_positions
    cfg, owner, cp, contract = prepared
    ids = contract['input']['eligible_group_ids'][:2]
    identity = {'groups': [{'group_id': gid} for gid in ids], 'protocol': {}}
    report = {'schema_version': 1, 'kind': 'fixed_grpo_preference', 'group_ids': ids,
              'checkpoint': {'path': str(cp), 'global_step': 1, 'receipt_sha256': sha256(cp / 'receipt.json')},
              'probe_identity': identity, 'probe_sha256': hash_json(identity),
              'groups': [{'group_id': gid, 'metrics': {}} for gid in ids]}
    write_json(tmp_path / 'legacy.json', report)
    # Rendering is orthogonal here; historical coordinates still use the real prepared fixture.
    monkeypatch.setattr('RL.probe_plot.render', lambda *a: b'chart')
    monkeypatch.setattr('RL.probe_plot._matplotlib_available', lambda: True)
    result = run({'project_root': str(tmp_path)}, report='legacy.json', output='historical.png')
    assert result['group_count'] >= 1
    assert read_json(tmp_path / 'historical.data.json')['groups'][0]['x'] == 1
    report['checkpoint']['receipt_sha256'] = 'wrong'
    write_json(tmp_path / 'bad.json', report)
    with pytest.raises(ValueError, match='receipt'):
        run({'project_root': str(tmp_path)}, report='bad.json', output='bad.png')


def test_invalid_coordinates_and_sealed_outputs_are_rejected(tmp_path):
    report, schedule = sample()
    wrong = copy.deepcopy(schedule)
    wrong['groups'][0]['scheduled_first_step'] = 7
    with pytest.raises(ValueError, match='steps'):
        plot_data(report, wrong)
    wrong = copy.deepcopy(schedule)
    wrong['groups'][0]['status_at_checkpoint'] = 'trained'
    with pytest.raises(ValueError, match='status'):
        plot_data(report, wrong)
    sealed = tmp_path / 'artifact'
    sealed.mkdir()
    write_json(sealed / 'receipt.json', {})
    with pytest.raises(ValueError, match='sealed'):
        publish(report, schedule, tmp_path, sealed / 'plot.png')
    with pytest.raises(ValueError, match='png'):
        publish(report, schedule, tmp_path, 'plot.html')
    assert not (sealed / 'plot.png').exists()


def test_plot_parser_and_probe_options():
    from RL.cli import register
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['lora-probe-plot', '--report', 'r.json', '--output', 'p.png', '--include-untrained'])
    assert args.include_untrained
    args = parser.parse_args(['lora-probe', '--checkpoint', 'cp', '--group-id', 'g0', '--output', 'r.json',
                              '--plot-output', 'p.png', '--plot-include-untrained', '--dry-run'])
    assert args.plot_output == 'p.png' and args.plot_include_untrained


def test_summary_weights_tokens_and_filters_scope_without_averaging_quantiles():
    report, schedule = sample()
    for i, group in enumerate(report['groups']):
        count = [1000, 1, 9, 10][i]
        delta = [.7, .1, -.1, 0][i]
        token = {'token_count': count, 'logprob_change_mean': delta,
                 'abs_logprob_change_mean': abs(delta), 'relative_probability_change': {
                     f'gt_{k}pct': {'either_fraction': float(i == 0), 'increase_fraction': float(i == 0),
                                    'decrease_fraction': 0.} for k in (10, 50)}}
        group['token_changes'] = {sign: copy.deepcopy(token) for sign in ('all', 'positive', 'negative')}
        group['branches'] = [{'token_count': count}]
        group['metrics']['positive_logprob_change'] = delta
    report['token_changes'] = {'all': {'sentinel': 'full report only'}}
    subset = plot_data(report, schedule)
    s = subset['summary']
    assert s['exact_pooled_token_changes'] is None
    assert s['cohorts'][0]['group_count'] == 3
    assert s['cohorts'][0]['metrics']['positive_logprob_change']['mean'] == pytest.approx(0.)
    assert s['cohorts'][0]['tokens']['all']['token_count'] == 20
    assert s['cohorts'][0]['tokens']['all']['logprob_change_mean'] == pytest.approx(-.04)
    assert s['cohorts'][0]['tokens']['all']['relative_probability_change']['gt_10pct']['either_fraction'] == 0
    complete = plot_data(report, schedule, include_untrained=True)['summary']
    assert complete['exact_pooled_token_changes'] == report['token_changes']
    assert complete['cohorts'][-1]['token_count'] == 1020
    assert complete['cohorts'][-1]['tokens']['all']['relative_probability_change']['gt_10pct']['either_fraction'] == pytest.approx(1000 / 1020)
    del report['groups'][1]['token_changes']
    assert plot_data(report, schedule)['summary']['cohorts'][0]['tokens']['all'] is None
