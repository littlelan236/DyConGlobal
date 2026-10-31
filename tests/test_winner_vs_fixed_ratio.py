import copy
import pytest
from test_entropy_mapping_report import frozen_input, message
from action.evaluation.entropy_mappings import mapping_rows, mapping_curve_tables, mapping_plot_index, VERSION
from core.action_identity import canonical_action


def test_vs_fixed_compares_actions_not_difference_of_rates():
    record, _, _ = frozen_input()
    actions = [canonical_action(message(cmd))[1] for cmd in ('x', 'y', 'y', 'y', 'x', 'x', 'x', 'x')]
    record['entropy_mapping_input']['ranked_candidates'] = [
        {'branch_id': i, 'score': float(i), 'action': action} for i, action in enumerate(actions)]
    trajectories = [{'source_run_id': 'run', 'rollout_id': 'r', 'assistant_steps': 1}]
    rows = mapping_rows([record], trajectories)
    a = next(r for r in rows if r['retention_rule'] == 'effective_count_v1')
    fixed = next(r for r in rows if r['retention_rule'] == 'fixed_ratio_v1')
    assert fixed['keep_count'] == 4
    assert fixed['mapped_winner_action'] == canonical_action(message('y'))[1]
    assert a['mapped_winner_action'] == canonical_action(message('x'))[1]
    assert a['winner_changed_vs_fixed_ratio'] == 1.0
    assert fixed['winner_changed_vs_fixed_ratio'] == 0.0
    assert a['winner_changed_rate'] == 0.0
    _, curves = mapping_curve_tables(rows, trajectories, 3)
    observed = [r for r in curves if r['retention_rule'] == 'effective_count_v1'
                and r['metric'] == 'winner_changed_vs_fixed_ratio' and r['mean'] is not None]
    assert observed and all(r['mean'] == 1.0 for r in observed)


def test_vs_fixed_missing_winner_is_null():
    record, _, _ = frozen_input()
    for pool in ([], record['entropy_mapping_input']['ranked_candidates'][:1]):
        edge = copy.deepcopy(record)
        edge['entropy_mapping_input']['ranked_candidates'] = pool
        rows = mapping_rows([edge], [{'source_run_id': 'run', 'rollout_id': 'r', 'assistant_steps': 1}])
        assert all(r['winner_changed_vs_fixed_ratio'] is None for r in rows)


def test_exactly_three_vs_fixed_plots():
    analysis = {'entropy_mapping_comparison': {'version': VERSION}, 'formats': ['png']}
    entries = mapping_plot_index([{'source_run_id': 'run'}], analysis)['entries']
    comparisons = [e for e in entries if e['metric'] == 'winner_changed_vs_fixed_ratio']
    assert len(comparisons) == 3
    assert not any(e['retention_rule'] == 'fixed_ratio_v1' for e in comparisons)
