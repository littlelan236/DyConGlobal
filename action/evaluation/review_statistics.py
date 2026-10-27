"""Descriptive review-length summaries; unknown observations never become zero."""
from collections import Counter
from statistics import mean, median


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0, 'sum': None, 'mean': None, 'median': None,
                'p25': None, 'p75': None, 'p90': None, 'min': None, 'max': None}
    def percentile(p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {'n': len(values), 'sum': sum(values), 'mean': mean(values),
            'median': median(values), 'p25': percentile(.25), 'p75': percentile(.75),
            'p90': percentile(.9), 'min': values[0], 'max': values[-1]}


def group_summary(rows):
    known = [r for r in rows if r['review_steps'] is not None]
    clean = [r for r in known if r['status'] == 'estimated']
    late_mutation = [r for r in rows if 'unresolved_late_mutation' in r['unknown_reasons']]
    mid_review_late_mutation = [r for r in late_mutation if r['all_episode_review_steps'] > 0]
    steps = sum(r['assistant_steps'] for r in known)
    return {'submitted': len(rows), 'assistant_steps': sum(r['assistant_steps'] for r in rows),
            'estimated': len(known), 'unknown': len(rows) - len(known),
            'estimated_without_warnings': len(clean),
            'estimated_with_warnings': len(known) - len(clean),
            'coverage': len(known) / len(rows) if rows else None,
            'review_steps': distribution([r['review_steps'] for r in known]),
            'review_steps_permissive': distribution([r['review_steps_permissive'] for r in known]),
            'review_span': distribution([r['review_span'] for r in known]),
            'review_share_macro': mean([r['review_step_share'] for r in known]) if known else None,
            'review_share_micro': sum(r['review_steps'] for r in known) / steps if steps else None,
            'without_warnings_review_steps': distribution([r['review_steps'] for r in clean]),
            'all_episode_review_steps': sum(r['all_episode_review_steps'] for r in rows),
            'rework_after_review': sum(r['rework_after_review'] for r in rows),
            'unresolved_late_mutation_trajectories': len(late_mutation),
            'mid_review_before_unresolved_late_mutation': len(mid_review_late_mutation),
            'mid_review_before_unresolved_late_mutation_rate': (
                len(mid_review_late_mutation) / len(late_mutation) if late_mutation else None),
            'unknown_reason_counts': dict(Counter(x for r in rows for x in r['unknown_reasons'])),
            'warning_counts': dict(Counter(x for r in rows for x in r['warnings']))}


def summarize(rows):
    return {'pooled': group_summary(rows),
            'by_run': {run: group_summary([r for r in rows if r['source_run_id'] == run])
                       for run in sorted({r['source_run_id'] for r in rows})},
            'by_evaluation_group': {group: group_summary([r for r in rows if r['evaluation_group'] == group])
                                    for group in sorted({r['evaluation_group'] for r in rows})}}
