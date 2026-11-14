"""Detached K3 cap comparisons; importing the field schema does not load Torch."""

K3_CAPS = (1, 5, 10)
K3_SCALARS = ('k3_diagnostic_tokens', 'k3_penalty_max') + tuple(
    f'k3_cap_{cap}_{suffix}'
    for cap in K3_CAPS
    for suffix in ('mean', 'exceeded_tokens', 'exceeded_fraction')
)


def token_statistics(k3, mask, rejected, group_indices):
    """Compare caps after ratio guard, preserving original loss denominators.

    Exceedance fractions count retained tokens equally. Candidate means use
    token-within-branch, branch-within-group, then equal group weighting.
    Nothing returned here participates in the objective's autograd graph.
    """
    import torch

    with torch.no_grad():
        retained = mask & ~rejected
        values = k3.detach().masked_fill(~retained, 0)
        count = int(retained.sum())
        counts = mask.sum(-1)
        result = {'k3_diagnostic_tokens': count, 'k3_penalty_max': float(values.max())}
        for cap in K3_CAPS:
            branch_means = values.clamp(max=cap).sum(-1) / counts
            mean = torch.stack([branch_means[indices].mean() for indices in group_indices]).mean()
            exceeded = int(((values > cap) & retained).sum())
            result.update({f'k3_cap_{cap}_mean': float(mean),
                           f'k3_cap_{cap}_exceeded_tokens': exceeded,
                           f'k3_cap_{cap}_exceeded_fraction': exceeded / count if count else 0.})
        return result


def summarize_statistics(groups):
    """Merge branch microbatches without token-weighting their objective means."""
    rows = [row for group in groups for row in group]
    if not rows or not all(all(key in row for key in K3_SCALARS) for row in rows):
        # Legacy/mixed producers provide no complete comparison; never invent zeros.
        return {}
    count = sum(row['k3_diagnostic_tokens'] for row in rows)
    result = {'k3_diagnostic_tokens': count,
              'k3_penalty_max': max(row['k3_penalty_max'] for row in rows)}
    for cap in K3_CAPS:
        key = f'k3_cap_{cap}_mean'
        mean = sum(sum(row[key] for row in group) / len(group) for group in groups) / len(groups)
        exceeded = sum(row[f'k3_cap_{cap}_exceeded_tokens'] for row in rows)
        result.update({key: mean, f'k3_cap_{cap}_exceeded_tokens': exceeded,
                       f'k3_cap_{cap}_exceeded_fraction': exceeded / count if count else 0.})
    return result
