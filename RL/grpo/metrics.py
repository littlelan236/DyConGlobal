"""Small, receipt-friendly objective metrics."""

from .k3_diagnostics import summarize_statistics


def summarize(results):
    if not results:
        return {"group_count": 0, "branch_count": 0, "token_count": 0}
    keys = ("ratio_mean", "ratio_min", "ratio_max", "clip_fraction", "k3_penalty_mean")
    output = {key: sum(float(r.get(key, 0.0)) for r in results) / len(results) for key in keys}
    for key in ("positive_ratio_mean", "negative_ratio_mean"):
        values = [r[key] for r in results if r.get(key) is not None]
        output[key] = sum(values) / len(values) if values else None
    output.update(group_count=len(results), branch_count=sum(r.get("branch_count", 0) for r in results),
                  token_count=sum(r.get("token_count", 0) for r in results))
    return output


def summarize_update(groups):
    """Branch means, then group means; quantiles/min/max use actual token ratios."""
    import torch
    rows = [r for group in groups for r in group]
    output = {}
    for key in ("ratio_mean", "k3_penalty_mean", "clip_fraction", "policy_loss"):
        output[key] = sum(sum(float(r.get(key, 0.)) for r in g) / len(g) for g in groups) / len(groups)
    for key in ("positive_logprob_change", "negative_logprob_change"):
        selected = [[r[key] for r in g if r.get(key) is not None] for g in groups]
        means = [sum(g) / len(g) for g in selected if g]
        output[key] = sum(means) / len(means) if means else None
    for key in ("positive_ratio_mean", "negative_ratio_mean"):
        selected = [[r[key] for r in g if r.get(key) is not None] for g in groups]
        means = [sum(values) / len(values) for values in selected if values]
        output[key] = sum(means) / len(means) if means else None
    excluded = sum(r.get("ratio_guard_excluded_tokens", 0) for r in rows)
    eligible = sum(r.get("ratio_guard_eligible_tokens", 0) for r in rows)
    output.update(ratio_guard_enabled=rows[0].get("ratio_guard_enabled", False),
                  ratio_guard_threshold=rows[0].get("ratio_guard_threshold", 1e-3), ratio_guard_excluded_tokens=excluded,
                  ratio_guard_eligible_tokens=eligible,
                  ratio_guard_excluded_fraction=excluded / eligible if eligible else 0.)
    ratios = [r["_ratio_values"] for r in rows if "_ratio_values" in r]
    if ratios:
        values = torch.cat(ratios).float()
        output.update(ratio_min=float(values.min()), ratio_max=float(values.max()),
                      ratio_quantiles=torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.])).tolist())
    output.update(positive_branches=sum(r.get("positive_branches", 0) for r in rows),
                  negative_branches=sum(r.get("negative_branches", 0) for r in rows))
    output.update(summarize_statistics(groups))
    return output


def read_metric_history(path):
    import json
    from pathlib import Path
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
