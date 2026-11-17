"""Detached learning diagnostics; no changes to the objective or its denominators."""
import math


def token_signals(log_ratio, ratio, advantages, mask, rejected, epsilon):
    import torch
    with torch.no_grad():
        d, r, a = log_ratio.detach(), ratio.detach(), advantages.detach()
        positive, negative = a[:, None] > 0, a[:, None] < 0
        saturated = ((positive & (r > 1 + epsilon)) | (negative & (r < 1 - epsilon))) & ~rejected
        active = mask & ~rejected & ~saturated & (positive | negative)
        rows = []
        for i in range(len(a)):
            selected = mask[i]
            rows.append({"advantage": float(a[i]), "tokens": int(selected.sum()),
                "logprob_change": float(d[i][selected].mean()),
                "active": int(active[i][selected].sum()),
                "saturated": int(saturated[i][selected].sum()),
                "excluded": int(rejected[i][selected].sum()),
                "ratio_low": int((r[i][selected] < 1 - epsilon).sum()),
                "ratio_high": int((r[i][selected] > 1 + epsilon).sum())})
        return {"_signal_branches": rows, "_log_ratio_values": d[mask].float().cpu()}


def summarize_signals(groups):
    """Merge complete group contributions, including uneven rank assignments.

    Objective fractions use token -> branch -> group means. Sign fractions and
    tail quantiles explicitly use token weights. Missing legacy fields stay absent.
    """
    import torch
    if not groups or any("_signal_branches" not in r for g in groups for r in g):
        return {}
    branches = [[b for row in group for b in row["_signal_branches"]] for group in groups]
    flat = [b for group in branches for b in group]
    mean = lambda values: math.fsum(values) / len(values)
    macro = lambda fn: mean([mean([fn(b) for b in group]) for group in branches])
    result = {"advantage_logprob_alignment": macro(lambda b: b["advantage"] * b["logprob_change"])}
    gaps = []
    for group in branches:
        pos = [b["logprob_change"] for b in group if b["advantage"] > 0]
        neg = [b["logprob_change"] for b in group if b["advantage"] < 0]
        if pos and neg:
            gaps.append(mean(pos) - mean(neg))
    result.update(preference_gap=mean(gaps) if gaps else None,
                  preference_positive_group_fraction=mean([float(x > 0) for x in gaps]) if gaps else None,
                  preference_group_count=len(gaps))
    magnitude = macro(lambda b: abs(b["advantage"]))
    for source, name in (("active", "policy_active"), ("saturated", "policy_saturated"),
                         ("excluded", "ratio_guard_excluded")):
        result[name + "_objective_fraction"] = macro(lambda b: b[source] / b["tokens"])
        result[name + "_advantage_weighted_fraction"] = (
            macro(lambda b: abs(b["advantage"]) * b[source] / b["tokens"]) / magnitude if magnitude else None)
        for sign, predicate in (("positive", lambda a: a > 0), ("negative", lambda a: a < 0)):
            selected = [b for b in flat if predicate(b["advantage"])]
            count = sum(b["tokens"] for b in selected)
            result[name + "_" + sign + "_token_fraction"] = sum(b[source] for b in selected) / count if count else None
    count = sum(b["tokens"] for b in flat)
    result.update(policy_active_tokens=sum(b["active"] for b in flat),
                  positive_advantage_tokens=sum(b["tokens"] for b in flat if b["advantage"] > 0),
                  negative_advantage_tokens=sum(b["tokens"] for b in flat if b["advantage"] < 0),
                  fully_guarded_branches=sum(b["excluded"] == b["tokens"] for b in flat),
                  fully_guarded_groups=sum(all(b["excluded"] == b["tokens"] for b in g) for g in branches),
                  ratio_low_token_fraction=sum(b["ratio_low"] for b in flat) / count,
                  ratio_high_token_fraction=sum(b["ratio_high"] for b in flat) / count)
    values = torch.cat([r["_log_ratio_values"] for g in groups for r in g])
    quantiles = torch.quantile(values, torch.tensor([.01, .5, .99]))
    result.update(log_ratio_min=float(values.min()), log_ratio_max=float(values.max()),
                  **{name: float(v) for name, v in zip(("log_ratio_p01", "log_ratio_p50", "log_ratio_p99"), quantiles)})
    return result


def throughput(metrics):
    """End-to-end update time, including loading/retries and peer wait."""
    seconds = metrics.get("seconds")
    if not seconds or seconds <= 0:
        return {}
    result = {"updates_per_second": 1 / seconds}
    if "failed_attempt_seconds" in metrics:
        result["retry_time_fraction"] = metrics["failed_attempt_seconds"] / seconds
    for source, target in (("group_count", "groups_per_second"), ("branch_count", "branches_per_second"),
                           ("token_count", "completion_tokens_per_second"),
                           ("policy_active_tokens", "policy_active_tokens_per_second")):
        if source in metrics:
            result[target] = metrics[source] / seconds
    return result
