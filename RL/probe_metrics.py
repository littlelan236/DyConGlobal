"""Fixed-group preference statistics; token mean, then branch mean, then group mean."""
import math


def mean(values):
    return math.fsum(values) / len(values) if values else None


class TokenChanges:
    """Exact CPU token pooling, independent of branch/group-equal preference metrics."""

    def __init__(self):
        self.chunks = {name: [] for name in ("positive", "negative", "zero")}

    def add(self, changes, advantage):
        import torch
        values = changes.detach().to(device="cpu", dtype=torch.float64).clone()
        if values.ndim != 1 or not torch.isfinite(values).all() or not math.isfinite(advantage):
            raise ValueError("Token changes require finite 1D values and advantage")
        name = "positive" if advantage > 0 else "negative" if advantage < 0 else "zero"
        self.chunks[name].append(values)

    def extend(self, other):
        for name in self.chunks:
            self.chunks[name].extend(other.chunks[name])

    def summary(self):
        import torch
        def pool(chunks):
            return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)
        return {"version": 1, "reference": "training_anchor", "weighting": "pooled_target_tokens",
                "quantile_interpolation": "linear", "top_fraction_rounding": "ceil_min_1",
                "all": token_change_metrics(pool([x for xs in self.chunks.values() for x in xs])),
                **{name: token_change_metrics(pool(chunks)) for name, chunks in self.chunks.items()}}


def token_change_metrics(changes):
    """Compact distributions of d=log(p_current)-log(p_anchor); no raw token output."""
    import torch
    values = changes.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("Token changes require finite 1D values")
    count = values.numel()
    absolute = values.abs()
    levels = (0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1)
    def quantiles(tensor):
        estimates = (torch.quantile(tensor, torch.tensor(levels, dtype=torch.float64),
                                    interpolation="linear").tolist() if count else [None] * len(levels))
        return {f"p{round(q * 100):02d}": value for q, value in zip(levels, estimates)}
    sorted_absolute = absolute.sort(descending=True).values
    total = float(absolute.sum())
    concentration = {}
    for percent in (1, 5, 10):
        size = max(1, (count * percent + 99) // 100) if count else 0
        concentration[f"top_{percent}pct"] = {
            "token_count": size,
            "abs_change_share": float(sorted_absolute[:size].sum()) / total if total > 0 else None}
    relative = {}
    for percent in (10, 50):
        threshold = percent / 100
        # |exp(d)-1| > t iff d > log(1+t) or d < log(1-t).
        # Compare in log space to avoid overflow even for extreme finite d.
        increased = int((values > math.log1p(threshold)).sum())
        decreased = int((values < math.log1p(-threshold)).sum())
        relative[f"gt_{percent}pct"] = {
            "increase_fraction": increased / count if count else None,
            "decrease_fraction": decreased / count if count else None,
            "either_fraction": (increased + decreased) / count if count else None}
    return {"token_count": count,
            "logprob_change_mean": float(values.mean()) if count else None,
            "abs_logprob_change_mean": float(absolute.mean()) if count else None,
            "logprob_change_quantiles": quantiles(values),
            "abs_logprob_change_quantiles": quantiles(absolute),
            "top_abs_change": concentration, "relative_probability_change": relative}


def group_metrics(branches):
    positive = [b for b in branches if b["advantage"] > 0]
    negative = [b for b in branches if b["advantage"] < 0]
    output = {}
    for name, rows in (("positive", positive), ("negative", negative)):
        for field in ("logprob_mean", "reference_logprob_mean", "logprob_change"):
            output[name + "_" + field] = mean([b[field] for b in rows])
    comparable = bool(positive and negative)
    output["preference_gap"] = (output["positive_logprob_mean"] - output["negative_logprob_mean"]
                                if comparable else None)
    output["reference_preference_gap"] = (
        output["positive_reference_logprob_mean"] - output["negative_reference_logprob_mean"]
        if comparable else None)
    output["preference_gap_change"] = (output["preference_gap"] - output["reference_preference_gap"]
                                       if comparable else None)
    output["pairwise_preference_rate"] = mean([
        float(p["logprob_mean"] > n["logprob_mean"]) + .5 * (p["logprob_mean"] == n["logprob_mean"])
        for p in positive for n in negative])
    output["advantage_logprob_alignment"] = mean([b["advantage"] * b["logprob_change"] for b in branches])
    output["logprob_mean"] = mean([b["logprob_mean"] for b in branches])
    output["logprob_change"] = mean([b["logprob_change"] for b in branches])
    return output


def aggregate(groups):
    keys = groups[0]["metrics"]
    output = {k: mean([g["metrics"][k] for g in groups if g["metrics"][k] is not None]) for k in keys}
    gaps = [g["metrics"]["preference_gap_change"] for g in groups
            if g["metrics"]["preference_gap_change"] is not None]
    output["improved_group_fraction"] = mean([float(x > 0) for x in gaps])
    branches = [b for g in groups for b in g["branches"]]
    support = {"group_count": len(groups), "comparable_group_count": len(gaps),
               "positive_group_count": sum(g["metrics"]["positive_logprob_mean"] is not None for g in groups),
               "negative_group_count": sum(g["metrics"]["negative_logprob_mean"] is not None for g in groups),
               "branch_count": len(branches), "token_count": sum(b["token_count"] for b in branches),
               "positive_branch_count": sum(b["advantage"] > 0 for b in branches),
               "negative_branch_count": sum(b["advantage"] < 0 for b in branches)}
    return output, support


def compare(report, baseline):
    if (baseline.get("schema_version") != report["schema_version"]
            or baseline.get("kind") != report["kind"]
            or baseline.get("probe_identity") != report["probe_identity"]):
        raise ValueError("Baseline report has incompatible groups/tokens/advantages/anchor/scoring protocol")
    def delta(current, previous):
        return {key: value - previous[key] if value is not None and previous.get(key) is not None else None
                for key, value in current.items()}
    previous = {g["group_id"]: g for g in baseline["groups"]}
    return {"baseline_checkpoint": baseline["checkpoint"],
            "aggregate_delta": delta(report["aggregate"], baseline["aggregate"]),
            "group_delta": [{"group_id": g["group_id"],
                             "metrics": delta(g["metrics"], previous[g["group_id"]]["metrics"])}
                            for g in report["groups"]]}
