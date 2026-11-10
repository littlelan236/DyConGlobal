"""Deterministic model-free score-to-advantage mapping; lower score is better."""
from dataclasses import dataclass
import math
import statistics

ADVANTAGE_VERSION = "historical_topk_zero_sum_v1"

@dataclass(frozen=True)
class AdvantageResult:
    advantages: list[float]
    order: list[int]
    ranks: list[int]
    positive_count: int
    negative_count: int
    total: float
    top_k: int
    metadata: dict

def compute_advantages(scores, *, top_k=4, top_k_fraction=None, score_scale_floor=0.1,
                       boundary_tie_sigma=0.05):
    if (top_k is None) == (top_k_fraction is None):
        raise ValueError("Specify exactly one top-k policy")
    if (not math.isfinite(score_scale_floor) or score_scale_floor <= 0
            or not math.isfinite(boundary_tie_sigma) or boundary_tie_sigma < 0):
        raise ValueError("Invalid advantage scale or tie tolerance")
    n = len(scores)
    if n < 2:
        raise ValueError("At least two branches are required")
    if top_k_fraction is not None:
        if not math.isfinite(top_k_fraction) or not 0 < top_k_fraction < 1:
            raise ValueError("Invalid top_k_fraction")
        k = min(n - 1, max(1, math.ceil(n * top_k_fraction)))
    else:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("Invalid top_k")
        k = min(n - 1, max(1, math.ceil(n / 8))) if top_k == 4 and n != 32 else top_k
        if k >= n:
            raise ValueError("Explicit top_k must be smaller than group size")
    try:
        values = [float(x) for x in scores]
    except (TypeError, ValueError):
        values = []
    if len(values) != n or any(not math.isfinite(x) for x in values):
        return AdvantageResult([], [], [], 0, 0, 0.0, k,
            {"version": ADVANTAGE_VERSION, "status": "skipped", "skip_reason": "missing_or_nonfinite_score"})
    order = list(sorted(range(n), key=lambda i: (values[i], i)))
    ranks = [sum(s < x for s in values) for x in values]
    center = sum(x / n for x in values)
    sigma = math.hypot(*(x - center for x in values)) / math.sqrt(n)
    gap = values[order[k]] - values[order[k - 1]]
    info = {"version": ADVANTAGE_VERSION, "population_std": sigma, "boundary_gap": gap}
    reason = "all_scores_equal" if values[order[0]] == values[order[-1]] else (
        "top_k_boundary_tie" if gap <= boundary_tie_sigma * sigma else None)
    if reason:
        return AdvantageResult([], order, ranks, 0, 0, 0.0, k,
            {**info, "status": "skipped", "skip_reason": reason})
    boundary = values[order[k - 1]]
    scale = max(boundary - values[order[0]], score_scale_floor * sigma, 1e-6)
    if not math.isfinite(scale) or not math.isfinite(sigma):
        raise ValueError("Nonfinite advantage scale")
    weights = [.5 + .5 * (boundary - values[i]) / scale for i in order[:k]]
    mean_weight = statistics.fmean(weights)
    result = [-k / (n - k)] * n
    for i, weight in zip(order[:k], weights):
        result[i] = weight / mean_weight
    return AdvantageResult(result, order, ranks, k, n - k, math.fsum(result), k,
        {**info, "status": "ok", "scale": scale})
