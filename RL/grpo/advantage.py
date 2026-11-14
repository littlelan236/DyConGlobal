"""Deterministic model-free score-to-advantage mapping; lower score is better."""
from dataclasses import dataclass
import math
import statistics

ADVANTAGE_VERSION = "historical_topk_zero_sum_v1"
CONSENSUS_ADVANTAGE_VERSION = "consensus_top_half_zero_sum_v1"


def validate_retained(retained, n):
    if not isinstance(retained, (list, tuple)) or len(retained) != n or any(
            type(flag) is not bool for flag in retained):
        raise ValueError("Consensus advantages require one explicit boolean retained flag per branch")

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
                       boundary_tie_sigma=0.05, advantage_policy=ADVANTAGE_VERSION, retained=None):
    if advantage_policy not in {ADVANTAGE_VERSION, CONSENSUS_ADVANTAGE_VERSION}:
        raise ValueError("Unknown advantage policy")
    consensus = advantage_policy == CONSENSUS_ADVANTAGE_VERSION
    if consensus:
        if top_k is not None or top_k_fraction != 0.5 or boundary_tie_sigma != 0:
            raise ValueError("Consensus top-half requires top_k=null, top_k_fraction=0.5 and boundary_tie_sigma=0")
        validate_retained(retained, len(scores))
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
            {"version": advantage_policy, "status": "skipped", "skip_reason": "missing_or_nonfinite_score"})
    order = list(sorted(range(n), key=lambda i: (values[i], i)))
    ranks = [sum(s < x for s in values) for x in values]
    center = sum(x / n for x in values)
    sigma = math.hypot(*(x - center for x in values)) / math.sqrt(n)
    gap = values[order[k]] - values[order[k - 1]]
    info = {"version": advantage_policy, "population_std": sigma, "boundary_gap": gap}
    reason = "all_scores_equal" if values[order[0]] == values[order[-1]] else (
        "top_k_boundary_tie" if boundary_tie_sigma > 0 and gap <= boundary_tie_sigma * sigma else None)
    if reason:
        return AdvantageResult([], order, ranks, 0, 0, 0.0, k,
            {**info, "status": "skipped", "skip_reason": reason})
    positive = [i for i in order[:k] if retained[i]] if consensus else order[:k]
    m = len(positive)
    if consensus:
        info.update(ranking_scope="all_branches", ranking_cutoff=k,
                    consensus_field="retained", retained_count=sum(retained),
                    positive_indices=positive, tie_break="frozen_native_branch_order")
    if not positive:
        return AdvantageResult([], order, ranks, 0, 0, 0.0, k,
            {**info, "status": "skipped", "skip_reason": "no_consensus_in_top_half"})
    boundary = values[order[k - 1]]
    scale = max(boundary - values[order[0]], score_scale_floor * sigma, 1e-6)
    if not math.isfinite(scale) or not math.isfinite(sigma):
        raise ValueError("Nonfinite advantage scale")
    weights = [.5 + .5 * (boundary - values[i]) / scale for i in positive]
    mean_weight = statistics.fmean(weights)
    result = [-m / (n - m)] * n
    for i, weight in zip(positive, weights):
        result[i] = weight / mean_weight
    return AdvantageResult(result, order, ranks, m, n - m, math.fsum(result), k,
        {**info, "status": "ok", "scale": scale})
