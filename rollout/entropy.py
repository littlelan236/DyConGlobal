"""Pure Shannon-nats retention rules; no model, GPU, or analysis dependencies."""

import math

RETENTION_RULES = ("effective_count_v1", "shannon_exp_keep_v1", "baseline_floor_v1",
                   "normalized_entropy_v1", "effective_count_linear_v1", "effective_count_quadratic_v1", "effective_count_type_linear_v1", "shannon_saturating_a25_v1", "effective_count_sigmoid_v1", "effective_count_sigmoid_4_v1", "effective_count_tilt_a1_v1", "effective_count_scale_a08_v1", "effective_count_sigmoid_c7_v1", "entropy_logistic_c18_a8_v1", "effective_width_logistic_c6_b2_v1", "effective_width_hill_c6_p8_v1", "effective_width_hill_c7_p4_v1", "effective_width_power_p2_v1")


RETENTION_RULES += ("effective_count_full_above_v1",)


def stable_ceil(value):
    """Absorb at most 16 machine ULPs at an integer boundary, then ceil."""
    nearest = round(value)
    if abs(value - nearest) <= 16 * math.ulp(max(1.0, abs(value))):
        return nearest
    return math.ceil(value)


def entropy_retention(counts, minimum=2, rule="shannon_exp_keep_v1", retention_ratio=0.5,
                      full_retention_threshold=10):
    """Return an auditable top-k budget from an empirical action histogram."""
    if rule not in RETENTION_RULES:
        raise ValueError("Unknown entropy retention rule: " + str(rule))
    if rule == "effective_count_full_above_v1" and (
            type(full_retention_threshold) is not int or full_retention_threshold < 1):
        raise ValueError("full_retention_threshold must be a positive integer")
    if type(minimum) is not int or minimum < 2:
        raise ValueError("min_retained_branches must be at least 2")
    if any(type(n) is not int or n < 1 for n in counts):
        raise ValueError("Action counts must be positive integers")
    if type(retention_ratio) not in (int, float) or not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be in (0, 1]")
    n = sum(counts)
    h = -math.fsum((count / n) * math.log(count / n) for count in counts) if n else None
    if h == 0:
        h = 0.0
    raw = None
    if n:
        if rule in {"effective_count_v1", "effective_count_full_above_v1"}:
            raw = math.exp(h)
        elif rule == "shannon_exp_keep_v1":
            raw = n - (n - minimum) * math.exp(-h)
        elif rule == "baseline_floor_v1":
            raw = n * (1 - (1 - retention_ratio) * math.exp(-h))
        elif rule == "normalized_entropy_v1":
            # Normalize by the maximum entropy of N observed branches.
            raw = minimum + (n - minimum) * (h / math.log(n) if n > 1 else 0.0)
        elif rule == "effective_count_linear_v1":
            # Linearly map effective action count exp(H) from 1 to N.
            raw = minimum + (n - minimum) * ((math.exp(h) - 1.0) / (n - 1.0) if n > 1 else 0.0)
        elif rule == "effective_count_quadratic_v1":
            # Compress low/intermediate effective counts; still reaches N at max H.
            fraction = (math.exp(h) - 1.0) / (n - 1.0) if n > 1 else 0.0
            raw = minimum + (n - minimum) * fraction * fraction
        elif rule == "effective_count_type_linear_v1":
            # Normalize effective action count by observed action types K.
            k = len(counts)
            fraction = (math.exp(h) - 1.0) / (k - 1.0) if k > 1 else 0.0
            raw = minimum + (n - minimum) * fraction
        elif rule == "shannon_saturating_a25_v1":
            raw = minimum + (n - minimum) * (1.0 - math.exp(-2.5 * h))
        elif rule == "effective_count_sigmoid_v1":
            effective = math.exp(h)
            center = math.sqrt(n)
            z = 1.5 * (effective - center)
            fraction = 1.0 / (1.0 + math.exp(-z))
            raw = minimum + (n - minimum) * fraction
        elif rule == "effective_count_sigmoid_4_v1":
            effective = math.exp(h)
            center = math.sqrt(n)
            z = 4.0 * (effective - center)
            fraction = 1.0 / (1.0 + math.exp(-z))
            raw = minimum + (n - minimum) * fraction
        elif rule == "effective_count_tilt_a1_v1":
            raw = math.exp(h + h * (h - 1.0))
        elif rule == "effective_count_scale_a08_v1":
            raw = math.exp(h) * (1.0 + 0.8 * h * h)
        elif rule == "effective_count_sigmoid_c7_v1":
            z = 4.0 * (math.exp(h) - 7.0)
            fraction = 1.0 / (1.0 + math.exp(-z))
            raw = minimum + (n - minimum) * fraction
        elif rule == "entropy_logistic_c18_a8_v1":
            z = 8.0 * (h - 1.8)
            fraction = 1.0 / (1.0 + math.exp(-z))
            raw = minimum + (n - minimum) * fraction
        elif rule == "effective_width_logistic_c6_b2_v1":
            effective = math.exp(h)
            z = 2.0 * (effective - 6.0)
            fraction = 1.0 / (1.0 + math.exp(-z))
            raw = minimum + (n - minimum) * fraction
        elif rule == "effective_width_hill_c6_p8_v1":
            effective = math.exp(h)
            ep = effective ** 8
            fraction = ep / (ep + 6.0 ** 8)
            raw = minimum + (n - minimum) * fraction
        elif rule == "effective_width_hill_c7_p4_v1":
            effective = math.exp(h)
            ep = effective ** 4
            fraction = ep / (ep + 7.0 ** 4)
            raw = minimum + (n - minimum) * fraction
        else:  # effective_width_power_p2_v1
            effective = math.exp(h)
            fraction = ((effective - 1.0) / (n - 1.0)) ** 2 if n > 1 else 0.0
            raw = minimum + (n - minimum) * fraction
    keep = min(n, max(minimum, stable_ceil(raw))) if n >= minimum else 0
    threshold_meta = {}
    if rule == "effective_count_full_above_v1":
        triggered = keep > full_retention_threshold
        threshold_meta = {"base_keep_count": keep,
                          "full_retention_threshold": full_retention_threshold,
                          "full_retention_triggered": triggered}
        if triggered:
            keep = n
    return {**threshold_meta, "definition": "shannon_nats", "stage": "post_loop_escape_pre_estimator_top_k",
            "retention_rule": rule, "numeric_policy": "integer_snap_16ulp_v1",
            "eligible_count": n, "action_types": len(counts),
            "group_sizes": sorted(counts, reverse=True), "entropy": h,
            "min_retained_branches": minimum, "raw_keep": raw, "keep_count": keep,
            "retention_ratio": retention_ratio if rule == "baseline_floor_v1" else None,
            "retention_ratio_used": rule == "baseline_floor_v1",
            "effective_retention_ratio": keep / n if n else None,
            "status": "available" if n >= minimum else "insufficient_branches" if n else "unavailable"}
