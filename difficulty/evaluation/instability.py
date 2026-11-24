"""Pure conditional score-consistency metrics; no model, filesystem or plotting IO."""

import itertools
import math
import statistics
from collections import defaultdict

import numpy as np

BASE_METRICS = ("mean_pair_gap", "mean_sample_std", "mean_range")
DENOMINATOR_TOLERANCE = 1e-12


def metric_names(multipliers, log1p):
    return [*BASE_METRICS, *([key for k in multipliers for key in
            (f"pair_rate_ge_{k}x", f"group_rate_ge_{k}x")] if log1p else [])]


def _mean(values):
    return statistics.fmean(values) if values else None


def evaluate_step(branches, *, view, log1p, multipliers):
    """Equal action-group weight within a step, with explicit raw/dedup views."""
    if view not in {"deduplicated", "sampling"}:
        raise ValueError("Unknown reasoning weighting view")
    contexts = {(r.get("decision_id"), r.get("parent_prefix_sha256")) for r in branches
                if r["status"] == "eligible" and r["identity_status"] == "available"}
    if len(contexts) > 1 or any(not decision or not prefix for decision, prefix in contexts):
        raise ValueError("A metric step must have one nonempty decision/parent-prefix identity")
    by_action = defaultdict(list)
    for row in branches:
        if row["status"] == "eligible" and row["identity_status"] == "available":
            score = row["score"]
            if not isinstance(score, (float, int)) or isinstance(score, bool) or not math.isfinite(score):
                raise ValueError("Identity-available eligible scores must be finite")
            if not row.get("action_hash") or not row.get("reasoning_hash"):
                raise ValueError("Available identity must include action and reasoning hashes")
            by_action[row["action_hash"]].append(row)
    units_by_action, groups = {}, []
    for action, candidates in sorted(by_action.items()):
        by_reason = defaultdict(list)
        for c in candidates:
            by_reason[c["reasoning_hash"]].append(c)
        if view == "deduplicated":
            units = [{"score": statistics.fmean(c["score"] for c in cs), "reasoning_hash": r}
                     for r, cs in sorted(by_reason.items())]
        else:
            units = [{"score": c["score"], "reasoning_hash": c["reasoning_hash"]} for c in candidates]
        units_by_action[action] = units
        values = [u["score"] for u in units]
        available = len(by_reason) >= 2
        gaps = [abs(a - b) for a, b in itertools.combinations(values, 2)] if available else []
        if any(not math.isfinite(gap) for gap in gaps):
            raise ValueError("Score difference overflow")
        group = {"action_hash": action, "action": candidates[0]["action"], "view": view,
                 "candidate_count": len(candidates), "distinct_reasonings": len(by_reason),
                 "unit_count": len(units), "available": available,
                 "branch_ids": [c["branch_id"] for c in candidates], "unit_scores": values,
                 "mean_pair_gap": _mean(gaps),
                 "sample_std": statistics.stdev(values) if available else None,
                 "range": max(values) - min(values) if available else None}
        if log1p:
            for k in multipliers:
                boundary = math.log(k)
                group[f"pair_rate_ge_{k}x"] = _mean([float(g >= boundary) for g in gaps])
                group[f"group_rate_ge_{k}x"] = float(group["range"] >= boundary) if available else None
        groups.append(group)
    supported = [g for g in groups if g["available"]]
    step = {"view": view, "eligible_actions": len(groups), "supported_groups": len(supported),
            "mean_pair_gap": _mean([g["mean_pair_gap"] for g in supported]),
            "mean_sample_std": _mean([g["sample_std"] for g in supported]),
            "mean_range": _mean([g["range"] for g in supported])}
    if log1p:
        for k in multipliers:
            for name in (f"pair_rate_ge_{k}x", f"group_rate_ge_{k}x"):
                step[name] = _mean([g[name] for g in supported])
    # Q uses distinct-reasoning pairs on both sides, including in the sampling view.
    same = []
    for units in units_by_action.values():
        gaps = [abs(a["score"] - b["score"]) for a, b in itertools.combinations(units, 2)
                if a["reasoning_hash"] != b["reasoning_hash"]]
        if gaps:
            same.append(statistics.fmean(gaps))
    different = []
    for left, right in itertools.combinations(units_by_action.values(), 2):
        gaps = [abs(a["score"] - b["score"]) for a in left for b in right
                if a["reasoning_hash"] != b["reasoning_hash"]]
        if gaps:
            if any(not math.isfinite(gap) for gap in gaps):
                raise ValueError("Score difference overflow")
            different.append(statistics.fmean(gaps))
    step["relative_same_gap"] = _mean(same) if different else None
    step["relative_different_gap"] = _mean(different) if same else None
    step["relative_ratio"] = (
        step["relative_same_gap"] / step["relative_different_gap"]
        if same and different and step["relative_different_gap"] > DENOMINATOR_TOLERANCE else None)
    return groups, step


def describe(values):
    if not values:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}
    return {"count": len(values), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "p95": float(np.quantile(values, .95)),
            "max": float(max(values))}


def _task_means(rows, keys):
    trajectories = defaultdict(list)
    for row in rows:
        if all(row.get(k) is not None for k in keys):
            trajectories[(row.get("task_key", row["task_id"]), row["repo"], row["rollout_id"])].append(row)
    tasks = defaultdict(list)
    for (task, repo, _), steps in sorted(trajectories.items()):
        tasks[(task, repo)].append([statistics.fmean(r[k] for r in steps) for k in keys])
    records = [{"task_id": task, "repo": repo, "values": np.mean(values, axis=0).tolist()}
               for (task, repo), values in sorted(tasks.items())]
    return records, len(trajectories), sum(len(x) for x in trajectories.values())


def _point(values, ratio):
    mean = np.mean(values, axis=0)
    if ratio:
        return float(mean[0] / mean[1]) if mean[1] > DENOMINATOR_TOLERANCE else None
    return float(mean[0])


def _bootstrap(records, count, seed, *, ratio=False, cluster="task"):
    buckets = defaultdict(list)
    for r in records:
        buckets[r["task_id"] if cluster == "task" else r["repo"]].append(r["values"])
    if count == 0 or len(buckets) < 2:
        return {"ci95": None, "clusters": len(buckets), "valid_resamples": 0}
    sums = np.asarray([np.sum(v, axis=0) for _, v in sorted(buckets.items())])
    sizes = np.asarray([len(v) for _, v in sorted(buckets.items())])
    draw = np.random.default_rng(seed).integers(len(sums), size=(count, len(sums)))
    means = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None]
    values = means[:, 0]
    if ratio:
        valid = means[:, 1] > DENOMINATOR_TOLERANCE
        values = means[valid, 0] / means[valid, 1]
    # Many undefined ratio replicates cannot yield a reliable conditional interval.
    ci = np.quantile(values, [.025, .975]).tolist() if len(values) >= .95 * count else None
    return {"ci95": ci, "clusters": len(buckets), "valid_resamples": len(values)}


def aggregate_metrics(rows, names, *, bootstrap_samples=2000, seed=42, require_repo=False):
    """Equal step -> rollout -> task means; bootstrap whole tasks/repositories."""
    report = {}
    for name in names:
        records, trajectories, steps = _task_means(rows, [name])
        report[name] = {
            "mean": _point([r["values"] for r in records], False) if records else None,
            "tasks": len(records), "rollouts": trajectories, "steps": steps,
            "task_bootstrap": _bootstrap(records, bootstrap_samples, seed),
            "repo_bootstrap": _bootstrap(
                [r for r in records if r["repo"]] if require_repo else records,
                bootstrap_samples, seed, cluster="repo")}
    return report


def aggregate_steps(rows, names, *, bootstrap_samples=2000, seed=42):
    """Aggregate standalone metrics and the paired relative-gap metric."""
    report = aggregate_metrics(rows, names, bootstrap_samples=bootstrap_samples, seed=seed)
    keys = ["relative_same_gap", "relative_different_gap"]
    records, trajectories, steps = _task_means(rows, keys)
    means = np.mean([r["values"] for r in records], axis=0) if records else [None, None]
    report["relative"] = {
        "same_gap": means[0], "different_gap": means[1],
        "ratio": _point([r["values"] for r in records], True) if records else None,
        "tasks": len(records), "rollouts": trajectories, "mixed_steps": steps,
        "zero_denominator_steps": sum(r.get(keys[1]) is not None
                                      and r[keys[1]] <= DENOMINATOR_TOLERANCE for r in rows),
        "task_bootstrap": _bootstrap(records, bootstrap_samples, seed, ratio=True),
        "repo_bootstrap": _bootstrap(records, bootstrap_samples, seed, ratio=True, cluster="repo")}
    return report


def summarize_groups(groups):
    supported = [r for r in groups if r["available"]]
    def distribution(rs):
        return {name: describe([r[name] for r in rs]) for name in ["mean_pair_gap", "sample_std", "range"]}
    return {"action_groups": len(groups), "supported_groups": len(supported),
            "pooled_descriptive": distribution(supported),
            "by_unit_count": {str(n): distribution([r for r in supported if r["unit_count"] == n])
                              for n in sorted({r["unit_count"] for r in supported})}}
