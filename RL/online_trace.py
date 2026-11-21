"""Deterministic training views of immutable online rollout rows."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math


def build_training_traces(rows, masses, *, max_input_tokens=32768):
    """Merge exact append-only baseline inputs, preserving per-token mass."""
    if len(rows) != len(masses) or max_input_tokens < 1:
        raise ValueError("invalid trace builder arguments")
    groups = {}
    units = []
    for row, mass in zip(rows, masses):
        if row.source != "baseline":
            units.append(([row], [mass], "dycon"))
        else:
            if not row.trajectory_id or row.agent_step is None:
                raise ValueError("baseline trace requires trajectory and step identities")
            groups.setdefault((row.task_id, row.trajectory_id), []).append((row, mass))
    reasons = Counter()
    for items in groups.values():
        items.sort(key=lambda item: item[0].agent_step)
        if len({r.agent_step for r, _ in items}) != len(items):
            raise ValueError("duplicate agent step")
        chain, weights = [], []
        for row, mass in items:
            if not math.isfinite(mass) or mass < 0:
                raise ValueError("invalid token mass")
            if row.supervision_positions is not None:
                raise ValueError("cannot merge an already derived trace")
            if (not row.completion_token_ids
                    or row.input_token_ids[-len(row.completion_token_ids):] != row.completion_token_ids
                    or len(row.rollout_logprobs) != len(row.completion_token_ids)):
                raise ValueError("invalid baseline token evidence")
            reason = None
            if chain:
                previous = chain[-1]
                prefix_length = len(previous.input_token_ids)
                if (row.advantage != previous.advantage or row.resolved != previous.resolved
                        or row.group_id != previous.group_id
                        or row.trajectory_token_count != previous.trajectory_token_count
                        or (row.kl_reference_logprobs is None) != (previous.kl_reference_logprobs is None)):
                    raise ValueError("inconsistent trajectory evidence")
                if not math.isclose(mass / len(row.completion_token_ids),
                                    weights[-1] / len(previous.completion_token_ids),
                                    rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("trajectory token masses differ")
                if (prefix_length > len(row.input_token_ids) - len(row.completion_token_ids)
                        or row.input_token_ids[:prefix_length] != previous.input_token_ids):
                    reason = "prefix_break"
                elif len(row.input_token_ids) > max_input_tokens:
                    reason = "length_budget"
            if reason:
                units.append((chain, weights, reason))
                reasons[reason] += 1
                chain, weights = [], []
            chain.append(row)
            weights.append(mass)
        units.append((chain, weights, "trajectory_end"))
    order = {row.row_id: i for i, row in enumerate(rows)}
    if len(order) != len(rows):
        raise ValueError("duplicate row identity")
    units.sort(key=lambda unit: order[unit[0][0].row_id])
    result, coefficients, manifest = [], [], []
    for chain, weights, reason in units:
        first, last = chain[0], chain[-1]
        positions = tuple(p for row in chain for p in range(
            len(row.input_token_ids) - len(row.completion_token_ids), len(row.input_token_ids)))
        if first.source == "dycon":
            merged = first
        else:
            merged = replace(first, input_token_ids=last.input_token_ids,
                completion_token_ids=tuple(t for r in chain for t in r.completion_token_ids),
                rollout_logprobs=tuple(t for r in chain for t in r.rollout_logprobs),
                kl_reference_logprobs=None if first.kl_reference_logprobs is None else
                    tuple(t for r in chain for t in r.kl_reference_logprobs),
                supervision_positions=positions)
        result.append(merged)
        coefficients.append(sum(weights))
        manifest.append({"rows": [r.row_id for r in chain], "reason": reason,
                         "input_tokens": len(last.input_token_ids),
                         "supervised_tokens": len(merged.completion_token_ids),
                         "mass": sum(weights),
                         "evidence_sha256": hashlib.sha256(json.dumps({
                             "input": merged.input_token_ids,
                             "positions": merged.supervision_positions,
                             "rollout_logprobs": merged.rollout_logprobs,
                             "kl_reference": merged.kl_reference_logprobs,
                             "advantage": merged.advantage,
                             "task": merged.task_id, "trajectory": merged.trajectory_id
                         }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return tuple(result), coefficients, {
        "version": "strict_token_prefix_v1", "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "max_input_tokens": max_input_tokens, "original_rows": len(rows),
        "training_units": len(result), "baseline_traces": len(result) - sum(r.source == "dycon" for r in rows),
        "original_input_tokens": sum(len(r.input_token_ids) for r in rows),
        "training_input_tokens": sum(len(r.input_token_ids) for r in result),
        "break_reasons": dict(reasons), "units": manifest}
