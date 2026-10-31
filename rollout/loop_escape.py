"""Attempt-local, per-slot action-first escape from cross-decision repetition."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from core.action_identity import ActionIdentityUnavailable, canonical_action
from core.io import hash_json

VERSION = "per_slot_action_then_reasoning_v1"


def fingerprints(response):
    """Use canonical trajectory tool actions; absent/empty reasoning is unknown."""
    choices = (response or {}).get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    try:
        action, _ = canonical_action(message)
    except (ActionIdentityUnavailable, TypeError, ValueError):
        action = None
    reasoning = message.get("reasoning_content")
    reason = hash_json({"reasoning_content": reasoning}) if isinstance(reasoning, str) and reasoning else None
    return {"action_fingerprint": action, "reasoning_fingerprint": reason}


@dataclass
class SlotHistory:
    count: int = 0
    action: dict = field(default_factory=dict)
    reasoning: dict = field(default_factory=dict)


@dataclass
class LoopEscapeHistory:
    """Owned by a single model/rollout attempt, never the serving broker."""

    decisions: int = 0
    slots: dict = field(default_factory=dict)

    def inspect(self, branch, current):
        history = self.slots.get(branch, SlotHistory())
        result = {**current, "history_count": history.count,
                  "history_available": history.count > 0}
        for name in ("action", "reasoning"):
            previous = getattr(history, name)
            fingerprint = current[name + "_fingerprint"]
            comparable = fingerprint is not None and bool(previous)
            result[name + "_history_available"] = bool(previous)
            result[name + "_comparison"] = (
                "repeated" if fingerprint in previous else "new") if comparable else "unknown"
            result[name + "_matched_decision_id"] = previous.get(fingerprint)
        return result

    def remember(self, decision):
        for row in decision["candidates"]:
            data = row["loop_escape"]
            if not data["output_available"]:
                continue
            slot = self.slots.setdefault(row["branch_id"], SlotHistory())
            slot.count += 1
            for name in ("action", "reasoning"):
                fingerprint = data[name + "_fingerprint"]
                if fingerprint is not None:
                    getattr(slot, name).setdefault(fingerprint, decision["decision_id"])
        self.decisions += 1


def filter_stage(rows, decision_index):
    """Unknown is never novelty or proof of repetition; retain it conservatively."""
    valid = [r for r in rows if r["loop_escape"]["output_available"]]
    if decision_index == 0:
        return "initial", [r["branch_id"] for r in valid]
    if any(r["loop_escape"]["action_comparison"] == "new" for r in valid):
        return "action", [r["branch_id"] for r in valid
                          if r["loop_escape"]["action_comparison"] != "repeated"]
    if valid and all(r["loop_escape"]["action_comparison"] == "repeated" for r in valid):
        return "reasoning", [r["branch_id"] for r in valid
                             if r["loop_escape"]["reasoning_comparison"] != "repeated"]
    return "unknown", [r["branch_id"] for r in valid]


def prepare_escape(records, formatted_ids, history):
    """Attach comparisons without changing selection eligibility or state yet."""
    for row in records:
        available = row["branch_id"] in formatted_ids
        current = fingerprints(row.get("response")) if available else {
            "action_fingerprint": None, "reasoning_fingerprint": None}
        row["loop_escape"] = {
            **history.inspect(row["branch_id"], current),
            "output_available": available,
            "status_before": row["status"], "reason_before": row["reason"],
            "selection_eligible_before": row.get("selection_eligible"),
            "excluded": False, "exclusion_reason": None,
        }
    stage, ids = filter_stage(records, history.decisions)
    before = sorted(formatted_ids)
    return {"enabled": True, "version": VERSION, "decision_index": history.decisions,
            "stage": stage, "candidate_ids_before": before,
            "proposed_candidate_ids": sorted(ids), "candidate_ids_after": before,
            "applied": False, "failure_reason": None}


def apply_escape(records, meta, failure_reason=None):
    """On exhaustion, keep actual outputs for the unchanged terminal detector."""
    if failure_reason is None and meta["candidate_ids_before"] and not meta["proposed_candidate_ids"]:
        failure_reason = "all_comparable_outputs_repeated"
    meta["failure_reason"] = failure_reason
    meta["applied"] = failure_reason is None
    meta["candidate_ids_after"] = (meta["proposed_candidate_ids"] if meta["applied"]
                                   else meta["candidate_ids_before"])
    for row in records:
        data = row["loop_escape"]
        data["excluded"] = (data["output_available"] and row["branch_id"] not in meta["candidate_ids_after"])
        if data["excluded"]:
            data["exclusion_reason"] = "repeated_" + meta["stage"]
            row.update(status="loop_excluded", reason="cross_step_loop")
            if "selection_eligible" in row:
                row["selection_eligible"] = False


def selection_failure(records, meta, mode, fallback_policy, consensus_config):
    """Check the original mode's feasibility before committing exclusions."""
    proposed = set(meta["proposed_candidate_ids"])
    if not proposed or proposed == set(meta["candidate_ids_before"]):
        return None
    eligible = [r for r in records if r["branch_id"] in proposed and r["loop_escape"]["status_before"] == "eligible"]
    if mode in {"parallel_consensus", "parallel_enthophy"}:
        from rollout.consensus import select_consensus

        candidates = [SimpleNamespace(branch_id=r["branch_id"], message={}, response=SimpleNamespace(
            model_dump=lambda mode="json", value=r["response"]: value)) for r in eligible]
        winner, _ = select_consensus(
            candidates, {r["branch_id"]: float(r["score"]) for r in eligible},
            consensus_config.get("retention_ratio", 0.5),
            consensus_config.get("min_retained_branches", 2),
            consensus_config.get("tie_policy", "estimator_score"),
            retention_rule=consensus_config.get("retention_rule"),
            full_retention_threshold=consensus_config.get("full_retention_threshold", 10))
        if winner is None:
            return "selection_unavailable_after_filter"
    elif mode != "parallel_baseline" and not eligible and fallback_policy != "first_valid":
        return "selection_unavailable_after_filter"
    return None


def validate_loop_escape_evidence(decision, previous_decisions=None):
    """Replay filtering; optionally reconstruct slot history from earlier receipts."""
    meta = decision.get("loop_escape")
    if meta is None:  # Legacy/off evidence is byte-shape compatible.
        return
    if meta.get("version") != VERSION or meta.get("enabled") is not True:
        raise ValueError("Loop escape version/enabled mismatch")
    history = None
    if previous_decisions is not None:
        history = LoopEscapeHistory()
        for previous in sorted((p for p in previous_decisions if p.get("loop_escape")),
                               key=lambda p: p["loop_escape"]["decision_index"]):
            if previous["loop_escape"]["decision_index"] != history.decisions:
                raise ValueError("Loop escape history has a gap")
            history.remember(previous)
        if meta["decision_index"] != history.decisions:
            raise ValueError("Loop escape decision index differs from history")
    rows = decision["candidates"]
    for row in rows:
        data = row["loop_escape"]
        if data["output_available"] and row.get("response") is not None:
            if any(data[key] != value for key, value in fingerprints(row["response"]).items()):
                raise ValueError("Loop escape fingerprint differs from response")
        if history is not None:
            current = {name + "_fingerprint": data[name + "_fingerprint"] for name in ("action", "reasoning")}
            expected = history.inspect(row["branch_id"], current)
            if any(data[key] != value for key, value in expected.items()):
                raise ValueError("Loop escape comparison differs from slot history")
    stage, proposed = filter_stage(rows, meta["decision_index"])
    before = sorted(r["branch_id"] for r in rows if r["loop_escape"]["output_available"])
    if stage != meta["stage"] or sorted(proposed) != meta["proposed_candidate_ids"] or before != meta["candidate_ids_before"]:
        raise ValueError("Loop escape stage/candidates mismatch")
    failure = meta["failure_reason"]
    if failure not in {None, "all_comparable_outputs_repeated", "selection_unavailable_after_filter"}:
        raise ValueError("Unknown loop escape failure reason")
    if failure == "all_comparable_outputs_repeated" and (proposed or not before):
        raise ValueError("Loop escape exhaustion evidence mismatch")
    if failure is None and before and not proposed:
        raise ValueError("Loop escape empty filter has no fallback")
    # Full receipts can also prove that a fallback was necessary. Compact agent
    # summaries omit raw responses; their source receipts carry this stronger check.
    if all(r.get("response") is not None for r in rows if r["loop_escape"]["output_available"]):
        expected_failure = selection_failure(
            rows, meta, meta["rollout_mode"], meta["fallback_policy"], meta["consensus_config"])
        if before and not proposed:
            expected_failure = "all_comparable_outputs_repeated"
        if failure != expected_failure:
            raise ValueError("Loop escape fallback differs from original selection constraints")
    after = sorted(proposed) if failure is None else before
    if meta["applied"] != (failure is None) or meta["candidate_ids_after"] != after:
        raise ValueError("Loop escape applied candidates mismatch")
    for row in rows:
        data = row["loop_escape"]
        excluded = data["output_available"] and row["branch_id"] not in after
        if data["excluded"] != excluded or data["exclusion_reason"] != ("repeated_" + stage if excluded else None):
            raise ValueError("Loop escape exclusion differs from filter")
        if row["status"] != ("loop_excluded" if excluded else data["status_before"]):
            raise ValueError("Loop escape candidate status mismatch")
        if row["reason"] != ("cross_step_loop" if excluded else data["reason_before"]):
            raise ValueError("Loop escape candidate reason mismatch")
        if data["selection_eligible_before"] is not None and row.get("selection_eligible") != (
                False if excluded else data["selection_eligible_before"]):
            raise ValueError("Loop escape random eligibility mismatch")
    selected = decision.get("selected_branch_id")
    if selected is not None and next(r for r in rows if r["branch_id"] == selected)["loop_escape"]["excluded"]:
        raise ValueError("Loop escape selected an excluded candidate")
