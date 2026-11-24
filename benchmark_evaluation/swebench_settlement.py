"""SWE-bench score projection, shared by execution and training consumers.

Official evidence is immutable. Fully audited scores and formal-run scores
backed by a completed official result can provide a terminal reward. Legacy
data remains readable.
"""
from __future__ import annotations

POLICY = "swebench_validity_settlement_v1"
SCORED_CLASSIFICATIONS = {"ValidOutcome", "OfficialOutcome"}
UNSCORED_STATUSES = {"infra_error", "ConnectionError", "completed_unknown"}


def official_result(outcome):
    if "official_result" in outcome:
        return dict(outcome["official_result"])
    raw = outcome.get("raw_outcome", outcome)
    return {"completed": raw.get("status") == "completed", "resolved": raw.get("resolved")}


def review_policy(audit, *, gold):
    """Use the official score for ambiguous formal evaluations."""
    if gold or audit.get("classification") != "Review":
        return audit
    official = audit.get("official_result", {})
    value = official.get("resolved")
    if official.get("completed") is True and type(value) is bool:
        return {**audit, "classification": "OfficialOutcome", "effective_resolved": value}
    return {**audit, "classification": "Unscored", "effective_resolved": None}


def project(outcome, audit, *, gold=False):
    """Produce a settled outcome without mutating raw input or audit evidence."""
    source = outcome.get("raw_outcome", outcome)
    raw = {k: source[k] for k in ("status", "resolved", "failure_type", "failure_detail", "error")
           if k in source}
    official = dict(audit.get("official_result", {}))
    audit = dict(review_policy(audit, gold=gold))
    classification = audit.get("classification")
    valid = classification in SCORED_CLASSIFICATIONS and type(audit.get("effective_resolved")) is bool
    if classification == "OfficialOutcome":
        valid = (not gold and valid and official.get("completed") is True
                 and audit["effective_resolved"] is official.get("resolved"))
    if classification == "ValidOutcome" and valid and audit["effective_resolved"] is not official.get("resolved"):
        # An invalid submission can be a supported zero without an official
        # test report. No other projection is allowed to override raw scoring.
        valid = audit.get("reason_code") in {"missing_patch", "invalid_patch"} and audit["effective_resolved"] is False
    if classification not in {"VerifierError", "Review", "Unscored"} and not valid:
        classification = "Review" if gold else "Unscored"
        audit.update(classification=classification, effective_resolved=None,
                     reason_code="invalid_score_projection")
    value = audit["effective_resolved"] if valid else None
    if valid:
        status = "completed"
    elif classification == "Unscored":
        status = raw.get("status", "completed_unknown")
        if status not in UNSCORED_STATUSES:
            status = "completed_unknown"
    else:
        status = classification
    return {**outcome, "raw_outcome": raw, "official_result": official,
            "status": status, "resolved": value, "effective_resolved": value,
            "reward": int(value) if type(value) is bool else None,
            "scoring_policy": POLICY, "scoring_eligible": valid,
            "verifier_audit": {k: audit.get(k) for k in
                ("policy_version", "classification", "reason_code", "effective_resolved", "evidence_digest",
                 "cause_confirmed", "diagnostic_status")}}


def label(outcome):
    """Read a terminal label, including legacy sidecars; never coerce null."""
    audit = outcome.get("verifier_audit")
    if audit is not None:
        if not isinstance(audit, dict) or audit.get("classification") not in SCORED_CLASSIFICATIONS:
            return None
        value = audit.get("effective_resolved")
        if type(value) is not bool or value is not outcome.get("resolved"):
            return None
    if outcome.get("scoring_policy"):
        if outcome["scoring_policy"] != POLICY or outcome.get("scoring_eligible") is not True:
            return None
        value = outcome.get("resolved")
        if type(value) is not bool or outcome.get("reward") != int(value):
            return None
    return outcome.get("resolved") if outcome.get("status") == "completed" and type(outcome.get("resolved")) is bool else None


def training_label(row, evidence, rollout_id):
    """Bind terminal eligibility to the selected physical evaluation outcome."""
    if row.get("evaluation_status") != "completed" or type(row.get("resolved")) is not bool:
        return None
    eid = row.get("selected_evaluation_id")
    outcome = evidence.get("evaluations", {}).get(eid, {}).get("outcomes", {}).get(rollout_id)
    if outcome is None:
        return None
    task = evidence.get("tasks", {}).get(row.get("task_key"), {})
    benchmark = str(task.get("benchmark", "")).lower().replace("-", "").replace("_", "")
    if "swebench" in benchmark and not outcome.get("verifier_audit"):
        # An old terminal score lacking validity evidence is not retroactively
        # trusted for newly built success labels. Explicitly re-evaluate it.
        return None
    value = label(outcome)
    return value if value is row.get("resolved") else None


def training_exclusion(row, evidence, rollout_id):
    """Explicit invalid evaluation excludes the whole trajectory's new groups.

    Unscored legacy consensus data is still representable. This guard only
    rejects known invalid terminal assessments, not every unevaluated rollout.
    """
    eid = row.get("selected_evaluation_id")
    outcome = evidence.get("evaluations", {}).get(eid, {}).get("outcomes", {}).get(rollout_id, {})
    audit = outcome.get("verifier_audit") or row.get("verifier_audit") or {}
    classification = audit.get("classification")
    if classification in {"VerifierError", "Review"}:
        return "terminal_verifier_error" if classification == "VerifierError" else "terminal_evaluation_review"
    status = outcome.get("status", row.get("evaluation_status"))
    if status in {"VerifierError", "Review", "infra_error", "ConnectionError"}:
        return "terminal_evaluation_ineligible"
    if outcome.get("scoring_policy") and label(outcome) is None:
        return "terminal_evaluation_ineligible"
    return None


def validate(outcome):
    """Reject inconsistent new settlement records at evidence write/read time."""
    if not outcome.get("scoring_policy"):
        return
    if outcome["scoring_policy"] != POLICY:
        raise ValueError("Unsupported SWE-bench scoring policy")
    classification = outcome.get("verifier_audit", {}).get("classification")
    value = label(outcome)
    if classification in SCORED_CLASSIFICATIONS:
        if type(value) is not bool or outcome.get("effective_resolved") is not value:
            raise ValueError("SWE-bench valid settlement requires a supported boolean reward")
        official = outcome.get("official_result", {})
        if classification == "OfficialOutcome" and (official.get("completed") is not True
                or official.get("resolved") is not value):
            raise ValueError("SWE-bench official settlement requires a completed matching result")
    elif classification in {"VerifierError", "Review", "Unscored"}:
        allowed = UNSCORED_STATUSES if classification == "Unscored" else {classification}
        if (outcome.get("status") not in allowed or outcome.get("resolved") is not None
                or outcome.get("effective_resolved") is not None or outcome.get("reward") is not None
                or outcome.get("scoring_eligible") is not False):
            raise ValueError("SWE-bench invalid evaluation cannot carry a training reward")
    else:
        raise ValueError("SWE-bench settlement requires an audit classification")
