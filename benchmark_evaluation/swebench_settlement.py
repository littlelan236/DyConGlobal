"""SWE-bench score projection, shared by execution and training consumers.

Official evidence is immutable. Only supported ValidOutcome results provide a
boolean success label or numeric terminal reward. Legacy data remains readable;
an existing sidecar must never be ignored when deriving a training label.
"""
from __future__ import annotations

POLICY = "swebench_validity_settlement_v1"


def project(outcome, audit):
    """Produce a settled outcome without mutating raw input or audit evidence."""
    raw = {k: outcome[k] for k in ("status", "resolved", "failure_type", "failure_detail", "error")
           if k in outcome}
    official = dict(audit.get("official_result", {}))
    audit = dict(audit)
    classification = audit.get("classification")
    valid = classification == "ValidOutcome" and type(audit.get("effective_resolved")) is bool
    if valid and audit["effective_resolved"] is not official.get("resolved"):
        # An invalid submission can be a supported zero without an official
        # test report. No other projection is allowed to override raw scoring.
        valid = audit.get("reason_code") in {"missing_patch", "invalid_patch"} and audit["effective_resolved"] is False
    if classification == "ValidOutcome" and not valid:
        classification = "Review"
        audit.update(classification=classification, effective_resolved=None,
                     reason_code="invalid_score_projection")
    value = audit["effective_resolved"] if valid else None
    status = "completed" if valid else "VerifierError" if classification == "VerifierError" else "Review"
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
        if not isinstance(audit, dict) or audit.get("classification") != "ValidOutcome":
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
    if classification == "ValidOutcome":
        if type(value) is not bool or outcome.get("effective_resolved") is not value:
            raise ValueError("SWE-bench valid settlement requires a supported boolean reward")
    elif classification in {"VerifierError", "Review"}:
        if (outcome.get("status") != classification or outcome.get("resolved") is not None
                or outcome.get("effective_resolved") is not None or outcome.get("reward") is not None
                or outcome.get("scoring_eligible") is not False):
            raise ValueError("SWE-bench invalid evaluation cannot carry a training reward")
    else:
        raise ValueError("SWE-bench settlement requires an audit classification")
