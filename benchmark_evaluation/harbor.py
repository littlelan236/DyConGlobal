"""Harbor TrialResult mapping and official Pro replay hooks."""
from __future__ import annotations

import math
from pathlib import Path

from core.io import hash_json, sha256


def trial_outcome(result, *, task, attempt_id, subject=None, evidence_dir=None, operation=None):
    def json_fact(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"invalid_numeric": repr(value)}
        if isinstance(value, dict):
            return {k: json_fact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_fact(v) for v in value]
        return value
    n_input = n_output = None
    cost = None
    if result.get("agent_result"):
        agent = result["agent_result"]
        n_input, n_output, cost = agent.get("n_input_tokens"), agent.get("n_output_tokens"), agent.get("cost_usd")
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards")
    exception = result.get("exception_info")
    valid_reward = (isinstance(rewards, dict) and bool(rewards) and all(
        type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in rewards.values()))
    if exception and not result.get("reward_valid", False):
        status, resolved = "infra_error", None
    elif valid_reward:
        status = "completed"
        resolved = all(v == 1 or v == 1.0 for v in rewards.values())
    else:
        status, resolved = "unknown", None
    audit = None
    if task.get("benchmark") == "terminalbench_2_1":
        if status == "infra_error":
            from benchmark_evaluation.harbor_verifier import POLICY
            audit = {"schema_version": 1, "policy": POLICY, "classification": "VerifierError",
                     "reason_code": "upstream_verifier_error", "sources": []}
        elif valid_reward and resolved and operation == "oracle" and evidence_dir is not None:
            from benchmark_evaluation.harbor_verifier import audit_verifier
            reference_audit = audit_verifier(evidence_dir, attempt_id, operation=operation, reference_only=True,
                                             trial_name=result.get("trial_name"))
            if reference_audit["classification"] == "VerifierError":
                audit = reference_audit
                status, resolved = "infra_error", None
            elif reference_audit["classification"] == "Review":
                audit = reference_audit
                status, resolved = "unknown", None
        elif valid_reward and not resolved and evidence_dir is not None:
            from benchmark_evaluation.harbor_verifier import audit_verifier
            audit = audit_verifier(evidence_dir, attempt_id, operation=operation,
                                   trial_name=result.get("trial_name"))
            if audit["classification"] == "VerifierError":
                status, resolved = "infra_error", None
            elif audit["classification"] == "Review":
                status, resolved = "unknown", None
    # Oracle is a positive control: a finite sub-one reward invalidates the
    # control even when pytest completed normally or the reference exited zero.
    # This does not establish the root cause, and must not affect model/nop zeros.
    if (task.get("benchmark") == "terminalbench_2_1" and operation == "oracle"
            and valid_reward and not all(v == 1 for v in rewards.values())):
        if audit is None or audit["classification"] != "VerifierError":
            from benchmark_evaluation.harbor_verifier import POLICY
            audit = {"schema_version": 1, "policy": POLICY,
                     "classification": "VerifierError",
                     "reason_code": "oracle_reference_rejected",
                     "evidence": {"expected_reward": "all_dimensions_equal_one",
                                  "verifier_health": audit},
                     "sources": (audit or {}).get("sources", []),
                     "incomplete_sources": (audit or {}).get("incomplete_sources", [])}
        status, resolved = "infra_error", None
    return {"schema_version": 1, "task": task["instance_id"], "attempt_id": attempt_id,
            "verifier_audit": audit, "exit_status": "VerifierError" if status == "infra_error" else None,
            "raw_reward": json_fact(rewards), "status": status, "resolved": resolved,
            "raw_reward_encoding": "nonfinite_text_v1",
            "exception": exception, "subject": subject, "reward_policy": "all_unit_interval_dimensions_equal_one_v1",
            "usage": {"input_tokens": n_input,
                "output_tokens": n_output, "cost": cost}, "trial_result_sha256": hash_json(json_fact(result)),
            "trial_result_encoding": "nonfinite_text_v1"}


def official_replay_reference(snapshot, task, patch):
    evaluator = snapshot.get("evaluator")
    if not evaluator:
        raise ValueError("This Harbor benchmark does not support post-hoc replay")
    path = Path(evaluator["path"])
    if sha256(path) != evaluator["sha256"]:
        raise ValueError("Official evaluator changed")
    return {"kind": "official_pro_replay", "task": task["instance_id"],
            "patch": {"path": str(patch), "sha256": sha256(patch)},
            "evaluator": evaluator, "new_model_requests": 0}
