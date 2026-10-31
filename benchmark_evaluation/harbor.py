"""Harbor TrialResult mapping and official Pro replay hooks."""
from __future__ import annotations

import math
from pathlib import Path

from core.io import hash_json, sha256


def trial_outcome(result, *, task, attempt_id, subject=None):
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
    return {"schema_version": 1, "task": task["instance_id"], "attempt_id": attempt_id,
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
