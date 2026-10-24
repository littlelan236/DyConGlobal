import copy
import json
from types import SimpleNamespace

import pytest

from core.config import Parallel
from rollout.consensus import select_consensus


class Candidate:
    def __init__(self, branch, command):
        self.branch_id = branch
        self.message = {"ok": True}
        self.response = SimpleNamespace(model_dump=lambda mode="json": {
            "choices": [{"message": {"tool_calls": [{"id": f"id-{branch}",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]}}]})


def test_consensus_retains_lowest_scores_and_votes_exact_action():
    candidates = [Candidate(i, "ls" if i < 3 else "pwd") for i in range(4)]
    winner, meta = select_consensus(candidates, {0: 1., 1: 2., 2: 3., 3: 4.}, 0.5, 2)
    assert winner.branch_id == 0
    assert meta["retained_branch_ids"] == [0, 1]
    assert meta["votes"][0]["count"] == 2


def test_consensus_tie_uses_estimator_score():
    candidates = [Candidate(0, "ls"), Candidate(1, "pwd"), Candidate(2, "echo")]
    winner, meta = select_consensus(candidates, {0: 1., 1: 2., 2: 3.}, 1.0, 2)
    assert winner.branch_id == 0
    assert meta["status"] == "tie_resolved_by_estimator"


def test_consensus_records_branch_filter_and_action_identity():
    candidates = [Candidate(0, "ls"), Candidate(1, "ls"), Candidate(2, "pwd"), Candidate(3, "echo")]
    winner, meta = select_consensus(candidates, {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}, 0.5, 2)
    assert winner.branch_id == 0
    assert meta["status"] == "reached"
    assert meta["valid_action_count"] == 2
    by_branch = {row["branch_id"]: row for row in meta["branch_metadata"]}
    assert by_branch[0]["retained"] is True
    assert by_branch[1]["action_hash"] == by_branch[0]["action_hash"]
    assert by_branch[2]["exclusion_reason"] == "estimator_filtered"


def test_consensus_tie_metadata_contains_comparison_keys():
    candidates = [Candidate(0, "ls"), Candidate(1, "pwd")]
    winner, meta = select_consensus(candidates, {0: 2.0, 1: 1.0}, 1.0, 2)
    assert winner.branch_id == 1
    assert meta["status"] == "tie_resolved_by_estimator"
    assert meta["tie_break"]["policy"] == "estimator_score"
    assert len(meta["tie_break"]["comparison_keys"]) == 2


def test_consensus_rejects_invalid_parameters():
    candidates = [Candidate(0, "ls"), Candidate(1, "pwd")]
    with pytest.raises(ValueError, match="retention_ratio"):
        select_consensus(candidates, {0: 1.0, 1: 2.0}, 0.0, 2)
    with pytest.raises(ValueError, match="tie policy"):
        select_consensus(candidates, {0: 1.0, 1: 2.0}, 1.0, 2, "stop")


def test_consensus_config_requires_stop_fallback():
    with pytest.raises(ValueError, match="fallback_policy"):
        Parallel(
            enabled=True,
            mode="parallel_consensus",
            branch_count=32,
            estimator_ref="estimator.json",
            fallback_policy="first_valid",
        )


def test_32_branches_plurality_beats_global_lowest_score():
    candidates = [Candidate(i, "pwd" if i == 0 else "ls" if i < 10 else "echo " + str(i))
                  for i in range(32)]
    winner, meta = select_consensus(candidates, {i: float(i) for i in range(32)})
    assert len(meta["retained_branch_ids"]) == 16
    assert winner.branch_id == 1
    assert max(v["count"] for v in meta["votes"]) == 9


def test_same_best_score_uses_mean_then_branch_id():
    candidates = [Candidate(0, "ls"), Candidate(1, "ls"),
                  Candidate(2, "pwd"), Candidate(3, "pwd")]
    winner, _ = select_consensus(candidates, {0: 1., 1: 4., 2: 1., 3: 2.}, 1.)
    assert winner.branch_id == 2
    winner, _ = select_consensus(candidates, {0: 1., 1: 2., 2: 1., 3: 2.}, 1.)
    assert winner.branch_id == 0


def test_insufficient_or_invalid_scores_never_execute_single_branch():
    candidates = [Candidate(0, "ls"), Candidate(1, "pwd")]
    winner, meta = select_consensus(candidates, {0: 1., 1: float("nan")})
    assert winner is None
    assert meta["status"] == "insufficient_branches"
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("ratio", [-1., 0., float("nan"), float("inf"), 1.01])
def test_invalid_retention_config(ratio):
    with pytest.raises(ValueError):
        Parallel(enabled=True, mode="parallel_consensus", branch_count=32,
                 estimator_ref="p.json", fallback_policy="stop",
                 consensus={"retention_ratio": ratio})


def coordinator_case(count=32):
    from rollout.parallel import ParallelReasoningCoordinator

    evidence, released = [], []
    class Client:
        def call(self, action, payload):
            if action == "release":
                released.append(payload["decision_id"])
            if action != "results":
                return {}
            return {"results": [{"branch_id": i, "score": float(i), "status": "scored"}
                                for i in range(count)], "estimator_sha256": "sha",
                    "target": "submission_remaining_reasoning_tokens_log1p", "endpoint": "submission"}
    config = {"retention_ratio": .5, "min_retained_branches": 2,
              "rule": "exact_action", "tie_policy": "estimator_score"}
    cfg = {"parallel": {"mode": "parallel_consensus", "branch_count": count,
                        "score_timeout_ms": 200, "fallback_policy": "stop", "consensus": config},
           "generation": {"seed": 42}, "capture": {"control_seconds": 1}}
    def record(value, timeout):
        evidence.append(copy.deepcopy(value))
        return value
    class Response:
        def __init__(self, bid):
            self.branch = bid
            self.choices = [SimpleNamespace(finish_reason="tool_calls")]
        def model_dump(self, **kwargs):
            command = "pwd" if self.branch == 0 else "ls"
            return {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "bash", "arguments": json.dumps({"command": command})}}]}}]}
    model = ParallelReasoningCoordinator(cfg, SimpleNamespace(record_decision=record), Client())
    result = model.run([], {"session_id": "s", "step_id": 0, "query_id": "q"},
                       lambda messages, md, seed: Response(md["branch_id"]),
                       lambda response: {"role": "assistant", "extra": {"actions": [response.branch]}},
                       lambda response: {"cost": 0.})
    return result, evidence[0], released


def test_coordinator_and_startup_audit_reuse_exact_selection():
    from rollout.consensus import validate_consensus_evidence
    from runtime.serving.parallel_validation import validate_selection

    result, decision, released = coordinator_case()
    assert result["extra"]["actions"] == [1]
    assert len(released) == 1
    validate_consensus_evidence(decision)
    summary = result["extra"]["dycon_parallel"]
    assert validate_selection(summary, 42)["selected_branch_id"] == 1
    changed = copy.deepcopy(decision)
    changed["consensus"]["votes"][0]["count"] += 1
    with pytest.raises(ValueError, match="evidence"):
        validate_consensus_evidence(changed)
    changed = copy.deepcopy(summary)
    changed["selected_branch_id"] = 0
    with pytest.raises(ValueError, match="selected branch"):
        validate_selection(changed, 42)
