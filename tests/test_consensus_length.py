import pytest

from core.action_identity import canonical_action
from difficulty.evaluation.consensus_length import evaluate_decision, evaluate_r2
from difficulty.evaluation.instability import aggregate_metrics


MESSAGE = {"tool_calls": [{"function": {"name": "run", "arguments": '{"command": "pwd"}'}}]}
ACTION, _ = canonical_action(MESSAGE)


def fixture(lengths, scores, *, ids=None, selected=0, policy="estimator_score"):
    ids = list(range(len(lengths))) if ids is None else ids
    return {"branch_count": len(lengths), "selected_branch_id": selected,
            "consensus_config": {"representative_policy": policy}, "consensus": {
                "status": "reached", "consensus_action_hash": ACTION, "retained_branch_ids": ids,
                "votes": [{"action_hash": ACTION, "branch_ids": ids, "count": len(ids)}]},
            "candidates": [{"branch_id": i, "completion_tokens": length, "score": scores[i],
                            "retained": i in ids, "action_hash": ACTION if i in ids else None}
                           for i, length in enumerate(lengths)]}


def test_winner_scope_and_rank_direction():
    decision = fixture([30, 10, 20, 1], [1, 2, 3, 0], ids=[0, 1, 2])
    step, branches = evaluate_decision(decision, MESSAGE)
    assert [row["branch_id"] for row in branches] == [0, 1, 2]
    assert step["same_branch"] is False
    assert step["token_excess"] == step["actual_token_regret"] == 20
    assert step["relative_token_excess"] == 2
    assert step["spearman"] == pytest.approx(-0.5)
    assert step["kendall_tau_b"] == pytest.approx(-1 / 3)
    positive, _ = evaluate_decision(fixture([10, 20, 30], [1, 2, 3]), MESSAGE)
    assert positive["spearman"] == pytest.approx(1)


def test_tie_sets_and_operational_representatives():
    step, _ = evaluate_decision(fixture([50, 10, 10], [1, 1, 2], selected=1,
                                        policy="min_step_tokens"), MESSAGE)
    assert step["shortest_branch_ids"] == [1, 2]
    assert step["estimator_best_branch_ids"] == [0, 1]
    assert step["shortest_branch_id"] == 1 and step["estimator_branch_id"] == 0
    assert step["optima_overlap"] is True and step["same_branch"] is False
    assert step["actual_is_estimator_best"] is True and step["actual_token_regret"] == 0
    assert step["token_excess"] == 40


def test_incomplete_group_and_undefined_statistics():
    step, branches = evaluate_decision(fixture([None, 10], [0.0, 1.0]), MESSAGE)
    assert step["status"] == "missing_length" and step["paired_count"] == 1
    assert step["shortest_branch_ids"] is None and branches[0]["is_shortest"] is None
    assert step["estimator_best_branch_ids"] == [0] and step["token_excess"] is None
    step, _ = evaluate_decision(fixture([0, 9], [float("nan"), 0.0]), MESSAGE)
    assert step["status"] == "missing_score" and step["actual_token_regret"] == 0
    step, _ = evaluate_decision(fixture([0, 9], [1, 1]), MESSAGE)
    assert step["relative_token_excess"] is None and step["spearman"] is None
    step, _ = evaluate_decision(fixture([10], [0.5]), MESSAGE)
    assert step["status"] == "singleton" and step["kendall_tau_b"] is None


def test_absent_winner_and_corrupted_winner_branch():
    step, branches = evaluate_decision({"consensus": {"status": "no_eligible_branch"}}, {})
    assert step["status"] == "no_winner" and not branches
    decision = fixture([10, 20], [1, 2], ids=[1])
    with pytest.raises(ValueError, match="outside the recorded winner vote"):
        evaluate_decision(decision, MESSAGE)


def test_task_balancing_with_missing_repo():
    rows = [{"task_id": "a", "repo": None, "rollout_id": "x", "metric": 1}] * 4
    rows += [{"task_id": "a", "repo": None, "rollout_id": "y", "metric": 0},
             {"task_id": "b", "repo": "repo", "rollout_id": "z", "metric": 0}]
    stat = aggregate_metrics(rows, ["metric"], bootstrap_samples=20, require_repo=True)["metric"]
    assert stat["mean"] == 0.25
    assert stat["task_bootstrap"]["clusters"] == 2 and stat["repo_bootstrap"]["clusters"] == 1


def _regression_groups(xs, scores_by_task):
    return [(str(i), str(i), [{"completion_tokens": x, "score": y} for x, y in zip(xs, ys)])
            for i, ys in enumerate(scores_by_task)]


def test_r2_distinguishes_raw_prediction_from_within_step_variation():
    groups = _regression_groups([10, 20], [[1, 2], [11, 12], [21, 22]])
    result = evaluate_r2(groups, seed=42)
    assert result["within_step_score_r2"] == pytest.approx(1)
    assert result["raw_score_r2"] < 0
    assert result["fold_task_counts"] == [1, 1, 1]


def test_r2_cross_validation_and_undefined_cases():
    groups = _regression_groups([10, 10], [[0, 0], [1, 1], [2, 2]])
    r2 = evaluate_r2(groups, seed=42)
    assert r2["raw_score_r2"] == pytest.approx(-1.25)
    assert r2["within_step_score_r2"] is None
    assert evaluate_r2(groups[:1], seed=42)["status"] == "insufficient_tasks"
    assert evaluate_r2(_regression_groups([10, 20], [[1, 1]] * 3), seed=42)["status"] == "constant_score"
