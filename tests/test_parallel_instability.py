"""Analytic consistency metrics and receipt-bound real filesystem integration."""

import json
import math
from pathlib import Path

import pytest
from test_parallel_visualization import published as published

from core.io import hash_json, read_json, sha256, write_json
from difficulty.evaluation.instability import aggregate_steps, evaluate_step, summarize_groups
from difficulty.evaluation.parallel import collect_parallel_scores
from difficulty.evaluation.parallel_identity import (
    CandidateIdentityReader,
    IdentityUnavailable,
    action_identity,
)
from difficulty.evaluation.parallel_instability import evaluate_parallel_instability


def branch(score, action="A", reasoning="r", bid=0, **extra):
    return {"score": score, "action_hash": action, "action": action, "reasoning_hash": reasoning,
            "status": "eligible", "identity_status": "available", "branch_id": bid,
            "decision_id": "decision", "parent_prefix_sha256": "prefix", **extra}


def measure(rows, view="deduplicated", log1p=True):
    return evaluate_step(rows, view=view, log1p=log1p, multipliers=[1.25, 1.5, 2., 3.])


def test_exact_primary_auxiliary_and_multiplier_boundaries():
    groups, step = measure([branch(0., reasoning="a"), branch(math.log(1.25), reasoning="b"),
                            branch(math.log(2), reasoning="c")])
    values = [0., math.log(1.25), math.log(2)]
    assert step["mean_pair_gap"] == pytest.approx(2 * math.log(2) / 3)
    assert groups[0]["sample_std"] == pytest.approx(math.sqrt(sum((v-sum(values)/3)**2 for v in values) / 2))
    assert groups[0]["range"] == math.log(2)
    assert step["pair_rate_ge_1.25x"] == 1.
    assert step["pair_rate_ge_1.5x"] == pytest.approx(2/3)
    assert step["pair_rate_ge_2.0x"] == pytest.approx(1/3)
    assert step["group_rate_ge_2.0x"] == 1.
    assert step["pair_rate_ge_3.0x"] == 0.
    assert step["relative_ratio"] is None


def test_duplicate_reasonings_dedup_before_statistics():
    rows = [branch(0., reasoning="a", bid=i) for i in range(3)] + [branch(2., reasoning="b", bid=3)]
    dedup, d = measure(rows)
    raw, r = measure(rows, view="sampling")
    assert d["mean_pair_gap"] == 2.
    assert d["mean_sample_std"] == pytest.approx(math.sqrt(2))
    assert dedup[0]["unit_count"] == 2
    assert r["mean_pair_gap"] == 1.
    assert r["mean_sample_std"] == 1.
    assert raw[0]["unit_count"] == 4
    # Repeated text/token identities use their mean score, not a first occurrence.
    _, average = measure([branch(0., reasoning="a"), branch(2., reasoning="a"), branch(3., reasoning="b")])
    assert average["mean_pair_gap"] == 2.


def test_equal_action_group_weight_not_pair_weight():
    rows = [branch(float(i), reasoning=str(i), bid=i) for i in range(3)]
    rows += [branch(0., action="B", reasoning="x"), branch(4., action="B", reasoning="y")]
    _, step = measure(rows)
    assert step["mean_pair_gap"] == pytest.approx((4/3+4)/2)
    assert step["mean_pair_gap"] != 2.  # Pooled pairs would give 2.


def test_relative_excludes_identical_reasoning_on_both_sides():
    _, step = measure([branch(0., reasoning="a"), branch(2., reasoning="b"),
                       branch(100., action="B", reasoning="a")])
    assert step["relative_same_gap"] == 2.
    assert step["relative_different_gap"] == 98.
    assert step["relative_ratio"] == pytest.approx(2/98)


def test_unavailable_is_null_not_zero_and_negative_scores_preserved():
    groups, step = measure([branch(-2.), branch(4.)])  # Same reasoning: no perturbation group.
    assert not groups[0]["available"]
    assert step["mean_pair_gap"] is None
    assert step["pair_rate_ge_2.0x"] is None
    _, step = measure([branch(-2., reasoning="a"), branch(-1., reasoning="b")])
    assert step["mean_pair_gap"] == 1.
    _, native = measure([branch(1., reasoning="a"), branch(2., reasoning="b")], log1p=False)
    assert not any("rate_ge" in key for key in native)


def test_constant_predictor_relative_is_undefined():
    _, step = measure([branch(0., reasoning="a"), branch(0., reasoning="b"),
                       branch(0., action="B", reasoning="c")])
    assert step["mean_pair_gap"] == 0.
    assert step["relative_ratio"] is None
    row = {**step, "task_id": "task", "repo": "repo", "rollout_id": "rollout"}
    report = aggregate_steps([row], ["mean_pair_gap"], bootstrap_samples=20)
    assert report["relative"]["ratio"] is None
    assert report["relative"]["zero_denominator_steps"] == 1
    assert report["mean_pair_gap"]["task_bootstrap"]["ci95"] is None


def test_task_rollout_balance_and_paired_ratio_of_means():
    def row(value, task, rollout, same=1., diff=1.):
        return {"mean_pair_gap": value, "task_id": task, "rollout_id": rollout, "repo": task,
                "relative_same_gap": same, "relative_different_gap": diff}
    rows = [row(10., "long", "long-a") for _ in range(100)]
    rows += [row(0., "long", "long-b"), row(3., "short", "short-a")]
    metrics = aggregate_steps(rows, ["mean_pair_gap"], bootstrap_samples=50, seed=3)
    assert metrics["mean_pair_gap"]["mean"] == 4.
    assert metrics["mean_pair_gap"]["tasks"] == 2
    assert metrics["mean_pair_gap"]["rollouts"] == 3
    assert metrics == aggregate_steps(rows, ["mean_pair_gap"], bootstrap_samples=50, seed=3)
    paired = aggregate_steps([row(1., "a", "a", 1., 1.), row(1., "b", "b", 3., 9.)],
                             ["mean_pair_gap"], bootstrap_samples=50)
    assert paired["relative"]["ratio"] == .4


def test_prefix_isolation_and_coverage_filters():
    rows = [branch(1., reasoning="a"), branch(100., reasoning="b", status="ineligible")]
    _, step = measure(rows)
    assert step["mean_pair_gap"] is None
    with pytest.raises(ValueError, match="parent-prefix"):
        measure([branch(0., reasoning="a"), branch(1., reasoning="b", parent_prefix_sha256="other")])
    with pytest.raises(ValueError, match="finite"):
        measure([branch(float("nan"), reasoning="a")])


def test_group_size_stratification():
    groups, _ = measure([branch(0., reasoning="a"), branch(2., reasoning="b")])
    report = summarize_groups(groups)
    assert report["by_unit_count"]["2"]["range"]["p95"] == 2.


def test_catalog_task_keys_are_the_clustering_authority():
    rows = [{"task_id": "same-visible-id", "task_key": key, "repo": "same-repo",
             "rollout_id": key, "mean_pair_gap": value} for key, value in [("key-a", 1.), ("key-b", 9.)]]
    result = aggregate_steps(rows, ["mean_pair_gap"], bootstrap_samples=20)
    assert result["mean_pair_gap"]["tasks"] == 2
    assert result["mean_pair_gap"]["mean"] == 5.
    assert result["mean_pair_gap"]["task_bootstrap"]["clusters"] == 2
    assert result["mean_pair_gap"]["repo_bootstrap"]["clusters"] == 1


def candidate(arguments, reasoning="reason", tool="bash", cid="id"):
    return {"response": {"choices": [{"message": {"reasoning_content": reasoning,
            "tool_calls": [{"id": cid, "function": {"name": tool, "arguments": arguments}}]}}]}}


def test_action_identity_preserves_order_and_command_bytes():
    a = candidate('{"command":"ls", "timeout":5}')
    b = candidate('{"timeout":5,"command":"ls"}', cid="different")
    assert action_identity(a) == action_identity(b)
    assert action_identity(a)[0] != action_identity(candidate('{"command":"ls ","timeout":5}'))[0]
    for args in ['{"command":"ls","command":"pwd"}', '{"timeout":NaN}', 'not json']:
        with pytest.raises(IdentityUnavailable):
            action_identity(candidate(args))
    a["response"]["choices"][0]["message"]["tool_calls"].append(
        {"function": {"name": "bash", "arguments": '{"command":"pwd"}'}})
    b = json.loads(json.dumps(a))
    b["response"]["choices"][0]["message"]["tool_calls"].reverse()
    assert action_identity(a)[0] != action_identity(b)[0]


def test_native_identity_uses_only_scored_reasoning_and_verifies_hashes(tmp_path):
    inventory = {"open_token_ids": [10], "close_token_ids": [11]}
    write_json(tmp_path / "inventory.json", inventory)
    request = {"inventory_ref": "inventory.json", "inventory_sha256": hash_json(inventory),
               "request_id": "native", "decision_id": "decision", "branch_id": 0,
               "parent_prefix_sha256": "prefix", "prompt_token_ids": [1, 10],
               "request_sha256": hash_json([1, 10]), "output_token_ids": [20, 21, 11, 99]}
    write_json(tmp_path / "request.json", request)
    c = {**candidate('{"command":"ls"}'), "ledger_ref": "request.json",
         "ledger_sha256": sha256(tmp_path / "request.json"), "branch_id": 0,
         "native_request_id": "native", "absolute_position": 4}
    decision = {"decision_id": "decision", "parent_prefix_sha256": "prefix"}
    reader = CandidateIdentityReader(tmp_path, "tokens")
    result = reader.read(c, decision)
    assert result["reasoning_hash"] == hash_json([20, 21])
    request["output_token_ids"][-1] = 100  # Action tokens do not enter reasoning identity.
    write_json(tmp_path / "request.json", request)
    c["ledger_sha256"] = sha256(tmp_path / "request.json")
    assert reader.read(c, decision)["reasoning_hash"] == result["reasoning_hash"]
    (tmp_path / "request.json").write_text('{}')
    with pytest.raises(ValueError, match="checksum"):
        reader.read(c, decision)


@pytest.fixture
def scored(published):
    cfg, catalog, session, ds, seal = published
    for d in ds:
        d["parent_prefix_sha256"] = "prefix"
    for i, c in enumerate(ds[0]["candidates"]):
        c["response"] = candidate(json.dumps({"command": "ls" if i < 2 else "pwd"}), reasoning=f"reason-{i}")["response"]
        c["score"] = [0., 2., 6., 10.][i]
    seal()
    return cfg, catalog, session, ds, seal


def test_workflow_reuses_plot_alignment_and_publishes_receipt(scored):
    cfg, _, _, _, _ = scored
    result = evaluate_parallel_instability(cfg, "run", "metrics", task_ids=["repo__task-1"], bootstrap_samples=20)
    out = Path(result["output"])
    summary = read_json(out / "summary.json")
    assert result["primary"]["mean_pair_gap"]["mean"] == 3.
    assert result["primary"]["relative"]["ratio"] == pytest.approx(3/7)
    assert summary["coverage"]["branch_candidates"] == 6  # No format/terminal pseudo-step.
    assert summary["coverage"]["identity_status_counts"]["available"] == 4
    assert summary["views"]["deduplicated"]["groups"]["supported_groups"] == 2
    for name, digest in read_json(out / "receipt.json")["files"].items():
        assert sha256(out / name) == digest
    plot_data = collect_parallel_scores(cfg, "run", "r-one")
    assert "action_hash" not in plot_data["branches"][0]  # Plot API remains backward compatible.
    with pytest.raises(FileExistsError):
        evaluate_parallel_instability(cfg, "run", "metrics", rollout_ids=["r-one"])


def test_capture_query_counter_may_differ_after_format_retries(scored):
    cfg, _, _, decisions, seal = scored
    decisions[0]["step_id"] = 7
    seal()
    data = collect_parallel_scores(cfg, "run", "r-one", identity_mode="text")
    assert data["steps"][0]["step_id"] == 0
    assert data["steps"][0]["query_step_id"] == 7
    assert all(r["query_step_id"] == 7 for r in data["branches"] if r["step_id"] == 0)
    result = evaluate_parallel_instability(cfg, "run", "retry-counter", bootstrap_samples=0)
    assert result["primary"]["mean_pair_gap"]["mean"] == 3.


def test_no_multipliers_for_native_nonlog_scores(scored):
    cfg = scored[0]
    result = evaluate_parallel_instability(cfg, "run", "native", score_space="native", bootstrap_samples=0)
    summary = read_json(Path(result["output"]) / "summary.json")
    assert not summary["multiplier_rates_available"]
    assert "pair_rate_ge_2.0x" not in result["primary"]


def test_unavailable_and_corruption_are_not_conflated(scored):
    cfg, _, session, _, _ = scored
    decision = session / "parallel/decisions/z-step0.json"
    original = decision.read_bytes()
    decision.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_parallel_instability(cfg, "run", "bad", skip_unavailable=True)
    assert not (Path(cfg["project_root"]) / "difficulty/evaluations/bad").exists()
    decision.write_bytes(original)
    (session / "finalization.json").unlink()
    with pytest.raises(ValueError, match="No published"):
        evaluate_parallel_instability(cfg, "run", "missing", skip_unavailable=True)


@pytest.mark.parametrize("kwargs", [{"multipliers": [1.]}, {"multipliers": [float("nan")]},
                                     {"bootstrap_samples": -1}, {"seed": -1},
                                     {"rollout_ids": ["r-one", "r-one"]}])
def test_invalid_parameters(scored, kwargs):
    with pytest.raises(ValueError):
        evaluate_parallel_instability(scored[0], "run", "invalid", **kwargs)
