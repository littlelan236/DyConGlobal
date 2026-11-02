"""CPU-only coverage for Shannon rules, canonical voting and both representatives."""

import copy
import json
import math
from types import SimpleNamespace

import pytest

from core.io import hash_json
from rollout.consensus import consensus_options, select_consensus, validate_consensus_evidence
from rollout.entropy import RETENTION_RULES, entropy_retention, stable_ceil


class Candidate:
    def __init__(self, branch, command="ls", arguments=None, reasoning=None):
        self.branch_id = branch
        self.message = {"ok": True}
        self.raw = {"choices": [{"message": {"role": "assistant", "reasoning_content": reasoning,
                   "tool_calls": [{"id": "call-" + str(branch), "function": {
                       "name": "bash", "arguments": arguments or json.dumps({"command": command})}}]}}]}
        self.response = SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(self.raw))


EXAMPLES = [([32], [2, 2, 16]), ([31, 1], [2, 6, 19]), ([24, 8], [2, 15, 23]),
            ([16, 16], [2, 17, 24]), ([8] * 4, [4, 25, 28]), ([4] * 8, [8, 29, 30]),
            ([2] * 16, [16, 31, 31]), ([1] * 32, [32, 32, 32])]


@pytest.mark.parametrize("counts,expected", EXAMPLES)
def test_all_documented_shannon_examples(counts, expected):
    for rule, keep in zip(RETENTION_RULES, expected):
        meta = entropy_retention(counts, rule=rule)
        assert meta["keep_count"] == keep
        h = -math.fsum((x / 32) * math.log(x / 32) for x in counts)
        assert meta["entropy"] == pytest.approx(h)
        assert meta["definition"] == "shannon_nats"
        json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("rule", RETENTION_RULES)
def test_entropy_missing_and_insufficient_not_zero(rule):
    assert entropy_retention([], rule=rule)["entropy"] is None
    assert entropy_retention([], rule=rule)["status"] == "unavailable"
    assert entropy_retention([1], rule=rule)["keep_count"] == 0
    assert entropy_retention([1], rule=rule)["status"] == "insufficient_branches"
    assert entropy_retention([32], rule=rule)["entropy"] == 0


def test_rule_boundaries_and_original_ratio_is_used_only_by_c():
    for rule in RETENTION_RULES:
        previous = None
        for singleton_count in range(32):
            counts = [32 - singleton_count] + [1] * singleton_count
            meta = entropy_retention(counts, rule=rule)
            assert 2 <= meta["keep_count"] <= 32
            if previous:
                assert meta["entropy"] >= previous["entropy"]
                assert meta["keep_count"] >= previous["keep_count"]
            previous = meta
    assert stable_ceil(math.nextafter(4.0, math.inf)) == 4
    assert stable_ceil(4.000001) == 5
    for rule in RETENTION_RULES[:2]:
        assert entropy_retention([16, 16], rule=rule, retention_ratio=.1)["keep_count"] == (
            entropy_retention([16, 16], rule=rule, retention_ratio=.9)["keep_count"])
    assert entropy_retention([32], rule=RETENTION_RULES[2], retention_ratio=.25)["keep_count"] == 8


@pytest.mark.parametrize("kwargs", [{"rule": "other"}, {"counts": [0]}, {"counts": [True]},
                                   {"minimum": 1}, {"retention_ratio": float("nan")}])
def test_invalid_retention_inputs(kwargs):
    with pytest.raises(ValueError):
        entropy_retention(**{"counts": [32], **kwargs})


@pytest.mark.parametrize("counts,threshold,expected", [
    ([32], 10, 2), ([4] * 2 + [3] * 8, 10, 10),
    ([4] * 10, 10, 10), ([3] * 10 + [2], 10, 32),
    ([4] * 8, 7, 32), ([4] * 8, 8, 8),
    ([2] * 6, 4, 12), ([], 10, 0), ([1], 10, 0),
])
def test_effective_count_full_above_uses_final_integer_budget(counts, threshold, expected):
    base = entropy_retention(counts, rule="effective_count_v1")
    result = entropy_retention(counts, rule="effective_count_full_above_v1",
                               full_retention_threshold=threshold)
    assert result["base_keep_count"] == base["keep_count"]
    assert result["raw_keep"] == base["raw_keep"]
    assert result["entropy"] == base["entropy"]
    assert result["keep_count"] == expected
    assert result["full_retention_triggered"] == (base["keep_count"] > threshold)


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True, None, float("inf"), float("nan")])
def test_effective_count_full_above_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="full_retention_threshold"):
        entropy_retention([32], rule="effective_count_full_above_v1",
                          full_retention_threshold=threshold)


def test_threshold_decision_replays_custom_value_and_rejects_tampering():
    candidates = [Candidate(i, "A" if i < 3 else "B") for i in range(4)]
    scores = {i: float(i) for i in range(4)}
    opts = consensus_options({"mode": "parallel_enthophy", "entropy": {
        "retention_rule": "effective_count_full_above_v1", "full_retention_threshold": 1}})
    winner, meta = select_consensus(candidates, scores,
        **{k: v for k, v in opts.items() if k != "rule"})
    assert meta["retained_count"] == 4
    by_id = {r["branch_id"]: r for r in meta["branch_metadata"]}
    decision = {"selection_mode": "parallel_enthophy", "consensus_config": opts,
                "consensus": meta, "selected_branch_id": winner.branch_id,
                "candidates": [{**by_id[c.branch_id], "status": "eligible", "response": c.raw}
                               for c in candidates]}
    validate_consensus_evidence(decision, opts)
    compact = copy.deepcopy(decision)
    for row in compact["candidates"]:
        row.pop("response")
    validate_consensus_evidence(compact, opts)
    compact["consensus"]["entropy"]["full_retention_threshold"] = 2
    with pytest.raises(ValueError, match="evidence differs"):
        validate_consensus_evidence(compact, opts)


@pytest.mark.parametrize("rule", [None, *RETENTION_RULES])
@pytest.mark.parametrize("policy", ["estimator_score", "min_step_tokens"])
def test_all_eight_fixed_dynamic_and_representative_combinations(rule, policy):
    candidates = [Candidate(i, reasoning="r" * (20 - i)) for i in range(4)]
    scores = {i: float(i) for i in range(4)}
    lengths = {0: 100, 1: 30, 2: 20, 3: 10}
    winner, meta = select_consensus(candidates, scores, retention_rule=rule,
                                    representative_policy=policy, token_lengths=lengths)
    kept = meta["retained_branch_ids"]
    expected = min(kept) if policy == "estimator_score" else max(kept)
    assert winner.branch_id == expected
    assert meta["consensus_vote_fraction"] == 1
    if rule:
        assert meta["entropy"]["entropy"] == 0
        assert meta["entropy"]["candidate_branch_ids"] == list(range(4))


def test_current_identity_normalizes_arguments_but_not_command_contents():
    candidates = [Candidate(0, arguments='{"command":"ls -l", "x":1}', reasoning="long"),
                  Candidate(1, arguments=' { "x" : 1, "command" : "ls -l" } ', reasoning="short"),
                  Candidate(2, command="ls  -l")]
    _, meta = select_consensus(candidates, {0: 0., 1: 1., 2: 2.}, retention_rule=RETENTION_RULES[0])
    assert meta["entropy"]["group_sizes"] == [2, 1]
    assert meta["entropy"]["entropy"] == pytest.approx(-2/3*math.log(2/3)-1/3*math.log(1/3))


def test_dynamic_excludes_invalid_identity_before_entropy_and_keeps_boundary_count_exact():
    candidates = [Candidate(i) for i in range(4)]
    candidates[0].raw["choices"][0]["message"]["tool_calls"] = []
    winner, meta = select_consensus(candidates, {i: 1. for i in range(4)}, retention_rule=RETENTION_RULES[0])
    assert winner.branch_id == 1
    assert meta["retained_branch_ids"] == [1, 2]
    assert meta["entropy"]["eligible_count"] == 3
    assert meta["entropy"]["cutoff_branch_id"] == 2
    assert meta["branch_metadata"][0]["exclusion_reason"] == "action_identity_unavailable"


def test_vote_group_tie_break_unchanged_and_no_rescue_from_filtered_branches():
    candidates = [Candidate(0, "A"), Candidate(1, "A"), Candidate(2, "B"), Candidate(3, "B")]
    scores = {0: 0., 1: 3., 2: 1., 3: 2.}
    winner, _ = select_consensus(candidates, scores, 1., representative_policy="min_step_tokens",
                                 token_lengths={0: 100, 1: 50, 2: 1, 3: 1})
    assert winner.branch_id == 1
    winner, _ = select_consensus(candidates, scores, .5, representative_policy="min_step_tokens",
                                 token_lengths={0: 100, 1: 1, 2: 10, 3: 1})
    assert winner.branch_id == 0  # A and B tie; A has the better score; b1 was filtered.


def test_missing_length_never_silently_falls_back_but_score_policy_ignores_it():
    candidates = [Candidate(0), Candidate(1)]
    winner, meta = select_consensus(candidates, {0: 0., 1: 1.},
                                    representative_policy="min_step_tokens", token_lengths={0: 2})
    assert winner is None
    assert meta["status"] == "token_length_unavailable"
    assert meta["representative"]["missing_length_branch_ids"] == [1]
    winner, _ = select_consensus(candidates, {0: 0., 1: 1.}, representative_policy="estimator_score")
    assert winner.branch_id == 0


def make_evidence(rule=RETENTION_RULES[1], policy="min_step_tokens"):
    candidates = [Candidate(i, "A" if i < 3 else "B") for i in range(4)]
    scores = {i: float(i) for i in range(4)}
    lengths = {0: 50, 1: 10, 2: 5, 3: 2}
    winner, meta = select_consensus(candidates, scores, retention_rule=rule,
                                    representative_policy=policy, token_lengths=lengths)
    config = consensus_options({"mode": "parallel_enthophy", "consensus": {"representative_policy": policy},
                                "entropy": {"retention_rule": rule}})
    by_id = {m["branch_id"]: m for m in meta["branch_metadata"]}
    rows = [{**by_id[c.branch_id], "status": "eligible", "response": c.raw,
             "completion_tokens": lengths[c.branch_id]} for c in candidates]
    return {"selection_mode": "parallel_enthophy", "consensus_config": config, "consensus": meta,
            "candidates": rows, "selected_branch_id": winner.branch_id}


def test_full_and_compact_evidence_replay_identically():
    full = make_evidence()
    validate_consensus_evidence(full)
    compact = copy.deepcopy(full)
    for row in compact["candidates"]:
        row.pop("response")
    validate_consensus_evidence(compact)
    assert hash_json(full["consensus"]) == hash_json(compact["consensus"])


@pytest.mark.parametrize("tamper", ["entropy", "groups", "k", "length", "winner", "policy", "rule", "action"])
def test_new_receipt_tampering_is_rejected(tamper):
    decision = make_evidence()
    expected_options = copy.deepcopy(decision["consensus_config"])
    if tamper == "entropy":
        decision["consensus"]["entropy"]["entropy"] = .7
    elif tamper == "groups":
        decision["consensus"]["entropy"]["groups"][0]["count"] += 1
    elif tamper == "k":
        decision["consensus"]["entropy"]["keep_count"] -= 1
    elif tamper == "length":
        decision["candidates"][0]["completion_tokens"] = 1
    elif tamper == "winner":
        decision["selected_branch_id"] = 3
    elif tamper == "policy":
        decision["consensus_config"]["representative_policy"] = "estimator_score"
    elif tamper == "rule":
        decision["consensus_config"]["retention_rule"] = RETENTION_RULES[0]
    else:
        decision["candidates"][0]["response"]["choices"][0]["message"]["tool_calls"] = []
    with pytest.raises(ValueError):
        validate_consensus_evidence(decision, expected_options)


def test_action_analysis_keeps_raw_decision_entropy_and_handles_mixed_csv_columns(tmp_path):
    from action.analyzers.branches import decision_record, step_statistics, summarize, write_csv

    full = make_evidence()
    full.update(branch_count=4, session_id="session", decision_id="decision", step_id=0)
    message = full["candidates"][full["selected_branch_id"]]["response"]["choices"][0]["message"]
    binding = {"source_run_id": "source", "rollout_id": "trajectory", "instance_id": "task", "cohort": None,
               "assistant_index": 0, "message_index": 0}
    record = decision_record(full, full, message, "both", binding, {"kind": "full"})
    row = step_statistics(record, "both")
    assert row["decision_pool_entropy"] == full["consensus"]["entropy"]["entropy"]
    assert row["retention_rule"] == "shannon_exp_keep_v1"
    assert row["representative_policy"] == "min_step_tokens"
    legacy = {k: v for k, v in row.items() if k not in {"representative_policy", "retention_rule",
                                                     "decision_pool_entropy", "decision_pool_count", "decision_keep_count"}}
    write_csv(tmp_path / "mixed.csv", [legacy, row])
    summary = summarize([legacy, row], [{**record, "parallel_steps": 1}], "both")
    assert "min_step_tokens" in summary["by_representative_policy"]
    assert "shannon_exp_keep_v1" in summary["by_retention_rule"]


@pytest.mark.parametrize("counts,minimum,n,expected", [
    ([32], 2, 6, 8), ([16, 16], 2, 6, 8), ([8] * 4, 2, 6, 10),
    ([2] * 16, 2, 6, 22), ([1] * 32, 2, 6, 32), ([32], 4, 6, 10),
    ([3, 2], 2, 6, 5), ([], 2, 6, 0), ([1], 2, 6, 0),
    ([32], 2, 0, 2), ([24, 8], 2, 1, 3),
])
def test_plus_n_adds_after_original_floor_and_rounding(counts, minimum, n, expected):
    base = entropy_retention(counts, minimum, "effective_count_v1")
    result = entropy_retention(counts, minimum, "effective_count_plus_n_v1", n=n)
    assert result["base_keep_count"] == base["keep_count"]
    assert result["raw_keep"] == base["raw_keep"]
    assert result["entropy"] == base["entropy"]
    assert result["status"] == base["status"]
    assert result["n"] == n
    assert result["keep_count"] == expected


@pytest.mark.parametrize("n", [-1, 1.5, True, None, "6", float("inf"), float("nan")])
def test_plus_n_rejects_invalid_offsets(n):
    with pytest.raises(ValueError, match="n must"):
        entropy_retention([32], rule="effective_count_plus_n_v1", n=n)


def test_plus_n_selection_and_compact_audit_replay_the_frozen_offset():
    candidates = [Candidate(i, "A" if i < 8 else "B") for i in range(10)]
    opts = consensus_options({"mode": "parallel_enthophy", "entropy": {
        "retention_rule": "effective_count_plus_n_v1", "n": 6}})
    winner, meta = select_consensus(candidates, {i: float(i) for i in range(10)},
        **{k: v for k, v in opts.items() if k != "rule"})
    assert meta["retained_branch_ids"] == list(range(8))
    by_id = {row["branch_id"]: row for row in meta["branch_metadata"]}
    decision = {"selection_mode": "parallel_enthophy", "consensus_config": opts,
                "consensus": meta, "selected_branch_id": winner.branch_id,
                "candidates": [{**by_id[c.branch_id], "status": "eligible", "response": c.raw}
                               for c in candidates]}
    validate_consensus_evidence(decision, opts)
    for row in decision["candidates"]:
        row.pop("response")
    validate_consensus_evidence(decision, opts)
    decision["consensus"]["entropy"]["n"] = 5
    with pytest.raises(ValueError, match="evidence differs"):
        validate_consensus_evidence(decision, opts)
