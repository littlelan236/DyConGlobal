"""Real CPU capture receipts plus mocked generation; never starts serving/GPU/tools."""

import copy

import pytest

from hidden.backends.vllm.online import capture_request_metadata
from rollout.consensus import consensus_options, validate_consensus_evidence
from rollout.decision.coordinator import DecisionCoordinator, ParallelSelectionError
from rollout.entropy import RETENTION_RULES
from rollout.loop_escape import LoopEscapeHistory
from runtime.serving.capture import OnlineCaptureController
from runtime.serving.parallel_validation import validate_selection
from tests.test_parallel_loop_escape import Response, parse


class Client:
    def __init__(self):
        self.calls = []
        self.scores = {i: float(i) for i in range(4)}

    def call(self, action, payload):
        self.calls.append((action, copy.deepcopy(payload)))
        if action == "branch-acquire":
            return {"acquired": True}
        if action != "results":
            return {}
        return {"results": [{"branch_id": i, "status": "scored", "score": value}
                            for i, value in self.scores.items()],
                "estimator_sha256": "sha", "target": "target", "endpoint": "submission"}


class Integration:
    def __init__(self, tmp_path, rule, policy, adapter="mini", loop_escape=False):
        self.controller = OnlineCaptureController(tmp_path, {"export": {"root": "online"}})
        self.metadata = capture_request_metadata("session", "task", "run", "pending", 0, ["think_close"])
        self.controller.start(self.metadata)
        self.cfg = {"parallel": {"mode": "parallel_enthophy" if rule else "parallel_consensus",
                                 "branch_count": 4, "fallback_policy": "stop", "score_timeout_ms": 200,
                                 "dynamic_branch_admission": True, "loop_escape": {"enabled": loop_escape},
                                 "consensus": {"retention_ratio": .5, "min_retained_branches": 2,
                                               "representative_policy": policy}},
                    "generation": {"seed": 42}, "capture": {"control_seconds": 2}}
        if rule:
            self.cfg["parallel"]["entropy"] = {"retention_rule": rule}
        self.client, self.history = Client(), LoopEscapeHistory()
        self.adapter = adapter
        self.lengths = [8, 4, 2, 1]

    def run(self, commands=None, step=0):
        commands = commands or ["A", "A", "A", "B"]
        metadata = {**self.metadata, "step_id": step, "query_id": "query-" + str(step)}

        def generate(messages, md, seed):
            bid = md["branch_id"]
            assert seed == 42 + bid
            native = "native-" + md["decision_id"] + "-" + str(bid)
            full_md = {**md, "prompt_token_ids": [10, 20]}
            self.controller.bind(native, full_md)
            self.controller.record_output({"request_id": native, "metadata": full_md,
                                           "prompt_token_ids": [10, 20],
                                           "output_token_ids": [30] * self.lengths[bid], "finish_reason": "stop"})
            response = Response(bid, commands[bid], "reason" * self.lengths[bid])
            response.value["usage"]["completion_tokens"] = self.lengths[bid]
            return response

        if self.adapter == "mini":
            from rollout.adapters.minisweagent import MiniPolicy

            policy, parser = MiniPolicy(), parse
        else:
            from rollout.adapters.openhands import OpenHandsPolicy, parse_response

            request = {"tools": [{"type": "function", "function": {"name": "bash"}}]}
            policy, parser = OpenHandsPolicy(), lambda value: parse_response(value, request)
        parser = getattr(self, "parse_override", parser)
        coordinator = DecisionCoordinator(self.cfg, self.controller, self.client, error_policy=policy,
                                          loop_escape_history=self.history)
        return coordinator.run([], metadata, generate, parser, lambda _: {"cost": .01})


@pytest.mark.parametrize("rule", [None, *RETENTION_RULES])
@pytest.mark.parametrize("representative", ["estimator_score", "min_step_tokens"])
@pytest.mark.parametrize("adapter", ["mini", "openhands"])
def test_every_combination_uses_exact_native_receipts(tmp_path, rule, representative, adapter):
    harness = Integration(tmp_path, rule, representative, adapter)
    try:
        value = harness.run()
        summary = value["extra"]["dycon_parallel"]
        assert summary["selection_mode"] == harness.cfg["parallel"]["mode"]
        retained = summary["consensus"]["retained_branch_ids"]
        a_group = [i for i in retained if i < 3]
        expected = min(a_group) if representative == "estimator_score" else max(a_group)
        assert summary["selected_branch_id"] == expected
        assert summary["total_usage"]["completion_tokens"] == sum(harness.lengths)
        for row in summary["candidates"]:
            assert row["completion_tokens"] == harness.lengths[row["branch_id"]]
            assert row["native_request_id"] and row["ledger_sha256"] and row["output_sha256"]
        if representative == "min_step_tokens":
            assert all(r["length_source"] == "native_receipt_output_token_ids" for r in summary["candidates"])
        validate_consensus_evidence(summary, consensus_options(harness.cfg["parallel"]))
        validate_selection(summary, 42, harness.cfg["parallel"])
        assert len(harness.controller.parallel_decisions) == 1
        assert [call for call, _ in harness.client.calls].count("release") == 1
    finally:
        harness.controller.close()


@pytest.mark.parametrize("rule", RETENTION_RULES)
@pytest.mark.parametrize("representative", ["estimator_score", "min_step_tokens"])
def test_loop_escape_pool_is_the_actual_entropy_input_and_failed_exclusions_restore(tmp_path, rule, representative):
    harness = Integration(tmp_path, rule, representative, loop_escape=True)
    try:
        harness.run(["A", "B", "C", "D"])
        value = harness.run(["A", "X", "Y", "D"], step=1)
        summary = value["extra"]["dycon_parallel"]
        assert summary["loop_escape"]["candidate_ids_after"] == [1, 2]
        assert summary["consensus"]["entropy"]["candidate_branch_ids"] == [1, 2]
        validate_consensus_evidence(summary)
        value = harness.run(["A", "X", "NEW", "D"], step=2)
        summary = value["extra"]["dycon_parallel"]
        assert summary["loop_escape"]["failure_reason"] == "selection_unavailable_after_filter"
        assert summary["loop_escape"]["candidate_ids_after"] == [0, 1, 2, 3]
        assert summary["consensus"]["entropy"]["eligible_count"] == 4
        validate_consensus_evidence(summary)
    finally:
        harness.controller.close()


def test_prepare_lengths_does_not_publish_a_winner_and_rechecks_native_proof(tmp_path):
    harness = Integration(tmp_path, RETENTION_RULES[0], "min_step_tokens")
    prepare = harness.controller.prepare_decision
    seen = []

    def checked(decision, timeout=30):
        assert not harness.controller.parallel_decisions
        result = prepare(decision, timeout)
        seen.append(decision["selected_branch_id"])
        if decision["selected_branch_id"] is None:
            result["candidates"][0]["completion_tokens"] += 1
        return result

    harness.controller.prepare_decision = checked
    try:
        with pytest.raises(ValueError, match="native_length_or_output_conflict"):
            harness.run()
        assert seen == [None]
        assert not harness.controller.parallel_decisions
        assert [call for call, _ in harness.client.calls].count("release") == 1
    finally:
        harness.controller.close()


@pytest.mark.parametrize("threshold,expected", [(1, 4), (2, 2)])
def test_coordinator_propagates_custom_full_retention_threshold(tmp_path, threshold, expected):
    harness = Integration(tmp_path, "effective_count_full_above_v1", "estimator_score")
    harness.cfg["parallel"]["entropy"]["full_retention_threshold"] = threshold
    try:
        summary = harness.run()["extra"]["dycon_parallel"]
        assert summary["consensus"]["retained_count"] == expected
        assert summary["consensus"]["entropy"]["full_retention_threshold"] == threshold
        validate_consensus_evidence(summary, consensus_options(harness.cfg["parallel"]))
        validate_selection(summary, 42, harness.cfg["parallel"])
    finally:
        harness.controller.close()


def test_entropy_empty_pool_retains_original_failure_and_auditable_unavailable_entropy(tmp_path):
    harness = Integration(tmp_path, RETENTION_RULES[0], "estimator_score")
    harness.client.scores = {i: float("nan") for i in range(4)}
    try:
        with pytest.raises(ParallelSelectionError):
            harness.run()
        summary = next(iter(harness.controller.parallel_decisions.values()))
        assert summary["selected_branch_id"] is None
        assert summary["consensus"]["entropy"]["entropy"] is None
        assert summary["consensus"]["entropy"]["status"] == "unavailable"
    finally:
        harness.controller.close()


@pytest.mark.parametrize("rule", RETENTION_RULES)
@pytest.mark.parametrize("representative", ["estimator_score", "min_step_tokens"])
def test_entropy_preserves_mini_format_correction_instead_of_relabeling_it(tmp_path, rule, representative):
    from minisweagent.exceptions import FormatError

    harness = Integration(tmp_path, rule, representative)

    def invalid_format(response):
        raise FormatError({"role": "user", "content": "invalid tool format"})

    harness.parse_override = invalid_format
    try:
        with pytest.raises(FormatError):
            harness.run()
        summary = next(iter(harness.controller.parallel_decisions.values()))
        assert summary["selected_branch_id"] == 0
        assert summary["fallback_reason"] == "all_branches_format_error"
        assert summary["consensus"]["entropy"]["entropy"] is None
        validate_consensus_evidence(summary)
    finally:
        harness.controller.close()


@pytest.mark.parametrize("adapter", ["mini", "openhands"])
def test_plus_n_custom_offset_flows_through_coordinator_and_audit(tmp_path, adapter):
    harness = Integration(tmp_path, "effective_count_plus_n_v1", "estimator_score", adapter)
    harness.cfg["parallel"]["entropy"]["n"] = 6
    try:
        summary = harness.run()["extra"]["dycon_parallel"]
        assert summary["consensus_config"]["n"] == 6
        assert summary["consensus"]["entropy"]["base_keep_count"] == 2
        assert summary["consensus"]["retained_count"] == 4
        validate_selection(summary, 42, harness.cfg["parallel"])
    finally:
        harness.controller.close()
