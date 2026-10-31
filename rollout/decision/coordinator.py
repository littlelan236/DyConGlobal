"""One logical query, concurrent native n=1 completions, one action winner."""

from __future__ import annotations

import copy
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.cancellation import Cancelled, CompletionWake
from dataclasses import dataclass
from uuid import uuid4

import requests

from core.io import hash_json
from rollout.consensus import CONSENSUS_MODES, consensus_options, select_consensus
from rollout.loop_escape import LoopEscapeHistory, apply_escape, prepare_escape, selection_failure


class ParallelSelectionError(RuntimeError):
    pass


RANDOM_SELECTION_MODE = "uniform_random_non_loop_v1"


def baseline_selection_seed(seed, task_id, step_id, parent_prefix_sha256):
    """Bind selection to frozen context, never scores or transport UUIDs."""
    return hash_json({"algorithm": RANDOM_SELECTION_MODE, "seed": seed,
                      "task_id": task_id, "step_id": step_id,
                      "parent_prefix_sha256": parent_prefix_sha256})


def random_branch_id(seed, branch_ids):
    """Replay a uniform draw in canonical branch order with a local RNG."""
    return random.Random(int(seed, 16)).choice(sorted(branch_ids))


@dataclass
class BranchCandidate:
    branch_id: int
    metadata: dict
    response: object = None
    message: dict | None = None
    error: Exception | None = None
    cost: float = 0.0


class ServingParallelClient:
    def __init__(self, cfg):
        self.base = cfg["serving"]["base_url"].removesuffix("/v1").rstrip("/")
        self.headers = {"Authorization": "Bearer " + os.environ.get(
            cfg["serving"]["api_key_env"], "EMPTY")}
        self.timeout = cfg.get("parallel", {}).get("control_request_seconds") or (cfg["timeouts"]["metadata_request_seconds"] + 35)
        self.admission_timeout = cfg["parallel"].get("admission_timeout_seconds", 1800) + 5

    def call(self, action, payload):
        response = requests.post(self.base + "/dycon/parallel/" + action,
                                 json=payload, timeout=self.admission_timeout if action in {"reserve", "branch-acquire"} else self.timeout,
                                 headers=self.headers)
        response.raise_for_status()
        return response.json()


def active(cfg, step):
    opts = cfg.get("parallel", {})
    if not opts.get("enabled") or opts.get("branch_count", 1) == 1:
        return False
    return (opts["policy"] == "every_step" or
            (opts["policy"] == "first_step" and step == 0) or
            (opts["policy"] == "every_k_steps" and step % opts["every_k_steps"] == 0))


class DecisionCoordinator:
    def __init__(self, cfg, controller, client=None, *, diagnostic_mode=None, diagnostic_branches=None,
                 loop_escape_history=None, cancellation=None, error_policy=None):
        self.cancellation = cancellation
        if error_policy is None:
            raise ValueError("An agent error policy is required")
        self.error_policy = error_policy
        self.cfg, self.controller = cfg, controller
        self.options = cfg["parallel"]
        self.loop_escape_history = loop_escape_history if loop_escape_history is not None else LoopEscapeHistory()
        self.client = client or ServingParallelClient(cfg)
        self.diagnostic_mode = diagnostic_mode
        self.branches = diagnostic_branches or self.options["branch_count"]
        if diagnostic_mode not in {None, "first_valid_unscored"}:
            raise ValueError("Unknown diagnostic branch selection mode")
        if not 1 <= self.branches <= self.options["branch_count"]:
            raise ValueError("Invalid diagnostic branch count")

    def run(self, messages, capture, generate, parse, calculate_cost, *, request_envelope=None):
        if not capture or self.controller is None:
            raise ParallelSelectionError("Parallel rollout requires a live capture controller")
        if self.cancellation is not None:
            self.cancellation.check()
        opts = self.options
        baseline = opts.get("mode", "parallel") == "parallel_baseline"
        consensus = opts.get("mode") in CONSENSUS_MODES and self.diagnostic_mode is None
        consensus_config = consensus_options(opts) if consensus else None
        random_selection = baseline and self.diagnostic_mode is None
        decision = "dec-" + uuid4().hex
        decision_deadline = time.time() + opts.get("decision_ttl_seconds", 28860)
        prefix = hash_json(request_envelope if request_envelope is not None else messages)
        candidates = []
        for branch in range(self.branches):
            metadata = {**copy.deepcopy(capture), "decision_id": decision, "branch_id": branch,
                        "branch_count": self.branches, "parent_prefix_sha256": prefix,
                        "decision_deadline_unix": decision_deadline,
                        "api_attempt_id": uuid4().hex}
            candidates.append(BranchCandidate(branch, metadata))
            if self.diagnostic_mode:
                metadata["diagnostic_score_disabled"] = True
        admission_start = time.monotonic()
        self.client.call("reserve", {"decision_id": decision,
                                     "decision_deadline_unix": decision_deadline,
                                     "branches": [c.branch_id for c in candidates]})
        admission_ms = (time.monotonic() - admission_start) * 1000
        start = time.monotonic()
        wake = None
        try:
            def one(candidate):
                acquired = False
                try:
                    if opts.get("dynamic_branch_admission", False):
                        lease = self.client.call("branch-acquire", {"decision_id": decision, "branch_id": candidate.branch_id})
                        if lease.get("acquired") is not True:
                            raise Cancelled("Parallel branch did not acquire a live lease")
                        acquired = True
                    if self.cancellation is not None:
                        self.cancellation.check()
                    candidate.response = generate(messages, candidate.metadata,
                                                  self.cfg["generation"]["seed"] + candidate.branch_id)
                    candidate.cost = calculate_cost(candidate.response)["cost"]
                    if self.cancellation is not None:
                        self.cancellation.check()
                    if candidate.response.choices[0].finish_reason == "abort":
                        candidate.error = ParallelSelectionError("Native branch aborted without loop evidence")
                except Exception as exc:
                    candidate.error = exc
                finally:
                    if acquired:
                        try:
                            self.client.call("branch-release", {"decision_id": decision, "branch_id": candidate.branch_id,
                                                                "reason": "coordinator_complete"})
                        except Exception:
                            pass
                return candidate

            with ThreadPoolExecutor(max_workers=len(candidates), thread_name_prefix="dycon-branch") as pool:
                futures = [pool.submit(one, c) for c in candidates]
                if self.cancellation is None:
                    for future in as_completed(futures):
                        future.result()
                else:
                    wake = CompletionWake(self.cancellation)
                    pending = set(futures)
                    for future in futures:
                        future.add_done_callback(wake.notify)
                    while pending:
                        finished = wake.wait(pending, self.cancellation)
                        if self.cancellation.event.is_set():
                            for future in pending:
                                future.cancel()
                            break
                        for future in finished:
                            future.result()
                        pending.difference_update(finished)
            if self.cancellation is not None:
                self.cancellation.check()
            cancelled = next((c.error for c in candidates if isinstance(c.error, Cancelled)), None)
            if cancelled is not None:
                raise cancelled
            decode_ms = (time.monotonic() - start) * 1000
            wait_start = time.monotonic()
            score_error = None
            try:
                # Missing/pending scores are backpressure, not algorithmic failure.
                result_deadline = min(time.monotonic() + opts.get("results_wait_seconds", 0),
                                      time.monotonic() + max(0.0, decision_deadline - time.time()))
                expected = {c.branch_id for c in candidates if c.response is not None
                            and c.response.choices[0].finish_reason != "length"}
                while True:
                    result = self.client.call("results", {"decision_id": decision,
                                                         "timeout": opts["score_timeout_ms"] / 1000})
                    statuses = {int(row["branch_id"]): row.get("status") for row in result["results"]}
                    pending = any(statuses.get(b) in {None, "pending", "missing"} for b in expected)
                    if not pending or time.monotonic() >= result_deadline:
                        break
                    if self.cancellation is not None:
                        self.cancellation.check()
                    time.sleep(min(.05, max(0.0, result_deadline-time.monotonic())))
                if result["estimator_sha256"] != opts.get("estimator_sha256", result["estimator_sha256"]):
                    raise ParallelSelectionError("Serving predictor differs from run snapshot")
            except (requests.RequestException, KeyError, ValueError) as exc:
                score_error = repr(exc)
                result = {"results": [], "estimator_sha256": opts.get("estimator_sha256"),
                          "target": None, "endpoint": None}
            scores = {int(row["branch_id"]): row for row in result["results"]}
            records = []
            eligible = []
            formatted = []
            for candidate in candidates:
                score = scores.get(candidate.branch_id, {})
                loop = score.get("reasoning_loop") or {}
                loop_terminated = (score.get("reason") == "reasoning_loop" and loop.get("abort_completed") is True
                        and score.get("api_attempt_id") == candidate.metadata["api_attempt_id"]
                        and score.get("native_request_id") == loop.get("native_request_id")
                        and all(loop.get(k) == candidate.metadata[k] for k in (
                            "session_id", "decision_id", "branch_id", "api_attempt_id"))
                        and candidate.response is not None)
                if loop_terminated:
                    candidate.message = None
                    candidate.error = self.error_policy.limit_error({
                        "role": "exit", "content": "Parallel reasoning loop detected",
                        "extra": {"exit_status": "LimitsExceeded", "submission": "",
                                  "limit_reason": "parallel_branch_reasoning_loop", "reasoning_loop": loop},
                    })
                elif random_selection and (score_error or score.get("reason") == "reasoning_loop"):
                    # A normalized 'stop' can be a native abort. Never parse a
                    # partial tool response when its loop status is unverified.
                    candidate.error = ParallelSelectionError("Unverified reasoning-loop state")
                elif candidate.response is not None and candidate.error is None:
                    # LiteLLM normalizes native 'abort' to 'stop'. Bind native
                    # loop evidence before parsing any partial tool output.
                    try:
                        candidate.message = parse(candidate.response)
                    except Exception as exc:
                        candidate.error = exc
                row = {"branch_id": candidate.branch_id, "api_attempt_id": candidate.metadata["api_attempt_id"],
                       "score_identity": {k: score[k] for k in (
                           "session_id", "decision_id", "branch_id", "api_attempt_id", "step_id", "request_sha256") if k in score},
                       "seed": self.cfg["generation"]["seed"] + candidate.branch_id,
                       "native_request_id": score.get("native_request_id"),
                       "close_sample_id": score.get("sample_id"),
                       "absolute_position": score.get("absolute_position"), "token_id": score.get("token_id"),
                       "score": score.get("score"), "predict_ms": score.get("predict_ms"),
                       "score_queue_ms": score.get("queue_ms"),
                       "status": "ineligible", "reason": score.get("reason", score.get("status", "missing")),
                       "response": candidate.response.model_dump(mode="json") if candidate.response else None,
                       "cost": candidate.cost}
                if type(row["score"]) not in (int, float) or not math.isfinite(row["score"]):
                    row["score"] = None
                    if score.get("status") == "scored":
                        row["reason"] = "invalid_score"
                if loop_terminated:
                    row["reasoning_loop"] = loop
                    row["native_finish_reason"] = "abort"
                if candidate.message is not None:
                    formatted.append(candidate)
                    if self.diagnostic_mode and score.get("status") == "unscored":
                        row.update(status="unscored_valid", reason="diagnostic_scoring_disabled")
                    if (score.get("status") == "scored" and isinstance(row["score"], (int, float))
                            and math.isfinite(row["score"])
                            and candidate.response.choices[0].finish_reason != "length"):
                        row.update(status="eligible", reason=None)
                        eligible.append(candidate)
                else:
                    row["reason"] = "reasoning_loop" if row.get("reasoning_loop") else (
                        type(candidate.error).__name__ if candidate.error else "missing_response")
                if random_selection:
                    row.update(scoring_status=score.get("status", "missing"),
                               scoring_reason=score.get("reason"),
                               selection_eligible=candidate.message is not None)
                    if candidate.message is not None:
                        row.update(status="eligible", reason=None)
                records.append(row)
            escape_meta = None
            if opts.get("loop_escape", {}).get("enabled") and self.diagnostic_mode is None:
                escape_meta = prepare_escape(records, {c.branch_id for c in formatted}, self.loop_escape_history)
                escape_meta.update(rollout_mode=opts.get("mode", "parallel"),
                                   fallback_policy=opts["fallback_policy"],
                                   consensus_config=consensus_config or opts.get("consensus", {}))
                failure = selection_failure(records, escape_meta, escape_meta["rollout_mode"],
                                            escape_meta["fallback_policy"], escape_meta["consensus_config"])
                apply_escape(records, escape_meta, failure)
                allowed = set(escape_meta["candidate_ids_after"])
                eligible = [c for c in eligible if c.branch_id in allowed]
                formatted = [c for c in formatted if c.branch_id in allowed]
            if consensus and consensus_config.get("representative_policy") == "min_step_tokens" and eligible:
                prepared = self.controller.prepare_decision({
                    "session_id": capture["session_id"], "decision_id": decision,
                    "step_id": capture["step_id"], "branch_count": len(candidates),
                    "decision_deadline_unix": decision_deadline,
                    "length_policy": "min_step_tokens",
                    "selected_branch_id": None, "candidates": records,
                }, self.cfg["capture"]["control_seconds"])
                records = prepared["candidates"]
            fallback = None
            selection_error = None
            consensus_meta = None
            selection_seed = None
            selection_ids = sorted(c.branch_id for c in formatted)
            if random_selection and score_error:
                winner = None
                selection_error = ParallelSelectionError(
                    "Cannot verify baseline reasoning-loop state: " + score_error)
            elif random_selection and formatted:
                selection_seed = baseline_selection_seed(
                    self.cfg["generation"]["seed"], capture["task_id"], capture["step_id"], prefix)
                selected = random_branch_id(selection_seed, selection_ids)
                winner = next(c for c in formatted if c.branch_id == selected)
            elif consensus and eligible:
                score_map = {c.branch_id: float(scores[c.branch_id]["score"]) for c in eligible}
                winner, consensus_meta = select_consensus(
                    eligible, score_map,
                    opts.get("consensus", {}).get("retention_ratio", 0.5),
                    opts.get("consensus", {}).get("min_retained_branches", 2),
                    opts.get("consensus", {}).get("tie_policy", "estimator_score"),
                    representative_policy=consensus_config.get("representative_policy"),
                    retention_rule=consensus_config.get("retention_rule"),
                    token_lengths={r["branch_id"]: r.get("completion_tokens") for r in records},
                )
                if winner is None:
                    selection_error = ParallelSelectionError("No exact-action consensus: " + consensus_meta["status"])
                else:
                    fallback = None
            elif eligible:
                winner = min(eligible, key=lambda c: (scores[c.branch_id]["score"], c.branch_id))
            elif self.diagnostic_mode == "first_valid_unscored" and formatted:
                winner = min(formatted, key=lambda c: c.branch_id)
            elif formatted and opts["fallback_policy"] == "first_valid":
                winner = min(formatted, key=lambda c: c.branch_id)
                fallback = "no_eligible_scored_branch"
            elif not formatted:
                winner, selection_error, fallback = self.error_policy.failure(
                    candidates, records, baseline, ParallelSelectionError)
            else:
                winner = None
                selection_error = ParallelSelectionError("No eligible scored parallel candidate")
            evidence = {"schema_version": 1, "session_id": capture["session_id"],
                        "decision_deadline_unix": decision_deadline,
                        "decision_id": decision, "step_id": capture["step_id"],
                        "query_id": capture["query_id"], "parent_prefix_sha256": prefix,
                        "branch_count": len(candidates), "transport": "concurrent_n1",
                        "selection_mode": self.diagnostic_mode or (
                            RANDOM_SELECTION_MODE if random_selection else "minimum_score"),
                        "selected_branch_id": winner.branch_id if winner else None,
                        "tie_rule": None if random_selection else "lowest_branch_id",
                        "estimator_sha256": result["estimator_sha256"], "target": result["target"],
                        "endpoint": result["endpoint"], "fallback_reason": fallback, "score_error": score_error,
                        "candidates": records, "decode_ms": decode_ms,
                        "admission_ms": admission_ms,
                        "response_to_selection_ms": (time.monotonic() - wait_start) * 1000,
                        "total_cost": sum(c.cost for c in candidates)}
            if escape_meta is not None:
                evidence["loop_escape"] = escape_meta
            if consensus:
                evidence["rollout_mode"] = opts["mode"]
                evidence["selection_mode"] = opts["mode"]
                evidence["tie_rule"] = "estimator_score"
                if consensus_meta is None:
                    consensus_meta = {
                        "status": "no_eligible_branch",
                        "retained_branch_ids": [],
                        "retained_count": 0,
                        "branch_metadata": [],
                        "votes": [],
                        "tie_break": None,
                        "unavailable": [],
                    }
                    if consensus_config.get("retention_rule"):
                        _, consensus_meta = select_consensus(
                            [], {}, consensus_config["retention_ratio"], consensus_config["min_retained_branches"],
                            consensus_config["tie_policy"], representative_policy=consensus_config.get("representative_policy"),
                            retention_rule=consensus_config["retention_rule"])
                metadata_by_branch = {
                    item["branch_id"]: item
                    for item in consensus_meta.get("branch_metadata", [])
                }
                for row in records:
                    item = metadata_by_branch.get(row["branch_id"])
                    if item:
                        row.update({
                            "estimator_rank": item.get("estimator_rank"),
                            "retained": item.get("retained", False),
                            "exclusion_reason": item.get("exclusion_reason"),
                            "identity_status": item.get("identity_status"),
                            "action_hash": item.get("action_hash"),
                            "action": item.get("action"),
                        })
                evidence["consensus"] = consensus_meta
                evidence["consensus_config"] = consensus_config
                if self.cfg.get("parallel_policy"):
                    evidence["policy_binding"] = {
                        key: self.cfg["parallel_policy"][key] for key in ("input_sha256", "effective_sha256")}
                    evidence["run_config_hash"] = self.cfg["config_hash"]
            if baseline:
                evidence["rollout_mode"] = "parallel_baseline"
            if random_selection:
                evidence.update(task_id=capture["task_id"], selection_seed=selection_seed,
                                selection_candidate_ids=selection_ids)
            evidence["total_usage"] = {
                key: sum(((row["response"] or {}).get("usage") or {}).get(key, 0) or 0 for row in records)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            # Binding native receipts is event-driven and never waits for NAS.
            # A receipt failure stops before tools are executed, retaining raw responses.
            if self.cancellation is not None:
                self.cancellation.check()
            evidence = self.controller.record_decision(evidence, self.cfg["capture"]["control_seconds"])
            if escape_meta is not None:
                self.loop_escape_history.remember(evidence)
            if selection_error is not None:
                if baseline:
                    for message in getattr(selection_error, "messages", []):
                        message.setdefault("extra", {}).update(
                            decision_id=decision, cost=evidence["total_cost"], total_usage=evidence["total_usage"])
                raise selection_error
            capture.update(winner.metadata)
            summary = {k: v for k, v in evidence.items() if k != "candidates"}
            summary["candidates"] = [{k: v for k, v in row.items() if k != "response"} for row in evidence["candidates"]]
            if winner.error is not None:
                for message in winner.error.messages:
                    message.setdefault("extra", {}).update(
                        dycon_parallel=summary, cost=evidence["total_cost"])
                raise winner.error
            message = winner.message
            message.setdefault("extra", {}).update(dycon_parallel=summary, cost=evidence["total_cost"])
            return message
        finally:
            if wake is not None:
                wake.close()
            try:
                self.client.call("release", {"decision_id": decision})
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Parallel reservation release failed; bounded TTL applies")
