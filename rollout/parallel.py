"""One logical query, concurrent native n=1 completions, one action winner."""

from __future__ import annotations

import copy
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import uuid4

import requests

from core.io import hash_json


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
        self.timeout = cfg["timeouts"]["metadata_request_seconds"] + 35
        self.admission_timeout = cfg["parallel"].get("admission_timeout_seconds", 1800) + 5

    def call(self, action, payload):
        response = requests.post(self.base + "/dycon/parallel/" + action,
                                 json=payload, timeout=self.admission_timeout if action == "reserve" else self.timeout,
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


class ParallelReasoningCoordinator:
    def __init__(self, cfg, controller, client=None, *, diagnostic_mode=None, diagnostic_branches=None):
        self.cfg, self.controller = cfg, controller
        self.options = cfg["parallel"]
        self.client = client or ServingParallelClient(cfg)
        self.diagnostic_mode = diagnostic_mode
        self.branches = diagnostic_branches or self.options["branch_count"]
        if diagnostic_mode not in {None, "first_valid_unscored"}:
            raise ValueError("Unknown diagnostic branch selection mode")
        if not 1 <= self.branches <= self.options["branch_count"]:
            raise ValueError("Invalid diagnostic branch count")

    def run(self, messages, capture, generate, parse, calculate_cost):
        if not capture or self.controller is None:
            raise ParallelSelectionError("Parallel rollout requires a live capture controller")
        opts = self.options
        baseline = opts.get("mode", "parallel") == "parallel_baseline"
        random_selection = baseline and self.diagnostic_mode is None
        decision = "dec-" + uuid4().hex
        prefix = hash_json(messages)
        candidates = []
        for branch in range(self.branches):
            metadata = {**copy.deepcopy(capture), "decision_id": decision, "branch_id": branch,
                        "branch_count": self.branches, "parent_prefix_sha256": prefix,
                        "api_attempt_id": uuid4().hex}
            candidates.append(BranchCandidate(branch, metadata))
            if self.diagnostic_mode:
                metadata["diagnostic_score_disabled"] = True
        admission_start = time.monotonic()
        self.client.call("reserve", {"decision_id": decision,
                                     "branches": [c.branch_id for c in candidates]})
        admission_ms = (time.monotonic() - admission_start) * 1000
        start = time.monotonic()
        try:
            def one(candidate):
                try:
                    candidate.response = generate(messages, candidate.metadata,
                                                  self.cfg["generation"]["seed"] + candidate.branch_id)
                    candidate.cost = calculate_cost(candidate.response)["cost"]
                    if candidate.response.choices[0].finish_reason == "abort":
                        candidate.error = ParallelSelectionError("Native branch aborted without loop evidence")
                except Exception as exc:
                    candidate.error = exc
                return candidate

            with ThreadPoolExecutor(max_workers=len(candidates), thread_name_prefix="dycon-branch") as pool:
                futures = [pool.submit(one, c) for c in candidates]
                for future in as_completed(futures):
                    future.result()
            decode_ms = (time.monotonic() - start) * 1000
            wait_start = time.monotonic()
            score_error = None
            try:
                result = self.client.call("results", {"decision_id": decision,
                                                     "timeout": opts["score_timeout_ms"] / 1000})
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
                    from minisweagent.exceptions import LimitsExceeded

                    candidate.message = None
                    candidate.error = LimitsExceeded({
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
                       "seed": self.cfg["generation"]["seed"] + candidate.branch_id,
                       "native_request_id": score.get("native_request_id"),
                       "close_sample_id": score.get("sample_id"),
                       "absolute_position": score.get("absolute_position"), "token_id": score.get("token_id"),
                       "score": score.get("score"), "predict_ms": score.get("predict_ms"),
                       "score_queue_ms": score.get("queue_ms"),
                       "status": "ineligible", "reason": score.get("reason", score.get("status", "missing")),
                       "response": candidate.response.model_dump(mode="json") if candidate.response else None,
                       "cost": candidate.cost}
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
            fallback = None
            selection_error = None
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
            elif eligible:
                winner = min(eligible, key=lambda c: (scores[c.branch_id]["score"], c.branch_id))
            elif self.diagnostic_mode == "first_valid_unscored" and formatted:
                winner = min(formatted, key=lambda c: c.branch_id)
            elif formatted and opts["fallback_policy"] == "first_valid":
                winner = min(formatted, key=lambda c: c.branch_id)
                fallback = "no_eligible_scored_branch"
            elif not formatted:
                from litellm.exceptions import ContextWindowExceededError
                from minisweagent.exceptions import FormatError, LimitsExceeded

                branch_errors = [c.error for c in candidates]
                # Preserve the existing agent FormatError correction contract.
                errors = [c for c in candidates if isinstance(c.error, FormatError)]
                if all(isinstance(error, (ContextWindowExceededError, LimitsExceeded))
                       for error in branch_errors):
                    winner = None
                    # Reuse the native exception so runner terminal categories and
                    # original error details survive. Context limits take priority
                    # in mixed limit groups; branch order breaks remaining ties.
                    selection_error = next(
                        (error for error in branch_errors if isinstance(error, ContextWindowExceededError)),
                        branch_errors[0],
                    )
                    if all(row["reason"] == "reasoning_loop" for row in records):
                        selection_error = LimitsExceeded({
                            "role": "exit", "content": "All parallel branches ended in reasoning loops",
                            "extra": {"exit_status": "reasoning_loop" if baseline else "LimitsExceeded",
                                      "submission": "",
                                      "limit_reason": "all_parallel_branches_reasoning_loop"},
                        })
                elif not errors:
                    winner = None
                    selection_error = ParallelSelectionError("All parallel API requests failed")
                else:
                    winner = min(errors, key=lambda c: c.branch_id)
                    fallback = "all_branches_format_error"
            else:
                winner = None
                selection_error = ParallelSelectionError("No eligible scored parallel candidate")
            evidence = {"schema_version": 1, "session_id": capture["session_id"],
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
            evidence = self.controller.record_decision(evidence, self.cfg["capture"]["control_seconds"])
            if selection_error is not None:
                if baseline:
                    for message in getattr(selection_error, "messages", []):
                        message.setdefault("extra", {}).update(
                            decision_id=decision, cost=evidence["total_cost"])
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
            message["extra"].update(dycon_parallel=summary, cost=evidence["total_cost"])
            return message
        finally:
            try:
                self.client.call("release", {"decision_id": decision})
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Parallel reservation release failed; bounded TTL applies")
