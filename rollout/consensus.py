"""Pure selection helpers for estimator-filtered exact-action consensus."""

from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace

from core.action_identity import ActionIdentityUnavailable, canonical_action
from core.io import hash_json
from rollout.entropy import entropy_retention

CONSENSUS_MODES = {"parallel_consensus", "parallel_enthophy"}


def consensus_options(parallel):
    """Only append new fields when the run explicitly freezes the new policies."""
    source = parallel.get("consensus") or {}
    options = {"retention_ratio": source.get("retention_ratio", 0.5),
               "min_retained_branches": source.get("min_retained_branches", 2),
               "rule": source.get("rule", "exact_action"),
               "tie_policy": source.get("tie_policy", "estimator_score")}
    if "representative_policy" in source:
        options["representative_policy"] = source["representative_policy"]
    if parallel.get("mode") == "parallel_enthophy":
        options["retention_rule"] = parallel["entropy"]["retention_rule"]
        if options["retention_rule"] == "effective_count_full_above_v1":
            options["full_retention_threshold"] = parallel["entropy"].get("full_retention_threshold", 10)
    return options


def select_consensus(
    candidates,
    scores,
    retention_ratio=0.5,
    min_retained_branches=2,
    tie_policy="estimator_score",
    *,
    representative_policy=None,
    retention_rule=None,
    full_retention_threshold=10,
    token_lengths=None,
):
    """Select one exact-action representative and return auditable metadata."""
    if not 0 < float(retention_ratio) <= 1:
        raise ValueError("retention_ratio must be in (0, 1]")
    if int(min_retained_branches) < 2:
        raise ValueError("min_retained_branches must be at least 2")
    if tie_policy != "estimator_score":
        raise ValueError("Unsupported consensus tie policy")
    if representative_policy not in {None, "estimator_score", "min_step_tokens"}:
        raise ValueError("Unsupported representative_policy")

    eligible = []
    branch_meta = {}
    for candidate in candidates:
        score = scores.get(candidate.branch_id)
        if candidate.message is None:
            branch_meta[candidate.branch_id] = {
                "branch_id": candidate.branch_id,
                "retained": False,
                "score": score,
                "identity_status": "not_action",
                "exclusion_reason": "not_formatted",
            }
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
        if score is None or not math.isfinite(score):
            branch_meta[candidate.branch_id] = {
                "branch_id": candidate.branch_id,
                "retained": False,
                "score": None,
                "identity_status": "unavailable",
                "exclusion_reason": "invalid_score",
            }
            continue
        eligible.append((candidate, score))

    identities = {}
    entropy_meta = None
    if retention_rule is not None:
        valid = []
        histogram = defaultdict(list)
        for candidate, score in eligible:
            try:
                message = candidate.response.model_dump(mode="json")["choices"][0]["message"]
                action_hash, action = canonical_action(message)
            except (ActionIdentityUnavailable, IndexError, KeyError, TypeError, ValueError) as exc:
                branch_meta[candidate.branch_id] = {
                    "branch_id": candidate.branch_id, "score": score, "retained": False,
                    "identity_status": "unavailable", "exclusion_reason": "action_identity_unavailable",
                    "identity_reason": str(exc)}
                continue
            identities[candidate.branch_id] = (action_hash, action)
            histogram[action_hash].append(candidate.branch_id)
            valid.append((candidate, score))
        eligible = valid
        entropy_meta = entropy_retention([len(ids) for ids in histogram.values()],
                                         int(min_retained_branches), retention_rule, float(retention_ratio),
                                         full_retention_threshold=full_retention_threshold)
        entropy_meta["candidate_branch_ids"] = sorted(identities)
        entropy_meta["groups"] = [
            {"action_hash": key, "action": identities[ids[0]][1], "branch_ids": sorted(ids), "count": len(ids)}
            for key, ids in sorted(histogram.items())]
    eligible.sort(key=lambda item: (item[1], item[0].branch_id))
    keep_count = (entropy_meta["keep_count"] if entropy_meta is not None else
                  max(int(min_retained_branches), math.ceil(float(retention_ratio) * len(eligible))))
    retained = eligible[: min(keep_count, len(eligible))]
    retained_ids = {candidate.branch_id for candidate, _ in retained}

    for rank, (candidate, score) in enumerate(eligible):
        branch_meta[candidate.branch_id] = {
            "branch_id": candidate.branch_id,
            "score": score,
            "estimator_rank": rank,
            "retained": candidate.branch_id in retained_ids,
            "identity_status": "pending" if candidate.branch_id in retained_ids else "not_evaluated",
            "exclusion_reason": None if candidate.branch_id in retained_ids else "estimator_filtered",
        }
        if candidate.branch_id in identities:
            action_hash, action = identities[candidate.branch_id]
            branch_meta[candidate.branch_id].update(identity_status="available", action_hash=action_hash, action=action)

    groups = defaultdict(list)
    unavailable = []
    for candidate, score in retained:
        try:
            dumped = candidate.response.model_dump(mode="json")
            message = dumped["choices"][0]["message"]
            action_hash, action = canonical_action(message)
        except (ActionIdentityUnavailable, IndexError, KeyError, TypeError, ValueError) as exc:
            reason = str(exc)
            unavailable.append({"branch_id": candidate.branch_id, "reason": reason})
            branch_meta[candidate.branch_id].update(
                identity_status="unavailable",
                exclusion_reason="action_identity_unavailable",
            )
            continue
        branch_meta[candidate.branch_id].update(
            identity_status="available",
            action_hash=action_hash,
            action=action,
        )
        groups[action_hash].append((candidate, score, action))

    base_meta = {
        "retained_branch_ids": [candidate.branch_id for candidate, _ in retained],
        "retained_count": len(retained),
        "branch_metadata": sorted(branch_meta.values(), key=lambda item: item["branch_id"]),
        "unavailable": unavailable,
        "votes": [],
        "tie_break": None,
    }
    if entropy_meta is not None:
        entropy_meta["cutoff_score"] = retained[-1][1] if retained else None
        entropy_meta["cutoff_branch_id"] = retained[-1][0].branch_id if retained else None
        base_meta["entropy"] = entropy_meta
    if len(eligible) < min_retained_branches or sum(len(rows) for rows in groups.values()) < min_retained_branches:
        return None, {**base_meta, "status": "insufficient_branches" if eligible else "no_eligible_branch"}
    if not groups:
        return None, {
            **base_meta,
            "status": "no_eligible_branch" if not eligible else "unavailable",
        }

    votes = []
    for action_hash, rows in groups.items():
        row_scores = [score for _, score, _ in rows]
        votes.append({
            "action_hash": action_hash,
            "action": rows[0][2],
            "count": len(rows),
            "branch_ids": [candidate.branch_id for candidate, _, _ in rows],
            "best_score": min(row_scores),
            "mean_score": sum(row_scores) / len(row_scores),
        })

    max_count = max(vote["count"] for vote in votes)
    tied = [vote for vote in votes if vote["count"] == max_count]
    winner_vote = min(
        tied,
        key=lambda vote: (
            vote["best_score"],
            vote["mean_score"],
            min(vote["branch_ids"]),
        ),
    )
    winner_candidates = [candidate for candidate, _, _ in groups[winner_vote["action_hash"]]]
    lengths = token_lengths or {}
    length_missing = (sorted(candidate.branch_id for candidate in winner_candidates
                             if type(lengths.get(candidate.branch_id)) is not int or lengths[candidate.branch_id] < 0)
                      if representative_policy == "min_step_tokens" else [])
    def comparison(candidate):
        key = (float(scores[candidate.branch_id]), candidate.branch_id)
        return (lengths[candidate.branch_id], *key) if representative_policy == "min_step_tokens" else key
    winner = min(winner_candidates, key=comparison) if not length_missing else None
    if representative_policy is not None:
        base_meta["representative"] = {
            "policy": representative_policy, "missing_length_branch_ids": length_missing,
            "comparison_keys": [{"branch_id": c.branch_id, "score": float(scores[c.branch_id]),
                                  **({"completion_tokens": lengths.get(c.branch_id)}
                                     if representative_policy == "min_step_tokens" else {})}
                                 for c in sorted(winner_candidates, key=lambda c: c.branch_id)],
            "selected_branch_id": winner.branch_id if winner else None}
    tie_break = None
    status = "reached"
    if len(tied) > 1:
        status = "tie_resolved_by_estimator"
        tie_break = {
            "policy": tie_policy,
            "candidate_action_hashes": [vote["action_hash"] for vote in tied],
            "comparison_keys": [
                {
                    "action_hash": vote["action_hash"],
                    "best_score": vote["best_score"],
                    "mean_score": vote["mean_score"],
                    "min_branch_id": min(vote["branch_ids"]),
                }
                for vote in tied
            ],
            "selected_action_hash": winner_vote["action_hash"],
        }

    valid_count = sum(vote["count"] for vote in votes)
    return winner, {
        **base_meta,
        "status": "token_length_unavailable" if length_missing else status,
        "votes": sorted(votes, key=lambda vote: (-vote["count"], vote["best_score"], vote["action_hash"])),
        "consensus_action_hash": winner_vote["action_hash"],
        "consensus_action": winner_vote["action"],
        "consensus_vote_fraction": winner_vote["count"] / valid_count,
        "valid_action_count": valid_count,
        "tie_break": tie_break,
    }


def validate_consensus_evidence(decision, options=None):
    """Audit stored selection with the same selector; never execute tools or replace the winner."""
    import json

    from rollout.loop_escape import validate_loop_escape_evidence

    validate_loop_escape_evidence(decision)
    if decision.get("selection_mode") not in CONSENSUS_MODES:
        return
    config = decision.get("consensus_config")
    if not isinstance(config, dict):
        raise ValueError("Consensus config is missing")
    if options is not None and config != options:
        raise ValueError("Consensus config differs from frozen run")
    if (decision["selection_mode"] == "parallel_enthophy") != ("retention_rule" in config):
        raise ValueError("Entropy mode and retention rule disagree")
    candidates, scores = [], {}
    for row in decision["candidates"]:
        if row["status"] != "eligible":
            continue
        score = row.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Consensus eligible score is invalid")
        if row.get("response") is not None:
            response = row["response"]
        elif row.get("action") is not None:
            actions = json.loads(row["action"])
            response = {"choices": [{"message": {"tool_calls": [
                {"function": {"name": action["name"], "arguments": action["arguments"]}}
                for action in actions
            ]}}]}
        else:
            # Filtered branches carry no identity and are never considered by the action parser.
            response = {"choices": [{"message": {}}]}
        candidates.append(SimpleNamespace(branch_id=row["branch_id"], message={},
                          response=SimpleNamespace(model_dump=lambda mode="json", value=response: value)))
        scores[row["branch_id"]] = score
    lengths = {row["branch_id"]: row.get("completion_tokens") for row in decision["candidates"]}
    winner, expected = select_consensus(
        candidates, scores, config["retention_ratio"], config["min_retained_branches"], config["tie_policy"],
        representative_policy=config.get("representative_policy"),
        retention_rule=config.get("retention_rule"), token_lengths=lengths,
        full_retention_threshold=config.get("full_retention_threshold", 10))
    if winner is None and not candidates:
        # Non-action format/limit responses retain the existing retry contract.
        if decision.get("consensus", {}).get("status") != "no_eligible_branch":
            raise ValueError("Consensus empty-candidate status differs")
        if decision["selection_mode"] == "parallel_enthophy":
            if hash_json(decision.get("consensus")) != hash_json(expected):
                raise ValueError("Consensus empty-pool entropy evidence differs")
            selected = decision.get("selected_branch_id")
            if selected is not None:
                corrective = [r["branch_id"] for r in decision["candidates"] if r.get("reason") == "FormatError"]
                if (decision.get("fallback_reason") != "all_branches_format_error" or
                        not corrective or selected != min(corrective)):
                    raise ValueError("Consensus empty pool cannot select an action winner")
        return
    actual = decision.get("consensus")
    if not isinstance(actual, dict) or hash_json(actual) != hash_json(expected):
        raise ValueError("Consensus vote evidence differs from candidates")
    if decision.get("selected_branch_id") != (winner.branch_id if winner else None):
        raise ValueError("Consensus selected branch differs from vote")
    by_branch = {row["branch_id"]: row for row in decision["candidates"]}
    for metadata in expected["branch_metadata"]:
        row = by_branch[metadata["branch_id"]]
        for key in ("retained", "estimator_rank", "exclusion_reason", "identity_status",
                    "action_hash", "action"):
            if row.get(key) != metadata.get(key):
                raise ValueError("Consensus branch metadata differs: " + key)
