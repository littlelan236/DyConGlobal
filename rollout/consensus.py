"""Pure selection helpers for estimator-filtered exact-action consensus."""

from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace

from core.action_identity import ActionIdentityUnavailable, canonical_action
from core.io import hash_json


def select_consensus(
    candidates,
    scores,
    retention_ratio=0.5,
    min_retained_branches=2,
    tie_policy="estimator_score",
):
    """Select one exact-action representative and return auditable metadata."""
    if not 0 < float(retention_ratio) <= 1:
        raise ValueError("retention_ratio must be in (0, 1]")
    if int(min_retained_branches) < 2:
        raise ValueError("min_retained_branches must be at least 2")
    if tie_policy != "estimator_score":
        raise ValueError("Unsupported consensus tie policy")

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

    eligible.sort(key=lambda item: (item[1], item[0].branch_id))
    keep_count = max(int(min_retained_branches), math.ceil(float(retention_ratio) * len(eligible)))
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
    winner = min(
        (candidate for candidate, _, _ in groups[winner_vote["action_hash"]]),
        key=lambda candidate: (float(scores[candidate.branch_id]), candidate.branch_id),
    )
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
        "status": status,
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

    if decision.get("selection_mode") != "parallel_consensus":
        return
    config = decision.get("consensus_config")
    if not isinstance(config, dict):
        raise ValueError("Consensus config is missing")
    if options is not None and config != options:
        raise ValueError("Consensus config differs from frozen run")
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
    winner, expected = select_consensus(candidates, scores, config["retention_ratio"],
                                        config["min_retained_branches"], config["tie_policy"])
    if winner is None and not candidates:
        # Non-action format/limit responses retain the existing retry contract.
        if decision.get("consensus", {}).get("status") != "no_eligible_branch":
            raise ValueError("Consensus empty-candidate status differs")
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
