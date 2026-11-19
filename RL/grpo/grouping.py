"""Derive logical training groups without changing the sealed source coordinates."""

from collections import Counter
import re

from core.io import hash_json
from .advantage import ADVANTAGE_VERSION, CONSENSUS_ADVANTAGE_VERSION, SAME_ACTION_ADVANTAGE_VERSION

ACTION_HASH = re.compile(r"[0-9a-f]{64}")
LINEAGE_KEYS = ("source_group_id", "action_hash")


def lineage(group):
    return {key: group[key] for key in LINEAGE_KEYS if key in group}


def same_action_group_id(source_group_id, action_hash):
    return "g-" + hash_json([SAME_ACTION_ADVANTAGE_VERSION, source_group_id, action_hash])[:40]


def training_groups(raw, policy, *, proof=None, max_sequence_length=None):
    """Yield training groups or source-step exclusions in native source order."""
    if policy != SAME_ACTION_ADVANTAGE_VERSION:
        for value in raw:
            yield value, None
        return

    counts = Counter()
    sizes = Counter()
    seen_groups, seen_branches = set(), set()
    for value in raw:
        group, members = value["group"], value["members"]
        gid = group["group_id"]
        ids = [row["branch_sample_id"] for row in members]
        ordinals = [row["branch_id"] for row in members]
        if (not members or group["expected_branch_count"] != len(members)
                or group["branch_ids"] != ids or ordinals != list(range(len(members)))
                or any(row["group_id"] != gid for row in members)):
            raise ValueError("Incomplete or misordered source group inventory for same-action GRPO")
        if gid in seen_groups or len(set(ids)) != len(ids) or not seen_branches.isdisjoint(ids):
            raise ValueError("Duplicate source group or branch for same-action GRPO")
        seen_groups.add(gid)
        seen_branches.update(ids)
        counts["source_step_count"] += 1
        if group.get("terminal_scoring_exclusion") or any(r.get("terminal_scoring_exclusion") for r in members):
            reason = "terminal_evaluation_ineligible"
        elif not group.get("data_ready"):
            reason = "incomplete_native_group"
        elif any(not isinstance(row.get("action"), str) or not row["action"]
                 or not isinstance(row.get("action_hash"), str) or not ACTION_HASH.fullmatch(row["action_hash"])
                 for row in members):
            reason = "missing_action_identity"
        else:
            reason = None
        if reason:
            counts["excluded_source_step_count"] += 1
            yield value, reason
            continue

        prompt = (members[0]["prompt_sha256"], members[0]["prompt_count"])
        for row in members:
            if (row["prompt_sha256"], row["prompt_count"]) != prompt:
                raise ValueError("Same-action source step prompt identities differ")
            if (max_sequence_length is not None
                    and row["prompt_count"] + row["completion_count"] > max_sequence_length):
                raise ValueError("Group exceeds explicit sequence limit; no truncation is allowed")

        by_action = {}
        for row in members:
            by_action.setdefault(row["action_hash"], []).append(row)
        for action_hash, rows in by_action.items():
            counts["derived_group_count"] += 1
            sizes[str(len(rows))] += 1
            subgroup = {**group, "group_id": same_action_group_id(gid, action_hash),
                        "source_group_id": gid, "action_hash": action_hash,
                        "expected_branch_count": len(rows), "available_branch_count": len(rows),
                        "raw_probability_complete": all(r.get("raw_probability_available", False) for r in rows),
                        "branch_ids": [r["branch_sample_id"] for r in rows]}
            # Members keep their dataset group ID for the probability reader.
            yield {**value, "group": subgroup, "members": rows}, None
    if proof is not None:
        proof["same_action_grouping"] = {
            "version": SAME_ACTION_ADVANTAGE_VERSION,
            "source_step_count": counts["source_step_count"],
            "excluded_source_step_count": counts["excluded_source_step_count"],
            "derived_group_count": counts["derived_group_count"],
            "derived_group_sizes": dict(sizes),
        }


def exclusion_record(value, reason):
    group = value["group"]
    return {"group_id": group["group_id"], **lineage(group), "reason": reason}


def order_record(value, policy=ADVANTAGE_VERSION):
    """Bind group lineage, scores, tokens and frozen advantages in native order."""
    # Preserve positional boolean callers of prepared.order_record from older code.
    if type(policy) is bool:
        policy = CONSENSUS_ADVANTAGE_VERSION if policy else ADVANTAGE_VERSION
    keys = ("branch_sample_id", "prompt_token_ids", "completion_token_ids", "historical_score")
    if policy == CONSENSUS_ADVANTAGE_VERSION:
        keys += ("retained",)
    group = value["group"]
    record = {"group_id": group["group_id"],
              "branches": [{k: row[k] for k in keys} for row in value["members"]],
              "advantages": value["frozen_advantages"]}
    if policy == SAME_ACTION_ADVANTAGE_VERSION:
        record.update({key: group[key] for key in LINEAGE_KEYS})
    return record
