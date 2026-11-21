"""Typed, immutable-ish group records and group-preserving batching."""

from dataclasses import dataclass, field
import math

from .advantage import (ADVANTAGE_VERSION, CONSENSUS_ADVANTAGE_POLICIES, SAME_ACTION_ADVANTAGE_VERSION,
                        compute_advantages, validate_retained)
from .grouping import lineage, same_action_group_id


def advantages_for_members(members, **settings):
    if settings.get("advantage_policy") in CONSENSUS_ADVANTAGE_POLICIES:
        settings["retained"] = [row.get("retained") for row in members]
    return compute_advantages([row.get("historical_score") for row in members], **settings)


@dataclass(frozen=True)
class GroupBatch:
    group_id: str
    branch_ids: list[str]
    prompt_token_ids: list[list[int]]
    completion_token_ids: list[list[int]]
    input_ids: object
    attention_mask: object
    labels: object
    loss_mask: object
    historical_scores: list[float]
    advantages: list[float]
    objective_version: str = "offline_grpo_k3_v1"
    metadata: dict = field(default_factory=dict)

    @property
    def group_size(self):
        return len(self.branch_ids)

    @classmethod
    def from_dataset_group(cls, value, *, top_k=4, top_k_fraction=None,
                           score_scale_floor=0.1, boundary_tie_sigma=0.05,
                           advantage_policy=ADVANTAGE_VERSION, pad_token_id=0, as_tensors=True,
                           collate=True):
        members = value.get("members", [])
        if not members:
            raise ValueError("Cannot construct an empty GRPO group")
        inventory = value.get("group", {})
        same_action = advantage_policy == SAME_ACTION_ADVANTAGE_VERSION
        if same_action and (not inventory.get("source_group_id") or not inventory.get("action_hash")
                            or inventory.get("group_id") != same_action_group_id(
                                inventory["source_group_id"], inventory["action_hash"])
                            or any(row.get("group_id") != inventory["source_group_id"]
                                   or row.get("action_hash") != inventory["action_hash"] for row in members)):
            raise ValueError("Same-action advantages require a derived training group")
        if advantage_policy in CONSENSUS_ADVANTAGE_POLICIES:
            validate_retained([row.get("retained") for row in members], len(members))
        if inventory.get("terminal_scoring_exclusion") or any(r.get("terminal_scoring_exclusion") for r in members):
            raise ValueError("terminal_evaluation_ineligible")
        if inventory.get("expected_branch_count", len(members)) != len(members):
            raise ValueError("Incomplete GRPO group inventory")
        if "branch_ids" in inventory and inventory["branch_ids"] != [r["branch_sample_id"] for r in members]:
            raise ValueError("Misordered GRPO branch identity")
        prompts = [list(row["prompt_token_ids"]) for row in members]
        completions = [list(row["completion_token_ids"]) for row in members]
        if any(not p or not c for p, c in zip(prompts, completions)):
            raise ValueError("GRPO groups require non-empty prompt and completion tokens")
        if any(p != prompts[0] for p in prompts[1:]):
            raise ValueError("GRPO group prompt tokens are not identical")
        scores = [row.get("historical_score") for row in members]
        if not same_action and any(type(s) not in (int, float) or not math.isfinite(float(s)) for s in scores):
            raise ValueError("GRPO groups require finite historical scores")
        if len(members) < 2:
            raise ValueError("singleton_action_group" if same_action else "GRPO groups require at least two branches")
        if any(not row.get("action") for row in members):
            raise ValueError("missing_action")
        from RL.dataset.loader import collate_group
        tensors = (collate_group({"members": members}, as_tensors=as_tensors, pad_token_id=pad_token_id)
                   if collate else dict.fromkeys(("input_ids", "attention_mask", "labels", "loss_mask")))
        adv = advantages_for_members(members, top_k=top_k, top_k_fraction=top_k_fraction,
                                     score_scale_floor=score_scale_floor,
                                     boundary_tie_sigma=boundary_tie_sigma,
                                     advantage_policy=advantage_policy)
        if adv.metadata["status"] != "ok":
            raise ValueError(adv.metadata["skip_reason"])
        return cls(value.get("group", {}).get("group_id", members[0].get("group_id", "")),
                   [str(row.get("branch_sample_id", row.get("branch_id"))) for row in members],
                   prompts, completions, tensors["input_ids"], tensors["attention_mask"],
                   tensors["labels"], tensors["loss_mask"], [float(x) for x in scores],
                   adv.advantages, metadata={**adv.metadata, **lineage(inventory),
                                             "order": adv.order, "ranks": adv.ranks})

    def branch_batch(self, index, device):
        """Physical microbatch=1, with complete native prefix and no padding."""
        length = len(self.prompt_token_ids[index]) + len(self.completion_token_ids[index])
        return {name: getattr(self, name)[index:index + 1, :length].to(device)
                for name in ("input_ids", "attention_mask", "labels", "loss_mask")}


def iter_group_batches(groups, group_batch_size):
    """Yield frozen windows; never split a group across optimizer updates."""
    if group_batch_size < 1:
        raise ValueError("group_batch_size must be positive")
    window = []
    for group in groups:
        if not hasattr(group, "group_size"):
            raise TypeError("iter_group_batches expects group records")
        window.append(group)
        if len(window) == group_batch_size:
            yield tuple(window)
            window = []
    if window:
        yield tuple(window)
