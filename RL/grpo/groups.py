"""Typed, immutable-ish group records and group-preserving batching."""

from dataclasses import dataclass, field
import math

from .advantage import compute_advantages


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
                           pad_token_id=0, as_tensors=True):
        members = value.get("members", [])
        if not members:
            raise ValueError("Cannot construct an empty GRPO group")
        inventory = value.get("group", {})
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
        if any(type(s) not in (int, float) or not math.isfinite(float(s)) for s in scores):
            raise ValueError("GRPO groups require finite historical scores")
        if len(members) < 2:
            raise ValueError("GRPO groups require at least two branches")
        if any(not row.get("action") for row in members):
            raise ValueError("missing_action")
        from RL.dataset.loader import collate_group
        tensors = collate_group({"members": members}, as_tensors=as_tensors, pad_token_id=pad_token_id)
        adv = compute_advantages(scores, top_k=top_k, top_k_fraction=top_k_fraction,
                                 score_scale_floor=score_scale_floor,
                                 boundary_tie_sigma=boundary_tie_sigma)
        if adv.metadata["status"] != "ok":
            raise ValueError(adv.metadata["skip_reason"])
        return cls(value.get("group", {}).get("group_id", members[0].get("group_id", "")),
                   [str(row.get("branch_sample_id", row.get("branch_id"))) for row in members],
                   prompts, completions, tensors["input_ids"], tensors["attention_mask"],
                   tensors["labels"], tensors["loss_mask"], [float(x) for x in scores],
                   adv.advantages, metadata={**adv.metadata, "order": adv.order, "ranks": adv.ranks})

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
