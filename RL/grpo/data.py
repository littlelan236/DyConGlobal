"""Torch-free preflight: dataset-wide eligibility, frozen advantages/order."""
from collections import Counter
from pathlib import Path
import math
import random

from core.io import hash_json, read_json, sha256
from RL.dataset.schema import MASK_POLICY
from .groups import GroupBatch
from .advantage import ADVANTAGE_VERSION

SYNTHETIC_DATASET = "synthetic-grpo-smoke-v1"


def prepare(config):
    binding = config.algorithm
    if binding.dataset_id == SYNTHETIC_DATASET:
        fixture = Path(__file__).parents[1] / "fixtures/grpo-native-smoke-v1.json"
        value = read_json(fixture)
        raw = value["groups"]
        proof = {"dataset_id": binding.dataset_id,
                 "dataset_receipt_sha256": sha256(fixture),
                 "synthetic": True, "tokenizer_fingerprint": "synthetic-token-ids-v1"}
    else:
        from RL.dataset.artifacts import directory, verify_local
        from RL.dataset.loader import iter_groups
        dataset = directory(binding.dataset_root, "rl-dataset", binding.dataset_id)
        verify_local(dataset, "rl-dataset")
        meta = read_json(dataset / "dataset.json")
        if not meta.get("data_ready") or meta.get("dataset_id") != binding.dataset_id:
            raise ValueError("Dataset binding is not data_ready or ID differs")
        digest = sha256(dataset / "receipt.json")
        raw = list(iter_groups(dataset, data_ready_only=False, with_tokens=False))
        from RL.dataset.loader import native_tokens
        for g in raw:
            if g["group"]["data_ready"]:
                for row in g["members"]:
                    p, c = native_tokens(row, audit=True)
                    row.update(prompt_token_ids=p, completion_token_ids=c)
        cohort = meta.get("cohort") or {}
        proof = {"dataset_id": binding.dataset_id,
                 "dataset_receipt_sha256": digest,
                 "synthetic": False, "tokenizer_fingerprint": cohort.get("tokenizer_fingerprint"),
                 "dataset_model": cohort.get("model")}
        for group in raw:
            for row in group["members"]:
                if group["group"]["data_ready"] and row.get("tokenizer_fingerprint") != proof["tokenizer_fingerprint"]:
                    raise ValueError("Dataset group tokenizer identity mismatch")
    settings = config.grpo.model_dump(include={"top_k", "top_k_fraction", "score_scale_floor", "boundary_tie_sigma"})
    eligible = []; exclusions = []
    for value in raw:
        group = value["group"]
        reason = None
        if not group.get("data_ready"):
            reason = "incomplete_native_group"
        elif group["expected_branch_count"] != len(value["members"]):
            raise ValueError("Group inventory is incomplete")
        elif any(len(r.get("prompt_token_ids", [])) + len(r.get("completion_token_ids", []))
                 > config.validation.max_sequence_length for r in value["members"]):
            raise ValueError("Group exceeds explicit sequence limit; no truncation is allowed")
        else:
            try:
                prepared = GroupBatch.from_dataset_group(value, as_tensors=False, **settings)
            except (ValueError, KeyError) as exc:
                # Identity/checksum/prompt corruption is a hard failure.
                if "prompt" in str(exc).lower() or "checksum" in str(exc).lower():
                    raise
                reason = str(exc)
        if reason:
            exclusions.append({"group_id": group["group_id"], "reason": reason})
        else:
            value["frozen_advantages"] = list(prepared.advantages)
            value["advantage_diagnostics"] = prepared.metadata
            eligible.append(value)
    if not eligible:
        raise ValueError("No eligible GRPO groups after exclusions")
    random.Random(config.grpo.shuffle_seed).shuffle(eligible)
    records = [{"group_id": g["group"]["group_id"], "branches": [
        {k: r[k] for k in ("branch_sample_id", "prompt_token_ids", "completion_token_ids", "historical_score")}
        for r in g["members"]], "advantages": g["frozen_advantages"]} for g in eligible]
    if len({r["group_id"] for r in records}) != len(records):
        raise ValueError("Duplicate group identity in dataset")
    advantages = [a for g in eligible for a in g["frozen_advantages"]]
    proof.update(objective_version="offline_grpo_k3_v1", mask_policy=MASK_POLICY,
                 advantage_policy=ADVANTAGE_VERSION, advantage_config_sha256=hash_json(settings),
                 group_order_seed=config.grpo.shuffle_seed, group_batch_size=config.grpo.group_batch_size,
                 max_epochs=1, probability_mode="raw_t1", clip_epsilon=config.grpo.clip_epsilon,
                 kl_beta=config.grpo.kl_beta,
                 eligible_group_ids=[g["group"]["group_id"] for g in eligible],
                 eligible_groups_sha256=hash_json(sorted(g["group"]["group_id"] for g in eligible)),
                 group_order_sha256=hash_json(records),
                 group_count=len(eligible), branch_count=len(advantages),
                 planned_updates=math.ceil(len(eligible) / config.grpo.group_batch_size),
                 exclusions=exclusions, exclusion_reasons=dict(Counter(x["reason"] for x in exclusions)),
                 total_groups=len(raw), group_coverage=len(eligible) / len(raw),
                 advantage_distribution={"min": min(advantages), "max": max(advantages),
                     "mean": sum(advantages) / len(advantages), "sum": math.fsum(advantages),
                     "positive": sum(a > 0 for a in advantages), "negative": sum(a < 0 for a in advantages)})
    if config.training.max_updates != proof["planned_updates"]:
        raise ValueError(f"GRPO max_updates must equal one epoch's planned_updates={proof['planned_updates']}")
    return eligible, proof


def tensor_groups(raw):
    """Reuse frozen advantages; never rerank within an optimizer update."""
    from RL.dataset.loader import collate_group
    for value in raw:
        rows = value["members"]
        batch = collate_group(value)
        yield GroupBatch(value["group"]["group_id"], [r["branch_sample_id"] for r in rows],
            [r["prompt_token_ids"] for r in rows], [r["completion_token_ids"] for r in rows],
            batch["input_ids"], batch["attention_mask"], batch["labels"], batch["loss_mask"],
            [r["historical_score"] for r in rows], tuple(value["frozen_advantages"]),
            metadata=value["advantage_diagnostics"])
