"""Torch-free preflight: dataset-wide eligibility, frozen advantages/order."""
from collections import Counter
from pathlib import Path
import math

from core.io import hash_json, read_json, sha256
from RL.dataset.schema import MASK_POLICY
from .groups import GroupBatch
from .advantage import CONSENSUS_ADVANTAGE_POLICIES, SAME_ACTION_ADVANTAGE_VERSION, advantage_settings, validate_retained
from .grouping import exclusion_record, order_record, training_groups
from .schedule import LEGACY_SHUFFLE, order_groups, shuffle_metadata

SYNTHETIC_DATASET = "synthetic-grpo-smoke-v1"


def source_groups(config):
    """Open sealed source metadata; native payloads are read one group at a time."""
    binding = config.algorithm
    if binding.dataset_id == SYNTHETIC_DATASET:
        if config.grpo.advantage_policy == SAME_ACTION_ADVANTAGE_VERSION:
            raise ValueError("Same-action GRPO requires sealed native action identities")
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
        verify_local(dataset, "rl-dataset", verify_files=config.preflight.verify_identity)
        meta = read_json(dataset / "dataset.json")
        if not meta.get("data_ready") or meta.get("dataset_id") != binding.dataset_id:
            raise ValueError("Dataset binding is not data_ready or ID differs")
        digest = sha256(dataset / "receipt.json")
        raw = iter_groups(dataset, data_ready_only=False, with_tokens=False,
                          verify_identity=config.preflight.verify_identity)
        cohort = meta.get("cohort") or {}
        proof = {"dataset_id": binding.dataset_id,
                 "dataset_receipt_sha256": digest,
                 "synthetic": False, "tokenizer_fingerprint": cohort.get("tokenizer_fingerprint"),
                 "dataset_model": cohort.get("model")}
    if binding.group_ids:
        selected = set(binding.group_ids)
        original = iter(raw)
        def selected_groups():
            seen = set()
            for row in original:
                gid = row["group"]["group_id"]
                if gid in selected:
                    if gid in seen:
                        raise ValueError("Duplicate selected parent in source dataset")
                    seen.add(gid)
                    yield row
            if seen != selected:
                raise ValueError("Selected full-policy parent IDs missing from dataset: " + repr(sorted(selected - seen)[:8]))
        raw = selected_groups()
        proof["explicit_parent_selection"] = {"group_ids": sorted(selected), "sha256": hash_json(sorted(selected))}
    return iter(raw), proof


def freeze_group(value, config):
    """The single eligibility/advantage implementation, shared by both formats."""
    if config.grpo.advantage_policy == SAME_ACTION_ADVANTAGE_VERSION and len(value["members"]) < 2:
        return "singleton_action_group"
    if value["group"].get("data_ready"):
        from RL.dataset.loader import native_tokens
        for row in value["members"]:
            if "prompt_token_ids" not in row:
                p, c = native_tokens(row, audit=config.preflight.audit_native_tokens)
                row.update(prompt_token_ids=p, completion_token_ids=c,
                           audit_enabled=config.preflight.audit_native_tokens)
    group = value["group"]
    if not group.get("data_ready"):
        return "incomplete_native_group"
    if group["expected_branch_count"] != len(value["members"]):
        raise ValueError("Group inventory is incomplete")
    if any(len(r.get("prompt_token_ids", [])) + len(r.get("completion_token_ids", []))
           > config.validation.max_sequence_length for r in value["members"]):
        raise ValueError("Group exceeds explicit sequence limit; no truncation is allowed")
    if config.preflight.verify_identity:
        for row in value["members"]:
            if (not config.algorithm.dataset_id == SYNTHETIC_DATASET
                    and not row.get("tokenizer_fingerprint")):
                raise ValueError("Dataset group tokenizer identity mismatch")
    settings = advantage_settings(config.grpo)
    if config.grpo.advantage_policy in CONSENSUS_ADVANTAGE_POLICIES:
        validate_retained([r.get("retained") for r in value["members"]], len(value["members"]))
    try:
        prepared = GroupBatch.from_dataset_group(value, as_tensors=False, collate=False, **settings)
    except (ValueError, KeyError) as exc:
        if "prompt" in str(exc).lower() or "checksum" in str(exc).lower():
            raise
        return str(exc)
    value["frozen_advantages"] = list(prepared.advantages)
    value["advantage_diagnostics"] = prepared.metadata
    return None


def prepare(config, *, validate_updates=True, shuffle_algorithm=None):
    raw, proof = source_groups(config)
    settings = advantage_settings(config.grpo)
    proof["preflight"] = config.preflight.model_dump()
    eligible = []; exclusions = []
    total_groups = 0
    for value, reason in training_groups(raw, config.grpo.advantage_policy, proof=proof,
                                         max_sequence_length=config.validation.max_sequence_length):
        total_groups += 1
        reason = reason or freeze_group(value, config)
        if reason:
            exclusions.append(exclusion_record(value, reason))
        else:
            eligible.append(value)
    if not eligible:
        raise ValueError("No eligible GRPO groups after exclusions")
    shuffle_algorithm = shuffle_algorithm or config.grpo.shuffle_algorithm
    eligible = order_groups(eligible, seed=config.grpo.shuffle_seed,
        batch_size=config.grpo.group_batch_size, algorithm=shuffle_algorithm,
        prompt_length=lambda g: len(g["members"][0]["prompt_token_ids"]))
    if shuffle_algorithm != LEGACY_SHUFFLE:
        proof.update(shuffle_metadata(shuffle_algorithm))
    records = [order_record(group, config.grpo.advantage_policy) for group in eligible]
    if len({r["group_id"] for r in records}) != len(records):
        raise ValueError("Duplicate group identity in dataset")
    advantages = [a for g in eligible for a in g["frozen_advantages"]]
    proof.update(objective_version="offline_grpo_k3_v1", mask_policy=MASK_POLICY,
                 advantage_policy=config.grpo.advantage_policy, advantage_config_sha256=hash_json(settings),
                 group_order_seed=config.grpo.shuffle_seed, group_batch_size=config.grpo.group_batch_size,
                 max_epochs=1, probability_mode="raw_t1", clip_epsilon=config.grpo.clip_epsilon,
                 kl_beta=config.grpo.kl_beta,
                 eligible_group_ids=[g["group"]["group_id"] for g in eligible],
                 eligible_groups_sha256=hash_json(sorted(g["group"]["group_id"] for g in eligible)),
                 group_order_sha256=hash_json(records),
                 group_count=len(eligible), branch_count=len(advantages),
                 planned_updates=math.ceil(len(eligible) / config.grpo.group_batch_size),
                 exclusions=exclusions, exclusion_reasons=dict(Counter(x["reason"] for x in exclusions)),
                 total_groups=total_groups, group_coverage=len(eligible) / total_groups,
                 advantage_distribution={"min": min(advantages), "max": max(advantages),
                     "mean": sum(advantages) / len(advantages), "sum": math.fsum(advantages),
                     "positive": sum(a > 0 for a in advantages), "negative": sum(a < 0 for a in advantages)})
    from .execution import enabled, parent_record, from_config, apply_proof
    if enabled(config):
        apply_proof(proof, from_config([parent_record(g) for g in eligible], config))
    if validate_updates and config.training.max_updates != proof["planned_updates"]:
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
            metadata={**value["advantage_diagnostics"], **(
                {"execution_chunk": value["execution_chunk"]} if "execution_chunk" in value else {})})
