"""Explicit, hash-bound standard-AdamW -> streamed-AdamW resume transition."""
import copy

from core.io import hash_json
from RL.config import schedule_contract


def without_adam_strategy(schedule):
    result = copy.deepcopy(schedule)
    result["training"]["memory"].pop("adam_moment_offload", None)
    return result


def validate(config, source_contract):
    migration = config.run.start.optimizer_backend_migration
    if migration is None:
        return False
    source, target = source_contract["schedule"], schedule_contract(config)
    if migration.source_schedule_sha256 != hash_json(source) or migration.target_schedule_sha256 != hash_json(target):
        raise ValueError("Optimizer migration source/target schedule hash mismatch")
    if source_contract.get("optimizer_backend") is not None:
        raise ValueError("Optimizer migration currently supports standard AdamW sources only")
    if config.training.memory.adam_moment_offload.mode != "memory_pressure":
        raise ValueError("Optimizer migration requires streamed AdamW target")
    if without_adam_strategy(target) != source:
        raise ValueError("Optimizer migration may change only Adam residency strategy; schedule/math must match")
    return True


def schedule_matches(config, source_contract):
    if config.run.start.optimizer_backend_migration is not None:
        return validate(config, source_contract)
    return schedule_contract(config) == source_contract["schedule"]


def contract_matches(config, source, target):
    if not validate(config, source):
        return source == target
    expected = copy.deepcopy(source)
    expected["schedule"] = target["schedule"]
    expected["optimizer_backend"] = target["optimizer_backend"]
    expected["optimizer_migration"] = target["optimizer_migration"]
    return expected == target
