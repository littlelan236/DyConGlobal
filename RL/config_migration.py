"""Explicit, hash-bound configuration revisions for state-preserving GRPO resume.

Every configuration field can be proposed. Existing adapter, backend, frozen
input/anchor, execution-plan and tensor compatibility checks still apply.
This changes future execution settings; it is not an exact-trajectory claim.
"""
import copy
from pathlib import Path

from core.io import hash_json, read_json, sha256, write_json


def differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            child = f"{path}.{key}" if path else key
            if key not in before or key not in after:
                result.append({"path": child, "before_present": key in before,
                               "after_present": key in after,
                               "before": before.get(key), "after": after.get(key)})
            else:
                result.extend(differences(before[key], after[key], child))
        return result
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def schedule_matches(config, source):
    from RL.config import schedule_contract
    target = schedule_contract(config)
    migration = config.run.start.config_migration
    if migration is None:
        return source == target
    if (hash_json(source) != migration.source_schedule_sha256
            or hash_json(target) != migration.target_schedule_sha256):
        raise ValueError("Config migration source/target schedule hash mismatch")
    # Changes to frozen selection/advantages/windows need a separately prepared
    # input, not a misleading claim that old cursors describe a new schedule.
    for section in ("algorithm",):
        if source.get(section) != target.get(section):
            raise ValueError("Config migration needs compatible frozen input: " + section)
    frozen = ("group_batch_size", "max_epochs", "shuffle_seed", "advantage_policy",
              "top_k", "top_k_fraction", "score_scale_floor", "boundary_tie_sigma",
              "parallel_branch_schedule", "parallel_branch_chunk_size", "branch_partition_seed",
              "reference", "probability_mode", "reference_cache")
    for key in frozen:
        if source.get("grpo", {}).get(key) != target.get("grpo", {}).get(key):
            raise ValueError("Config migration needs a new frozen plan/anchor for grpo." + key)
    for key in ("max_updates", "gradient_accumulation_steps", "supervision"):
        if source["training"].get(key) != target["training"].get(key):
            raise ValueError("Config migration conflicts with frozen optimizer windows: " + key)
    return True


def contract_matches(config, source, target):
    from RL.artifacts import resume_contract_matches
    if not schedule_matches(config, source["schedule"]):
        return False
    adjusted = copy.deepcopy(target)
    adjusted["schedule"] = copy.deepcopy(source["schedule"])
    return resume_contract_matches(source, adjusted)


def restore_contract(config, directory, target):
    source = read_json(Path(directory) / "state.json")["contract"]
    if not contract_matches(config, source, target):
        raise ValueError("Migrated checkpoint immutable state/backend/anchor mismatch")
    return source


def write_transition(config, metadata, directory):
    if config.run.start.config_migration is None:
        return
    from RL.config import schedule_contract
    source, target = metadata["contract"]["schedule"], schedule_contract(config)
    if not schedule_matches(config, source):
        raise ValueError("Config migration schedule mismatch")
    path = Path(config.run.start.checkpoint)
    previous = path / "config-transition.json"
    source_config = path.parent.parent / "resolved-config.json"
    write_json(Path(directory) / "config-transition.json", {
        "schema_version": 1, "policy": "explicit_config_revision_preserve_state_v1",
        "reason": config.run.start.config_migration.reason,
        "source_checkpoint": str(path), "source_receipt_sha256": sha256(path / "receipt.json"),
        "source_state": metadata["state"],
        "source_schedule_sha256": hash_json(source), "target_schedule_sha256": hash_json(target),
        "schedule_changes": differences(source, target),
        "configuration_changes": differences(read_json(source_config), config.model_dump()) if source_config.is_file() else None,
        "parent_transition_sha256": sha256(previous) if previous.is_file() else None,
        "guarantee": "restored_parameters_optimizer_moments_scheduler_step_rng_and_cursor; revised_future_settings_not_exact_trajectory"})


def apply_optimizer_config(config, optimizer, scheduler):
    if config.run.start.config_migration is None:
        return
    # Restore moments/step first; explicitly apply new hyperparameters afterward.
    # LambdaLR functions are constructed from the new config; keep last_epoch.
    settings = config.training
    scheduler.base_lrs = [settings.learning_rate] * len(optimizer.param_groups)
    for index, group in enumerate(optimizer.param_groups):
        group["initial_lr"] = settings.learning_rate
        group["lr"] = settings.learning_rate * scheduler.lr_lambdas[index](scheduler.last_epoch)
        group["weight_decay"] = settings.weight_decay
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
