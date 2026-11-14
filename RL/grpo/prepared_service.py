"""Public prepared-data operations used by the existing LoRA CLI/service."""
from pathlib import Path
import time

from core.io import hash_json, read_json, sha256, write_json
from RL.artifacts import source_identity
from RL.config import load, schedule_contract
from RL.dataset.artifacts import publication, seal, verify_local
from .prepared import (PreparedReader, binding, resolve, materialize, summary,
                       validate_config, validate_dataset_binding)


def checkpoint_reader(checkpoint, *, project_root=None):
    from RL.checkpoints import verify_checkpoint
    verify_checkpoint(checkpoint)
    metadata = read_json(Path(checkpoint)/"state.json")
    contract = metadata["contract"]
    if "prepared" not in contract:
        raise ValueError("Legacy checkpoint has no prepared contract; materialize its frozen groups explicitly and use warm_start for a new stage")
    reader = PreparedReader(resolve(contract["prepared"]))
    if reader.proof != contract["input"] or reader.schedule["sha256"] != contract["prepared_schedule_sha256"]:
        raise ValueError("Checkpoint prepared/schedule contract identity mismatch")
    reader.validate_state(metadata["state"])
    return reader, metadata


def read_extension(path, *, summary_only=False):
    path = Path(path)
    folder = path.parent if path.name == "manifest.json" else path
    verify_local(folder, "rl-extension")
    request = read_json(folder/"manifest.json")
    if (request.get("schema_version") != 1 or request.get("extension_id") != folder.name
            or request.get("scheduler_extension") != "constant"):
        raise ValueError("Unsupported append extension schema/scheduler rule")
    cp = Path(request["parent_checkpoint"])
    if sha256(cp/"receipt.json") != request["parent_receipt_sha256"]:
        raise ValueError("Append parent checkpoint receipt changed")
    parent, metadata = checkpoint_reader(cp)
    if hash_json(metadata["contract"]) != request["parent_contract_sha256"]:
        raise ValueError("Append parent checkpoint contract changed")
    child = PreparedReader(resolve(request["prepared"]))
    if (child.manifest["parent"] != binding(parent.path)
            or child.schedule["extension_request_id"] != request["extension_id"]):
        raise ValueError("Append extension parent/prepared/request binding mismatch")
    if summary_only:
        return {"extension_id":request["extension_id"], "parent_checkpoint":str(cp),
                "scheduler_extension":request["scheduler_extension"], **summary(child)}
    return request, parent, child, metadata


def create(training_config, root_config, artifact_id, *, shard_groups=16,
           checkpoint=None, extension_id=None, append_seed=None, source_attempt=None):
    root = Path(root_config["project_root"])
    config, _ = load(training_config, root)
    if config.algorithm is None:
        raise ValueError("Prepared data requires offline GRPO")
    if source_attempt:
        from RL.artifacts import verify_receipt
        attempt = Path(source_attempt)
        verify_receipt(attempt, "lora-attempt-v1")
        class LegacyFrozen:
            proof = read_json(attempt/"input-identity.json")
            def __iter__(self):
                return iter(read_json(attempt/"frozen-groups.json"))
        legacy = LegacyFrozen()
        result = materialize(config, root, artifact_id, shard_groups=shard_groups,
            legacy=legacy, source_attempt=str(attempt))
        if result.proof["group_order_sha256"] != legacy.proof["group_order_sha256"]:
            raise ValueError("Legacy frozen-group order/content hash differs; migration rejected")
    elif checkpoint:
        if not extension_id or append_seed is None:
            raise ValueError("Append materialization requires extension_id and append_seed")
        parent, metadata = checkpoint_reader(checkpoint, project_root=root)
        validate_config(parent, config, exact=False)
        if config.training.scheduler != "constant":
            raise ValueError("Append scheduler extension requires constant; linear extension is not implemented")
        result = materialize(config, root, artifact_id, shard_groups=shard_groups,
            parent=parent, append_seed=append_seed, extension_id=extension_id)
        with publication(config.runtime.output_root, "rl-extension", extension_id,
                         signature=hash_json(binding(result.path))) as (stage, destination):
            write_json(stage/"manifest.json", {"schema_version":1, "extension_id":extension_id,
                "parent_checkpoint":str(Path(checkpoint).resolve()),
                "parent_receipt_sha256":sha256(Path(checkpoint)/"receipt.json"),
                "parent_contract_sha256":hash_json(metadata["contract"]), "prepared":binding(result.path),
                "producer_fingerprint":source_identity(root)["fingerprint"],
                "scheduler_extension":"constant", "old_total_updates":parent.proof["planned_updates"],
                "new_total_updates":result.proof["planned_updates"]})
            seal(stage, "rl-extension", extension_id)
        return {**summary(result), "prepared_data":binding(result.path),
                "extension_manifest":str(destination/"manifest.json")}
    else:
        result = materialize(config, root, artifact_id, shard_groups=shard_groups)
    return {**summary(result), "prepared_data":binding(result.path)}


def for_training(config, root, parent):
    """Resolve the immutable input before any tensor collation or reference import."""
    metadata = None
    extension = None
    mode = config.run.start.mode
    if mode == "append_resume":
        request, old, reader, metadata = read_extension(config.run.start.extension_manifest)
        if Path(request["parent_checkpoint"]).resolve() != Path(parent["path"]).resolve():
            raise ValueError("append_resume checkpoint differs from extension request")
        validate_config(old, config, exact=False)
        validate_config(reader, config)
        validate_dataset_binding(reader.proof, config)
        if config.training.scheduler != "constant":
            raise ValueError("Append scheduler requires the explicit constant extension rule")
        if (config.validation.stop_after_update is not None
                and config.validation.stop_after_update <= metadata["state"]["global_step"]):
            raise ValueError("Pause target must follow the restored global step")
        extension = {"extension_id":request["extension_id"],
            "extension_manifest_sha256":sha256(config.run.start.extension_manifest),
            "extension_receipt_sha256":sha256(Path(config.run.start.extension_manifest).parent/"receipt.json"),
            "extension_of_attempt":Path(parent["path"]).parent.parent.name,
            "parent_checkpoint":parent["path"], "scheduler_extension":"constant",
            "old_total_updates":old.proof["planned_updates"], "new_total_updates":reader.proof["planned_updates"],
            "new_groups":len(reader)-len(old),
            "new_branches":reader.proof["branch_count"]-old.proof["branch_count"],
            "new_tokens":sum(reader.records[g]["token_count"] for g in reader.local_ids)}
    elif mode == "resume":
        reader, metadata = checkpoint_reader(parent["path"], project_root=root)
        validate_config(reader, config)
        validate_dataset_binding(reader.proof, config)
        from .contract import validate_resume
        validate_resume(config, reader.proof, metadata)
        if "extension" in metadata["contract"]:
            extension = metadata["contract"]["extension"]
    elif config.prepared_data is not None:
        reader = PreparedReader(resolve(config.prepared_data.model_dump()))
        validate_config(reader, config)
        validate_dataset_binding(reader.proof, config)
        if reader.manifest.get("parent"):
            raise ValueError("Extended prepared schedule requires append_resume")
    else:
        artifact_id = config.run.attempt_id[:110]+"-prepared"
        if config.runtime.data_preparation_async:
            from RL.dataset.artifacts import directory as artifact_directory
            destination = artifact_directory(config.runtime.output_root, "rl-prepared", artifact_id)
            marker = Path(root) / "temp" / "async-preparation" / (config.run.attempt_id + ".json")
            deadline = time.monotonic() + config.runtime.distributed_timeout_seconds
            while not destination.exists():
                if marker.is_file():
                    state = read_json(marker)
                    if state.get("status") == "failed":
                        raise RuntimeError("Asynchronous prepared-data worker failed: " + state.get("error", "unknown error"))
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for asynchronous prepared-data artifact")
                time.sleep(2)
            reader = PreparedReader(destination)
        else:
            reader = materialize(config, root, artifact_id)
    if config.prepared_data is not None and binding(reader.path) != config.prepared_data.model_dump():
        raise ValueError("Configured prepared artifact differs from checkpoint/extension")
    reused = mode in {"resume", "append_resume"} or config.prepared_data is not None
    observations = {"prepared_cache_hit":reused,
        "source_scan_groups":0 if reused else reader.manifest["observations"]["source_scan_groups"],
        "native_scan_groups":0 if reused else reader.manifest["observations"]["native_scan_groups"],
        "materialized_groups":0 if reused else len(reader), "schedule_segments":len(reader.schedule["segments"]),
        **reader.statistics}
    return reader, metadata, extension, observations
