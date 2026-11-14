"""Standard PEFT exports and atomically published complete update-boundary states."""

import os
import tempfile
import shutil
from pathlib import Path

from core.io import hash_json, read_json, sha256, write_json
from RL.artifacts import commit_receipt, inventory, verify_receipt, resume_contract_matches


def adapter_contract(base, targets, lora):
    return {"base_fingerprint": base["fingerprint"], "targets": targets,
            "lora": lora.model_dump() if hasattr(lora, "model_dump") else lora}


def export_adapter(model, directory, contract):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Adapter output already exists: {directory}")
    directory.mkdir(parents=True)
    model.save_pretrained(directory, safe_serialization=True)
    files = inventory(directory)
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
        raise ValueError("PEFT export is incomplete")
    write_json(directory / "identity.json", {"schema_version": 1, "contract": contract, "files": files})
    verify_adapter(directory, contract, verify_owner=False)


def verify_adapter(directory, expected_contract=None, *, verify_owner=True):
    directory = Path(directory).resolve()
    if directory.name != "adapter":
        raise ValueError("Only this module's named adapter exports are accepted")
    owner = directory.parent
    if verify_owner:
        receipt = verify_receipt(owner)
        if receipt.get("kind") not in {"lora-attempt-v1", "lora-checkpoint-v1"}:
            raise ValueError("Adapter has no supported LoRA owner")
    identity = read_json(directory / "identity.json")
    if identity.get("schema_version") != 1:
        raise ValueError("Invalid adapter identity schema")
    files = inventory(directory)
    files.pop("identity.json", None)
    if files != identity.get("files"):
        raise ValueError("Adapter content/hash mismatch")
    contract = identity["contract"]
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("Adapter base/targets/LoRA identity mismatch")
    config = read_json(directory / "adapter_config.json")
    settings = contract["lora"]
    if (config.get("peft_type") != "LORA" or config.get("bias") != "none"
            or config.get("r") != settings["rank"] or config.get("lora_alpha") != settings["alpha"]
            or config.get("lora_dropout") != settings["dropout"]
            or sorted(config.get("target_modules", [])) != sorted(t["path"] for t in contract["targets"])
            or config.get("modules_to_save") or config.get("use_dora") or config.get("use_rslora")):
        raise ValueError("Saved PEFT adapter configuration differs from the frozen contract")
    return identity


def verify_checkpoint(directory):
    directory = Path(directory).resolve()
    if not (directory / "training-state.pt").is_file():
        raise ValueError("resume requires a complete training checkpoint; use warm_start for adapter-only exports")
    receipt = verify_receipt(directory, "lora-checkpoint-v1")
    required = {"training-state.pt", "state.json", "adapter/identity.json", "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors"}
    if not required.issubset(receipt["files"]):
        raise ValueError("Checkpoint required files missing")
    metadata = read_json(directory / "state.json")
    state = metadata["state"]
    if (receipt.get("contract_sha256") != hash_json(metadata["contract"])
            or metadata["contract"].get("adapter", metadata["adapter_contract"]) != metadata["adapter_contract"]):
        raise ValueError("Checkpoint receipt/adapter contract mismatch")
    if (not isinstance(state.get("global_step"), int) or state["global_step"] < 1
            or directory.name != f"step-{state['global_step']:06d}"
            or receipt.get("global_step") != state["global_step"]):
        raise ValueError("Checkpoint step identity mismatch")
    verify_adapter(directory / "adapter", metadata["adapter_contract"], verify_owner=False)
    if ("grpo" in metadata["contract"] and metadata["contract"].get("streaming") is None
            and metadata["contract"]["grpo"].get("cache_schema_version") != 3
            and "grpo/reference-logprobs.pt" not in receipt["files"]):
        raise ValueError("GRPO checkpoint reference cache missing")
    if "grpo" in metadata["contract"]:
        from RL.grpo.contract import validate_reference_files, receipt_metadata
        if metadata["contract"].get("streaming"):
            from RL.grpo.stream_resume import resolve_checkpoint
            resolve_checkpoint(directory, metadata["contract"])
        else:
            validate_reference_files(directory, metadata["contract"])
        expected = receipt_metadata(metadata["contract"], state)
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("GRPO checkpoint receipt objective/cursor mismatch")
        if "prepared" in metadata["contract"]:
            from RL.grpo.prepared import PreparedReader, resolve
            prepared = PreparedReader(resolve(metadata["contract"]["prepared"]))
            if (prepared.proof != metadata["contract"]["input"] or
                    prepared.schedule["sha256"] != metadata["contract"]["prepared_schedule_sha256"]):
                raise ValueError("Checkpoint prepared/schedule identity mismatch")
            prepared.validate_state(state)
    return receipt


def save_checkpoint(model, optimizer, scheduler, state, directory, contract, adapter_identity, *, extra_files=None):
    import torch

    from RL.training import capture_rng

    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError("Checkpoint output already exists")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Checkpoints may only be saved after gradients have been cleared")
    staging = Path(tempfile.mkdtemp(prefix=".write-checkpoint-", dir=directory.parent))
    export_adapter(model, staging / "adapter", adapter_identity)
    metadata = {"schema_version": 1, "state": dict(state), "contract": contract,
                "adapter_contract": adapter_identity}
    write_json(staging / "state.json", metadata)
    if extra_files:
        import shutil
        from core.io import owned_path
        for name, source_file in extra_files.items():
            target = owned_path(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
    payload = {"metadata": metadata, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "rng": capture_rng()}
    from RL.parallel import current
    parallel = current()
    if parallel:
        if not parallel.rank_rng or len(parallel.rank_rng) != parallel.world_size:
            raise ValueError("Distributed checkpoint missing rank RNG states")
        payload["rank_rng"] = parallel.rank_rng
    with (staging / "training-state.pt").open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    extra_metadata = {}
    if "grpo" in contract:
        from RL.grpo.contract import validate_reference_files, receipt_metadata
        if contract.get("streaming"):
            from RL.grpo.stream_resume import resolve_checkpoint
            resolve_checkpoint(staging, contract)
        else:
            validate_reference_files(staging, contract, verify_artifacts=False)
        extra_metadata = receipt_metadata(contract, state)
    commit_receipt(staging, "lora-checkpoint-v1", global_step=state["global_step"],
                   contract_sha256=hash_json(contract), **extra_metadata)
    staging.rename(directory)
    fd = os.open(directory.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return directory


def restore_checkpoint(directory, optimizer, scheduler, expected_contract, expected_adapter_contract, *, verify=True):
    import torch

    from RL.training import restore_rng

    directory = Path(directory)
    if verify:
        verify_checkpoint(directory)
    metadata = read_json(directory / "state.json")
    # Historical project code hashes are provenance, not a resume gate.
    if (not resume_contract_matches(metadata["contract"], expected_contract)
            or metadata["adapter_contract"] != expected_adapter_contract):
        raise ValueError("Resume schedule/input/base/targets/source contract mismatch")
    # Accept only verified local module output, and use the restricted Torch loader.
    payload = torch.load(directory / "training-state.pt", map_location="cpu", weights_only=True)
    if payload["metadata"] != metadata:
        raise ValueError("Training state and checkpoint metadata disagree")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    from RL.parallel import current
    parallel = current()
    rank_rng = payload.get("rank_rng")
    if parallel:
        if not rank_rng or len(rank_rng) != parallel.world_size:
            raise ValueError("Checkpoint rank RNG topology mismatch")
        restore_rng(rank_rng[parallel.rank])
    else:
        if rank_rng:
            raise ValueError("Distributed checkpoint requires its original world size")
        restore_rng(payload["rng"])
    return dict(metadata["state"])


def parent_reference(source, mode):
    source = Path(source).resolve()
    owner = source if mode == "resume" else source.parent
    verify_receipt(owner)
    return {"mode": mode, "path": str(source), "owner": str(owner),
            "receipt_sha256": sha256(owner / "receipt.json"),
            "adapter_identity_sha256": sha256((source / "adapter" if mode == "resume" else source) / "identity.json")}
