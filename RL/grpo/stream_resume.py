"""Frozen streaming recovery evidence. Metadata resolution never imports Torch."""
import shutil
from pathlib import Path, PurePosixPath

from core.io import hash_json, owned_path, read_json, sha256, write_json

PLAN = "grpo/stream-plan.json"
PROVENANCE = "grpo/source-provenance.json"
ANCHOR = "grpo/anchor-tensors.json"
BUNDLE = "grpo/stream-resume.json"
SNAPSHOT = "grpo/reference/adapter"


def checkpoint_kind(contract):
    if "streaming" in contract:
        value = contract["streaming"]
        if (not isinstance(value, dict) or value.get("version") != 1
                or value.get("window_queue") != 2
                or value.get("assignment") != "window_ordinal_mod_world_size"
                or "prepared" in contract):
            raise ValueError("Unsupported streaming checkpoint version/execution contract")
        return "streaming"
    return "prepared" if "prepared" in contract else "legacy"


def execution_mode(config, parent_contract=None):
    kind = checkpoint_kind(parent_contract) if parent_contract is not None else None
    mode = config.run.start.mode
    if kind == "streaming":
        if mode != "resume":
            raise ValueError("Streaming checkpoints support exact resume only")
        if not config.runtime.data_preparation_async or config.prepared_data is not None:
            raise ValueError("Streaming resume requires data_preparation_async=true and prepared_data=null")
        return "streaming"
    if mode == "fresh" and config.runtime.data_preparation_async and config.prepared_data is None:
        return "streaming"
    return kind or "prepared"


def checked_path(root, name):
    relative = PurePosixPath(name)
    if (not isinstance(name, str) or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or ":" in name or not relative.parts):
        raise ValueError("Invalid streaming recovery relative path: " + str(name))
    root = Path(root)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink in streaming recovery evidence: " + name)
    return owned_path(root, name)


def recovery_files(directory):
    directory = Path(directory)
    names = [PLAN, PROVENANCE, ANCHOR]
    snapshot = checked_path(directory, SNAPSHOT)
    if not snapshot.is_dir():
        raise ValueError("Missing streaming reference adapter: " + str(snapshot))
    for path in sorted(snapshot.rglob("*")):
        name = path.relative_to(directory).as_posix()
        checked_path(directory, name)
        if path.is_file():
            names.append(name)
    return {name: checked_path(directory, name) for name in names}


def write_bundle(directory, contract, *, origin=None):
    files = recovery_files(directory)
    write_json(Path(directory) / BUNDLE, {
        "schema_version": 1,
        "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for name, path in files.items()},
        "reference_identity": contract["grpo"]["reference_identity"],
        "origin": origin,
    })
    return {**files, BUNDLE: Path(directory) / BUNDLE}


def resolve_checkpoint(directory, contract):
    """Resolve one sealed checkpoint, or its tightly bounded legacy evidence."""
    from .streaming import validate_reference
    if checkpoint_kind(contract) != "streaming":
        raise ValueError("Expected a streaming checkpoint")
    directory = Path(directory)
    bundle_path = checked_path(directory, BUNDLE)
    if bundle_path.exists():
        evidence = directory
        bundle = read_json(bundle_path)
        if (bundle.get("schema_version") != 1
                or bundle.get("reference_identity") != contract["grpo"]["reference_identity"]):
            raise ValueError("Streaming recovery bundle schema/reference identity mismatch")
        files = recovery_files(evidence)
        if set(bundle.get("files", {})) != set(files):
            raise ValueError("Streaming recovery bundle file inventory mismatch")
        for name, path in files.items():
            if bundle["files"][name] != {"sha256": sha256(path), "bytes": path.stat().st_size}:
                raise ValueError("Streaming recovery bundle checksum mismatch: " + name)
        origin = bundle.get("origin")
    else:
        # A partial new bundle must never be disguised as historical output.
        if any(checked_path(directory, name).exists() for name in (PLAN, PROVENANCE, ANCHOR, SNAPSHOT)):
            raise ValueError("Incomplete streaming recovery bundle; refusing legacy fallback")
        if (directory.parent.name != "checkpoints" or not directory.name.startswith("step-")
                or directory.parent.is_symlink() or directory.is_symlink()):
            raise ValueError("Legacy streaming checkpoint must be ATTEMPT/checkpoints/step-N")
        evidence = directory.parent.parent
        anchor = checked_path(evidence, "initial-adapter.json")
        if not anchor.is_file():
            raise ValueError("Missing legacy streaming initial-adapter.json")
        origin = {"legacy_parent_attempt": str(evidence),
                  "source_checkpoint_receipt_sha256": sha256(directory / "receipt.json")}
    for name in (PLAN, PROVENANCE, SNAPSHOT):
        checked_path(evidence, name)
    for path in checked_path(evidence, SNAPSHOT).rglob("*"):
        checked_path(evidence, path.relative_to(evidence).as_posix())
    plan = validate_reference(evidence, contract)
    state_path = directory / "state.json"
    if state_path.is_file():
        state = read_json(state_path)["state"]
        count, step = state.get("group_cursor"), state.get("global_step")
        batch = contract["input"]["group_batch_size"]
        if plan["proof"].get("execution_plan"):
            from .execution import validate_state
            validate_state(plan["proof"]["execution_plan"], state)
        elif (type(count) is not int or type(step) is not int or not 0 <= count <= len(plan["groups"])
                or count != min(len(plan["groups"]), step * batch)
                or state.get("consumed_groups") != count
                or state.get("consumed_branches") != sum(len(g["branch_ids"]) for g in plan["groups"][:count])
                or state.get("segment_id") != "streaming" or state.get("segment_ordinal") != count
                or state.get("consumed_group_ids_sha256") != hash_json(plan["proof"]["eligible_group_ids"][:count])
                or state.get("last_consumed_group_id") != (plan["groups"][count-1]["group_id"] if count else None)):
            raise ValueError("Streaming checkpoint committed cursor/prefix mismatch")
    anchor_path = checked_path(evidence, ANCHOR if bundle_path.exists() else "initial-adapter.json")
    fingerprints = read_json(anchor_path)
    if not fingerprints or hash_json(fingerprints) != contract["grpo"]["reference_identity"]["adapter_hash"]:
        raise ValueError("Streaming original anchor tensor manifest mismatch")
    files = recovery_files(evidence) if bundle_path.exists() else {
        PLAN: checked_path(evidence, PLAN), PROVENANCE: checked_path(evidence, PROVENANCE),
        ANCHOR: anchor_path,
        **{p.relative_to(evidence).as_posix(): checked_path(evidence, p.relative_to(evidence).as_posix())
           for p in checked_path(evidence, SNAPSHOT).rglob("*") if p.is_file()}}
    return {"plan": plan, "files": files, "evidence": evidence, "fingerprints": fingerprints,
            "origin": origin, "tensor_validation": "pending_training_model"}


def inherit(resolved, directory):
    for name, source in resolved["files"].items():
        target = checked_path(directory, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def validate_anchor(model, directory, contract):
    """Use PEFT's own name/dtype mapping, then restore the current adapter."""
    from peft.utils.save_and_load import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
    from RL.modeling import adapter_fingerprint
    saved = {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(model).items()}
    anchor = load_peft_weights(str(Path(directory) / SNAPSHOT), device="cpu")
    if set(anchor) != set(saved) or any(anchor[k].shape != saved[k].shape or anchor[k].dtype != saved[k].dtype for k in saved):
        raise ValueError("Streaming anchor tensor inventory/shape/dtype mismatch")
    try:
        set_peft_model_state_dict(model, anchor)
        actual = adapter_fingerprint(model)
        if (actual != read_json(Path(directory) / ANCHOR)
                or hash_json(actual) != contract["grpo"]["reference_identity"]["adapter_hash"]):
            raise ValueError("Streaming original anchor actual tensor fingerprint mismatch")
    finally:
        set_peft_model_state_dict(model, saved)


def validate_snapshot_tensors(resolved, contract):
    """Read the anchor alone in the training environment without a base-model load."""
    from safetensors.torch import load_file
    from RL.modeling import tensor_sha
    expected = resolved["fingerprints"]
    tensors = load_file(str(resolved["evidence"] / SNAPSHOT / "adapter_model.safetensors"), device="cpu")
    mapped = {}
    for name, tensor in tensors.items():
        candidates = [key for key in expected if key == name or
                      key.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.") == name]
        if len(candidates) != 1 or candidates[0] in mapped:
            raise ValueError("Ambiguous/missing streaming anchor PEFT tensor mapping: " + name)
        mapped[candidates[0]] = tensor_sha(tensor)
    if mapped != expected or hash_json(mapped) != contract["grpo"]["reference_identity"]["adapter_hash"]:
        raise ValueError("Streaming original anchor actual tensor fingerprint mismatch")
    return {"tensor_validation": "verified", "tensor_count": len(mapped), "adapter_hash": hash_json(mapped)}


def restore_training(directory, optimizer, scheduler, contract, *, expected_state=None, groups=None):
    """One restore rule for rank zero and followers; RNG restoration is last."""
    from RL.checkpoints import restore_checkpoint
    state = restore_checkpoint(directory, optimizer, scheduler, contract, contract["adapter"], verify=True)
    if expected_state is not None and state != expected_state:
        raise ValueError("Restored rank state differs from broadcast checkpoint state")
    if groups is not None:
        groups.validate_state(state)
    return state
