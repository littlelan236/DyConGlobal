"""Torch-free identities and verification for the full-policy artifact protocol."""
from pathlib import Path

from core.io import hash_json, read_json, sha256
from RL.artifacts import read_receipt, verify_receipt
from RL.config import schedule_contract

KIND = "rl-full-checkpoint-v1"


def verify_checkpoint(path, *, deep=True):
    path = Path(path)
    receipt = (verify_receipt if deep else read_receipt)(path, KIND)
    required = {"state.json", "parameters.json", "anchor/manifest.json"}
    if not required.issubset(receipt["files"]):
        raise ValueError("Full checkpoint required metadata missing")
    metadata = read_json(path / "state.json")
    contract = metadata["contract"]
    from RL.full_backend import checkpoint_identity
    identity = checkpoint_identity(contract)
    if receipt.get("contract_sha256") != hash_json(contract):
        raise ValueError("Full checkpoint contract mismatch")
    if contract["parameters_sha256"] != hash_json(read_json(path / "parameters.json")):
        raise ValueError("Full checkpoint parameter inventory mismatch")
    anchor = read_json(path / "anchor/manifest.json")
    if hash_json(anchor) != contract["anchor_sha256"] or anchor["policy_identity"] != contract["initial_policy"]:
        raise ValueError("Full checkpoint frozen anchor mismatch")
    expected = set()
    for rank in range(contract["world_size"]):
        prefix = f"ranks/{rank}/"
        if identity is not None:
            name = prefix + "gdn.json"
            if name not in receipt["files"] or sha256(path / name) != receipt["files"][name]["sha256"]:
                raise ValueError("Full checkpoint GDN rank evidence missing or changed")
            gdn = read_json(path / name)
            if gdn["execution"] != identity or gdn["observations"]["calls"].get("chunk", 0) < 1:
                raise ValueError("Full checkpoint GDN rank execution mismatch or no chunk calls")
        for name in ("manifest.json", "runtime.pt"):
            if prefix + name not in receipt["files"]:
                raise ValueError("Full checkpoint rank shard missing")
        manifest = read_json(path / prefix / "manifest.json")
        if manifest["rank"] != rank or manifest["world_size"] != contract["world_size"]:
            raise ValueError("Full checkpoint shard topology mismatch")
        if set(manifest["parameters"]) != {p["name"] for p in read_json(path / "parameters.json")["parameters"]}:
            raise ValueError("Full checkpoint parameter shard coverage mismatch")
        for entry in manifest["parameters"].values():
            for tensor in (entry["weight"], *entry["optimizer"].values()):
                for part in tensor["parts"]:
                    expected.add(prefix + part["file"])
    if not expected.issubset(receipt["files"]):
        raise ValueError("Full checkpoint tensor shard missing")
    for name in required | {f"ranks/{r}/manifest.json" for r in range(contract["world_size"])}:
        if sha256(path / name) != receipt["files"][name]["sha256"]:
            raise ValueError("Full checkpoint metadata checksum mismatch")
    from RL.grpo.prepared import PreparedReader, resolve
    reader = PreparedReader(resolve(contract["prepared"]))
    reader.validate_state(metadata["state"])
    if reader.proof != contract["input"]:
        raise ValueError("Full checkpoint prepared input mismatch")
    return metadata


def source(config):
    if config.run.start.mode == "fresh":
        return None
    path = Path(config.run.start.checkpoint)
    metadata = verify_checkpoint(path, deep=True)
    return {"mode": config.run.start.mode, "path": str(path),
            "receipt_sha256": sha256(path / "receipt.json"), "metadata": metadata}


def base_identity(config, root):
    from core.model_identity import describe, verify_model
    from core.io import write_json
    path = Path(config.model.path)
    definition = read_json(path / "config.json")
    if (definition.get("model_type") != "qwen3_5"
            or definition.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or definition.get("text_config", {}).get("model_type") != "qwen3_5_text"):
        raise ValueError("Full training requires the dense Qwen3.5 conditional-generation model")
    registry = Path(root) / "runtime/model-identities"
    if config.preflight.verify_identity:
        identity = describe(path)
        write_json(registry / (identity["fingerprint"] + ".json"), identity)
    else:
        matches = []
        for candidate in registry.glob("*.json"):
            try:
                matches.append(verify_model({"project_root": str(root), "model": {
                    "path": str(path), "fingerprint": candidate.stem, "served_name": path.name}}, deep=False))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if len(matches) != 1:
            raise ValueError("Full startup needs one registered model identity; enable verify_identity")
        identity = matches[0]
    return {**identity, "path": str(path)}


def prepared(config, root, parent):
    from RL.grpo.prepared import PreparedReader, resolve, materialize, validate_config
    if config.run.start.mode == "resume":
        reader = PreparedReader(resolve(parent["metadata"]["contract"]["prepared"]))
    elif config.prepared_data is not None:
        reader = PreparedReader(resolve(config.prepared_data.model_dump()))
    else:
        reader = materialize(config, root, config.run.attempt_id[:110] + "-prepared")
    try:
        validate_config(reader, config)
    except ValueError:
        # Residency strategy is independent of immutable data preparation. Only
        # accept the exact checkpoint-bound artifact and unchanged data/math.
        from RL.optimizer_migration import schedule_matches, without_adam_strategy
        from RL.grpo.prepared import binding
        source = parent["metadata"]["contract"] if parent else None
        if (not source or config.run.start.mode != "resume"
                or source.get("prepared") != binding(reader.path)
                or not schedule_matches(config, source)
                or without_adam_strategy(schedule_contract(config)) != without_adam_strategy(reader.manifest["training_schedule"])):
            raise
    if config.prepared_data is not None:
        from RL.grpo.prepared import binding
        if binding(reader.path) != config.prepared_data.model_dump():
            raise ValueError("Configured prepared data differs from frozen checkpoint")
    return reader


def preflight(config, root, parent):
    if config.training.memory.host_offload.enabled:
        from RL.resources import memory_plan
        memory_plan(config)  # Cooperative all-local-rank admission, before CUDA.
    base = base_identity(config, root)
    if parent:
        contract = parent["metadata"]["contract"]
        if contract["base_fingerprint"] != base["fingerprint"]:
            raise ValueError("Full source base/tokenizer identity mismatch")
        if config.run.start.mode == "resume":
            from RL.optimizer_migration import schedule_matches
            if (not schedule_matches(config, contract)
                    or contract["world_size"] != len(config.runtime.gpus or [config.runtime.gpu])):
                raise ValueError("Full resume schedule/backend/topology mismatch")
            step = parent["metadata"]["state"]["global_step"]
            if step >= config.training.max_updates:
                raise ValueError("Full source schedule is already complete")
            if config.validation.stop_after_update is not None and config.validation.stop_after_update <= step:
                raise ValueError("Full pause target must follow restored update")
    if config.prepared_data is not None or (parent and config.run.start.mode == "resume"):
        reader = prepared(config, root, parent)
        from RL.grpo.contract import validate_input
        validate_input(base, reader.proof)
        base["grpo_dataset"] = reader.proof
    return base


def artifact(root_config, artifact_id, storage_root, *, deep):
    from RL.artifacts import _directory
    directory = _directory(root_config, artifact_id, storage_root)
    receipt = (verify_receipt if deep else read_receipt)(directory, "rl-full-attempt-v1")
    attempt = read_json(directory / "attempt.json")
    if attempt["attempt_id"] != artifact_id or attempt["status"] not in {"completed", "paused"}:
        raise ValueError("Full attempt identity/status mismatch")
    if attempt.get("validation_only") and attempt.get("final_checkpoint") is None:
        if attempt.get("resumable") is not False or receipt["status"] != attempt["status"]:
            raise ValueError("Invalid no-export validation attempt")
        if read_json(directory / "progress.json")["state"] != attempt["final_state"]:
            raise ValueError("No-export committed state mismatch")
        return {"attempt_id": artifact_id, "status": attempt["status"], "kind": receipt["kind"],
                "checkpoint": None, "resumable": False, "state": attempt["final_state"], "verified": deep}
    final = Path(attempt["final_checkpoint"])
    if not final.is_relative_to(directory / "checkpoints"):
        raise ValueError("Full checkpoint escapes attempt")
    metadata = verify_checkpoint(final, deep=deep)
    if metadata["state"] != attempt["final_state"] or receipt["status"] != attempt["status"]:
        raise ValueError("Full attempt committed state mismatch")
    return {"attempt_id": artifact_id, "status": attempt["status"],
            "kind": receipt["kind"], "checkpoint": str(final), "state": metadata["state"],
            "verified": deep}
