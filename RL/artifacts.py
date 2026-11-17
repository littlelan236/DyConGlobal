"""LoRA receipt scope and content identities using the project's IO and root registry."""

import importlib.metadata
import os
import subprocess
from pathlib import Path

from core.io import artifact_roots, hash_json, identifier, now, owned_path, read_json, sha256, write_json


def inventory(directory):
    directory = Path(directory).resolve()
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted inside LoRA artifacts: {path.name}")
        if path.is_file() and path != directory / "receipt.json":
            files[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return files


def commit_receipt(directory, kind, **metadata):
    directory = Path(directory)
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    receipt = {"schema_version": 1, "kind": kind, "created_at": now(), **metadata, "files": inventory(directory)}
    write_json(directory / "receipt.json", receipt)
    # A published directory is immutable. Flush directory entries after the commit marker.
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def read_receipt(directory, kind=None):
    """Read the committed identity without rescanning artifact contents."""
    directory = Path(directory).resolve()
    if not (directory / "receipt.json").is_file():
        raise ValueError("Uncommitted or incomplete LoRA artifact: receipt.json missing")
    receipt = read_json(directory / "receipt.json")
    if receipt.get("schema_version") != 1 or (kind and receipt.get("kind") != kind):
        raise ValueError("LoRA receipt kind/schema mismatch")
    if not isinstance(receipt.get("files"), dict) or not receipt["files"]:
        raise ValueError("LoRA receipt file inventory missing")
    for name in receipt["files"]:
        owned_path(directory, name)
    return receipt


def verify_receipt(directory, kind=None):
    receipt = read_receipt(directory, kind)
    if receipt["files"] != inventory(directory):
        raise ValueError("LoRA receipt file inventory/hash mismatch")
    return receipt


def base_identity(path, project_root, verify_content=True):
    from core.model_identity import describe

    path = Path(path)
    config = read_json(path / "config.json")
    text = config.get("text_config", {})
    if (config.get("model_type") != "qwen3_5"
            or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or text.get("model_type") != "qwen3_5_text"
            or (text.get("hidden_size"), text.get("num_hidden_layers"), text.get("vocab_size"))
            != (4096, 32, 248320)):
        raise ValueError("Only the local Qwen3.5-9B conditional-generation checkpoint is supported")
    if not (path / "tokenizer.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise ValueError("Local tokenizer files are required")
    registry = Path(project_root) / "runtime/model-identities"
    if verify_content:
        identity = describe(path)  # Deep content hashes, including every weight shard and tokenizer file.
        write_json(registry / (identity["fingerprint"] + ".json"), identity)
    else:
        from core.model_identity import verify_model
        identities = []
        for manifest in sorted(registry.glob("*.json")):
            try:
                identity = verify_model({"project_root": str(project_root),
                    "model": {"path": str(path), "fingerprint": manifest.stem,
                              "served_name": path.name}}, deep=False)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            identities.append(identity)
        if len(identities) != 1:
            raise ValueError("Unchecked startup needs one registered model identity matching file names/sizes; "
                             "set preflight.verify_identity=true to register/verify contents")
        identity = identities[0]
    return {**identity, "path": str(path)}


def dependency_versions():
    names = ["torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "dycon-global",
             "causal-conv1d", "flash-linear-attention"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def resume_contract_matches(expected, actual):
    """Project code hashes are audit metadata; all other training pins still match."""
    expected = dict(expected)
    actual = dict(actual)
    expected.pop("source_fingerprint", None)
    actual.pop("source_fingerprint", None)
    # Only an explicitly selected reference path may ignore unused optional
    # kernels. Core package versions and every other contract field stay strict.
    def reference(contract):
        identity = contract.get("gdn_execution")
        if identity is not None:
            return (identity.get("resolved_backend") == "reference_pytorch"
                    and identity.get("conv_backend") == "reference_pytorch")
        historical = (contract.get("attention") or {}).get("execution_backend", {})
        return historical == {"causal_conv1d": "reference_pytorch", "gated_delta_rule": "reference_pytorch"}
    if reference(expected) and reference(actual):
        if "gdn_execution" not in expected:
            # Interpret historical reference metadata without editing its bytes.
            actual.pop("gdn_execution", None)
        for contract in (expected, actual):
            if "dependencies" in contract:
                contract["dependencies"] = {k:v for k,v in contract["dependencies"].items()
                    if k not in {"causal-conv1d", "flash-linear-attention"}}
    return expected == actual


def source_identity(root):
    root = Path(root)
    paths = [*sorted((root / "RL").rglob("*.py")), root / "core/cli.py", root / "core/io.py",
             root / "core/locking.py", root / "core/model_identity.py", root / "core/config.py",
             root / "pyproject.toml", root / "locks/rl.in", root / "locks/rl.txt"]
    files = {str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=15).strip()
    return {"head": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines(),
            "files": files, "fingerprint": hash_json(files), "dependencies": dependency_versions()}


def _directory(cfg_or_root, artifact_id, storage_root=None):
    name = identifier(artifact_id)
    cfg = cfg_or_root if isinstance(cfg_or_root, dict) else {"project_root": str(cfg_or_root)}
    roots = [Path(storage_root).resolve()] if storage_root else artifact_roots(cfg, "lora-attempt")
    found = [(r / "RL/checkpoints" / name).resolve() for r in roots if (r / "RL/checkpoints" / name).exists()]
    if len(set(found)) != 1:
        raise ValueError("LoRA attempt missing or ambiguous; specify --storage-root")
    directory = found[0]
    if not any(directory == r / "RL/checkpoints" / name for r in roots):
        raise ValueError("LoRA attempt path escapes its storage root")
    return directory


def verify(cfg_or_root, scope, artifact_id, *, storage_root=None, dataset_id=None):
    if scope != "lora-attempt" or dataset_id is not None:
        raise ValueError("Invalid LoRA artifact scope/options")
    directory = _directory(cfg_or_root, artifact_id, storage_root)
    receipt = verify_receipt(directory, "lora-attempt-v1")
    attempt = read_json(directory / "attempt.json")
    required = {"training-config.yaml", "resolved-config.json", "attempt.json", "environment.json",
                "model-identity.json", "trainable-parameters.json", "metrics.jsonl", "validation.json",
                "report.md", "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                "adapter/identity.json"}
    if not required.issubset(receipt["files"]):
        raise ValueError("LoRA attempt required files missing")
    if (attempt.get("attempt_id") != artifact_id or attempt.get("purpose") != "engineering_smoke"
            or attempt.get("status") not in {"completed", "paused"}
            or receipt.get("status") != attempt["status"]):
        raise ValueError("LoRA attempt identity/status mismatch")
    from RL.checkpoints import verify_adapter, verify_checkpoint
    from RL.config import Config, schedule_contract

    config = Config.model_validate(read_json(directory / "resolved-config.json"))
    contract = attempt["contract"]
    base = read_json(directory / "model-identity.json")
    manifest = read_json(directory / "trainable-parameters.json")
    validation = read_json(directory / "validation.json")
    if (receipt.get("contract_sha256") != hash_json(contract)
            or contract["schedule"] != schedule_contract(config)
            or hash_json(base["files"]) != base["fingerprint"]
            or contract["adapter"]["base_fingerprint"] != base["fingerprint"]
            or contract["adapter"]["targets"] != manifest["targets"]
            or manifest["target_count"] != len(manifest["targets"])
            or manifest["trainable_parameters"] != config.lora.rank * sum(
                t["in_features"] + t["out_features"] for t in manifest["targets"])
            or attempt["final_state"] != validation["state"]
            or validation["input"] != contract["input"]
            or attempt["start_mode"] != config.run.start.mode):
        raise ValueError("LoRA attempt config/model/targets/state identity mismatch")
    if "supervision" in contract["input"]:
        if (not {"input-identity.json", "selection.json"}.issubset(receipt["files"])
                or read_json(directory / "input-identity.json") != contract["input"]
                or read_json(directory / "selection.json") != contract["input"]["supervision"]):
            raise ValueError("LoRA input/selection identity mismatch")
    expected_step = config.validation.stop_after_update or config.training.max_updates
    if (validation["state"]["global_step"] != expected_step
            or (attempt["status"] == "completed") != (expected_step == config.training.max_updates)):
        raise ValueError("LoRA attempt schedule/status mismatch")
    if config.algorithm is not None:
        from RL.grpo.contract import validate_reference_files, receipt_metadata
        from RL.grpo.metrics import read_metric_history
        validate_reference_files(directory, contract)
        extra = receipt_metadata(contract, validation["state"])
        if any(receipt.get(k) != v for k, v in extra.items()):
            raise ValueError("GRPO attempt receipt objective/cursor mismatch")
        required_grpo = {"grpo-report.json", "grpo-contract.json",
                         "input-identity.json", "selection.json"}
        from RL.grpo.stream_resume import checkpoint_kind
        kind = checkpoint_kind(contract)
        if kind == "streaming":
            from RL.grpo.stream_resume import resolve_checkpoint
            resolve_checkpoint(directory, contract)
            required_grpo.add("grpo/stream-resume.json")
        else:
            required_grpo.add("grpo/prepared-binding.json" if kind == "prepared" else "frozen-groups.json")
        if not required_grpo.issubset(receipt["files"]):
            raise ValueError("GRPO objective/report evidence missing")
        if (read_json(directory / "grpo-contract.json") != contract or
                read_json(directory / "input-identity.json") != contract["input"]):
            raise ValueError("GRPO frozen input/objective contract mismatch")
        history = read_metric_history(directory / "metrics.jsonl")
        report = read_json(directory / "grpo-report.json")
        if (len(history) != validation["state"]["global_step"] or not history
                or {k:v for k,v in report.items() if k not in {"preparation", "extension"}} != {"status": attempt["status"], "input": contract["input"],
                              "reference": contract["grpo"], "state": validation["state"], "updates": history}
                or any(history[-1].get(k) != v for k, v in validation["state"].items())):
            raise ValueError("GRPO report/metrics/cursor cannot be reconstructed")
        if "prepared" in contract:
            from RL.grpo.prepared import PreparedReader, resolve
            prepared = PreparedReader(resolve(contract["prepared"]))
            prepared.validate_state(validation["state"])
            if report.get("extension") != contract.get("extension") or "preparation" not in report:
                raise ValueError("GRPO prepared reuse/extension report missing")
    verify_adapter(directory / "adapter", contract["adapter"], verify_owner=False)
    count = len(receipt["files"])
    for checkpoint in sorted((directory / "checkpoints").glob("step-*")):
        item = verify_checkpoint(checkpoint)
        count += len(item["files"])
    parent = attempt.get("parent")
    if parent:
        owner = Path(parent["owner"])
        if sha256(owner / "receipt.json") != parent["receipt_sha256"]:
            raise ValueError("LoRA parent receipt changed")
        verify_receipt(owner)
    return {"scope": scope, "artifact_id": artifact_id, "path": str(directory), "verified": True,
            "status": attempt["status"], "files_checked": count,
            "receipt_sha256": sha256(directory / "receipt.json")}


def show(cfg_or_root, scope, artifact_id, **kwargs):
    result = verify(cfg_or_root, scope, artifact_id, **kwargs)
    return {**result, "attempt": read_json(Path(result["path"]) / "attempt.json")}
