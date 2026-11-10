"""Torch-free checkpoint/reference contracts shared by service and verification."""
import math
from pathlib import Path

from core.io import hash_json, read_json, sha256


def validate_input(base, proof):
    if proof["synthetic"]:
        return
    model = proof.get("dataset_model") or {}
    if not isinstance(model, dict) or model.get("fingerprint") != base["fingerprint"]:
        raise ValueError("GRPO dataset model/tokenizer fingerprint differs from the training base")
    if not proof.get("tokenizer_fingerprint"):
        raise ValueError("GRPO dataset tokenizer fingerprint is missing")


def validate_resume(config, proof, metadata):
    from RL.config import schedule_contract
    contract = metadata["contract"]
    if contract.get("schedule") != schedule_contract(config) or contract.get("input") != proof:
        raise ValueError("GRPO resume schedule/dataset/objective/order identity mismatch")
    state = metadata["state"]
    n = proof["group_count"]
    step = state["global_step"]
    if step >= proof["planned_updates"] or state["group_cursor"] >= n:
        raise ValueError("GRPO epoch is already complete; use explicit warm_start")
    if (state["group_cursor"] != min(n, step * config.grpo.group_batch_size)
            or state["consumed_groups"] != state["group_cursor"]):
        raise ValueError("GRPO resume group cursor mismatch")
    if config.validation.stop_after_update is not None and config.validation.stop_after_update <= step:
        raise ValueError("Pause target must follow the restored global step")


def validate_reference_files(directory, contract):
    directory = Path(directory)
    reference = contract["grpo"]
    cache = directory / "grpo/reference-logprobs.pt"
    if not cache.is_file() or sha256(cache) != reference["reference_logprob_cache_sha256"]:
        raise ValueError("GRPO reference cache missing or checksum mismatch")
    snapshot = directory / "grpo/reference/adapter/identity.json"
    if not snapshot.is_file() or sha256(snapshot) != reference["reference_snapshot_sha256"]:
        raise ValueError("GRPO reference snapshot missing or identity mismatch")
    identity = read_json(snapshot)
    if identity["contract"] != contract["adapter"]:
        raise ValueError("GRPO reference model/LoRA identity mismatch")
    for name, proof in identity["files"].items():
        if sha256(snapshot.parent / name) != proof["sha256"]:
            raise ValueError("GRPO reference adapter snapshot checksum mismatch")
    manifest = directory / "grpo/reference.json"
    if not manifest.is_file() or sha256(manifest) != reference["reference_manifest_sha256"]:
        raise ValueError("GRPO reference manifest checksum mismatch")
    manifest_value = read_json(manifest)
    if (manifest_value["identity"] != reference["reference_identity"] or
            manifest_value["input_sha256"] != hash_json(contract["input"])):
        raise ValueError("GRPO reference manifest/input identity mismatch")


def receipt_metadata(contract, state):
    proof = contract["input"]
    reference = contract["grpo"]
    keys = ("objective_version", "dataset_id", "dataset_receipt_sha256", "advantage_policy",
            "advantage_config_sha256", "group_order_seed", "group_order_sha256", "group_batch_size",
            "max_epochs", "clip_epsilon", "kl_beta", "probability_mode")
    return {"algorithm": {"name": "offline_grpo_k3_v1"}, **{k: proof[k] for k in keys},
            **reference, **{k: state[k] for k in ("group_cursor", "consumed_groups", "consumed_branches", "consumed_tokens")}}
