"""Frozen initial-policy reference helpers."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import torch

from core.io import hash_json, read_json, sha256, write_json


@dataclass(frozen=True)
class ReferenceIdentity:
    model_fingerprint: str
    tokenizer_fingerprint: str
    adapter_hash: str
    objective_version: str = "offline_grpo_k3_v1"

    def as_dict(self):
        return self.__dict__.copy()


def freeze_reference(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cache_key(identity, *, dataset_receipt_sha256, group_id, branch_id, mask_policy,
              probability_mode="raw_t1"):
    identity_values = identity.as_dict() if hasattr(identity, "as_dict") else dict(identity)
    payload = {**identity_values,
               "dataset_receipt_sha256": dataset_receipt_sha256, "group_id": group_id,
               "branch_id": branch_id, "mask_policy": mask_policy,
               "probability_mode": probability_mode}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ReferenceLogprobCache:
    """Small identity-bound cache; a miss is explicit and never silently reused."""
    def __init__(self, root, identity):
        self.root = __import__("pathlib").Path(root)
        self.identity = identity

    def path(self, key):
        return self.root / (str(key) + ".pt")

    def put(self, key, values, mask):
        self.root.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": self.identity.as_dict() if hasattr(self.identity, "as_dict") else dict(self.identity),
                    "key": key, "values": values.detach().cpu(), "mask": mask.detach().cpu()}, self.path(key))

    def get(self, key):
        path = self.path(key)
        if not path.is_file():
            return None
        value = torch.load(path, map_location="cpu", weights_only=True)
        expected = self.identity.as_dict() if hasattr(self.identity, "as_dict") else dict(self.identity)
        if value.get("key") != key or value.get("identity") != expected:
            raise ValueError("Reference logprob cache identity mismatch")
        return value["values"], value["mask"]


@torch.no_grad()
def reference_logprobs(model, batch, *, selected_fn):
    was_training = model.training
    model.eval()
    logits = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits
    result = selected_fn(logits, batch["input_ids"], batch.get("labels"), batch.get("loss_mask"))
    if was_training:
        model.train()
    return result


def _entries(groups, proof, identity):
    entries = []
    for group in groups:
        for index, branch_id in enumerate(group.branch_ids):
            entries.append({"group_id": group.group_id, "branch_id": branch_id,
                "cache_key": cache_key(identity, dataset_receipt_sha256=proof["dataset_receipt_sha256"],
                    group_id=group.group_id, branch_id=branch_id, mask_policy=proof["mask_policy"]),
                "prompt_sha256": hash_json(group.prompt_token_ids[index]),
                "completion_sha256": hash_json(group.completion_token_ids[index]),
                "loss_mask_sha256": hash_json(group.loss_mask[index].tolist()),
                "target_count": int(group.loss_mask[index].sum())})
    return entries


@torch.no_grad()
def build_reference(model, groups, directory, proof, base, adapter_hash, block_size):
    from .logprobs import model_target_logprobs
    directory = Path(directory)
    identity = ReferenceIdentity(base["fingerprint"], proof["tokenizer_fingerprint"], adapter_hash).as_dict()
    values = {}
    training = model.training
    model.eval()
    try:
        for group in groups:
            members = {}
            for index, branch_id in enumerate(group.branch_ids):
                batch = group.branch_batch(index, next(model.parameters()).device)
                targets = model_target_logprobs(model, batch, block_size).float().cpu()
                if not torch.isfinite(targets).all():
                    raise ValueError("Nonfinite initial-policy reference logprobs")
                members[branch_id] = targets
            values[group.group_id] = members
    finally:
        model.train(training)
    manifest = {"schema_version": 1, "identity": identity,
                "input_sha256": hash_json(proof), "entries": _entries(groups, proof, identity)}
    write_json(directory / "grpo/reference.json", manifest)
    torch.save({"identity": identity, "input_sha256": hash_json(proof), "groups": values},
               directory / "grpo/reference-logprobs.pt")
    return values, {"objective_version": "offline_grpo_k3_v1", "reference_identity": identity,
        "reference_manifest_sha256": sha256(directory / "grpo/reference.json"),
        "reference_snapshot_sha256": sha256(directory / "grpo/reference/adapter/identity.json"),
        "reference_logprob_cache_sha256": sha256(directory / "grpo/reference-logprobs.pt")}


def load_reference(directory, groups, proof, reference_proof):
    directory = Path(directory)
    manifest = read_json(directory / "grpo/reference.json")
    identity = reference_proof["reference_identity"]
    if (manifest.get("identity") != identity or manifest.get("input_sha256") != hash_json(proof)
            or manifest.get("entries") != _entries(groups, proof, identity)):
        raise ValueError("Reference cache group/token/mask identity mismatch")
    payload = torch.load(directory / "grpo/reference-logprobs.pt", map_location="cpu", weights_only=True)
    if payload.get("identity") != identity or payload.get("input_sha256") != hash_json(proof):
        raise ValueError("Reference cache probability/model/input identity mismatch")
    values = payload["groups"]
    if set(values) != {g.group_id for g in groups}:
        raise ValueError("Reference cache group coverage mismatch")
    for group in groups:
        if set(values[group.group_id]) != set(group.branch_ids):
            raise ValueError("Reference cache branch coverage mismatch")
        for i, branch_id in enumerate(group.branch_ids):
            row = values[group.group_id][branch_id]
            if (row.ndim != 1 or len(row) != len(group.completion_token_ids[i])
                    or row.dtype != torch.float32 or not torch.isfinite(row).all()):
                raise ValueError("Reference cache target values invalid")
    return values
