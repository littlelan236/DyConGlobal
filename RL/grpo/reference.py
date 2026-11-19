"""Frozen initial-policy reference helpers."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
import torch

from core.io import hash_json, read_json, sha256, write_json
from .contract import validate_scoring_identity


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
    return _write_reference(directory, groups, proof, identity, values)


def _validate_values(values, groups):
    if list(values) != [g.group_id for g in groups]:
        raise ValueError("Reference cache group coverage/order mismatch")
    for group in groups:
        if list(values[group.group_id]) != list(group.branch_ids):
            raise ValueError("Reference cache branch coverage/order mismatch")
        for i, branch_id in enumerate(group.branch_ids):
            row = values[group.group_id][branch_id]
            if (not isinstance(row, torch.Tensor) or row.ndim != 1
                    or len(row) != len(group.completion_token_ids[i])
                    or row.dtype != torch.float32 or row.requires_grad
                    or not torch.isfinite(row).all()):
                raise ValueError("Reference cache target values invalid")


def _write_reference(directory, groups, proof, identity, values, *, binding=None):
    """One writer; manifest is the last commit marker, never a partial cache."""
    directory = Path(directory)
    _validate_values(values, groups)
    manifest = {"schema_version": 2 if binding else 1, "identity": identity,
                "input_sha256": hash_json(proof), "entries": _entries(groups, proof, identity)}
    if binding:
        manifest["probability_binding"] = binding
    target = directory / "grpo/reference-logprobs.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target.parent / "reference.json").exists() or target.exists():
        raise FileExistsError("Reference cache is immutable; choose a new attempt")
    fd, name = tempfile.mkstemp(prefix=".reference-", suffix=".pt", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save({"identity": identity, "input_sha256": hash_json(proof), "groups": values}, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, target)
        write_json(directory / "grpo/reference.json", manifest)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    result = {"objective_version": "offline_grpo_k3_v1", "reference_identity": identity,
        "reference_manifest_sha256": sha256(directory / "grpo/reference.json"),
        "reference_snapshot_sha256": sha256(directory / "grpo/reference/adapter/identity.json"),
        "reference_logprob_cache_sha256": sha256(target)}
    if binding:
        digest = result["reference_logprob_cache_sha256"]
        result.update(cache_schema_version=2, probability_binding=binding,
                      role_binding={"ratio_denominator": digest, "k3_reference": digest})
    return values, result


def load_reference(directory, groups, proof, reference_proof, *, verify=True):
    directory = Path(directory)
    if not verify:
        payload = torch.load(directory / "grpo/reference-logprobs.pt", map_location="cpu", weights_only=True)
        values = payload["groups"]
        _validate_values(values, groups)
        return values
    manifest = read_json(directory / "grpo/reference.json")
    if (sha256(directory / "grpo/reference.json") != reference_proof["reference_manifest_sha256"]
            or sha256(directory / "grpo/reference-logprobs.pt") != reference_proof["reference_logprob_cache_sha256"]):
        raise ValueError("Reference cache manifest/content checksum mismatch")
    identity = reference_proof["reference_identity"]
    from .contract import validate_probability_binding
    validate_probability_binding(directory, manifest, reference_proof)
    if (manifest.get("identity") != identity or manifest.get("input_sha256") != hash_json(proof)
            or manifest.get("entries") != _entries(groups, proof, identity)):
        raise ValueError("Reference cache group/token/mask identity mismatch")
    payload = torch.load(directory / "grpo/reference-logprobs.pt", map_location="cpu", weights_only=True)
    if payload.get("identity") != identity or payload.get("input_sha256") != hash_json(proof):
        raise ValueError("Reference cache probability/model/input identity mismatch")
    values = payload["groups"]
    _validate_values(values, groups)
    return values


def load_reference_group(directory, group, proof, reference_proof):
    """Read one verified group from a sealed cache without loading all storage.

    The caller verifies checkpoint/artifact receipts before using this reader.
    Token/mask/anchor checks remain identical to the full-cache reader.
    """
    directory = Path(directory)
    manifest = read_json(directory / "grpo/reference.json")
    identity = reference_proof["reference_identity"]
    expected = _entries([group], proof, identity)
    selected = set(group.branch_ids)
    entries = [e for e in manifest["entries"]
               if e["group_id"] == group.group_id and e["branch_id"] in selected]
    if (manifest.get("identity") != identity or manifest.get("input_sha256") != hash_json(proof)
            or entries != expected):
        raise ValueError("Selected reference group/token/mask identity mismatch")
    payload = torch.load(directory / "grpo/reference-logprobs.pt", map_location="cpu", weights_only=True, mmap=True)
    if payload.get("identity") != identity or payload.get("input_sha256") != hash_json(proof):
        raise ValueError("Selected reference anchor/input identity mismatch")
    values = {group.group_id: {bid: payload["groups"][group.group_id][bid] for bid in group.branch_ids}}
    _validate_values(values, [group])
    return {bid: value.clone() for bid, value in values[group.group_id].items()}


def adapter_compatibility(model, *, start_mode, adapter_hash):
    """Prove base equivalence from actual PEFT state, not the word 'fresh'."""
    if start_mode == "warm_start":
        return {"kind": "adapter", "fingerprint": adapter_hash}
    if start_mode != "fresh":
        raise ValueError("External import requires fresh/warm_start; resume uses frozen cache")
    configs = getattr(model, "peft_config", {})
    if set(configs) != {"default"}:
        raise ValueError("Fresh LoRA base equivalence requires exactly one default adapter")
    settings = configs["default"]
    if (settings.bias != "none" or settings.modules_to_save
            or getattr(settings, "layer_replication", None)
            or getattr(settings, "trainable_token_indices", None)
            or getattr(settings, "use_dora", False)
            or getattr(settings, "lora_bias", False)
            or getattr(settings, "init_lora_weights", None) is not True):
        raise ValueError("Fresh LoRA initialization is not proven equivalent to base")
    layers = 0
    for module in model.modules():
        if hasattr(module, "modules_to_save") and module.modules_to_save:
            raise ValueError("Fresh LoRA modules-to-save changes base function")
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            layers += 1
            if (getattr(module, "merged", False) or list(module.active_adapters) != ["default"]
                    or "default" not in module.lora_B
                    or torch.count_nonzero(module.lora_B["default"].weight).item() != 0
                    or module.lora_B["default"].bias is not None
                    or module.lora_A["default"].bias is not None):
                raise ValueError("Fresh LoRA B/bias/active state is not zero increment")
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if (not layers or not trainable
            or any(".lora_A.default." not in n and ".lora_B.default." not in n for n in trainable)):
        raise ValueError("Fresh LoRA contains unproven trainable parameters")
    return {"kind": "base_equivalent", "zero_B_layers": layers,
            "bias": "none", "modules_to_save": None, "training_adapter_hash": adapter_hash}


def import_reference(model, groups, raw_groups, directory, proof, base, adapter_hash, config, *,
                     strict=False, anchor_adapter=None, source=None, shared_provenance_path=None,
                     shared_provenance_sha256=None):
    """Materialize a single shared anchor; this function never evaluates the model."""
    from RL.dataset.probability import open_probability_stream
    settings = config.grpo.reference_cache
    checked = strict or config.preflight.verify_model_state or config.preflight.verify_probabilities
    adapter = anchor_adapter or (adapter_compatibility(model, start_mode=config.run.start.mode, adapter_hash=adapter_hash)
               if checked else {"kind": "unchecked", "start_mode": config.run.start.mode,
                                "training_adapter_hash": adapter_hash, "verification": "disabled"})
    source = source or open_probability_stream(settings, config.algorithm, proof, strict=strict)
    if strict or config.preflight.verify_probabilities:
        validate_scoring_identity(source.scoring_identity, base=base, proof=proof,
            training_dtype=config.model.dtype, policy=settings.dtype_compatibility, adapter=adapter)
    rows = {}
    for raw in raw_groups:
        for row in raw["members"]:
            # Reindex training outputs by logical group. The unchanged row below
            # still addresses the original dataset group in the probability stream.
            key = (raw["group"]["group_id"], row["branch_sample_id"])
            if key in rows:
                raise ValueError("Duplicate frozen branch identity")
            rows[key] = row
    values = {}
    for group in groups:
        values[group.group_id] = {}
        for branch_id in group.branch_ids:
            row = rows[(group.group_id, branch_id)]
            if proof.get("synthetic"):
                from RL.dataset.probability import synthetic_branch
                row = synthetic_branch(row, proof["tokenizer_fingerprint"])
            array = source.values(row, verify=config.preflight.verify_trajectory_integrity)
            values[group.group_id][branch_id] = torch.from_numpy(array.copy()).detach()
    if strict or config.preflight.verify_probabilities:
        source.verify_unchanged()
    provenance = source.provenance
    if shared_provenance_path is None:
        provenance_path = Path(directory) / "grpo/source-provenance.json"
        write_json(provenance_path, provenance)
        provenance_path_value = "source-provenance.json"
        provenance_digest = sha256(provenance_path)
        binding_schema = 1
    else:
        provenance_path_value = str(shared_provenance_path)
        provenance_digest = shared_provenance_sha256
        binding_schema = 2
    binding = {"schema_version": binding_schema, "selection": settings.model_dump(),
        "source_receipt_sha256": source.receipt_sha256,
        "source_provenance_sha256": provenance_digest,
        "source_provenance_path": provenance_path_value,
        "scoring_identity": source.scoring_identity,
        "collector_dtype": source.scoring_identity["collector_dtype"],
        "training_compute_dtype": config.model.dtype,
        "dtype_compatibility": settings.dtype_compatibility,
        "training_initial_adapter": adapter,
        "preflight": config.preflight.model_dump(),
        "training_snapshot_purpose": "compatibility_evidence_only",
        "probability_semantics": "raw_t1_completion_targets_v1"}
    if "execution_cohorts" in provenance:
        binding["execution_cohorts"] = provenance["execution_cohorts"]
        binding["dataset_receipt_sha256"] = proof["dataset_receipt_sha256"]
    identity = ReferenceIdentity(base["fingerprint"], proof["tokenizer_fingerprint"], adapter_hash).as_dict()
    return _write_reference(directory, groups, proof, identity, values, binding=binding)


@torch.no_grad()
def diagnose_initial_current(model, groups, values, *, branch_count, block_size):
    """Explicit small diagnostic; computes current, never changes the anchor."""
    from .logprobs import model_target_logprobs
    rows = []
    was_training = model.training
    model.eval()
    try:
        # Round-robin across groups so a small request need not cover only one group.
        branches = [(group, i) for i in range(max(len(g.branch_ids) for g in groups))
                    for group in groups if i < len(group.branch_ids)]
        for group, index in branches[:branch_count]:
            branch_id = group.branch_ids[index]
            current = model_target_logprobs(model, group.branch_batch(index,
                next(model.parameters()).device), block_size).float().cpu()
            delta = values[group.group_id][branch_id] - current
            ratio = torch.exp(-delta)
            k3 = torch.expm1(delta) - delta
            if not all(torch.isfinite(v).all() for v in (current, delta, ratio, k3)):
                raise ValueError("Nonfinite external anchor/initial current diagnostic")
            q = torch.tensor([0., .25, .5, .75, 1.])
            rows.append({"group_id": group.group_id, "branch_id": branch_id,
                "delta_quantiles": torch.quantile(delta, q).tolist(),
                "ratio_quantiles": torch.quantile(ratio, q).tolist(),
                "k3_mean": float(k3.mean()), "k3_max": float(k3.max())})
    finally:
        model.train(was_training)
    return {"purpose": "initial_current_diagnostic", "branches": rows,
            "anchor_modified": False, "exact_equality_required": False}
