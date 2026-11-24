"""Torch-free checkpoint/reference contracts shared by service and verification."""
import math
import hashlib
import json
from pathlib import Path

from core.io import hash_json, read_json, sha256

_PROVENANCE_VALIDATION_CACHE = set()


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
    from RL.config_migration import schedule_matches
    if not schedule_matches(config, contract.get("schedule")) or contract.get("input") != proof:
        raise ValueError("GRPO resume schedule/dataset/objective/order identity mismatch")
    state = metadata["state"]
    n = proof["group_count"]
    step = state["global_step"]
    if step >= proof["planned_updates"] or state["group_cursor"] >= n:
        raise ValueError("GRPO epoch is already complete; use explicit warm_start")
    if "prepared" in contract:
        from .prepared import PreparedReader, resolve, validate_config
        reader = PreparedReader(resolve(contract["prepared"]))
        validate_config(reader, config)
        reader.validate_state(state)
    elif proof.get("execution_plan"):
        from .execution import validate_state
        validate_state(proof["execution_plan"], state)
    elif (state["group_cursor"] != min(n, step * config.grpo.group_batch_size)
            or state["consumed_groups"] != state["group_cursor"]):
        raise ValueError("GRPO resume group cursor mismatch")
    if config.validation.stop_after_update is not None and config.validation.stop_after_update <= step:
        raise ValueError("Pause target must follow the restored global step")


def validate_reference_files(directory, contract, *, verify_artifacts=True, snapshot_directory=None):
    directory = Path(directory)
    reference = contract["grpo"]
    from .stream_resume import checkpoint_kind
    if checkpoint_kind(contract) == "streaming":
        from .streaming import validate_reference
        return validate_reference(directory, contract)
    if reference.get("cache_schema_version") == 3:
        from .reference_artifact import validate_attempt
        return validate_attempt(directory, contract, verify_artifacts=verify_artifacts)
    cache = directory / "grpo/reference-logprobs.pt"
    if not cache.is_file() or sha256(cache) != reference["reference_logprob_cache_sha256"]:
        raise ValueError("GRPO reference cache missing or checksum mismatch")
    snapshot = (Path(snapshot_directory) / "identity.json" if snapshot_directory is not None
                else directory / "grpo/reference/adapter/identity.json")
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
    validate_probability_binding(directory, manifest_value, reference)
    if manifest_value.get("schema_version") == 1 and contract.get("schedule", {}).get("grpo", {}).get("reference_cache"):
        raise ValueError("External probability schedule cannot use a legacy model-forward cache")
    if manifest_value.get("schema_version") == 2:
        if contract["schedule"]["grpo"].get("reference_cache") != reference["probability_binding"]["selection"]:
            raise ValueError("GRPO reference source selection differs from schedule")
        binding = reference["probability_binding"]
        if "preflight" in binding and binding["preflight"] != contract["schedule"].get("preflight"):
            raise ValueError("GRPO source validation settings differ from training schedule")
        provenance = read_json(probability_provenance_path(directory, binding))
        if binding["selection"]["source"] == "dataset_probability":
            if binding["source_receipt_sha256"] != contract["input"]["dataset_receipt_sha256"]:
                raise ValueError("GRPO dataset probability source receipt mismatch")
        elif (provenance["manifest"]["dataset"]["receipt_sha256"] if provenance.get("adapter_schema") == "canonical_probability_v3"
              else provenance["manifest"]["dataset_receipt_sha256"]) != contract["input"]["dataset_receipt_sha256"]:
            raise ValueError("GRPO external probability source dataset receipt mismatch")
    if (manifest_value["identity"] != reference["reference_identity"] or
            manifest_value["input_sha256"] != hash_json(contract["input"])):
        raise ValueError("GRPO reference manifest/input identity mismatch")


def probability_provenance_path(directory, binding):
    chunk_path = Path(directory).resolve()
    if binding.get("schema_version") == 2:
        root = chunk_path
        while root.parent != root and not (root / "manifest.json").is_file():
            root = root.parent
        relative = Path(binding.get("source_provenance_path", ""))
        if not relative or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GRPO shared source provenance path is invalid")
        provenance_path = (root / relative).resolve()
        if not provenance_path.is_relative_to(root):
            raise ValueError("GRPO shared source provenance escapes artifact root")
    else:
        provenance_path = chunk_path / "grpo/source-provenance.json"
    return provenance_path


def validate_probability_binding(directory, manifest, reference):
    version = manifest.get("schema_version")
    if version == 1:
        if any(k in reference for k in ("cache_schema_version", "probability_binding", "role_binding")):
            raise ValueError("Legacy reference cannot carry an external binding")
        return
    if version != 2 or reference.get("cache_schema_version") != 2:
        raise ValueError("Unsupported GRPO reference cache schema")
    binding = reference.get("probability_binding")
    if not binding or manifest.get("probability_binding") != binding:
        raise ValueError("GRPO probability source binding mismatch")
    digest = reference["reference_logprob_cache_sha256"]
    if reference.get("role_binding") != {"ratio_denominator": digest, "k3_reference": digest}:
        raise ValueError("GRPO shared anchor role/hash binding mismatch")
    if binding.get("schema_version") not in (1, 2) or binding.get("probability_semantics") != "raw_t1_completion_targets_v1":
        raise ValueError("Unsupported GRPO probability binding semantics")
    from RL.config import ArtifactReferenceCache, DatasetReferenceCache
    selection = binding["selection"]
    cls = ArtifactReferenceCache if selection.get("source") == "probability_artifact" else DatasetReferenceCache
    selected = cls.model_validate(selection)
    if binding.get("dtype_compatibility") != selected.dtype_compatibility:
        raise ValueError("GRPO probability dtype rule binding mismatch")
    source_dtype = binding.get("collector_dtype")
    training_dtype = binding.get("training_compute_dtype")
    if not ((selected.dtype_compatibility == "same_dtype" and source_dtype == training_dtype
             and source_dtype in {"float16", "bfloat16"})
            or (selected.dtype_compatibility == "fp16_source_bf16_training"
                and source_dtype == "float16" and training_dtype == "bfloat16")):
        raise ValueError("GRPO probability dtype contract invalid")
    provenance_path = probability_provenance_path(directory, binding)
    if not provenance_path.is_file() or sha256(provenance_path) != binding["source_provenance_sha256"]:
        raise ValueError("GRPO local source provenance checksum mismatch")
    cache_key = (str(provenance_path), binding["source_provenance_sha256"], hash_json(binding), hash_json(reference.get("reference_identity", {})))
    if cache_key in _PROVENANCE_VALIDATION_CACHE:
        return
    provenance = read_json(provenance_path)
    if (hashlib.sha256(provenance["receipt_bytes"].encode("utf-8")).hexdigest() != binding["source_receipt_sha256"]
            or provenance["scoring_identity"] != binding["scoring_identity"]
            or binding["scoring_identity"]["collector_dtype"] != source_dtype):
        raise ValueError("GRPO source receipt/scoring identity mismatch")
    source_receipt = json.loads(provenance["receipt_bytes"])
    source_manifest_bytes = provenance["manifest_bytes"].encode("utf-8")
    source_manifest = json.loads(source_manifest_bytes)
    from RL.dataset.probability import RAW_CONTRACT, RAW_SEMANTICS, normalize_replay_manifest, REPLAY_METADATA
    normalized_manifest = source_manifest
    if provenance.get("adapter_schema") == "canonical_probability_v3":
        from RL.dataset.probability_identity import canonical_scoring
        scoring = canonical_scoring(source_manifest)
        if (selected.source != "probability_artifact" or selected.artifact.scope != "rl-probability"
                or source_manifest["probability_id"] != selected.artifact.id
                or source_manifest["dataset"]["receipt_sha256"] != binding.get("dataset_receipt_sha256")
                or source_manifest["execution_cohorts"] != binding.get("execution_cohorts")
                or source_manifest["execution_cohorts"] != provenance.get("execution_cohorts")):
            raise ValueError("GRPO canonical probability cohort/scope binding mismatch")
        normalized_manifest = {**source_manifest, "schema_version": 2, "scoring_identity": scoring}
    if provenance.get("adapter_schema") == "native_replay_v2":
        metadata_bytes = provenance["metadata_bytes"]
        if set(metadata_bytes) != set(REPLAY_METADATA):
            raise ValueError("Native replay local source metadata is incomplete")
        for name, raw in metadata_bytes.items():
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != source_receipt["files"][name]["sha256"]:
                raise ValueError("Native replay local source metadata checksum mismatch")
        normalized_manifest = normalize_replay_manifest(source_manifest,
            {k: json.loads(v) for k, v in metadata_bytes.items()}, binding["scoring_identity"]["tokenizer_fingerprint"])
    if (hashlib.sha256(source_manifest_bytes).hexdigest() != provenance["manifest_sha256"]
            or source_manifest != provenance["manifest"]
            or source_receipt["files"][provenance["manifest_name"]]["sha256"] != provenance["manifest_sha256"]
            or source_receipt["files"][provenance["index_name"]]["sha256"] != provenance["index_sha256"]
            or normalized_manifest.get("schema_version") != 2
            or normalized_manifest.get("probability_semantics") not in RAW_SEMANTICS
            or normalized_manifest.get("raw_contract") != RAW_CONTRACT
            or normalized_manifest.get("scoring_identity") != binding["scoring_identity"]):
        raise ValueError("GRPO source receipt/manifest raw semantics mismatch")
    scoring = binding["scoring_identity"]
    identity = reference["reference_identity"]
    adapter = binding["training_initial_adapter"]
    if (scoring["checkpoint_fingerprint"] != identity["model_fingerprint"]
            or scoring["tokenizer_fingerprint"] != identity["tokenizer_fingerprint"]
            or scoring["effective_model_overrides"] != {}):
        raise ValueError("GRPO source model/tokenizer compatibility mismatch")
    if adapter["kind"] == "unchecked":
        checks = binding.get("preflight", {})
        if (checks.get("verify_model_state") is not False or checks.get("verify_probabilities") is not False
                or adapter.get("verification") != "disabled" or adapter.get("start_mode") not in {"fresh", "warm_start"}
                or adapter.get("training_adapter_hash") != identity["adapter_hash"]):
            raise ValueError("GRPO unchecked adapter declaration/settings mismatch")
    elif adapter["kind"] == "base_equivalent":
        if (scoring["adapter_fingerprint"] is not None
                or adapter["training_adapter_hash"] != identity["adapter_hash"]
                or adapter["zero_B_layers"] < 1 or adapter["bias"] != "none"
                or adapter["modules_to_save"] is not None):
            raise ValueError("GRPO fresh LoRA base equivalence binding mismatch")
    elif adapter["kind"] != "adapter" or not (adapter["fingerprint"] == identity["adapter_hash"] == scoring["adapter_fingerprint"]):
        raise ValueError("GRPO scoring/training adapter binding mismatch")
    _PROVENANCE_VALIDATION_CACHE.add(cache_key)


def receipt_metadata(contract, state):
    proof = contract["input"]
    reference = contract["grpo"]
    keys = ("objective_version", "dataset_id", "dataset_receipt_sha256", "advantage_policy",
            "advantage_config_sha256", "group_order_seed", "group_order_sha256", "group_batch_size",
            "max_epochs", "clip_epsilon", "kl_beta", "probability_mode")
    extra = {}
    if "prepared" in contract:
        extra = {"resume_mode":"append_resume" if "extension" in contract else "exact",
                 "prepared_artifact_id":contract["prepared"]["id"],
                 "prepared_manifest_sha256":contract["prepared"]["manifest_sha256"],
                 "schedule_id":contract["prepared_schedule_id"],
                 "schedule_manifest_sha256":contract["prepared_schedule_sha256"],
                 **{k:state[k] for k in ("segment_id", "segment_ordinal", "last_consumed_group_id", "consumed_group_ids_sha256")}}
    if "same_action_grouping" in proof:
        extra["same_action_grouping"] = proof["same_action_grouping"]
    if proof.get("execution_plan"):
        from .execution import validate, validate_state
        validate(proof["execution_plan"])
        validate_state(proof["execution_plan"], state)
        extra.update({k: state[k] for k in ("cursor_schema", "execution_schedule_sha256",
            "execution_window_cursor", "execution_round", "consumed_execution_chunks",
            "planned_parent_groups", "planned_execution_chunks", "distinct_parent_groups_seen",
            "completed_parent_groups", "consumed_half_ids_sha256")})
    return {**extra, "algorithm": {"name": "offline_grpo_k3_v1"}, **{k: proof[k] for k in keys},
            **reference, **{k: state[k] for k in ("group_cursor", "consumed_groups", "consumed_branches", "consumed_tokens")}}


def validate_scoring_identity(scoring, *, base, proof, training_dtype, policy, adapter):
    if (scoring.get("checkpoint_fingerprint") != base["fingerprint"]
            or scoring.get("tokenizer_fingerprint") != proof["tokenizer_fingerprint"]):
        raise ValueError("External probability checkpoint/tokenizer identity mismatch")
    expected_adapter = None if adapter["kind"] == "base_equivalent" else adapter["fingerprint"]
    if scoring.get("adapter_fingerprint") != expected_adapter or "adapter_fingerprint" not in scoring:
        raise ValueError("External scoring adapter differs from training initial policy")
    dtype = scoring.get("collector_dtype")
    if not ((policy == "same_dtype" and dtype == training_dtype)
            or (policy == "fp16_source_bf16_training" and dtype == "float16" and training_dtype == "bfloat16")):
        raise ValueError("External probability dtype compatibility mismatch")
    # Checkpoint fingerprint covers config/rope/tokenizer/weights. Runtime overrides
    # that alter the model function must be absent, not guessed from a model name.
    if scoring.get("effective_model_overrides") != {}:
        raise ValueError("External scoring model overrides/quantization/steering are unsupported")
