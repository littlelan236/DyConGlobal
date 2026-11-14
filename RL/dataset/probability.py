"""Neutral, receipt-bound raw probability reader. No training roles or Torch."""
from pathlib import Path
import json

import numpy as np

from core.io import hash_json, owned_path, read_json, sha256
from .artifacts import directory, verify_local
from .build import iter_rows
from .materialize import read_array
from .schema import spans

RAW_CONTRACT = {"temperature": 1.0, "normalization": "full_vocabulary",
                "logits_stage": "pre_transform", "targets": "completion_tokens"}
RAW_SEMANTICS = {"raw_t1_output_tokens_v1", "raw_t1_sampled_token_v1"}
BINDINGS = ("group_id", "branch_sample_id", "native_request_id", "prompt_sha256",
            "completion_sha256", "prompt_count", "completion_count", "tokenizer_fingerprint")


def synthetic_branch(row, tokenizer_fingerprint):
    """Explicit engineering fixture binding; never used for native datasets."""
    p, c = row["prompt_token_ids"], row["completion_token_ids"]
    return {**row, "native_request_id": "synthetic:" + row["branch_sample_id"],
            "prompt_count": len(p), "completion_count": len(c),
            "tokenizer_fingerprint": tokenizer_fingerprint, "mask_spans": spans(len(p), len(c))}


REPLAY_METADATA = ("root-sampling.json", "models.json", "effective-policies.json", "validation.json")


def normalize_replay_manifest(manifest, metadata, tokenizer_fingerprint, *, verify=True):
    """Adapt the producer's native replay schema; retain all original evidence."""
    from .replay.validation import raw_contract, raw_policy, RAW_ORIGIN
    contract = metadata["root-sampling.json"]
    if verify and (manifest.get("schema_version") != 2
            or manifest.get("probability_semantics") != "raw_t1_output_tokens_v1"
            or manifest.get("origin") != RAW_ORIGIN or contract != raw_contract()
            or manifest.get("raw_contract_sha256") != hash_json(contract)):
        raise ValueError("Native replay raw scoring contract mismatch")
    models = metadata["models.json"]["models"]
    runtime = metadata["models.json"]["runtime"]
    identity = metadata["validation.json"]["identity"]
    if verify and len(models) != 1:
        raise ValueError("Native replay requires one scoring model cohort")
    key, value = next(iter(models.items()))
    model = value["model"]
    if verify and (key != hash_json(value) or value.get("dtype") != "float16"
            or runtime.get("logprobs_mode") != "raw_logprobs"
            or runtime.get("enable_trace_replay") is not True
            or runtime.get("model_inventory_sha256") != model.get("fingerprint")
            or not runtime.get("replay_implementation")
            or identity.get("raw_contract_sha256") != manifest["raw_contract_sha256"]
            or identity.get("dataset_receipt_sha256") != manifest["dataset_receipt_sha256"]):
        raise ValueError("Native replay scoring runtime/checkpoint identity mismatch")
    unsupported = ("adapter", "lora", "lora_request", "quantization", "rope_scaling", "rope_theta",
                   "hf_overrides", "steering", "speculative_config")
    if verify and any(model.get(k) or runtime.get("engine_options", {}).get(k) for k in unsupported):
        raise ValueError("Native replay model-function overrides are unsupported")
    policy = raw_policy()
    if verify and metadata["effective-policies.json"] != {policy["policy_sha256"]: policy}:
        raise ValueError("Native replay policy is not the fixed raw scoring policy")
    scoring = {"checkpoint_fingerprint": model["fingerprint"],
        "tokenizer_fingerprint": tokenizer_fingerprint, "adapter_fingerprint": None,
        "collector_dtype": value["dtype"], "effective_model_overrides": {},
        "runtime": runtime, "implementation": runtime["replay_implementation"]}
    return {**manifest, "raw_contract": RAW_CONTRACT, "scoring_identity": scoring}


class ProbabilityStream:
    def __init__(self, path, *, kind, manifest_name, index_name, dataset_receipt_sha256,
                 stream=None, verify_dataset_identity=True, verify=True, metadata_only=False):
        self.path = Path(path)
        self.verify = verify
        self.metadata_only = metadata_only
        if self.path.name.startswith(".") or self.path.name.endswith(".staging"):
            raise ValueError("Probability source is staging, not committed")
        if kind == "rl-probability":
            self._canonical(dataset_receipt_sha256)
            return
        receipt = verify_local(self.path, kind,
            verify_files=verify and not metadata_only and (verify_dataset_identity if kind == "rl-dataset" else True))
        if verify and metadata_only:
            for name in (manifest_name, index_name):
                if sha256(owned_path(self.path, name)) != receipt["files"].get(name, {}).get("sha256"):
                    raise ValueError("Probability source metadata checksum mismatch: " + name)
        self.receipt_sha256 = sha256(self.path / "receipt.json")
        manifest_path = owned_path(self.path, manifest_name)
        index_path = owned_path(self.path, index_name)
        manifest = read_json(manifest_path)
        source_manifest = manifest
        metadata_bytes = {}
        native_models = None
        native_policy_sha256 = None
        native_replay = (kind == "rl-replay" and "raw_contract_sha256" in manifest
                         and "scoring_identity" not in manifest)
        if native_replay:
            for name in REPLAY_METADATA:
                if name not in receipt["files"]:
                    raise ValueError("Native replay metadata missing from receipt")
                if verify and sha256(owned_path(self.path, name)) != receipt["files"][name]["sha256"]:
                    raise ValueError("Native replay identity metadata checksum mismatch: " + name)
                metadata_bytes[name] = owned_path(self.path, name).read_bytes().decode("utf-8")
            first = next(iter(iter_rows(index_path)), None)
            if not first:
                raise ValueError("Native replay probability index is empty")
            metadata = {k: json.loads(v) for k, v in metadata_bytes.items()}
            manifest = normalize_replay_manifest(manifest, metadata, first["tokenizer_fingerprint"], verify=verify)
            native_models = metadata["models.json"]["models"]
            from .replay.validation import raw_policy
            native_policy_sha256 = raw_policy()["policy_sha256"]
        if verify and (manifest_name not in receipt["files"] or index_name not in receipt["files"]
                or manifest.get("schema_version") != 2
                or manifest.get("probability_semantics") not in RAW_SEMANTICS
                or manifest.get("raw_contract") != RAW_CONTRACT
                or manifest.get("dataset_receipt_sha256") != dataset_receipt_sha256):
            raise ValueError("Probability source raw T=1 semantics/dataset binding mismatch")
        if verify and stream is not None and manifest.get("stream") != stream:
            raise ValueError("Dataset probability stream identifier mismatch")
        scoring = manifest.get("scoring_identity")
        required = {"checkpoint_fingerprint", "tokenizer_fingerprint", "adapter_fingerprint",
                    "collector_dtype", "effective_model_overrides", "runtime", "implementation"}
        if verify and (not isinstance(scoring, dict) or not required <= scoring.keys()
                or not scoring["runtime"] or not scoring["implementation"]):
            raise ValueError("Probability source scoring/runtime identity is incomplete")
        self.scoring_identity = scoring
        scoring_digest = hash_json(scoring)
        self.records = {}
        for record in iter_rows(index_path):
            if native_replay:
                from .replay.validation import RAW_ORIGIN
                if verify and (record.get("probability_semantics") != manifest["probability_semantics"]
                        or record.get("origin") != RAW_ORIGIN
                        or record.get("raw_contract_sha256") != manifest["raw_contract_sha256"]
                        or record.get("model_sha256") not in native_models
                        or record.get("policy_sha256") != native_policy_sha256
                        or record.get("count") != record["completion_count"]):
                    raise ValueError("Native replay branch raw/model/policy binding mismatch")
                record = {**record, "scoring_identity_sha256": scoring_digest,
                          "mask_spans": spans(record["prompt_count"], record["completion_count"])}
            key = (record["group_id"], record["branch_sample_id"])
            if verify and key in self.records:
                raise ValueError("Duplicate probability source branch identity")
            if verify and (record.get("probability_semantics", manifest["probability_semantics"]) != manifest["probability_semantics"]
                    or record.get("scoring_identity_sha256") != scoring_digest
                    or record["tokenizer_fingerprint"] != scoring["tokenizer_fingerprint"]):
                raise ValueError("Probability branch scoring/semantic identity mismatch")
            ref = record["values_ref"]
            if verify and (ref["dtype"] != "<f4" or ref["count"] != record["completion_count"]
                    or ref["shard"] not in receipt["files"]
                    or record.get("mask_spans") != spans(record["prompt_count"], record["completion_count"])):
                raise ValueError("Probability branch dtype/count/mask/shard binding mismatch")
            self.records[key] = record
        self.provenance = {"receipt_bytes": (self.path / "receipt.json").read_bytes().decode("utf-8"),
            "manifest": source_manifest, "manifest_sha256": sha256(manifest_path),
            "manifest_bytes": manifest_path.read_bytes().decode("utf-8"),
            "manifest_name": manifest_name, "index_name": index_name,
            "index_sha256": sha256(index_path), "scoring_identity": scoring,
            "source_path": str(self.path), "stream": stream}
        if native_replay:
            self.provenance.update(adapter_schema="native_replay_v2", metadata_bytes=metadata_bytes)

    def _canonical(self, dataset_receipt_sha256):
        from .probability_identity import canonical_scoring
        from .probability_artifacts import check_probability
        if self.verify:
            check_probability(self.path, metadata_only=self.metadata_only)
        manifest_path = self.path / "probability.json"
        index_path = self.path / "branches.jsonl"
        manifest = read_json(manifest_path)
        if self.verify and manifest["dataset"]["receipt_sha256"] != dataset_receipt_sha256:
            raise ValueError("Canonical probability dataset receipt mismatch")
        self.receipt_sha256 = sha256(self.path / "receipt.json")
        self.scoring_identity = canonical_scoring(manifest)
        self.records = {(r["group_id"], r["branch_sample_id"]): r for r in iter_rows(index_path)}
        self.provenance = {"receipt_bytes": (self.path / "receipt.json").read_text(encoding="utf-8"),
            "manifest": manifest, "manifest_sha256": sha256(manifest_path),
            "manifest_bytes": manifest_path.read_text(encoding="utf-8"),
            "manifest_name": "probability.json", "index_name": "branches.jsonl",
            "index_sha256": sha256(index_path), "scoring_identity": self.scoring_identity,
            "source_path": str(self.path), "stream": "raw_t1_output_tokens",
            "execution_cohorts": manifest["execution_cohorts"], "adapter_schema": "canonical_probability_v3"}

    def values(self, row, *, verify=None):
        verify = self.verify if verify is None else verify
        record = self.records.get((row["group_id"], row["branch_sample_id"]))
        if record is None:
            raise ValueError("Missing required probability source branch")
        if verify and (any(record[k] != row[k] for k in BINDINGS)
                or record["mask_spans"] != row["mask_spans"]
                or hash_json(row["prompt_token_ids"]) != record["prompt_sha256"]
                or hash_json(row["completion_token_ids"]) != record["completion_sha256"]):
            raise ValueError("Probability branch token/request/mask binding mismatch")
        values = read_array(self.path, record["values_ref"], verify=verify)
        if (values.dtype != np.dtype("<f4") or values.ndim != 1
                or len(values) != row["completion_count"] or not np.isfinite(values).all() or (values > 0).any()):
            raise ValueError("Probability source requires finite float32 completion targets")
        return values

    def verify_unchanged(self):
        if self.verify:
            if sha256(self.path / "receipt.json") != self.receipt_sha256:
                raise ValueError("Probability source receipt changed during import")
            for name, expected in ((self.provenance["manifest_name"], self.provenance["manifest_sha256"]),
                                   (self.provenance["index_name"], self.provenance["index_sha256"])):
                if sha256(owned_path(self.path, name)) != expected:
                    raise ValueError("Probability source metadata changed during import: " + name)

    def validate_dataset_rows(self, rows):
        remaining = set(self.records)
        for row in rows:
            key = (row["group_id"], row["branch_sample_id"])
            record = self.records.get(key)
            if record is not None:
                if any(record[k] != row[k] for k in BINDINGS) or record["mask_spans"] != row["mask_spans"]:
                    raise ValueError("Probability source has wrong dataset branch binding")
                remaining.discard(key)
        if remaining:
            raise ValueError("Probability source contains extra unbound dataset records")


def open_probability_stream(settings, dataset, proof, *, strict=False, scan_dataset=True, metadata_only=False):
    verify = strict or proof.get("preflight", {}).get("verify_probabilities", True)
    if settings.source == "probability_artifact":
        path = directory(settings.artifact.root, settings.artifact.scope, settings.artifact.id)
        canonical = settings.artifact.scope == "rl-probability"
        source = ProbabilityStream(path, kind=settings.artifact.scope,
            manifest_name="probability.json" if canonical else "replay.json",
            index_name="branches.jsonl", dataset_receipt_sha256=proof["dataset_receipt_sha256"],
            verify=verify, metadata_only=metadata_only)
        if verify and source.provenance["manifest"].get("probability_id" if canonical else "replay_id") != settings.artifact.id:
            raise ValueError("Probability artifact ID differs from manifest")
    elif settings.source == "dataset_probability":
        path = directory(dataset.dataset_root, "rl-dataset", dataset.dataset_id)
        verify_dataset_identity = proof.get("preflight", {}).get("verify_identity", True)
        verify_local(path, "rl-dataset", verify_files=verify and verify_dataset_identity and not metadata_only)
        if verify and sha256(path / "receipt.json") != proof["dataset_receipt_sha256"]:
            raise ValueError("Dataset probability receipt differs from frozen selection")
        streams = read_json(path / "dataset.json").get("probability_streams", {})
        if settings.stream not in streams:
            raise ValueError("Dataset has no explicitly selected probability stream")
        descriptor = streams[settings.stream]
        # Manifest resides within the dataset receipt. Its dataset binding is the
        # frozen dataset content identity, avoiding a circular receipt hash.
        source = ProbabilityStream(path, kind="rl-dataset", manifest_name=descriptor["manifest"],
            index_name=descriptor["index"], dataset_receipt_sha256=descriptor["dataset_binding"],
            stream=settings.stream, verify_dataset_identity=verify_dataset_identity, verify=verify,
            metadata_only=metadata_only)
    else:
        raise ValueError("External probability reader needs an explicit external source")
    if not verify or not scan_dataset:
        return source
    if proof.get("synthetic"):
        fixture = read_json(Path(__file__).parents[1] / "fixtures/grpo-native-smoke-v1.json")
        rows = (synthetic_branch(r, proof["tokenizer_fingerprint"])
                for g in fixture["groups"] for r in g["members"])
    else:
        from .loader import iter_groups
        dataset_path = directory(dataset.dataset_root, "rl-dataset", dataset.dataset_id)
        if sha256(dataset_path / "receipt.json") != proof["dataset_receipt_sha256"]:
            raise ValueError("Probability source training dataset receipt changed")
        rows = (r for g in iter_groups(dataset_path, with_tokens=False, data_ready_only=False,
            verify_identity=proof.get("preflight", {}).get("verify_identity", True)) for r in g["members"])
    source.validate_dataset_rows(rows)
    return source
