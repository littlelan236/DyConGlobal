"""Sealed reference chunks, one initial-policy anchor, and immutable parent reuse."""
from pathlib import Path
import shutil

from core.io import hash_json, read_json, sha256, write_json
from RL.dataset.artifacts import publication, seal, verify_local
from .prepared import PreparedReader, binding, resolve
from .data import tensor_groups


class ReferenceReader:
    def __init__(self, path, *, prepared=None, ancestors=()):
        self.path = Path(path)
        if str(self.path.resolve()) in ancestors or len(ancestors) >= 100:
            raise ValueError("Reference parent cycle/depth exceeded")
        verify_local(self.path, "rl-reference")
        self.manifest = read_json(self.path / "manifest.json")
        m = self.manifest
        if m.get("schema_version") != 1 or m.get("reference_id") != self.path.name:
            raise ValueError("Reference artifact schema/ID mismatch")
        self.prepared = prepared or PreparedReader(resolve(m["prepared"]))
        if binding(self.prepared.path) != m["prepared"]:
            raise ValueError("Reference/prepared artifact binding mismatch")
        self.parent = ReferenceReader(resolve(m["parent"], "rl-reference"), ancestors=(*ancestors, str(self.path.resolve()))) if m.get("parent") else None
        if self.parent and (self.parent.manifest["identity"] != m["identity"]
                            or self.parent.manifest["anchor_adapter"] != m["anchor_adapter"]
                            or self.prepared.manifest["parent"] != self.parent.manifest["prepared"]):
            raise ValueError("Append reference initial-policy anchor/parent mismatch")
        self.owners = dict(self.parent.owners) if self.parent else {}
        from .contract import validate_reference_files
        from RL.checkpoints import verify_adapter
        verify_adapter(self.path/"anchor/adapter", m["adapter_contract"], verify_owner=False)
        for chunk in m["chunks"]:
            folder = self.path / chunk["name"]
            if (not folder.is_relative_to(self.path) or ".." in Path(chunk["name"]).parts
                    or sha256(folder / "grpo/reference.json") != chunk["reference"]["reference_manifest_sha256"]):
                raise ValueError("Reference chunk manifest checksum mismatch")
            if chunk["reference"]["reference_identity"] != m["identity"]:
                raise ValueError("Reference chunk anchor identity mismatch")
            validate_reference_files(folder, chunk["contract"], snapshot_directory=self.path/"anchor/adapter")
            entries = read_json(folder / "grpo/reference.json")["entries"]
            indexed = [(e["group_id"], e["branch_id"]) for e in entries]
            expected = []
            for group_id in chunk["group_ids"]:
                if group_id in self.owners or group_id not in self.prepared.local_ids:
                    raise ValueError("Reference chunk duplicate/nonlocal group identity")
                expected.extend((group_id, bid) for bid in self.prepared.branch_ids_for(group_id))
                self.owners[group_id] = (self, chunk)
            if indexed != expected:
                raise ValueError("Reference chunk branch coverage/order mismatch")
        if set(self.owners) != set(self.prepared.ids):
            raise ValueError("Reference artifact has incomplete group coverage")
        self.cached_chunk = None
        self.cached_values = None

    def __getitem__(self, group_id):
        owner, chunk = self.owners[group_id]
        if owner is not self:
            return owner[group_id]
        if self.cached_chunk != chunk["name"]:
            from .reference import load_reference
            groups = list(tensor_groups(self.prepared.raw(g) for g in chunk["group_ids"]))
            self.cached_values = load_reference(self.path / chunk["name"], groups,
                chunk["contract"]["input"], chunk["reference"], verify=True)
            self.cached_chunk = chunk["name"]
        return self.cached_values[group_id]

    def for_group(self, group):
        """Map the sealed parent cache, materializing only this execution half.

        The artifact and chunk hashes were checked on attachment. Tensor storage
        is memory-mapped; unselected completion arrays are not copied into RAM.
        Selected token/mask identities use the same reference manifest entries.
        """
        from .reference import load_reference_group
        owner, chunk = self.owners[group.group_id]
        if owner is not self:
            return owner.for_group(group)
        return load_reference_group(self.path / chunk["name"], group,
                                    chunk["contract"]["input"], chunk["reference"])

    def summary(self):
        return {"reference_id":self.path.name, "groups":len(self.owners),
                "local_chunks":len(self.manifest["chunks"]), "identity":self.manifest["identity"],
                "parent":self.manifest["parent"], "manifest_sha256":sha256(self.path/"manifest.json")}


def publish(model, config, prepared, base, adapter_hash, adapter_identity, snapshot,
            *, parent=None, chunk_groups=1):
    """Compute/import only local new groups; old shards remain receipt-bound references."""
    from .reference import build_reference, import_reference, adapter_compatibility
    from RL.memory import attention_context
    anchor_adapter = (parent.manifest["anchor_adapter"] if parent else
        adapter_compatibility(model, start_mode=config.run.start.mode, adapter_hash=adapter_hash))
    reference_id = config.run.attempt_id[:110] + "-reference"
    identity = {"model_fingerprint":base["fingerprint"], "tokenizer_fingerprint":prepared.proof["tokenizer_fingerprint"],
                "adapter_hash":adapter_hash, "objective_version":"offline_grpo_k3_v1"}
    if parent and identity != parent.manifest["identity"]:
        raise ValueError("Append reference tokenizer/model/adapter anchor mismatch")
    stream = None
    if config.grpo.reference_cache.source != "model_forward":
        from RL.dataset.probability import open_probability_stream
        stream = open_probability_stream(config.grpo.reference_cache, config.algorithm, prepared.proof,
                                         strict=True, scan_dataset=False)
        from .contract import validate_scoring_identity
        validate_scoring_identity(stream.scoring_identity, base=base, proof=prepared.proof,
            training_dtype=config.model.dtype, policy=config.grpo.reference_cache.dtype_compatibility,
            adapter=anchor_adapter)
        if parent and parent.manifest.get("scoring_identity") not in (None, stream.scoring_identity):
            raise ValueError("Append reference source scoring identity/dtype mismatch")
    elif parent and parent.manifest.get("scoring_identity") is not None:
        raise ValueError("Append must preserve the external source scoring identity and dtype")
    with publication(config.runtime.output_root, "rl-reference", reference_id,
                     signature=hash_json({"prepared":binding(prepared.path), "identity":identity})) as (stage, destination):
        shutil.copytree(snapshot, stage/"anchor/adapter")
        shared_provenance_path = None
        shared_provenance_sha256 = None
        if stream is not None:
            shared_provenance_path = "source-provenance.json"
            write_json(stage / shared_provenance_path, stream.provenance)
            shared_provenance_sha256 = sha256(stage / shared_provenance_path)
        chunks = []
        ids = [g for g in prepared.ids if g in prepared.local_ids]
        for offset in range(0, len(ids), chunk_groups):
            group_ids = ids[offset:offset+chunk_groups]
            name = f"chunks/chunk-{len(chunks):06d}"
            folder = stage / name
            local_identity = folder/"grpo/reference/adapter/identity.json"
            local_identity.parent.mkdir(parents=True)
            local_identity.write_bytes((Path(snapshot)/"identity.json").read_bytes())
            raw = [prepared.raw(g) for g in group_ids]
            groups = list(tensor_groups(raw))
            chunk_proof = {k:prepared.proof[k] for k in
                ("dataset_id", "dataset_receipt_sha256", "tokenizer_fingerprint", "mask_policy", "synthetic")}
            if stream is None:
                with attention_context(config.training.memory.attention_policy):
                    _, reference = build_reference(model, groups, folder, chunk_proof, base, adapter_hash,
                                                  config.training.memory.loss_chunk_tokens)
            else:
                _, reference = import_reference(model, groups, raw, folder, chunk_proof, base,
                    adapter_hash, config, strict=True, anchor_adapter=anchor_adapter, source=stream,
                    shared_provenance_path=shared_provenance_path,
                    shared_provenance_sha256=shared_provenance_sha256)
            contract = {"grpo":reference, "adapter":adapter_identity,
                        "input":chunk_proof, "schedule":schedule_contract_for_chunk(config)}
            chunks.append({"name":name, "group_ids":group_ids, "reference":reference, "contract":contract})
        if stream is not None:
            stream.verify_unchanged()
        write_json(stage / "manifest.json", {"schema_version":1, "reference_id":reference_id,
            "identity":identity, "anchor_adapter":anchor_adapter, "adapter_contract":adapter_identity,
            "prepared":binding(prepared.path), "parent":binding(parent.path, "rl-reference") if parent else None,
            "scoring_identity":stream.scoring_identity if stream else None,
            "training_dtype":config.model.dtype, "chunks":chunks})
        seal(stage, "rl-reference", reference_id)
    return ReferenceReader(destination, prepared=prepared)


def schedule_contract_for_chunk(config):
    from RL.config import schedule_contract
    return schedule_contract(config)


def bind_attempt(directory, reader, proof):
    directory = Path(directory)
    manifest = {"schema_version":3, "reference_artifact":binding(reader.path, "rl-reference"),
                "input_sha256":hash_json(proof), "identity":reader.manifest["identity"]}
    write_json(directory / "grpo/reference.json", manifest)
    digest = manifest["reference_artifact"]["manifest_sha256"]
    return {"cache_schema_version":3, "objective_version":"offline_grpo_k3_v1",
        "reference_artifact":manifest["reference_artifact"],
        "reference_identity":manifest["identity"], "reference_manifest_sha256":sha256(directory/"grpo/reference.json"),
        "reference_snapshot_sha256":sha256(directory/"grpo/reference/adapter/identity.json"),
        "reference_logprob_cache_sha256":digest,
        "role_binding":{"ratio_denominator":digest, "k3_reference":digest}}


def validate_attempt(directory, contract, *, verify_artifacts=True):
    directory = Path(directory)
    proof = contract["grpo"]
    pointer = read_json(directory / "grpo/reference.json")
    if (pointer.get("schema_version") != 3 or pointer["reference_artifact"] != proof["reference_artifact"]
            or pointer["input_sha256"] != hash_json(contract["input"])
            or pointer["identity"] != proof["reference_identity"]
            or sha256(directory/"grpo/reference.json") != proof["reference_manifest_sha256"]
            or sha256(directory/"grpo/reference/adapter/identity.json") != proof["reference_snapshot_sha256"]):
        raise ValueError("Reference artifact/input/snapshot binding mismatch")
    digest = proof["reference_artifact"]["manifest_sha256"]
    if (proof["reference_logprob_cache_sha256"] != digest or
            proof["role_binding"] != {"ratio_denominator":digest, "k3_reference":digest}):
        raise ValueError("Reference shared anchor role binding mismatch")
    from RL.checkpoints import verify_adapter
    verify_adapter(directory/"grpo/reference/adapter", contract["adapter"], verify_owner=False)
    if not verify_artifacts:
        path = resolve(proof["reference_artifact"], "rl-reference")
        verify_local(path, "rl-reference", verify_files=False)
        manifest = read_json(path/"manifest.json")
        if manifest["prepared"] != contract.get("prepared") or manifest["identity"] != pointer["identity"]:
            raise ValueError("Reference prepared/anchor identity mismatch")
        return None
    reader = ReferenceReader(resolve(proof["reference_artifact"], "rl-reference"))
    if reader.manifest["prepared"] != contract.get("prepared") or reader.manifest["identity"] != pointer["identity"]:
        raise ValueError("Reference prepared/anchor identity mismatch")
    return reader
