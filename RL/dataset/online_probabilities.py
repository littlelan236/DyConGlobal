"""Neutral dataset-bound raw probability stream; independent of training roles."""
from pathlib import Path

from core.io import hash_json, owned_path, read_json, sha256
from runtime.serving.raw_logprobs import STREAM, read_values


def bind_online_probability(native, session, row, source_config, dependencies):
    descriptor = native.get("raw_t1_probability")
    if not descriptor or descriptor.get("status") == "failed":
        return None
    # Always validate probability integrity, including audit-disabled datasets.
    read_values(session["root"], descriptor, output_ids=native["output_token_ids"], record=native)
    identity = descriptor["identity"]
    for key, expected in (("request_id", row["native_request_id"]), ("session_id", row["session_id"]),
                          ("decision_id", row["decision_id"]), ("branch_id", row["branch_id"]),
                          ("step_id", row["agent_step"]), ("api_attempt_id", row["api_attempt_id"]),
                          ("run_id", row["source_run_id"])):
        if identity.get(key) != expected:
            raise ValueError("Online probability branch binding mismatch: " + key)
    source = descriptor["source_identity"]
    if (source.get("model_fingerprint") != source_config["model"]["fingerprint"] or
            source.get("serving_dtype") != source_config["serving"].get("dtype")):
        raise ValueError("Online probability scoring model/dtype mismatch")
    ledger = owned_path(session["root"], row["native_evidence"]["ledger_ref"])
    sidecar = owned_path(session["root"], descriptor["locator"])
    receipt = Path(session["base"]) / "receipt.json"
    # Dataset receipt freezes the actual published bytes, regardless of the
    # optional general-purpose native audit setting.
    proof = {"ledger_ref": row["native_evidence"]["ledger_ref"],
             "ledger_sha256": sha256(ledger), "session_receipt_ref": str(receipt.relative_to(session["root"])),
             "session_receipt_sha256": sha256(receipt),
             "publication_sha256": hash_json(session["publication"])}
    for path in (ledger, sidecar, receipt):
        dependencies[str(path.resolve())] = sha256(path)
    return {"root": str(session["root"]), "descriptor": descriptor, "source_receipt": proof}


class OnlineProbabilityReader:
    def __init__(self, dataset, stream=STREAM):
        if stream != STREAM:
            raise ValueError("Unknown online probability stream: " + stream)
        from .artifacts import verify_local
        self.dataset = Path(dataset)
        verify_local(self.dataset, "rl-dataset")
        self.stream = stream
        self.receipt_sha256 = sha256(self.dataset / "receipt.json")

    def record(self, row):
        binding = (row.get("probability_streams") or {}).get(self.stream)
        if not binding:
            raise ValueError("Dataset branch has no raw T=1 probability stream: " + row["branch_sample_id"])
        root = binding["root"]
        proof = binding["source_receipt"]
        ledger = owned_path(root, proof["ledger_ref"])
        receipt = owned_path(root, proof["session_receipt_ref"])
        if sha256(ledger) != proof["ledger_sha256"] or sha256(receipt) != proof["session_receipt_sha256"]:
            raise ValueError("Online probability source receipt changed")
        native = read_json(ledger)
        descriptor = binding["descriptor"]
        if descriptor != native.get("raw_t1_probability"):
            raise ValueError("Online probability descriptor changed")
        values = read_values(root, descriptor, output_ids=native["output_token_ids"], record=native)
        if (hash_json(native["prompt_token_ids"]) != row["prompt_sha256"] or
                hash_json(native["output_token_ids"]) != row["completion_sha256"] or
                descriptor["count"] != row["completion_count"]):
            raise ValueError("Online probability dataset token binding mismatch")
        return {**descriptor, "values": values, "branch_sample_id": row["branch_sample_id"],
                "mask_spans": row["mask_spans"], "source_receipt": proof,
                "dataset_receipt_sha256": self.receipt_sha256}

    def values(self, row):
        return self.record(row)["values"]


def freeze_dataset_stream(stage, *, max_shard_bytes):
    """Seal source vectors into the shared reader's existing typed shard format."""
    from .build import iter_rows, json_line
    from .materialize import ShardWriter
    from .probability import BINDINGS, RAW_CONTRACT
    from core.io import write_json
    stage = Path(stage)
    scoring = None
    writer = None
    count = 0
    binding = hash_json({"branches_sha256": sha256(stage / "branches.jsonl"),
                         "groups_sha256": sha256(stage / "groups.jsonl")})
    manifest_name = "raw-t1-probability.json"
    index_name = "raw-t1-probability-branches.jsonl"
    index = None
    try:
        for row in iter_rows(stage / "branches.jsonl"):
            source_binding = (row.get("probability_streams") or {}).get(STREAM)
            if not source_binding:
                continue
            descriptor = source_binding["descriptor"]
            source = descriptor["source_identity"]
            identity = {"checkpoint_fingerprint": source["model_fingerprint"],
                        "tokenizer_fingerprint": descriptor["tokenizer_fingerprint"],
                        "adapter_fingerprint": source.get("adapter"),
                        "collector_dtype": source["serving_dtype"],
                        "effective_model_overrides": {},
                        "runtime": {"backend": "vllm_v2", "source": source},
                        "implementation": {"collector": descriptor["collector"]}}
            if scoring is None:
                scoring = identity
                writer = ShardWriter(stage, "<f4", max_shard_bytes)
                index = (stage / index_name).open("w", encoding="utf-8")
            elif scoring != identity:
                raise ValueError("Online raw stream contains mixed scoring identities")
            values = read_values(source_binding["root"], descriptor)
            json_line(index, {**{k: row[k] for k in BINDINGS},
                             "mask_spans": row["mask_spans"],
                             "scoring_identity_sha256": hash_json(scoring),
                             "values_ref": writer.put(values),
                             "source_receipt": source_binding["source_receipt"]})
            count += 1
    finally:
        if writer:
            writer.close()
        if index:
            index.close()
    if scoring is None:
        return {}
    write_json(stage / manifest_name, {"schema_version": 2,
               "stream": STREAM, "probability_semantics": "raw_t1_sampled_token_v1",
               "raw_contract": RAW_CONTRACT, "dataset_receipt_sha256": binding,
               "scoring_identity": scoring, "origin": "online_rollout", "branch_count": count})
    return {STREAM: {"manifest": manifest_name, "index": index_name, "dataset_binding": binding}}
