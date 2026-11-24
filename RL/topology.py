"""Physical GPU remapping of immutable GRPO optimizer windows.

Portable execution deliberately accepts only RNG-free training computations.
Rank RNG remains checkpointed for bookkeeping; it is not presented as an
unchanged stochastic trajectory when the number of physical workers changes.
"""
from dataclasses import replace
import copy

POLICY = "preserve_updates"
PROTOCOL = "logical_window_remap_v1"
TASK_BRANCHES = 16


def enabled(config):
    return config.runtime.topology_policy == POLICY


def execution_record(config):
    from RL.parallel import devices
    return {"version": PROTOCOL, "policy": POLICY,
            "world_size": len(devices(config)), "gpu_ids": list(devices(config)),
            "whole_parent_task_branches": TASK_BRANCHES,
            "rng_policy": "reject_training_rng_consumption_v1"}


def validate_record(record):
    if (not isinstance(record, dict) or set(record) != {
            "version", "policy", "world_size", "gpu_ids", "whole_parent_task_branches", "rng_policy"}
            or record["version"] != PROTOCOL or record["policy"] != POLICY
            or type(record["world_size"]) is not int or record["world_size"] not in {2, 4, 8}
            or not isinstance(record["gpu_ids"], list)
            or len(record["gpu_ids"]) != record["world_size"]
            or any(type(i) is not int or i < 0 for i in record["gpu_ids"])
            or len(set(record["gpu_ids"])) != record["world_size"]
            or record["whole_parent_task_branches"] != TASK_BRANCHES
            or record["rng_policy"] != "reject_training_rng_consumption_v1"):
        raise ValueError("Invalid portable physical execution record")
    return record


def source_world_size(metadata):
    record = metadata.get("physical_execution")
    if record is not None:
        return validate_record(record)["world_size"]
    return (metadata["contract"].get("parallel") or {}).get("world_size", 1)


def validate_resume(config, metadata):
    """Torch-free admission; every nonphysical training pin is checked elsewhere."""
    from RL.parallel import execution_contract, devices
    contract = metadata["contract"]
    target = execution_contract(config)
    saved = contract.get("parallel")
    if not enabled(config):
        if metadata.get("physical_execution") is not None:
            raise ValueError("Portable checkpoints require runtime.topology_policy=preserve_updates")
        if saved != target:
            raise ValueError("Resume requires the same data-parallel world size and execution contract")
        return
    if config.run.start.mode != "resume":
        raise ValueError("Portable checkpoint migration supports resume only")
    previous = source_world_size(metadata)
    if previous not in {1, 2, 4, 8} or len(devices(config)) not in {2, 4, 8}:
        raise ValueError("Portable resume supports 1/2/4/8 source and 2/4/8 target workers")
    if "streaming" not in contract and "prepared" not in contract:
        raise ValueError("Portable resume requires streaming or indexed prepared input")
    if saved is None:
        if previous != 1 and metadata.get("physical_execution") is None:
            raise ValueError("Missing original single-worker logical contract")
        if target.get("version") != "whole_group_sum_v1" or contract["input"].get("execution_plan"):
            raise ValueError("Single-worker migration requires a whole-parent logical plan")
        return
    if not isinstance(saved, dict) or not isinstance(target, dict):
        raise ValueError("Portable resume requires a distributed GRPO contract")
    # Keep the historical logical protocol, including two-pass vs whole-parent.
    if ({k: v for k, v in saved.items() if k != "world_size"}
            != {k: v for k, v in target.items() if k != "world_size"}):
        raise ValueError("Portable resume cannot change the logical execution protocol")


def bind(config, metadata, contract, directory):
    """Keep the source logical contract byte-for-byte; record physical lineage."""
    if not enabled(config):
        return
    from pathlib import Path
    from core.io import hash_json, sha256, write_json
    if metadata is not None:
        validate_resume(config, metadata)
        if "parallel" in metadata["contract"]:
            contract["parallel"] = copy.deepcopy(metadata["contract"]["parallel"])
        else:
            contract.pop("parallel", None)
    record = execution_record(config)
    validate_record(record)
    write_json(Path(directory) / "topology-transition.json", {
        "schema_version": 1, "physical_execution": record,
        "logical_contract_sha256": hash_json(contract),
        "source_checkpoint": config.run.start.checkpoint,
        "source_receipt_sha256": sha256(Path(config.run.start.checkpoint)/"receipt.json") if metadata else None,
        "source_world_size": source_world_size(metadata) if metadata else None,
        "source_contract_sha256": hash_json(metadata["contract"]) if metadata else None,
        "source_physical_execution": metadata.get("physical_execution") if metadata else None,
        "guarantee": "same_logical_windows_weights_and_optimizer_state_not_bitwise_identity"})


def audit_model(model):
    import torch
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
            raise ValueError("Portable execution rejects nonzero dropout: " + name)
        for field in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob",
                      "attention_probs_dropout_prob"):
            value = getattr(module, field, 0)
            if isinstance(value, (float, int)) and value != 0:
                raise ValueError("Portable execution rejects stochastic model setting: " + name + "." + field)


def configure(config, model, parallel):
    if enabled(config) and parallel is None:
        raise ValueError("Portable training requires the formal distributed execution context")
    if parallel is not None:
        parallel.topology_policy = config.runtime.topology_policy
        parallel.physical_execution = execution_record(config) if enabled(config) else None
    if enabled(config):
        audit_model(model)


def portable(parallel):
    return parallel is not None and getattr(parallel, "topology_policy", "exact") == POLICY


def assert_rng_unchanged(before):
    import torch
    from RL.training import capture_rng
    after = capture_rng()
    if (before["python"] != after["python"] or before["numpy"] != after["numpy"]
            or not torch.equal(before["torch_cpu"], after["torch_cpu"])
            or len(before["torch_cuda"]) != len(after["torch_cuda"])
            or any(not torch.equal(a, b) for a, b in zip(before["torch_cuda"], after["torch_cuda"]))):
        raise ValueError("Portable execution detected training RNG consumption; optimizer update rejected")


def tasks(window, *, rank, world_size):
    """A physical task view; parent windows and their cursors stay untouched."""
    ordinal = 0
    for index, parent in enumerate(window):
        for start in range(0, parent.group_size, TASK_BRANCHES):
            end = min(start + TASK_BRANCHES, parent.group_size)
            if ordinal % world_size == rank:
                indices = list(range(start, end))
                values = {key: [getattr(parent, key)[i] for i in indices] for key in (
                    "branch_ids", "prompt_token_ids", "completion_token_ids", "historical_scores", "advantages")}
                values.update({key: getattr(parent, key)[start:end] if getattr(parent, key) is not None else None
                               for key in ("input_ids", "attention_mask", "labels", "loss_mask")})
                values["metadata"] = {**parent.metadata, "physical_task": {
                    "task_ordinal": ordinal, "native_branch_ordinals": indices,
                    "parent_branch_count": parent.group_size}}
                yield index, replace(parent, **values), (end-start)/parent.group_size
            ordinal += 1


def owns_parent(index, branch_counts, rank, world_size):
    first = sum((n + TASK_BRANCHES - 1)//TASK_BRANCHES for n in branch_counts[:index])
    count = (branch_counts[index]+TASK_BRANCHES-1)//TASK_BRANCHES
    return any(i % world_size == rank for i in range(first, first+count))


def merge_rows(window, peers):
    """Reconstruct original parent means and prove exactly-once branch coverage."""
    merged = [[] for _ in window]
    for peer in peers:
        rank = peer["rank"]
        for index, rows in peer["rows"]:
            if type(index) is not int or not 0 <= index < len(window):
                raise ValueError("Portable window has invalid parent index")
            parent = window[index]
            first = sum((g.group_size+TASK_BRANCHES-1)//TASK_BRANCHES for g in window[:index])
            for row in rows:
                ordinal = row.get("native_branch_ordinal")
                if (type(ordinal) is not int or not 0 <= ordinal < parent.group_size
                        or row.get("branch_id") != parent.branch_ids[ordinal]
                        or (first+ordinal//TASK_BRANCHES) % len(peers) != rank):
                    raise ValueError("Portable window branch identity/assignment mismatch")
            merged[index].extend(rows)
    for parent, rows in zip(window, merged):
        rows.sort(key=lambda row: row["native_branch_ordinal"])
        if [row["native_branch_ordinal"] for row in rows] != list(range(parent.group_size)):
            raise ValueError("Portable window has missing or duplicate branch contributions")
    return merged


def restore_rank_rng(payload, metadata, parallel):
    """Only RNG-free forward/backward is admissible for cross-topology recovery."""
    from RL.training import restore_rng
    states = payload.get("rank_rng")
    if source_world_size(metadata) == 1 and states is None:
        states = [payload["rng"]]
    if not states or len(states) != source_world_size(metadata):
        raise ValueError("Checkpoint rank RNG topology mismatch")
    record = validate_record(parallel.physical_execution)
    if record["world_size"] != parallel.world_size:
        raise ValueError("Portable target physical topology mismatch")
    # Preserve original rank states where possible. Extra physical workers reuse
    # bookkeeping state; every training task must prove it did not advance RNG.
    restore_rng(states[parallel.rank % len(states)])
