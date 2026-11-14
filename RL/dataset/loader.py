"""Stream groups in frozen order, preserving full prefixes and branch multiplicity."""
from pathlib import Path

from core.io import read_json, sha256, owned_path
from hidden.artifacts import read_request
from .artifacts import verify_local
from .build import iter_rows
from .schema import spans, validate


def native_tokens(row, *, audit=True, verify_hashes=None):
    evidence = row.get("native_evidence")
    if not evidence:
        raise ValueError("Branch has no exact native token evidence")
    if audit:
        value = read_request(evidence["root"], evidence)
    else:
        value = read_json(owned_path(evidence["root"], evidence["ledger_ref"]))
    for key, expected in (("request_id", row["native_request_id"]), ("session_id", row["session_id"]),
                          ("decision_id", row["decision_id"]), ("branch_id", row["branch_id"]),
                          ("step_id", row["agent_step"]), ("message_index", row["message_index"]),
                          ("tokenizer_fingerprint", row["tokenizer_fingerprint"])):
        if audit and value.get(key) != expected:
            raise ValueError("Frozen native payload identity mismatch: " + key)
    p, c = value.get("prompt_token_ids"), value.get("output_token_ids")
    validate(p, c, row["prompt_sha256"], row["completion_sha256"], verify_hashes=audit if verify_hashes is None else verify_hashes)
    if (len(p), len(c)) != (row["prompt_count"], row["completion_count"]):
        raise ValueError("Frozen native payload lengths mismatch")
    if row["mask_spans"] != spans(len(p), len(c)):
        raise ValueError("Original target-coordinate spans mismatch")
    return p, c


def iter_groups(path, *, with_tokens=True, data_ready_only=True, max_sequence_length=None,
                cache=None, replay=None, allow_replay_estimates=False, start_group=0,
                verify_identity=True):
    path = Path(path)
    verify_local(path, "rl-dataset", verify_files=verify_identity)
    cached = None
    if cache is not None:
        from .materialize import CacheReader
        cached = CacheReader(cache, sha256(path / "receipt.json"))
    probabilities = None
    audit = read_json(path / "dataset.json").get("audit", {}).get("enabled", False)
    if replay is not None:
        from .replay.artifacts import ReplayReader
        probabilities = ReplayReader(replay, path, allow_estimates=allow_replay_estimates)
    source = iter(iter_rows(path / "branches.jsonl"))
    for ordinal, group in enumerate(iter_rows(path / "groups.jsonl")):
        members = [next(source, None) for _ in range(group["expected_branch_count"])]
        if any(r is None or r["group_id"] != group["group_id"] for r in members):
            raise ValueError("Missing or misordered group members")
        if [r["branch_sample_id"] for r in members] != group["branch_ids"]:
            raise ValueError("Member order differs from frozen group")
        if ordinal < start_group or (data_ready_only and not group["data_ready"]):
            continue
        if max_sequence_length is not None and any(r["prompt_count"] + r["completion_count"] > max_sequence_length for r in members):
            raise ValueError("Group exceeds explicit sequence limit; no truncation is allowed")
        prompt = None
        if with_tokens:
            for row in members:
                p, c = cached.tokens(row) if cached else native_tokens(row, audit=audit)
                if audit and prompt is not None and p != prompt:
                    raise ValueError("Frozen group actual prompts differ")
                if prompt is None:
                    prompt = p
                row.update(prompt_token_ids=p, completion_token_ids=c)
                if probabilities:
                    row["behavior_logprobs"] = probabilities.values(row)
        yield {"group": group, "members": members, "cursor": {"group_ordinal": ordinal, "next_group": ordinal + 1}}
    if next(source, None) is not None:
        raise ValueError("Dataset contains unbound branch rows")


def collate_group(group, *, as_tensors=True, pad_token_id=0):
    members = group["members"]
    if not members:
        raise ValueError("Cannot collate an empty group")
    sequences = []
    for row in members:
        p, c = row["prompt_token_ids"], row["completion_token_ids"]
        validate(p, c, row["prompt_sha256"], row["completion_sha256"], verify_hashes=row.get("audit_enabled", True))
        sequences.append(p + c)
    length = max(map(len, sequences))
    output = {name: [] for name in ("input_ids", "attention_mask", "labels", "loss_mask")}
    for row, sequence in zip(members, sequences):
        p, c = row["prompt_token_ids"], row["completion_token_ids"]
        padding = length - len(sequence)
        output["input_ids"].append(sequence + [pad_token_id] * padding)
        output["attention_mask"].append([1] * len(sequence) + [0] * padding)
        output["labels"].append([-100] * len(p) + c + [-100] * padding)
        output["loss_mask"].append([False] * len(p) + [True] * len(c) + [False] * padding)
    if as_tensors:
        import torch
        output = {k: torch.tensor(v, dtype=torch.bool if k == "loss_mask" else torch.int64) for k,v in output.items()}
    return output


def iter_ce_batches(path, **kwargs):
    """Engineering CE/mask adapter; the existing trainer is not a GRPO objective."""
    for group in iter_groups(path, **kwargs):
        for row in group["members"]:
            yield collate_group({"members": [row]})


def iter_grpo_groups(path, *, top_k=4, top_k_fraction=None, score_scale_floor=0.1,
                     boundary_tie_sigma=0.05, advantage_policy="historical_topk_zero_sum_v1", **kwargs):
    """Yield objective-ready frozen groups without changing native coordinates."""
    from RL.grpo.groups import GroupBatch

    for value in iter_groups(path, **kwargs):
        yield GroupBatch.from_dataset_group(
            value, top_k=top_k, top_k_fraction=top_k_fraction,
            score_scale_floor=score_scale_floor, boundary_tie_sigma=boundary_tie_sigma,
            advantage_policy=advantage_policy)


def iter_grpo_batches(path, group_batch_size=4, **kwargs):
    from RL.grpo.groups import iter_group_batches

    yield from iter_group_batches(iter_grpo_groups(path, **kwargs), group_batch_size)
