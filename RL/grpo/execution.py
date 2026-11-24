"""Frozen 8x16 execution views of fully scored native parent groups (Torch-free).

Parent artifacts and scoring remain authoritative. Only this module selects
branches; neither a loader nor a rank derives a partition from live settings.
"""
import copy
from dataclasses import dataclass
import math

from core.io import hash_json
from .schedule import order_groups

SCHEDULE = "distinct_parent_two_pass_v1"
PARTITION = "sha256_ordinal_permutation_v1"
WEIGHT = "token_mean_branch_mean_actual_window_mean_sum_fp32_v1"
CURSOR = "execution_window_cursor_v1"


def enabled(config):
    return bool(config.grpo and getattr(config.grpo, "parallel_branch_chunk_size", None) is not None)


def parent_record(value):
    rows = value["members"]
    ids = [r["branch_sample_id"] for r in rows]
    if (len(rows) != 32 or len(set(ids)) != 32
            or value["group"]["expected_branch_count"] != 32
            or value["group"].get("branch_ids", ids) != ids):
        raise ValueError("8x16 requires exactly 32 distinct native branches per eligible parent")
    advantages = value["frozen_advantages"]
    scores = [r["historical_score"] for r in rows]
    if len(advantages) != 32 or any(not math.isfinite(v) for v in [*advantages, *scores]):
        raise ValueError("8x16 parent scores/frozen advantages must be finite and complete")
    return {"parent_group_id": value["group"]["group_id"], "branch_ids": ids,
            "prompt_length": rows[0].get("prompt_count", len(rows[0].get("prompt_token_ids", []))),
            "completion_counts": [r.get("completion_count", len(r.get("completion_token_ids", []))) for r in rows],
            "frozen_metadata_sha256": hash_json({"scores": scores, "advantages": advantages,
                "retained": [r.get("retained") for r in rows],
                "diagnostics": value["advantage_diagnostics"]})}


def partition(parent_id, seed):
    permutation = sorted(range(32), key=lambda i: hash_json([PARTITION, seed, parent_id, i]))
    return [sorted(permutation[:16]), sorted(permutation[16:])]


def build(parents, *, partition_seed, shuffle_seed, shuffle_algorithm):
    parents = sorted(copy.deepcopy(list(parents)), key=lambda p: p["parent_group_id"])
    ids = [p["parent_group_id"] for p in parents]
    if len(parents) < 8 or len(set(ids)) != len(ids):
        raise ValueError("8x16 requires at least 8 eligible distinct parent groups")
    all_branches = set()
    for p in parents:
        if (len(p["branch_ids"]) != 32 or len(set(p["branch_ids"])) != 32
                or all_branches.intersection(p["branch_ids"])
                or len(p["completion_counts"]) != 32
                or any(type(n) is not int or n < 1 for n in p["completion_counts"])
                or type(p["prompt_length"]) is not int or p["prompt_length"] < 1):
            raise ValueError("8x16 parent branch identities/native token counts are invalid")
        all_branches.update(p["branch_ids"])
    chunks, windows = [], []
    for round_index in range(2):
        seed = int(hash_json([SCHEDULE, shuffle_seed, round_index])[:16], 16)
        ordered = order_groups(parents, seed=seed, batch_size=8, algorithm=shuffle_algorithm,
                               prompt_length=lambda p: p["prompt_length"])
        for offset in range(0, len(ordered), 8):
            start = len(chunks)
            for p in ordered[offset:offset + 8]:
                ordinals = partition(p["parent_group_id"], partition_seed)[round_index]
                chunks.append({"parent_group_id": p["parent_group_id"], "half_index": round_index,
                    "native_branch_ordinals": ordinals,
                    "branch_ids": [p["branch_ids"][i] for i in ordinals],
                    "token_count": sum(p["completion_counts"][i] for i in ordinals)})
            windows.append({"window_index": len(windows), "round_index": round_index,
                            "start": start, "end": len(chunks)})
    result = {"schema_version": 1, "schedule_algorithm": SCHEDULE,
        "partition_algorithm": PARTITION, "partition_seed": partition_seed,
        "shuffle_seed": shuffle_seed, "shuffle_algorithm": shuffle_algorithm,
        "round_seed_algorithm": "sha256_schedule_seed_round_first64_v1",
        "world_size": 8, "parent_branch_count": 32, "branch_chunk_size": 16,
        "window_parent_count": 8, "loss_weight_protocol": WEIGHT,
        "assignment": "window_ordinal_mod_world_size", "parents": parents,
        "chunks": chunks, "windows": windows}
    result["sha256"] = hash_json(result)
    return result


def from_config(parents, config):
    return build(parents, partition_seed=config.grpo.branch_partition_seed,
                 shuffle_seed=config.grpo.shuffle_seed, shuffle_algorithm=config.grpo.shuffle_algorithm)


def validate(plan):
    expected = build(plan["parents"], partition_seed=plan["partition_seed"],
                     shuffle_seed=plan["shuffle_seed"], shuffle_algorithm=plan["shuffle_algorithm"])
    if (plan != expected
            or plan.get("sha256") != hash_json({k:v for k,v in plan.items() if k != "sha256"})):
        raise ValueError("8x16 execution plan partition/window/hash mismatch")
    return plan


def apply_proof(proof, plan):
    proof.update(execution_plan=plan, execution_schedule_sha256=plan["sha256"],
                 planned_parent_groups=len(plan["parents"]),
                 planned_execution_chunks=len(plan["chunks"]), planned_updates=len(plan["windows"]),
                 group_batch_semantics="distinct_parent_execution_chunks")


def cursor(plan, count):
    ends = [0, *[w["end"] for w in plan["windows"]]]
    if type(count) is not int or count not in ends:
        raise ValueError("8x16 cursor must be a frozen optimizer window boundary")
    consumed = plan["chunks"][:count]
    seen, completed = set(), set()
    halves = {}
    for c in consumed:
        pid = c["parent_group_id"]
        seen.add(pid)
        halves.setdefault(pid, set()).add(c["half_index"])
        if len(halves[pid]) == 2:
            completed.add(pid)
    step = ends.index(count)
    return {"cursor_schema": CURSOR, "execution_schedule_sha256": plan["sha256"],
        "execution_window_cursor": step, "consumed_execution_chunks": count,
        "planned_parent_groups": len(plan["parents"]), "planned_execution_chunks": len(plan["chunks"]),
        "distinct_parent_groups_seen": len(seen), "completed_parent_groups": len(completed),
        # Explicit compatibility projection: only fully committed parents count.
        "group_cursor": len(completed), "consumed_groups": len(completed),
        "consumed_branches": count * 16,
        "consumed_tokens": sum(c["token_count"] for c in consumed),
        "consumed_half_ids_sha256": hash_json([[c["parent_group_id"], c["half_index"]] for c in consumed]),
        "execution_round": plan["windows"][step]["round_index"] if step < len(plan["windows"]) else 2,
        "segment_id": plan["sha256"], "segment_ordinal": count,
        "last_consumed_group_id": next((c["parent_group_id"] for c in reversed(consumed)
                                        if c["parent_group_id"] in completed), None),
        "consumed_group_ids_sha256": hash_json(sorted(completed))}


def validate_state(plan, state):
    expected = cursor(plan, state.get("consumed_execution_chunks"))
    if (type(state.get("global_step")) is not int
            or state["global_step"] != expected["execution_window_cursor"]
            or any(type(state.get(k)) is not type(v) or state.get(k) != v for k, v in expected.items())):
        raise ValueError("8x16 checkpoint execution cursor/counters/coverage mismatch")


def select_raw(parent, chunk, *, partial=False):
    """Select already-frozen branch-aligned data; never score an execution half."""
    ordinals = chunk["native_branch_ordinals"]
    if (parent["group"]["group_id"] != chunk["parent_group_id"]
            or len(ordinals) != 16 or ordinals != sorted(set(ordinals))
            or any(type(i) is not int or not 0 <= i < 32 for i in ordinals)):
        raise ValueError("Invalid 8x16 parent/half branch selection")
    if not partial:
        record = parent_record(parent)
        if [record["branch_ids"][i] for i in ordinals] != chunk["branch_ids"]:
            raise ValueError("8x16 native branch order/identity mismatch")
    result = copy.deepcopy(parent)
    if not partial:
        result["members"] = [result["members"][i] for i in ordinals]
        result["frozen_advantages"] = [result["frozen_advantages"][i] for i in ordinals]
    if [r["branch_sample_id"] for r in result["members"]] != chunk["branch_ids"]:
        raise ValueError("8x16 selected native branches mismatch")
    result["execution_chunk"] = copy.deepcopy(chunk)
    return result


def decorate(group, chunk, window, rank):
    from dataclasses import replace
    return replace(group, metadata={**group.metadata, "execution_chunk": {
        **copy.deepcopy(chunk), "window_index": window["window_index"],
        "round_index": window["round_index"], "rank": rank, "parent_branch_count": 32}})


def validate_window(plan, step, window):
    if type(step) is not int or not 0 <= step < len(plan["windows"]):
        raise ValueError("8x16 execution window is beyond the frozen plan")
    w = plan["windows"][step]
    expected = plan["chunks"][w["start"]:w["end"]]
    if len(window) != len(expected) or len({g.group_id for g in window}) != len(window):
        raise ValueError("8x16 window has missing or duplicate parent contributions")
    for group, chunk in zip(window, expected):
        metadata = group.metadata.get("execution_chunk", {})
        if (group.group_id != chunk["parent_group_id"] or group.group_size != 16
                or group.branch_ids != chunk["branch_ids"]
                or any(metadata.get(k) != v for k, v in chunk.items())
                or metadata.get("window_index") != step or metadata.get("round_index") != w["round_index"]
                or int(group.loss_mask.sum()) != chunk["token_count"]):
            raise ValueError("8x16 parent/half/native branch window identity mismatch")


@dataclass(frozen=True)
class TokenCount:
    count: int

    def sum(self):
        return self.count


def placeholder(parent, chunk, window, rank):
    """Remote-rank metadata: counts and identities without native payload IO."""
    from .groups import GroupBatch
    p = parent["prompt_length"]
    ordinals = chunk["native_branch_ordinals"]
    group = GroupBatch(chunk["parent_group_id"], chunk["branch_ids"],
        [range(p)] * 16, [range(parent["completion_counts"][i]) for i in ordinals],
        None, None, None, TokenCount(chunk["token_count"]), [], [])
    return decorate(group, chunk, window, rank)


class PreparedExecution:
    """Trainer view over the existing immutable indexed parent artifact."""
    def __init__(self, reader, *, rank=0, world_size=1, fsdp_replicate_empty=False):
        self.reader = reader
        self.plan = validate(reader.proof["execution_plan"])
        self.execution_plan = self.plan
        self.rank, self.world_size = rank, world_size
        self.fsdp_replicate_empty = bool(fsdp_replicate_empty)
        if self.fsdp_replicate_empty and world_size not in {2, 4, 8}:
            raise ValueError("FSDP2 8x16 requires world_size=2/4/8")
        self.parents = {p["parent_group_id"]: p for p in self.plan["parents"]}

    def __len__(self):
        return len(self.plan["chunks"])

    def cursor(self, count):
        return cursor(self.plan, count)

    def validate_state(self, state):
        validate_state(self.plan, state)

    def windows(self, start):
        from .data import tensor_groups
        cursor(self.plan, start)
        for window in self.plan["windows"]:
            if window["start"] < start:
                continue
            groups = []
            for index, chunk in enumerate(self.plan["chunks"][window["start"]:window["end"]]):
                assigned_rank = index % self.world_size
                replicate_for_empty = self.fsdp_replicate_empty and self.rank >= window["end"] - window["start"] and index == 0
                if assigned_rank != self.rank and not replicate_for_empty:
                    group = placeholder(self.parents[chunk["parent_group_id"]], chunk, window, assigned_rank)
                else:
                    parent = self.reader.metadata(chunk["parent_group_id"])
                    if parent_record(parent) != self.parents[chunk["parent_group_id"]]:
                        raise ValueError("Prepared parent differs from frozen 8x16 metadata")
                    raw = self.reader.raw(chunk["parent_group_id"], branch_ordinals=chunk["native_branch_ordinals"])
                    group = decorate(next(tensor_groups([select_raw(raw, chunk, partial=True)])), chunk, window, self.rank)
                groups.append(group)
            yield tuple(groups)
