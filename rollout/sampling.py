"""Immutable sample identities for repetition-aware rollout runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.io import hash_json, identifier, read_json, write_json

WORK_ITEMS_SCHEMA = 1
SEED_POLICY = "repetition_offset_v1"


@dataclass(frozen=True)
class SampleContext:
    run_id: str
    task_key: str
    instance_id: str
    repetition_id: str
    repetition_index: int
    sample_key: str
    sample_seed: int
    seed_policy: str = SEED_POLICY

    @property
    def attempt_prefix(self) -> str:
        return f"{self.repetition_id}-attempt"

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task_key": self.task_key,
            "instance_id": self.instance_id,
            "repetition_id": self.repetition_id,
            "repetition_index": self.repetition_index,
            "sample_key": self.sample_key,
            "sample_seed": self.sample_seed,
            "seed_policy": self.seed_policy,
        }


def make_work_items(selection: dict, run_id: str, repeat_count: int, base_seed: int, branch_count: int = 1):
    """Expand frozen tasks into deterministic task x repetition items."""
    identifier(run_id)
    if type(repeat_count) is not int or repeat_count < 1:
        raise ValueError("repeat_count must be a positive integer")
    if type(base_seed) is not int:
        raise ValueError("base seed must be an integer")
    stride = branch_count if type(branch_count) is int and branch_count > 1 else 1
    items = []
    for index in range(repeat_count):
        for task_key in selection["task_keys"]:
            task = selection["tasks"][task_key]
            repetition_id = f"rep-{index:03d}"
            sample_key = "s-" + hash_json({
                "version": 1,
                "run_id": run_id,
                "task_key": task_key,
                "repetition_index": index,
            })[:32]
            sample = SampleContext(
                run_id=run_id,
                task_key=task_key,
                instance_id=task["instance_id"],
                repetition_id=repetition_id,
                repetition_index=index,
                sample_key=sample_key,
                sample_seed=base_seed + index * stride,
            )
            items.append(sample.as_dict())
    return {
        "schema_version": WORK_ITEMS_SCHEMA,
        "run_id": run_id,
        "repeat_count": repeat_count,
        "seed_policy": {
            "name": SEED_POLICY,
            "base_seed": base_seed,
            "branch_stride": stride,
        },
        "selection_sha256": hash_json(selection),
        "items": items,
    }


def load_work_items(root, run_id, *, selection=None):
    path = Path(root) / "runs" / identifier(run_id) / "rollout" / "work-items.json"
    value = read_json(path)
    if value.get("schema_version") != WORK_ITEMS_SCHEMA or value.get("run_id") != run_id:
        raise ValueError(f"Unsupported work-item manifest: {path}")
    if selection is not None and value.get("selection_sha256") != hash_json(selection):
        raise ValueError("Work-item selection hash mismatch")
    required = {"sample_key", "task_key", "instance_id", "repetition_id", "repetition_index", "sample_seed"}
    seen = set()
    for item in value.get("items", []):
        if not required <= item.keys():
            raise ValueError("Incomplete work item")
        key = item["sample_key"]
        if key in seen:
            raise ValueError("Duplicate sample key")
        seen.add(key)
    return value


def write_work_items(run, value):
    value = dict(value)
    value["items_sha256"] = hash_json(value["items"])
    write_json(Path(run) / "rollout" / "work-items.json", value)
    return value


def sample_from_item(item: dict) -> SampleContext:
    return SampleContext(
        run_id=item["run_id"],
        task_key=item["task_key"],
        instance_id=item["instance_id"],
        repetition_id=item["repetition_id"],
        repetition_index=item["repetition_index"],
        sample_key=item["sample_key"],
        sample_seed=item["sample_seed"],
        seed_policy=item.get("seed_policy", SEED_POLICY),
    )


@dataclass(frozen=True)
class AttemptContext:
    run_id: str
    task_key: str
    sample_key: str
    repetition_id: str
    attempt_id: str
    seed_round: int
    effective_seed: int
    sample_seed: int
    seed_policy: str

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


def effective_seed(sample, manifest, seed_round, actual_branches=1):
    """Match the installed vLLM OpenAI chat API signed-int64 seed contract."""
    if type(seed_round) is not int or seed_round < 0:
        raise ValueError('Invalid seed round')
    stride = manifest['seed_policy']['branch_stride']
    if not 1 <= actual_branches <= stride:
        raise ValueError('Frozen branch stride cannot cover active branches')
    seed = sample.sample_seed + seed_round * manifest['repeat_count'] * stride
    if not -(2**63) <= seed <= seed + actual_branches - 1 <= 2**63 - 1:
        raise ValueError('seed_space_exhausted: vLLM seed must fit signed int64')
    return seed


def execution_seed(value):
    """Actual execution seed, with an explicit legacy fallback."""
    if isinstance(value, dict):
        return value.get('effective_seed', value.get('sample_seed'))
    return getattr(value, 'effective_seed', value.sample_seed)
