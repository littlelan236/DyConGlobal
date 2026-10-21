import copy

import pytest

from difficulty.probe_config import RidgeProbe
from difficulty.training.split import inherit_task_split, make_split, rollout_roles


def fixture_rows():
    return [{"repo": "a", "instance_id": f"task-{i}", "base_commit": "abc",
             "rollout_id": f"r{i}", "trajectory_sha256": f"sha{i}", "source_kind": "legacy_close"}
            for i in range(6)]


def test_repeated_task_rollouts_inherit_original_partition():
    original = fixture_rows()
    samples = [{"sample_id": r["rollout_id"], "rollout_id": r["rollout_id"]} for r in original]
    parent = make_split(original, samples, validation_tasks=2)
    expanded = original + [{**r, "rollout_id": r["rollout_id"]+"-new",
                             "trajectory_sha256": r["trajectory_sha256"]+"-new"} for r in original]
    new_samples = samples + [{"sample_id": r["rollout_id"], "rollout_id": r["rollout_id"]}
                             for r in expanded[6:]]
    result = inherit_task_split(expanded, new_samples, parent, 2, 42)
    roles = rollout_roles(result)
    assert result["task_counts"] == {"train": 4, "validation": 2}
    assert result["rollout_counts"] == {"train": 8, "validation": 4}
    assert result["sample_counts"] == {"train": 8, "validation": 4}
    assert all(roles[f"r{i}"] == roles[f"r{i}-new"] for i in range(6))
    assert result == inherit_task_split(list(reversed(expanded)), new_samples, parent, 2, 42)
    assert result["tasks"] == parent["tasks"]
    assert rollout_roles(parent) == {r["rollout_id"]: r["split"] for r in parent["tasks"]}


def test_inherited_split_rejects_cross_partition_hash_and_changed_task():
    original = fixture_rows()
    samples = [{"sample_id": r["rollout_id"], "rollout_id": r["rollout_id"]} for r in original]
    parent = make_split(original, samples, validation_tasks=2)
    roles = rollout_roles(parent)
    tr = next(r for r in original if roles[r["rollout_id"]] == "train")
    va = next(r for r in original if roles[r["rollout_id"]] == "validation")
    bad = {**tr, "rollout_id": "cross-partition", "trajectory_sha256": va["trajectory_sha256"]}
    with pytest.raises(ValueError, match="SHA would cross"):
        inherit_task_split(original+[bad], samples, parent, 2, 42)
    changed = copy.deepcopy(original)
    changed[0]["base_commit"] = "changed"
    with pytest.raises(ValueError, match="base_commit"):
        inherit_task_split(changed, samples, parent, 2, 42)
    with pytest.raises(ValueError, match="task set"):
        inherit_task_split(original[:-1], samples[:-1], parent, 2, 42)


def test_inherited_configuration_requires_frozen_parent():
    with pytest.raises(ValueError, match="parent experiment"):
        RidgeProbe(split_strategy="inherit_task_split_v1")
    cfg = RidgeProbe(split_strategy="inherit_task_split_v1", expected_rollouts=328,
                     parent_experiment_id="original", parent_receipt_sha256="a"*64)
    assert cfg.expected_tasks == 238 and cfg.expected_rollouts == 328
