import json

import pytest

from core.io import sha256, write_json
from difficulty.datasets.compose import compose_datasets
from difficulty.training.splits import create_split


def _skills_dataset(root, dataset_id, prefix, count=5):
    directory = root / "difficulty/datasets" / dataset_id
    directory.mkdir(parents=True)
    rollouts, samples = [], []
    revision = "f" * 40
    for index in range(count):
        instance_id = f"{prefix}-task-{index}"
        rid = f"rollout-{prefix}-{index}"
        trajectory = (prefix + str(index)).encode().hex().ljust(64, "0")
        rollouts.append({
            "rollout_id": rid, "benchmark": "skillsbench", "revision": revision,
            "instance_id": instance_id, "task_package_sha256": (prefix + str(index)).encode().hex().ljust(64, "0"),
            "agent": "openhands", "source_kind": "online_close", "trajectory_sha256": trajectory,
        })
        samples.append({"sample_id": f"sample-{prefix}-{index}", "rollout_id": rid,
                        "benchmark": "skillsbench", "revision": revision,
                        "instance_id": instance_id,
                        "task_group": f"skillsbench@{revision}::{instance_id}",
                        "source_kind": "online_close"})
    write_json(directory / "rollouts.json", {"rollouts": rollouts})
    (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
    write_json(directory / "source-evidence.json", {"source_run_id": f"source-{prefix}"})
    write_json(directory / "summary.json", {"samples": len(samples)})
    write_json(directory / "dataset.json", {"benchmark": "skillsbench", "agent": "openhands"})
    write_json(directory / "receipt.json", {
        "state": "complete", "eligibility": "resolved_only", "source_run_id": f"source-{prefix}",
        "files": {name: sha256(directory / name) for name in
                  ["rollouts.json", "samples.jsonl", "source-evidence.json", "summary.json", "dataset.json"]},
    })


def test_skillsbench_compose_and_task_split_use_package_identity(tmp_path):
    _skills_dataset(tmp_path, "skills-a", "a")
    _skills_dataset(tmp_path, "skills-b", "b")
    result = compose_datasets(tmp_path, "skills-combined", ["skills-a", "skills-b"])
    assert result["samples"] == 10
    dataset = tmp_path / "difficulty/datasets/skills-combined"
    metadata = json.loads((dataset / "dataset.json").read_text())
    assert metadata["benchmark"] == "skillsbench"
    split = create_split(tmp_path, "skills-combined", "task-split",
                         strategy="task_random_v1", validation_fraction=.2, test_fraction=.2)
    assert split["task_counts"] == {"train": 6, "validation": 2, "test": 2}


def test_skillsbench_rejects_repo_quota_split(tmp_path):
    _skills_dataset(tmp_path, "skills-a", "a")
    with pytest.raises(ValueError, match="not defined for SkillsBench"):
        create_split(tmp_path, "skills-a", "repo-split", strategy="repo_source_quota_v1")
