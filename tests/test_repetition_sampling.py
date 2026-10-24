from pathlib import Path

import pytest

from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from rollout.sampling import make_work_items
from tests.conftest import task


def test_repeat_count_freezes_task_repetitions(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("sample-task")])
    run = create_run(cfg, "sample-run", operation="rollout", task_ids=["sample-task"], repeat_count=3)
    loaded = load_plan(cfg["project_root"], run.name)
    assert loaded["plan"]["schema_version"] == 2
    assert loaded["plan"]["repeat_count"] == 3
    assert [item["repetition_id"] for item in loaded["work_items"]["items"]] == ["rep-000", "rep-001", "rep-002"]
    assert [item["sample_seed"] for item in loaded["work_items"]["items"]] == [42, 43, 44]


def test_repeat_count_rejects_non_rollout(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("sample-task")])
    with pytest.raises(ValueError, match="only configurable"):
        create_run(cfg, "analysis-repeat", operation="analysis", task_ids=["sample-task"], repeat_count=2)


def test_sample_seed_uses_parallel_branch_stride(cfg):
    selection = {"task_keys": ["key"], "tasks": {"key": task("sample-task")}}
    items = make_work_items(selection, "run", 3, 42, 4)["items"]
    assert [item["sample_seed"] for item in items] == [42, 46, 50]
    assert len({item["sample_key"] for item in items}) == 3
