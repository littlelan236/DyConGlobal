from pathlib import Path

import pytest

from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.io import read_json, write_json
from tests.conftest import task


def test_create_run_freezes_tasks_and_metadata(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("run-task")])
    run = create_run(cfg, "run-plan-test", operation="rollout", task_ids=["run-task"])
    assert (run / "plan.json").exists()
    assert (run / "tasks.json").exists()
    assert (run / "code-identity.json").exists()
    assert read_json(run / "tasks.json")["tasks"][0]["instance_id"] == "run-task"
    assert load_plan(cfg["project_root"], "run-plan-test")["plan"]["operation"] == "rollout"


def test_tampered_plan_is_rejected(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("tamper-task")])
    run = create_run(cfg, "run-plan-tamper", operation="analysis", task_ids=["tamper-task"])
    plan = read_json(run / "plan.json")
    plan["operation"] = "rollout"
    write_json(run / "plan.json", plan)
    with pytest.raises(ValueError, match="plan was modified"):
        load_plan(cfg["project_root"], "run-plan-tamper")
