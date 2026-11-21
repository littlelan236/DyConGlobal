
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
    assert read_json(run / "resolved-config.json")["execution"]["supervisor_max_open_files"] == 262144
    assert read_json(run / "status-summary.json")["sample_count"] == 1


def test_tampered_plan_is_rejected(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("tamper-task")])
    run = create_run(cfg, "run-plan-tamper", operation="analysis", task_ids=["tamper-task"])
    assert read_json(run / "status-summary.json")["operation"] == "analysis"
    plan = read_json(run / "plan.json")
    plan["operation"] = "rollout"
    write_json(run / "plan.json", plan)
    with pytest.raises(ValueError, match="plan was modified"):
        load_plan(cfg["project_root"], "run-plan-tamper")


def test_create_run_freezes_online_source_profiles(cfg):
    from RL.online_runtime import make_online_run_work_items
    Catalog(cfg["project_root"]).register_tasks([task("online-profile-task")])
    selection = Catalog(cfg["project_root"]).select_tasks(task_ids=["online-profile-task"],
                                                        benchmark=cfg["benchmark"]["backend"],
                                                        revision=None)
    manifest = make_online_run_work_items(selection, "online-profile-run", m=1,
                    base_seed=cfg["generation"]["seed"], branch_count=1)
    run = create_run(cfg, "online-profile-run", operation="rollout",
                     task_ids=["online-profile-task"], repeat_count=2,
                     online_work_items=manifest)
    items = read_json(run / "rollout/work-items.json")["items"]
    assert [item["online_source"] for item in items] == ["dycon", "baseline"]
    assert read_json(run / "rollout/work-items.json")["items_sha256"]
    assert load_plan(cfg["project_root"], "online-profile-run")["plan"]["online_contract"]
    changed = read_json(run / "rollout/work-items.json")
    changed["items"][0]["online_source"] = "baseline"
    from core.io import hash_json
    changed["items_sha256"] = hash_json(changed["items"])
    write_json(run / "rollout/work-items.json", changed)
    with pytest.raises(ValueError, match="Online profile contract"):
        load_plan(cfg["project_root"], "online-profile-run")


def test_invalid_online_profile_leaves_no_snapshot(cfg):
    from RL.online_runtime import make_online_run_work_items
    from pathlib import Path
    Catalog(cfg["project_root"]).register_tasks([task("online-bad-task")])
    selection = Catalog(cfg["project_root"]).select_tasks(task_ids=["online-bad-task"],
                                                        benchmark=cfg["benchmark"]["backend"], revision=None)
    manifest = make_online_run_work_items(selection, "online-bad-run", m=1,
                    base_seed=cfg["generation"]["seed"], branch_count=1)
    manifest["items"][0]["sample_seed"] += 1
    with pytest.raises(ValueError, match="identities/profiles"):
        create_run(cfg, "online-bad-run", operation="rollout", task_ids=["online-bad-task"],
                   repeat_count=2, online_work_items=manifest)
    assert not (Path(cfg["project_root"]) / "runs/online-bad-run").exists()
