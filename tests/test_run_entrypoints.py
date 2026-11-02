import json
from pathlib import Path

import pytest

from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.io import hash_json, read_json, write_json
from tests.conftest import task


def make_run(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("entry-task")])
    return create_run(cfg,"entry-test",operation="rollout",task_ids=["entry-task"])


def test_run_status_uses_snapshot_even_when_root_yaml_breaks(cfg,capsys):
    from core.cli import main
    run=make_run(cfg)
    Path(cfg["config_source"]).write_text("invalid: [")
    main(["--config",cfg["config_source"],"run","status",run.name])
    assert json.loads(capsys.readouterr().out)["task_count"]==1


def test_run_status_reports_sample_summary_before_other_details(cfg,capsys):
    from core.cli import main
    run=make_run(cfg)
    (run/"execution").mkdir(parents=True,exist_ok=True)
    write_json(run/"execution/latest.json",{"attempt":"001"})
    write_json(run/"progress.json",{"samples":{
        "s0":{"disposition":"accepted","exit_status":"Submitted","serving_assignment":{"execution_attempt_id":"000"}},
        "s1":{"disposition":"accepted","exit_status":"ContextWindowExceededError","serving_assignment":{"execution_attempt_id":"001"}},
        "s2":{"disposition":"retry_pending","exit_status":"LimitsExceeded","serving_assignment":{"execution_attempt_id":"001"}},
        "s3":{"disposition":"pending","exit_status":"Running"},
    }})

    main(["--config",cfg["config_source"],"run","status",run.name])
    output=capsys.readouterr().out
    result=json.loads(output)
    assert list(result)[:3]==["run_id","sample_summary","plan"]
    assert result["sample_summary"]=={
        "sample_count":4,
        "accepted_count":2,
        "pending_count":1,
        "retry_pending_count":1,
        "failure_counts":{"ContextWindowExceededError":1,"LimitsExceeded":1},
        "latest_attempt":{
            "execution_attempt_id":"001",
            "sample_count":2,
            "accepted_count":1,
            "pending_count":0,
            "retry_pending_count":1,
            "failure_counts":{"ContextWindowExceededError":1,"LimitsExceeded":1},
        },
    }


def test_rehashed_task_sidecar_cannot_bypass_plan(cfg):
    run=make_run(cfg)
    data=read_json(run/"tasks.json")
    data["tasks"][0]["instance_id"]="other"
    data["tasks_sha256"]=hash_json(data["tasks"])
    write_json(run/"tasks.json",data)
    with pytest.raises(ValueError,match="tasks and plan"):
        load_plan(cfg["project_root"],run.name)


def test_supervisor_returns_failure_and_event_wait_receipt(cfg):
    from application.execution.supervisor import launch, wait
    run=make_run(cfg)
    # Fixture has no checkpoint: official startup must fail before GPU/container work.
    result=launch(cfg["project_root"],run.name)
    assert result["exit_code"]!=0
    assert wait(cfg["project_root"],run.name)==result
    assert (run/"execution/attempts/000/exit.json").exists()


def test_replay_group_sorts_zero_integer_and_missing_branches():
    from hidden.adapters.vllm import replay_group
    rows=[{"rollout_id":"r","session_id":"s","branch_id":i} for i in [1,None,0,2]]
    keys=[replay_group(r) for r in rows]
    assert len(set(keys))==4
    sorted(keys)


def test_installed_cli_imports_catalog_from_outside_project(cfg):
    import subprocess,sys
    import catalog
    p=subprocess.run([sys.executable,"-m","core.cli","--help"],cwd="/tmp",text=True,capture_output=True)
    assert p.returncode==0,p.stderr


def test_task_file_accepts_string_ids(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("entry-task")])
    path=Path(cfg["project_root"])/"ids.json"
    write_json(path,["entry-task"])
    run=create_run(cfg,"file-selection",operation="analysis",tasks_file=path)
    assert read_json(run/"tasks.json")["tasks"][0]["instance_id"]=="entry-task"
