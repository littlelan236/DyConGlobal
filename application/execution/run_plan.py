"""Immutable run plan creation and loading.

The planner freezes task selection and the resolved root configuration once.
Execution code never re-reads the mutable root YAML for an existing run.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from catalog.store import Catalog
from core.config import load_snapshot, snapshot
from core.io import hash_json, identifier, now, read_json, write_json
from rollout.sampling import load_work_items, make_work_items, write_work_items


def _selection_from_tasks(catalog: Catalog, task_ids=(), partition=None):
    selection = catalog.select_tasks(partition=partition, task_ids=task_ids)
    tasks = selection["tasks"]
    return {
        "selection_type": "tasks",
        "catalog_revision": selection["catalog_revision"],
        "catalog_sha256": selection["catalog_sha256"],
        "task_keys": selection["task_keys"],
        "tasks": tasks,
        "rollout_ids": [],
    }


def create_run(cfg, run_id, *, operation, tasks_file=None, task_ids=(), partition=None, repeat_count=1):
    run_id = identifier(run_id)
    if operation not in {"rollout", "harness", "enrichment", "hidden", "analysis"}:
        raise ValueError(f"Unsupported operation: {operation}")
    if type(repeat_count) is not int or repeat_count < 1:
        raise ValueError("repeat_count must be a positive integer")
    if operation != "rollout" and repeat_count != 1:
        raise ValueError("repeat_count is only configurable for rollout runs")
    root = Path(cfg["project_root"])
    catalog = Catalog(root)
    if tasks_file:
        payload = read_json(tasks_file)
        if isinstance(payload, dict) and "tasks" in payload:
            payload = payload["tasks"]
        if not isinstance(payload, list):
            raise ValueError("tasks file must contain a JSON list or {tasks: [...]}")
        requested = [row if isinstance(row, str) else row.get("instance_id") or row.get("instance") for row in payload]
        task_ids = [x if isinstance(x, str) else x["instance_id"] for x in requested]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs")
    selection = _selection_from_tasks(catalog, task_ids=task_ids, partition=partition)
    if not selection["task_keys"]:
        raise ValueError("Run task selection is empty")
    run = snapshot(cfg, run_id, selection)
    frozen = read_json(run / "resolved-config.json")
    frozen.update({"run_id": run_id, "operation": operation})
    base = {k: v for k, v in frozen.items() if k not in {"config_hash", "selection", "selection_sha256"}}
    frozen["config_hash"] = hash_json(base)
    write_json(run / "resolved-config.json", frozen)
    effective_config = yaml.safe_load(Path(cfg["config_source"]).read_text(encoding="utf-8"))
    for override in cfg.get("overrides", []):
        dotted, raw_value = override.split("=", 1)
        target = effective_config
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = yaml.safe_load(raw_value)
    (run / "config.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    tasks = {
        "schema_version": 2,
        "run_id": run_id,
        "operation": operation,
        "selection_sha256": hash_json(selection),
        "tasks": [
            {"task_key": key, "instance_id": selection["tasks"][key]["instance_id"], "task": selection["tasks"][key]}
            for key in selection["task_keys"]
        ],
    }
    tasks["tasks_sha256"] = hash_json(tasks["tasks"])
    write_json(run / "tasks.json", tasks)
    plan = {
        "schema_version": 2,
        "run_id": run_id,
        "operation": operation,
        "migration_state": "active",
        "repeat_count": repeat_count,
        "config_hash": frozen["config_hash"],
        "selection_sha256": hash_json(selection),
        "tasks_sha256": tasks["tasks_sha256"],
        "created_at": now(),
    }
    plan["plan_sha256"] = hash_json(plan)
    write_json(run / "plan.json", plan)
    for folder in (
        "logs", "events", "execution/attempts", "rollout/tasks", "rollout/trajectories",
        "rollout/patches", "enrichment", "hidden/sessions", "hidden/index",
        "hidden/branch", "harness", "reports",
    ):
        (run / folder).mkdir(parents=True, exist_ok=True)
    branch_count = cfg.get("parallel", {}).get("branch_count", 1) if operation == "rollout" else 1
    write_work_items(run, make_work_items(selection, run_id, repeat_count, cfg["generation"]["seed"], branch_count))
    if operation == "rollout":
        from rollout.retry import initialize_counts
        initialize_counts(run, read_json(run / "rollout/work-items.json"))
    write_json(run / "status.json", {"stage": "planned", "run_id": run_id, "operation": operation})
    if operation == "rollout":
        from rollout.progress import Progress
        Progress(run, frozen, read_json(run / "rollout/work-items.json")["items"])._write()
    else:
        write_json(run / "progress.json", {})
    write_json(run / "artifacts.json", {"schema_version": 1, "run_id": run_id, "artifacts": []})
    write_json(run / "code-identity.json", {"schema_version": 1, "git": _git_identity(root)})
    write_json(run / "events" / "000001.json", {
        "event": "run.planned", "run_id": run_id, "operation": operation,
        "config_hash": frozen["config_hash"], "selection_sha256": hash_json(selection),
    })
    return run


def _git_identity(root):
    import subprocess

    git = "/home/wangtianyi/.local/bin/git"
    try:
        head = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output([git, "-C", str(root), "status", "--porcelain"], text=True)
        return {"head": head, "dirty": bool(dirty), "dirty_sha256": hash_json(dirty)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"unavailable": str(exc)}


def load_plan(root, run_id):
    root = Path(root).resolve()
    run_id = identifier(run_id)
    run = root / "runs" / run_id
    plan = read_json(run / "plan.json")
    if plan.get("schema_version") != 2 or plan.get("migration_state", "active") != "active":
        raise ValueError(f"Run is not an active schema-2 run: {run_id}")
    expected = plan.pop("plan_sha256", None)
    if expected != hash_json(plan):
        raise ValueError(f"Run plan was modified: {run_id}")
    plan["plan_sha256"] = expected
    cfg = load_snapshot(root, run_id)
    if cfg.get("run_id") != run_id or cfg.get("operation") != plan["operation"]:
        raise ValueError("Run plan and resolved configuration disagree")
    if cfg["config_hash"] != plan["config_hash"] or cfg["selection_sha256"] != plan["selection_sha256"]:
        raise ValueError("Run plan and snapshot hashes disagree")
    if read_json(run / "selection.json") != cfg["selection"]:
        raise ValueError("Run selection sidecar changed")
    tasks = read_json(run / "tasks.json")
    if tasks["tasks_sha256"] != hash_json(tasks["tasks"]):
        raise ValueError("Run tasks were modified")
    if tasks["tasks_sha256"] != plan["tasks_sha256"]:
        raise ValueError("Run tasks and plan hashes disagree")
    expected_tasks = [{"task_key": key, "instance_id": cfg["selection"]["tasks"][key]["instance_id"],
                       "task": cfg["selection"]["tasks"][key]} for key in cfg["selection"]["task_keys"]]
    if tasks["tasks"] != expected_tasks:
        raise ValueError("Run tasks and frozen selection disagree")
    if tasks["selection_sha256"] != plan["selection_sha256"]:
        raise ValueError("Run tasks do not match selection")
    work_items = load_work_items(root, run_id, selection=cfg["selection"])
    if work_items.get("items_sha256") != hash_json(work_items["items"]):
        raise ValueError("Run work items were modified")
    if work_items.get("repeat_count") != plan.get("repeat_count"):
        raise ValueError("Run repeat count disagrees with work items")
    expected = make_work_items(
        cfg["selection"], run_id, plan["repeat_count"], cfg["generation"]["seed"],
        cfg.get("parallel", {}).get("branch_count", 1),
    )
    if expected["items"] != work_items["items"]:
        raise ValueError("Run work items disagree with frozen selection/config")
    from rollout.retry import enabled
    if enabled(cfg) and expected["seed_policy"] != work_items.get("seed_policy"):
        raise ValueError("Run seed policy disagrees with frozen configuration")
    return {"run": run, "plan": plan, "config": cfg, "tasks": tasks, "work_items": work_items}


def show_run(root, run_id):
    loaded = load_plan(root, run_id)
    return {k: loaded[k] for k in ("plan", "tasks", "work_items")}


def list_runs(root):
    root = Path(root).resolve()
    result = []
    for run in sorted((root / "runs").iterdir()):
        if not run.is_dir() or not (run / "plan.json").exists():
            continue
        try:
            loaded = load_plan(root, run.name)
            result.append({"run_id": run.name, **loaded["plan"]})
        except Exception as exc:  # status listing must expose corrupt plans
            result.append({"run_id": run.name, "error": str(exc)})
    return result
