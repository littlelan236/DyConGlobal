"""Immutable run plan creation and loading.



The planner freezes task selection and the resolved root configuration once.

Execution code never re-reads the mutable root YAML for an existing run.

"""



from __future__ import annotations

import copy
from pathlib import Path

import yaml

from catalog.store import Catalog
from core.config import load_snapshot, snapshot
from core.io import artifact_roots, hash_json, identifier, now, read_json, run_path, sha256, write_json
from rollout.sampling import load_work_items, make_work_items, write_work_items


def _selection_from_tasks(catalog: Catalog, task_ids=(), partition=None, cfg=None):

    from core.backends import identity
    benchmark = identity(cfg or {})[0]
    revision = None
    if benchmark in {"skillsbench", "webarena_verified", "swebench_pro_v2", "terminalbench_2_1"}:
        ref = (cfg or {}).get("benchmark", {}).get("snapshot_ref")
        if not ref:
            raise ValueError(f"{benchmark} requires benchmark.snapshot_ref")
        revision = read_json(Path(cfg["project_root"]) / ref)["revision"]
    selection = catalog.select_tasks(partition=partition, task_ids=task_ids,
                                     benchmark=benchmark, revision=revision)
    if benchmark in {"webarena_verified", "swebench_pro_v2", "terminalbench_2_1"}:
        snapshot_tasks = read_json(Path(cfg["project_root"]) / ref)["tasks"]
        selected = {key: task for key, task in selection["tasks"].items() if key in snapshot_tasks}
        if task_ids and {task["instance_id"] for task in selected.values()} != set(task_ids):
            raise ValueError("Requested WebArena/Harbor tasks are outside the frozen benchmark snapshot")
        selection["tasks"], selection["task_keys"] = selected, sorted(selected)

    if benchmark == 'skillsbench':
        snapshot_tasks = read_json(Path(cfg['project_root']) / ref)['tasks']
        selected = {key: copy.deepcopy(snapshot_tasks[key]) for key in selection['task_keys']
                    if key in snapshot_tasks}
        if task_ids and {t['instance_id'] for t in selected.values()} != set(task_ids):
            raise ValueError('Requested SkillsBench tasks are outside the frozen benchmark snapshot')
        for key, task in selected.items():
            task['partition'] = selection['tasks'][key]['partition']
        selection['tasks'], selection['task_keys'] = selected, sorted(selected)

    tasks = selection["tasks"]

    return {

        "selection_type": "tasks",

        "catalog_revision": selection["catalog_revision"],

        "catalog_sha256": selection["catalog_sha256"],

        "task_keys": selection["task_keys"],

        "tasks": tasks,

        "rollout_ids": [],

    }





def create_run(cfg, run_id, *, operation, tasks_file=None, task_ids=(), partition=None, repeat_count=1,
               from_run=None, clone_overrides=(), parallel_config=None,
               reverse_task_order=False, online_work_items=None):

    run_id = identifier(run_id)

    if operation not in {"rollout", "harness", "enrichment", "hidden", "analysis"}:

        raise ValueError(f"Unsupported operation: {operation}")

    if type(repeat_count) is not int or repeat_count < 1:

        raise ValueError("repeat_count must be a positive integer")

    if operation != "rollout" and repeat_count != 1:

        raise ValueError("repeat_count is only configurable for rollout runs")

    if parallel_config is not None and operation != "rollout":
        raise ValueError("--parallel-config is only configurable for rollout runs")

    root = Path(cfg["project_root"])

    if reverse_task_order and from_run is None:
        raise ValueError("--reverse-task-order requires --from-run")
    if online_work_items is not None and operation != "rollout":
        raise ValueError("online_work_items are only supported for rollout runs")

    catalog = Catalog(root)
    clone_receipt = None
    frozen_selection = None
    if from_run is not None:
        if tasks_file or partition is not None:
            raise ValueError('A frozen clone accepts only an optional --task-id subset')
        source = load_plan(root, identifier(from_run))
        if operation != 'rollout' or source['plan']['operation'] != operation:
            raise ValueError('Frozen cloning requires a rollout source and destination')
        if repeat_count != source['plan']['repeat_count']:
            raise ValueError('Frozen cloning retains source repeat_count')
        allowed = {'serving.gpus', 'serving.tensor_parallel_size', 'serving.base_url',
                   'serving.gpu_memory_utilization', 'serving.max_model_len',
                   'serving.max_num_seqs', 'serving.max_num_batched_tokens',
                   'serving.trajectory_sharding', 'serving.replica_base_urls',
                   'serving.max_cpu_workers_per_gpu', 'serving.startup_mode',
                   'generation.max_tokens', 'generation.thinking_token_budget',
                   'generation.require_reasoning_close',
                   'webarena.environment_pool_ref', 'webarena.browser_image_ref',
                   'parallel.estimator_ref',
                   'paths.artifact_roots', 'model.verify_identity'}
        cfg = copy.deepcopy(source['config'])
        frozen_selection = copy.deepcopy(cfg.pop('selection'))
        cfg.pop('selection_sha256', None)
        for key in ['run_id', 'operation']:
            cfg.pop(key, None)
        changes = []
        for override in clone_overrides:
            dotted, value = override.split("=", 1)
            if dotted not in allowed:
                raise ValueError("Unsupported frozen-clone override: " + dotted)
            section, field = dotted.split(".", 1)
            parsed = yaml.safe_load(value)
            changes.append({"path": dotted, "before": cfg[section].get(field), "after": parsed})
            cfg[section][field] = parsed
        if task_ids:
            # A clone may reuse the source runtime/config while selecting a
            # different subset from the same frozen benchmark snapshot.
            frozen_selection = _selection_from_tasks(catalog, task_ids=task_ids, partition=None, cfg=cfg)
        if reverse_task_order:
            frozen_selection["task_keys"].reverse()
        from core.config import Paths, Serving, validate_generation_contract
        Paths.model_validate(cfg['paths'])
        artifact_roots(cfg, 'run')
        Serving.model_validate(cfg['serving'])
        # Rebuild the derived context below; never carry the source budget.
        cfg.pop('execution_context', None)
        validate_generation_contract(cfg)
        if cfg.get('parallel', {}).get('enabled'):
            from runtime.serving.parallel import artifact_contract
            contract = artifact_contract(cfg['parallel']['estimator_ref'],
                model_fingerprint=cfg['model']['fingerprint'])
            if list(contract['sites']) != list(cfg['parallel']['save_sites']):
                raise ValueError('Estimator public sites differ from the frozen capture/scoring sites')
            cfg['parallel']['estimator_sha256'] = sha256(cfg['parallel']['estimator_ref'])
        source_dir = run_path(root, from_run)
        cfg['config_source'] = str(source_dir / 'config.yaml')
        cfg['config_source_sha256'] = sha256(source_dir / 'config.yaml')
        cfg['overrides'] = list(clone_overrides)
        cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
        clone_receipt = {'source_run_id': from_run, 'source_config_hash': source['config']['config_hash'],
            'source_plan_sha256': source['plan']['plan_sha256'],
            'source_selection_sha256': source['plan']['selection_sha256'],
            'source_tasks_sha256': source['plan']['tasks_sha256'], 'overrides': changes,
            'task_order': 'reversed' if reverse_task_order else 'source',
            'created_at': now()}

    from core.config import with_derived_parallel_capacity
    from core.parallel_policy import apply_run_parallel

    # This is creation, including clones of older snapshots. Loading/resuming
    # the source still uses its frozen capacity without modifying its bytes.
    cfg = with_derived_parallel_capacity(cfg)
    if "supervisor_max_open_files" not in cfg["execution"]:
        from core.config import Execution
        cfg = copy.deepcopy(cfg)
        cfg["execution"]["supervisor_max_open_files"] = Execution.model_fields["supervisor_max_open_files"].default
        cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    parallel = cfg.get("parallel", {})
    if (parallel.get("enabled") and not parallel.get("dynamic_branch_admission", False)
            and parallel.get("branch_count", 1) > parallel["max_inflight_branches"]):
        raise ValueError("parallel capacity must accommodate one decision")

    cfg, parallel_input = apply_run_parallel(
        cfg, parallel_config, from_run=from_run, overrides=clone_overrides or cfg.get("overrides", ()))

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

    selection = frozen_selection if frozen_selection is not None else _selection_from_tasks(
        catalog, task_ids=task_ids, partition=partition, cfg=cfg)

    if not selection["task_keys"]:

        raise ValueError("Run task selection is empty")

    from core.backends import execution_context
    context = execution_context(cfg, selection["tasks"])
    if context is not None:
        cfg = copy.deepcopy(cfg)
        cfg["execution_context"] = context
        cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    # New runs bind the runtime/driver/evaluator explicitly. Loading legacy runs
    # derives this in memory and never changes their original snapshot hashes.
    from runtime.backends.registry import resolve as resolve_execution
    cfg = copy.deepcopy(cfg)
    cfg["execution_binding"] = resolve_execution(cfg, selection["tasks"]).as_dict()
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    branch_count = cfg.get("parallel", {}).get("branch_count", 1) if operation == "rollout" else 1
    online_contract = None
    if online_work_items is None:
        work_items = make_work_items(selection, run_id, repeat_count, cfg["generation"]["seed"], branch_count)
    else:
        if repeat_count < 2:
            raise ValueError("online runs require one DyCon and at least one baseline repetition")
        from RL.online_runtime import make_online_run_work_items
        work_items = make_online_run_work_items(selection, run_id, m=repeat_count - 1,
            base_seed=cfg["generation"]["seed"], branch_count=branch_count)
        if (not isinstance(online_work_items, dict)
                or online_work_items.get("run_id") != run_id
                or sorted(online_work_items.get("items", []), key=lambda item: item["sample_key"])
                   != sorted(work_items["items"], key=lambda item: item["sample_key"])
                or online_work_items.get("seed_policy") != work_items["seed_policy"]):
            raise ValueError("online work-item identities/profiles differ from frozen selection/config")
        work_items["items"] = copy.deepcopy(online_work_items["items"])
        online_contract = dict(work_items["online_contract"])
        online_contract["sample_order"] = [item["sample_key"] for item in work_items["items"]]
        work_items["online_contract"] = dict(online_contract)
        work_items["source_selection_sha256"] = online_work_items.get("selection_sha256")
    run = snapshot(cfg, run_id, selection)

    if parallel_input is not None:
        (run / "parallel-config.yaml").write_bytes(parallel_input)

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

    if cfg.get("parallel_policy"):
        effective_config["parallel"] = {
            k: v for k, v in cfg["parallel"].items() if k != "estimator_sha256"}
    # Human-readable YAML is an input surface; the derived value is recorded
    # only in resolved-config.json and the audit/hash chain.
    effective_config.get("parallel", {}).pop("max_inflight_branches", None)

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

    if online_contract is not None:
        plan["online_contract"] = online_contract
        plan["work_items_sha256"] = hash_json(work_items["items"])
    plan["plan_sha256"] = hash_json(plan)

    write_json(run / "plan.json", plan)

    for folder in (

        "logs", "events", "execution/attempts", "rollout/tasks", "rollout/trajectories",

        "rollout/patches", "enrichment", "hidden/sessions", "hidden/index",

        "hidden/branch", "harness", "reports",

    ):

        (run / folder).mkdir(parents=True, exist_ok=True)

    write_work_items(run, work_items)

    if operation == "rollout":

        from rollout.retry import initialize_counts

        initialize_counts(run, read_json(run / "rollout/work-items.json"))

    write_json(run / "status.json", {"stage": "planned", "run_id": run_id, "operation": operation})

    if operation == "rollout":

        from rollout.progress import Progress

        Progress(run, frozen, read_json(run / "rollout/work-items.json")["items"])._write()

    else:

        write_json(run / "progress.json", {})
        from application.execution.status import refresh_run_status_summary
        refresh_run_status_summary(root, run_id)

    write_json(run / "artifacts.json", {"schema_version": 1, "run_id": run_id, "artifacts": []})

    write_json(run / "code-identity.json", {"schema_version": 1, "git": _git_identity(root)})

    write_json(run / "events" / "000001.json", {

        "event": "run.planned", "run_id": run_id, "operation": operation,

        "config_hash": frozen["config_hash"], "selection_sha256": hash_json(selection),

    })

    if clone_receipt is not None:
        clone_receipt.update(new_run_id=run_id, new_config_hash=frozen['config_hash'])
        write_json(run / 'cloned-from.json', clone_receipt)
        write_json(run / 'events' / 'cloned-from.json', {'event': 'run.cloned', **clone_receipt})
    return run





def _git_identity(root):
    import shutil
    import subprocess

    git = shutil.which("git") or "git"

    try:

        head = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()

        dirty = subprocess.check_output([git, "-C", str(root), "status", "--porcelain"], text=True)

        return {"head": head, "dirty": bool(dirty), "dirty_sha256": hash_json(dirty),
                "source_files_sha256": {relative: sha256(Path(root) / relative) for relative in (
                    "rollout/adapters/openhands.py", "rollout/adapters/openhands_bridge.py",
                    "rollout/decision/coordinator.py", "runtime/serving/parallel.py",
                    "runtime/serving/vllm_plugin.py", "runtime/agents/model_socket.py",
                    "runtime/backends/native/swe_openhands_worker.py")
                    if (Path(root) / relative).is_file()}}

    except (OSError, subprocess.CalledProcessError) as exc:

        return {"unavailable": str(exc)}





def load_plan(root, run_id, *, allow_incomplete_revision=None, allow_incomplete_extension=None):

    root = Path(root).resolve()

    run_id = identifier(run_id)

    run = run_path(root, run_id)

    for journal in (run / 'config-revisions').glob('*/progress.json'):
        if (read_json(journal).get('phase') in {'prepared', 'committing', 'rolling_back'}
                and journal.parent.name != allow_incomplete_revision):
            raise RuntimeError('Incomplete configuration revision ' + journal.parent.name +
                               '; recover it with run revise-config --recover before execution')
    for journal in (run / 'repetition-extensions').glob('*/progress.json'):
        if (read_json(journal).get('phase') in {'prepared', 'committing'}
                and journal.parent.name != allow_incomplete_extension):
            raise RuntimeError('Incomplete repetition extension; recover with run add-repetitions --recover')
    plan = read_json(run / "plan.json")

    if plan.get("schema_version") != 2 or plan.get("migration_state", "active") != "active":

        raise ValueError(f"Run is not an active schema-2 run: {run_id}")

    expected = plan.pop("plan_sha256", None)

    if expected != hash_json(plan):

        raise ValueError(f"Run plan was modified: {run_id}")

    plan["plan_sha256"] = expected

    cfg = load_snapshot(root, run_id)
    from runtime.backends.registry import from_snapshot
    from_snapshot(cfg, cfg["selection"]["tasks"])

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
        extension=plan.get("sampling_extension"),

    )

    if plan.get("online_contract") is not None:
        from RL.online_runtime import make_online_run_work_items
        expected = make_online_run_work_items(cfg["selection"], run_id,
            m=plan["repeat_count"] - 1, base_seed=cfg["generation"]["seed"],
            branch_count=cfg.get("parallel", {}).get("branch_count", 1))
        order = plan["online_contract"].get("sample_order", [])
        by_key = {item["sample_key"]: item for item in expected["items"]}
        if len(order) != len(by_key) or set(order) != set(by_key):
            raise ValueError("Online sample order is not an exact frozen work permutation")
        expected["items"] = [by_key[key] for key in order]
        expected["online_contract"]["sample_order"] = order
        if (expected["online_contract"] != plan["online_contract"]
                or plan.get("work_items_sha256") != hash_json(work_items["items"])
                or work_items.get("online_contract") != plan["online_contract"]):
            raise ValueError("Online profile contract differs from immutable run plan")

    if expected["items"] != work_items["items"]:

        raise ValueError("Run work items disagree with frozen selection/config")

    from rollout.retry import enabled

    if (enabled(cfg) or plan.get("sampling_extension")) and expected["seed_policy"] != work_items.get("seed_policy"):

        raise ValueError("Run seed policy disagrees with frozen configuration")

    return {"run": run, "plan": plan, "config": cfg, "tasks": tasks, "work_items": work_items}





def show_run(root, run_id):

    loaded = load_plan(root, run_id)

    return {k: loaded[k] for k in ("plan", "tasks", "work_items")}





def list_runs(root):
    root = Path(root).resolve()
    values = set()
    config = root / "config.yaml"
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}
    paths = (raw or {}).get("paths", {})
    scan = list(paths.get("artifact_scan_roots", []))
    preferred = paths.get("artifact_roots", {}).get("run")
    if preferred:
        scan.insert(0, preferred)
    scan.insert(0, str(root))
    for value in scan:
        directory = Path(value).expanduser() / "runs"
        if directory.exists():
            values.update(p.name for p in directory.iterdir() if p.is_dir())
    return sorted(name for name in values if (run_path(root, name) / "plan.json").exists())
