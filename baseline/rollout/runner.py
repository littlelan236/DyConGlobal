"""Manifest-driven runner; only infrastructure failures receive fresh attempts."""

from __future__ import annotations

import concurrent.futures
import os
import traceback
from pathlib import Path

from baseline.catalog import Catalog
from baseline.importers.legacy import import_rollout
from core.config import load_snapshot, snapshot
from core.io import identifier, now, read_json, update_status, write_json
from core.locking import exclusive_run, file_lock

TERMINAL = {
    "Submitted",
    "LimitsExceeded",
    "ContextWindowExceededError",
    "RepeatedFormatError",
    "TimeExceeded",
}
INFRASTRUCTURE = {
    "ContainerCorrupted",
    "EnvironmentSetupFailed",
    "APIError",
    "Timeout",
    "ReadTimeout",
    "ConnectError",
    "APIConnectionError",
    "DockerException",
    "NotFound",
    "ConnectionError",
}


def run_task(cfg, run_id, task_key, task, factory):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from baseline.rollout.agent import DurableAgent
    from baseline.rollout.model import make_model
    from runtime.containers.factory import TaskContext
    from runtime.containers.rollout_adapter import RolloutEnvironment

    iid = identifier(task["instance_id"])
    root = Path(cfg["project_root"])
    work = root / "runs" / run_id / "tasks" / iid
    with file_lock(work / "execution.lock", nonblocking=True):
        if (work / "result.json").exists():
            old = read_json(work / "result.json")
            if old["exit_status"] in TERMINAL:
                return old
        attempts = work / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        existing = list(attempts.iterdir())
        offset = len(existing)
        for p in existing:
            if not (p / "attempt.json").exists():
                write_json(p / "attempt.json", {"attempt_id": p.name, "exit_status": "Interrupted"})
        result = None
        for n in range(offset, cfg["execution"]["infrastructure_retry_limit"] + 1):
            factory.deadline.check()
            attempt_id = f"attempt-{n:03d}"
            out = attempts / attempt_id
            out.mkdir()
            path = out / "raw.traj.json"
            ctx = TaskContext(run_id, iid, attempt_id, "rollout", out)
            env = agent = None
            status, patch, error, started = "Interrupted", "", None, now()
            try:
                env = RolloutEnvironment(factory, task["instance"], ctx)
                agent = DurableAgent(
                    make_model(cfg, factory.deadline),
                    env,
                    system_template=cfg["agent"]["system_template"],
                    instance_template=cfg["agent"]["instance_template"],
                    step_limit=cfg["agent"]["step_limit"],
                    cost_limit=cfg["agent"]["cost_limit"],
                    max_consecutive_format_errors=cfg["agent"]["max_consecutive_format_errors"],
                    output_path=path,
                )
                info = agent.run(task["instance"]["problem_statement"])
                status, patch = info.get("exit_status", "Unknown"), info.get("submission", "")
            except Exception as exc:  # noqa: BLE001 -- persist and classify a worker failure
                status = type(exc).__name__
                error = {"error": str(exc), "traceback": traceback.format_exc()}
            finally:
                try:
                    if agent is not None:
                        agent.save(
                            path,
                            {
                                "instance_id": iid,
                                "info": {
                                    "exit_status": status,
                                    "submission": patch,
                                    "attempt_id": attempt_id,
                                    "dycon": {"run_id": run_id, "config_hash": cfg["config_hash"]},
                                },
                            },
                        )
                    elif not path.exists():
                        write_json(
                            path,
                            {
                                "instance_id": iid,
                                "messages": [],
                                "info": {"exit_status": status, "submission": "", "attempt_id": attempt_id},
                            },
                        )
                finally:
                    if env is not None:
                        env.close()
            rid = import_rollout(
                root,
                {
                    "source": str(path),
                    "task": task,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "purposes": ["baseline"],
                    "model_fingerprint": cfg["model"]["fingerprint"],
                },
            )
            path.unlink()  # canonical bytes now live under baseline/ only
            result = {
                "instance_id": iid,
                "rollout_id": rid,
                "attempt_id": attempt_id,
                "exit_status": status,
                "started_at": started,
                "finished_at": now(),
                "error": error,
            }
            write_json(out / "attempt.json", result)
            write_json(work / "result.json", result)
            if status not in INFRASTRUCTURE:
                return result
        if result is None:
            raise RuntimeError(f"Retry budget exhausted for {iid}; interrupted attempts are preserved")
        return result


@exclusive_run
def run_manifest(cfg, run_id, resume=False, factory=None):
    # Prevent mini-swe-agent from loading the user's unrelated global .env.
    os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(Path(cfg["project_root"]) / "state/miniswe-config")
    os.environ["MSWEA_SILENT_STARTUP"] = "1"
    if (Path(os.environ["MSWEA_GLOBAL_CONFIG_DIR"]) / ".env").exists():
        raise ValueError(
            "mini-swe-agent .env overrides are disabled; use config.yaml and API-key environment"
        )
    from core.model_identity import verify_model
    from runtime.containers.factory import ContainerFactory
    from runtime.serving.manager import ServingManager

    root = Path(cfg["project_root"])
    run_id = identifier(run_id)
    if resume:
        cfg = load_snapshot(root, run_id)
        selection = cfg["selection"]
    else:
        selection = Catalog(root).select_tasks(partition=cfg["benchmark"]["partition"])
        if not selection["task_keys"]:
            raise ValueError("Baseline task scope is empty")
        verify_model(cfg)
        snapshot(cfg, run_id, selection)
        cfg = load_snapshot(root, run_id)
    verify_model(cfg)
    run = root / "runs" / run_id
    factory = factory or ContainerFactory(cfg)
    outcomes = {}
    update_status(run / "status.json", "rollout")
    try:
        factory.lock_images([t["instance"] for t in selection["tasks"].values()])
        with ServingManager(cfg, run / "serving"):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=cfg["execution"]["rollout_workers"]
            ) as pool:
                futures = {
                    pool.submit(run_task, cfg, run_id, key, task, factory): key
                    for key, task in selection["tasks"].items()
                }
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    try:
                        outcomes[key] = future.result()
                    except Exception as exc:  # noqa: BLE001 -- retain cohort progress on worker errors
                        outcomes[key] = {"exit_status": "WorkerFailure", "error": repr(exc)}
                    write_json(run / "progress.json", outcomes)
        failed = [k for k, v in outcomes.items() if v["exit_status"] not in TERMINAL]
        update_status(run / "status.json", "failed" if failed else "complete", failed=failed)
        return outcomes
    except BaseException as exc:
        update_status(run / "status.json", "failed", error=repr(exc))
        raise
    finally:
        factory.cleanup_owned(run_id)
