"""Manifest-driven runner; only infrastructure failures receive fresh attempts."""

from __future__ import annotations

import concurrent.futures
import os
import traceback
from pathlib import Path

from catalog.store import Catalog
from application.execution.evidence import RunEvidence
from core.config import load_snapshot, snapshot
from core.io import identifier, now, read_json, sha256, update_status, write_json
from core.locking import exclusive_run, file_lock
from rollout.artifacts import import_rollout

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


def run_task(cfg, run_id, task_key, task, factory, runtime=None):
    runtime = runtime or {}
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
    from rollout.agent import DurableAgent
    from rollout.model import make_model
    from runtime.containers.factory import TaskContext
    from runtime.containers.rollout_adapter import RolloutEnvironment

    iid = identifier(task["instance_id"])
    root = Path(cfg["project_root"])
    work = root / "runs" / run_id / "rollout/tasks" / iid
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
            capture_client = None
            capture_session_id = None
            capture_errors = []
            capture_status = "disabled"

            def capture_failure(exc, *, abort=True):
                capture_errors.append(repr(exc))
                write_json(out / "capture-errors.json", capture_errors)
                if abort and cfg["capture"]["failure_policy"] == "block_rollout":
                    from runtime.serving.errors import CaptureError

                    raise CaptureError(str(exc)) from exc

            status, patch, error, started = "Interrupted", "", None, now()
            try:
                env = RolloutEnvironment(factory, task["instance"], ctx)
                capture_context = None
                if cfg.get("capture", {}).get("enabled"):
                    capture_context = False
                    capture_status = "collecting"
                    try:
                        inventory = runtime["capture_token_inventory"]
                        capture_context = capture_request_metadata(
                            f"cap-{run_id}-{iid}-{attempt_id}",
                            iid,
                            run_id,
                            f"pending-{iid}-{attempt_id}",
                            0,
                            cfg["capture"]["anchors"],
                            newline_token_ids=inventory["newline_token_ids"],
                            close_token_ids=inventory["close_token_ids"],
                            tokenizer_fingerprint=inventory["tokenizer_fingerprint"],
                            newline_policy=cfg["capture"]["newline_policy"],
                            capture_select=inventory["capture_select"],
                        )
                        capture_client = CaptureClient(
                            runtime["capture_control_url"], timeout=cfg["capture"]["control_seconds"]
                        )
                        capture_session_id = capture_context["session_id"]
                        capture_client.start(
                            capture_context, graph_mode="FULL", prefix_cache_policy="safe_boundary"
                        )
                    except Exception as exc:
                        capture_context = False
                        capture_status = "failed"
                        capture_failure(exc)
                agent = DurableAgent(
                    make_model(cfg, factory.deadline, capture_context, runtime),
                    env,
                    system_template=cfg["agent"]["system_template"],
                    instance_template=cfg["agent"]["instance_template"],
                    step_limit=cfg["agent"]["step_limit"],
                    cost_limit=cfg["agent"]["cost_limit"],
                    max_consecutive_format_errors=cfg["agent"]["max_consecutive_format_errors"],
                    loop_repeat_limit=cfg["agent"].get("loop_repeat_limit"),
                    loop_max_period=cfg["agent"].get("loop_max_period", 7),
                    output_path=path,
                )
                info = agent.run(task["instance"]["problem_statement"])
                status, patch = info.get("exit_status", "Unknown"), info.get("submission", "")
            except Exception as exc:  # noqa: BLE001 -- persist and classify a worker failure
                if agent is not None and getattr(exc, "response_message", None) is not None:
                    agent.messages.append(exc.response_message)
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
                    if capture_client is not None and capture_session_id is not None:
                        try:
                            import requests

                            endpoint = cfg["serving"]["base_url"].removesuffix("/v1").rstrip("/")
                            flushed = requests.post(
                                endpoint + "/dycon/capture/engine/flush",
                                timeout=cfg["capture"]["finalize_seconds"],
                            )
                            flushed.raise_for_status()
                            if flushed.json().get("failures"):
                                raise RuntimeError(f"Capture drain failed: {flushed.json()['failures']}")
                            # Stop only this session, after native rows have been drained.
                            capture_client.timeout = (
                                cfg["capture"]["finalize_seconds"] + cfg["capture"]["control_seconds"]
                            )
                            capture_client.stop(capture_session_id)
                            capture_client.prepare(capture_session_id, path)
                        except Exception as exc:
                            capture_failure(exc, abort=False)
                            capture_status = "partial"
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
                    "run_local": True,
                },
            )
            if capture_client is not None and capture_session_id is not None:
                canonical = root / RunEvidence(root, run_id).read()["rollouts"][rid]["trajectory_path"]
                try:
                    capture_client.timeout = (
                        cfg["capture"]["export"]["upload_max_elapsed_seconds"]
                        + cfg["capture"]["finalize_seconds"]
                    )
                    finalized = capture_client.finalize(
                        capture_session_id,
                        rollout_id=rid,
                        trajectory_path=str(canonical),
                        trajectory_sha256=sha256(canonical),
                        model_fingerprint=cfg["model"]["fingerprint"],
                    )
                    capture_status = finalized["capture_status"]
                    if capture_status not in {"complete", "not_exported"}:
                        capture_failure(RuntimeError("Capture is " + capture_status), abort=False)
                except Exception as exc:
                    capture_status = "partial"
                    capture_failure(exc, abort=False)
            path.unlink()  # canonical bytes now live under the run-local rollout tree
            result = {
                "instance_id": iid,
                "rollout_id": rid,
                "attempt_id": attempt_id,
                "exit_status": "CaptureError"
                if capture_errors and cfg["capture"]["failure_policy"] == "block_rollout"
                else status,
                "agent_exit_status": status,
                "started_at": started,
                "finished_at": now(),
                "error": error,
                "capture_status": capture_status,
                "capture_errors": capture_errors,
            }
            write_json(out / "attempt.json", result)
            write_json(work / "result.json", result)
            if status not in INFRASTRUCTURE:
                return result
        if result is None:
            raise RuntimeError(f"Retry budget exhausted for {iid}; interrupted attempts are preserved")
        return result


@exclusive_run
def run_manifest(cfg, run_id, resume=False, factory=None, *, task_ids=(), check_only=False):
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
        selection = Catalog(root).select_tasks(
            partition=cfg["benchmark"]["partition"], task_ids=task_ids
        )
        if not selection["task_keys"]:
            raise ValueError("Baseline task scope is empty")
        verify_model(cfg)
        snapshot(cfg, run_id, selection)
        cfg = load_snapshot(root, run_id)
    from core.config import assert_frozen

    assert_frozen(cfg)
    if resume and not check_only:
        completed = {}
        current = RunEvidence(root, run_id).read()
        for key, task in selection["tasks"].items():
            result_path = root / "runs" / run_id / "rollout/tasks" / task["instance_id"] / "result.json"
            if not result_path.exists():
                break
            result = read_json(result_path)
            if result["exit_status"] not in TERMINAL:
                break
            row = current["rollouts"][result["rollout_id"]]
            if row["run_id"] != run_id or row["task_key"] != key:
                raise ValueError("Completed result identity changed")
            for kind in ("trajectory", "patch"):
                if sha256(root / row[kind + "_path"]) != row[kind + "_sha256"]:
                    raise ValueError("Completed canonical bytes changed")
            completed[key] = result
        else:
            if completed:
                write_json(root / "runs" / run_id / "progress.json", completed)
                update_status(root / "runs" / run_id / "status.json", "complete", failed=[])
                return completed

    verify_model(cfg)
    run = root / "runs" / run_id
    write_json(run / "process.json", {"pid": os.getpid(), "supervisor_pid": os.getppid()})
    factory = factory or ContainerFactory(cfg)
    outcomes = {}
    update_status(run / "status.json", "rollout")
    try:
        factory.lock_images([t["instance"] for t in selection["tasks"].values()])
        with ServingManager(cfg, run / "serving") as serving:
            runtime = getattr(serving, "runtime", {})
            runtime["validation_mixed"] = check_only
            from runtime.serving.startup import check_tool_chain

            check_tool_chain(cfg, run_id, next(iter(selection["tasks"].values())), factory, runtime)
            if check_only:
                update_status(run / "status.json", "validated")
                return {"startup": "passed"}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=cfg["execution"]["rollout_workers"]
            ) as pool:
                futures = {
                    pool.submit(admitted_task, cfg, run_id, key, task, factory, runtime): key
                    for key, task in selection["tasks"].items()
                }
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    try:
                        outcomes[key] = future.result()
                    except Exception as exc:  # noqa: BLE001 -- retain cohort progress on worker errors
                        outcomes[key] = {"exit_status": "WorkerFailure", "error": repr(exc)}
                    write_json(run / "progress.json", outcomes)
        if cfg["capture"]["enabled"] and cfg["capture"]["export"]["enabled"]:
            from enrichment.online_trace import finalize_token_events

            source = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
            target = root / "runs" / run_id / "enrichment/online"
            if not target.exists() and (source / "requests").exists():
                try:
                    inventory = read_json(source / "inventory.json")
                    finalize_token_events(
                        source,
                        target,
                        open_ids=inventory.get("open_token_ids", [248068]),
                        close_ids=inventory["close_token_ids"],
                    )
                except Exception as exc:
                    write_json(run / "capture-finalization-error.json", {"error": repr(exc)})
                    if cfg["capture"]["failure_policy"] == "block_rollout":
                        from runtime.serving.errors import CaptureError

                        raise CaptureError(str(exc)) from exc
        failed = [k for k, v in outcomes.items() if v["exit_status"] not in TERMINAL]
        update_status(run / "status.json", "failed" if failed else "complete", failed=failed)
        return outcomes
    except BaseException as exc:
        update_status(run / "status.json", "failed", error=repr(exc))
        raise
    finally:
        factory.cleanup_owned(run_id)


def admitted_task(cfg, run_id, key, task, factory, runtime):
    controller = runtime.get("capture_controller")
    if controller and controller.spool:
        options = cfg["capture"]["export"]
        controller.spool.budget.admit(
            options["spool_high_watermark"],
            options["spool_low_watermark"],
            options["upload_max_elapsed_seconds"],
        )
    return run_task(cfg, run_id, key, task, factory, runtime)
