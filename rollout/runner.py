"""Manifest-driven runner; only infrastructure failures receive fresh attempts."""

from __future__ import annotations

import concurrent.futures
import copy
import os
import traceback
from pathlib import Path

from catalog.store import Catalog
from application.execution.evidence import RunEvidence
from core.config import load_snapshot
from core.io import identifier, now, read_json, sha256, update_status, write_json
from core.locking import exclusive_run, file_lock
from rollout.artifacts import import_rollout
from application.execution.run_plan import create_run, load_plan
from rollout.sampling import SampleContext, sample_from_item

TERMINAL = {
    "Submitted",
    "reasoning_loop",
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


def run_task(cfg, run_id, task_key, task, factory, runtime=None, sample: SampleContext | None = None):
    runtime = runtime or {}
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
    from rollout.agent import DurableAgent
    from rollout.model import make_model
    from runtime.containers.factory import TaskContext
    from runtime.containers.rollout_adapter import RolloutEnvironment

    iid = identifier(task["instance_id"])
    sample = sample or SampleContext(
        run_id=run_id, task_key=task_key, instance_id=iid,
        repetition_id="rep-000", repetition_index=0,
        sample_key="legacy-sample-" + task_key[:24], sample_seed=cfg["generation"]["seed"],
    )
    local_cfg = copy.deepcopy(cfg)
    local_cfg["generation"]["seed"] = sample.sample_seed
    root = Path(cfg["project_root"])
    work = root / "runs" / run_id / "rollout/tasks" / iid / "repetitions" / sample.repetition_id
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
            attempt_id = f"{sample.repetition_id}-attempt-{n:03d}"
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
                            "cap-" + sample.sample_key + "-" + attempt_id,
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
                    make_model(local_cfg, factory.deadline, capture_context, runtime),
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
                    "sample": sample.as_dict(),
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
                "sample_key": sample.sample_key,
                "repetition_id": sample.repetition_id,
                "repetition_index": sample.repetition_index,
                "sample_seed": sample.sample_seed,
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
    if not resume:
        if (root / "runs" / run_id).exists():
            raise FileExistsError(f"Run already exists: {run_id}")
        create_run(
            cfg, run_id, operation="rollout", task_ids=task_ids,
            repeat_count=1,
        )
    loaded = load_plan(root, run_id)
    cfg = loaded["config"]
    selection = cfg["selection"]
    work_items = loaded["work_items"]["items"]
    from core.config import assert_frozen

    assert_frozen(cfg)
    if resume and not check_only:
        evidence = RunEvidence(root, run_id).read()
        settled = {}
        for item in work_items:
            result_path = root / "runs" / run_id / "rollout/tasks" / item["instance_id"] / "repetitions" / item["repetition_id"] / "result.json"
            if not result_path.exists():
                continue
            result = read_json(result_path)
            if result.get("sample_key") != item["sample_key"]:
                raise ValueError("Completed sample identity changed")
            if result.get("exit_status") in TERMINAL or result.get("exit_status") == "infrastructure_exhausted":
                rid = result.get("rollout_id")
                if rid and rid in evidence["rollouts"]:
                    row = evidence["rollouts"][rid]
                    for kind in ("trajectory", "patch"):
                        if sha256(root / row[kind + "_path"]) != row[kind + "_sha256"]:
                            raise ValueError("Completed canonical bytes changed")
                settled[item["sample_key"]] = result
        if len(settled) == len(work_items):
            write_json(root / "runs" / run_id / "progress.json", {"schema_version": 2, "samples": settled})
            update_status(root / "runs" / run_id / "status.json", "complete", failed=[])
            return settled

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
            pending = iter(work_items)
            running = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=cfg["execution"]["rollout_workers"]
            ) as pool:
                def fill():
                    while len(running) < cfg["execution"]["rollout_workers"]:
                        try:
                            item = next(pending)
                        except StopIteration:
                            return
                        task = selection["tasks"][item["task_key"]]
                        if item["sample_key"] in outcomes:
                            continue
                        future = pool.submit(
                            admitted_task, cfg, run_id, item["task_key"], task, factory, runtime,
                            sample_from_item({**item, "run_id": run_id}),
                        )
                        running[future] = item

                fill()
                while running:
                    finished, _ = concurrent.futures.wait(
                        running, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in finished:
                        item = running.pop(future)
                        try:
                            value = future.result()
                        except Exception as exc:  # noqa: BLE001 -- retain cohort progress on worker errors
                            value = {
                                "sample_key": item["sample_key"],
                                "repetition_id": item["repetition_id"],
                                "exit_status": "WorkerFailure",
                                "error": repr(exc),
                            }
                        outcomes[item["sample_key"]] = value
                    write_json(run / "progress.json", {"schema_version": 2, "samples": outcomes})
                    fill()
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
        failed = [k for k, v in outcomes.items() if v.get("exit_status") not in TERMINAL]
        update_status(run / "status.json", "failed" if failed else "complete", failed=failed)
        return outcomes
    except BaseException as exc:
        update_status(run / "status.json", "failed", error=repr(exc))
        raise
    finally:
        factory.cleanup_owned(run_id)


def admitted_task(cfg, run_id, key, task, factory, runtime, sample=None):
    controller = runtime.get("capture_controller")
    if controller and controller.spool:
        options = cfg["capture"]["export"]
        controller.spool.budget.admit(
            options["spool_high_watermark"],
            options["spool_low_watermark"],
            options["upload_max_elapsed_seconds"],
        )
    return run_task(cfg, run_id, key, task, factory, runtime, sample=sample)
