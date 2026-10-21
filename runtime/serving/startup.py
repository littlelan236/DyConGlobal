"""Functional startup gate on the official model, parser and container path."""

from pathlib import Path

import numpy as np
import requests

from rollout.model import make_model
from core.io import read_json, sha256, write_json
from hidden.artifacts import event_records, read_arrays, read_request
from hidden.backends.vllm.online import capture_request_metadata
from runtime.containers.factory import TaskContext
from runtime.containers.rollout_adapter import RolloutEnvironment


def check_tool_chain(cfg, run_id, task, factory, runtime):
    root = Path(cfg["project_root"])
    out = root / "runs" / run_id / "startup"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "receipt.json").exists() and read_json(out / "receipt.json").get("passed"):
        return read_json(out / "receipt.json")
    cap, controller, metadata = cfg["capture"]["enabled"], runtime.get("capture_controller"), None
    sid = "startup-" + run_id
    if cap:
        inv = runtime["capture_token_inventory"]
        metadata = capture_request_metadata(
            sid,
            "startup",
            run_id,
            sid,
            0,
            cfg["capture"]["anchors"],
            newline_token_ids=inv["newline_token_ids"],
            close_token_ids=inv["close_token_ids"],
            tokenizer_fingerprint=inv["tokenizer_fingerprint"],
            newline_policy=cfg["capture"]["newline_policy"],
            capture_select=inv["capture_select"],
        )
        controller.start(metadata)
    messages = [
        {"role": "system", "content": cfg["agent"]["system_template"]},
        {
            "role": "user",
            "content": "This is a startup tool check. Think briefly about the current directory and environment, "
            "then call bash exactly once to run: pwd; printf 'DYCON_STARTUP_OK\\n'. Do not modify any files.",
        },
    ]
    env = RolloutEnvironment(
        factory, task["instance"], TaskContext(run_id, task["instance_id"], "startup", "rollout", out)
    )
    try:
        message = make_model(cfg, factory.deadline, metadata, runtime).query(messages)
        messages.append(message)
        actions = message["extra"]["actions"]
        from core.io import hash_json

        recorder = runtime.get("capture_controller") if message.get("extra", {}).get("dycon_parallel") else None
        if recorder:
            recorder.record_parallel_action(message["extra"]["dycon_parallel"], "started")
        observations = [env.execute(action) for action in actions]
        if recorder:
            recorder.record_parallel_action(message["extra"]["dycon_parallel"], "completed",
                                            {"observations_sha256": hash_json(observations)})
        if not observations or not any(
            "DYCON_STARTUP_OK" in o["output"] and o["returncode"] == 0 for o in observations
        ):
            raise ValueError("Startup tool did not execute successfully in the real container")
        path = out / "trajectory.json"
        write_json(path, {"messages": messages, "observations": observations})
        receipt = {
            "passed": False,
            "config_hash": cfg["config_hash"],
            "actions": actions,
            "observations": observations,
            "steering_enabled": cfg["steering"]["enabled"],
            "capture_enabled": cap,
            "parser": cfg["serving"]["tool_parser"],
        }
        if cap:
            endpoint = cfg["serving"]["base_url"].removesuffix("/v1").rstrip("/")
            response = requests.post(
                endpoint + "/dycon/capture/engine/flush", timeout=cfg["capture"]["finalize_seconds"]
            )
            response.raise_for_status()
            if response.json().get("failures"):
                raise ValueError(f"Native capture drain errors: {response.json()['failures']}")
            controller.stop(sid)
            controller.prepare_trajectory(sid, str(path))
            finalized = controller.finalize(
                sid,
                rollout_id=sid,
                trajectory_path=str(path),
                trajectory_sha256=sha256(path),
                model_fingerprint=cfg["model"]["fingerprint"],
            )
            if not cfg["capture"]["export"]["enabled"]:
                if finalized["capture_status"] != "not_exported":
                    raise ValueError("Disabled export must be reported as not_exported")
                receipt.update(passed=True, capture_status="not_exported")
                write_json(out / "receipt.json", receipt)
                return receipt
            if message.get("extra", {}).get("dycon_parallel"):
                from hidden.artifacts import read_branch_index

                branch_root = controller.spool.nas if controller.spool else controller.export_root
                branch = read_branch_index(branch_root, sid)
                if branch["coverage"]["missing"]:
                    raise ValueError("Startup parallel close coverage incomplete")
                for row in branch["samples"]:
                    if not row["close_events"]:
                        raise ValueError("Startup parallel branch has no close")
                    for event in row["close_events"]:
                        arrays = read_arrays(branch_root, event)
                        if not all(np.isfinite(a).all() for a in arrays.values()):
                            raise ValueError("Startup parallel branch nonfinite hidden")
                receipt["parallel"] = message["extra"]["dycon_parallel"]
            if finalized["capture_status"] != "complete":
                raise ValueError("Startup capture is not published and complete")
            source = controller.spool.nas if controller.spool else controller.export_root
            rows = [row for _, row in event_records(source) if row["session_id"] == sid]
            anchors = {r["anchor"] for r in rows}
            if not set(cfg["capture"]["anchors"]) <= anchors:
                raise ValueError(f"Startup missing configured anchors: {anchors}")
            if not any(r["graph_execution"] == "full" for r in rows):
                raise ValueError("Startup has no actual FULL decode capture evidence")
            for row in rows:
                request = controller.request_outputs[row["request_id"]]
                ledger = read_request(
                    source, {"ledger_ref": request["ledger_ref"], "ledger_sha256": request["ledger_sha256"]}
                )
                position = row["absolute_position"]
                ids = ledger["prompt_token_ids"] + ledger["output_token_ids"]
                if position >= len(ids) - 1 or ids[position] != row["token_id"]:
                    raise ValueError("Startup token coordinate mismatch")
                arrays = read_arrays(source, row)
                if len(arrays) != len(cfg["capture"]["layers"]) or not all(
                    np.isfinite(a).all() for a in arrays.values()
                ):
                    raise ValueError("Startup incomplete or nonfinite hidden layers")
            receipt.update(
                capture_status=finalized["capture_status"],
                anchors=sorted(anchors),
                events=len(rows),
                source_root=str(source),
                session_id=sid,
            )
        receipt["passed"] = True
        if cap and runtime.get("validation_mixed"):
            receipt["mixed_batch"] = check_mixed_batches(cfg, run_id, factory, runtime)
        if (runtime.get("validation_mixed") and cfg.get("parallel", {}).get("enabled")
                and cfg["parallel"]["branch_count"] > 1
                and cfg["parallel"].get("validation_repetitions", 0)):
            from runtime.serving.parallel_validation import validate_parallel_queries

            receipt["parallel_validation"] = validate_parallel_queries(cfg, run_id, factory, runtime)
        write_json(out / "receipt.json", receipt)
        return receipt
    finally:
        env.close()


def check_mixed_batches(cfg, run_id, factory, runtime):
    """Controlled functional overlap; native scheduling, no timing measurements."""
    if "reasoning_newline" not in cfg["capture"]["anchors"]:
        return {"status": "skipped", "reason": "requires_reasoning_newline_capture"}
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from minisweagent.exceptions import FormatError

    controller, inventory = runtime["capture_controller"], runtime["capture_token_inventory"]
    sid = "startup-mixed-" + run_id
    md = capture_request_metadata(
        sid,
        "startup-mixed",
        run_id,
        sid,
        0,
        cfg["capture"]["anchors"],
        newline_token_ids=inventory["newline_token_ids"],
        close_token_ids=inventory["close_token_ids"],
        tokenizer_fingerprint=inventory["tokenizer_fingerprint"],
        capture_select=inventory["capture_select"],
    )
    controller.start(md)
    newline = inventory["newline_token_ids"][0]
    barrier = threading.Barrier(2)

    def one(index):
        # This native scheduler diagnostic constrains output to newline tokens,
        # deliberately has no close/tool call, and is not a branch decision.
        model = make_model(cfg, factory.deadline, md, {**runtime, "diagnostic_single_query": True})
        model._dycon_capture_step = index
        messages = [
            {
                "role": "user",
                "content": "Functional scheduling check. "
                + ("repository note " * 10000 if index else "short"),
            }
        ]
        extra = dict(model.config.model_kwargs["extra_body"])
        extra["allowed_token_ids"] = [newline]
        barrier.wait()
        try:
            return model.query(messages, max_tokens=32 if not index else 8, extra_body=extra)
        except FormatError as exc:
            return exc.messages[0]  # constrained diagnostic deliberately has no tool call

    with ThreadPoolExecutor(max_workers=2) as executor:
        messages = list(executor.map(one, [0, 1]))
    endpoint = cfg["serving"]["base_url"].removesuffix("/v1").rstrip("/")
    response = requests.post(
        endpoint + "/dycon/capture/engine/flush", timeout=cfg["capture"]["finalize_seconds"]
    )
    response.raise_for_status()
    controller.stop(sid)
    path = Path(cfg["project_root"]) / "runs" / run_id / "startup/mixed-trajectory.json"
    write_json(path, {"messages": messages})
    controller.prepare_trajectory(sid, path)
    result = controller.finalize(
        sid,
        rollout_id=sid,
        trajectory_path=str(path),
        trajectory_sha256=sha256(path),
        model_fingerprint=cfg["model"]["fingerprint"],
    )
    if result["capture_status"] != "complete":
        raise ValueError("Mixed-batch capture did not publish completely")
    source = controller.spool.nas if controller.spool else controller.export_root
    rows = [row for _, row in event_records(source) if row["session_id"] == sid]
    if not rows or not any(row["batch_kind"] == "mixed" for row in rows):
        raise ValueError("Controlled overlap did not observe a native mixed batch")
    if any(row["token_id"] != newline for row in rows):
        raise ValueError("Controlled mixed batch returned an unexpected token")
    return {
        "passed": True,
        "events": len(rows),
        "batch_kinds": sorted({row["batch_kind"] for row in rows}),
        "executions": sorted({row["graph_execution"] for row in rows}),
        "session_id": sid,
    }
