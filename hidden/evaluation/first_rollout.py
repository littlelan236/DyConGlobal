"""Event-triggered first-completion audit and bounded independent vLLM replay.

Existing completions are handled once at startup; future completions wake on
the official runner's atomic progress write. No timer-based status polling.
"""

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from baseline.catalog import Catalog
from core.config import load_snapshot
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import directory_changes, file_lock
from enrichment.online_trace import reasoning_targets
from hidden.artifacts import event_records, read_arrays, read_request, response_records
from hidden.backends.vllm.capture import native_layers


def first_completed(run, task_ids):
    rows = []
    for path in (Path(run) / "tasks").glob("*/attempts/*/attempt.json"):
        row = read_json(path)
        if row.get("instance_id") in task_ids and row.get("rollout_id") and row.get("finished_at"):
            rows.append((datetime.fromisoformat(row["finished_at"]), str(path), row))
    if not rows:
        return None
    _, path, row = min(rows, key=lambda value: (value[0], value[1]))
    return {"attempt_receipt": path, **row}


def wait_first(root, run_id, task_ids, timeout):
    run = Path(root) / "runs" / identifier(run_id)
    deadline = time.monotonic() + timeout
    pid = read_json(run / "process.json")["pid"]
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        pidfd = None
    try:
        with directory_changes(run) as changed:
            while True:
                result = first_completed(run, task_ids)
                if result is not None:
                    return result
                if pidfd is None:
                    raise RuntimeError("Runner exited without a completed rollout")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No first completion before the event-wait deadline")
                ready = changed(remaining, [pidfd])
                if pidfd in ready:
                    result = first_completed(run, task_ids)
                    if result is None:
                        raise RuntimeError("Runner exited without a completed rollout")
                    return result
    finally:
        if pidfd is not None:
            os.close(pidfd)


def inspect_first(cfg, result, out):
    root = Path(cfg["project_root"])
    catalog = Catalog(root).read()
    rollout_id = result["rollout_id"]
    record = catalog["rollouts"][rollout_id]
    task = catalog["tasks"][record["task_key"]]
    if record["run_id"] != Path(out).parent.name or task["instance_id"] != result["instance_id"]:
        raise ValueError("First completion identity disagrees with canonical manifest")
    trajectory = root / record["trajectory_path"]
    if (
        sha256(trajectory) != record["trajectory_sha256"]
        or sha256(root / record["patch_path"]) != record["patch_sha256"]
    ):
        raise ValueError("Canonical trajectory/patch checksum mismatch")
    source = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / record["run_id"]
    data = read_json(trajectory)
    responses = list(response_records(data))
    if not responses:
        raise ValueError("First rollout contains no model responses")
    requests, reference_rows, notes = {}, {}, []
    session = None
    for message_index, message, response in responses:
        ledger = read_request(source, response)
        native_id = response["dycon_native_request_id"]
        if native_id in requests or ledger["request_id"] != native_id:
            raise ValueError("Duplicate or mismatched native request ID")
        if ledger["task_id"] != result["instance_id"] or ledger["run_id"] != record["run_id"]:
            raise ValueError("Request crossed a task/run boundary")
        if session is not None and session != ledger["session_id"]:
            raise ValueError("One rollout unexpectedly spans capture sessions")
        session = ledger["session_id"]
        metadata = (message.get("extra", {}).get("request") or {}).get("dycon_capture", {})
        for key in ("query_id", "api_attempt_id"):
            if metadata.get(key) != ledger.get(key) or not ledger.get(key):
                raise ValueError(f"Missing/mismatched {key}")
        if (
            hash_json(ledger["prompt_token_ids"]) != ledger["request_sha256"]
            or hash_json(ledger["output_token_ids"]) != ledger["output_sha256"]
        ):
            raise ValueError("Native token ledger checksum mismatch")
        if ledger["output_token_ids"] and not any(ledger["output_token_ids"]):
            raise ValueError("Degenerate all-zero native generation")
        requests[native_id] = ledger
        reference_rows[native_id] = {
            "ledger_ref": response["ledger_ref"],
            "ledger_sha256": response["ledger_sha256"],
        }
        actions = (message.get("extra") or {}).get("actions") or []
        notes.append(
            {
                "message_index": message_index,
                "role": message["role"],
                "native_request_id": native_id,
                "reasoning": message.get("reasoning_content"),
                "content": message.get("content"),
                "actions": actions,
                "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
                "prompt_tokens": len(ledger["prompt_token_ids"]),
                "output_tokens": len(ledger["output_token_ids"]),
            }
        )
    finalization = read_json(source / "sessions" / session / "finalization.json")
    if finalization["rollout_id"] != rollout_id or finalization["trajectory_sha256"] != sha256(trajectory):
        raise ValueError("NAS finalization does not bind canonical bytes")
    if set(finalization["accepted_native_request_ids"]) != set(requests):
        raise ValueError("Finalization does not bind every persisted response exactly once")
    inventory = read_json(source / "inventory.json")
    expected = set()
    for rid, request in requests.items():
        targets, _ = reasoning_targets(request, inventory["open_token_ids"], inventory["close_token_ids"])
        expected.update((rid, position, anchor) for position, anchor in targets)
    model_json = read_json(Path(cfg["model"]["path"]) / "config.json")
    text = model_json.get("text_config", model_json)
    sites = native_layers(cfg["capture"]["layers"], text["num_hidden_layers"])
    required_keys = {f"layer_{site}" for site in sites}
    selected, observed, checked_shards, raw_count = [], set(), {}, 0
    for path, event in event_records(source):
        rid = event["request_id"]
        if rid not in requests:
            continue
        raw_count += 1
        key = (rid, event["absolute_position"], event["anchor"])
        if key not in expected:
            continue
        if key in observed:
            raise ValueError("Duplicate reasoning event coordinate")
        request = requests[rid]
        prompt, output = request["prompt_token_ids"], request["output_token_ids"]
        pos = event["absolute_position"]
        gi = pos - len(prompt)
        if not 0 <= gi < len(output) - 1 or output[gi] != event["token_id"]:
            raise ValueError("Online event has no exact subsequent input-forward evidence")
        if event.get("generation_index") != gi or event["request_sha256"] != hash_json(prompt):
            raise ValueError("Online coordinate/request checksum mismatch")
        locator = event.get("tensor_locator")
        if locator and locator["shard"] not in checked_shards:
            digest = sha256(source / locator["shard"])
            if digest != locator["shard_sha256"]:
                raise ValueError("Published NAS shard checksum mismatch")
            checked_shards[locator["shard"]] = digest
        arrays = read_arrays(source, event)
        if set(arrays) != required_keys or any(
            a.shape != (text["hidden_size"],) or not np.isfinite(a).all() for a in arrays.values()
        ):
            raise ValueError("Online hidden site/shape/finiteness mismatch")
        selected.append(
            {
                **event,
                **reference_rows[rid],
                "rollout_id": rollout_id,
                "trajectory_sha256": sha256(trajectory),
                "alignment_status": "exact",
                "token_ref": [rollout_id, request["step_id"], "generation_input", pos, event["token_id"]],
            }
        )
        observed.add(key)
    missing = sorted(expected - observed)
    extras = []
    for rid, pos, anchor in missing:
        request = requests[rid]
        ids = request["prompt_token_ids"] + request["output_token_ids"]
        prefix = ids[: pos + 1]
        extras.append(
            {
                "sample_id": "missing-" + hash_json([rid, pos, anchor])[:32],
                "request_id": rid,
                "absolute_position": pos,
                "anchor": anchor,
                "token_id": ids[pos],
                "prefix_token_ids": prefix,
                "prefix_sha256": hash_json(prefix),
                "trajectory_sha256": sha256(trajectory),
                **reference_rows[rid],
            }
        )
    selected.sort(key=lambda row: (requests[row["request_id"]]["step_id"], row["absolute_position"]))
    write_json(out / "paired-events.json", selected)
    write_json(out / "missing-prefixes.json", extras)
    write_json(out / "reasoning-and-actions.json", notes)
    commands = [a["command"] for n in notes for a in n["actions"]]
    observations = [m for m in data["messages"] if m.get("role") == "tool"]
    format_errors = sum(n["role"] != "assistant" for n in notes)
    export_failures = source / "sessions" / session / "failures.json"
    coverage = {
        "expected": len(expected),
        "observed": len(observed),
        "missing": missing,
        "by_anchor_expected": dict(Counter(k[2] for k in expected)),
        "by_anchor_observed": dict(Counter(k[2] for k in observed)),
    }
    audit = {
        "run_id": record["run_id"],
        "instance_id": result["instance_id"],
        "rollout_id": rollout_id,
        "trajectory": str(trajectory),
        "trajectory_sha256": sha256(trajectory),
        "nas_root": str(source),
        "session_id": session,
        "agent_exit_status": result.get("agent_exit_status", result["exit_status"]),
        "capture_status": result["capture_status"],
        "submission_status": record["submission_status"],
        "responses": len(responses),
        "tool_actions": len(commands),
        "tool_observations": len(observations),
        "format_errors": format_errors,
        "repeated_commands": {c: n for c, n in Counter(commands).items() if n > 1},
        "protocol_ok": len(commands) == len(observations) and format_errors == 0,
        "coverage": coverage,
        "capture_complete": not missing and result["capture_status"] == "complete",
        "raw_events": raw_count,
        "excluded_non_reasoning_events": raw_count - len(selected),
        "site_count": len(sites),
        "all_paired_arrays_finite": True,
        "verified_shards": checked_shards,
        "graph_modes": dict(Counter(row["graph_execution"] for row in selected)),
        "batch_kinds": dict(Counter(row["batch_kind"] for row in selected)),
        "export_failures": dict(Counter(read_json(export_failures))) if export_failures.exists() else {},
        "inspected_at": now(),
    }
    write_json(out / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=float, default=86400)
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = load_snapshot(root, args.run_id)
    out = root / "hidden/reports" / args.run_id / "first-rollout"
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / "execution.lock", nonblocking=True):
        if (out / "result.json").exists():
            print(json.dumps(read_json(out / "result.json")), flush=True)
            return
        task_ids = {t["instance_id"] for t in cfg["selection"]["tasks"].values()}
        first = wait_first(root, args.run_id, task_ids, args.timeout)
        write_json(
            out / "selected.json",
            {"trigger": "inotify or pre-existing completion backlog", "selected_at": now(), **first},
        )
        print("FIRST_ROLLOUT", first["instance_id"], first["finished_at"], flush=True)
        audit = inspect_first(cfg, first, out)
        print(
            "AUDIT",
            json.dumps({k: audit[k] for k in ["responses", "format_errors", "coverage", "capture_status"]}),
            flush=True,
        )
        command = [
            str(root / ".venvs/latent-vllm/bin/python"),
            "-m",
            "hidden.evaluation.replay_online_artifacts",
            "--source-run",
            args.run_id,
            "--online-dir",
            audit["nas_root"],
            "--event-index",
            str(out / "paired-events.json"),
            "--extra-prefixes",
            str(out / "missing-prefixes.json"),
            "--output",
            str(out / "replay"),
        ]
        options = {
            "gpus": cfg["serving"]["gpus"],
            "tensor_parallel_size": cfg["serving"]["tensor_parallel_size"],
            "max_num_seqs": 2,
            "max_inflight_requests": 2,
            "max_num_batched_tokens": 512,
            "gpu_memory_utilization": 0.12,
            "kv_cache_memory_bytes": 2147483648,
            "allow_shared_gpus": True,
            "min_free_gpu_bytes": 15032385536,
            "capture_budget_bytes": 16777216,
        }
        # Replay only needs the exact recorded prefixes plus one generated token.
        # Advertising the serving context limit would reserve KV for 256k even
        # when the first completed trajectory has only a few thousand tokens.
        pairs = read_json(out / "paired-events.json")
        missing = read_json(out / "missing-prefixes.json")
        options["max_model_len"] = max(
            [row["absolute_position"] + 2 for row in pairs]
            + [len(row["prefix_token_ids"]) + 1 for row in missing]
            + [2]
        )
        for key, value in options.items():
            command += ["--set", "hidden.vllm." + key + "=" + json.dumps(value)]
        write_json(
            out / "replay-launch.json",
            {"command": command, "source_config_hash": cfg["config_hash"], "resource_settings": options},
        )
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        with (out / "replay.log").open("w") as log:
            child = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=environment)
        report_path = out / "replay/report.json"
        report = read_json(report_path) if report_path.exists() else {}
        result = {
            "instance_id": first["instance_id"],
            "rollout_id": first["rollout_id"],
            "protocol_ok": audit["protocol_ok"],
            "capture_complete": audit["capture_complete"],
            "numerical_passed": child.returncode == 0 and report.get("passed", False),
            "paired_points": len(read_json(out / "paired-events.json")),
            "missing_online_points": len(read_json(out / "missing-prefixes.json")),
            "offline_only_missing_points": report.get("offline_only_missing_points"),
            "phases": {
                key: {k: v[k] for k in ["passed", "worst_cosine", "worst_relative_l2"]}
                for key, v in report.get("phases", {}).items()
            },
            "replay_exit_code": child.returncode,
            "finished_at": now(),
        }
        result["overall_passed"] = (
            result["protocol_ok"] and result["capture_complete"] and result["numerical_passed"]
        )
        write_json(out / "result.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
