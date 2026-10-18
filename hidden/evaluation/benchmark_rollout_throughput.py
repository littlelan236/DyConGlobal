"""Benchmark native coding-agent rollout throughput with optional online capture."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import time
from multiprocessing.connection import wait as wait_connections

from baseline.catalog import Catalog
from baseline.rollout.runner import run_task
from core.config import resolve
from core.io import now, read_json, sha256, write_json
from core.model_identity import verify_model
from runtime.containers.factory import ContainerFactory
from runtime.serving.manager import ServingManager


MODEL = "/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B"
FINGERPRINT = "f97643c15ea3cb04517379dd13482b48548458c3a3f6c3a7eb93dcaa6adf86a2"


def _worker_process(cfg, run_id, connection):
    try:
        config_dir = Path(cfg["project_root"]) / "state" / "miniswe-config"
        os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(config_dir)
        os.environ["MSWEA_SILENT_STARTUP"] = "1"
        while True:
            assignment = connection.recv()
            if assignment is None:
                return
            task_key, task = assignment
            try:
                if (config_dir / ".env").exists():
                    raise ValueError("mini-swe-agent .env overrides are disabled for benchmark workers")
                task_cfg = json.loads(json.dumps(cfg))
                task_cfg["benchmark_task_id"] = task_key
                factory = ContainerFactory(task_cfg)
                result = run_task(task_cfg, run_id, task_key, task, factory)
                connection.send({"task_key": task_key, "result": result})
            except BaseException as exc:
                connection.send({"task_key": task_key, "error": repr(exc)})
    finally:
        connection.close()


def _trajectory_stats(root: Path, result: dict) -> dict:
    rid = result.get("rollout_id")
    row = Catalog(root).read()["rollouts"].get(rid, {}) if rid else {}
    path = root / row["trajectory_path"] if row else None
    steps = 0
    tokens = 0
    responses = 0
    if path and path.exists():
        data = read_json(path)
        for message in data.get("messages", []):
            if message.get("role") != "assistant":
                continue
            steps += 1
            response = (message.get("extra") or {}).get("response") or {}
            responses += 1 if response else 0
            output_ids = response.get("output_token_ids")
            if isinstance(output_ids, list):
                tokens += len(output_ids)
            else:
                usage = response.get("usage") or {}
                tokens += int(usage.get("completion_tokens") or 0)
    return {"rollout_id": rid, "instance_id": result.get("instance_id"),
            "exit_status": result.get("exit_status"), "steps": steps,
            "completion_tokens": tokens, "responses": responses,
            "finished_at": result.get("finished_at")}


def run_group(cfg, tasks, run_id, *, capture_enabled: bool, port: int, output: Path) -> dict:
    group_cfg = json.loads(json.dumps(cfg))
    group_cfg["serving"].update(
        management="managed", base_url=f"http://127.0.0.1:{port}/v1",
        gpus=[4, 5, 6, 7], tensor_parallel_size=4, dtype="float16",
        max_num_seqs=12, max_num_batched_tokens=4096,
        max_model_len=32768, gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=4294967296, tool_parser="hermes", reasoning_parser="qwen3",
        startup_seconds=1800,
    )
    group_cfg["execution"].update(rollout_workers=12, infrastructure_retry_limit=0)
    group_cfg["generation"].update(temperature=0.6, top_p=0.95, top_k=20,
                                    max_tokens=8192, seed=42)
    group_cfg["capture"].update(enabled=capture_enabled, layers=list(range(32)),
                                  anchors=["reasoning_newline", "think_close"],
                                  prefill_token_chunk_size=0,
                                  export={"enabled": True, "root": str(output / "capture"),
                                          "queue_size": 512, "flush_interval_ms": 50})
    if not capture_enabled:
        group_cfg["capture"] = {**group_cfg["capture"], "enabled": False}
    verify_model(group_cfg)
    metric_dir = output / "request-metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    group_cfg["benchmark_metrics_dir"] = str(metric_dir)
    factory = ContainerFactory(group_cfg)
    factory.lock_images([task["instance"] for task in tasks.values()])
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    first_idle = {"at": None, "task_key": None}
    processes = {}
    parents = {}
    cancelled = []
    with ServingManager(group_cfg, output / "serving") as serving:
        if capture_enabled:
            group_cfg["capture_control_url"] = serving.capture_server.base_url
            group_cfg["capture_token_inventory"] = serving.capture_token_inventory
        context = multiprocessing.get_context("spawn")
        task_items = list(tasks.items())
        next_task = 0

        def launch(worker_id, task_key, task):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(target=_worker_process,
                args=(group_cfg, run_id, child), name=f"rollout-worker-{worker_id:02d}")
            process.start()
            child.close()
            parent.send((task_key, task))
            processes[process.sentinel] = (process, task_key)
            parents[parent] = (process, task_key)

        for worker_id in range(min(12, len(task_items))):
            task_key, task = task_items[next_task]
            next_task += 1
            launch(worker_id, task_key, task)
        launched = time.monotonic()
        pending = set(parents)
        while pending:
            ready = wait_connections(pending)
            for connection in ready:
                process, task_key = parents[connection]
                payload = connection.recv()
                if "result" in payload:
                    results[task_key] = payload["result"]
                else:
                    results[task_key] = {"instance_id": tasks[task_key]["instance_id"],
                                         "exit_status": "WorkerFailure", "error": payload["error"]}
                if next_task < len(task_items):
                    next_key, next_value = task_items[next_task]
                    next_task += 1
                    parents[connection] = (process, next_key)
                    connection.send((next_key, next_value))
                elif first_idle["at"] is None:
                    # Every task has been assigned; this completed worker found
                    # the queue empty and became idle. End the group here.
                    first_idle["at"] = time.monotonic()
                    first_idle["task_key"] = task_key
                    connection.send(None)
                    pending.remove(connection)
            if first_idle["at"] is not None:
                # Preserve any other results that became ready at the same
                # instant, then stop workers still processing assigned tasks.
                for other in list(pending):
                    other_process, other_key = parents[other]
                    if other_process.is_alive():
                        other_process.terminate()
                    cancelled.append(other_key)
                    other.close()
                    pending.remove(other)
                break
        for connection, (process, _) in parents.items():
            connection.close()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
        for process, _ in processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
        cutoff = first_idle["at"] or time.monotonic()
        active_records = []
        for path in metric_dir.glob("*.json"):
            row = read_json(path)
            if row["completed_monotonic_ns"] <= int(cutoff * 1_000_000_000):
                active_records.append(row)
        active_start = min((row["started_monotonic_ns"] for row in active_records),
                           default=int(launched * 1_000_000_000)) / 1_000_000_000
        active_elapsed = max(cutoff - active_start, 0.001)
        full_run_elapsed = cutoff - launched
        if capture_enabled and serving.capture_controller is not None:
            serving.capture_controller.queue.queue.join()
        status = (requests_status(serving) if capture_enabled else None)
    factory.cleanup_owned(run_id)
    rows = [_trajectory_stats(Path(group_cfg["project_root"]), result)
            for result in results.values() if result.get("rollout_id")]
    completed_rows = [row for row in rows if row["exit_status"] in {"Submitted", "LimitsExceeded",
                                                                     "ContextWindowExceededError", "RepeatedFormatError", "TimeExceeded"}]
    total_tokens = sum(row["completion_tokens"] for row in completed_rows)
    total_steps = sum(row["steps"] for row in completed_rows)
    report = {
        "run_id": run_id, "capture_enabled": capture_enabled, "task_count": len(tasks),
        "worker_count": 12, "serving_max_num_seqs": 12, "gpus": [4, 5, 6, 7],
        "dtype": "float16", "model": MODEL, "model_fingerprint": FINGERPRINT,
        "started_at": now(), "elapsed_seconds_until_queue_exhausted_worker_idle": full_run_elapsed,
        "active_window_seconds_until_queue_exhausted_worker_idle": active_elapsed,
        "first_idle_task_key": first_idle["task_key"],
        "assigned_task_count": next_task,
        "completed_task_count_including_failures": len(results),
        "terminated_or_unfinished_task_count": len(tasks) - len(results),
        "terminated_or_unfinished_task_keys": [key for key in tasks if key not in results],
        "terminated_other_workers": len(cancelled), "cancelled_task_keys": cancelled,
        "completed_task_count": len(completed_rows), "total_completion_tokens": total_tokens,
        "completed_task_statuses": {key: value.get("exit_status") for key, value in results.items()},
        "total_steps": total_steps,
        "completed_model_requests_in_active_window": len(active_records),
        "active_window_completion_tokens": sum(row["completion_tokens"] for row in active_records),
        "active_window_steps": len(active_records),
        "all_tasks_tokens_per_second": None,
        "all_tasks_steps_per_minute": None,
        "completed_rollout_tokens_per_second": total_tokens / full_run_elapsed if full_run_elapsed else 0.0,
        "completed_rollout_steps_per_minute": total_steps / full_run_elapsed * 60 if full_run_elapsed else 0.0,
        "active_window_tokens_per_second": sum(row["completion_tokens"] for row in active_records) / active_elapsed if active_elapsed else 0.0,
        "active_window_request_steps_per_minute": len(active_records) / active_elapsed * 60,
        "task_results": rows, "capture_engine_status": status,
    }
    write_json(output / "report.json", report)
    return report


def requests_status(serving):
    import requests

    return requests.get(serving.cfg["serving"]["base_url"].replace("/v1", "") +
                        "/dycon/capture/engine/status", timeout=60).json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=16)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.capture_only and args.baseline_only:
        raise ValueError("capture-only and baseline-only are mutually exclusive")
    cfg = resolve(args.config, ["model.path=" + MODEL,
                                "model.fingerprint=" + FINGERPRINT,
                                "model.served_name=Qwen3.5-9B"])
    catalog = Catalog(cfg["project_root"])
    selected = catalog.select(partition="train", purposes=["training"])
    task_keys = sorted(selected["tasks"])
    if len(task_keys) < args.task_count:
        raise ValueError(f"Only {len(task_keys)} training tasks are available")
    tasks = {key: selected["tasks"][key] for key in task_keys[:args.task_count]}
    args.output.mkdir(parents=True, exist_ok=False)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    reports = {}
    if not args.capture_only:
        reports["baseline"] = run_group(cfg, tasks, f"speed-baseline-{stamp}", capture_enabled=False,
                                         port=8121, output=args.output / "baseline")
    if not args.baseline_only:
        reports["capture"] = run_group(cfg, tasks, f"speed-capture-{stamp}", capture_enabled=True,
                                        port=8122, output=args.output / "capture")
    write_json(args.output / "comparison.json", {
        "task_keys": list(tasks), "worker_count": 12, "max_num_seqs": 12,
        "baseline": reports.get("baseline"), "capture": reports.get("capture"),
        "speed_ratio_tokens_per_second": (
            reports["capture"]["active_window_tokens_per_second"] /
            reports["baseline"]["active_window_tokens_per_second"]
            if reports.get("baseline") and reports["baseline"]["active_window_tokens_per_second"] else None),
        "speed_ratio_steps_per_minute": (
            reports["capture"]["active_window_request_steps_per_minute"] /
            reports["baseline"]["active_window_request_steps_per_minute"]
            if reports.get("baseline") and reports["baseline"]["active_window_request_steps_per_minute"] else None),
    })


if __name__ == "__main__":
    main()
