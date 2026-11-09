"""Read-only per-run, per-GPU live generation throughput from vLLM metrics."""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from core.io import read_json
from application.execution.run_plan import load_plan
from application.execution.status import active_run_statuses

_METRIC = re.compile(r"^(?P<name>vllm:generation_tokens_total)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$")


def _read(url: str, timeout: float) -> float:
    endpoint = url.rstrip("/").removesuffix("/v1") + "/metrics"
    request = Request(endpoint, headers={"Accept": "text/plain"})
    with urlopen(request, timeout=timeout) as response:
        text = response.read(4 * 1024 * 1024).decode("utf-8", "replace")
    values = []
    for line in text.splitlines():
        match = _METRIC.match(line)
        if match:
            try:
                values.append(float(match.group("value")))
            except ValueError:
                pass
    if not values:
        raise RuntimeError("vLLM generation token counter is missing")
    return sum(values)


def query_active_gpu_token_speed(root, *, window_seconds: float = 2.0, timeout: float = 3.0) -> dict:
    root = str(root)
    active = active_run_statuses(root).get("runs", [])
    targets = []
    runs = []
    for row in active:
        run_id = row["run_id"]
        run_dir = Path(root) / "runs" / run_id
        cfg = load_plan(root, run_id)["config"]
        serving = cfg["serving"]
        pool_path = run_dir / "serving" / "worker-pool.json"
        status = read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
        pool = read_json(pool_path) if pool_path.exists() else None
        is_current_pool = bool(pool and pool.get("state") == "running" and
                               (not status.get("execution_attempt_id") or
                                pool.get("execution_attempt_id") == status.get("execution_attempt_id")))
        gpu_rows = []
        if is_current_pool and pool.get("trajectory_sharding"):
            for worker in pool.get("workers", []):
                gpu_rows.append({"gpu": worker["gpu"], "worker_id": worker.get("worker_id"),
                                 "endpoint": worker["base_url"], "status": "measuring"})
        elif is_current_pool and pool.get("workers"):
            worker = pool["workers"][0]
            gpus = serving.get("gpus", [worker.get("gpu")])
            if len(gpus) == 1:
                gpu_rows.append({"gpu": gpus[0], "worker_id": worker.get("worker_id"),
                                 "endpoint": worker["base_url"], "status": "measuring"})
            else:
                gpu_rows.extend({"gpu": gpu, "endpoint": worker["base_url"],
                                 "status": "shared_tensor_parallel", "tokens_per_second": None}
                                for gpu in gpus)
        else:
            gpu_rows.extend({"gpu": gpu, "status": "serving_pool_not_running",
                             "tokens_per_second": None} for gpu in serving.get("gpus", []))
        runs.append({"run_id": run_id, "gpus": gpu_rows})
        for gpu_row in gpu_rows:
            if gpu_row["status"] == "measuring":
                targets.append((run_id, gpu_row, gpu_row["endpoint"]))
    before = {}
    errors = {}
    for run_id, gpu_row, endpoint in targets:
        key = (run_id, gpu_row["gpu"], endpoint)
        try:
            before[key] = _read(endpoint, timeout)
        except Exception as exc:
            errors[key] = str(exc)[:240]
    started = time.monotonic()
    time.sleep(window_seconds)
    elapsed = max(time.monotonic() - started, 1e-9)
    for run in runs:
        for gpu_row in run["gpus"]:
            if gpu_row["status"] != "measuring":
                continue
            key = (run["run_id"], gpu_row["gpu"], gpu_row["endpoint"])
            try:
                after = _read(gpu_row["endpoint"], timeout)
                if key in errors:
                    raise RuntimeError(errors[key])
                delta = max(0.0, after - before[key])
                gpu_row.update(status="ok", generated_tokens=int(delta) if delta.is_integer() else delta,
                               elapsed_seconds=elapsed, tokens_per_second=delta / elapsed)
            except Exception as exc:
                gpu_row.update(status="metrics_unavailable", error=str(exc)[:240], tokens_per_second=None)
            gpu_row.pop("endpoint", None)
    return {"schema_version": 1, "measurement_window_seconds": elapsed,
            "metric": "vllm:generation_tokens_total", "runs": runs}
