"""Per-GPU TP=1 serving pool for trajectory-level rollout scheduling."""

from __future__ import annotations

import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from core.cancellation import policy
from core.io import now, write_json
from runtime.serving.manager import ServingManager


class ServingWorker:
    def __init__(self, pool, index, gpu, cfg, endpoint, manager, output):
        self.pool = pool
        self.index = index
        self.gpu = gpu
        self.cfg = cfg
        self.endpoint = endpoint
        self.manager = manager
        self.output = output
        self.active = 0

    def runtime(self):
        value = dict(self.manager.runtime)
        value.update({
            "serving_worker_id": self.index,
            "serving_worker_gpu": self.gpu,
            "serving_worker_endpoint": self.endpoint,
            "serving_cfg": self.cfg,
            "serving_pool": self.pool,
            "startup_output": str(self.pool.output.parent / "startup" / f"worker-{self.index:02d}"),
            "startup_suffix": f"-worker-{self.index:02d}",
        })
        return value


class ServingPool:
    """Own one managed service per GPU and dispatch complete trajectories to it.

    The pool deliberately does not split a trajectory's parallel branches across
    services. A worker reservation lasts until that trajectory's attempt returns,
    which makes capture/session state and per-request endpoint identity stable.
    """

    def __init__(self, cfg, output, run_id):
        self.cfg = cfg
        self.output = Path(output)
        self.run_id = run_id
        self.workers = []
        self._lock = threading.RLock()
        self._entered = False
        self._startup_cancel = threading.Event()

    @property
    def enabled(self):
        return self.cfg["serving"].get("trajectory_sharding", "none") == "per_gpu_tp1"

    @property
    def max_cpu_workers_per_gpu(self):
        return int(self.cfg["serving"].get("max_cpu_workers_per_gpu", 1))

    @property
    def capacity(self):
        return len(self.workers) * self.max_cpu_workers_per_gpu

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _enter(self):
        if self._entered:
            raise RuntimeError("ServingPool cannot be entered twice")
        self.output.mkdir(parents=True, exist_ok=True)
        serving = self.cfg["serving"]
        if not self.enabled:
            manager = ServingManager(self.cfg, self.output, run_id=self.run_id)
            manager.__enter__()
            self.workers.append(ServingWorker(
                self, 0, serving["gpus"][0], self.cfg, serving["base_url"], manager, self.output
            ))
        else:
            for index, (gpu, endpoint) in enumerate(zip(serving["gpus"], serving["replica_base_urls"])):
                worker_cfg = copy.deepcopy(self.cfg)
                worker_cfg["serving"]["gpus"] = [gpu]
                worker_cfg["serving"]["tensor_parallel_size"] = 1
                worker_cfg["serving"]["base_url"] = endpoint
                worker_cfg["serving"]["trajectory_sharding"] = "none"
                # Pool topology is validated before reducing each worker to one MIG.
                worker_cfg["serving"]["startup_mode"] = serving.get("startup_mode", "serial")
                worker_cfg["serving"]["replica_base_urls"] = []
                output = self.output / "workers" / f"worker-{index:02d}"
                manager = ServingManager(worker_cfg, output, run_id=self.run_id,
                                         capture_owner=self.workers[0].manager if self.workers else None)
                self.workers.append(ServingWorker(
                    self, index, gpu, worker_cfg, endpoint, manager, output
                ))
                if serving.get("startup_mode", "serial") == "serial":
                    manager.__enter__()
            if serving.get("startup_mode", "serial") == "parallel":
                # Prepare the owner first, then attach every replica to the same
                # initialized capture controller. All device locks precede spawn.
                for worker in self.workers:
                    worker.manager.pool_managed = True
                    worker.manager.startup_cancel = self._startup_cancel
                    worker.manager.prepare()
                with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                    futures = [executor.submit(worker.manager.__enter__) for worker in self.workers]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except BaseException:
                        self._startup_cancel.set()
                        for future in futures:
                            future.cancel()
                        # The executor joins startup threads before reverse-order
                        # cleanup closes replicas, followed by their capture owner.
                        raise
        self._entered = True
        self._write_receipt()
        return self

    def _write_receipt(self):
        write_json(self.output / "worker-pool.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "trajectory_sharding": self.enabled,
            "startup_mode": self.cfg["serving"].get("startup_mode", "serial"),
            "tensor_parallel_size": 1 if self.enabled else self.cfg["serving"]["tensor_parallel_size"],
            "max_cpu_workers_per_gpu": self.max_cpu_workers_per_gpu,
            "capacity": self.capacity,
            "state": "running" if self._entered else "stopped",
            "execution_attempt_id": os.environ.get("DYCON_EXECUTION_ATTEMPT"),
            "workers": [{
                "worker_id": worker.index,
                "gpu": worker.gpu,
                "base_url": worker.endpoint,
                "output": str(worker.output),
                "startup_timing": getattr(worker.manager, "startup_timing", {}),
                "active_cpu_workers": worker.active,
                "available_cpu_workers": self.max_cpu_workers_per_gpu - worker.active,
            } for worker in self.workers],
            "updated_at": now(),
        })

    def acquire(self):
        """Reserve the least-loaded GPU, or return None when all are full."""
        with self._lock:
            available = [w for w in self.workers if w.active < self.max_cpu_workers_per_gpu]
            if not available:
                return None
            worker = min(available, key=lambda item: (item.active, item.index))
            worker.active += 1
            self._write_receipt()
            return worker

    def release(self, worker):
        with self._lock:
            if worker.active <= 0:
                raise RuntimeError(f"Serving worker {worker.index} released without a reservation")
            worker.active -= 1
            self._write_receipt()

    def cancel(self, run_id, execution_attempt_id):
        errors = []
        for worker in self.workers[:1]:
            controller = worker.manager.runtime.get("capture_controller")
            if controller is not None:
                try:
                    controller.cancel_run(run_id)
                except Exception as exc:
                    errors.append({"worker_id": worker.index, "stage": "capture", "error": repr(exc)})
        def cancel_engine(worker):
            if (self.cfg['serving']['management'] != 'managed'
                    or not (self.cfg.get('capture', {}).get('enabled')
                            or worker.manager.runtime.get('request_control_enabled'))):
                return None
            try:
                endpoint = worker.endpoint.removesuffix("/v1").rstrip("/")
                requests.post(
                    endpoint + "/dycon/capture/engine/cancel",
                    json={"run_id": run_id, "execution_attempt_id": execution_attempt_id},
                    headers={"X-DyCon-Cancel-Token": os.environ.get("DYCON_CANCEL_TOKEN", "")},
                    timeout=policy(self.cfg)["control_request_seconds"],
                ).raise_for_status()
            except Exception as exc:
                return {"worker_id": worker.index, "stage": "engine", "error": repr(exc)}
            return None
        # One slow endpoint must not delay waking the other GPU workers.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, len(self.workers))) as executor:
            errors.extend(value for value in executor.map(cancel_engine, self.workers) if value is not None)
        return errors

    def __exit__(self, *exc):
        errors = []
        self._startup_cancel.set()
        for worker in reversed(self.workers):
            try:
                worker.manager.__exit__(*exc)
            except Exception as error:
                errors.append(error)
        for worker in self.workers:
            worker.active = 0
        self._entered = False
        if self.workers:
            self._write_receipt()
        self.workers.clear()
        if errors:
            raise errors[0]
