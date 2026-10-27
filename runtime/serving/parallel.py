"""Graph-external scoring contract and bounded per-decision result delivery.

The provider owns model loading/preprocessing/prediction. No estimator is
implemented here. Tensor data never crosses the HTTP result interface.
"""

from __future__ import annotations

import importlib
import queue
import threading
import time
from pathlib import Path

import numpy as np

from core.io import hash_json, owned_path, read_json, sha256
from core.cancellation import Cancelled


def artifact_contract(path, expected_sha=None, model_fingerprint=None):
    path = Path(path)
    if expected_sha is not None and sha256(path) != expected_sha:
        raise ValueError("Parallel estimator manifest changed")
    value = read_json(path)
    required = {"schema_version", "loader", "model_fingerprint", "sites", "hidden_size",
                "target", "endpoint", "files"}
    if not required <= value.keys() or value["schema_version"] != 1:
        raise ValueError("Invalid external predictor contract")
    if model_fingerprint is not None and value["model_fingerprint"] != model_fingerprint:
        raise ValueError("Parallel estimator checkpoint mismatch")
    sites = value["sites"]
    if (not isinstance(sites, list) or not sites or len(set(sites)) != len(sites)
            or any(type(x) is not int or x < 0 for x in sites)
            or type(value["hidden_size"]) is not int or value["hidden_size"] <= 0
            or not isinstance(value["files"], dict) or not value["files"]):
        raise ValueError("Invalid predictor features/files")
    if not all(isinstance(value[k], str) and value[k] for k in ("loader", "target", "endpoint")):
        raise ValueError("Invalid predictor identity")
    module, separator, name = value["loader"].partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError("Predictor loader must be module:callable")
    for relative, digest in value["files"].items():
        if sha256(owned_path(path.parent, relative)) != digest:
            raise ValueError("Parallel estimator file changed: " + relative)
    for module, digest in value.get("implementation_sha256", {}).items():
        implementation = importlib.import_module(module)
        if not getattr(implementation, "__file__", None) or sha256(implementation.__file__) != digest:
            raise ValueError("Online predictor implementation changed: " + module)
    return value


class ScoringBroker:
    """One bounded CPU worker; a stuck predictor cannot accumulate more work.

    Decisions retain their complete branch evidence, while dynamic admission
    can acquire and release native branch slots independently. Expiry bounds
    abandoned groups.
    """

    def __init__(self, cfg, model_fingerprint, num_layers):
        self.cfg = cfg
        self.contract = artifact_contract(cfg["estimator_ref"], cfg["estimator_sha256"], model_fingerprint)
        contract = self.contract
        if any(site >= num_layers for site in contract["sites"]):
            raise ValueError("Predictor site outside checkpoint")
        self.layer_names = [f"layer_{num_layers if x == num_layers - 1 else x}"
                            for x in contract["sites"]]
        module, name = contract["loader"].split(":")
        self.predictor = getattr(importlib.import_module(module), name)(
            Path(cfg["estimator_ref"]).parent, contract)
        if not callable(getattr(self.predictor, "predict", None)):
            raise ValueError("External loader must return an object with predict(batch)")
        self.condition = threading.Condition()
        self.groups = {}
        self.jobs = queue.Queue(maxsize=cfg["max_inflight_branches"])
        self.outstanding = 0
        self.closed = False
        self.cancelled = {}
        self.run_cancelled = False
        self.peak_active = 0
        self.saturated_waiting_peak = 0
        threading.Thread(target=self._run, name="dycon-score", daemon=True).start()

    def _run(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                job()
            finally:
                self.jobs.task_done()

    def status(self):
        with self.condition:
            active = sum(len(group['active']) for group in self.groups.values())
            pending = sum(len(group.get('pending', ())) for group in self.groups.values())
            self.peak_active = max(self.peak_active, active)
            if active >= self.cfg['max_inflight_branches']:
                self.saturated_waiting_peak = max(self.saturated_waiting_peak, pending)
            return {'active': active, 'pending': pending, 'decisions': len(self.groups),
                    'capacity': self.cfg['max_inflight_branches'], 'peak_active': self.peak_active,
                    'saturated_waiting_peak': self.saturated_waiting_peak, 'cancelled': self.run_cancelled}

    def check_cancelled(self, decision_id):
        if self.run_cancelled or decision_id in self.cancelled:
            raise Cancelled("Parallel decision cancelled")

    def cancel(self, decision_id=None):
        with self.condition:
            if decision_id is None:
                self.run_cancelled = True
            decisions = list(self.groups) if decision_id is None else [decision_id]
            for key in decisions:
                self.cancelled[key] = time.monotonic() + self.cfg.get('decision_ttl_seconds', 28860)
                group = self.groups.get(key)
                if group is not None:
                    group['cancelled'] = True
                    # Active native work retains its lease until stream terminal.
                    group.get('pending', set()).clear()
            self.condition.notify_all()
            return {'cancel_requested': True, 'decision_ids': decisions}

    def reserve(self, decision_id, branches, decision_deadline_unix=None):
        with self.condition:
            self.check_cancelled(decision_id)
            if self.closed:
                raise RuntimeError("Scoring broker closed")
            if not isinstance(decision_id, str) or not decision_id:
                raise ValueError("Invalid decision identity")
            if any(type(b) is not int or b < 0 for b in branches):
                raise ValueError("Invalid branch identity")
            now = time.monotonic()
            self.cancelled = {key: expiry for key, expiry in self.cancelled.items() if expiry > now or key in self.groups}
            for key in list(self.groups):
                if self.groups[key]["expires"] < now:
                    del self.groups[key]
            def active_branches():
                return sum(len(group["active"]) for group in self.groups.values())
            if (len(branches) != len(set(branches)) or not branches
                    or len(branches) > self.cfg["branch_count"]):
                raise ValueError("Invalid scoring group")
            if decision_id in self.groups:
                raise ValueError("Decision already reserved")
            ttl = self.cfg.get("decision_ttl_seconds", 28860)
            if decision_deadline_unix is not None:
                if type(decision_deadline_unix) not in {int, float} or not np.isfinite(decision_deadline_unix):
                    raise ValueError("Invalid parallel decision deadline")
                ttl = min(ttl, decision_deadline_unix - time.time())
                if ttl <= 0:
                    raise ValueError("Parallel decision deadline exceeded")
            dynamic = bool(self.cfg.get("dynamic_branch_admission", False))
            if not dynamic and not self.condition.wait_for(
                lambda: self.closed or self.run_cancelled or decision_id in self.cancelled or active_branches() + len(branches) <= self.cfg["max_inflight_branches"],
                min(self.cfg.get("admission_timeout_seconds", 1800), ttl),
            ):
                raise RuntimeError("Parallel scoring capacity exhausted")
            self.check_cancelled(decision_id)
            if self.closed:
                raise RuntimeError("Scoring broker closed")
            if time.monotonic() >= now + ttl:
                raise ValueError("Parallel decision deadline exceeded")
            self.groups[decision_id] = {
                "branches": set(branches),
                "pending": set(branches) if dynamic else set(),
                "active": set() if dynamic else set(branches),
                "results": {},
                "expires": now + ttl,
            }
            return {"reserved": True}

    def mark_terminated(self, metadata, request_id, evidence):
        decision, branch = metadata["decision_id"], metadata["branch_id"]
        with self.condition:
            group = self.groups.get(decision)
            if self.closed or group is None or group["expires"] <= time.monotonic() or branch not in group["branches"]:
                return
            group["results"][branch] = {
                "branch_id": branch, "native_request_id": request_id,
                "api_attempt_id": metadata["api_attempt_id"], "status": "failed",
                "reason": "reasoning_loop", "reasoning_loop": dict(evidence),
            }
            group["active"].discard(branch)
            self.condition.notify_all()

    def acquire_branch(self, decision_id, branch_id):
        """Wait for one branch slot without reserving the other branch slots."""
        with self.condition:
            self.check_cancelled(decision_id)
            initial = self.groups.get(decision_id)
            if initial is None:
                raise RuntimeError("Unknown parallel decision branch")
            remaining = initial["expires"] - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Parallel decision deadline exceeded")
            def ready():
                self.status()
                group = self.groups.get(decision_id)
                return self.closed or self.run_cancelled or decision_id in self.cancelled or group is None or branch_id not in group.get("pending", set()) or sum(len(g["active"]) for g in self.groups.values()) < self.cfg["max_inflight_branches"]
            if not self.condition.wait_for(ready, min(self.cfg.get("admission_timeout_seconds", 1800), remaining)):
                raise RuntimeError("Parallel scoring capacity exhausted")
            self.check_cancelled(decision_id)
            group = self.groups.get(decision_id)
            if self.closed or group is None or branch_id not in group["branches"]:
                raise RuntimeError("Unknown parallel decision branch")
            if group["expires"] <= time.monotonic():
                raise RuntimeError("Parallel decision deadline exceeded")
            if branch_id in group["active"]:
                return {"acquired": True, "already_acquired": True}
            if branch_id not in group.get("pending", set()):
                if branch_id in group.get("results", {}):
                    return {"acquired": False, "already_released": True}
                raise RuntimeError("Parallel branch is not pending")
            group["pending"].remove(branch_id)
            group["active"].add(branch_id)
            self.status()
            self.condition.notify_all()
            return {"acquired": True, "active_branches": sum(len(g["active"]) for g in self.groups.values())}

    def release_branch(self, metadata, request_id=None, reason="native_output"):
        """Release one vLLM sequence slot while retaining decision evidence."""
        decision, branch = metadata.get("decision_id"), metadata.get("branch_id")
        with self.condition:
            group = self.groups.get(decision)
            if self.closed or group is None or branch not in group["branches"]:
                return {"released": False, "reason": "unknown_branch"}
            if branch in group.get("pending", set()):
                group["pending"].remove(branch)
            elif branch in group["active"]:
                group["active"].remove(branch)
            else:
                return {"released": False, "reason": "already_released"}
            group.setdefault("release_reasons", {})[str(branch)] = reason
            if request_id is not None:
                group.setdefault("native_request_ids", {})[str(branch)] = request_id
            self.condition.notify_all()
            return {"released": True, "active_branches": sum(len(g["active"]) for g in self.groups.values())}

    def submit(self, metadata, request_id, position, token_id, arrays):
        decision, branch = metadata.get("decision_id"), metadata.get("branch_id")
        with self.condition:
            group = self.groups.get(decision)
            if (self.closed or self.run_cancelled or decision in self.cancelled or group is None or group["expires"] <= time.monotonic()
                    or branch not in group["branches"] or branch in group["results"]):
                return
            identity = {"branch_id": branch, "native_request_id": request_id,
                        **{k: metadata[k] for k in ("session_id", "decision_id", "api_attempt_id", "step_id", "request_sha256") if k in metadata},
                        "absolute_position": position, "token_id": token_id,
                        "sample_id": hash_json({"session": metadata["session_id"], "request": request_id,
                                                "anchor": "think_close", "position": position})[:32]}
            group["results"][branch] = {**identity, "status": "pending"}
            pending = group["results"][branch]
            if metadata.get("diagnostic_score_disabled"):
                group["results"][branch] = {**identity, "status": "unscored"}
                self.condition.notify_all()
                return
            if self.outstanding >= self.cfg["max_inflight_branches"]:
                group["results"][branch] = {**identity, "status": "failed", "reason": "scorer_busy"}
                self.condition.notify_all()
                return
            try:
                features = np.stack([np.array(arrays[name], copy=True) for name in self.layer_names])
                if features.shape != (len(self.layer_names), self.contract["hidden_size"]) or not np.isfinite(features).all():
                    raise ValueError("Invalid online predictor features")
            except (KeyError, ValueError) as exc:
                group["results"][branch] = {**identity, "status": "failed", "reason": str(exc)}
                self.condition.notify_all()
                return
            self.outstanding += 1
            reserved_group = group

        queued_at = time.monotonic()

        def predict():
            start = time.monotonic()
            with self.condition:
                if self.groups.get(decision) is not reserved_group or self.closed or self.run_cancelled or decision in self.cancelled:
                    self.outstanding -= 1
                    return
            try:
                scores = np.asarray(self.predictor.predict(features[None, ...]), dtype=float)
                if scores.shape != (1,) or not np.isfinite(scores).all():
                    raise ValueError("Predictor must return finite [N] scores")
                result = {**identity, "status": "scored", "score": float(scores[0])}
            except Exception as exc:
                result = {**identity, "status": "failed", "reason": repr(exc)}
            result["predict_ms"] = (time.monotonic() - start) * 1000
            result["queue_ms"] = (start - queued_at) * 1000
            with self.condition:
                group = self.groups.get(decision)
                if group is reserved_group and not group.get("cancelled") and group["results"].get(branch) is pending:
                    group["results"][branch] = result
                self.outstanding -= 1
                self.condition.notify_all()

        self.jobs.put_nowait(predict)

    def results(self, decision_id, timeout):
        if not 0 <= timeout <= 30:
            raise ValueError("Invalid score wait timeout")
        with self.condition:
            self.check_cancelled(decision_id)
            if decision_id not in self.groups:
                raise KeyError("Unknown scoring decision")
            group = self.groups[decision_id]
            remaining = group["expires"] - time.monotonic()
            if remaining <= 0:
                raise ValueError("Parallel decision deadline exceeded")
            self.condition.wait_for(lambda: self.closed or self.run_cancelled or decision_id in self.cancelled or self.groups.get(decision_id) is not group or all(
                group["results"].get(b, {}).get("status") in {"scored", "failed", "unscored"}
                for b in group["branches"]), min(timeout, remaining))
            self.check_cancelled(decision_id)
            if self.closed or self.groups.get(decision_id) is not group:
                raise RuntimeError("Scoring decision released")
            if group["expires"] <= time.monotonic():
                raise ValueError("Parallel decision deadline exceeded")
            return {"results": [dict(group["results"].get(b, {"branch_id": b, "status": "missing"}))
                                for b in sorted(group["branches"])],
                    "estimator_sha256": self.cfg["estimator_sha256"],
                    "target": self.contract["target"], "endpoint": self.contract["endpoint"]}

    def release(self, decision_id):
        with self.condition:
            group = self.groups.pop(decision_id, None)
            self.condition.notify_all()
            return {"released": group is not None}

    def close(self):
        with self.condition:
            self.closed = True
            self.groups.clear()
            self.condition.notify_all()
        while True:
            try:
                self.jobs.get_nowait()
                self.jobs.task_done()
            except queue.Empty:
                break
        self.jobs.put_nowait(None)
