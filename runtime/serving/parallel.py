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

    Decisions reserve all their slots atomically before generation. Discarded
    results cannot recreate a released decision. Expiry bounds abandoned groups.
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

    def reserve(self, decision_id, branches):
        with self.condition:
            if self.closed:
                raise RuntimeError("Scoring broker closed")
            if not isinstance(decision_id, str) or not decision_id:
                raise ValueError("Invalid decision identity")
            if any(type(b) is not int or b < 0 for b in branches):
                raise ValueError("Invalid branch identity")
            now = time.monotonic()
            for key in list(self.groups):
                if self.groups[key]["expires"] < now:
                    del self.groups[key]
            if (len(branches) != len(set(branches)) or not branches
                    or len(branches) > self.cfg["branch_count"]):
                raise ValueError("Invalid scoring group")
            if decision_id in self.groups:
                raise ValueError("Decision already reserved")
            if not self.condition.wait_for(lambda: sum(len(g["branches"]) for g in self.groups.values())
                                           + len(branches) <= self.cfg["max_inflight_branches"],
                                           self.cfg.get("admission_timeout_seconds", 1800)):
                raise RuntimeError("Parallel scoring capacity exhausted")
            self.groups[decision_id] = {"branches": set(branches), "results": {},
                                       "expires": time.monotonic() + self.cfg.get("decision_ttl_seconds", 28860)}
            return {"reserved": True}

    def submit(self, metadata, request_id, position, token_id, arrays):
        decision, branch = metadata.get("decision_id"), metadata.get("branch_id")
        with self.condition:
            group = self.groups.get(decision)
            if self.closed or group is None or branch not in group["branches"] or branch in group["results"]:
                return
            identity = {"branch_id": branch, "native_request_id": request_id,
                        "absolute_position": position, "token_id": token_id,
                        "sample_id": hash_json({"session": metadata["session_id"], "request": request_id,
                                                "anchor": "think_close", "position": position})[:32]}
            group["results"][branch] = {**identity, "status": "pending"}
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
                if self.groups.get(decision) is not reserved_group or self.closed:
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
                if group is reserved_group:
                    group["results"][branch] = result
                self.outstanding -= 1
                self.condition.notify_all()

        self.jobs.put_nowait(predict)

    def results(self, decision_id, timeout):
        if not 0 <= timeout <= 30:
            raise ValueError("Invalid score wait timeout")
        with self.condition:
            if decision_id not in self.groups:
                raise KeyError("Unknown scoring decision")
            group = self.groups[decision_id]
            self.condition.wait_for(lambda: all(
                group["results"].get(b, {}).get("status") in {"scored", "failed", "unscored"}
                for b in group["branches"]), timeout)
            return {"results": [dict(group["results"].get(b, {"branch_id": b, "status": "missing"}))
                                for b in sorted(group["branches"])],
                    "estimator_sha256": self.cfg["estimator_sha256"],
                    "target": self.contract["target"], "endpoint": self.contract["endpoint"]}

    def release(self, decision_id):
        with self.condition:
            self.groups.pop(decision_id, None)
            self.condition.notify_all()
            return {"released": True}

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
