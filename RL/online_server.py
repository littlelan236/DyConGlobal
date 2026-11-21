"""HTTP boundary for formal online rollout backends.

The server owns wire validation, request identity and idempotent receipts. A
backend must execute the frozen DyCon run work-items through project services
and return a sealed ``online_bootstrap_rollout_batch_v1`` payload; this module
never fabricates tokens, probabilities, evaluation labels or hidden vectors.
"""
from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Protocol

from core.io import now, write_json, sha256
from RL.online_runtime import ONLINE_ROLLOUT_SCHEMA, OnlineRolloutBatch, validate_trajectory_coverage

REQUEST_SCHEMA = ONLINE_ROLLOUT_SCHEMA


class OnlineRolloutBackend(Protocol):
    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def request_digest(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("request_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class OnlineRolloutService:
    def __init__(self, backend: OnlineRolloutBackend, receipt_root: str | Path):
        self.backend = backend
        self.root = Path(receipt_root)
        self.requests = self.root / "requests"
        self.responses = self.root / "responses"

    @staticmethod
    def _validate_request(payload: Mapping[str, Any]) -> str:
        if payload.get("schema") != REQUEST_SCHEMA:
            raise ValueError("unsupported online rollout request schema")
        if type(payload.get("step")) is not int or payload["step"] < 0:
            raise ValueError("online rollout request step must be a non-negative integer")
        if "max_inflight" in payload and (type(payload["max_inflight"]) is not int or payload["max_inflight"] < 1):
            raise ValueError("online max_inflight must be a positive integer")
        tasks = tuple(str(x) for x in payload.get("tasks", ()))
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("online rollout request tasks must be non-empty and unique")
        worker_ids = payload.get("worker_ids", ["shared"])
        if not isinstance(worker_ids, list) or not worker_ids or len(worker_ids) != len(set(worker_ids)):
            raise ValueError("online rollout request worker_ids must be non-empty and unique")
        items = payload.get("work_items")
        if not isinstance(items, list) or not items:
            raise ValueError("online rollout request work_items must be non-empty")
        ids, per_task = set(), {task: {"dycon": 0, "baseline": 0} for task in tasks}
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("online rollout work item must be an object")
            trajectory_id, task, source = item.get("trajectory_id"), item.get("task_id"), item.get("source")
            if not isinstance(trajectory_id, str) or not trajectory_id or trajectory_id in ids:
                raise ValueError("online rollout work item trajectory identity is invalid or duplicated")
            if task not in per_task or source not in {"dycon", "baseline"}:
                raise ValueError("online rollout work item task/source is outside request")
            if item.get("worker_id") not in worker_ids:
                raise ValueError("online rollout work item worker is outside request worker_ids")
            ids.add(trajectory_id)
            per_task[task][source] += 1
        if any(counts["dycon"] != 1 or counts["baseline"] < 1 for counts in per_task.values()):
            raise ValueError("each task requires one DyCon and at least one baseline work item")
        if len({counts["baseline"] for counts in per_task.values()}) != 1:
            raise ValueError("baseline repetitions must be equal for every task")
        if not all(isinstance(payload.get(key), str) and payload[key] for key in ("policy_version", "estimator_version")):
            raise ValueError("online request requires generating policy and estimator versions")
        expected = request_digest(payload)
        if payload.get("request_sha256") != expected:
            raise ValueError("online rollout request SHA does not match work manifest")
        return expected

    def handle(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import fcntl
        from RL.online_bootstrap import EstimatorDatasetStore
        digest = self._validate_request(payload)
        self.requests.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)
        # flock also serializes duplicate submissions across server processes.
        with (self.requests / f"{digest}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            response_path = self.responses / f"{digest}.json"
            receipt_path = self.requests / f"{digest}.json"
            if response_path.is_file():
                sealed = json.loads(response_path.read_text(encoding="utf-8"))
                value = sealed["batch"]
                encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()
                if sealed.get("request_sha256") != digest or sealed.get("response_sha256") != hashlib.sha256(encoded).hexdigest():
                    raise ValueError("cached online response checksum/identity mismatch")
                validate_trajectory_coverage(OnlineRolloutBatch.from_dict(value), payload)
            else:
                result = self.backend(dict(payload))
                if not isinstance(result, Mapping):
                    raise ValueError("online rollout backend must return an object")
                batch = OnlineRolloutBatch.from_dict(result)
                validate_trajectory_coverage(batch, payload)
                value = json.loads(json.dumps(batch.as_dict(), allow_nan=False))
                encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()
                sealed = {"schema_version": 2, "request_sha256": digest,
                          "response_sha256": hashlib.sha256(encoded).hexdigest(), "batch": value}
                EstimatorDatasetStore._atomic_json(response_path, sealed)
            # A crash after sealed response publication can recreate the receipt
            # without invoking the physical backend again.
            EstimatorDatasetStore._atomic_json(receipt_path, {
                "schema_version": 2, "state": "complete", "request_sha256": digest,
                "step": payload["step"], "response_sha256": sealed["response_sha256"],
                "response": str(response_path)})
            return value


class RunLifecycleBackend:
    """Bridge one request to the formal run/supervisor lifecycle.

    ``assemble`` is intentionally mandatory: it must read canonical trajectory,
    evaluation, native probability and hidden publication receipts and return a
    sealed online batch.  The bridge never guesses those fields.
    """
    def __init__(self, cfg: Mapping[str, Any], assemble, *, bind_deployment=None):
        if not callable(assemble):
            raise TypeError("online evidence assembler must be callable")
        self.cfg = dict(cfg)
        self.assemble = assemble
        self.bind_deployment = bind_deployment

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        from rollout.runner import run_online_manifest
        digest = str(request.get("request_sha256") or request_digest(request))
        tasks = [str(task) for task in request["tasks"]]
        baseline = sum(item.get("source") == "baseline" for item in request["work_items"])
        task_count = len(tasks)
        if baseline < task_count or baseline % task_count:
            raise ValueError("online request baseline work items are not evenly distributed")
        m = baseline // task_count
        run_id = "online-bootstrap-" + digest[:24]
        cfg = self.bind_deployment(request) if self.bind_deployment is not None else self.cfg
        outcomes = run_online_manifest(
            cfg, run_id, task_ids=tasks, m=m,
            branch_count=int(cfg.get("parallel", {}).get("branch_count", 1)),
            resume=True, request=request)
        result = self.assemble(request, outcomes, run_id=run_id)
        if not isinstance(result, Mapping):
            raise ValueError("online evidence assembler must return a batch object")
        return dict(result)


def make_http_server(host: str, port: int, service: OnlineRolloutService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DyConOnlineRollout/1"

        def do_POST(self):
            if self.path.rstrip("/") != "/rollout":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 128 * 1024 * 1024:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = service.handle(payload)
                body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": repr(exc)}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_args):
            return

    return ThreadingHTTPServer((host, int(port)), Handler)


def native_backend(cfg):
    """Production factory for ``dycon online-rollout-serve --backend``.

    Every request supplies immutable, verified policy/estimator exports. Run
    snapshots bind the chosen pair; serving switches only between batch runs.
    """
    import copy
    from RL.online_evidence import NativeRunAssembler
    from RL.checkpoints import verify_adapter
    from runtime.serving.parallel import artifact_contract

    def bind(request):
        deployment = request.get("deployment") or {}
        required = {"adapter", "adapter_identity_sha256", "base_fingerprint", "policy_fingerprint",
                    "estimator_manifest", "estimator_manifest_sha256", "feature", "advantage_settings"}
        if not required <= deployment.keys():
            raise ValueError("native online rollout requires immutable deployment and feature bindings")
        if deployment["base_fingerprint"] != cfg["model"]["fingerprint"]:
            raise ValueError("online deployment base differs from configured base")
        path = Path(deployment["adapter"]).resolve()
        if sha256(path / "identity.json") != deployment["adapter_identity_sha256"]:
            raise ValueError("online deployment adapter identity changed")
        identity = verify_adapter(path, verify_owner=False)
        if identity["contract"]["base_fingerprint"] != deployment["base_fingerprint"]:
            raise ValueError("online adapter belongs to another base model")
        if deployment["policy_fingerprint"] != request_digest({"base": deployment["base_fingerprint"],
                                                               "adapter": deployment["adapter_identity_sha256"]}):
            raise ValueError("online policy content fingerprint mismatch")
        predictor = artifact_contract(deployment["estimator_manifest"], deployment["estimator_manifest_sha256"],
                                      deployment["base_fingerprint"])
        feature = deployment["feature"]
        if (predictor["sites"] != [int(str(feature["site"]).removeprefix("layer_"))]
                or predictor["hidden_size"] != int(feature["width"])
                or predictor["target"] != feature["target"]):
            raise ValueError("online estimator differs from initial feature/target contract")
        result = copy.deepcopy(cfg)
        if deployment.get('failure_policy') is not None:
            from core.config import OnlineRLFailurePolicy
            result['execution']['online_rl_failure_policy'] = OnlineRLFailurePolicy(**deployment['failure_policy']).model_dump()
            if deployment['failure_policy'].get('baseline_failure_reward') is not None:
                result['execution']['limits_retry'] = {
                    'enabled': True, 'retry_statuses': ['LimitsExceeded', 'reasoning_loop'],
                    'seed_policy': 'repetition_round_offset_v1'}
        result['execution'].setdefault('selection_retry', {'enabled': True, 'max_retries': 3})
        if "max_inflight" in request:
            result["execution"]["rollout_workers"] = min(
                int(request["max_inflight"]), result["execution"]["rollout_workers"])
        if (result["agent"].get("backend", "minisweagent") not in {"openhands", "minisweagent"} or not result["capture"]["enabled"]
                or not result["capture"]["export"]["enabled"] or not result["capture"]["logprobs"]["enabled"]
                or not result.get("parallel", {}).get("enabled")
                or result["parallel"].get("mode") != "parallel_consensus"):
            raise ValueError("native online rollout requires a qualified agent, native capture/raw probabilities and consensus")
        if list(request["worker_ids"]) != list(result["serving"]["gpus"]):
            raise ValueError("online worker GPUs differ from configured serving pool")
        result["serving"]["lora"] = {"adapter": str(path), "identity_sha256": deployment["adapter_identity_sha256"],
                                     "rank": int(identity["contract"]["lora"]["rank"])}
        result["model"]["served_name"] = "online-" + deployment["policy_fingerprint"][:24]
        result["parallel"]["estimator_ref"] = deployment["estimator_manifest"]
        result["parallel"]["estimator_sha256"] = deployment["estimator_manifest_sha256"]
        result["parallel"]["branch_count"] = int(deployment["branch_count"])
        result["parallel"]["dynamic_branch_admission"] = True
        result["parallel"]["request_admission"] = deployment["request_admission"]
        result["parallel"]["dycon_candidate_portion"] = int(deployment["dycon_candidate_portion"])
        from core.io import hash_json
        result["config_hash"] = hash_json({k: v for k, v in result.items() if k != "config_hash"})
        return result

    def assemble(request, execution, *, run_id):
        if execution.get("exit_code") != 0:
            raise RuntimeError("online rollout supervisor did not exit successfully")
        from RL.online_evidence import ensure_evaluated
        ensure_evaluated(cfg["project_root"], run_id)
        deployment = request["deployment"]
        return NativeRunAssembler(cfg, feature=deployment["feature"],
            advantage_settings=deployment["advantage_settings"])(request, execution, run_id=run_id)
    return RunLifecycleBackend(cfg, assemble, bind_deployment=bind)
