"""Opt-in vLLM endpoint plugin for online FULL-graph capture.

The plugin is loaded only when ``VLLM_PLUGINS=dycon_capture``.  It starts the
native EasySteer capture stream and drains labelled rows through the existing
vLLM collective RPC.  Tensor bytes are staged on the shared filesystem and
submitted to DyConGlobal's localhost control plane; no scheduler, KV cache,
TP assembly or graph logic is duplicated here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import hmac
import logging
import os
import tempfile
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from uuid import uuid4

import numpy as np
from starlette.requests import Request

STREAM = "tact_states"
LOG = logging.getLogger("dycon.capture")
REQUEST_CONTEXT = ContextVar("dycon_capture_request", default=None)
OWNER_CONTEXT = ContextVar("dycon_request_owner", default=None)
LOOP_CONTEXT = ContextVar("dycon_reasoning_loop_request", default=None)
NATIVE_CONTEXT = ContextVar("dycon_native_request", default=None)
CONTROL_DEADLINE = ContextVar("dycon_control_deadline", default=None)


def capture_rows_need_processing(raw, export_enabled, scoring_enabled):
    return bool(raw and any(raw) and (export_enabled or scoring_enabled))


class DyConCapturePlugin:
    name = "dycon_capture"
    required_tasks = None

    def __init__(self):
        self.engine = None
        self.task: asyncio.Task | None = None
        self.control_url = os.environ.get("DYCON_CAPTURE_CONTROL_URL")
        self.control_only = os.environ.get('DYCON_REQUEST_CONTROL_ONLY') == '1'
        self.staging = Path(os.environ.get("DYCON_CAPTURE_STAGING_ROOT", "hidden/online/staging"))
        self.layers = json.loads(os.environ.get("DYCON_CAPTURE_LAYERS", "[]"))
        self.select = json.loads(os.environ.get("DYCON_CAPTURE_SELECT", "{}"))
        self.newline_ids = set(json.loads(os.environ.get("DYCON_CAPTURE_NEWLINE_IDS", "[]")))
        self.close_ids = set(json.loads(os.environ.get("DYCON_CAPTURE_CLOSE_IDS", "[248069]")))
        self.open_ids = set(json.loads(os.environ.get("DYCON_CAPTURE_OPEN_IDS", "[248068]")))
        self.tp_size = 1
        self.failures = []
        self.rows_published = 0
        self.stage_diagnostics = {}
        self.stage_diagnostics_points = {}
        self.stage_diagnostics_point_map = {}
        self.stage_diagnostics_truncated = {}
        self.selector_observer_errors = 0
        self.worker_statuses = []
        self.drain_lock = asyncio.Lock()
        self.stream_options = {}
        self.behavior_failures = []
        self.execution_intervals = {}
        self.request_metadata = {}
        self.failed_capture_decisions = set()
        self.request_prompt_lengths = {}
        self.request_last_position = {}
        self.request_close_seen = set()
        self.filtered_events = {"prompt": 0, "after_close": 0, "duplicate": 0, "unknown": 0}
        self.failure_policy = os.environ.get("DYCON_CAPTURE_FAILURE_POLICY", "record_and_continue")
        self.control_seconds = float(os.environ.get("DYCON_CAPTURE_CONTROL_SECONDS", "30"))
        self.flush_seconds = max(0.001, float(os.environ.get("DYCON_CAPTURE_FLUSH_SECONDS", "0.1")))
        self.export_options = json.loads(os.environ.get("DYCON_CAPTURE_EXPORT_OPTIONS", "{}"))
        self.budget = None
        self.parallel_options = json.loads(os.environ.get("DYCON_PARALLEL_OPTIONS", "{}"))
        self.reasoning_loop_options = json.loads(os.environ.get('DYCON_REASONING_LOOP_OPTIONS', '{}'))
        self.request_receipt_root = Path(os.environ.get('DYCON_REQUEST_RECEIPT_ROOT', str(self.staging.parent / 'request-terminals')))
        self.scoring = None
        self.cancelled_run = False
        self.cancelled_decisions = set()
        self.cancelled_sessions = set()
        self.native_changed = asyncio.Event()
        self.native_requests = {}
        self.cancel_token = os.environ.get("DYCON_CANCEL_TOKEN", "")
        self.owner_run = os.environ.get("DYCON_CANCEL_OWNER_RUN")
        self.owner_attempt = os.environ.get("DYCON_EXECUTION_ATTEMPT")
        from runtime.serving.execution import ExecutionLane

        capacity = max(128, self.parallel_options.get("max_inflight_branches", 16) * 4)
        self.wait_lane = ExecutionLane("dycon-admission", 16, capacity)
        self.receipt_workers = int(os.environ.get("DYCON_CAPTURE_RECEIPT_WORKERS", "4"))
        if not 1 <= self.receipt_workers <= 128:
            raise ValueError("Capture receipt workers must be in [1, 128]")
        self.receipt_lane = ExecutionLane("dycon-receipt", self.receipt_workers, max(capacity, self.receipt_workers))
        self.publish_lane = ExecutionLane("dycon-publish", 1, 4)
        self.control_workers = int(os.environ.get("DYCON_CAPTURE_CONTROL_WORKERS", "2"))
        self.result_workers = int(self.parallel_options.get("result_workers", 16))
        self.control_lane = ExecutionLane("dycon-control", self.control_workers, capacity)
        self.result_lane = ExecutionLane("dycon-results", self.result_workers, capacity)
        self.failure_lane = ExecutionLane("dycon-diagnostics", 4, capacity)
        if self.export_options.get("enabled", True) and self.export_options.get("spool_root"):
            from core.disk_budget import DiskBudget

            project = os.environ["DYCON_CAPTURE_PROJECT"]
            self.budget = DiskBudget(
                project,
                Path(project) / self.export_options["spool_root"],
                self.export_options["spool_max_bytes"],
            )

    def _check_cancelled(self, metadata):
        if (self.cancelled_run or (metadata and metadata.get("decision_id") in self.cancelled_decisions)
                or (metadata and metadata.get("session_id") in self.cancelled_sessions)):
            from core.cancellation import Cancelled
            raise Cancelled("Native request cancelled")

    async def cancel_native(self, decision_id=None, session_id=None, wait_seconds=0):
        if session_id is not None:
            self.cancelled_sessions.add(session_id)
        elif decision_id is None:
            self.cancelled_run = True
        else:
            self.cancelled_decisions.add(decision_id)
        if self.scoring is not None and hasattr(self.scoring, "cancel"):
            decisions = ({state['metadata'].get('decision_id') for state in self.native_requests.values()
                          if state['metadata'].get('session_id') == session_id}
                         if session_id is not None else {decision_id})
            for owned_decision in decisions:
                if owned_decision is not None or session_id is None:
                    await self.control_lane.run(self.scoring.cancel, owned_decision)
        requested, errors, aborts = [], [], []
        async def abort_one(native_id):
            try:
                async with asyncio.timeout(min(3.0, self.control_seconds)):
                    await self.engine.abort(native_id, internal=True)
                requested.append(native_id)
            except Exception as exc:
                errors.append({"native_request_id": native_id, "error": repr(exc)})
        for native_id, state in list(self.native_requests.items()):
            if session_id is not None and state["metadata"].get("session_id") != session_id:
                continue
            if decision_id is not None and state["metadata"].get("decision_id") != decision_id:
                continue
            state["cancel_requested"] = True
            if state["phase"] in {"submitting", "running"}:
                aborts.append(abort_one(native_id))
        await asyncio.gather(*aborts)
        def pending():
            return [key for key, state in self.native_requests.items() if state.get('cancel_requested')
                    and (session_id is None or state['metadata'].get('session_id') == session_id)
                    and (decision_id is None or state['metadata'].get('decision_id') == decision_id)]
        if wait_seconds:
            deadline = asyncio.get_running_loop().time() + min(float(wait_seconds), 30.0)
            while pending():
                self.native_changed.clear()
                if not pending():
                    break
                try:
                    await asyncio.wait_for(self.native_changed.wait(), max(0.001, deadline - asyncio.get_running_loop().time()))
                except TimeoutError:
                    break
        return {"cancel_requested": True, "abort_requested_ids": requested,
                "pending_native_ids": pending(), "errors": errors, "session_id": session_id,
                "abort_completed": not pending()}

    async def _receipt_post(self, route, value, metadata):
        deadline = CONTROL_DEADLINE.set(time.monotonic() + self.control_seconds)
        try:
            async with asyncio.timeout(self.control_seconds):
                return await self.receipt_lane.run(self._safe_post, route, value, metadata)
        except TimeoutError:
            # Match _safe_post: retain native completion even for block_rollout;
            # the model wrapper raises CaptureError outside API retry handling.
            from core.io import hash_json, write_json
            key = (metadata.get('session_id'), metadata.get('decision_id'))
            if key[1] is not None:
                self.failed_capture_decisions.add(key)
            failure = {**{k: metadata[k] for k in ('session_id', 'decision_id', 'branch_id', 'api_attempt_id', 'step_id') if k in metadata},
                       'reason': route + ': capture control total deadline exceeded', 'payload': value}
            self.failures.append(failure)
            await self.failure_lane.run(write_json, self.staging.parent / 'failures' / (hash_json(failure) + '.json'), failure)
            return None
        finally:
            CONTROL_DEADLINE.reset(deadline)

    def attach_router(self, app):
        @app.middleware("http")
        async def provenance(request, call_next):
            native_probe = (request.url.path == "/v1/completions"
                            and self.parallel_options.get("branch_probe") is not None)
            if request.method != "POST" or (request.url.path != "/v1/chat/completions" and not native_probe):
                return await call_next(request)
            body = await request.json()
            metadata = body.pop("dycon_capture", None)
            owner = body.pop('dycon_request_owner', None)
            if owner is not None and (owner.get('run_id') != self.owner_run
                    or owner.get('execution_attempt_id') != self.owner_attempt):
                from starlette.responses import JSONResponse
                return JSONResponse({'error': 'request_owner_mismatch'}, status_code=403)
            body.pop("capture_select", None)
            if metadata is not None:
                metadata = dict(metadata)
                metadata["capture_select"] = self.select
                if not native_probe:
                    metadata["message_index"] = len(body.get("messages", []))
            request._body = json.dumps(body).encode()
            context = REQUEST_CONTEXT.set(metadata)
            owner_context = OWNER_CONTEXT.set(owner)
            try:
                return await call_next(request)
            finally:
                REQUEST_CONTEXT.reset(context)
                OWNER_CONTEXT.reset(owner_context)

        def authorized(request, value):
            return (bool(self.cancel_token) and
                    hmac.compare_digest(request.headers.get("X-DyCon-Cancel-Token", ""), self.cancel_token) and
                    value.get("run_id") == self.owner_run and
                    value.get("execution_attempt_id") == self.owner_attempt)

        @app.post("/dycon/capture/engine/cancel")
        async def cancel_engine(request: Request):
            from starlette.responses import JSONResponse
            value = await request.json()
            if not authorized(request, value):
                return JSONResponse({"error": "cancel_scope_or_token_mismatch"}, status_code=403)
            return await self.cancel_native(value.get("decision_id"), value.get("session_id"), value.get("wait_seconds", 0))

        @app.post('/dycon/capture/engine/request-terminal')
        async def request_terminal(request: Request):
            from starlette.responses import JSONResponse
            from core.io import hash_json, read_json
            value = await request.json()
            if not authorized(request, value):
                return JSONResponse({'error': 'request_terminal_scope_or_token_mismatch'}, status_code=403)
            key = hash_json([value.get('session_id'), value.get('api_attempt_id')])
            path = self.request_receipt_root / (key + '.json')
            if not path.exists():
                return JSONResponse({'error': 'native_terminal_receipt_pending'}, status_code=409)
            return await self.control_lane.run(read_json, path)

        @app.post("/dycon/parallel/{action}")
        async def parallel(action: str, request: Request):
            from starlette.responses import JSONResponse

            if self.scoring is None:
                return JSONResponse({"error": "parallel_scoring_not_enabled"}, status_code=503)
            value = await request.json()
            try:
                if action == "cancel":
                    if not authorized(request, value):
                        return JSONResponse({"error": "cancel_scope_or_token_mismatch"}, status_code=403)
                    return await self.cancel_native(value["decision_id"])
                if action == "status":
                    return await self.control_lane.run(self.scoring.status)
                if action == "reserve":
                    return await self.wait_lane.run(self.scoring.reserve, value["decision_id"], value["branches"],
                                                   value.get("decision_deadline_unix"))
                if action == "branch-acquire":
                    return await self.wait_lane.run(self.scoring.acquire_branch, value["decision_id"], value["branch_id"])
                if action == "branch-release":
                    return await self.control_lane.run(self.scoring.release_branch, value, value.get("native_request_id"), value.get("reason", "client_release"))
                if action == "results":
                    # Results must not queue behind long branch admission or
                    # synchronously drain another session's tensor I/O. The
                    # periodic drain notifies the scorer; clients may poll pending.
                    return await self.result_lane.run(self.scoring.results, value["decision_id"], value["timeout"])
                if action == "release":
                    return await self.control_lane.run(self.scoring.release, value["decision_id"])
                return JSONResponse({"error": "unknown_parallel_action"}, status_code=404)
            except asyncio.CancelledError:
                # Disconnect cannot strand a synchronous acquire after its HTTP waiter vanishes.
                if value.get("decision_id"):
                    await asyncio.shield(self.cancel_native(value["decision_id"]))
                raise
            except (ValueError, KeyError, RuntimeError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

        @app.post("/dycon/capture/engine/flush")
        async def flush(request: Request):
            body = await request.body()
            value = json.loads(body) if body else {}
            sid = value.get("session_id")
            from runtime.serving.diagnostics import safe_stack, sanitize_error_text, bounded_records
            from core.io import write_json
            from starlette.responses import JSONResponse
            supplied = request.headers.get("X-DyCon-Correlation-ID", "")
            correlation = supplied if 1 <= len(supplied) <= 64 and all(c.isascii() and (c.isalnum() or c in "_-") for c in supplied) else uuid4().hex
            started = time.monotonic()
            try:
                pending = [key for key, state in self.native_requests.items()
                           if sid is None or state['metadata'].get('session_id') == sid]
                if pending:
                    from starlette.responses import JSONResponse
                    return JSONResponse({'error': 'native_generation_still_in_flight', 'pending_native_ids': pending}, status_code=409)
                await self.drain_once()
                try:
                    diagnostic = self._session_stage_diagnostics(sid) if sid else {"schema_version": 1, "requests": {}, "truncated_count": 0}
                except Exception:
                    diagnostic = {"schema_version": 1, "requests": {}, "reason_code": "diagnostic_summary_unavailable"}
                failures = [f for f in self.failures if sid is None or
                            (isinstance(f, dict) and f.get("session_id") == sid)]
                unscoped = [f for f in self.failures if not isinstance(f, dict) or f.get('session_id') is None]
                return {"rows_published": self.rows_published, "failures": failures,
                        'unscoped_failures': unscoped, 'session_id': sid,
                        'stage_diagnostics': diagnostic}
            except Exception as exc:
                evidence = {"schema_version": 1, "stage": "engine_flush", "session_id": sid,
                            "run_id": self.owner_run, "execution_attempt_id": self.owner_attempt,
                            "correlation_id": correlation, "error_type": type(exc).__name__,
                            "reason_code": "engine_flush_failed", "error_summary": sanitize_error_text(exc),
                            "duration_seconds": time.monotonic() - started, "stack": safe_stack(exc)}
                try:
                    project = os.environ["DYCON_CAPTURE_PROJECT"]
                    if bounded_records(project, self.owner_run, self.owner_attempt, [evidence], kind="control"):
                        write_json(self.staging.parent / "control-errors" / (correlation + ".json"), evidence)
                except Exception:
                    pass
                return JSONResponse({k: evidence[k] for k in ("error_type", "reason_code", "stage", "correlation_id")},
                                    status_code=500, headers={"X-DyCon-Correlation-ID": correlation})

        @app.post("/dycon/capture/engine/mode")
        async def mode(request: Request):
            body = await request.json()
            method = "start_capture" if body["enabled"] else "stop_capture"
            results = await self.engine.collective_rpc(
                method, args=(STREAM,), kwargs=self.stream_options if body["enabled"] else {}
            )
            return {"enabled": body["enabled"], "acknowledged": all(results)}

        @app.post("/dycon/capture/engine/replay")
        async def replay(request: Request):
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind

            body = await request.json()

            async def one(row):
                metadata = {
                    **row["metadata"],
                    "replay": True,
                    "capture_select": {STREAM: {"prompt_positions": [-1]}},
                }
                from core.io import hash_json

                metadata["capture_select_sha256"] = hash_json(metadata["capture_select"])
                token = REQUEST_CONTEXT.set(metadata)
                try:
                    result = None
                    async for output in self.engine.generate(
                        {"prompt_token_ids": row["prefix_token_ids"]},
                        SamplingParams(
                            temperature=0,
                            max_tokens=1,
                            ignore_eos=True,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                        ),
                        row["request_id"],
                    ):
                        result = output
                    return {"request_id": result.request_id, "cached_tokens": result.num_cached_tokens}
                finally:
                    REQUEST_CONTEXT.reset(token)

            outputs = await asyncio.gather(*(one(row) for row in body["rows"]))
            await self.drain_once()
            return {"outputs": outputs, "failures": self.failures}

        @app.get("/dycon/capture/engine/status")
        async def status():
            if self.engine is not None:
                self.worker_statuses = await self.engine.collective_rpc("capture_status", args=(STREAM,))
            return {
                "enabled": self.engine is not None,
                "layers": self.layers,
                "stream": STREAM,
                "rows_published": self.rows_published,
                "control_seconds": self.control_seconds,
                "receipt_workers": self.receipt_workers,
                "control_workers": self.control_workers,
                "result_workers": self.result_workers,
                "filtered_events": self.filtered_events,
                "execution_lanes": {"wait": self.wait_lane.status(), "receipt": self.receipt_lane.status(),
                                    "publish": self.publish_lane.status(), "control": self.control_lane.status(),
                                    "results": self.result_lane.status(), "diagnostics": self.failure_lane.status()},
                "native_lifecycle": {key: state["phase"] for key, state in self.native_requests.items()},
                "cancelling": self.cancelled_run,
                "failures": self.failures,
                "workers": self.worker_statuses,
            }

        @app.post("/dycon/capture/behavior-failure")
        async def behavior_failure(request: Request):
            body = await request.json()
            self.behavior_failures.append(body)
            return {"recorded": True, "failures": len(self.behavior_failures)}

    async def init_state(self, engine_client, state, args):
        if self.control_only and engine_client is not None:
            self.engine = engine_client
            self._attach_admission()
            return
        if not self.control_url or engine_client is None:
            return
        if not self.layers:
            raise RuntimeError("DYCON_CAPTURE_LAYERS must contain at least one layer")
        from vllm.capture import SelectSpec

        from hidden.backends.vllm.capture import native_layers

        self.engine = engine_client
        self.tp_size = int(engine_client.vllm_config.parallel_config.tensor_parallel_size)
        text_config = engine_client.model_config.hf_text_config
        self.layers = native_layers(self.layers, text_config.num_hidden_layers)
        if self.parallel_options.get("enabled"):
            from runtime.serving.parallel import ScoringBroker

            self.scoring = ScoringBroker(self.parallel_options,
                                         os.environ["DYCON_PARALLEL_MODEL_FINGERPRINT"],
                                         text_config.num_hidden_layers)
        spec = self.select.get(STREAM, self.select)
        self.stream_options = {
            "layers": self.layers,
            "dtype": "float32",
            "select": SelectSpec(**spec).to_wire() if "phase" not in spec else spec,
            "budget_bytes": int(os.environ.get("DYCON_CAPTURE_BUDGET_BYTES", "268435456")),
            "device_budget_bytes": int(os.environ.get("DYCON_CAPTURE_BUDGET_BYTES", "268435456")),
        }
        await engine_client.collective_rpc("configure_capture_execution")
        results = await engine_client.collective_rpc(
            "start_capture", args=(STREAM,), kwargs=self.stream_options
        )
        if not results or not all(results):
            raise RuntimeError("native vLLM capture stream failed to start")
        self.staging.mkdir(parents=True, exist_ok=True)
        self._attach_admission()
        self.task = asyncio.create_task(self._drain_loop(), name="dycon-capture-drain")
        await self._warmup_capture_graphs()

    async def _warmup_capture_graphs(self):
        # Initialize the native capture graphs with a dedicated request before
        # rollout traffic. Normal multi-token prefill initializes model state.
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from hidden.backends.vllm.online import capture_request_metadata

        # All serving replicas share one controller; each engine and restart
        # must own a distinct diagnostic session and request identity.
        session_id = "engine-warmup-" + uuid4().hex
        metadata = capture_request_metadata(
            session_id,
            "engine-warmup",
            "engine-warmup",
            session_id,
            0,
            ["reasoning_newline", "think_close"],
            newline_token_ids=sorted(self.newline_ids),
            close_token_ids=sorted(self.close_ids),
        )
        await self.receipt_lane.run(self._post, "/dycon/capture/session/start", {"metadata": metadata})
        token = REQUEST_CONTEXT.set(metadata)
        try:
            # Token 0 is valid for the supported causal tokenizers; warmup must
            # not assume Qwen3.5's private <think> IDs when validating another
            # dense checkpoint.
            async for _ in self.engine.generate(
                {"prompt_token_ids": [0, 0]},
                SamplingParams(
                    temperature=0, max_tokens=1, ignore_eos=True, output_kind=RequestOutputKind.FINAL_ONLY
                ),
                session_id,
            ):
                pass
            await self.drain_once()
        finally:
            REQUEST_CONTEXT.reset(token)
        await self.receipt_lane.run(self._post, f"/dycon/capture/session/{session_id}/stop", {})

    @staticmethod
    def _raw_t1_values(final, generated):
        if os.environ.get("DYCON_RAW_T1_ENABLED") != "1":
            return None
        values = []
        for token_id, entry in zip(generated, final.outputs[0].logprobs or []):
            item = entry.get(int(token_id)) if isinstance(entry, dict) else None
            if item is None:
                raise RuntimeError("raw T=1 output did not contain sampled-token value")
            value = getattr(item, "logprob", item.get("logprob") if isinstance(item, dict) else None)
            if value is None:
                raise RuntimeError("raw T=1 output value is missing")
            values.append(float(value))
        if len(values) != len(generated):
            raise RuntimeError("raw T=1 output/logit token alignment mismatch")
        return values

    def _post(self, route, value):
        import requests

        timeout = self.control_seconds
        total_deadline = CONTROL_DEADLINE.get()
        if total_deadline is not None:
            timeout = min(timeout, total_deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Capture control total deadline exceeded")
        deadline = (value.get("metadata") or value).get("decision_deadline_unix")
        if deadline is not None:
            timeout = min(timeout, deadline - time.time())
            if timeout <= 0:
                raise TimeoutError("Capture decision deadline exceeded")
        response = requests.post(
            self.control_url.rstrip("/") + route, json=value, timeout=timeout
        )
        if not response.ok:
            raise RuntimeError(f"capture control {route}: {response.status_code}: {response.text[:500]}")
        return response.json()

    def _safe_post(self, route, value, metadata):
        import requests

        key = (metadata.get("session_id"), metadata.get("decision_id"))
        try:
            if key[1] is not None and key in self.failed_capture_decisions:
                raise RuntimeError("Capture decision already unavailable")
            # Only these endpoints have an idempotent identity/content contract.
            attempts = 2 if route.endswith(("/request/bind", "/request/output")) else 1
            for attempt in range(attempts):
                try:
                    return self._post(route, value)
                except (requests.ConnectionError, requests.Timeout):
                    if attempt + 1 == attempts:
                        raise
        except Exception as exc:
            if key[1] is not None:
                self.failed_capture_decisions.add(key)
            failure = {"session_id": metadata.get("session_id"), "reason": f"{route}: {exc!r}",
                       **{k: metadata[k] for k in ("decision_id", "branch_id", "api_attempt_id", "step_id")
                          if k in metadata}}
            self.failures.append(failure)
            from core.io import hash_json, write_json

            write_json(self.staging.parent / "failures" / (hash_json({**failure, "payload": value}) + ".json"),
                       {**failure, "payload": value})
            try:
                self._post("/dycon/capture/failure", failure)
            except Exception:
                pass
            # Preserve the native completion even for block_rollout. The common
            # client raises CaptureError after recording this response, outside
            # mini-swe-agent's API retry loop; a 500 here would resample it.
            return None

    def _attach_admission(self):
        # Adapt this engine instance only. vLLM retains rendering, sampling,
        # request ID randomization and batching; the plugin attaches provenance
        # at the already-tokenized EngineCoreRequest boundary.
        original_process = self.engine.input_processor.process_inputs
        original_add = self.engine._add_request
        original_generate = self.engine.generate

        @wraps(original_process)
        def process(*args, **kwargs):
            metadata = REQUEST_CONTEXT.get()
            # Keep provenance separate from the native stream-wide selector.
            request = original_process(*args, **kwargs)
            owner = OWNER_CONTEXT.get()
            if owner and request.sampling_params is not None:
                required = owner.get('required_max_tokens')
                if required is not None and (len(request.prompt_token_ids or []) + required > owner['max_model_len']
                        or request.sampling_params.max_tokens != required):
                    raise ValueError('ContextWindowExceeded: rendered prompt plus frozen output budget exceeds context; output budget cannot be silently reduced')
            if metadata is None and not self.control_only:
                # Missing provenance must not inherit the enabled stream's
                # selector (e.g. OpenHands condensation and warmup calls).
                # An empty mapping inherits it, and None disables a filter,
                # not capture. Native SelectSpec exclusions override includes
                # for both phases and remain local to this request.
                request.capture_select = {STREAM: {
                    "prompt": "all", "generation": "all",
                    "exclude_prompt_window": [0, None],
                    "exclude_generation_window": [0, None],
                }}
            if metadata is not None and request.sampling_params is not None:
                if os.environ.get("DYCON_RAW_T1_ENABLED") == "1":
                    params = request.sampling_params.clone()
                    params.logprobs = 0
                    params.prompt_logprobs = None
                    params.logprob_token_ids = None
                    params.extra_args = dict(params.extra_args or {})
                    params.extra_args["dycon_raw_t1_sampled_token_v1"] = True
                    request.sampling_params = params
                if metadata.get("replay"):
                    # Native input admission owns prompt-target cache policy.
                    from vllm.capture import SelectSpec

                    request.capture_select = {STREAM: SelectSpec(prompt_positions=[-1]).to_wire()}
                ids = list(request.prompt_token_ids or [])
                if (
                    "think_open" in metadata.get("anchors", [])
                    and not metadata.get("replay")
                    and (len(ids) < 2 or ids[-2:] != [248068, 198])
                ):
                    raise ValueError("capture prompt must end with the active <think>\\n boundary")
                # Decode capture targets are generated after the prompt: APC
                # can read the entire reusable prompt. Native capture policy
                # preserves explicit skips and prompt-target restrictions.
            return request

        @wraps(original_add)
        async def add(request, *args, **kwargs):
            metadata = REQUEST_CONTEXT.get()
            owner = OWNER_CONTEXT.get()
            self._check_cancelled(metadata or owner)
            native_state = NATIVE_CONTEXT.get()
            loop_state = LOOP_CONTEXT.get()
            if loop_state is not None:
                loop_state["native_request_id"] = request.request_id
                ids = request.prompt_token_ids or []
                loop_state["reasoning_active"] = bool(ids and (
                    ids[-1] in self.open_ids or (len(ids) >= 2 and ids[-2] in self.open_ids)))
            if metadata is not None:
                self.native_requests[request.request_id] = {
                    "metadata": dict(metadata), "phase": "binding", "cancel_requested": False}
                self.request_metadata[request.request_id] = dict(metadata)
                self.request_prompt_lengths[request.request_id] = len(request.prompt_token_ids or [])
                from core.io import hash_json

                binding = dict(metadata)
                binding["prompt_token_ids"] = list(request.prompt_token_ids or [])
                binding["request_sha256"] = hash_json(binding["prompt_token_ids"])
                self.request_metadata[request.request_id] = binding
                if native_state is None:
                    raise RuntimeError("Native capture admission has no request-local context")
                native_state.update(native_request_id=request.request_id, binding=binding)
                if os.environ.get("DYCON_RAW_T1_ENABLED") == "1":
                    import msgspec
                    policy = msgspec.to_builtins(request.sampling_params)
                    policy.pop("logprobs", None)
                    policy.pop("prompt_logprobs", None)
                    policy.pop("logprob_token_ids", None)
                    policy.pop("extra_args", None)
                    native_state["raw_t1_generation_identity"] = {
                        "generation_policy_sha256": hash_json(policy),
                        "grammar_sha256": hash_json(policy.get("structured_outputs")),
                        "steering": None}
                await self._receipt_post(
                    "/dycon/capture/request/bind",
                    {"request_id": request.request_id, "metadata": binding},
                    binding,
                )
                self._check_cancelled(metadata)
                self.native_requests[request.request_id]["phase"] = "submitting"
            if metadata is None and owner is not None:
                self.native_requests[request.request_id] = {
                    'metadata': dict(owner), 'phase': 'submitting', 'cancel_requested': False}
                native_state.update(native_request_id=request.request_id)
            result = await original_add(request, *args, **kwargs)
            if metadata is None and owner is not None:
                self.native_requests[request.request_id]['phase'] = 'running'
                try:
                    self._check_cancelled(owner)
                except BaseException:
                    await self.engine.abort(request.request_id, internal=True)
                    raise
            if metadata is not None:
                state = self.native_requests[request.request_id]
                state["phase"] = "running"
                if state["cancel_requested"] or self.cancelled_run or metadata.get("decision_id") in self.cancelled_decisions:
                    # Cancellation may have traversed the registry before engine add completed.
                    await self.engine.abort(request.request_id, internal=True)
            return result

        @wraps(original_generate)
        async def generate(*args, **kwargs):
            metadata = REQUEST_CONTEXT.get()
            owner = OWNER_CONTEXT.get()
            final = None
            generated = []
            raw_values = []
            raw_error = None
            branch_released = False
            params = args[1] if len(args) > 1 else kwargs["sampling_params"]
            if (metadata is not None and params.n != 1 and
                    (metadata.get("decision_id") or os.environ.get("DYCON_RAW_T1_ENABLED") == "1")):
                raise ValueError("Parallel transport requires native n=1 for exact capture identity")
            options = self.parallel_options.get('reasoning_loop') or self.reasoning_loop_options
            guarded = bool(options.get('enabled') and not (metadata or {}).get('replay')
                           and ((metadata and metadata.get('decision_id'))
                                or (owner and owner.get('request_role') == 'agent_decision')))
            state, detector = {}, None
            if guarded:
                from vllm.sampling_params import RequestOutputKind

                from runtime.serving.reasoning_loop import ReasoningLoopDetector

                detector = ReasoningLoopDetector(options, self.close_ids)
                params = params.clone()
                if getattr(params.output_kind, "name", "") == "FINAL_ONLY":
                    params.output_kind = RequestOutputKind.CUMULATIVE
                params.stream_interval = options["check_interval_tokens"]
                if len(args) > 1:
                    args = (*args[:1], params, *args[2:])
                else:
                    kwargs = {**kwargs, "sampling_params": params}
            context = LOOP_CONTEXT.set(state if guarded else None)
            native_state = {} if metadata is not None or owner is not None else None
            native_context = NATIVE_CONTEXT.set(native_state)
            stream = original_generate(*args, **kwargs)
            output_published = False
            owner_published = False
            async def publish_owner():
                nonlocal owner_published
                if owner_published or owner is None or final is None:
                    return
                from core.io import hash_json, now, write_json
                loop = state.get('reasoning_loop')
                if loop is not None:
                    loop.update(final_output_tokens=len(generated),
                                abort_completed=final.outputs[0].finish_reason == 'abort')
                value = {'schema_version': 1, 'owner': owner, 'native_request_id': (native_state or {}).get('native_request_id'),
                         'prompt_token_ids': list(final.prompt_token_ids or []), 'output_token_ids': list(generated),
                         'finish_reason': final.outputs[0].finish_reason, 'reasoning_loop': loop, 'finished_at': now()}
                key = hash_json([owner.get('session_id'), owner.get('api_attempt_id')])
                await self.failure_lane.run(write_json, self.request_receipt_root / (key + '.json'), value)
                owner_published = True
            async def publish_output():
                nonlocal branch_released, output_published
                if output_published or metadata is None or final is None:
                    return
                if not native_state or not native_state.get("native_request_id"):
                    raise RuntimeError("Native output has no request-local admission identity")
                native_id = native_state["native_request_id"]
                loop = state.get("reasoning_loop")
                if loop is not None:
                    loop.update(final_output_tokens=len(generated),
                                abort_completed=final.outputs[0].finish_reason == "abort")
                    if loop["abort_completed"] and self.scoring is not None:
                        self.scoring.mark_terminated(metadata, state["native_request_id"], loop)
                generation_entropy = None
                if os.environ.get("DYCON_GENERATION_ENTROPY") == "1":
                    stats = await self.engine.collective_rpc("generation_entropy", args=(native_id, generated))
                    if len(stats) != 1:
                        raise ValueError("Generation entropy requires TP=1")
                    generation_entropy = stats[0]
                await self._receipt_post(
                    "/dycon/capture/request/output",
                    {
                        "metadata": native_state["binding"],
                        "request_id": native_id,
                        **({"generation_entropy": generation_entropy} if generation_entropy is not None else {}),
                        "external_response_id": final.request_id,
                        "prompt_token_ids": list(final.prompt_token_ids or []),
                        "output_token_ids": generated,
                        **({"raw_t1_logprobs_b64": base64.b64encode(
                            np.asarray(raw_values, dtype="<f4").tobytes()).decode("ascii")}
                           if os.environ.get("DYCON_RAW_T1_ENABLED") == "1" and raw_error is None else {}),
                        **({"raw_t1_generation_identity": native_state["raw_t1_generation_identity"]}
                           if "raw_t1_generation_identity" in native_state else {}),
                        "finish_reason": final.outputs[0].finish_reason,
                        "num_cached_tokens": getattr(final, "num_cached_tokens", 0) or 0,
                        **({"reasoning_loop": loop} if loop is not None else {}),
                    },
                    metadata,
                )
                if (self.scoring is not None and hasattr(self.scoring, "mark_native_terminal")
                        and metadata.get("decision_id") and final.finished):
                    self.scoring.mark_native_terminal(
                        native_state["binding"], native_id, generated, self.close_ids,
                        final.outputs[0].finish_reason)
                if self.scoring is not None and hasattr(self.scoring, "release_branch") and metadata.get("decision_id"):
                    self.scoring.release_branch(metadata, native_id, "native_output")
                    branch_released = True
                output_published = True

            try:
                delta = getattr(params.output_kind, "name", "") == "DELTA"
                async for output in stream:
                    final = output
                    ids = output.outputs[0].token_ids
                    new_ids = list(ids if delta else ids[len(generated):])
                    if os.environ.get("DYCON_RAW_T1_ENABLED") == "1" and metadata is not None and raw_error is None:
                        try:
                            chunk = self._raw_t1_values(output, list(ids))
                            raw_values.extend(chunk if delta else chunk[len(generated):])
                            if len(raw_values) > json.loads(os.environ.get("DYCON_RAW_T1_OPTIONS", "{}")).get("max_request_tokens", 131072):
                                raise ValueError("Raw probability request token budget exceeded")
                        except Exception as exc:
                            raw_error = type(exc).__name__
                            raw_values.clear()
                    generated.extend(new_ids)
                    # Finished/closed requests win races with a new abort.
                    if (detector is not None and not state.get("reasoning_loop")
                            and (state.get("reasoning_active") or options.get("include_post_close", False))
                            and not output.finished):
                        if not state.get("reasoning_active") and options.get("include_post_close", False):
                            detector.phase = "post_close"
                        if (not options.get("include_post_close", False)
                                and any(token in self.close_ids for token in new_ids)):
                            detector.closed = True
                        hit = detector.feed(new_ids)
                        if hit is not None:
                            native_id = state["native_request_id"]
                            state["reasoning_loop"] = {
                                **hit, **{k: (metadata or owner or {})[k] for k in (
                                    "session_id", "decision_id", "branch_id", "api_attempt_id") if k in (metadata or owner or {})},
                                "native_request_id": native_id,
                                "observed_at_output_tokens": len(generated),
                                "abort_requested": True,
                            }
                            await self.engine.abort(native_id, internal=True)
                    if output.finished:
                        await publish_output()
                        await publish_owner()
                    yield output
                await publish_output()
                await publish_owner()
            finally:
                try:
                    await stream.aclose()
                finally:
                    if self.scoring is not None and hasattr(self.scoring, "release_branch") and metadata and metadata.get("decision_id") and not branch_released:
                        native_id = (native_state or {}).get("native_request_id")
                        if native_id is not None:
                            self.scoring.release_branch(metadata, native_id, "stream_exit")
                    if native_state is not None and native_state.get("native_request_id"):
                        self.native_requests.pop(native_state["native_request_id"], None)
                        self.native_changed.set()
                    LOOP_CONTEXT.reset(context)
                    NATIVE_CONTEXT.reset(native_context)

        self.engine.input_processor.process_inputs = process
        self.engine._add_request = add
        self.engine.generate = generate

    async def drain_once(self):
        async with self.drain_lock:
            raw = await self.engine.collective_rpc("fetch_captured", args=(STREAM,), kwargs={"clear": True})
            executions = await self.engine.collective_rpc("fetch_capture_execution")
            for rank, worker in enumerate(executions):
                selector = worker.pop("__capture_diagnostics__", {})
                if rank == 0:
                    self.selector_observer_errors += selector.get("observer_errors", 0)
                    for request_id, report in selector.get("requests", {}).items():
                        entry = self.stage_diagnostics.setdefault(request_id, {})
                        entry["selector_observed"] = True
                        entry["observer_errors"] = entry.get("observer_errors", 0) + report.get("observer_errors", 0)
                        for position in report.get("positions", []):
                            self._count_diagnostic(request_id, "selector_seen", "think_close", position)
                        remainder = max(0, report.get("count", 0) - len(report.get("positions", [])))
                        if remainder:
                            counts = entry.setdefault("selector_seen", {})
                            counts["think_close"] = counts.get("think_close", 0) + remainder
                            sid = self.request_metadata.get(request_id, {}).get("session_id")
                            self.stage_diagnostics_truncated[sid] = self.stage_diagnostics_truncated.get(sid, 0) + remainder
                for request_id, rows in worker.items():
                    self.execution_intervals.setdefault(request_id, []).extend(rows)
            self.worker_statuses = await self.engine.collective_rpc("capture_status", args=(STREAM,))
            if capture_rows_need_processing(raw, self.export_options.get("enabled", True),
                                            self.scoring is not None):
                await self.publish_lane.run(self._publish, raw)

    async def _drain_loop(self):
        while True:
            try:
                await self.drain_once()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self.failures.append(repr(exc))
                LOG.exception("DyCon capture drain failed")
            await asyncio.sleep(self.flush_seconds)

    def _count_diagnostic(self, request_id, stage, anchor, position=None):
        try:
            self._record_diagnostic_count(request_id, stage, anchor, position)
        except Exception:
            pass  # Observability cannot change publish/submit behavior.

    def _record_diagnostic_count(self, request_id, stage, anchor, position=None):
        metadata = self.request_metadata.get(request_id, {})
        session_id = metadata.get("session_id")
        if not session_id:
            return
        entry = self.stage_diagnostics.setdefault(request_id, {
            "selector_seen": {}, "capture_row": {}, "submit_accepted": {}, "submit_failed": {}})
        counts = entry.setdefault(stage, {})
        counts[anchor] = counts.get(anchor, 0) + 1
        if anchor == "think_close" and position is not None:
            point_map = self.stage_diagnostics_point_map.setdefault(session_id, {})
            key = (request_id, int(position))
            point = point_map.get(key)
            if point is None:
                if sum(len(points) for points in self.stage_diagnostics_point_map.values()) >= 256:
                    self.stage_diagnostics_truncated[session_id] = self.stage_diagnostics_truncated.get(session_id, 0) + 1
                    return
                point = {"request_id": request_id, "position": int(position), "stages": []}
                point_map[key] = point
            if stage not in point["stages"]:
                point["stages"].append(stage)
            if stage == "submit_accepted":
                # The controller's existing sample receipt proves successful
                # points. Keep coordinate metadata only for unresolved points.
                point_map.pop(key, None)

    def _session_stage_diagnostics(self, session_id):
        requests = {}
        for request_id, metadata in self.request_metadata.items():
            if metadata.get("session_id") != session_id or request_id not in self.stage_diagnostics:
                continue
            requests[request_id] = {k: v for k, v in self.stage_diagnostics[request_id].items() if k != "think_close_points"}
        for point in self.stage_diagnostics_point_map.get(session_id, {}).values():
            requests.setdefault(point["request_id"], {}).setdefault("think_close_points", []).append({
                "position": point["position"], "stages": list(point["stages"])})
        return {"schema_version": 1, "requests": requests,
                "selector_observer_errors": sum(r.get("observer_errors", 0) for r in requests.values()),
                "truncated_count": self.stage_diagnostics_truncated.get(session_id, 0)}

    def _publish(self, raw):
        from vllm.model_hooks.capture.serialization import assemble_captured

        from runtime.serving.capture_worker import execution_at

        tensors, metadata, _ = assemble_captured(raw, tp_size=self.tp_size)
        if not metadata:
            return
        first_layer = next(iter(metadata))
        labels = metadata[first_layer]
        events = list(zip(labels.req_ids, labels.positions.tolist(), labels.token_ids.tolist(), range(len(labels.req_ids)), strict=True))
        # Native streams can be fetched in batches.  Sort only the small event
        # metadata here; tensor rows remain indexed by the original row number.
        events.sort(key=lambda row: (row[0], int(row[1]), row[3]))
        prepared = []
        for request_id, position, token_id, row_number in events:
            if token_id in self.open_ids:
                anchor = "think_open"
            elif token_id in self.close_ids:
                anchor = "think_close"
            elif token_id in self.newline_ids:
                anchor = "reasoning_newline"
            else:
                self.filtered_events["unknown"] += 1
                continue
            if not self._accept_reasoning_event(request_id, int(position), int(token_id), anchor):
                continue
            sample = {
                f"layer_{layer}": tensor[row_number].detach().cpu().numpy()
                for layer, tensor in tensors.items()
            }
            self._count_diagnostic(request_id, "capture_row", anchor, position)
            if anchor == "think_close" and self.scoring is not None:
                self.scoring.submit(self.request_metadata.get(request_id, {}), request_id,
                                    int(position), int(token_id), sample)
            execution = execution_at(self.execution_intervals, request_id, int(position))
            prepared.append((request_id, position, token_id, anchor, sample, execution))
        # Publish all scoring features before a slow per-point disk/HTTP handoff.
        self._publish_prepared(prepared)

    def _publish_prepared(self, prepared):
        for request_id, position, token_id, anchor, sample, execution in prepared:
            fd, name = tempfile.mkstemp(prefix=".dycon-", suffix=".npz", dir=self.staging)
            os.close(fd)
            path = Path(name)
            reservation = "retained-handoff:" + str(path)
            confirmed = False
            submit_recorded = False
            response = None
            correlation_id = uuid4().hex
            request_started = time.monotonic()
            try:
                if self.budget:
                    self.budget.acquire(reservation, sum(v.nbytes + 4096 for v in sample.values()) + 4096)
                np.savez(path, **sample)
                import requests

                response = requests.post(
                    self.control_url.rstrip("/") + "/dycon/capture/sample-file",
                    json={
                        "request_id": request_id,
                        "path": path.name,
                        "anchor": anchor,
                        "position": int(position),
                        "token_id": int(token_id),
                        "absolute_position": int(position),
                        "phase": "prompt" if anchor == "think_open" else "generation_input",
                        "graph_replays": int(execution["graph_execution"] == "full"),
                        **execution,
                    },
                    timeout=self.control_seconds,
                    headers={"X-DyCon-Correlation-ID": correlation_id},
                )
                if not response.ok:
                    from runtime.serving.diagnostics import safe_response_summary
                    LOG.error("DyCon capture control rejected staged row: %s",
                              safe_response_summary(response.content[:512], 512))
                response.raise_for_status()
                acknowledgment = response.json()
                if isinstance(acknowledgment, dict) and acknowledgment.get('status') == 'failed':
                    self._count_diagnostic(request_id, "submit_failed", anchor, position)
                    submit_recorded = True
                    raise RuntimeError('Capture controller did not accept the staged sample')
                self._count_diagnostic(request_id, "submit_accepted", anchor, position)
                submit_recorded = True
                confirmed = True
                self.rows_published += 1
            except Exception as exc:
                if not submit_recorded:
                    self._count_diagnostic(request_id, "submit_failed", anchor, position)
                from runtime.serving.diagnostics import safe_response_summary, sanitize_error_text
                failure = {
                    "request_id": request_id, "stage": "submit",
                    "reason": sanitize_error_text(exc, 512),
                    "session_id": self.request_metadata.get(request_id, {}).get("session_id"),
                    "correlation_id": (response.headers.get("X-DyCon-Correlation-ID", correlation_id)
                                       if response is not None else correlation_id),
                    "status_code": getattr(response, "status_code", None),
                    "duration_seconds": round(time.monotonic() - request_started, 6),
                    "error_type": type(exc).__name__,
                    "response_summary": safe_response_summary(response.content[:512], 512)
                                       if response is not None else "",
                    "retained_staging_path": str(path), "reservation_key": reservation,
                }
                self.failures.append(failure)
                from core.io import hash_json, write_json

                write_json(
                    self.staging.parent / "failures" / (hash_json([request_id, position]) + ".json"),
                    failure,
                )
                if self.failure_policy == "block_rollout":
                    raise
            finally:
                if confirmed:
                    path.unlink(missing_ok=True)
                    if self.budget:
                        self.budget.release(reservation)

    def _accept_reasoning_event(self, request_id, position, token_id, anchor):
        """Filter answer/prompt events before allocating an export handoff."""
        from runtime.serving.capture import accept_reasoning_event

        prompt_length = self.request_prompt_lengths.get(request_id)
        if prompt_length is None:
            prompt_length = len(self.request_metadata.get(request_id, {}).get("prompt_token_ids", ()))
        state = {
            "prompt_length": prompt_length,
            "last_position": self.request_last_position.get(request_id, -1),
            "close_seen": request_id in self.request_close_seen,
        }
        accepted = accept_reasoning_event(state, position, anchor, self.filtered_events)
        self.request_last_position[request_id] = state["last_position"]
        if state["close_seen"]:
            self.request_close_seen.add(request_id)
        return accepted

    async def shutdown(self):
        try:
            if self.scoring is not None:
                self.scoring.close()  # Wake synchronous admission/result waiters first.
            if self.task is not None:
                self.task.cancel()
                await self.task
            if self.engine is not None and self.control_url:
                async with asyncio.timeout(self.control_seconds):
                    await self.drain_once()
                    await self.engine.collective_rpc("stop_capture", args=(STREAM,), kwargs={})
        finally:
            for lane in (self.wait_lane, self.receipt_lane, self.publish_lane, self.control_lane, self.result_lane, self.failure_lane):
                lane.close()


def plugin():
    return DyConCapturePlugin()
