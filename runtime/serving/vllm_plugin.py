"""Opt-in vLLM endpoint plugin for online FULL-graph capture.

The plugin is loaded only when ``VLLM_PLUGINS=dycon_capture``.  It starts the
native EasySteer capture stream and drains labelled rows through the existing
vLLM collective RPC.  Tensor bytes are staged on the shared filesystem and
submitted to DyConGlobal's localhost control plane; no scheduler, KV cache,
TP assembly or graph logic is duplicated here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import numpy as np
from starlette.requests import Request

STREAM = "tact_states"
LOG = logging.getLogger("dycon.capture")
REQUEST_CONTEXT = ContextVar("dycon_capture_request", default=None)


class DyConCapturePlugin:
    name = "dycon_capture"
    required_tasks = None

    def __init__(self):
        self.engine = None
        self.task: asyncio.Task | None = None
        self.control_url = os.environ.get("DYCON_CAPTURE_CONTROL_URL")
        self.staging = Path(os.environ.get("DYCON_CAPTURE_STAGING_ROOT", "hidden/online/staging"))
        self.layers = json.loads(os.environ.get("DYCON_CAPTURE_LAYERS", "[]"))
        self.select = json.loads(os.environ.get("DYCON_CAPTURE_SELECT", "{}"))
        self.newline_ids = set(json.loads(os.environ.get("DYCON_CAPTURE_NEWLINE_IDS", "[]")))
        self.close_ids = set(json.loads(os.environ.get("DYCON_CAPTURE_CLOSE_IDS", "[248069]")))
        self.open_ids = set(json.loads(os.environ.get("DYCON_CAPTURE_OPEN_IDS", "[248068]")))
        self.tp_size = 1
        self.failures = []
        self.rows_published = 0
        self.worker_statuses = []
        self.drain_lock = asyncio.Lock()
        self.stream_options = {}
        self.behavior_failures = []
        self.execution_intervals = {}
        self.request_metadata = {}
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
        self.scoring = None
        if self.export_options.get("enabled", True) and self.export_options.get("spool_root"):
            from core.disk_budget import DiskBudget

            project = os.environ["DYCON_CAPTURE_PROJECT"]
            self.budget = DiskBudget(
                project,
                Path(project) / self.export_options["spool_root"],
                self.export_options["spool_max_bytes"],
                int(os.environ.get("DYCON_CAPTURE_RESERVE_BYTES", "0")),
            )

    def attach_router(self, app):
        @app.middleware("http")
        async def provenance(request, call_next):
            if request.method != "POST" or request.url.path != "/v1/chat/completions":
                return await call_next(request)
            body = await request.json()
            metadata = body.pop("dycon_capture", None)
            body.pop("capture_select", None)
            if metadata is not None:
                metadata = dict(metadata)
                metadata["capture_select"] = self.select
                metadata["message_index"] = len(body.get("messages", []))
            request._body = json.dumps(body).encode()
            context = REQUEST_CONTEXT.set(metadata)
            try:
                return await call_next(request)
            finally:
                REQUEST_CONTEXT.reset(context)

        @app.post("/dycon/parallel/{action}")
        async def parallel(action: str, request: Request):
            from starlette.responses import JSONResponse

            if self.scoring is None:
                return JSONResponse({"error": "parallel_scoring_not_enabled"}, status_code=503)
            value = await request.json()
            try:
                if action == "reserve":
                    return await asyncio.to_thread(self.scoring.reserve, value["decision_id"], value["branches"])
                if action == "results":
                    # A completed generate can reach this endpoint before the
                    # periodic drain loop has published its close row. Drain
                    # once synchronously so scoring observes the native event
                    # without increasing the model request or GPU wait.
                    await self.drain_once()
                    return await asyncio.to_thread(self.scoring.results, value["decision_id"], value["timeout"])
                if action == "release":
                    return self.scoring.release(value["decision_id"])
                return JSONResponse({"error": "unknown_parallel_action"}, status_code=404)
            except (ValueError, KeyError, RuntimeError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

        @app.post("/dycon/capture/engine/flush")
        async def flush():
            await self.drain_once()
            return {"rows_published": self.rows_published, "failures": self.failures}

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
                "filtered_events": self.filtered_events,
                "failures": self.failures,
                "workers": self.worker_statuses,
            }

        @app.post("/dycon/capture/behavior-failure")
        async def behavior_failure(request: Request):
            body = await request.json()
            self.behavior_failures.append(body)
            return {"recorded": True, "failures": len(self.behavior_failures)}

    async def init_state(self, engine_client, state, args):
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
        # Initialize the native capture graphs with a dedicated request before
        # rollout traffic. Normal multi-token prefill initializes model state.
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from hidden.backends.vllm.online import capture_request_metadata

        metadata = capture_request_metadata(
            "engine-warmup",
            "engine-warmup",
            "engine-warmup",
            "engine-warmup",
            0,
            ["reasoning_newline", "think_close"],
            newline_token_ids=sorted(self.newline_ids),
            close_token_ids=sorted(self.close_ids),
        )
        await asyncio.to_thread(self._post, "/dycon/capture/session/start", {"metadata": metadata})
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
                "engine-warmup",
            ):
                pass
            await self.drain_once()
        finally:
            REQUEST_CONTEXT.reset(token)
        await asyncio.to_thread(self._post, "/dycon/capture/session/engine-warmup/stop", {})

    def _post(self, route, value):
        import requests

        response = requests.post(
            self.control_url.rstrip("/") + route, json=value, timeout=self.control_seconds
        )
        if not response.ok:
            raise RuntimeError(f"capture control {route}: {response.status_code}: {response.text[:500]}")
        return response.json()

    def _safe_post(self, route, value, metadata):
        try:
            return self._post(route, value)
        except Exception as exc:
            failure = {"session_id": metadata.get("session_id"), "reason": f"{route}: {exc!r}"}
            self.failures.append(failure)
            from core.io import hash_json, write_json

            write_json(self.staging.parent / "failures" / (hash_json(failure) + ".json"), failure)
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
            if metadata is not None and request.sampling_params is not None:
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
            if metadata is not None:
                self.request_metadata[request.request_id] = dict(metadata)
                self.request_prompt_lengths[request.request_id] = len(request.prompt_token_ids or [])
                from core.io import hash_json

                binding = dict(metadata)
                binding["prompt_token_ids"] = list(request.prompt_token_ids or [])
                binding["request_sha256"] = hash_json(binding["prompt_token_ids"])
                await asyncio.to_thread(
                    self._safe_post,
                    "/dycon/capture/request/bind",
                    {"request_id": request.request_id, "metadata": binding},
                    binding,
                )
            return await original_add(request, *args, **kwargs)

        @wraps(original_generate)
        async def generate(*args, **kwargs):
            metadata = REQUEST_CONTEXT.get()
            final = None
            generated = []
            params = args[1] if len(args) > 1 else kwargs["sampling_params"]
            if metadata is not None and metadata.get("decision_id") and params.n != 1:
                raise ValueError("Parallel transport requires native n=1 for exact capture identity")
            delta = getattr(params.output_kind, "name", "") == "DELTA"
            async for output in original_generate(*args, **kwargs):
                final = output
                if delta:
                    generated.extend(output.outputs[0].token_ids)
                else:
                    generated = list(output.outputs[0].token_ids)
                yield output
            if metadata is not None and final is not None:
                await asyncio.to_thread(
                    self._safe_post,
                    "/dycon/capture/request/output",
                    {
                        "metadata": metadata,
                        "request_id": final.request_id,
                        "prompt_token_ids": list(final.prompt_token_ids or []),
                        "output_token_ids": generated,
                        "finish_reason": final.outputs[0].finish_reason,
                        "num_cached_tokens": getattr(final, "num_cached_tokens", 0) or 0,
                    },
                    metadata,
                )

        self.engine.input_processor.process_inputs = process
        self.engine._add_request = add
        self.engine.generate = generate

    async def drain_once(self):
        async with self.drain_lock:
            raw = await self.engine.collective_rpc("fetch_captured", args=(STREAM,), kwargs={"clear": True})
            executions = await self.engine.collective_rpc("fetch_capture_execution")
            for worker in executions:
                for request_id, rows in worker.items():
                    self.execution_intervals.setdefault(request_id, []).extend(rows)
            self.worker_statuses = await self.engine.collective_rpc("capture_status", args=(STREAM,))
            if raw and any(raw) and self.export_options.get("enabled", True):
                await asyncio.to_thread(self._publish, raw)

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

    def _publish(self, raw):
        from vllm.model_hooks.capture.serialization import assemble_captured

        from runtime.serving.capture_worker import execution_at

        tensors, metadata, _ = assemble_captured(raw, tp_size=self.tp_size)
        if not metadata:
            return
        first_layer = next(iter(metadata))
        labels = metadata[first_layer]
        LOG.info("DyCon capture assembled %d rows across %d layers", len(labels), len(tensors))
        events = list(zip(labels.req_ids, labels.positions.tolist(), labels.token_ids.tolist(), range(len(labels.req_ids)), strict=True))
        # Native streams can be fetched in batches.  Sort only the small event
        # metadata here; tensor rows remain indexed by the original row number.
        events.sort(key=lambda row: (row[0], int(row[1]), row[3]))
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
            if anchor == "think_close" and self.scoring is not None:
                self.scoring.submit(self.request_metadata.get(request_id, {}), request_id,
                                    int(position), int(token_id), sample)
            execution = execution_at(self.execution_intervals, request_id, int(position))
            fd, name = tempfile.mkstemp(prefix=".dycon-", suffix=".npz", dir=self.staging)
            os.close(fd)
            path = Path(name)
            reservation = "handoff:" + str(path)
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
                )
                if not response.ok:
                    LOG.error("DyCon capture control rejected staged row: %s", response.text[:500])
                response.raise_for_status()
                LOG.info(
                    "DyCon capture staged anchor=%s request=%s position=%s", anchor, request_id, position
                )
                self.rows_published += 1
            except Exception as exc:
                failure = {
                    "request_id": request_id,
                    "reason": repr(exc),
                    "session_id": self.request_metadata.get(request_id, {}).get("session_id"),
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
        if self.scoring is not None:
            self.scoring.close()
        if self.task is not None:
            self.task.cancel()
            await self.task
        if self.engine is not None:
            await self.drain_once()
            await self.engine.collective_rpc("stop_capture", args=(STREAM,), kwargs={})


def plugin():
    return DyConCapturePlugin()
