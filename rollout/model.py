"""Native mini-swe-agent model, augmented with request provenance."""

import copy
import os
import time

from hidden.backends.vllm.online import build_capture_select
from runtime.steering.easysteer.payload import steering_payload


def model_config(cfg, capture_context=None, sample=None):
    s, g, a = cfg["serving"], cfg["generation"], cfg["agent"]
    extra_body = {
        "top_k": g["top_k"],
        "min_p": g.get("min_p", 0.0),
        "repetition_penalty": g.get("repetition_penalty", 1.0),
        "return_token_ids": bool(cfg.get("benchmark_metrics_dir")),
        "steering": steering_payload(cfg),
        "chat_template_kwargs": {k: g[k] for k in ["enable_thinking", "preserve_thinking"]},
    }
    if g.get("thinking_token_budget") is not None:
        extra_body["thinking_token_budget"] = g["thinking_token_budget"]
    # Ordinary rollout requests keep their exact historical body.  Metadata is
    # opt-in and is consumed only by the managed online capture plane.
    if cfg.get("capture", {}).get("enabled") and capture_context is not False:
        if not capture_context:
            raise ValueError("capture-enabled model requires request provenance context")
        extra_body["dycon_capture"] = {**capture_context, "anchors": list(cfg["capture"]["anchors"])}
        # This is the native vLLM request field consumed by the EasySteer
        # capture policy. Provenance remains in dycon_capture alongside it.
        newline_ids = capture_context.get("newline_token_ids") or [198, 271, 1358]
        close_ids = capture_context.get("close_token_ids") or [248069]
        extra_body["capture_select"] = capture_context.get(
            "capture_select",
            build_capture_select(
                newline_ids if "reasoning_newline" in cfg["capture"]["anchors"] else [],
                close_ids if "think_close" in cfg["capture"]["anchors"] else [],
                include_open="think_open" in cfg["capture"]["anchors"],
            ),
        )
    return {
        "model_name": "openai/" + cfg["model"]["served_name"],
        "cost_tracking": "ignore_errors",
        "observation_template": a["observation_template"],
        "format_error_template": a["format_error_template"],
        "model_kwargs": {
            "api_base": s["base_url"],
            "api_key": os.environ.get(s["api_key_env"], "EMPTY"),
            "timeout": cfg["timeouts"]["model_seconds"],
            **{k: g[k] for k in ["temperature", "top_p", "max_tokens", "seed"]},
            "presence_penalty": g.get("presence_penalty", 0.0),
            "extra_body": extra_body,
        },
    }


def make_model(cfg, deadline, capture_context=None, runtime=None, sample=None):
    from uuid import uuid4

    from minisweagent.exceptions import FormatError
    from minisweagent.models.litellm_model import LitellmModel

    from enrichment.trajectory import tools_schema

    class RecordingModel(LitellmModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dycon_capture_step = 0
            self._dycon_metric_step = 0

        def record_action_phase(self, message, phase, details=None):
            summary = message.get("extra", {}).get("dycon_parallel")
            if summary is not None:
                controller = (runtime or {}).get("capture_controller")
                controller.record_parallel_action(summary, phase, details)

        def serialize(self):
            import copy

            value = copy.deepcopy(super().serialize())

            def redact(node):
                if isinstance(node, dict):
                    return {
                        k: "<environment>" if k in {"api_key", "password", "authorization"} else redact(v)
                        for k, v in node.items()
                    }
                if isinstance(node, list):
                    return [redact(x) for x in node]
                return node

            return redact(value)

        def _query(self, messages, **kwargs):
            capture = self.config.model_kwargs.get("extra_body", {}).get("dycon_capture")
            if capture is not None:
                capture["api_attempt_id"] = uuid4().hex
            return super()._query(messages, **kwargs)

        def query(self, messages, **kwargs):
            from core.io import hash_json

            self.config.model_kwargs["timeout"] = deadline.remaining(cfg["timeouts"]["model_seconds"])
            capture = self.config.model_kwargs.get("extra_body", {}).get("dycon_capture")
            if capture is not None:
                for key in ("decision_id", "branch_id", "branch_count", "parent_prefix_sha256"):
                    capture.pop(key, None)
                capture.update(step_id=self._dycon_capture_step, query_id=uuid4().hex)
                self._dycon_capture_step += 1

            def attach(message):
                record = {
                    "tools": tools_schema(),
                    "chat_template_kwargs": {
                        k: cfg["generation"][k] for k in ["enable_thinking", "preserve_thinking"]
                    },
                    "config_hash": cfg["config_hash"],
                    "model_fingerprint": cfg["model"]["fingerprint"],
                    "steering_payload_sha256": cfg["steering"].get("payload_sha256"),
                }
                prepared = self._prepare_messages_for_api(messages)
                if capture is None:
                    record["messages"] = prepared
                else:
                    record["messages_sha256"] = hash_json(prepared)
                    record["dycon_capture"] = {
                        k: v
                        for k, v in capture.items()
                        if k not in {"newline_token_ids", "close_token_ids", "capture_select"}
                    }
                message.setdefault("extra", {})["request"] = record
                return message

            def check_capture(message=None):
                if capture is not None and cfg["capture"]["failure_policy"] == "block_rollout":
                    controller = (runtime or {}).get("capture_controller")
                    if controller:
                        failures = controller.sessions[capture["session_id"]].options.get("failures", [])
                        failures = list(failures)
                        failures.extend(
                            f["error"]
                            for f in controller.queue.status()["failed"]
                            if f.get("session_id") == capture["session_id"]
                        )
                        for path in (controller.export_root / "failures").glob("*.json"):
                            from core.io import read_json

                            record = read_json(path)
                            if record.get("session_id") == capture["session_id"]:
                                failures.append(record["reason"])
                        if failures:
                            from runtime.serving.errors import CaptureError

                            error = CaptureError(str(failures))
                            error.response_message = message
                            raise error

            check_capture()
            try:
                from rollout.parallel import ParallelReasoningCoordinator, active

                step = self._dycon_metric_step
                self._dycon_metric_step += 1
                if active(cfg, step) and not (runtime or {}).get("diagnostic_single_query"):
                    coordinator = ParallelReasoningCoordinator(
                        cfg, (runtime or {}).get("capture_controller"),
                        client=(runtime or {}).get("parallel_client"),
                        diagnostic_mode=(runtime or {}).get("diagnostic_selection_mode"),
                        diagnostic_branches=(runtime or {}).get("diagnostic_branch_count"))

                    def generate(prepared, metadata, seed):
                        options = copy.deepcopy(self.config.model_kwargs)
                        extra = {**options["extra_body"], **kwargs.get("extra_body", {})}
                        options.update(kwargs)
                        extra["dycon_capture"] = metadata
                        options["extra_body"] = extra
                        options["seed"] = seed
                        options["timeout"] = deadline.remaining(cfg["timeouts"]["model_seconds"])
                        # Call the same native model transport and parsers, with
                        # per-branch immutable kwargs; no shared metadata races.
                        return LitellmModel._query(self, prepared, **options)

                    def parse(response):
                        try:
                            actions = self._parse_actions(response)
                        except FormatError as error:
                            for msg in error.messages:
                                msg.setdefault("extra", {})["response"] = response.model_dump(mode="json")
                            raise
                        msg = response.choices[0].message.model_dump()
                        msg["extra"] = {"actions": actions, "response": response.model_dump(mode="json"),
                                        "timestamp": time.time()}
                        return msg

                    message = attach(coordinator.run(
                        self._prepare_messages_for_api(messages), capture, generate, parse, self._calculate_cost))
                else:
                    message = attach(super().query(messages, **kwargs))
                check_capture(message)
                return message
            except FormatError as exc:
                for message in exc.messages:
                    if isinstance((message.get("extra") or {}).get("response"), dict):
                        attach(message)
                        check_capture(message)
                raise

    return RecordingModel(**model_config(cfg, capture_context, sample))
