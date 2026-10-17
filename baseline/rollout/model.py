"""Native mini-swe-agent model, augmented with request provenance."""

import os

from hidden.backends.vllm.online import build_capture_select
from runtime.steering.easysteer.payload import steering_payload


def model_config(cfg, capture_context=None):
    s, g, a = cfg["serving"], cfg["generation"], cfg["agent"]
    extra_body = {
        "top_k": g["top_k"],
        "steering": steering_payload(cfg),
        "chat_template_kwargs": {k: g[k] for k in ["enable_thinking", "preserve_thinking"]},
    }
    # Ordinary rollout requests keep their exact historical body.  Metadata is
    # opt-in and is consumed only by the managed online capture plane.
    if cfg.get("capture", {}).get("enabled"):
        if not capture_context:
            raise ValueError("capture-enabled model requires request provenance context")
        extra_body["dycon_capture"] = {**capture_context, "anchors": list(cfg["capture"]["anchors"])}
        # This is the native vLLM request field consumed by the EasySteer
        # capture policy. Provenance remains in dycon_capture alongside it.
        newline_ids = capture_context.get("newline_token_ids") or [198, 271, 1358]
        close_ids = capture_context.get("close_token_ids") or [248069]
        extra_body["capture_select"] = capture_context.get(
            "capture_select", build_capture_select(newline_ids, close_ids)
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
            "extra_body": extra_body,
        },
    }


def make_model(cfg, deadline, capture_context=None):
    from minisweagent.models.litellm_model import LitellmModel

    from enrichment.trajectory import tools_schema

    class RecordingModel(LitellmModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dycon_capture_step = 0

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

        def query(self, messages, **kwargs):
            self.config.model_kwargs["timeout"] = deadline.remaining(cfg["timeouts"]["model_seconds"])
            capture = self.config.model_kwargs.get("extra_body", {}).get("dycon_capture")
            if capture is not None:
                capture["step_id"] = self._dycon_capture_step
                self._dycon_capture_step += 1
            message = super().query(messages, **kwargs)
            request_record = {
                "messages": self._prepare_messages_for_api(messages),
                "tools": tools_schema(),
                "chat_template_kwargs": {
                    k: cfg["generation"][k] for k in ["enable_thinking", "preserve_thinking"]
                },
                "config_hash": cfg["config_hash"],
                "model_fingerprint": cfg["model"]["fingerprint"],
            }
            if capture is not None:
                request_record["dycon_capture"] = dict(capture)
            message.setdefault("extra", {})["request"] = request_record
            return message

    return RecordingModel(**model_config(cfg, capture_context))
