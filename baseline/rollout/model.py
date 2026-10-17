"""Native mini-swe-agent model, augmented with request provenance."""

import os

from runtime.steering.easysteer.payload import steering_payload


def model_config(cfg):
    s, g, a = cfg["serving"], cfg["generation"], cfg["agent"]
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
            "extra_body": {
                "top_k": g["top_k"],
                "steering": steering_payload(cfg),
                "chat_template_kwargs": {k: g[k] for k in ["enable_thinking", "preserve_thinking"]},
            },
        },
    }


def make_model(cfg, deadline):
    from minisweagent.models.litellm_model import LitellmModel

    from enrichment.trajectory import tools_schema

    class RecordingModel(LitellmModel):
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
            message = super().query(messages, **kwargs)
            message.setdefault("extra", {})["request"] = {
                "messages": self._prepare_messages_for_api(messages),
                "tools": tools_schema(),
                "chat_template_kwargs": {
                    k: cfg["generation"][k] for k in ["enable_thinking", "preserve_thinking"]
                },
                "config_hash": cfg["config_hash"],
                "model_fingerprint": cfg["model"]["fingerprint"],
            }
            return message

    return RecordingModel(**model_config(cfg))
