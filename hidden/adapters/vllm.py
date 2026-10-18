"""Batched adapter; vLLM owns scheduling, prefix caching and state restoration."""

import time

from core.io import hash_json
from hidden.backends.vllm.capture import capture_window, native_layers


class VLLMAdapter:
    def __init__(self, llm, cfg, layer_count, hidden_size, runtime):
        self.llm, self.cfg = llm, cfg
        self.layer_count, self.hidden_size = layer_count, hidden_size
        self.fingerprint = cfg["model"]["fingerprint"]
        self.batch_size = cfg["hidden"]["vllm"]["max_inflight_requests"]
        self.metadata = {
            **runtime,
            "options": cfg["hidden"]["vllm"],
            "model_fingerprint": self.fingerprint,
            "dtype": cfg["hidden"]["dtype"],
        }
        self.metrics = []

    def sites(self):
        return [{"kind": "decoder_output", "block_index": i} for i in range(self.layer_count - 1)] + [
            {"kind": "final_norm", "block_index": None}
        ]

    def capture_batch(self, rows, layers, chunk_size):
        native_layers(layers, self.layer_count)
        if not rows or len(rows) > self.batch_size:
            raise ValueError("Capture window must be nonempty and bounded")
        maximum = self.cfg["hidden"]["vllm"]["max_model_len"]
        for row in rows:
            ids = row["prefix_token_ids"]
            if not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("Prefix must contain nonnegative integer token IDs")
            if len(ids) >= maximum:
                raise ValueError("Prefix must fit max_model_len with one trigger output token; no truncation")
            if hash_json(ids) != row["prefix_sha256"]:
                raise ValueError("Prefix checksum mismatch")
        start = time.monotonic()
        values, metrics = capture_window(
            self.llm, rows, layers, self.layer_count, self.hidden_size, self.cfg["hidden"]["vllm"]
        )
        self.metrics.append({**metrics, "window_id": len(self.metrics), "seconds": time.monotonic() - start})
        return values

    def close(self):
        # Engine shutdown is public in the pinned release. Never touch other processes.
        self.llm.llm_engine.engine_core.shutdown()


def load_adapter(cfg, *, steering_replay=False):
    from core.model_identity import verify_model
    from hidden.backends.vllm.engine import engine_args, verify_runtime

    verify_model(cfg)
    runtime = verify_runtime(cfg["project_root"])
    from transformers import AutoConfig
    from vllm import LLM

    config = AutoConfig.from_pretrained(
        cfg["model"]["path"],
        local_files_only=True,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    text = getattr(config, "text_config", config)
    native_layers(cfg["hidden"]["layers"], text.num_hidden_layers)
    args = engine_args(cfg)
    if steering_replay:
        args.update(enable_steer_vector=True, steer_algorithms="direct", steer_graph_mode="split",
                    steer_vector_dtype="float32")
    llm = LLM(**args)
    return VLLMAdapter(llm, cfg, text.num_hidden_layers, text.hidden_size, runtime)
