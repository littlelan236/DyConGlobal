"""Dependency-ordered replay; vLLM owns prefix caching and state restoration."""

import time
from collections import OrderedDict, deque

from core.io import hash_json
from hidden.backends.vllm.capture import capture_window, native_layers


def replay_group(row):
    """A request ID is not a trajectory: consecutive steps must share a queue.

    Rows without trajectory metadata conservatively share one serial queue.
    Native token hashes, not this grouping, determine cache compatibility.
    """
    return tuple(row.get(key) or "" for key in ("rollout_id", "session_id", "branch_id"))


def replay_windows(rows, limit):
    """Submit at most one anchor per trajectory until that wave completes."""
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(replay_group(row), deque()).append(row)
    active = deque(groups.values())
    while active:
        window = []
        for _ in range(min(limit, len(active))):
            group = active.popleft()
            window.append(group.popleft())
            if group:
                active.append(group)
        yield window


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
            "replay_policy": "trajectory_apc",
        }
        self.metrics = []
        # Bound diagnostic CPU history independently of vLLM's GPU cache.
        self._previous = OrderedDict()

    def sites(self):
        return [{"kind": "decoder_output", "block_index": i} for i in range(self.layer_count - 1)] + [
            {"kind": "final_norm", "block_index": None}
        ]

    def capture_batch(self, rows, layers, chunk_size):
        native_layers(layers, self.layer_count)
        if not rows or len(rows) > self.batch_size:
            raise ValueError("Capture window must be nonempty and bounded")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate sample_id in capture window")
        maximum = self.cfg["hidden"]["vllm"]["max_model_len"]
        for row in rows:
            ids = row["prefix_token_ids"]
            if not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("Prefix must contain nonnegative integer token IDs")
            if len(ids) >= maximum:
                raise ValueError("Prefix must fit max_model_len with one trigger output token; no truncation")
            if hash_json(ids) != row["prefix_sha256"]:
                raise ValueError("Prefix checksum mismatch")
        values = {}
        options = self.cfg["hidden"]["vllm"]
        for window in replay_windows(rows, min(self.batch_size, options["max_num_seqs"])):
            start = time.monotonic()
            captured, metrics = capture_window(
                self.llm, window, layers, self.layer_count, self.hidden_size, options
            )
            if set(captured) != {row["sample_id"] for row in window}:
                raise ValueError("Replay returned mismatched sample identities")
            for row, sample in zip(window, metrics["samples"], strict=True):
                if sample["sample_id"] != row["sample_id"]:
                    raise ValueError("Replay metrics differ from submitted sample identities")
                key, ids = replay_group(row), row["prefix_token_ids"]
                previous = self._previous.pop(key, None)
                lcp = 0
                if previous is not None:
                    old = previous["prefix_token_ids"]
                    lcp = next(
                        (i for i, (a, b) in enumerate(zip(old, ids)) if a != b), min(len(old), len(ids))
                    )
                sample.update(
                    prefix_tokens=len(ids),
                    uncached_prompt_tokens=len(ids) - sample["cached_tokens"],
                    previous_sample_id=previous["sample_id"] if previous else None,
                    common_prefix_tokens=lcp,
                    append_only=previous is not None and lcp == len(previous["prefix_token_ids"]),
                )
                self._previous[key] = {"sample_id": row["sample_id"], "prefix_token_ids": ids}
                while len(self._previous) > self.batch_size:
                    self._previous.popitem(last=False)
            metrics["uncached_prompt_tokens"] = metrics["prompt_tokens"] - metrics["cached_tokens"]
            self.metrics.append(
                {
                    **metrics,
                    "window_id": len(self.metrics),
                    "seconds": time.monotonic() - start,
                    "replay_policy": "trajectory_apc",
                }
            )
            values.update(captured)
        return {row["sample_id"]: values[row["sample_id"]] for row in rows}

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
        args.update(
            enable_steer_vector=True,
            steer_algorithms="direct",
            steer_graph_mode="split",
            steer_vector_dtype="float32",
        )
    llm = LLM(**args)
    return VLLMAdapter(llm, cfg, text.num_hidden_layers, text.hidden_size, runtime)
