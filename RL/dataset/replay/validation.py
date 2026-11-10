"""Pure policy and token checks; no inference imports in the runner."""
from __future__ import annotations

import math
from core.io import hash_json

ROOT_KEYS = ("temperature", "top_p", "top_k")
DEFAULTS = {"min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "repetition_penalty": 1.0, "min_tokens": 0, "ignore_eos": False,
            "stop_token_ids": [], "logit_bias": None, "allowed_token_ids": None}
UNSUPPORTED = ("structured_outputs", "grammar", "bad_words", "thinking_token_budget",
               "repetition_detection", "speculative_config", "require_reasoning_close",
               "logits_processors", "stop", "response_format")


def root_sampling(root_config):
    generation = root_config["generation"]
    values = {key: generation[key] for key in ROOT_KEYS}
    if (any(type(values[k]) not in (int, float) or not math.isfinite(values[k]) for k in ROOT_KEYS)
            or values["temperature"] < 0 or not 0 < values["top_p"] <= 1
            or type(values["top_k"]) is not int or values["top_k"] < -1 or values["top_k"] == 0):
        raise ValueError("Root temperature/top_p/top_k are invalid")
    return {**values, "semantics": "processed_sampling_v1",
            "root_config_sha256": root_config.get("config_source_sha256"),
            "sampling_sha256": hash_json(values)}


def unsupported_request(request):
    unsupported = [k for k in UNSUPPORTED if request.get(k) not in (None, False, 0, [], {}, "")]
    if request.get("prompt_logprobs") is not None: unsupported.append("prompt_logprobs")
    if request.get("n", 1) != 1: unsupported.append("n")
    return unsupported


def effective_policy(source, sampling, request=None):
    """Explicit overrides win. Absent historical runtime defaults remain estimates."""
    generation = source.get("generation", {})
    request = request or {}
    extra = request.get("extra_body") or {}
    merged = {**generation, **request, **extra}
    unsupported = unsupported_request(merged)
    if (source.get("steering") or {}).get("enabled"): unsupported.append("steering")
    for key in ("adapter", "lora", "lora_request", "quantization", "speculative_config"):
        if source.get("model", {}).get(key) or source.get("serving", {}).get(key): unsupported.append(key)
    if unsupported: raise ValueError("Unsupported historical constraints: " + ", ".join(sorted(set(unsupported))))
    comparisons = {}
    for key in ROOT_KEYS:
        historical = merged.get(key)
        comparisons[key] = "matched" if historical is not None else "root_config_user_asserted_constant"
        if historical is not None and historical != sampling[key]:
            raise ValueError(f"Historical/root sampling conflict: {key}: {historical} != {sampling[key]}")
    policy = {k: sampling[k] for k in ROOT_KEYS}
    if policy["temperature"] == 0:
        raise ValueError("Greedy sampling probability semantics are not supported by this replay backend")
    policy.update({k: merged.get(k, v) for k, v in DEFAULTS.items()})
    policy["seed"] = merged.get("seed")
    defaults = [k for k in DEFAULTS if k not in merged]
    return {"parameters": policy, "historical_comparison": comparisons,
            "unknown_historical_defaults": defaults, "policy_sha256": hash_json(policy),
            "historical_runtime_status": "not_archived", "status": "replay_estimate"}


def validate_values(values, count, *, allow_zero_support=True, tolerance=1e-6):
    if len(values) != count: raise ValueError("Replay output count differs from native completion")
    for index, value in enumerate(values):
        value = float(value)
        if math.isnan(value) or value == math.inf or value > tolerance:
            raise ValueError(f"Invalid processed logprob at position {index}")
        if value == -math.inf and not allow_zero_support:
            raise ValueError(f"Zero-support token at position {index}")
    return [float(v) for v in values]


def verify_identity(row, output, values):
    if output.get("token_ids") != row["completion_token_ids"]:
        raise ValueError("Trace replay output token IDs differ from native completion")
    if len(values) != len(row["completion_token_ids"]):
        raise ValueError("Trace replay output length differs from native completion")
    return True


def target_logprobs(request_output, completion):
    """vLLM RequestOutput -> generated-token values, indexed by token ID."""
    if not request_output.finished or len(request_output.outputs) != 1:
        raise ValueError("Trace replay did not return exactly one finished output")
    output = request_output.outputs[0]
    if list(output.token_ids) != completion:
        raise ValueError("Trace replay output token IDs differ from native completion")
    if output.logprobs is None or len(output.logprobs) != len(completion):
        raise ValueError("Trace replay missing per-token processed logprobs")
    values = []
    for token, entries in zip(completion, output.logprobs):
        if entries is None or token not in entries:
            raise ValueError("Trace replay target token missing from logprob entries")
        values.append(entries[token].logprob)
    return validate_values(values, len(completion))
