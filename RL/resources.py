"""Torch-free host budgeting and dry-run memory estimates (not capacity claims)."""

import json
import os
from pathlib import Path


def host_memory():
    values = {}
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(value.split()[0]) * 1024
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmLck"}:
                values[key] = int(value.split()[0]) * 1024
    return values


def memory_plan(config):
    model = json.loads((Path(config.model.path) / "config.json").read_text())
    text = model["text_config"]
    length = config.validation.max_sequence_length
    settings = config.training.memory
    logits_tokens = min(length, settings.loss_chunk_tokens) if settings.loss_chunk_tokens else length
    boundary = length * text["hidden_size"] * 2 * (text["num_hidden_layers"] + 2)
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    budget = int(settings.host_budget_gib * 2**30)
    required = int(boundary * 1.5)  # Concurrent graphs are bounded to one microbatch per rank.
    host = host_memory()
    if settings.activation_offload != "none":
        if required > budget:
            raise ValueError(f"Offload activation estimate {required} exceeds per-rank host budget {budget}")
        available = host.get("MemAvailable")
        if available is None or available < budget * world_size + max(8 * 2**30, available // 10):
            raise ValueError("Insufficient available host RAM for all local ranks' offload budgets and reserve")
    return {"strategy": settings.model_dump(), "checkpoint_policy": config.training.checkpoint_policy,
            "first_order_custom_autograd": bool(settings.loss_chunk_tokens or settings.mlp_chunk_tokens),
            "temporary_fp32_logits_bytes": logits_tokens * text["vocab_size"] * 4,
            "full_sequence_fp32_logits_bytes": length * text["vocab_size"] * 4,
            "dense_bool_mask_bytes_if_fallback": length**2,
            "boundary_activation_estimate_bytes_per_rank": required,
            "host_budget_bytes_per_rank": budget, "local_world_size": world_size,
            "host": host, "capacity_verified": False,
            "attention_requirement": "Flash SDPA forward/backward; no math fallback" if
            settings.attention_policy == "flash_only" else "Profile dispatch; auto may fall back"}
