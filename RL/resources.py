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
    # Resolve this process's actual cgroup, including ancestor limits. Host-global
    # MemAvailable and cgroup remaining are separate constraints, never summed.
    values.update(cgroup_memory())
    return values


def cgroup_memory():
    try:
        line = next(x for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
        root = Path("/sys/fs/cgroup")
        path = root / line.split(":", 2)[2].lstrip("/")
        limits, remaining = [], []
        while path == root or root in path.parents:
            maximum = (path / "memory.max").read_text().strip()
            if maximum != "max":
                limit = int(maximum)
                limits.append(limit)
                remaining.append(max(0, limit - int((path / "memory.current").read_text())))
            if path == root: break
            path = path.parent
        return {"cgroup_limit_bytes": min(limits) if limits else None,
                "cgroup_remaining_bytes": min(remaining) if remaining else None,
                "cgroup_capability": "v2"}
    except (OSError, StopIteration, ValueError):
        return {"cgroup_limit_bytes": None, "cgroup_remaining_bytes": None,
                "cgroup_capability": "unavailable"}


def check_host_capacity(settings, *, requested_bytes=0, world_size=1, host=None):
    """Cooperative admission, not an OS reservation against concurrent allocators."""
    host = host_memory() if host is None else host
    total, available = host.get("MemTotal"), host.get("MemAvailable")
    if total is None or available is None:
        raise RuntimeError("Host memory capacity unavailable for managed offload")
    reserve = max(int(settings.reserve_gib * 2**30), int(total * settings.reserve_fraction))
    required = requested_bytes * world_size
    remaining = host.get("cgroup_remaining_bytes")
    if available < required + reserve or (remaining is not None and remaining < required + reserve):
        raise RuntimeError("Managed offload host safety reserve exhausted before allocation")
    return {"host_global": host, "reserve_bytes": reserve, "requested_bytes_all_local_ranks": required,
            "local_world_size": world_size, "hard_os_reservation": False}


def memory_plan(config):
    model = json.loads((Path(config.model.path) / "config.json").read_text())
    text = model["text_config"]
    length = config.validation.max_sequence_length
    settings = config.training.memory
    logits_tokens = min(length, settings.loss_chunk_tokens) if settings.loss_chunk_tokens else length
    element_bytes = 4 if config.model.dtype == "float32" else 2
    boundary = length * text["hidden_size"] * element_bytes * (text["num_hidden_layers"] + 2)
    world_size = len(getattr(config.runtime, "gpus", None) or [config.runtime.gpu])
    budget = int(settings.host_budget_gib * 2**30)
    required = int(boundary * 1.5)  # Concurrent graphs are bounded to one microbatch per rank.
    host = host_memory()
    managed = settings.host_offload
    admission = None
    if managed.enabled:
        admission = check_host_capacity(managed,
            requested_bytes=int(managed.allocator_budget_gib * 2**30), world_size=world_size, host=host)
    if config.preflight.check_resources and (settings.activation_offload != "none" or settings.oom_retry.enabled):
        if required > budget:
            raise ValueError(f"Offload activation estimate {required} exceeds per-rank host budget {budget}")
        available = host.get("MemAvailable")
        if not managed.enabled and (available is None or available < budget * world_size + max(8 * 2**30, available // 10)):
            raise ValueError("Insufficient available host RAM for all local ranks' offload budgets and reserve")
    adam = settings.adam_moment_offload
    return {"strategy": settings.model_dump(), "checkpoint_policy": config.training.checkpoint_policy,
            "adam_moment_offload": {"enabled": adam.mode != "disabled", "backend": adam.backend,
                "rank_cpu_state_limit_bytes": int(adam.host_state_budget_gib * 2**30),
                "node_cpu_state_limit_bytes": int(adam.node_state_budget_gib * 2**30),
                "pinned_staging_is_subset_of_allocator_budget": True,
                "host_state_is_durable_not_cache": True,
                "capacity_verified": False},
            "model_dtype": config.model.dtype, "activation_element_bytes": element_bytes,
            "first_order_custom_autograd": bool(settings.loss_chunk_tokens or settings.mlp_chunk_tokens),
            "temporary_fp32_logits_bytes": logits_tokens * text["vocab_size"] * 4,
            "full_sequence_fp32_logits_bytes": length * text["vocab_size"] * 4,
            "dense_bool_mask_bytes_if_fallback": length**2,
            "boundary_activation_estimate_bytes_per_rank": required,
            "host_budget_bytes_per_rank": budget, "local_world_size": world_size,
            "host_allocator_budget_bytes_per_rank": int(managed.allocator_budget_gib * 2**30) if managed.enabled else None,
            "host_admission": admission,
            "host": host, "capacity_verified": False,
            "attention_requirement": "Flash SDPA forward/backward; no math fallback" if
            settings.attention_policy == "flash_only" else "Profile dispatch; auto may fall back"}
