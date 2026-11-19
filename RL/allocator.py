"""Torch-free allocator setup; called before CUDA in parent and spawn workers."""
import os
import sys
from contextlib import contextmanager
import warnings

ALIASES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
HOST_POLICY_VERSION = "scoped_host_reclaim_v2"
HOST_NATIVE_CACHE_MB = 1


@contextmanager
def capture_allocator_warnings():
    """Retain detected allocator warnings while preserving ordinary stderr."""
    records = []
    with warnings.catch_warnings():
        original = warnings.showwarning
        def show(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if ("allocator" in text.lower() or "expandable_segments" in text.lower()) and len(records) < 64:
                records.append({"message": text, "category": category.__name__})
            original(message, category, filename, lineno, file=file, line=line)
        warnings.showwarning = show
        yield records


def _parse(value):
    # Commas inside bracketed size classes are part of a single option.
    parts, start, depth = [], 0, 0
    for index, char in enumerate(value):
        if char == "[": depth += 1
        elif char == "]": depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index]); start = index + 1
        if depth < 0: raise ValueError("Malformed CUDA allocator settings")
    if depth: raise ValueError("Malformed CUDA allocator settings")
    parts.append(value[start:])
    result = {}
    for part in parts:
        if not part.strip(): continue
        key, separator, val = part.partition(":")
        key, val = key.strip(), val.strip()
        if not separator or not key or not val:
            raise ValueError("Malformed CUDA allocator option: " + part)
        if key in result and result[key].lower() != val.lower():
            raise ValueError("Conflicting CUDA allocator option: " + key)
        result[key] = val
    return result


def configure_cuda_allocator(settings, *, apply=True, host_offload=None):
    original = {name: os.environ.get(name) for name in ALIASES}
    options = {}
    for value in original.values():
        for key, val in _parse(value or "").items():
            if key in options and options[key].lower() != val.lower():
                raise ValueError("CUDA allocator aliases conflict: " + key)
            options[key] = val
    requested = {"expandable_segments": str(settings.expandable_segments)}
    if host_offload is not None and host_offload.enabled:
        # Small native blocks remain cacheable (including scalar readbacks).
        # The managed pool requests strictly larger blocks, owns their reuse,
        # and can return them without leaving large native cached allocations.
        requested.update(pinned_max_cached_size_mb=str(HOST_NATIVE_CACHE_MB),
                         pinned_max_round_threshold_mb=str(HOST_NATIVE_CACHE_MB),
                         pinned_use_background_threads="False", pinned_use_cuda_host_register="False")
    if settings.backend: requested["backend"] = settings.backend
    for key, val in requested.items():
        if key in options and options[key].lower() != val.lower():
            raise ValueError("CUDA allocator " + key + " conflicts with training config")
        options[key] = val
    backend = options.get("backend", "native")
    if settings.expandable_segments and backend != "native":
        raise ValueError("expandable_segments requires the native allocator backend; found " + backend)
    normalized = ",".join(f"{key}:{value}" for key, value in sorted(options.items()))
    if apply:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized() and any(
                original[name] != normalized for name in ALIASES):
            raise RuntimeError("CUDA allocator must be configured before CUDA initialization")
        for name in ALIASES: os.environ[name] = normalized
    return {"original_environment": original, "normalized": normalized, "backend_requested": backend,
            "configured": apply, "expandable_segments_requested": settings.expandable_segments,
            "effect_verification": "configuration_passed; runtime support is not independently proven",
            "support_warnings": [], "warning_capture": "PyTorch warnings remain in training stderr"}
