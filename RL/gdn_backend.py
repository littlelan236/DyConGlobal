"""Explicit Gated Delta Net backend binding for the pinned Qwen model.

The binding is installed once per worker, before model construction.  It
patches only the two GDN chunk/recurrent symbols used by Qwen's native layer;
causal convolution and Hub kernel dispatch remain on their verified reference
paths.  A failed FLA import or capability check is fatal by design.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import functools
import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


TRANSFORMERS_VERSION = "5.17.0"
QWEN_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
# Updated when the pinned Transformers source changes.  Keeping this explicit
# makes an accidental environment upgrade fail at binding time.
QWEN_SOURCE_SHA256 = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
CHUNK_SIZE = 64


class GDNBackendError(RuntimeError):
    """The requested backend cannot satisfy the execution contract."""


def _source_hash(module: Any) -> str | None:
    path = getattr(module, "__file__", None)
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _unwrap_public(function: Callable[..., Any]) -> Callable[..., Any]:
    """Unwrap only the public decorator chain, never private closure cells."""
    result = inspect.unwrap(function)
    if not callable(result):
        raise GDNBackendError(f"GDN symbol is not callable: {function!r}")
    return result


def _validate_reference(module: Any) -> dict[str, Any]:
    import transformers

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise GDNBackendError(
            f"Qwen GDN binding requires Transformers {TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__}"
        )
    actual_hash = _source_hash(module)
    if actual_hash != QWEN_SOURCE_SHA256:
        raise GDNBackendError(
            "Qwen implementation source hash differs from the pinned contract: "
            f"{actual_hash}"
        )
    required = {
        "torch_chunk_gated_delta_rule": ("query", "key", "value", "g", "beta", "initial_state"),
        "torch_recurrent_gated_delta_rule": ("query", "key", "value", "g", "beta", "initial_state"),
    }
    result: dict[str, Any] = {}
    for name, parameters in required.items():
        original = getattr(module, name, None)
        if original is None:
            raise GDNBackendError(f"Pinned Qwen module has no {name}")
        unwrapped = _unwrap_public(original)
        signature = inspect.signature(unwrapped)
        missing = [p for p in parameters if p not in signature.parameters]
        if missing:
            raise GDNBackendError(f"{name} signature is missing {missing}")
        result[name] = unwrapped
    return result


def _fla_adapter(function: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(function)
    names = set(signature.parameters)
    required = {"q", "k", "v", "g", "beta", "scale", "state_v_first",
                "use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel", "cp_context"}
    if not required.issubset(names):
        raise GDNBackendError(
            "FLA chunk_gated_delta_rule lacks the required Qwen arguments: "
            + repr(sorted(required - names))
        )
    if "initial_state" not in names or "output_final_state" not in names:
        raise GDNBackendError("FLA chunk implementation does not expose state-gradient outputs")

    def call(query, key, value, g, beta, chunk_size=CHUNK_SIZE,
             initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, **kwargs):
        import torch
        if chunk_size != CHUNK_SIZE:
            raise GDNBackendError("fla_chunk requires chunk_size=64")
        if any(x.dtype != torch.bfloat16 or x.device.type != "cuda" for x in (query, key, value)):
            raise GDNBackendError("fla_chunk requires CUDA BF16 q/k/v")
        if query.ndim != 4 or key.shape != query.shape or value.shape[:3] != query.shape[:3]:
            raise GDNBackendError("Qwen must supply matching [B,T,HV,D] q/k/v heads")
        if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
            raise GDNBackendError("GDN g/beta shape differs from q/k/v")
        if initial_state is not None and (initial_state.shape !=
                (query.shape[0], value.shape[2], key.shape[-1], value.shape[-1])
                or initial_state.dtype != torch.float32 or initial_state.device != query.device):
            raise GDNBackendError("GDN initial state must be CUDA FP32 [B,HV,K,V]")
        for name in ("cu_seqlens", "cp_context", "chunk_indices", "cu_seqlens_cpu"):
            if kwargs.pop(name, None) is not None:
                raise GDNBackendError(f"fla_chunk does not support {name}")
        # These are decoder metadata ignored by the pinned reference operator.
        for name in ("position_ids", "use_cache"):
            kwargs.pop(name, None)
        if kwargs:
            raise GDNBackendError("Unsupported GDN arguments: " + repr(sorted(kwargs)))
        return function(q=query, k=key, v=value, g=g, beta=beta,
                        scale=key.shape[-1] ** -0.5,
                        initial_state=initial_state, output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        use_beta_sigmoid_in_kernel=False, use_gate_in_kernel=False,
                        state_v_first=False, cp_context=None, chunk_size=CHUNK_SIZE)

    call.__name__ = "fla_chunk_gated_delta_rule_adapter"
    call.__module__ = function.__module__
    return call


@dataclass
class BackendBinding:
    configured_backend: str
    resolved_backend: str
    module: str
    chunk_callable: str
    recurrent_callable: str
    source_hash: str | None
    dependency_version: str | None
    state_layout: str = "[B,HV,K,V], state_v_first=false"
    state_dtype_policy: str = "preserve_boundary_dtype; explicit conversion only"
    chunk_size: int = CHUNK_SIZE
    _chunk: Callable[..., Any] = field(repr=False, compare=False, default=None)
    _recurrent: Callable[..., Any] = field(repr=False, compare=False, default=None)
    _observed_calls: Counter = field(default_factory=Counter, repr=False, compare=False)
    _observed_lengths: Counter = field(default_factory=Counter, repr=False, compare=False)
    _originals: dict = field(default_factory=dict, repr=False, compare=False)
    _source_identity: dict = field(default_factory=dict, repr=False, compare=False)

    def chunk(self, *args, **kwargs):
        self._observed_calls["chunk"] += 1
        if args:
            self._observed_lengths[int(args[0].shape[1])] += 1
        elif "query" in kwargs:
            self._observed_lengths[int(kwargs["query"].shape[1])] += 1
        if kwargs.get("initial_state") is not None:
            self._observed_calls["chunk_with_initial_state"] += 1
        if kwargs.get("output_final_state", False):
            self._observed_calls["chunk_with_final_state"] += 1
        return self._chunk(*args, **kwargs)

    def recurrent(self, *args, **kwargs):
        self._observed_calls["recurrent"] += 1
        return self._recurrent(*args, **kwargs)

    def execution_contract(self) -> dict[str, Any]:
        return {
            "version": 1,
            "configured_backend": self.configured_backend,
            "resolved_backend": self.resolved_backend,
            "chunk_implementation": {"module": self.module, "callable": self.chunk_callable,
                                      "source_sha256": self.source_hash},
            "recurrent_backend": "reference_pytorch",
            "recurrent_implementation": self.recurrent_callable,
            "dependency_version": self.dependency_version,
            "sources_and_versions": self._source_identity,
            "conv_backend": "reference_pytorch",
            "norm_backend": "native_qwen",
            "state_layout": self.state_layout,
            "state_dtype_policy": self.state_dtype_policy,
            "chunk_size": self.chunk_size,
            "qk_norm": "use_qk_l2norm_in_kernel=True as supplied by Qwen",
            "gate_beta_semantics": "g=log-space decay; beta=sigmoid(beta projection)",
            "determinism_policy": "caller requires torch deterministic algorithms; no fallback",
        }

    def observations(self) -> dict[str, Any]:
        return {"calls": dict(self._observed_calls),
                "chunk_lengths": {str(k): v for k, v in self._observed_lengths.items()}}


_LOCK = threading.RLock()
_ACTIVE: BackendBinding | None = None


def install_gdn_backend(model_config: Any, expected_identity: dict[str, Any] | None = None) -> BackendBinding:
    """Install the explicitly requested backend exactly once in this worker."""
    global _ACTIVE
    configured = getattr(model_config, "gdn_backend", model_config if isinstance(model_config, str) else None)
    if configured not in {"reference_pytorch", "fla_chunk"}:
        raise GDNBackendError(f"Unsupported gdn_backend: {configured!r}")
    if configured == "fla_chunk" and getattr(model_config, "dtype", None) != "bfloat16":
        raise GDNBackendError("fla_chunk requires model.dtype=bfloat16")
    with _LOCK:
        if _ACTIVE is not None:
            if _ACTIVE.configured_backend != configured:
                raise GDNBackendError("Cannot switch GDN backend during a live worker lifetime")
            if expected_identity is not None and expected_identity != _ACTIVE.execution_contract():
                raise GDNBackendError("Existing GDN binding does not match expected execution identity")
            return _ACTIVE
        module = importlib.import_module(QWEN_MODULE)
        reference = _validate_reference(module)
        from importlib.metadata import version
        versions = {name: version(name) for name in ("torch", "transformers", "peft")}
        sources = {"qwen": QWEN_SOURCE_SHA256}
        if configured == "reference_pytorch":
            chunk = reference["torch_chunk_gated_delta_rule"]
            resolved = "reference_pytorch"
            dependency_version = None
        else:
            try:
                fla = importlib.import_module("fla.ops.gated_delta_rule")
                candidate = getattr(fla, "chunk_gated_delta_rule")
            except Exception as exc:  # noqa: BLE001 - preserve the requested backend failure
                raise GDNBackendError("fla_chunk requested but FLA cannot be imported") from exc
            chunk = _fla_adapter(candidate)
            resolved = configured
            try:
                dependency_version = version("flash-linear-attention")
            except Exception:
                dependency_version = None
            if dependency_version is None:
                raise GDNBackendError("fla_chunk requires an installed flash-linear-attention distribution")
            if dependency_version != "0.6.0" or version("fla-core") != "0.6.0" or version("triton") != "3.7.1":
                raise GDNBackendError("fla_chunk requires flash-linear-attention/fla-core 0.6.0 and Triton 3.7.1")
            versions.update({name: version(name) for name in
                             ("flash-linear-attention", "fla-core", "triton", "einops")})
            fla_root = Path(importlib.import_module("fla.ops").__file__).parent
            fla_sources = {}
            # Freeze the whole ops/utils implementation, including state backward
            # and l2norm, rather than merely recording the exported callable name.
            for directory in (fla_root, fla_root.parent / "utils"):
                for path in sorted(directory.rglob("*.py")):
                    fla_sources[str(path.relative_to(fla_root.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
            source_digest = hashlib.sha256(json.dumps(fla_sources, sort_keys=True).encode()).hexdigest()
            if source_digest != "418533bd6dc3c055cfd7d9440a57883fd342d2b52d16ba1600c153bbd223f6aa":
                raise GDNBackendError("FLA source hash differs from the pinned 0.6.0 wheel")
            sources.update(fla_ops_utils_sha256=source_digest,
                           **{name: digest for name, digest in fla_sources.items()
                              if name.startswith("ops/gated_delta_rule/") or name in {
                                  "ops/common/chunk_delta_h.py", "ops/common/chunk_o.py",
                                  "ops/utils/l2norm.py", "ops/utils/softmax.py"}})
        recurrent = reference["torch_recurrent_gated_delta_rule"]
        # Keep causal convolution and all other Hub-dispatched operators on the
        # original reference path.  Qwen resolves these globals at call time.
        replacements = {}
        for name in ("causal_conv1d_fn", "causal_conv1d_update"):
            original = getattr(module, name, None)
            if original is not None:
                replacements[name] = _unwrap_public(original)
        binding = BackendBinding(
            configured_backend=configured, resolved_backend=resolved, module=QWEN_MODULE,
            chunk_callable=getattr(chunk, "__name__", type(chunk).__name__),
            recurrent_callable=getattr(recurrent, "__name__", type(recurrent).__name__),
            source_hash=_source_hash(module), dependency_version=dependency_version,
            _chunk=chunk, _recurrent=recurrent,
            _source_identity={"versions": versions, "sources": sources})
        # Count real calls while preserving the exact callable selected above.
        if expected_identity is not None and expected_identity != binding.execution_contract():
            raise GDNBackendError("Requested GDN execution identity does not match binding")
        replacements.update(torch_chunk_gated_delta_rule=binding.chunk,
                            torch_recurrent_gated_delta_rule=binding.recurrent)
        @functools.wraps(reference["torch_chunk_gated_delta_rule"])
        def counted_chunk(*args, **kwargs):
            return binding.chunk(*args, **kwargs)
        @functools.wraps(reference["torch_recurrent_gated_delta_rule"])
        def counted_recurrent(*args, **kwargs):
            return binding.recurrent(*args, **kwargs)
        replacements.update(torch_chunk_gated_delta_rule=counted_chunk,
                            torch_recurrent_gated_delta_rule=counted_recurrent)
        binding._originals = {name: getattr(module, name) for name in replacements}
        for name, function in replacements.items():
            setattr(module, name, function)
        _ACTIVE = binding
        return binding


def active_binding() -> BackendBinding | None:
    with _LOCK:
        return _ACTIVE


def reset_for_tests() -> None:
    """Clear the process binding after all model graphs have been released."""
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            module = importlib.import_module(QWEN_MODULE)
            for name, function in _ACTIVE._originals.items():
                setattr(module, name, function)
        _ACTIVE = None
