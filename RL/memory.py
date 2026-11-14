"""Composable first-order memory backends; parameter names/objects stay unchanged.

Only token positions in the MLP and output projection are chunked. Attention always
sees the complete causal sequence. No parameter gradients or collectives are written
here; autograd owns both. FSDP, compilation and higher-order gradients are unsupported.
"""

import contextlib
import json
import math
import os
import time
import weakref
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def local_saved_tensors():
    """Short-lived recomputation must never inherit activation offload hooks."""
    return torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, lambda tensor: tensor)


def autocast_state(tensor):
    device = tensor.device.type
    return device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device)


def autocast_context(state):
    device, enabled, dtype = state
    return torch.autocast(device_type=device, enabled=enabled, dtype=dtype)


def target_coordinates(labels):
    """Shared CE/GRPO causal coordinates: [batch, predicting logit position]."""
    return (labels[:, 1:] != -100).nonzero(as_tuple=False)


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, weight, bias, chunk_tokens):
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError("Chunked CE requires a frozen output head")
        if chunk_tokens <= 0 or hidden.shape[:-1] != labels.shape:
            raise ValueError("Invalid CE block size or hidden/label shapes")
        # Shift once, preserving the full backbone graph through the hidden input.
        valid = target_coordinates(labels)
        ctx.save_for_backward(hidden, labels, valid)
        ctx.weight, ctx.bias = weight, bias
        ctx.versions = (weight._version, bias._version if bias is not None else None)
        ctx.chunk_tokens = chunk_tokens
        ctx.autocast = autocast_state(hidden)
        loss = hidden.new_zeros((), dtype=torch.float32)
        for block in valid.split(chunk_tokens):
            b, position = block.unbind(1)
            logits = F.linear(hidden[b, position], weight, bias)
            loss.add_(F.cross_entropy(logits.float(), labels[b, position + 1], reduction="sum"))
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        if torch.is_grad_enabled():
            raise RuntimeError("Chunked CE supports first-order gradients only")
        if ctx.versions != (ctx.weight._version, ctx.bias._version if ctx.bias is not None else None):
            raise RuntimeError("Frozen output head changed between forward and backward")
        hidden, labels, valid = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden)
        for block in valid.split(ctx.chunk_tokens):
            b, position = block.unbind(1)
            with torch.enable_grad(), local_saved_tensors(), autocast_context(ctx.autocast):
                part = hidden[b, position].detach().requires_grad_(True)
                logits = F.linear(part, ctx.weight, ctx.bias)
                loss = F.cross_entropy(logits.float(), labels[b, position + 1], reduction="sum")
                (gradient,) = torch.autograd.grad(loss, part, grad_outputs=grad_loss)
            grad_hidden[b, position] = gradient
        return grad_hidden, None, None, None, None


def chunked_causal_ce(hidden, labels, head, chunk_tokens):
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("Chunked CE supports a frozen nn.Linear output head only")
    return _ChunkedCE.apply(hidden, labels, head.weight, head.bias, chunk_tokens)


def exact_length_batch(batch):
    """Trim only a single, right-padded sequence; refuse ambiguous masks."""
    mask = batch["attention_mask"]
    if mask.shape[0] != 1:
        raise ValueError("exact_length currently supports microbatch=1 only")
    length = int(mask.sum())
    expected = torch.arange(mask.shape[1], device=mask.device) < length
    if not torch.equal(mask[0].bool(), expected) or length < 2:
        raise ValueError("exact_length requires contiguous right padding and at least two tokens")
    if (batch["labels"][:, length:] != -100).any():
        raise ValueError("Cannot trim supervised padding")
    if "loss_mask" in batch and batch["loss_mask"][:, length:].any():
        raise ValueError("Cannot trim selected loss_mask padding")
    return {key: tensor[:, :length] for key, tensor in batch.items()}


def text_backbone(model):
    raw = model.get_base_model() if hasattr(model, "get_base_model") else model
    base = raw.base_model
    return base.language_model if hasattr(base, "language_model") else base


def attention_context(policy):
    if policy == "flash_only":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    return contextlib.nullcontext()


class LossBackend:
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.tokens = self.blocks = 0
        self.sequence_lengths = []

    def validate_prepared_batch(self, batch):
        """Fail on exact-length constraints before any backward in the window."""
        if self.settings.exact_length:
            exact_length_batch(batch)

    def __call__(self, model, batch):
        if "loss_mask" in batch:
            raise ValueError("Resolve raw loss_mask with prepare_batch/train before the loss backend")
        if self.settings.exact_length:
            batch = exact_length_batch(batch)
        count = int((batch["labels"][:, 1:] != -100).sum())
        self.sequence_lengths.append(batch["input_ids"].shape[1])
        if not self.settings.loss_chunk_tokens:
            from RL.validation import causal_ce
            self.tokens += batch["input_ids"].numel()
            self.blocks += 1
            return causal_ce(model, batch)
        hidden = text_backbone(model)(input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
        head = model.get_output_embeddings()
        loss = chunked_causal_ce(hidden, batch["labels"], head, self.settings.loss_chunk_tokens)
        self.tokens += count
        self.blocks += math.ceil(count / self.settings.loss_chunk_tokens)
        return loss, count


def parameter_signature(module):
    return tuple((name, id(p), tuple(p.shape), p.requires_grad, p._version)
                 for name, p in module.named_parameters())


class _PositionMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, backend, *parameters):
        ctx.backend, ctx.parameters = backend, parameters
        ctx.signature = parameter_signature(backend)
        ctx.autocast = autocast_state(hidden)
        ctx.save_for_backward(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        output = None
        for start in range(0, flat.shape[0], backend.chunk_tokens):
            part = backend.dense(flat[start:start + backend.chunk_tokens])
            if output is None:
                output = part.new_empty(flat.shape)
            output[start:start + part.shape[0]].copy_(part)
            backend.forward_blocks += 1
        return output.reshape_as(hidden)

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("Position-chunked MLP supports first-order gradients only")
        backend = ctx.backend
        if parameter_signature(backend) != ctx.signature:
            raise RuntimeError("MLP parameter version, identity, shape or trainable inventory changed")
        (hidden,) = ctx.saved_tensors
        flat = hidden.reshape(-1, hidden.shape[-1])
        grad_output = gradient.reshape_as(flat)
        grad_hidden = torch.empty_like(flat)
        grad_parameters = [None] * len(ctx.parameters)
        for start in range(0, flat.shape[0], backend.chunk_tokens):
            with torch.enable_grad(), local_saved_tensors(), autocast_context(ctx.autocast):
                part = flat[start:start + backend.chunk_tokens].detach().requires_grad_(True)
                output = backend.dense(part)
                grads = torch.autograd.grad(output, (part, *ctx.parameters),
                                            grad_outputs=grad_output[start:start + part.shape[0]],
                                            allow_unused=False)
            grad_hidden[start:start + part.shape[0]].copy_(grads[0])
            for index, grad in enumerate(grads[1:]):
                if grad_parameters[index] is None:
                    grad_parameters[index] = grad
                else:
                    grad_parameters[index].add_(grad)
            backend.backward_blocks += 1
        return grad_hidden.reshape_as(hidden), None, *grad_parameters


class PositionChunkedMLP(torch.nn.Module):
    """Registered backend preserving the original child names and Parameter objects."""
    def __init__(self, original, chunk_tokens):
        super().__init__()
        # Keep the original module out of the registered tree: shared children retain
        # the exact PEFT export paths. Dense execution still uses the original class.
        self.__dict__.update(original.__dict__.copy())
        object.__setattr__(self, "dense", original)
        self.chunk_tokens = chunk_tokens
        self.forward_blocks = self.backward_blocks = 0
        if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in original.modules()):
            raise ValueError("Position-chunked MLP requires dropout=0")
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError("Position-chunked MLP requires trainable adapter parameters")

    def forward(self, hidden):
        parameters = tuple(p for p in self.parameters() if p.requires_grad)
        return _PositionMLP.apply(hidden, self, *parameters)


def install_mlp_backend(model, chunk_tokens):
    if not chunk_tokens:
        return []
    before = [(name, id(p), p.requires_grad, tuple(p.shape)) for name, p in model.named_parameters()]
    backends = []
    for layer in text_backbone(model).layers:
        backend = PositionChunkedMLP(layer.mlp, chunk_tokens)
        layer.mlp = backend
        backends.append(backend)
    after = [(name, id(p), p.requires_grad, tuple(p.shape)) for name, p in model.named_parameters()]
    if before != after:
        raise RuntimeError("MLP backend changed parameter names, identities, shapes or freezing")
    return backends


class BoundaryOffload:
    """Offload only non-reentrant checkpoint input activations, never model weights."""
    def __init__(self, model, budget_bytes, *, enabled=True):
        self.budget_bytes = budget_bytes
        self.enabled = enabled
        self.parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self.live_bytes = self.peak_bytes = 0
        self.reset()

    def reset(self):
        self.d2h_bytes = self.h2d_bytes = 0
        self.pack_seconds = self.unpack_seconds = 0.0
        self.events = {"d2h": [], "h2d": []}
        self.peak_bytes = self.live_bytes
        self.budget_rejection = None

    def _release(self, size):
        self.live_bytes -= size

    def _copy(self, kind, callback):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = callback()
        end.record()
        self.events[kind].append((start, end))
        return result

    def context(self):
        native = torch.autograd.graph.save_on_cpu(pin_memory=True)

        def pack(tensor):
            # Non-reentrant checkpoint's input saver also saves an empty sentinel.
            # Only rank-3 hidden-state boundaries can be moved; weights stay resident.
            if (tensor.device.type != "cuda" or tensor.ndim != 3 or not tensor.numel()
                    or tensor.untyped_storage().data_ptr() in self.parameter_storages):
                return False, tensor
            size = tensor.numel() * tensor.element_size()
            if self.live_bytes + size > self.budget_bytes:
                self.budget_rejection = {"requested_bytes": size, "live_bytes": self.live_bytes,
                                         "budget_bytes": self.budget_bytes}
                raise RuntimeError("Pinned activation host budget exceeded; offload was not disabled")
            started = time.monotonic()
            packed = self._copy("d2h", lambda: native.pack_hook(tensor))
            self.pack_seconds += time.monotonic() - started
            self.d2h_bytes += size
            self.live_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            weakref.finalize(packed[1], self._release, size)
            return True, packed

        def unpack(packed):
            moved, value = packed
            if not moved:
                return value
            started = time.monotonic()
            tensor = self._copy("h2d", lambda: native.unpack_hook(value))
            self.unpack_seconds += time.monotonic() - started
            self.h2d_bytes += tensor.numel() * tensor.element_size()
            return tensor

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def checkpoint(self, function, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint
        if not self.enabled:
            return checkpoint(function, *args, use_reentrant=False, **kwargs)
        with self.context():
            return checkpoint(function, *args, use_reentrant=False, **kwargs)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def metrics(self):
        # The training core synchronizes before calling the update callback.
        values = {"offload_d2h_bytes": self.d2h_bytes, "offload_h2d_bytes": self.h2d_bytes,
                  "offload_pinned_peak_bytes": self.peak_bytes, "offload_live_bytes": self.live_bytes,
                  "offload_pack_wall_seconds": self.pack_seconds,
                  "offload_unpack_wall_seconds": self.unpack_seconds,
                  "offload_host_budget_bytes": self.budget_bytes,
                  "offload_budget_rejection": self.budget_rejection}
        for kind, events in self.events.items():
            try:
                seconds = sum(start.elapsed_time(end) for start, end in events) / 1000
            except RuntimeError as exc:
                values[f"offload_{kind}_timing_error"] = str(exc)
                seconds = 0.
            values[f"offload_{kind}_device_seconds"] = seconds
            values[f"offload_{kind}_GiB_per_second"] = (
                getattr(self, kind + "_bytes") / 2**30 / seconds if seconds else None)
        return values


def is_cuda_oom(exc):
    """Only the explicit CUDA allocation exception is recoverable."""
    return isinstance(exc, torch.OutOfMemoryError)


class MemoryEventLogger:
    """Append scalar JSON evidence without retaining tensors or exception objects."""
    def __init__(self, directory, *, attempt_id="", rank=0, world_size=1,
                 physical_gpu=None, enabled=True, device="cpu"):
        self.enabled, self.device = enabled, torch.device(device)
        self.identity = {"schema_version": 1, "attempt_id": attempt_id,
                         "rank": rank, "world_size": world_size, "physical_gpu": physical_gpu,
                         "logical_device": str(self.device)}
        self.path = Path(directory) / "memory-events.jsonl"
        if enabled: self.path.parent.mkdir(parents=True, exist_ok=True)

    def memory(self):
        if self.device.type != "cuda": return {}
        try:
            free, total = torch.cuda.mem_get_info(self.device)
            stats = torch.cuda.memory_stats(self.device)
            return {"allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                    "device_free_bytes": int(free), "device_total_bytes": int(total),
                    "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
                    "allocation_retries": stats.get("num_alloc_retries"), "allocator_ooms": stats.get("num_ooms")}
        except Exception as exc:
            return {"memory_sample_error": f"{type(exc).__name__}: {exc}"}

    def write(self, *, event, phase=None, **fields):
        if not self.enabled: return
        item = {**self.identity, "observed_at": datetime.now(timezone.utc).isoformat(),
                "event": event, "phase": phase, **fields}
        # No default=str: accidental Tensor payloads are an implementation error.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, allow_nan=False) + "\n")
            handle.flush()


class MemoryController:
    """Fixed per-attempt offload policy and one device-wide phase sampler.

    Phase intervals do not overlap. Every reset is preceded by a sample and its
    peak is folded into attempt totals. Branch samples aggregate by group/phase;
    no per-layer or per-token records are emitted.
    """
    def __init__(self, config, directory, *, rank=0, world_size=1, physical_gpu=None,
                 offloader=None, device="cpu"):
        self.config, self.offloader = config, offloader
        self.rank, self.world_size = rank, world_size
        self.device = torch.device(device)
        memory = config.training.memory
        self.mode = memory.activation_offload
        self.max_retries = memory.oom_retry.max_retries if memory.oom_retry.enabled else 0
        self.logger = MemoryEventLogger(directory, attempt_id=config.run.attempt_id, rank=rank,
            world_size=world_size, physical_gpu=physical_gpu, device=device,
            enabled=memory.memory_telemetry.enabled)
        self.last_decision = None
        self.context = {}
        self.phase_name = None
        self.fields = {}
        self.records = {}
        self.peaks = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        self.baseline = {}

    def decide(self, window, *, global_step, group_cursor=0, parallel=None):
        auto = self.config.training.memory.auto_offload
        local = []
        for index, group in enumerate(window):
            if index % self.world_size != self.rank: continue
            prompts = [len(v) for v in group.prompt_token_ids]
            completions = [len(v) for v in group.completion_token_ids]
            row = {"group_id": group.group_id, "group_ordinal": group_cursor + index,
                   "window_ordinal": index, "prompt_tokens": max(prompts, default=0),
                   "completion_min": min(completions, default=0),
                   "completion_max": max(completions, default=0), "completion_sum": sum(completions),
                   "max_sequence_tokens": max((p+c for p,c in zip(prompts, completions)), default=0),
                   "branch_count": group.group_size}
            reasons = []
            if row["prompt_tokens"] >= auto.prompt_tokens_threshold: reasons.append("prompt_tokens_threshold")
            if row["max_sequence_tokens"] >= auto.sequence_tokens_threshold: reasons.append("sequence_tokens_threshold")
            local.append({**row, "reasons": reasons})
        payload = {"rank": self.rank, "groups": local, "mode": self.mode}
        peers = parallel.control_step(global_step + 1, 0, "risk", payload) if parallel else [payload]
        if any(peer["mode"] != self.mode for peer in peers):
            raise ValueError("Distributed offload configurations differ")
        trigger_ranks = [peer["rank"] for peer in peers if any(g["reasons"] for g in peer["groups"])]
        enabled = self.mode == "checkpoint_boundary" or (self.mode == "auto" and bool(trigger_ranks))
        self.last_decision = {"mode": "checkpoint_boundary" if enabled else "none", "configured_mode": self.mode,
            "trigger_ranks": trigger_ranks if self.mode == "auto" else [], "ranks": peers,
            "thresholds": auto.model_dump() if hasattr(auto, "model_dump") else vars(auto),
            "policy_version": "global_step_length_or_v1", "capacity_prediction": None,
            "headroom_usage": "recorded_only_no_capacity_model"}
        if self.offloader: self.offloader.set_enabled(enabled)
        return self.last_decision

    def begin_attempt(self, window, *, global_step, attempt_id):
        self.context = {"global_step": global_step, "global_step_semantics": "committed",
                        "inflight_step": global_step + 1, "step_attempt": attempt_id, "committed": False}
        self.fields, self.records = {}, {}
        self.peaks = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        self.phase_name = None
        self.baseline = self.logger.memory()
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        self.logger.write(event="retry_start" if attempt_id else "step_start", phase="step_start",
                          **self.context, **self.baseline, offload=self.last_decision,
                          phase_timing="existing_cuda_sync_boundaries_monotonic_wall",
                          extra_cuda_synchronizations=False)
        self.phase("step_start")

    def _sample(self):
        sample = self.logger.memory()
        for key in self.peaks: self.peaks[key] = max(self.peaks[key], sample.get(key, 0))
        if self.phase_name is not None:
            key = (self.fields.get("group_ordinal"), self.phase_name)
            record = self.records.setdefault(key, {"phase": self.phase_name, **self.fields,
                                                   "seconds": 0., "samples": 0,
                                                   "peak_allocated_bytes": 0, "peak_reserved_bytes": 0})
            record["seconds"] += time.monotonic() - self.phase_started
            record["samples"] += 1
            for metric in self.peaks: record[metric] = max(record[metric], sample.get(metric, 0))
            record["last_memory"] = sample
        return sample

    def phase(self, name, **fields):
        self._sample()
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        self.phase_name, self.phase_started = name, time.monotonic()
        self.fields.update(fields)

    def flush_phases(self):
        self._sample()
        if self.logger.enabled and self.config.training.memory.memory_telemetry.granularity == "phase":
            for row in self.records.values():
                self.logger.write(event="phase_end", **self.context, offload=self.last_decision, **row)
        self.records = {}
        self.phase_name = None

    def pause(self):
        # Waiting for peers must not inflate the final local compute phase.
        self._sample()
        self.phase_name = None

    def offload_metrics(self):
        return self.offloader.metrics() if self.offloader else {"offload_d2h_bytes": 0, "offload_h2d_bytes": 0,
                                                               "offload_live_bytes": 0, "offload_pinned_peak_bytes": 0}

    def result(self, statuses):
        self.flush_phases()
        sample = self.logger.memory()
        deltas = {key + "_delta": (sample.get(key, 0) or 0) - (self.baseline.get(key, 0) or 0)
                  for key in ("allocation_retries", "allocator_ooms")}
        self.logger.write(event="attempt_result", phase="local_compute_end", **self.context,
                          **sample, attempt_peaks=self.peaks, offload=self.last_decision,
                          offload_metrics=self.offload_metrics(), rank_results=statuses, **deltas)

    def oom(self, error, *, phase):
        # Sample before resetting peaks or collecting garbage. The failed requested
        # allocation is only available in the preserved original error text.
        self.logger.write(event="oom", phase=phase, **self.context, **self.logger.memory(),
                          **self.fields, error=error, allocation_parse="best_effort_original_text_only",
                          offload=self.last_decision)

    def cleanup(self, *, before, statuses):
        self.logger.write(event="oom_cleanup", phase="oom_cleanup", **self.context,
                          cleanup_stage="before" if before else "after", **self.logger.memory(),
                          offload_metrics=self.offload_metrics(), rank_results=statuses)

    def retry(self, window, *, global_step, attempt_id):
        if self.offloader is None: raise RuntimeError("OOM retry lacks an offloader")
        self.offloader.set_enabled(True)
        self.last_decision = {**self.last_decision, "mode": "checkpoint_boundary", "reason": "cuda_oom_retry"}

    def committed(self, *, global_step, window, attempt_id):
        self.flush_phases()
        self.logger.write(event="step_commit", phase="step_commit",
                          **{**self.context, "global_step": global_step, "committed": True},
                          **self.logger.memory(), attempt_peaks=self.peaks,
                          offload=self.last_decision, offload_metrics=self.offload_metrics())

def configure_checkpointing(model, config, budget_bytes):
    from RL.modeling import enable_checkpointing
    offloader = None
    memory = config.training.memory
    if (memory.activation_offload != "none" or memory.oom_retry.enabled):
        # Keep a controller available for an auto decision or an OOM retry even
        # when the first attempt intentionally stays entirely on device.
        offloader = BoundaryOffload(model, budget_bytes,
                                    enabled=memory.activation_offload == "checkpoint_boundary")
    if config.training.checkpoint_policy == "layer":
        enable_checkpointing(model)
        if offloader:
            for layer in text_backbone(model).layers:
                layer._gradient_checkpointing_func = offloader.checkpoint
    else:
        model.gradient_checkpointing_disable()
        model.enable_input_require_grads()
        if config.training.checkpoint_policy == "layer_group":
            raw = model.get_base_model() if hasattr(model, "get_base_model") else model
            original = text_backbone(model)
            before = [(name, id(p)) for name, p in model.named_parameters()]
            raw.base_model.language_model = GroupedTextBackbone(
                original, config.training.checkpoint_group_layers, offloader)
            if before != [(name, id(p)) for name, p in model.named_parameters()]:
                raise RuntimeError("Grouped checkpoint backend changed PEFT parameter paths or identities")
    return offloader


class GroupedTextBackbone(torch.nn.Module):
    """Explicit Qwen3.5 text backend for group checkpoint scheduling.

    This is a registered model adapter, not an attention monkey patch. It retains
    Qwen's position/mask helpers and exact child tree. Only text, no KV cache, and
    the pinned Transformers Qwen3.5 interface are supported.
    """
    def __init__(self, original, group_layers, offloader):
        super().__init__()
        self.__dict__.update(original.__dict__.copy())
        self.group_layers, self.offloader = group_layers, offloader

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=False, **kwargs):
        from torch.utils.checkpoint import checkpoint
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        if use_cache or past_key_values is not None:
            raise ValueError("Grouped training backend does not support KV caches")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device).view(1, 1, -1).expand(4, hidden.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_positions, rotary_positions = position_ids[0], position_ids[1:]
        else:
            text_positions, rotary_positions = None, position_ids
        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            mask_args = dict(config=self.config, inputs_embeds=hidden, attention_mask=attention_mask,
                             past_key_values=None, position_ids=text_positions)
            masks = {"full_attention": qwen.create_causal_mask(**mask_args),
                     "linear_attention": qwen.create_recurrent_attention_mask(**mask_args)}
        position_embeddings = self.rotary_emb(hidden, rotary_positions)
        layers = self.layers[:self.config.num_hidden_layers]
        for start in range(0, len(layers), self.group_layers):
            group = tuple(layers[start:start + self.group_layers])
            # Bind each closure now; non-reentrant backward runs after this loop.
            def run_group(value, group=group):
                for layer in group:
                    value = layer(value, position_embeddings=position_embeddings,
                                  attention_mask=masks[layer.block_type], position_ids=text_positions,
                                  past_key_values=None, use_cache=False, **kwargs)
                return value
            if self.training and torch.is_grad_enabled():
                hidden = (self.offloader.checkpoint(run_group, hidden) if self.offloader else
                          checkpoint(run_group, hidden, use_reentrant=False))
            else:
                hidden = run_group(hidden)
        return qwen.Qwen3_5ModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=None)
