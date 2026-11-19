"""Pressure-selected FP32 Adam moments with bounded GPU streamed updates.

Torch 2.13 non-fused/non-foreach functional AdamW is the numerical backend.
Each tile owns storage; CPU backing is an independently releasable mmap, never
a view retaining a whole GPU moment. Transfers use one event-guarded slot.
"""
from contextlib import contextmanager
import math
import mmap
import time

import torch
from torch.optim.adamw import adamw

from RL.full_policy import local_tensor

BACKEND = "gpu_streamed_fp32_v1"
LAYOUT = 1
PAGE = mmap.PAGESIZE


class CPUBacking:
    live_mappings = 0
    max_mappings = 8192

    def __init__(self, count, admission=None):
        if CPUBacking.live_mappings >= CPUBacking.max_mappings:
            raise RuntimeError("CPU Adam mapping-count safety limit exhausted before allocation")
        self.size = math.ceil(count * 4 / PAGE) * PAGE
        self.admission = admission
        self.token = admission.reserve(self.size, "adam") if admission else None
        self.mapping = None
        self.tensor = None
        try:
            self.mapping = mmap.mmap(-1, self.size)
            CPUBacking.live_mappings += 1
            # Retain the exported memoryview in Torch's storage. Escaping tensor
            # aliases prevent unmap (BufferError), instead of use-after-unmap.
            self.tensor = torch.frombuffer(memoryview(self.mapping), dtype=torch.float32, count=count)
            self.tensor.zero_()  # Fault in and charge actual pages before commit.
            if admission:
                admission.commit(self.token)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.tensor = None
        if self.mapping is not None:
            self.mapping.close()  # Deliberately refuses surviving exported views.
            self.mapping = None
            CPUBacking.live_mappings -= 1
        if self.token is not None:
            self.admission.release(self.token)
            self.token = None


class MomentTile:
    def __init__(self, offset, count, device, *, cpu=False, admission=None):
        self.offset, self.count = offset, count
        self.bytes = count * 8
        self.device = device
        self.backing = CPUBacking(2 * count, admission) if cpu else None
        self.pair = (self.backing.tensor.view(2, count) if cpu else
                     torch.zeros((2, count), device=device, dtype=torch.float32))
        self.residency = "cpu" if cpu else "gpu"
        self.version = 0
        self.since = 0
        self.event = None


class StreamedAdamW(torch.optim.Optimizer):
    def __init__(self, groups, *, lr, settings, host_settings, pool=None, admission=None):
        if torch.__version__.split("+")[0] != "2.13.0":
            raise RuntimeError("Streamed AdamW functional backend validated for Torch 2.13.0 only")
        super().__init__(groups, dict(lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.0,
                                     amsgrad=False, maximize=False, foreach=False, fused=False,
                                     capturable=False, differentiable=False))
        self.settings, self.host_settings = settings, host_settings
        self.pool, self.admission = pool, admission
        self.device = local_tensor(self.param_groups[0]["params"][0]).device
        self.tile_elements = int(settings.tile_pair_mib * 2**20) // 8
        self.scratch = None
        self.initialized = False
        self.failed = False
        self.closed = False
        self.window = 0
        self.stable = 0
        self.pending_backings = []
        self.last_decision = None
        self.counters = {k: 0 for k in ("evict_d2h_bytes", "restore_h2d_bytes", "step_h2d_bytes",
            "step_d2h_bytes", "cpu_backing_released_bytes", "step_count", "evictions", "restores")}
        self.seconds = {k: 0.0 for k in ("control", "transfer_wait", "transfer_gpu", "host_memcpy",
                                       "adam_enqueue", "adam_gpu", "step", "branch_forward_backward_reclaim")}
        self.peak_cpu = 0
        self.observed_window_peak = 0
        self.observed_window_reserved = 0
        self.workspace_bytes_per_token = 0.0
        self.branch_observation = None
        for group in self.param_groups:
            if any(group.get(k) for k in ("amsgrad", "maximize", "foreach", "fused", "capturable", "differentiable")):
                raise ValueError("Streamed AdamW supports non-fused FP32 AdamW only")
            for p in group["params"]:
                local = local_tensor(p)
                if local.dtype != torch.float32 or not local.is_contiguous() or local.device != self.device:
                    raise ValueError("Streamed AdamW requires contiguous FP32 local shards on one device")

    def tiles(self):
        for group in self.param_groups:
            for p in group["params"]:
                for tile in self.state[p].get("tiles", ()):
                    yield p, tile

    def total_bytes(self):
        return sum(local_tensor(p).numel() * 8 for g in self.param_groups for p in g["params"])

    def cpu_bytes(self):
        return sum(t.backing.size for _, t in self.tiles() if t.backing is not None) + sum(b.size for b in self.pending_backings)

    def _new_tile(self, offset, count, cpu):
        charge = math.ceil(count * 8 / PAGE) * PAGE
        if cpu and self.cpu_bytes() + charge > int(self.settings.host_state_budget_gib * 2**30):
            raise RuntimeError("Rank Adam state budget rejected before allocation")
        return MomentTile(offset, count, self.device, cpu=cpu, admission=self.admission)

    def initialize(self, cpu_target=0):
        if self.initialized:
            return
        placed = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state.update(step=torch.tensor(0., dtype=torch.float32), tiles=[])
                size = local_tensor(p).numel()
                for offset in range(0, size, self.tile_elements):
                    count = min(self.tile_elements, size - offset)
                    tile = self._new_tile(offset, count, placed < cpu_target)
                    tile.since = self.window
                    state["tiles"].append(tile)
                    if tile.residency == "cpu":
                        placed += tile.bytes
        self.initialized = True
        self.peak_cpu = max(self.peak_cpu, self.cpu_bytes())

    def _ensure_scratch(self):
        if self.scratch is None:
            # The denominator and functional temporaries are bounded by another
            # tile; schema reserves four tile-pair sizes as a conservative cap.
            self.scratch = torch.empty((2, self.tile_elements), device=self.device, dtype=torch.float32)

    @contextmanager
    def staging(self):
        if self.device.type != "cuda":
            yield torch.empty((2, self.tile_elements), dtype=torch.float32)
            return
        if self.pool is None:
            raise RuntimeError("CUDA streamed AdamW requires the shared managed pinned pool")
        size = self.tile_elements * 8
        block = self.pool.acquire(size)
        stage = block["tensor"][:size].view(torch.float32).view(2, self.tile_elements)
        try:
            yield stage
        finally:
            # Every transfer records and waits for its event before source reuse.
            # Final stream completion also covers the last functional update.
            torch.cuda.current_stream(self.device).synchronize()
            del stage
            self.pool.release(block)
            self.pool.poll()

    def _copy(self, target, source, tile):
        started = time.monotonic()
        begin = None
        if self.device.type == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(self.device))
        target.copy_(source, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            tile.event = event
            event.synchronize()
            self.seconds["transfer_gpu"] += begin.elapsed_time(event) / 1000
            tile.event = None
        self.seconds["transfer_wait"] += time.monotonic() - started

    def _host_copy(self, target, source):
        started = time.monotonic()
        target.copy_(source)
        self.seconds["host_memcpy"] += time.monotonic() - started

    def _release_backing(self, backing):
        size = backing.size
        try:
            backing.close()
        except BufferError:
            self.pending_backings.append(backing)
        else:
            self.counters["cpu_backing_released_bytes"] += size

    def poll(self):
        old, self.pending_backings = self.pending_backings, []
        for backing in old:
            self._release_backing(backing)

    @torch.no_grad()
    def evict(self, amount):
        moved = 0
        candidates = sorted((t for _, t in self.tiles() if t.residency == "gpu"), key=lambda t: -t.count)
        with self.staging() as stage:
            for tile in candidates:
                if moved >= amount:
                    break
                charge = math.ceil(tile.bytes / PAGE) * PAGE
                if self.cpu_bytes() + charge > int(self.settings.host_state_budget_gib * 2**30):
                    raise RuntimeError("Insufficient rank Adam CPU budget for GPU pressure")
                backing = CPUBacking(2 * tile.count, self.admission)
                try:
                    self._copy(stage[:, :tile.count], tile.pair, tile)
                    self._host_copy(backing.tensor.view(2, tile.count), stage[:, :tile.count])
                except BaseException:
                    backing.close()
                    raise
                tile.pair = backing.tensor.view(2, tile.count)
                tile.backing, tile.residency, tile.since = backing, "cpu", self.window
                moved += tile.bytes
                self.counters["evictions"] += 1
        self.counters["evict_d2h_bytes"] += moved
        self.peak_cpu = max(self.peak_cpu, self.cpu_bytes())
        return moved

    @torch.no_grad()
    def restore_gpu(self, amount, *, minimum_residency=True):
        moved = 0
        with self.staging() as stage:
            for _, tile in self.tiles():
                if moved >= amount:
                    break
                if tile.residency != "cpu" or (minimum_residency and self.window - tile.since < self.settings.min_residency_windows):
                    continue
                if tile.bytes > amount - moved:
                    continue
                target = torch.empty((2, tile.count), device=self.device, dtype=torch.float32)
                self._host_copy(stage[:, :tile.count], tile.pair)
                self._copy(target, stage[:, :tile.count], tile)
                backing = tile.backing
                tile.pair, tile.backing, tile.residency = target, None, "gpu"
                self._release_backing(backing)
                moved += tile.bytes
                self.counters["restores"] += 1
        self.counters["restore_h2d_bytes"] += moved
        self.poll()
        return moved

    def sample(self):
        if self.device.type != "cuda":
            return {"driver_free_bytes": 2**60, "allocated_bytes": 0, "reserved_bytes": 0}
        free, _ = torch.cuda.mem_get_info(self.device)
        return {"driver_free_bytes": free, "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device)}

    def begin_branch(self, length):
        if self.device.type != "cuda":
            return
        self.observed_window_peak = max(self.observed_window_peak, torch.cuda.max_memory_allocated(self.device))
        self.observed_window_reserved = max(self.observed_window_reserved, torch.cuda.max_memory_reserved(self.device))
        missing_grad = sum(local_tensor(p).numel() * 4 for g in self.param_groups
                           for p in g["params"] if p.grad is None)
        self.branch_observation = (length, torch.cuda.memory_allocated(self.device), missing_grad, time.monotonic())
        torch.cuda.reset_peak_memory_stats(self.device)

    def finish_branch(self):
        if self.branch_observation is None:
            return
        length, baseline, missing_grad, started = self.branch_observation
        peak = torch.cuda.max_memory_allocated(self.device)
        self.observed_window_peak = max(self.observed_window_peak, peak)
        self.observed_window_reserved = max(self.observed_window_reserved, torch.cuda.max_memory_reserved(self.device))
        # Tiny sequences are dominated by constant FSDP gathers; extrapolating
        # their per-token ratio to 80K would produce an invalid linear forecast.
        if length >= 4096:
            self.workspace_bytes_per_token = max(self.workspace_bytes_per_token,
                max(0, peak - baseline - missing_grad) / length)
        self.branch_observation = None
        self.seconds["branch_forward_backward_reclaim"] += time.monotonic() - started

    def prepare(self, *, extra_bytes=0, new_window=False, allow_restore=False, sample=None):
        if self.failed or self.closed:
            raise RuntimeError("Unusable streamed Adam optimizer")
        started = time.monotonic()
        self.poll()
        if new_window:
            self.window += 1
            self.observed_window_peak = 0
            self.observed_window_reserved = 0
        before = self.sample() if sample is None else sample
        low = int(self.settings.low_watermark_gib * 2**30)
        high = int(self.settings.high_watermark_gib * 2**30)
        # Trim cached device memory only on actual pressure; then credit driver
        # free, never assume a reserved fragment can satisfy a continuous tensor.
        if sample is None and before["driver_free_bytes"] < extra_bytes + low and self.device.type == "cuda":
            torch.cuda.empty_cache()
            before = self.sample()
        missing = 0 if self.initialized else self.total_bytes()
        scratch = 0 if self.scratch is not None else self.tile_elements * 8
        deficit = max(0, missing + extra_bytes + scratch + low - before["driver_free_bytes"])
        moved = 0
        if not self.initialized:
            if deficit > int(self.settings.host_state_budget_gib * 2**30):
                raise RuntimeError(f"Adam initial capacity deficit {deficit} exceeds host state budget")
            self.initialize(cpu_target=deficit)
        elif deficit:
            moved = self.evict(deficit)
            if moved < deficit:
                raise RuntimeError(f"Insufficient Adam moments to cover predicted GPU deficit: {deficit - moved} bytes")
        self._ensure_scratch()
        after = self.sample() if sample is None else dict(before, driver_free_bytes=before["driver_free_bytes"] + moved - missing - scratch)
        # Recently released tiles are reusable in Torch's allocator. Trim once
        # after a pressure batch so the shared activation policy sees the gain.
        if sample is None and (moved or missing) and self.device.type == "cuda":
            torch.cuda.empty_cache()
            after = self.sample()
        # Independent tensors still share allocator segments. Dropping N tensor
        # bytes does not guarantee that exactly N driver bytes become free.
        # Re-sample and evict another bounded batch before declaring failure.
        if sample is None:
            for _ in range(16):
                shortfall = extra_bytes + low - after["driver_free_bytes"]
                if shortfall <= 0:
                    break
                additional = self.evict(shortfall + min(self.tile_elements * 8, 20 * 2**20))
                moved += additional
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                after = self.sample()
                if not additional:
                    break
        if allow_restore:
            spare = after["driver_free_bytes"] - extra_bytes - high
            self.stable = self.stable + 1 if spare > 0 else 0
            if self.stable >= self.settings.restore_stable_windows:
                self.restore_gpu(min(spare, int(self.settings.max_restore_gib_per_window * 2**30)))
        elif deficit:
            self.stable = 0
        if sample is None and after["driver_free_bytes"] < extra_bytes + low:
            raise RuntimeError("GPU headroom remains below prediction after Adam migration: " + str(after))
        self.last_decision = {"window": self.window, "extra_bytes": extra_bytes, "predicted_deficit_bytes": deficit,
                              "before": before, "after": after, "migrated_bytes": moved}
        self.seconds["control"] += time.monotonic() - started

    @torch.no_grad()
    def step(self, closure=None):
        if self.failed or self.closed:
            raise RuntimeError("Unusable streamed Adam optimizer")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        started = time.monotonic()
        if not self.initialized:
            if self.device.type == "cuda":
                self.prepare(extra_bytes=int(self.settings.gpu_scratch_budget_mib * 2**20))
            else:
                self.initialize()
        self._ensure_scratch()
        # Validate every input before the first parameter mutation.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad = local_tensor(p.grad)
                    if grad.is_sparse or grad.dtype != torch.float32 or grad.shape != local_tensor(p).shape:
                        raise ValueError("Invalid streamed Adam gradient")
        try:
            has_cpu = any(t.residency == "cpu" for _, t in self.tiles())
            with self.staging() if has_cpu else _no_stage() as stage:
                pending_steps = []
                compute_events = []
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        parameter, grad = local_tensor(p).view(-1), local_tensor(p.grad).view(-1)
                        state = self.state[p]
                        for tile in state["tiles"]:
                            n, offset = tile.count, tile.offset
                            if tile.residency == "cpu":
                                pair = self.scratch[:, :n]
                                self._host_copy(stage[:, :n], tile.pair)
                                self._copy(pair, stage[:, :n], tile)
                                self.counters["step_h2d_bytes"] += tile.bytes
                            else:
                                pair = tile.pair
                            old_step = state["step"].clone()
                            compute = time.monotonic()
                            if self.device.type == "cuda":
                                begin = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                begin.record(torch.cuda.current_stream(self.device))
                            adamw([parameter[offset:offset+n]], [grad[offset:offset+n]],
                                  [pair[0]], [pair[1]], [], [old_step], foreach=False,
                                  capturable=False, differentiable=False, fused=False, has_complex=False,
                                  amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1],
                                  lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"], maximize=False)
                            if self.device.type == "cuda":
                                end.record(torch.cuda.current_stream(self.device))
                                compute_events.append((begin, end))
                            self.seconds["adam_enqueue"] += time.monotonic() - compute
                            if tile.residency == "cpu":
                                self._copy(stage[:, :n], pair, tile)
                                self._host_copy(tile.pair, stage[:, :n])
                                self.counters["step_d2h_bytes"] += tile.bytes
                            tile.version += 1
                        pending_steps.append(state["step"])
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                    self.seconds["adam_gpu"] += sum(begin.elapsed_time(end) for begin, end in compute_events) / 1000
                for step in pending_steps:
                    step.add_(1)
            self.counters["step_count"] += 1
        except BaseException:
            self.failed = True  # No rollback mirror, no retry of partial updates.
            raise
        self.seconds["step"] += time.monotonic() - started
        return loss

    def iter_local_state_chunks(self, p, key):
        if key not in {"exp_avg", "exp_avg_sq"}:
            raise ValueError("Unknown Adam moment")
        index = 0 if key == "exp_avg" else 1
        for tile in self.state[p]["tiles"]:
            yield tile.offset, tile.pair[index]

    def validate(self):
        if self.failed:
            raise RuntimeError("Partially updated Adam state")
        count = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if set(state) != {"step", "tiles"} or not torch.isfinite(state["step"]):
                    raise ValueError("Invalid tiled Adam state")
                offset = 0
                for tile in state["tiles"]:
                    if tile.offset != offset or tile.pair.shape != (2, tile.count) or tile.pair.dtype != torch.float32:
                        raise ValueError("Invalid Adam tile coverage")
                    if tile.event is not None or tile.version != int(state["step"]):
                        raise ValueError("Uncommitted Adam tile")
                    for part in tile.pair.reshape(-1).split(1024 * 1024):
                        if not bool(torch.isfinite(part).all()):
                            raise ValueError("Nonfinite Adam moments")
                    offset += tile.count
                if offset != local_tensor(p).numel():
                    raise ValueError("Incomplete Adam shard")
                for part in local_tensor(p).detach().reshape(-1).split(1024 * 1024):
                    if not bool(torch.isfinite(part).all()):
                        raise ValueError("Nonfinite master")
                count += 1
        return {"parameter_tensors": count, "master_moments_dtype": "float32", "all_finite": True,
                "optimizer_backend": BACKEND, "layout_version": LAYOUT}

    def metrics(self):
        cpu = sum(t.bytes for _, t in self.tiles() if t.residency == "cpu")
        gpu = sum(t.bytes for _, t in self.tiles() if t.residency == "gpu")
        return {**self.counters, "seconds_cumulative": dict(self.seconds), "cpu_authoritative_bytes": cpu,
            "gpu_authoritative_bytes": gpu, "uninitialized_bytes": 0 if self.initialized else self.total_bytes(),
            "logical_state_bytes": self.total_bytes(), "cpu_mapped_bytes": self.cpu_bytes(),
            "cpu_peak_mapped_bytes": self.peak_cpu, "pending_backings": len(self.pending_backings),
            "cpu_mapping_count_process": CPUBacking.live_mappings,
            "gpu_scratch_bytes": self.scratch.numel() * 4 if self.scratch is not None else 0,
            "observed_window_peak_allocated_bytes": self.observed_window_peak,
            "observed_window_peak_reserved_bytes": self.observed_window_reserved,
            "observed_workspace_bytes_per_token": self.workspace_bytes_per_token,
            "last_decision": self.last_decision,
            "admission": self.admission.metrics() if self.admission else None}

    def state_dict(self):
        # Standard small-state interoperability; production full checkpoint uses
        # the streaming iterator instead and never materializes whole moments.
        result = super().state_dict()
        packed = {}
        for group, packed_group in zip(self.param_groups, result["param_groups"], strict=True):
            for p, index in zip(group["params"], packed_group["params"], strict=True):
                if p not in self.state or not self.state[p]:
                    continue
                size = local_tensor(p).numel()
                values = {k: torch.empty(size, dtype=torch.float32) for k in ("exp_avg", "exp_avg_sq")}
                for key in values:
                    for offset, tensor in self.iter_local_state_chunks(p, key):
                        values[key][offset:offset+tensor.numel()].copy_(tensor)
                packed[index] = {"step": self.state[p]["step"].clone(), **values}
        result["state"] = packed
        result["streamed_backend"] = {"backend": BACKEND, "layout": LAYOUT}
        return result

    def load_state_dict(self, state_dict):
        if state_dict.get("streamed_backend") != {"backend": BACKEND, "layout": LAYOUT}:
            raise ValueError("Adam backend transition requires explicit migration; no silent conversion")
        if self.initialized:
            raise ValueError("Load requires fresh streamed optimizer")
        self.prepare()
        saved = state_dict["param_groups"]
        if len(saved) != len(self.param_groups):
            raise ValueError("Adam group count mismatch")
        for group, previous in zip(self.param_groups, saved, strict=True):
            for p, index in zip(group["params"], previous["params"], strict=True):
                values = state_dict["state"][index]
                for key in ("exp_avg", "exp_avg_sq"):
                    flat = values[key].reshape(-1)
                    if flat.numel() != local_tensor(p).numel() or flat.dtype != torch.float32:
                        raise ValueError("Adam restore shape/dtype mismatch")
                    for offset, tensor in self.iter_local_state_chunks(p, key):
                        tensor.copy_(flat[offset:offset+tensor.numel()])
                self.state[p]["step"].copy_(values["step"])
                for tile in self.state[p]["tiles"]:
                    tile.version = int(values["step"])
            group.update({k: v for k, v in previous.items() if k != "params"})

    def close(self):
        if self.closed:
            return
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        for _, tile in self.tiles():
            tile.pair = None
            if tile.backing is not None:
                backing, tile.backing = tile.backing, None
                self._release_backing(backing)
        self.poll()
        if self.pending_backings:
            raise RuntimeError("CPU Adam views survived optimizer close; storage remains charged")
        self.scratch = None
        self.closed = True


@contextmanager
def _no_stage():
    yield None


def coordinated(context, operation):
    """Same control sequence on every rank before the next FSDP collective."""
    error = None
    result = None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = context.gather({"rank": context.rank, "error": error})
    if any(row["error"] for row in errors):
        raise RuntimeError("Adam offload phase failed: " + str(errors))
    return result


def workspace_estimate(policy, length):
    """Conservative initial linear estimate, refined by observed branch peaks.

It is an estimate, not an arbitrary-length capacity guarantee. Leave the low
watermark in addition for non-linear/native workspaces and FSDP communication.
"""
    definition = policy.language_model.config
    hidden, intermediate = definition.hidden_size, definition.intermediate_size
    element = 4 if next(policy.parameters()).dtype == torch.float32 and getattr(definition, "dtype", None) == torch.float32 else 2
    # Real 9B backward peaks include recompute, gradient and temporary MLP/GDN
    # tensors beyond one forward's intermediates. Use two forward workspaces
    # initially; subsequent observations can only raise this estimate.
    return int(length * (hidden * 24 + intermediate * 8) * element * 2)


def safety_point(opt, policy, context, *, length=0, new_window=False, allow_restore=False, step=False):
    if not isinstance(opt, StreamedAdamW):
        return
    # Gradients are absent immediately after zero_grad; protect their full next
    # allocation as well as branch workspace before allowing any restoration.
    missing_grad = sum(local_tensor(p).numel() * 4 for p in policy.parameters() if p.grad is None)
    work = (int(opt.settings.gpu_scratch_budget_mib * 2**20) if step else
            max(workspace_estimate(policy, length), int(opt.workspace_bytes_per_token * length)))
    coordinated(context, lambda: opt.prepare(extra_bytes=missing_grad + work,
                new_window=new_window, allow_restore=allow_restore))
