"""Event-safe activation reclamation while retaining small native pinned caches.

Torch 2.13 caches blocks up to 1 MiB. Every project allocation is strictly larger
and page aligned, so dropping a completed block returns its native allocation.
The per-block native cutoff is NOT a total native-cache budget.
"""
import time
import weakref

from RL.resources import check_host_capacity
from RL.allocator import HOST_POLICY_VERSION, HOST_NATIVE_CACHE_MB

PAGE = 4096
POLICY_VERSION = HOST_POLICY_VERSION
NATIVE_CACHE_BYTES = HOST_NATIVE_CACHE_MB * 2**20


def host_allocator_stats():
    import torch
    try:
        fn = getattr(torch.cuda.memory, "host_memory_stats", None)
        if fn is None:
            raise RuntimeError("host_memory_stats API unavailable")
        raw = fn()
        required = ("active_bytes.current", "allocated_bytes.current")
        if any(k not in raw for k in required):
            raise RuntimeError("host_memory_stats unavailable before CUDA initialization")
        active, allocated = (int(raw[k]) for k in required)
        return {"host_stats_available": True, "host_stats_reason": None,
                "host_active_bytes": active, "host_allocated_bytes": allocated,
                "host_active_peak_bytes": raw.get("active_bytes.peak"),
                "host_allocated_peak_bytes": raw.get("allocated_bytes.peak"),
                "host_native_cache_bytes": max(0, allocated - active)}
    except Exception as exc:
        return {"host_stats_available": False, "host_stats_reason": f"{type(exc).__name__}: {exc}",
                **{k: None for k in ("host_active_bytes", "host_allocated_bytes", "host_active_peak_bytes",
                                     "host_allocated_peak_bytes", "host_native_cache_bytes")}}


class TorchHostAllocator:
    """Version-scoped adaptation; no private cache-release symbol is called."""
    drain_bytes = PAGE

    def __init__(self):
        import os
        import torch
        from RL.allocator import _parse
        expected = {"pinned_max_cached_size_mb": str(HOST_NATIVE_CACHE_MB),
                    "pinned_max_round_threshold_mb": str(HOST_NATIVE_CACHE_MB),
                    "pinned_use_background_threads": "false", "pinned_use_cuda_host_register": "false"}
        options = _parse(os.environ.get("PYTORCH_ALLOC_CONF", ""))
        if torch.__version__ != "2.13.0+cu130" or any(options.get(k, "").lower() != v for k,v in expected.items()):
            raise RuntimeError("Managed host allocator requires validated Torch 2.13 size-separated pinned caching")

    @staticmethod
    def charge(size):
        return max(NATIVE_CACHE_BYTES + PAGE, (size + PAGE - 1) // PAGE * PAGE)

    def stats(self):
        sample = host_allocator_stats()
        if not sample["host_stats_available"]:
            raise RuntimeError(sample["host_stats_reason"])
        return sample

    def allocate(self, size):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Managed host reclamation does not support CUDA graph capture")
        return torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)

    def release(self, block):
        block["tensor"] = None

    def drain(self):
        # Torch 2.13 allocate() first processes completed host stream events.
        # Completed large blocks bypass its cache; the small tick is cacheable.
        # This preserves other small caches and is not a global empty_cache().
        tick = self.allocate(self.drain_bytes)
        del tick

    def synchronize(self):
        import torch
        torch.cuda.synchronize()


class ManagedHostPool:
    def __init__(self, settings, *, allocator=None, capacity=None):
        self.settings = settings
        self.allocator = allocator if allocator is not None else TorchHostAllocator()
        self.capacity = capacity if capacity is not None else check_host_capacity
        self.budget = int(settings.allocator_budget_gib * 2**30)
        self.reclaim_policy = settings.reclaim_policy
        self.cache_limit = (0 if self.reclaim_policy == "after_use" else int(settings.cache_limit_gib * 2**30))
        self.blocks = []
        self.baseline = self.allocator.stats()["host_allocated_bytes"]
        self.peak = self.baseline
        self.reclaims = []
        self.rejection = None
        self.last_allocation = None
        self._needs_reclaim = False
        self.closed = False
        self.native_release_pending = False
        self.pending_admission_releases = []

    @staticmethod
    def ready(block):
        if block.get("poisoned") or block["leased"] or any(not event.query() for event in block["events"]):
            return False
        block["events"].clear()
        return True

    def _reject(self, message, **fields):
        self.rejection = {"reason": message, "budget_bytes": self.budget, **fields}
        raise RuntimeError("Managed pinned host budget: " + message)

    def acquire(self, size):
        if self.closed: raise RuntimeError("Managed host pool is closed")
        self.poll()
        charge = (self.allocator.charge(size) if hasattr(self.allocator, "charge")
                  else (size + PAGE - 1) // PAGE * PAGE)
        stats = self.allocator.stats()
        if stats["host_allocated_bytes"] + self.allocator.drain_bytes > self.budget:
            self._reject("process-wide pinned allocation exceeds budget", **stats)
        candidates = [b for b in self.blocks if b["size"] == charge and self.ready(b)]
        if candidates:
            self.capacity(self.settings, requested_bytes=0)
            block = candidates[0]
            block["leased"] = True
            return block
        # Evict only unused blocks with completed events. Never synchronize a
        # tensor here; unsafe pending blocks count against the actual budget.
        if stats["host_allocated_bytes"] + charge + self.allocator.drain_bytes > self.budget:
            self.trim(0, reason="allocation_admission", drain=True)
            stats = self.allocator.stats()
        estimate = stats["host_allocated_bytes"] + charge + self.allocator.drain_bytes
        if estimate > self.budget:
            self._reject("allocation rejected before native allocation", requested_bytes=size,
                         charged_bytes=charge, estimated_total_bytes=estimate, **stats)
        self.capacity(self.settings, requested_bytes=charge + self.allocator.drain_bytes)
        from RL.host_admission import current
        admission = current()
        token = admission.reserve(charge + self.allocator.drain_bytes, "pinned") if admission else None
        try:
            tensor = self.allocator.allocate(charge)
            if admission:
                admission.commit(token)
        except BaseException:
            if admission:
                admission.release(token)
            raise
        block = {"tensor": tensor, "size": charge, "leased": True, "events": [],
                 "admission": admission, "reservation": token}
        self.blocks.append(block)
        after = self.allocator.stats()["host_allocated_bytes"]
        self.peak = max(self.peak, after)
        self.last_allocation = {"logical_bytes": size, "charged_bytes": charge,
            "before_bytes": stats["host_allocated_bytes"], "after_bytes": after,
            "estimate_error_bytes": after - (estimate - self.allocator.drain_bytes)}
        if after + self.allocator.drain_bytes > self.budget:
            # Capability violation is terminal; this check cannot substitute for
            # pre-allocation admission. GPU tests verify the exact-page strategy.
            self._reject("native allocator violated validated allocation charge", **self.last_allocation)
        return block

    def release(self, block):
        # Called by finalizers: no CUDA calls, synchronization, or allocation.
        block["leased"] = False
        self._needs_reclaim = True

    def poll(self):
        """Nonblocking progress at allocation/branch/group control points."""
        if not self._needs_reclaim: return None
        idle = sum(b["size"] for b in self.blocks if self.ready(b))
        if idle > self.cache_limit:
            return self.trim(reason="after_use" if not self.cache_limit else "cache_limit")
        self._needs_reclaim = any(not b["leased"] and b["events"] for b in self.blocks)
        return None

    def trim(self, limit=None, *, reason="safe_boundary", drain=True, finalize=False):
        limit = self.cache_limit if limit is None else limit
        started = time.monotonic()
        before = self.allocator.stats()["host_allocated_bytes"]
        idle = [b for b in self.blocks if self.ready(b)]
        cached = sum(b["size"] for b in idle)
        released = 0
        for block in sorted(idle, key=lambda b: b["size"], reverse=True):
            if cached <= limit: break
            cached -= block["size"]
            released += block["size"]
            self.allocator.release(block)
            if block.get("reservation"):
                self.pending_admission_releases.append((block["admission"], block["reservation"]))
                block["reservation"] = None
            # Native free records its own stream event, which can be later than
            # our explicit copy event. An empty pool alone is not a clean boundary.
            self.native_release_pending = True
            self.blocks = [b for b in self.blocks if b is not block]
        if drain:
            if finalize and self.native_release_pending:
                # free() itself may have recorded fresh native events after
                # the caller's earlier synchronization. Complete these too.
                sync = getattr(self.allocator, "synchronize", None)
                if sync is not None: sync()
            # Reserved scratch space is included in every admission calculation.
            if self.allocator.stats()["host_allocated_bytes"] + self.allocator.drain_bytes > self.budget:
                self._reject("process pinned baseline leaves no event-drain scratch space")
            self.allocator.drain()
            if finalize:
                self.native_release_pending = False
                for admission, token in self.pending_admission_releases:
                    admission.release(token)
                self.pending_admission_releases.clear()
        after = self.allocator.stats()["host_allocated_bytes"]
        row = {"reason": reason, "before_bytes": before, "after_bytes": after,
               "released_pool_bytes": released, "seconds": time.monotonic() - started,
               "confirmed_net_reclaimed_bytes": min(released, max(0, before - after)),
               "api": "size_separated_allocation_with_event_drain", "cache_limit_bytes": limit}
        self._needs_reclaim = any(not b["leased"] and b["events"] for b in self.blocks)
        self.reclaims.append(row)
        self.reclaims = self.reclaims[-8:]
        if after + self.allocator.drain_bytes > self.budget:
            self._reject("process pinned baseline exceeds actual budget after trim", **row)
        return row

    def safe_boundary(self):
        if any(b["leased"] for b in self.blocks):
            raise RuntimeError("Active managed host lease survived safe boundary")
        if any(not self.ready(b) for b in self.blocks):
            raise RuntimeError("Pending CUDA transfer survived synchronized safe boundary")
        # Host pressure overrides idle retention; otherwise bounded cache is kept
        # across short windows without allocation/free churn (a fixed hysteresis).
        try:
            self.capacity(self.settings, requested_bytes=0)
            limit, reason = self.cache_limit, "cache_limit"
        except RuntimeError:
            limit, reason = 0, "host_pressure"
        return self.trim(limit, reason=reason, finalize=True)

    def close(self):
        """Caller must reach the same synchronized boundary as safe_boundary."""
        if self.closed: return None
        self.safe_boundary()
        row = self.trim(0, reason="close", finalize=True)
        self.closed = True
        return row

    def metrics(self):
        stats = self.allocator.stats()
        return {**stats, "host_allocator_baseline_bytes": self.baseline,
            "host_allocator_delta_bytes": stats["host_allocated_bytes"] - self.baseline,
            "host_actual_budget_bytes": self.budget, "host_pool_reserved_bytes": sum(b["size"] for b in self.blocks),
            "host_pool_idle_bytes": sum(b["size"] for b in self.blocks if self.ready(b)),
            "host_pool_leased_bytes": sum(b["size"] for b in self.blocks if b["leased"]),
            "host_pool_pending_bytes": sum(b["size"] for b in self.blocks if not b["leased"] and not self.ready(b)),
            "host_pool_reclaim_policy": self.reclaim_policy,
            "host_pool_native_release_pending": self.native_release_pending,
            "host_allocator_backend": "torch_size_separated",
            "host_native_cache_max_block_bytes": NATIVE_CACHE_BYTES,
            "host_pool_peak_bytes": self.peak, "host_pool_reclaims": list(self.reclaims),
            "host_pool_rejection": self.rejection, "host_last_allocation": self.last_allocation,
            "host_offload_policy_version": POLICY_VERSION}


class HostLease:
    def __init__(self, pool, block, tensor):
        self.block = block
        self.device = tensor.device
        size = tensor.numel() * tensor.element_size()
        self.tensor = block["tensor"][:size].view(tensor.dtype).view(tensor.shape)
        weakref.finalize(self, pool.release, block)


def pressure_choice(*, driver_free_bytes, reserved_bytes, allocated_bytes, tensor_bytes,
                    headroom_bytes, retry_headroom_bytes=0):
    # Use driver free only. Native cached capacity is recorded but not credited;
    # no predicted release of the candidate or other aliases is booked in advance.
    required = headroom_bytes + retry_headroom_bytes + tensor_bytes
    return {"offload": driver_free_bytes < required, "driver_free_bytes": driver_free_bytes,
            "reserved_bytes": reserved_bytes, "allocated_bytes": allocated_bytes,
            "required_headroom_bytes": required, "candidate_bytes": tensor_bytes,
            "reason": "compute_and_h2d_headroom" if driver_free_bytes < required else "gpu_headroom_available",
            "capacity_prediction": False, "credited_native_cache_bytes": 0}
