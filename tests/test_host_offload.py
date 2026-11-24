"""Deterministic admission/lifetime tests. CPU simulation is not CUDA evidence."""
import json
from types import SimpleNamespace

import pytest
import torch

from RL.config import Memory, HostOffload, memory_contract
from RL.host_offload import ManagedHostPool, PAGE, pressure_choice
from RL.resources import check_host_capacity


class Allocator:
    drain_bytes = PAGE
    def __init__(self, baseline=0):
        self.allocated = baseline
        self.calls = 0
        self.fail_drain = False
    def stats(self):
        return {"host_allocated_bytes": self.allocated, "host_active_bytes": self.allocated,
                "host_native_cache_bytes": 0, "host_stats_available": True}
    def allocate(self, size):
        self.calls += 1
        self.allocated += size
        return size
    def release(self, block):
        self.allocated -= block["size"]
        block["tensor"] = None
    def drain(self):
        if self.fail_drain: raise RuntimeError("injected cleanup failure")


def pool(pages=8, cache_pages=2, baseline=0):
    settings = HostOffload(enabled=True, allocator_budget_gib=pages * PAGE / 2**30,
                          cache_limit_gib=cache_pages * PAGE / 2**30, reclaim_policy="bounded_reuse")
    allocator = Allocator(baseline)
    return ManagedHostPool(settings, allocator=allocator, capacity=lambda *a, **kw: None), allocator


def test_bounded_short_long_short_multiple_sizes_twenty_rounds():
    current, native = pool(pages=12)
    for _ in range(20):
        for size in [17, 3*PAGE+1, 17, 5*PAGE+7]:
            block = current.acquire(size)
            assert block["size"] % PAGE == 0
            current.release(block)
            assert current.metrics()["host_pool_idle_bytes"] > 0
            row = current.safe_boundary()
            assert current.metrics()["host_pool_idle_bytes"] <= 2*PAGE
            assert native.allocated + PAGE <= current.budget
            assert row["after_bytes"] <= row["before_bytes"]


def test_idle_counts_as_real_pinned_allocation_even_with_no_logical_live():
    current, native = pool()
    block = current.acquire(PAGE)
    current.release(block)
    assert native.stats()["host_allocated_bytes"] == PAGE
    assert current.metrics()["host_pool_idle_bytes"] == PAGE
    current.trim(0)
    assert native.allocated == 0


def test_preallocation_refusal_includes_other_component_baseline_and_page_rounding():
    current, native = pool(pages=4, cache_pages=0, baseline=2*PAGE)
    before = native.calls
    with pytest.raises(RuntimeError, match="before native allocation"):
        current.acquire(PAGE+1)
    assert native.calls == before
    assert current.rejection["charged_bytes"] == 2*PAGE


def test_idle_eviction_allows_new_size_but_never_evicts_active_or_pending():
    current, native = pool(pages=5)
    a = current.acquire(2*PAGE)
    current.release(a)
    b = current.acquire(3*PAGE)
    assert a["tensor"] is None and native.allocated == 3*PAGE
    with pytest.raises(RuntimeError): current.acquire(2*PAGE)
    assert b["tensor"] is not None
    current.release(b)
    event = SimpleNamespace(query=lambda: False)
    b["events"].append(event)
    with pytest.raises(RuntimeError): current.acquire(2*PAGE)
    with pytest.raises(RuntimeError, match="Pending CUDA transfer"): current.safe_boundary()
    event.query = lambda: True
    current.safe_boundary()
    assert native.allocated <= 2*PAGE


def test_reuse_waits_for_all_copy_dependencies_and_cleanup_errors_propagate():
    current, native = pool()
    b = current.acquire(PAGE)
    current.release(b)
    ready = [False]
    b["events"].append(SimpleNamespace(query=lambda: ready[0]))
    other = current.acquire(PAGE)
    assert other is not b
    current.release(other)
    ready[0] = True
    again = current.acquire(PAGE)
    assert again is b
    with pytest.raises(RuntimeError, match="Active managed host lease"): current.safe_boundary()
    current.release(again)
    native.fail_drain = True
    with pytest.raises(RuntimeError, match="injected cleanup"): current.safe_boundary()


def test_process_baseline_growth_fails_even_on_reuse():
    current, native = pool()
    b = current.acquire(PAGE)
    current.release(b)
    native.allocated += 8*PAGE
    with pytest.raises(RuntimeError, match="process-wide"): current.acquire(PAGE)


def test_host_pressure_overrides_retained_idle_cache():
    current, native = pool()
    b = current.acquire(PAGE)
    current.release(b)
    def no_capacity(*a, **kw): raise RuntimeError("system under pressure")
    current.capacity = no_capacity
    row = current.safe_boundary()
    assert row["reason"] == "host_pressure" and native.allocated == 0


def test_unavailable_allocator_stats_fail_closed():
    class Missing(Allocator):
        def stats(self): raise RuntimeError("host stats missing")
    with pytest.raises(RuntimeError, match="host stats missing"):
        ManagedHostPool(HostOffload(enabled=True), allocator=Missing())


def test_pressure_selection_ignores_length_and_does_not_double_count_cache():
    def choose(free, retry=0):
        return pressure_choice(driver_free_bytes=free, reserved_bytes=1000, allocated_bytes=500,
            tensor_bytes=100, headroom_bytes=300, retry_headroom_bytes=retry)
    assert not choose(500)["offload"]
    assert choose(350)["offload"]
    assert choose(500, retry=200)["offload"]
    assert choose(350)["credited_native_cache_bytes"] == 0


@pytest.mark.parametrize("world", [1, 2, 4])
def test_host_budget_checks_all_local_ranks_even_without_optional_preflight(world):
    gib = 2**30
    settings = HostOffload(enabled=True, reserve_gib=8.0, reserve_fraction=0.1)
    host = {"MemTotal": 100*gib, "MemAvailable": 15*gib, "cgroup_remaining_bytes": None}
    if world < 4:
        result = check_host_capacity(settings, requested_bytes=2*gib, world_size=world, host=host)
        assert result["requested_bytes_all_local_ranks"] == world*2*gib
    else:
        with pytest.raises(RuntimeError, match="reserve exhausted"):
            check_host_capacity(settings, requested_bytes=2*gib, world_size=world, host=host)
    host["cgroup_remaining_bytes"] = 10*gib
    with pytest.raises(RuntimeError): check_host_capacity(settings, requested_bytes=gib, host=host)


@pytest.mark.parametrize("raw", [
    {"activation_offload": "memory_pressure"},
    {"host_offload": {"enabled": True, "allocator_budget_gib": 0.2, "cache_limit_gib": 0.3}},
    {"host_offload": {"reserve_fraction": float("inf")}},
    {"memory_pressure": {"max_retries": 4}},
    {"memory_pressure": {"max_retries": True}},
])
def test_strict_managed_config(raw):
    with pytest.raises(ValueError): Memory.model_validate(raw)


def test_default_contract_compatibility_and_explicit_version():
    legacy = memory_contract(Memory())
    assert "host_offload" not in legacy and "memory_pressure" not in legacy
    managed = memory_contract(Memory(activation_offload="memory_pressure", host_offload={"enabled": True}))
    assert managed["offload_policy_version"] == "scoped_host_reclaim_v2" and managed != legacy
    assert managed["host_allocator_backend"] == "torch_size_separated"


def test_pressure_retry_from_already_enabled_policy_preserves_transaction(tmp_path):
    from tests.test_lora_oom_recovery import execute
    from tests.test_grpo_lifecycle import assert_tree_close
    want = execute(tmp_path/"direct", mode="checkpoint_boundary")
    got = execute(tmp_path/"retry", fault="once", mode="memory_pressure")
    for key in ("model", "optimizer", "scheduler", "state", "rng", "grads"):
        assert_tree_close(want[key], got[key])
    assert got["commits"] == got["yields"] == 2
    assert got["updates"][0][1]["step_attempt"] == 1
    rows = [json.loads(x) for x in (tmp_path/"retry/memory-events.jsonl").read_text().splitlines()]
    assert next(x for x in rows if x["event"] == "retry_start")["offload"]["retry_headroom_bytes"] > 0
    assert rows[-1]["offload"]["state_scope"] == "window_local_reset_before_each_update"


def test_pressure_retry_exhaustion_never_commits(tmp_path):
    from tests.test_lora_oom_recovery import execute
    with pytest.raises(RuntimeError, match="GRPO step failed"):
        execute(tmp_path, fault="always", mode="memory_pressure")
    rows = [json.loads(x) for x in (tmp_path/"memory-events.jsonl").read_text().splitlines()]
    assert len([x for x in rows if x["event"] in {"step_start", "retry_start"}]) == 3
    assert not any(x["event"] == "step_commit" for x in rows)


def test_after_use_progress_never_frees_a_lease_or_pending_event():
    current, native = pool()
    current.reclaim_policy = "after_use"
    current.cache_limit = 0
    b = current.acquire(PAGE)
    current.poll()
    assert native.allocated == PAGE
    ready = [False]
    b["events"].append(SimpleNamespace(query=lambda: ready[0]))
    current.release(b)
    current.poll()
    assert current.metrics()["host_pool_pending_bytes"] == PAGE
    ready[0] = True
    row = current.poll()
    assert native.allocated == 0 and row["confirmed_net_reclaimed_bytes"] == PAGE
    current.close()
    assert current.close() is None
    with pytest.raises(RuntimeError, match="closed"): current.acquire(PAGE)


def test_native_small_cache_is_allowed_and_size_separation_is_charged_before_allocation():
    from RL.host_offload import TorchHostAllocator, NATIVE_CACHE_BYTES
    assert TorchHostAllocator.charge(4) == NATIVE_CACHE_BYTES + PAGE
    assert TorchHostAllocator.charge(NATIVE_CACHE_BYTES) > NATIVE_CACHE_BYTES
    assert TorchHostAllocator.charge(NATIVE_CACHE_BYTES + PAGE + 1) == NATIVE_CACHE_BYTES + 2*PAGE
    current, native = pool(pages=8)
    native.charge = TorchHostAllocator.charge
    native.stats = lambda: {"host_allocated_bytes": PAGE, "host_native_cache_bytes": PAGE}
    with pytest.raises(RuntimeError, match="before native allocation"):
        current.acquire(4)
    assert native.calls == 0


def test_scoped_allocator_configuration_preserves_small_native_cache(monkeypatch):
    from RL.allocator import configure_cuda_allocator
    from RL.config import CudaAllocator
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"): monkeypatch.delenv(key, raising=False)
    info = configure_cuda_allocator(CudaAllocator(), host_offload=HostOffload(enabled=True), apply=False)
    assert "pinned_max_cached_size_mb:1" in info["normalized"]
    assert "pinned_max_round_threshold_mb:1" in info["normalized"]
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "pinned_max_cached_size_mb:0")
    with pytest.raises(ValueError, match="conflicts"): configure_cuda_allocator(CudaAllocator(), host_offload=HostOffload(enabled=True), apply=False)


def test_empty_pool_with_native_deferred_release_still_synchronizes_boundary(monkeypatch):
    from RL.memory import BoundaryOffload
    current, native = pool()
    current.cache_limit = 0
    block = current.acquire(PAGE)
    current.release(block)
    current.poll()
    assert not current.blocks and current.native_release_pending
    offload = object.__new__(BoundaryOffload)
    offload.pool, offload.live_bytes, offload._boundary_clean = current, 0, False
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append('sync'))
    offload.safe_boundary()
    assert calls == ['sync'] and not current.native_release_pending


def test_trace_leases_share_capacity_and_reuse_on_release(tmp_path, monkeypatch):
    from RL.host_admission import TraceAdmission
    a = TraceAdmission(tmp_path, budget=100, reserve=1, attempt="baseline", rank=0)
    b = TraceAdmission(tmp_path, budget=100, reserve=1, attempt="baseline", rank=1)
    host = {"MemAvailable": 500, "cgroup_remaining_bytes": 78}
    monkeypatch.setattr("RL.host_admission.host_memory", lambda: host)
    token, _ = a.try_acquire(65)
    assert token
    assert b.try_acquire(20)[0] is None
    host["cgroup_remaining_bytes"] = 200
    assert b.try_acquire(36)[0] is None
    a.account_overrun(token, 75)
    assert b.try_acquire(26)[0] is None
    next_token, status = b.try_acquire(25)
    assert next_token is not None and status["shared_reserved_bytes"] == 100
    b.release(next_token)
    a.account_overrun(token, 105)
    rejected, status = b.try_acquire(1)
    assert rejected is None and status["shared_reserved_bytes"] == 105
    a.release(token)
    token, _ = b.try_acquire(90)
    assert token  # One rank can borrow more than half the shared budget.
    b.release(token)
    assert not a.tokens and not b.tokens


def test_trace_admission_reuses_resident_capacity_without_ignoring_pending_bytes(tmp_path, monkeypatch):
    from RL.host_admission import TraceAdmission
    gib = 2**30
    a = TraceAdmission(tmp_path, budget=460*gib, reserve=gib, attempt="baseline", rank=0)
    b = TraceAdmission(tmp_path, budget=460*gib, reserve=gib, attempt="baseline", rank=1)
    host = {"MemAvailable": 503*gib, "cgroup_path": "/training"}
    monkeypatch.setattr("RL.host_admission.host_memory", lambda: host)
    token, _ = a.try_acquire(304*gib)
    a.account_residency(token, 238*gib)
    host.update(MemAvailable=227*gib, cgroup_remaining_bytes=147*gib)
    next_token, status = b.try_acquire(80*gib)
    assert next_token and status["shared_reserved_bytes"] == 384*gib
    assert status["trace_pending_bytes"] == 146*gib
    b.release(next_token)
    assert b.try_acquire(81*gib)[0] is None
    host["cgroup_path"] = "/another-group"
    assert b.try_acquire(80*gib)[0] is None
    host.pop("cgroup_remaining_bytes")
    a.account_overrun(token, 390*gib)
    assert b.try_acquire(71*gib)[0] is None
    a.account_residency(token, -238*gib)
    assert b.try_acquire(60*gib)[0] is None
    a.release(token)


def test_complete_trace_reclaims_before_releasing_reservation(monkeypatch):
    from RL.memory import BoundaryOffload
    offload = BoundaryOffload(torch.nn.Linear(2, 2), None)
    offload.pool, native = pool(pages=2**18, cache_pages=0)
    offload.trace_bytes_per_token = 100
    stages, overruns = [], []
    def acquire(size, _wait):
        assert size == 1000 + 512 * 2**20
        return "trace", {"shared_reserved_bytes": size}, 0.
    def release(token):
        assert token == "trace"
        assert not offload.pool.blocks and native.allocated == 0
        assert not offload.pool.native_release_pending
    residency = []

    def account_residency(token, delta):
        assert native.allocated == abs(delta)
        residency.append((token, delta))

    offload.trace_admission = SimpleNamespace(acquire=acquire, release=release,
                                             account_overrun=lambda *args: overruns.append(args),
                                             account_residency=account_residency)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    actual = 512 * 2**20 + PAGE
    with offload.trace_scope(10, lambda phase, **_: stages.append(phase)):
        block = offload.pool.acquire(actual)
        offload.pool.release(block)
        assert native.allocated == actual
    assert overruns == [("trace", actual + PAGE)]
    assert residency == [("trace", actual), ("trace", -actual)]
    assert stages == ["trace_host_admitted", "trace_host_released"]
    assert offload.pool.budget == 2**18 * PAGE


def test_failed_trace_releases_only_after_pool_drain(monkeypatch):
    from RL.memory import BoundaryOffload
    offload = BoundaryOffload(torch.nn.Linear(2, 2), None)
    offload.pool, native = pool(pages=2**18, cache_pages=2)
    offload.trace_bytes_per_token = 100
    released = []

    def release(token):
        assert not offload.pool.blocks and native.allocated == 0
        released.append(token)

    resident = []
    offload.trace_admission = SimpleNamespace(
        acquire=lambda *_: ("trace", {}, 0.), release=release, account_overrun=lambda *_: None,
        account_residency=lambda _, delta: resident.append(delta))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with offload.trace_scope(10, lambda *_a, **_kw: None):
            block = offload.pool.acquire(PAGE)
            offload.pool.release(block)
            offload._boundary_clean = False
            raise torch.cuda.OutOfMemoryError("injected")
    assert not released and offload.failed_trace == "trace"
    offload.release_failed_trace()
    assert released == ["trace"] and offload.failed_trace is None
    assert resident == [PAGE, -PAGE]
