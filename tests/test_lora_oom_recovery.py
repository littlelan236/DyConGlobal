"""CPU state-machine/Gloo tests; they do not claim CUDA memory savings."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from RL.allocator import configure_cuda_allocator
from RL.config import Config, CudaAllocator, Memory, load
from RL.memory import BoundaryOffload, MemoryController, MemoryEventLogger
from RL.parallel import GroupParallel
from RL.grpo.objective import ObjectiveResult
from RL.training import capture_rng, seed_everything, train_grpo

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def no_ambient_allocator(monkeypatch):
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)


def test_allocator_merges_aliases_and_bracket_options(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "roundup_power2_divisions:[256:1,512:2]")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    result = configure_cuda_allocator(CudaAllocator())
    assert "max_split_size_mb:128" in result["normalized"]
    assert "roundup_power2_divisions:[256:1,512:2]" in result["normalized"]
    assert "expandable_segments:True" in result["normalized"]
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == result["normalized"]
    assert configure_cuda_allocator(CudaAllocator())["normalized"] == result["normalized"]


@pytest.mark.parametrize("primary,legacy", [
    ("expandable_segments:False", ""), ("max_split_size_mb:64", "max_split_size_mb:128"),
    ("backend:cudaMallocAsync", ""), ("roundup_power2_divisions:[1:2", "")])
def test_allocator_conflicts_and_unsupported_backend_fail(monkeypatch, primary, legacy):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", primary)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", legacy)
    with pytest.raises(ValueError): configure_cuda_allocator(CudaAllocator())


def test_allocator_dry_run_preserves_environment(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    result = configure_cuda_allocator(CudaAllocator(), apply=False)
    assert not result["configured"] and "PYTORCH_ALLOC_CONF" not in os.environ


@pytest.mark.parametrize("field,value", [("oom_retry", {"enabled": True, "max_retries": True}),
    ("oom_retry", {"max_retries": 2}), ("auto_offload", {"prompt_tokens_threshold": -1}),
    ("auto_offload", {"device_headroom_gib": float("inf")}),
    ("memory_telemetry", {"granularity": "layer"})])
def test_new_fields_strict(field, value):
    with pytest.raises(ValueError): Memory.model_validate({field: value})


def settings(mode="none", retry=True):
    cfg, _ = load(ROOT / "RL/experiments/grpo-offline-k3-smoke.yaml", ROOT)
    cfg.training.memory = Memory(activation_offload=mode,
        host_offload={"enabled": mode == "memory_pressure"},
        oom_retry={"enabled": retry, "max_retries": 1},
        auto_offload={"prompt_tokens_threshold": 8, "sequence_tokens_threshold": 12, "device_headroom_gib": 8.0})
    return cfg


class Switch:
    def __init__(self): self.enabled = False; self.live_bytes = 0
    def set_enabled(self, value): self.enabled = value
    def metrics(self): return {"offload_d2h_bytes": 0, "offload_h2d_bytes": 0}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.4, -.2]))
        self.register_buffer("ticks", torch.tensor(0))


def groups(count=5):
    return [SimpleNamespace(group_id=str(i), group_size=2, index=i,
        prompt_token_ids=[[1, 2]] * 2, completion_token_ids=[[3], [4, 5]],
        loss_mask=torch.ones(2, 2, dtype=torch.bool)) for i in range(count)]


class Objective:
    def __init__(self, fault=None, rank=0):
        self.fault, self.rank, self.calls, self.draws = fault, rank, [], []
    def reset(self): pass
    def close_group(self): pass
    def iter_results(self, model, group):
        self.calls.append(group.group_id)
        for branch in range(2):
            model.ticks.add_(1)
            factor = random.random() + float(np.random.rand()) + float(torch.rand(()))
            self.draws.append((group.index, branch, factor))
            loss = (model.weight - group.index * .2).square().sum() * factor / 2
            if self.fault and group.index == 2 and branch == 1 and (self.fault != "once" or not hasattr(self, "failed")):
                self.failed = True
                if self.fault == "fatal": raise ValueError("data invalid")
                if self.fault == "host": raise RuntimeError("Pinned activation host budget exceeded")
                if self.fault == "cpu": raise MemoryError("host memory")
                raise torch.OutOfMemoryError("CPU simulation of CUDA OOM")
            yield ObjectiveResult(loss, loss, loss * 0, {"policy_loss": float(loss.detach())})


class Windows:
    """Single-use window and reference identities must survive retry."""
    def __init__(self, rows):
        self.rows = rows; self.yields = self.commits = self.closed = 0
        self.reference = object()
    def __len__(self): return len(self.rows)
    def validate_state(self, state): pass
    def cursor(self, cursor): return {}
    def windows(self, start):
        try:
            for i in range(start, len(self.rows), 4):
                self.active = tuple(self.rows[i:i+4]); self.yields += 1
                yield self.active
        finally: self.closed += 1
    def commit_window(self, window):
        assert window is self.active
        self.commits += 1


def execute(tmp, *, fault=None, parallel=None, mode="none", count=5):
    # Production loads modeling before training starts. Match that lifecycle;
    # first-time PEFT imports can initialize module RNG state in a CPU fixture.
    from RL.modeling import adapter_fingerprint
    seed_everything(110 + (parallel.rank if parallel else 0))
    model, objective = Model(), Objective(fault)
    switch = Switch()
    controller = MemoryController(settings(mode), tmp, offloader=switch,
        rank=parallel.rank if parallel else 0, world_size=parallel.world_size if parallel else 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    windows = Windows(groups(count)); updates, grads = [], []
    state = train_grpo(model, windows, objective, optimizer, scheduler, group_batch_size=4,
        grad_clip=100., parallel=parallel, memory_controller=controller,
        on_gradients=lambda: grads.append(model.weight.grad.detach().clone()),
        on_update=lambda s, m: updates.append((s, m)))
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "state": state, "rng": capture_rng(),
            "updates": updates, "grads": grads, "calls": objective.calls,
            "draws": objective.draws,
            "yields": windows.yields, "commits": windows.commits, "closed": windows.closed,
            "switch": switch.enabled}


def test_whole_step_retry_restores_rng_gradients_buffers_and_cursor(tmp_path):
    from tests.test_grpo_lifecycle import assert_tree_close
    want = execute(tmp_path / "direct", mode="checkpoint_boundary")
    got = execute(tmp_path / "retry", fault="once")
    for key in ("model", "optimizer", "scheduler", "state", "rng", "grads"):
        assert_tree_close(want[key], got[key])
    assert got["calls"] == ["0", "1", "2", "0", "1", "2", "3", "4"]
    assert got["yields"] == got["commits"] == 2 and got["closed"] == 1
    assert [m["step_attempt"] for _,m in got["updates"]] == [1, 0]
    assert not got["switch"]  # next step returns to its initial non-offloaded policy
    events = [json.loads(line) for line in (tmp_path / "retry/memory-events.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in events if row["event"] == "step_commit"] == [1, 2]
    assert any(row["event"] == "oom" and row["global_step"] == 0 for row in events)


@pytest.mark.parametrize("fault,mode", [("always", "none"), ("once", "checkpoint_boundary"),
    ("fatal", "none"), ("host", "none"), ("cpu", "none")])
def test_unrecoverable_errors_never_commit(tmp_path, fault, mode):
    model, objective = Model(), Objective(fault)
    controller = MemoryController(settings(mode), tmp_path, offloader=Switch())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    before = model.weight.detach().clone()
    with pytest.raises((RuntimeError, ValueError)):
        train_grpo(model, groups(4), objective, optimizer, scheduler, memory_controller=controller)
    torch.testing.assert_close(before, model.weight, atol=0, rtol=0)
    assert not optimizer.state and scheduler.last_epoch == 0 and model.weight.grad is None
    assert not any(json.loads(line)["event"] == "step_commit" for line in
                   (tmp_path / "memory-events.jsonl").read_text().splitlines())


def _worker(rank, world, path, case):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(path) / "store").as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=40))
    try:
        parallel = GroupParallel(rank, world)
        if case == "risk":
            rows = groups(4)
            rows[-1].prompt_token_ids = [list(range(10))] * 2
            controller = MemoryController(settings("auto"), Path(path)/str(rank), rank=rank,
                world_size=world, offloader=Switch())
            decision = controller.decide(rows, global_step=0, parallel=parallel)
            assert decision["mode"] == "checkpoint_boundary"
            assert decision["trigger_ranks"] == [3 % world]
        else:
            # Rank zero can finish all local groups before the failing rank reports.
            fault = "once" if rank == 2 % world and case in {"retry", "pressure_retry"} else None
            want = execute(Path(path)/f"direct-{rank}", parallel=parallel, mode="checkpoint_boundary")
            got = execute(Path(path)/f"retry-{rank}", parallel=parallel, fault=fault,
                          mode="memory_pressure" if case == "pressure_retry" else "none")
            torch.save({"want": want, "got": got}, Path(path)/f"comparison-{rank}.pt")
            from tests.test_grpo_lifecycle import assert_tree_close
            for key in ("model", "optimizer", "scheduler", "state", "rng", "grads"):
                try:
                    assert_tree_close(want[key], got[key])
                except AssertionError as exc:
                    raise AssertionError(f"rank={rank} key={key} draws={want['draws']} vs {got['draws']}") from exc
            assert got["yields"] == got["commits"] == 2
            assert got["updates"][0][1]["step_attempt"] == (1 if case in {"retry", "pressure_retry"} else 0)
            assert len(got["calls"]) > len(want["calls"]) if case == "retry" and rank < 4 else True
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("case", ["risk", "retry", "pressure_retry"])
def test_real_gloo_risk_or_and_whole_window_retry(tmp_path, world, case):
    mp.spawn(_worker, args=(world, str(tmp_path), case), nprocs=world, join=True)


def test_logger_rejects_tensor_payloads(tmp_path):
    logger = MemoryEventLogger(tmp_path)
    with pytest.raises(TypeError): logger.write(event="invalid", accidental=torch.ones(1))

@pytest.mark.parametrize("enabled", [False, True])
def test_shared_prefix_uses_single_switchable_checkpoint_wrapper(enabled):
    import contextlib
    import gc
    import weakref
    from tests.test_grpo_prefix import model_pair, group, branch_loss
    from RL.grpo.prefix import PrefixContext

    oracle, shared = model_pair(True)
    row = group(5, 3)
    copied = []
    class CPUHookOffload(BoundaryOffload):
        # Exercise the real nonreentrant wrapper/hook nesting on CPU. These
        # cloned tensors are instrumentation, not pinned CUDA transfer evidence.
        @contextlib.contextmanager
        def context(self):
            def pack(tensor):
                if tensor.ndim == 3 and tensor.numel():
                    value = tensor.detach().clone()
                    copied.append(weakref.ref(value))
                    return value
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                yield
    offloader = CPUHookOffload(shared, 2**30, enabled=enabled)
    contexts = [PrefixContext(oracle, row.prompt_token_ids[0]),
                PrefixContext(shared, row.prompt_token_ids[0], offloader=offloader)]
    for i, completion in enumerate(row.completion_token_ids):
        for model, prefix in zip((oracle, shared), contexts):
            current = prefix.branch_logprobs(model, completion, 2)
            branch_loss(current, i, 3).backward()
            del current
            prefix.after_branch_backward()
    for prefix in contexts: prefix.finish(); prefix.close()
    for (name, want), (_, got) in zip(oracle.named_parameters(), shared.named_parameters()):
        if want.requires_grad:
            torch.testing.assert_close(got.grad, want.grad, atol=2e-6, rtol=2e-4, msg=name)
    gc.collect()
    assert bool(copied) == enabled
    assert all(ref() is None for ref in copied)


def test_phase_sampler_folds_peaks_before_every_reset(tmp_path, monkeypatch):
    controller = MemoryController(settings(), tmp_path, offloader=Switch())
    controller.device = torch.device("cuda:0")
    observed = {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20,
                "allocated_bytes": 5, "reserved_bytes": 10}
    sampled = []
    def sample(): sampled.append(dict(observed)); return dict(observed)
    def reset(device):
        assert sampled and sampled[-1] == observed
        observed["peak_allocated_bytes"] = observed["allocated_bytes"]
        observed["peak_reserved_bytes"] = observed["reserved_bytes"]
    monkeypatch.setattr(controller.logger, "memory", sample)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    controller.begin_attempt([], global_step=0, attempt_id=0)
    controller.phase("prefix_forward", group_ordinal=0)
    observed["peak_allocated_bytes"] = 120
    controller.phase("suffix_forward")
    observed["peak_allocated_bytes"] = 80
    controller.phase("prefix_backward")
    observed["peak_allocated_bytes"] = 150
    controller.flush_phases()
    assert controller.peaks["peak_allocated_bytes"] == 150
    rows = [json.loads(line) for line in (tmp_path / "memory-events.jsonl").read_text().splitlines()]
    phases = {row["phase"]: row for row in rows if row["event"] == "phase_end"}
    assert phases["prefix_forward"]["peak_allocated_bytes"] == 120
    assert phases["suffix_forward"]["peak_allocated_bytes"] == 80
    assert phases["prefix_backward"]["peak_allocated_bytes"] == 150


def _failure_worker(rank, path, case):
    torch.set_num_threads(1)
    from RL.modeling import adapter_fingerprint
    dist.init_process_group('gloo', init_method=(Path(path)/'store').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        if case == 'exit' and rank == 1:
            os._exit(7)
        model = Model()
        class FailureObjective(Objective):
            def iter_results(self, model, group):
                if case == 'fatal_precedence' or group.index == 2:
                    if rank == 0: raise torch.OutOfMemoryError('injected CUDA allocation failure')
                    if case == 'fatal_precedence': raise ValueError('data invalid')
                yield from super().iter_results(model, group)
        parallel = GroupParallel(rank, 2)
        def forbidden(*args, **kwargs): raise AssertionError('entered gradient collective after failed compute')
        parallel.finish_window = forbidden
        switch = Switch()
        if case == 'cleanup_failure' and rank == 1: switch.live_bytes = 1
        controller = MemoryController(settings(), Path(path)/str(rank), rank=rank, world_size=2, offloader=switch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        before = model.weight.detach().clone()
        with pytest.raises((ValueError, RuntimeError)):
            train_grpo(model, groups(4), FailureObjective(), optimizer, scheduler,
                       parallel=parallel, memory_controller=controller)
        torch.testing.assert_close(before, model.weight, atol=0, rtol=0)
        assert model.weight.grad is None and not optimizer.state and scheduler.last_epoch == 0
        events = [json.loads(line) for line in controller.logger.path.read_text().splitlines()]
        if case != 'exit': assert not any(row['event'] == 'retry_start' for row in events)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('case', ['fatal_precedence', 'cleanup_failure', 'exit'])
def test_gloo_fatal_cleanup_failure_and_process_exit_are_bounded(tmp_path, case):
    import time
    started = time.monotonic()
    if case == 'exit':
        with pytest.raises(mp.ProcessExitedException):
            mp.spawn(_failure_worker, args=(str(tmp_path), case), nprocs=2, join=True)
    else:
        mp.spawn(_failure_worker, args=(str(tmp_path), case), nprocs=2, join=True)
    assert time.monotonic() - started < 40


def test_memory_summary_separates_committed_and_failed_samples(tmp_path):
    from RL.memory_summary import summarize_files
    execute(tmp_path, fault='once')
    result = summarize_files([tmp_path/'memory-events.jsonl'])
    assert {'committed', 'CUDA_OOM'} <= {row['outcome'] for row in result['buckets']}
    assert {row['offload_mode'] for row in result['buckets']} == {'none', 'checkpoint_boundary'}
    filtered = summarize_files([tmp_path/'memory-events.jsonl'], rank=0, phase='suffix_backward', limit=1)
    assert len(filtered['buckets']) == 1


def test_real_streaming_producer_and_shared_prefix_retry_cpu(tmp_path, monkeypatch):
    from core.io import read_json
    from RL.grpo.backend import GroupObjective
    from RL.grpo.streaming import create_plan
    from RL.memory import configure_checkpointing, install_mlp_backend
    from RL.service import _train_grpo
    from tests.lora_helpers import model_and_targets
    from tests.test_grpo_prepared import ROOT, BASE
    from tests.test_grpo_streaming_resume import fixture
    from tests.test_grpo_lifecycle import assert_tree_close

    cfg = fixture(tmp_path, n=5)
    cfg.grpo.prefix_reuse.mode = 'boundary_grad'
    original = GroupObjective.iter_results
    snapshots = []
    for label, inject in [('direct', False), ('retry', True)]:
        current = cfg.model_copy(deep=True)
        current.run.attempt_id = label
        current.training.memory = Memory(activation_offload='none' if inject else 'checkpoint_boundary',
            loss_chunk_tokens=2, mlp_chunk_tokens=3, oom_retry={'enabled': True, 'max_retries': 1})
        selected = create_plan(current)['proof']['eligible_group_ids'][1]
        path = tmp_path / label
        path.mkdir()
        model, targets = model_and_targets(settings=current.lora)
        mlps = install_mlp_backend(model, 3)
        offloader = configure_checkpointing(model, current, 2**30)
        controller = MemoryController(current, path, offloader=offloader)
        failed, refs = [], []
        def stochastic(self, model, group):
            refs.append((group.group_id, id(self.reference_values.values)))
            for index, result in enumerate(original(self, model, group)):
                factor = .8 + .1*(random.random()+float(np.random.rand())+float(torch.rand(())))
                if inject and not failed and group.group_id == selected and index == 1:
                    failed.append(True)
                    raise torch.OutOfMemoryError('CPU fixture local shared-prefix OOM')
                result.loss = result.loss * factor
                yield result
        monkeypatch.setattr(GroupObjective, 'iter_results', stochastic)
        result = _train_grpo(current, ROOT, path, BASE, None,
            {'attempt_id': label, 'purpose': 'engineering_smoke', 'start_mode': 'fresh'},
            model, targets, optimizer_settings=current.training, offloader=offloader,
            mlps=mlps, memory_controller=controller)
        progress = read_json(path/'grpo/stream-progress.json')
        assert progress['consumed_windows'] == 3 and progress['queue_depth'] <= 2
        if inject:
            assert failed and refs[0][1] == refs[2][1]  # exact reference mapping during retry
        state = torch.load(path/'checkpoints/step-000003/training-state.pt', weights_only=True)
        snapshots.append((result['state'], copy.deepcopy(model.state_dict()),
                          {key: state[key] for key in ('optimizer', 'scheduler', 'rng')}))
    for left, right in zip(snapshots[0], snapshots[1]):
        assert_tree_close(left, right)


def test_allocator_support_warnings_are_captured_and_preserved(monkeypatch):
    import warnings
    from RL.allocator import capture_allocator_warnings
    forwarded = []
    monkeypatch.setattr(warnings, 'showwarning', lambda message, *a, **kw: forwarded.append(str(message)))
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        with capture_allocator_warnings() as observed:
            warnings.warn('expandable_segments is not supported on this platform')
            warnings.warn('ordinary unrelated warning')
    assert observed == [{'message': 'expandable_segments is not supported on this platform', 'category': 'UserWarning'}]
    assert len(forwarded) == 2
