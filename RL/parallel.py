"""Synchronous, whole-group LoRA data parallelism behind the training service.

Forward/backward remains entirely local, including the shared-prefix VJP.
Only complete-window adapter gradients are SUM-reduced, before clipping.
"""
from datetime import timedelta
from pathlib import Path
import os
import time

_context = None
_device_info = None


def devices(config):
    return config.runtime.gpus or [config.runtime.gpu]


def execution_contract(config):
    count = len(devices(config))
    from RL.config import is_full
    if is_full(config):
        return {"version": "fsdp2_branch_rounds_v1", "world_size": count,
                "gradient_reduction": "fsdp2_mean", "loss_scaling": "world_over_parents_over_branches"}
    if (config.algorithm is not None and config.algorithm.name in {"online_bootstrap_grpo_v1", "offline_baseline_grpo_v1", "key_step_grpo_v1"}
            and count > 1):
        return {"version": "online_training_unit_sum_v1", "world_size": count,
                "assignment": "training_unit_ordinal_mod_world_size", "gradient_reduction": "sum_fp32"}
    from RL.grpo.execution import enabled, SCHEDULE, PARTITION, WEIGHT
    if enabled(config):
        return {"version": SCHEDULE, "world_size": count,
                "partition_algorithm": PARTITION, "loss_weight_protocol": WEIGHT,
                "assignment": "window_ordinal_mod_world_size", "gradient_reduction": "sum_fp32"}
    return ({"version": "whole_group_sum_v1", "world_size": count,
             "assignment": "window_ordinal_mod_world_size", "gradient_reduction": "sum_fp32"}
            if count > 1 else None)


def current():
    return _context


class GroupParallel:
    def __init__(self, rank, world_size, *, data_group=None):
        self.rank, self.world_size, self.data_group = rank, world_size, data_group
        self.rank_rng = None

    @property
    def primary(self):
        return self.rank == 0

    def gather(self, value):
        import torch.distributed as dist
        output = [None] * self.world_size
        dist.all_gather_object(output, value)
        return output

    def broadcast(self, value=None):
        import torch.distributed as dist
        items = [value]
        dist.broadcast_object_list(items, src=0)
        return items[0]

    def synchronize_model(self, model):
        import torch
        import torch.distributed as dist
        parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        schema = [(name, list(p.shape), str(p.dtype)) for name, p in parameters]
        if any(other != schema for other in self.gather(schema)):
            raise ValueError("Distributed adapter parameter manifests differ")
        with torch.no_grad():
            for _, parameter in parameters:
                dist.broadcast(parameter, src=0, group=self.data_group)

    def reduce_gradients(self, parameters):
        import torch
        import torch.distributed as dist
        # Explicit presence avoids changing AdamW behavior for globally unused parameters.
        device = parameters[0].device
        present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=device)
        dist.all_reduce(present, group=self.data_group)
        flags = present.tolist()
        # Bounded FP32 communication buckets; never communicate frozen base weights.
        bucket, count = [], 0
        def flush():
            if not bucket:
                return
            flat = torch.cat([(p.grad.detach().float().reshape(-1) if p.grad is not None else
                               torch.zeros(p.numel(), device=device)) for p in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.data_group)
            offset = 0
            for parameter in bucket:
                value = flat[offset:offset + parameter.numel()].view_as(parameter).to(parameter.dtype)
                if parameter.grad is None:
                    parameter.grad = value.clone()
                else:
                    parameter.grad.copy_(value)
                offset += parameter.numel()
            bucket.clear()
        for parameter, used in zip(parameters, flags):
            if not used:
                parameter.grad = None
                continue
            if bucket and count + parameter.numel() > 4 * 1024 * 1024:
                flush()
                count = 0
            bucket.append(parameter)
            count += parameter.numel()
        flush()

    def control_step(self, step, attempt, phase, payload):
        """Gloo control messages have a checked collective sequence and identity."""
        envelope = {"step": step, "attempt": attempt, "phase": phase,
                    "rank": self.rank, "payload": payload}
        peers = self.gather(envelope)
        if (sorted(p["rank"] for p in peers) != list(range(self.world_size)) or
                any((p["step"], p["attempt"], p["phase"]) != (step, attempt, phase) for p in peers)):
            raise ValueError("Distributed step/attempt/control-phase mismatch")
        return [peer["payload"] for peer in peers]

    def finish_window(self, model, parameters, window, rows, loss, prefix, compute_seconds):
        import torch
        from RL.topology import portable, merge_rows
        chunk_mode = bool(window and window[0].metadata.get("execution_chunk")) if hasattr(window[0], "metadata") else False
        remapped = portable(self) and not chunk_mode
        identities = [dict(g.metadata["execution_chunk"]) for g in window] if chunk_mode else None
        if remapped:
            branch_count = sum(len(values) for _, values in rows)
            token_count = sum(len(window[i].completion_token_ids[row["native_branch_ordinal"]])
                              for i, values in rows for row in values)
        else:
            branch_count = sum(window[index].group_size for index, _ in rows)
            token_count = sum(int(window[index].loss_mask.sum()) for index, _ in rows)
        payload = {"identities": identities, "groups": [index for index, _ in rows], "rows": rows, "loss": loss,
                   "prefix": prefix, "rank": self.rank, "compute_seconds": compute_seconds,
                   "branch_count": branch_count,
                   "token_count": token_count,
                   "peak_allocated_bytes": getattr(self, "attempt_peak_allocated_bytes", 0) or
                        (torch.cuda.max_memory_allocated() if parameters[0].is_cuda else 0),
                   "peak_reserved_bytes": getattr(self, "attempt_peak_reserved_bytes", 0) or
                        (torch.cuda.max_memory_reserved() if parameters[0].is_cuda else 0),
                   "phases": getattr(self, "local_phase_seconds", {}),
                   "memory": getattr(self, "local_memory", None),
                   "offload_metrics": getattr(self, "local_offload_metrics", None),
                   "finite": all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)}
        peers = self.gather(payload)
        indexes = [index for peer in peers for index in peer["groups"]]
        parent_rows = merge_rows(window, peers) if remapped else None
        if not remapped and sorted(indexes) != list(range(len(window))):
            raise ValueError("Distributed window has missing or duplicate group contributions")
        if chunk_mode:
            if (any(peer["identities"] != identities for peer in peers)
                    or len({c["parent_group_id"] for c in identities}) != len(window)
                    or any(peer["groups"] != [i for i in range(len(window)) if i % self.world_size == peer["rank"]]
                           for peer in peers)):
                raise ValueError("8x16 distributed parent/half/rank identity mismatch")
            for peer in peers:
                for index, contributions in peer["rows"]:
                    c = identities[index]
                    if (len(contributions) != 16
                            or [r.get("branch_id") for r in contributions] != c["branch_ids"]
                            or [r.get("native_branch_ordinal") for r in contributions] != c["native_branch_ordinals"]):
                        raise ValueError("8x16 distributed branch coverage/order mismatch")
        if not all(peer["finite"] for peer in peers):
            raise ValueError("Nonfinite distributed gradient; optimizer update rejected")
        started = time.monotonic()
        self.reduce_gradients(parameters)
        if parameters[0].is_cuda:
            torch.cuda.synchronize()
        ordered = sorted((row for peer in peers for row in peer["rows"]), key=lambda row: row[0])
        stats = {"world_size": self.world_size, "gradient_sync_seconds": time.monotonic() - started,
                 "local_telemetry_rank": 0,
                 "rank_critical_path_seconds": max(p["compute_seconds"] for p in peers),
                 "ranks": [{"rank": peer["rank"], "group_count": len(set(peer["groups"])) if remapped else len(peer["groups"]),
                            **({"physical_task_count": len(peer["groups"])} if remapped else {}),
                            "group_ordinals": peer["groups"], "compute_seconds": peer["compute_seconds"],
                            "branch_count": peer["branch_count"], "token_count": peer["token_count"],
                            "peak_allocated_bytes": peer["peak_allocated_bytes"],
                            "peak_reserved_bytes": peer["peak_reserved_bytes"], **peer["phases"],
                            "memory": peer.get("memory"), "offload_metrics": peer.get("offload_metrics"),
                            **({"parent_group_count": len(peer["groups"]),
                                "execution_chunk_count": len(peer["groups"])} if chunk_mode else {})}
                           for peer in peers]}
        return (parent_rows if remapped else [row for _, row in ordered], sum(peer["loss"] for peer in peers),
                [item for peer in peers for item in peer["prefix"]], stats)

    def after_step(self, model):
        from RL.modeling import adapter_fingerprint
        from RL.training import capture_rng
        # Compact hashes detect divergence before publishing a new committed update.
        fingerprints = self.gather(adapter_fingerprint(model))
        if any(item != fingerprints[0] for item in fingerprints):
            raise ValueError("Distributed adapter replicas diverged after optimizer update")
        self.rank_rng = self.gather(capture_rng())


def launch(config, root, directory, base, parent, attempt):
    """The public service owns all locks and all spawned worker lifetimes."""
    import torch.multiprocessing as mp
    import tempfile
    temporary_root = Path(root) / "temp"
    temporary_root.mkdir(exist_ok=True)
    rendezvous = Path(tempfile.mkdtemp(prefix="rl-data-parallel-", dir=temporary_root))
    result_file = rendezvous / "result.json"
    import shutil
    import signal
    context = None
    previous = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    def terminated(signum, frame):
        raise RuntimeError("Distributed training supervisor received SIGTERM")
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        context = mp.start_processes(_worker, args=(config, str(root), str(directory), base, parent, attempt,
                                                   str(rendezvous / "store"), str(result_file)),
                                     nprocs=len(devices(config)), join=False, start_method="spawn")
        while not context.join(timeout=1):
            failures = [Path(directory) / "worker-error.json",
                        *[Path(directory) / "distributed" / f"rank-{rank}" / "worker-error.json"
                          for rank in range(1, len(devices(config)))]]
            for failure in failures:
                if failure.exists():
                    detail = (failure.parent / "worker-traceback.txt").read_text()
                    raise RuntimeError("Distributed worker failed:\n" + detail)
        from core.io import read_json
        return read_json(result_file)
    finally:
        if context:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
        signal.signal(signal.SIGINT, previous[signal.SIGINT])
        shutil.rmtree(rendezvous)


def _worker(rank, config, root, directory, base, parent, attempt, store, result_file):
    global _context, _device_info
    selected = devices(config)
    os.environ["LOCAL_WORLD_SIZE"] = str(len(selected))
    config = config.model_copy(deep=True)
    config.runtime.gpu = selected[rank]
    from RL.service import _select_gpu, _train
    _device_info = _select_gpu(config)
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(0)
    duration = timedelta(seconds=config.runtime.distributed_timeout_seconds)
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank,
                            world_size=len(selected), timeout=duration)
    try:
        data = dist.new_group(backend="nccl", timeout=duration)
        _context = GroupParallel(rank, len(selected), data_group=data)
        location = Path(directory) if rank == 0 else Path(directory) / "distributed" / f"rank-{rank}"
        location.mkdir(parents=True, exist_ok=True)
        from RL.config import is_full
        result = _train(config, Path(root), Path(directory) if is_full(config) else location,
                        base, parent, dict(attempt))
        # Keep all workers alive until rank zero has published the attempt receipt.
        dist.barrier()
        if rank == 0:
            from core.io import write_json
            write_json(result_file, result)
    except BaseException:
        # destroy_process_group can wait for a peer still inside a collective.
        # Exit through the spawn error path so the supervisor terminates peers.
        import traceback
        from core.io import write_json
        location = Path(directory) if rank == 0 else Path(directory) / "distributed" / f"rank-{rank}"
        location.mkdir(parents=True, exist_ok=True)
        (location / "worker-traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(location / "worker-error.json", {"rank": rank, "status": "failed"})
        traceback.print_exc()
        raise
    else:
        dist.destroy_process_group()
    finally:
        _context = None


def follower(config, root, directory, parent, model, targets, *, optimizer_settings, mlps=(), offloader=None,
             memory_controller=None):
    """Attach read-only prepared/reference artifacts published once by rank zero."""
    from RL.offline_baseline import build_updater as baseline_updater
    from RL.policy_reference import dycon_scorer
    context = current()
    shared = context.broadcast()
    contract, state, stop = shared["contract"], shared["state"], shared["stop"]
    if shared.get("streaming"):
        from RL.grpo.streaming import StreamingGroups, attach_source
        from core.io import read_json
        plan = read_json(Path(shared["directory"]) / "grpo/stream-plan.json")
        groups = StreamingGroups(config, plan, rank=context.rank, world_size=context.world_size,
                                 progress_path=Path(directory) / "stream-progress.json")
        attach_source(config, groups, contract["grpo"]["probability_binding"], directory=shared["directory"])
        from RL.grpo.backend import GroupObjective
        from RL.training import optimizer_and_scheduler, train_grpo
        from RL.memory import attention_context
        from RL.profiling import UpdateTelemetry
        objective = GroupObjective(groups.reference, config.grpo, config.training.memory.loss_chunk_tokens,
                                   offloader=offloader,
                                   reference_scorer=dycon_scorer(config, model, shared["directory"]))
        optimizer, scheduler = optimizer_and_scheduler(model, optimizer_settings)
        telemetry = UpdateTelemetry(model, objective, mlps, offloader, directory,
                                    config.training.memory.profile_first_update,
                                    config.training.memory.attention_policy == "flash_only")
        from RL.grpo.stream_resume import validate_anchor, restore_training
        validate_anchor(model, shared["directory"], contract)
        if parent and config.run.start.mode == "resume":
            state = restore_training(parent["path"], optimizer, scheduler, contract,
                                     expected_state=state, groups=groups, config=config)
        groups.validate_state(state)
        try:
            with attention_context(config.training.memory.attention_policy):
                result = train_grpo(model, groups, objective, optimizer, scheduler, state=state,
                    max_updates=stop, group_batch_size=config.grpo.group_batch_size,
                    grad_clip=config.training.grad_clip,
                    on_update=lambda *_: telemetry.finish(), parallel=context,
                    on_window_start=telemetry.start, memory_controller=memory_controller,
                    on_attempt_abort=telemetry.abort,
                    baseline=baseline_updater(config, model, shared["directory"], contract, optimizer, scheduler,
                                              memory_controller=memory_controller))
        finally:
            telemetry.close()
            groups.close()
        states = context.gather(result)
        if any(item != states[0] for item in states):
            raise ValueError("Distributed final cursors differ")
        return result
    from RL.grpo.prepared import PreparedGroups, PreparedReader, resolve
    from RL.grpo.contract import validate_reference_files
    from RL.grpo.backend import GroupObjective
    from RL.training import optimizer_and_scheduler, train_grpo
    from RL.checkpoints import restore_checkpoint
    from RL.memory import attention_context
    from RL.profiling import UpdateTelemetry
    prepared_reader = PreparedReader(resolve(contract["prepared"]))
    if prepared_reader.execution_plan:
        from RL.grpo.execution import PreparedExecution
        groups = PreparedExecution(prepared_reader, rank=context.rank, world_size=context.world_size)
    else:
        groups = PreparedGroups(prepared_reader)
    reference = validate_reference_files(Path(shared["directory"]), contract)
    objective = GroupObjective(reference, config.grpo, config.training.memory.loss_chunk_tokens,
                               offloader=offloader,
                               reference_scorer=dycon_scorer(config, model, shared["directory"]))
    optimizer, scheduler = optimizer_and_scheduler(model, optimizer_settings)
    if parent and config.run.start.mode in {"resume", "append_resume"}:
        from core.io import read_json
        saved_contract = read_json(Path(parent["path"]) / "state.json")["contract"]
        restore_checkpoint(parent["path"], optimizer, scheduler, saved_contract, contract["adapter"], verify=True)
        from RL.config_migration import apply_optimizer_config
        apply_optimizer_config(config, optimizer, scheduler)
    telemetry = UpdateTelemetry(model, objective, mlps, offloader, directory,
                                config.training.memory.profile_first_update,
                                config.training.memory.attention_policy == "flash_only")
    try:
        with attention_context(config.training.memory.attention_policy):
            result = train_grpo(model, groups, objective, optimizer, scheduler, state=state,
                max_updates=stop, group_batch_size=config.grpo.group_batch_size,
                grad_clip=config.training.grad_clip, on_window_start=telemetry.start,
                on_update=lambda *_: telemetry.finish(), parallel=context,
                memory_controller=memory_controller, on_attempt_abort=telemetry.abort,
                    baseline=baseline_updater(config, model, shared["directory"], contract, optimizer, scheduler,
                                              memory_controller=memory_controller))
    finally:
        telemetry.close()
    states = context.gather(result)
    if any(state != states[0] for state in states):
        raise ValueError("Distributed final cursors differ")
    return result
