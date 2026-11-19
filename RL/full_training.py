"""FSDP2 execution strategy inside the common RL service/attempt lifecycle."""
from datetime import timedelta
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from core.io import hash_json, now, read_json, write_json
from RL.artifacts import commit_receipt, dependency_versions
from RL.config import schedule_contract
from RL.full_policy import global_clip, load, local_tensor, microsteps, optimizer as make_optimizer
from RL.grpo.objective import grpo_objective


def _context(config, root):
    from RL.parallel import current, GroupParallel
    context = current()
    if context is not None:
        return context, None
    from RL.service import _select_gpu
    _select_gpu(config)
    torch.cuda.set_device(0)
    folder = tempfile.TemporaryDirectory(prefix="fsdp2-single-", dir=Path(root) / "temp")
    dist.init_process_group("gloo", init_method=Path(folder.name, "store").as_uri(),
                            rank=0, world_size=1,
                            timeout=timedelta(seconds=config.runtime.distributed_timeout_seconds))
    data = dist.new_group(backend="nccl", timeout=timedelta(seconds=config.runtime.distributed_timeout_seconds))
    return GroupParallel(0, 1, data_group=data), folder


def _group(reader, group_id):
    from RL.grpo.data import tensor_groups
    return next(iter(tensor_groups([reader.raw(group_id)])))


def _anchor(config, reader, policy, base, inventory, directory, context, parent):
    folder = Path(directory) / "anchor"
    if config.run.start.mode == "resume":
        if context.primary:
            shutil.copytree(Path(parent["path"]) / "anchor", folder)
        dist.barrier()
        return read_json(folder / "manifest.json")
    initial = {"schema_version": 1, "kind": "full_language_policy",
               "base_fingerprint": base["fingerprint"], "parameters_sha256": hash_json(inventory),
               "warm_start_receipt_sha256": parent["receipt_sha256"] if parent else None}
    if config.model.gdn_backend == "fla_chunk":
        from RL.gdn_backend import active_binding
        initial["gdn_execution"] = active_binding().execution_contract()
    from RL.full_policy import shard_digest
    initial["initial_parameter_shards"] = context.gather({"rank": context.rank, "sha256": shard_digest(policy)})
    source = None
    if config.grpo.reference_cache.source != "model_forward":
        from RL.dataset.probability import open_probability_stream
        from RL.grpo.contract import validate_scoring_identity
        source = open_probability_stream(config.grpo.reference_cache, config.algorithm, reader.proof, strict=True)
        validate_scoring_identity(source.scoring_identity, base=base, proof=reader.proof,
            training_dtype=config.model.dtype, policy=config.grpo.reference_cache.dtype_compatibility,
            adapter={"kind": "base_equivalent"})
    if context.primary:
        folder.mkdir()
    dist.barrier()
    entries = []
    policy.eval()
    with torch.no_grad():
        for g, group_id in enumerate(reader.ids):
            group = _group(reader, group_id)
            raw = reader.raw(group_id) if source else None
            for b, branch_id in enumerate(group.branch_ids):
                if source:
                    row = raw["members"][b]
                    if reader.proof["synthetic"]:
                        from RL.dataset.probability import synthetic_branch
                        row = synthetic_branch(row, reader.proof["tokenizer_fingerprint"])
                    values = torch.from_numpy(source.values(row, verify=True).copy()).float()
                else:
                    # Every rank follows the same inference collective schedule.
                    values = policy(group.branch_batch(b, "cuda:0"),
                                    config.training.memory.loss_chunk_tokens or 256).float().cpu()
                if values.shape != (len(group.completion_token_ids[b]),) or not bool(torch.isfinite(values).all()):
                    raise ValueError("Full frozen anchor has invalid targets")
                name = f"g{g:08d}-b{b:05d}.safetensors"
                entry = {"group_id": group_id, "branch_id": branch_id, "file": name,
                         "target_count": values.numel(), "prompt_sha256": hash_json(group.prompt_token_ids[b]),
                         "completion_sha256": hash_json(group.completion_token_ids[b]),
                         "values_sha256": __import__("hashlib").sha256(values.numpy().tobytes()).hexdigest()}
                entries.append(entry)
                if context.primary:
                    save_file({"values": values}, str(folder / name))
    if source:
        source.verify_unchanged()
    manifest = {"schema_version": 1, "policy_identity": initial,
                "input_sha256": hash_json(reader.proof), "probability_mode": "raw_t1",
                "roles": ["ratio_denominator", "k3_reference"], "entries": entries,
                "source": config.grpo.reference_cache.model_dump()}
    if source:
        manifest["provenance"] = source.provenance
    if context.primary:
        write_json(folder / "manifest.json", manifest)
    dist.barrier()
    return manifest


def window_update(policy, window, references, config, context, memory_controller=None, moment_optimizer=None, capacity_limit=None):
    """One fixed microstep table; placeholders have exactly zero contribution."""
    rows, local_loss = [], 0.0
    table = microsteps(window, context.world_size)
    if capacity_limit is not None:
        if capacity_limit > len(table):
            raise ValueError("Capacity microsteps exceeds this window's canonical microstep table")
        table = table[:capacity_limit]
    for microstep_index, assignments in enumerate(table):
        task = assignments[context.rank]
        g, b = task if task is not None else (0, 0)
        # Idle ranks still traverse the same FSDP2 modules/collectives. Reuse a
        # locally loaded branch; its zero loss never contributes to the update.
        group = window[g] if task is not None else next(
            (candidate for candidate in window if candidate.input_ids is not None), window[0])
        if moment_optimizer is not None:
            from RL.optimizer_offload import safety_point
            branch_length = len(group.prompt_token_ids[b]) + len(group.completion_token_ids[b])
            safety_point(moment_optimizer, policy, context,
                length=branch_length)
            moment_optimizer.begin_branch(branch_length)
            if memory_controller is not None:
                memory_controller.logger.write(event="adam_moment_control", phase="before_branch",
                    **memory_controller.context, microstep=microstep_index, sequence_tokens=branch_length,
                    adam_moment_offload=moment_optimizer.metrics())
        values = policy(group.branch_batch(b, "cuda:0"), config.training.memory.loss_chunk_tokens or 256)
        if task is None:
            loss = values.sum() * 0.0
        else:
            reference = references[(group.group_id, group.branch_ids[b])].to(values.device)
            settings = config.grpo
            result = grpo_objective(values[None], reference[None], [group.advantages[b]],
                torch.ones_like(values[None], dtype=torch.bool), clip_epsilon=settings.clip_epsilon,
                kl_beta=settings.kl_beta, ratio_guard_enabled=settings.ratio_guard.enabled,
                ratio_guard_threshold=settings.ratio_guard.threshold, k3_cap=settings.k3_cap)
            weight = 1 / (len(window) * group.group_size)
            loss = result.loss * (context.world_size * weight)
            local_loss += float(result.loss.detach()) * weight
            metrics = dict(result.metrics)
            if settings.token_diagnostics.enabled:
                from RL.grpo.token_diagnostics import branch_diagnostics
                metrics["_token_diagnostics"] = branch_diagnostics(values, reference, group, b, settings.token_diagnostics)
            rows.append({"group_id": group.group_id, "branch_id": group.branch_ids[b],
                         "weight": weight, "metrics": metrics,
                         "policy_loss": float(result.policy_loss.detach()),
                         "k3": float(result.k3_penalty.detach())})
        loss.backward()  # FSDP2 synchronizes and averages once at each microstep.
        del loss, values
        if memory_controller is not None:
            memory_controller.reclaim_completed()
            memory_controller.safe_boundary()
        if moment_optimizer is not None:
            moment_optimizer.finish_branch()
    peers = context.gather({"rows": rows, "loss": local_loss})
    all_rows = [r for p in peers for r in p["rows"]]
    if capacity_limit is None and window and "execution_chunk" in window[0].metadata:
        expected = [(group.group_id, bid) for group in window for bid in group.branch_ids]
        actual = [(row["group_id"], row["branch_id"]) for row in all_rows]
        if sorted(actual) != sorted(expected):
            raise ValueError("Full 8x16 missing or duplicate branch contributions")
    return sum(p["loss"] for p in peers), all_rows, table


def execute(config, root, directory, base, parent, attempt):
    context, single = _context(config, root)
    try:
        if attempt.get("operation") == "full_gradient_compare":
            from RL.full_gradient_compare import evaluate
            return evaluate(config, root, directory, base, attempt, context)
        if attempt.get("operation") == "full_probe":
            from RL.full_probe import evaluate
            return evaluate(config, parent, attempt["group_ids"], context)
        return _execute(config, root, directory, base, parent, attempt, context)
    finally:
        import sys
        active_error = sys.exc_info()[0] is not None
        controller = getattr(context, "full_memory_controller", None)
        if controller is not None:
            try:
                controller.close()
            except Exception:
                if not active_error:
                    raise
                # A failed autograd traceback may still own leases. Preserve
                # the primary failure; worker exit releases its CUDA context.
                import warnings
                warnings.warn("Full offload cleanup deferred to failed-worker exit", RuntimeWarning)
        opt = getattr(context, "full_moment_optimizer", None)
        if opt is not None:
            try:
                opt.close()
            except Exception:
                if not active_error:
                    raise
        from RL.host_admission import current, reset
        if current() is not None:
            try:
                reset()
            except Exception:
                if not active_error:
                    raise
        telemetry = getattr(context, "full_telemetry", None)
        if telemetry is not None:
            telemetry.close()
        if single is not None:
            dist.destroy_process_group()
            single.cleanup()


def _execute(config, root, directory, base, parent, attempt, context):
    from torch.distributed.device_mesh import DeviceMesh
    from RL.full_artifacts import prepared
    from RL.full_checkpoint import restore, save
    from RL.grpo.prepared import PreparedReader, binding
    from RL.grpo.contract import validate_input
    from RL.training import seed_everything
    from RL.resources import host_memory
    from RL.memory import attention_context
    directory = Path(directory)
    if context.primary:
        reader = prepared(config, root, parent)
        validate_input(base, reader.proof)
        shared = str(reader.path)
    else:
        shared = None
    reader = PreparedReader(context.broadcast(shared))
    training_groups = reader
    if reader.execution_plan:
        from RL.grpo.execution import PreparedExecution
        training_groups = PreparedExecution(reader, rank=context.rank, world_size=context.world_size,
                                            fsdp_replicate_empty=True)
    seed_everything(config.run.seed)
    from RL.probe import _configure_numerics
    _configure_numerics(config)
    from RL.full_backend import bind, checkpoint_identity, evidence
    expected = checkpoint_identity(parent["metadata"]["contract"]) if parent and config.run.start.mode == "resume" else None
    backend_binding = bind(config.model, context, expected)
    mesh = DeviceMesh.from_group(context.data_group, "cuda", mesh_dim_names=("data",))
    policy, inventory = load(config, mesh, context.rank, context.world_size)
    from RL.full_policy import activation_offloader
    from RL.memory import MemoryController
    admission = None
    if config.training.memory.adam_moment_offload.mode != "disabled":
        from RL.host_admission import configure
        admission = configure(root, config.training.memory.adam_moment_offload,
            config.training.memory.host_offload, attempt=config.run.attempt_id, rank=context.rank)
    offloader = activation_offloader(policy, config)
    controller = None
    if offloader is not None:
        controller = MemoryController(config, directory, rank=context.rank,
            world_size=context.world_size, offloader=offloader, device="cuda:0")
        context.full_memory_controller = controller
    opt, scheduler = make_optimizer(policy, inventory, config.training,
        pool=offloader.pool if offloader is not None else None, admission=admission)
    if hasattr(opt, "iter_local_state_chunks"):
        context.full_moment_optimizer = opt
    if parent:
        if parent["metadata"]["contract"]["world_size"] != context.world_size:
            raise ValueError("Full warm_start/resume requires the same shard topology")
        restore(parent["path"], policy, opt, scheduler, context, training=config.run.start.mode == "resume", config=config)
    if context.primary:
        write_json(directory / "trainable-parameters.json", inventory)
    with attention_context(config.training.memory.attention_policy):
        anchor = _anchor(config, reader, policy, base, inventory, directory, context, parent)
    contract = {"schema_version": 1, "schedule": schedule_contract(config),
                "base_fingerprint": base["fingerprint"], "input": reader.proof,
                "prepared": binding(reader.path), "parameters_sha256": hash_json(inventory),
                "prepared_schedule_id": reader.schedule["schedule_id"],
                "prepared_schedule_sha256": reader.schedule["sha256"],
                "world_size": context.world_size, "backend": "fsdp2_branch_rounds_v1",
                "initial_policy": anchor["policy_identity"], "anchor_sha256": hash_json(anchor),
                "dependencies": dependency_versions(),
                "precision": {"master": "float32", "moments": "float32", "reduce": "float32",
                              "compute": config.model.dtype},
                "hardware": [torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)]}
    if hasattr(opt, "iter_local_state_chunks"):
        from RL.optimizer_offload import BACKEND, LAYOUT
        contract["optimizer_backend"] = {"backend": BACKEND, "layout": LAYOUT,
            "arithmetic": "torch_2.13_nonfused_nonforeach_fp32", "policy_version": 1,
            "resume_equivalence": "validated_tolerance"}
        if config.run.start.optimizer_backend_migration is not None:
            contract["optimizer_migration"] = config.run.start.optimizer_backend_migration.model_dump()
        elif parent and "optimizer_migration" in parent["metadata"]["contract"]:
            contract["optimizer_migration"] = parent["metadata"]["contract"]["optimizer_migration"]
    if config.model.gdn_backend == "fla_chunk" or (parent and "gdn_execution" in parent["metadata"]["contract"]):
        contract["gdn_execution"] = backend_binding.execution_contract()
    # JSON normalization makes tuple/list identities stable across fresh/resume.
    contract = json.loads(json.dumps(contract))
    if config.run.start.mode == "resume":
        from RL.optimizer_migration import contract_matches
        if not contract_matches(config, parent["metadata"]["contract"], contract):
            raise ValueError("Full resume parameter/anchor/dependency/hardware contract mismatch")
    if config.run.start.mode == "resume":
        state = dict(parent["metadata"]["state"])
    else:
        state = {"global_step": 0, "group_cursor": 0, "consumed_groups": 0,
                 "consumed_branches": 0, "consumed_tokens": 0, **training_groups.cursor(0)}
    reader.validate_state(state)
    if context.primary:
        attempt.update(status="training", started_at=now(), contract=contract, initial_state=dict(state))
        write_json(directory / "attempt.json", attempt)
        write_json(directory / "grpo-contract.json", contract)
        write_json(directory / "input-identity.json", reader.proof)
        write_json(directory / "environment.json", {"dependencies": contract["dependencies"],
            "hardware": contract["hardware"], "execution": config.execution.model_dump(),
            "gdn": evidence(backend_binding)})
        from RL.observability import LiveTelemetry
        context.full_telemetry = LiveTelemetry({"project_root": str(root)}, config.run.attempt_id,
                                              config.runtime.output_root)
    lookup = {(e["group_id"], e["branch_id"]): e for e in anchor["entries"]}
    stop = config.validation.stop_after_update or config.training.max_updates
    policy.train()
    final = None
    capacity_observation = None
    start_cursor = state.get("consumed_execution_chunks", state["group_cursor"])
    for window in training_groups.windows(start_cursor):
        if state["global_step"] >= stop:
            break
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        if context.primary:
            context.full_telemetry.emit("full_branch_forward_backward", inflight_update=state["global_step"] + 1)
        if reader.execution_plan:
            from RL.grpo.execution import validate_window
            validate_window(reader.execution_plan, state["global_step"], window)
        references = {}
        for group in window:
            if reader.execution_plan and group.input_ids is None:
                continue
            for b, bid in enumerate(group.branch_ids):
                entry = lookup[(group.group_id, bid)]
                values = load_file(str(directory / "anchor" / entry["file"]))["values"]
                if (values.shape != (len(group.completion_token_ids[b]),)
                        or entry["prompt_sha256"] != hash_json(group.prompt_token_ids[b])
                        or entry["completion_sha256"] != hash_json(group.completion_token_ids[b])
                        or entry["values_sha256"] != __import__("hashlib").sha256(values.numpy().tobytes()).hexdigest()):
                    raise ValueError("Full frozen anchor changed or target coordinates differ")
                references[(group.group_id, bid)] = values
        opt.zero_grad(set_to_none=True)
        if hasattr(opt, "prepare"):
            from RL.optimizer_offload import safety_point
            maximum_length = max(len(g.prompt_token_ids[b]) + len(g.completion_token_ids[b])
                for g in window if g.input_ids is not None for b in range(g.group_size))
            safety_point(opt, policy, context, length=maximum_length, new_window=True, allow_restore=True)
        if controller is not None:
            offloader.reset()
            controller.decide(window, global_step=state["global_step"],
                              group_cursor=state.get("consumed_execution_chunks", state["group_cursor"]), parallel=context)
            controller.begin_attempt(window, global_step=state["global_step"], attempt_id=0)
        with attention_context(config.training.memory.attention_policy):
            loss, rows, table = window_update(policy, window, references, config, context, controller,
                moment_optimizer=opt if hasattr(opt, "prepare") else None,
                capacity_limit=config.validation.capacity_microsteps)
        if config.validation.capacity_microsteps is not None:
            # Capacity probes execute whole native branches through this same
            # FSDP/GRPO path, but never apply their partial-window gradients.
            # All moments were initialized by safety_point before the probe.
            from RL.full_policy import validate_optimizer_state
            from RL.optimizer_offload import coordinated
            norm = global_clip(policy, config.training.grad_clip, context.data_group)
            validation = coordinated(context, lambda: validate_optimizer_state(policy, opt))
            torch.cuda.synchronize()
            observed = context.gather({"rank": context.rank,
                "adam_moment_offload": opt.metrics(), "optimizer_validation": validation,
                "activation_offload": offloader.metrics() if offloader is not None else None,
                "peak_allocated_bytes": max(torch.cuda.max_memory_allocated(), opt.observed_window_peak),
                "peak_reserved_bytes": max(torch.cuda.max_memory_reserved(), opt.observed_window_reserved),
                "host": host_memory(), "branches": [
                    {"group_id": window[g].group_id, "branch_id": window[g].branch_ids[b],
                     "sequence_tokens": len(window[g].prompt_token_ids[b]) + len(window[g].completion_token_ids[b])}
                    for assignment in table if (task := assignment[context.rank]) is not None for g,b in [task]]})
            capacity_observation = {"kind": "full_native_branch_capacity_v1", "committed_updates": 0,
                "partial_gradient_norm": norm, "microsteps_executed": len(table), "ranks": observed,
                "trainable_parameters": inventory["trainable_parameters"],
                "compute_seconds": time.monotonic() - started,
                "optimizer_step_applied": False, "cursor_unchanged": True}
            if context.primary:
                write_json(directory / "capacity.json", capacity_observation)
            break
        norm = global_clip(policy, config.training.grad_clip, context.data_group)
        from RL.full_policy import parameter_samples
        before = parameter_samples(policy)
        used_learning_rate = opt.param_groups[0]["lr"]
        if hasattr(opt, "prepare"):
            from RL.optimizer_offload import safety_point, coordinated
            safety_point(opt, policy, context, step=True)
            coordinated(context, opt.step)
        else:
            opt.step()
        after = parameter_samples(policy)
        sampled_delta = max((float((after[n] - value).abs().max())
                             for n, value in before.items() if value.numel()), default=0.0)
        del before, after
        from RL.full_policy import validate_optimizer_state
        if hasattr(opt, "prepare"):
            from RL.optimizer_offload import coordinated
            optimizer_validation = coordinated(context, lambda: validate_optimizer_state(policy, opt))
        else:
            optimizer_validation = validate_optimizer_state(policy, opt)
        # Validate the complete distributed update before committing scheduler
        # or data cursor progress. Validation failure makes this attempt terminal.
        scheduler.step()
        if reader.execution_plan:
            state["global_step"] += 1
            state.update(training_groups.cursor(state["consumed_execution_chunks"] + len(window)))
        else:
            state["global_step"] += 1
            state["group_cursor"] += len(window)
            state["consumed_groups"] += len(window)
            state["consumed_branches"] += sum(g.group_size for g in window)
            state["consumed_tokens"] += sum(len(c) for g in window for c in g.completion_token_ids)
            state.update(reader.cursor(state["group_cursor"]))
        training_groups.validate_state(state)
        torch.cuda.synchronize()
        if controller is not None:
            controller.safe_boundary()
            controller.committed(global_step=state["global_step"], window=window, attempt_id=0)
        telemetry = context.gather({"rank": context.rank,
            **({"activation_offload": offloader.metrics(), "offload_decision": controller.last_decision}
               if controller is not None else {}),
            **({"adam_moment_offload": opt.metrics()} if hasattr(opt, "metrics") else {}),
            "gdn": evidence(backend_binding), "optimizer_validation": optimizer_validation,
            "sampled_parameter_max_delta": sampled_delta,
            "peak_allocated_bytes": max(torch.cuda.max_memory_allocated(), getattr(opt, "observed_window_peak", 0)),
            "peak_reserved_bytes": max(torch.cuda.max_memory_reserved(), getattr(opt, "observed_window_reserved", 0)), "host": host_memory(),
            "local_master_bytes": sum(local_tensor(p).numel() * 4 for p in policy.parameters())})
        metrics = {"global_step": state["global_step"], "loss": loss,
                   "policy_loss": sum(r["weight"] * r["policy_loss"] for r in rows),
                   "k3_unweighted": sum(r["weight"] * r["k3"] for r in rows),
                   "global_grad_norm": norm, "learning_rate": used_learning_rate,
                   "next_learning_rate": scheduler.get_last_lr()[0],
                   "gradient_clip_threshold": config.training.grad_clip,
                   "gradient_clip_scale": min(1.0, config.training.grad_clip / (norm + 1e-6)),
                   "policy_changed": any(t["sampled_parameter_max_delta"] > 0 for t in telemetry),
                   "update_evidence": "up_to_64_evenly_spaced_values_per_local_parameter",
                   "trainable_parameters": inventory["trainable_parameters"],
                   "compute_seconds": time.monotonic() - started, "ranks": telemetry,
                   "state_dtypes": contract["precision"], "microsteps": len(table), "branches": rows,
                   "state": dict(state)}
        from RL.grpo.metrics import summarize_update
        metrics.update(summarize_update([[r["metrics"] for r in rows if r["group_id"] == g.group_id] for g in window]))
        for row in rows:
            row["metrics"] = {k: v for k, v in row["metrics"].items() if not k.startswith("_")}
        metrics.update({k: v for k, v in state.items() if k != "state"})
        metrics.update(gradient_norm=norm, step_seconds=metrics["compute_seconds"], observed_at=now())
        if context.primary:
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            write_json(directory / "progress.json", {"status": "training", "state": state,
                "planned_updates": config.training.max_updates, "updated_at": now()})
        if not config.validation.skip_checkpoint_export and (state["global_step"] % config.training.checkpoint_every_updates == 0 or state["global_step"] == stop):
            if context.primary:
                context.full_telemetry.emit("checkpoint", inflight_update=state["global_step"])
            final = save(directory, policy, opt, scheduler, state, contract, inventory, context)
        if context.primary:
            context.full_telemetry.emit("update_committed", inflight_update=None, global_step=state["global_step"])
    if capacity_observation is None and (state["global_step"] != stop or (final is None and not config.validation.skip_checkpoint_export)):
        raise ValueError("Full prepared schedule ended before requested update boundary")
    if controller is not None:
        # Closing emits rank-local memory evidence. Flush every rank BEFORE
        # primary hashes the attempt; immutable receipt files never change later.
        controller.close()
        dist.barrier()
    if hasattr(opt, "close") and hasattr(opt, "iter_local_state_chunks"):
        opt.close()
        dist.barrier()
    status = "completed" if capacity_observation is not None or stop == config.training.max_updates else "paused"
    if context.primary:
        attempt.update(status=status, ended_at=now(), final_state=state, final_checkpoint=final,
                       resumable=final is not None, validation_only=config.validation.skip_checkpoint_export,
                       capacity_only=capacity_observation is not None)
        write_json(directory / "attempt.json", attempt)
        write_json(directory / "progress.json", {"status": status, "state": state,
            "planned_updates": config.training.max_updates, "updated_at": now()})
        write_json(directory / "validation.json", {"state": state, "checkpoint": final,
            "language_parameters": inventory["trainable_parameters"], "vision": "immutable_base_on_disk",
            **({"capacity": capacity_observation} if capacity_observation is not None else {}),
            "gdn": evidence(backend_binding)})
        commit_receipt(directory, "rl-full-attempt-v1", status=status, contract_sha256=hash_json(contract))
        context.full_telemetry.emit(status, inflight_update=None)
    dist.barrier()
    return {"attempt_id": config.run.attempt_id, "status": status, "checkpoint": final, "state": state}
