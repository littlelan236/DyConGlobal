"""Loss-independent single-device optimizer loop with token-weighted accumulation."""

import itertools
import random
import time

import numpy as np
import torch

from core.io import hash_json
from RL.supervision import prepare_batch
from RL.optimizer_signals import gradient_signals, parameter_sample, update_signals


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG topology differs from the current single-device runtime")
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_and_scheduler(model, settings):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate, betas=(0.9, 0.999),
                                  weight_decay=settings.weight_decay)
    from RL.lr_schedule import factor
    schedule = lambda step: factor(step, settings)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def train(model, batches, loss_callback, optimizer, scheduler, *, max_updates,
          gradient_accumulation_steps, grad_clip, vocab_size, max_length,
          state=None, on_update=None, on_gradients=None, on_window_start=None, supervision=None):
    state = dict(state or {"global_step": 0, "consumed_microbatches": 0, "consumed_tokens": 0})
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(expected) or {id(p) for p in actual} != expected:
        raise ValueError("Optimizer must contain each trainable adapter parameter exactly once")
    device = next(model.parameters()).device
    iterator = iter(batches)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while state["global_step"] < max_updates:
        window = list(itertools.islice(iterator, gradient_accumulation_steps))
        if not window:
            break
        # Resolve every sample before this window's first forward/backward. Loss
        # callbacks consume these effective labels, never select a second time.
        selections = [prepare_batch(batch, vocab_size=vocab_size, max_length=max_length,
                                    supervision=supervision) for batch in window]
        validate_prepared = getattr(loss_callback, "validate_prepared_batch", None)
        if validate_prepared:
            for selection in selections:
                validate_prepared(selection.model_batch())
        total_tokens = sum(selection.selected_count for selection in selections)
        available_tokens = sum(selection.available_count for selection in selections)
        if on_window_start:
            on_window_start()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        total_loss = 0.0
        forward_seconds = backward_seconds = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for selection in selections:
            batch, expected_count = selection.model_batch(), selection.selected_count
            phase_started = time.monotonic()
            loss_sum, count = loss_callback(model, {key: value.to(device) for key, value in batch.items()})
            if count != expected_count or loss_sum.ndim != 0 or not torch.isfinite(loss_sum):
                raise ValueError("Nonfinite loss or invalid loss callback supervision contract")
            total_loss += float(loss_sum.detach())
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds += time.monotonic() - phase_started
            phase_started = time.monotonic()
            (loss_sum / total_tokens).backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            backward_seconds += time.monotonic() - phase_started
        for parameter in actual:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite gradient; optimizer update rejected")
        if on_gradients:
            on_gradients()
        norm = torch.nn.utils.clip_grad_norm_(actual, grad_clip, error_if_nonfinite=True)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("No effective trainable gradient")
        optimizer_metrics = gradient_signals(actual, norm, grad_clip)
        before_parameters = parameter_sample(actual, state["global_step"] + 1)
        optimizer.step()
        optimizer_metrics.update(update_signals(actual, before_parameters))
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["global_step"] += 1
        state["consumed_microbatches"] += len(window)
        state["consumed_tokens"] += total_tokens
        memory = {}
        if device.type == "cuda":
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "free_bytes": free, "total_bytes": total}
        metrics = {**state, "loss": total_loss / total_tokens, "supervised_tokens": total_tokens,
                   "microbatches": len(window), "learning_rate": learning_rate,
                   "next_learning_rate": optimizer.param_groups[0]["lr"],
                   "gradient_norm": float(norm), "seconds": time.monotonic() - started, **memory,
                   "evaluated_at_global_step": state["global_step"] - 1, **optimizer_metrics}
        metrics.update(forward_seconds=forward_seconds, backward_seconds=backward_seconds)
        metrics.update(input_tokens=sum(selection.input_tokens for selection in selections),
                       available_supervised_tokens=available_tokens,
                       supervision_fraction=total_tokens / available_tokens,
                       selection_sha256=hash_json([selection.sha256 for selection in selections]))
        if on_update:
            on_update(dict(state), metrics)
    return state


def _grpo_local_attempt(model, window, objective, parameters, *, cursor, parallel,
                        stage, memory, attempt, step, on_start, rng, execution_plan=None, kl_only=False):
    """Return only scalar evidence; no exception traceback escapes this scope."""
    import traceback
    from RL.memory import is_cuda_oom
    device = next(model.parameters()).device
    started = time.monotonic()
    forward = backward = loss_value = 0.0
    rows, phase = [], "step_start"
    def enter(name, **fields):
        nonlocal phase
        phase = name
        if memory: memory.phase(name, **fields)
    try:
        if rng is not None: restore_rng(rng)
        if not kl_only and hasattr(objective, "reset"): objective.reset()
        if on_start: on_start()
        if execution_plan:
            from RL.grpo.execution import validate_window
            validate_window(execution_plan, step, window)
        if memory: memory.begin_attempt(window, global_step=step, attempt_id=attempt)
        elif device.type == "cuda":
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        from RL.topology import portable, tasks, assert_rng_unchanged
        remapped = portable(parallel)
        work = (tasks(window, rank=parallel.rank, world_size=parallel.world_size)
                if remapped and not execution_plan else
                ((i, g, 1.0) for i, g in enumerate(window)
                 if parallel is None or i % parallel.world_size == parallel.rank))
        for index, group, parent_fraction in work:
            compute_rng = capture_rng() if remapped else None
            fields = {"group_id": group.group_id, "group_ordinal": cursor + index,
                      "window_ordinal": index, "branch_ordinal": None}
            if execution_plan:
                fields.update(group.metadata["execution_chunk"], execution_chunk_ordinal=cursor + index,
                              group_ordinal_semantics="execution_chunk")
            stage("group_forward", **{k:v for k,v in fields.items() if k != "window_ordinal"},
                  branch_count=group.group_size)
            results = None
            try:
                begin = getattr(objective, "begin_group", None)
                if begin:
                    enter("prefix_forward", **fields); stage("prefix_forward")
                    before = time.monotonic(); begin(model, group)
                    if device.type == "cuda": torch.cuda.synchronize(device)
                    forward += time.monotonic() - before
                results = (objective.iter_results(model, group) if hasattr(objective, "iter_results") else
                           (objective(model, item) for item in (group,)))
                group_rows = []
                while True:
                    enter("suffix_forward", branch_ordinal=len(group_rows), **{k:v for k,v in fields.items() if k != "branch_ordinal"})
                    stage("group_forward", contribution_ordinal=len(group_rows))
                    before = time.monotonic(); result = next(results, None)
                    if device.type == "cuda": torch.cuda.synchronize(device)
                    forward += time.monotonic() - before
                    if result is None: break
                    loss = (result.k3_penalty * objective.settings.kl_beta / group.group_size
                            if kl_only else result.loss)
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise ValueError("Nonfinite GRPO objective; optimizer update rejected")
                    loss_value += float(loss.detach()) * parent_fraction / len(window)
                    enter("suffix_backward", branch_ordinal=len(group_rows))
                    stage("group_backward", contribution_ordinal=len(group_rows))
                    before = time.monotonic(); (loss * parent_fraction / len(window)).backward()
                    if device.type == "cuda": torch.cuda.synchronize(device)
                    duration = time.monotonic() - before; backward += duration
                    if hasattr(objective, "record_branch_backward"): objective.record_branch_backward(duration)
                    group_rows.append(result.metrics)
                    del loss, result
                    enter("boundary_accumulate")
                    if hasattr(objective, "after_branch_backward"): objective.after_branch_backward()
                    if memory and hasattr(memory, "reclaim_completed"): memory.reclaim_completed()
                if not group_rows: raise ValueError("Objective returned no group contribution")
                if execution_plan:
                    chunk = group.metadata["execution_chunk"]
                    if ([r.get("branch_id") for r in group_rows] != group.branch_ids
                            or [r.get("native_branch_ordinal") for r in group_rows] != chunk["native_branch_ordinals"]):
                        raise ValueError("8x16 missing/duplicate/misordered branch contributions")
                if hasattr(objective, "finish_group"):
                    enter("prefix_backward", branch_ordinal=None); stage("prefix_backward")
                    before = time.monotonic(); objective.finish_group()
                    if device.type == "cuda": torch.cuda.synchronize(device)
                    backward += time.monotonic() - before
                if memory and hasattr(memory, "reclaim_completed"): memory.reclaim_completed()
                if remapped:
                    assert_rng_unchanged(compute_rng)
                rows.append((index, group_rows))
                stage("group_complete", completed_local_groups=len(rows),
                      completed_groups_in_update=(None if parallel else
                          (index + 1 if not execution_plan or group.metadata["execution_chunk"]["half_index"] == 1 else 0)),
                      completed_local_execution_chunks=len(rows) if execution_plan else None)
            except BaseException as exc:
                if memory and is_cuda_oom(exc):
                    memory.oom(f"{type(exc).__name__}: {exc}", phase=phase)
                raise
            finally:
                result = loss = None
                try:
                    if results is not None and hasattr(results, "close"): results.close()
                finally:
                    if hasattr(objective, "close_group"): objective.close_group()
        for parameter in parameters:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite GRPO gradient; optimizer update rejected")
        if memory: memory.safe_boundary()
        return {"status": {"state": "OK", "phase": phase}, "rows": rows, "loss": loss_value,
                "prefixes": (objective.drain_prefix_metrics() if hasattr(objective, "drain_prefix_metrics")
                             else list(getattr(objective, "prefix_metrics", []))),
                "seconds": time.monotonic() - started, "forward_seconds": forward, "backward_seconds": backward}
    except BaseException as exc:
        if hasattr(objective, "drain_prefix_metrics"): objective.drain_prefix_metrics()
        # The helper returns, dropping all loss/result/generator/exception frame
        # references before the caller runs gc or allocator cleanup.
        return {"status": {"state": "CUDA_OOM" if is_cuda_oom(exc) else "FATAL", "phase": phase,
                           "error_type": type(exc).__name__, "error": str(exc)[:16384],
                           "traceback": traceback.format_exc()[-16384:]},
                "seconds": time.monotonic() - started}


def _rollback_grpo(model, objective, optimizer, buffers, on_abort, memory):
    """No optimizer update has run. Return cleanup status for the control group."""
    import gc
    try:
        optimizer.zero_grad(set_to_none=True)
        if hasattr(objective, "close_group"): objective.close_group()
        if hasattr(objective, "reset"): objective.reset()
        if on_abort: on_abort()
        if buffers is not None:
            modules = dict(model.named_modules())
            with torch.no_grad():
                for name, value in buffers.items():
                    owner, _, leaf = name.rpartition(".")
                    module = modules[owner]
                    current = module._buffers[leaf]
                    if current.shape != value.shape:
                        module._buffers[leaf] = value.to(current.device).clone()
                    else:
                        current.copy_(value)
        gc.collect()
        device = next(model.parameters()).device
        if device.type == "cuda":
            # A poisoned context or an asynchronous error must become FATAL.
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        if memory and memory.offloader and memory.offloader.live_bytes:
            raise RuntimeError("Pinned checkpoint activations survived whole-step cleanup")
        if memory:
            release = getattr(memory.offloader, "release_failed_trace", None)
            if release is not None:
                release()
            memory.safe_boundary()
        return {"state": "OK"}
    except BaseException as exc:
        return {"state": "FATAL", "error": f"Cleanup failed: {type(exc).__name__}: {exc}"}


def train_grpo(model, groups, objective_callback, optimizer, scheduler, *,
               group_batch_size=4, max_epochs=1, grad_clip=1.0, state=None,
               max_updates=None, on_update=None, on_gradients=None, on_window_start=None,
               on_stage=None, parallel=None, memory_controller=None, on_attempt_abort=None, baseline=None):
    """Whole-group local attempts, Gloo acknowledgement, then one committed update."""
    if max_epochs != 1: raise ValueError("offline GRPO supports exactly one epoch")
    from RL.grpo.groups import iter_group_batches
    def stage(name, **fields):
        if on_stage:
            try: on_stage(name, **fields)
            except Exception as exc:
                import warnings
                warnings.warn("RL stage callback failed: " + str(exc))
    execution_plan = getattr(groups, "execution_plan", None)
    state = dict(state or {"global_step": 0, "group_cursor": 0,
                           "consumed_groups": 0, "consumed_branches": 0, "consumed_tokens": 0})
    if execution_plan and state["global_step"] == 0 and "cursor_schema" not in state:
        state.update(groups.cursor(0))
    def consumed():
        return state["consumed_execution_chunks"] if execution_plan else state["group_cursor"]
    if state["group_cursor"] != state["consumed_groups"] or state["group_cursor"] < 0:
        raise ValueError("Invalid GRPO resume group cursor")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if not expected or len(parameters) != len(expected) or {id(p) for p in parameters} != expected:
        raise ValueError("Optimizer must contain each trainable adapter parameter exactly once")
    prepared = hasattr(groups, "validate_state") and hasattr(groups, "windows")
    all_groups = groups if prepared else list(groups)
    expected_step = (state["group_cursor"] + group_batch_size - 1) // group_batch_size
    if not prepared and (state["global_step"] != expected_step or
            (state["group_cursor"] != len(all_groups) and state["group_cursor"] % group_batch_size)):
        raise ValueError("Resume cursor is not an optimizer update boundary")
    if consumed() > len(all_groups): raise ValueError("Resume cursor is beyond the frozen dataset")
    if prepared: all_groups.validate_state(state)
    else:
        skipped = all_groups[:state["group_cursor"]]
        if state["consumed_branches"] != sum(g.group_size for g in skipped): raise ValueError("Resume branch cursor mismatch")
        if state["consumed_tokens"] != sum(int(g.loss_mask.sum()) for g in skipped): raise ValueError("Resume token cursor mismatch")
    device = next(model.parameters()).device
    memory = memory_controller
    model.train(); optimizer.zero_grad(set_to_none=True)
    windows = iter(all_groups.windows(consumed()) if prepared else
                   iter_group_batches(all_groups[state["group_cursor"]:], group_batch_size))
    try:
        while max_updates is None or state["global_step"] < max_updates:
            # The yielded CPU window/reference stays alive across both attempts.
            step_started = time.monotonic()
            window = next(windows, None)
            if window is None: break
            queue_wait_seconds = time.monotonic() - step_started
            step = state["global_step"] + 1
            stage("window", inflight_update=step,
                  window_end_group_ordinal=None if execution_plan else state["group_cursor"] + len(window),
                  execution_window_cursor=state.get("execution_window_cursor"),
                  window_end_execution_chunk=consumed() + len(window) if execution_plan else None,
                  completed_groups_in_update=0, group_id=None, branch_ordinal=None)
            decision = (memory.decide(window, global_step=state["global_step"],
                        group_cursor=consumed(), parallel=parallel) if memory else {"mode": "none"})
            retry_limit = memory.retry_limit(decision) if memory else 0
            step_rng = capture_rng() if retry_limit else None
            # Qwen's explicit cache facade is per-forward, not a model buffer.
            # Snapshot registered rotary/other mutable buffers without copying weights.
            buffers = {name: value.detach().clone() for name, value in model.named_buffers()} if retry_limit else None
            failed_seconds = 0.
            for attempt in range(retry_limit + 1):
                if attempt: memory.retry(window, global_step=state["global_step"], attempt_id=attempt)
                attempt_rng = (capture_rng() if baseline is not None and baseline.replay_kl
                               and step % baseline.attribution_every == 0 else None)
                local = _grpo_local_attempt(model, window, objective_callback, parameters,
                    cursor=consumed(), parallel=parallel, stage=stage, memory=memory,
                    attempt=attempt, step=state["global_step"], on_start=on_window_start,
                    rng=step_rng if attempt else None, execution_plan=execution_plan)
                if baseline is not None and local["status"]["state"] == "OK":
                    try:
                        def replay_kl():
                            after_rng = capture_rng()
                            try:
                                restore_rng(attempt_rng)
                                result = _grpo_local_attempt(
                                    model, window, objective_callback, parameters,
                                    cursor=consumed(), parallel=parallel, stage=lambda *_args, **_kw: None,
                                    memory=None, attempt=attempt, step=state["global_step"],
                                    on_start=None, rng=None, execution_plan=execution_plan, kl_only=True)
                                if result["status"]["state"] == "CUDA_OOM":
                                    raise torch.cuda.OutOfMemoryError(result["status"]["error"])
                                if result["status"]["state"] != "OK":
                                    raise RuntimeError(str(result["status"]))
                            finally:
                                restore_rng(after_rng)
                        local["baseline"], local["baseline_cursor"], local["attribution"] = baseline(
                            state, parallel, replay_kl=replay_kl)
                    except Exception as exc:
                        retryable = isinstance(exc, torch.cuda.OutOfMemoryError) and retry_limit
                        local["status"] = {"state": "CUDA_OOM" if retryable else "FATAL",
                                           "phase": "offline_baseline", "error": str(exc),
                                           "error_type": type(exc).__name__}
                if memory: memory.pause()
                status = {**local["status"], "rank": parallel.rank if parallel else 0,
                          "memory": memory.logger.memory() if memory else None,
                          "offload_metrics": memory.offload_metrics() if memory else None}
                statuses = parallel.control_step(step, attempt, "compute_result", status) if parallel else [status]
                if memory: memory.result(statuses)
                errors = [peer for peer in statuses if peer["state"] != "OK"]
                if not errors: break
                failed_seconds += local["seconds"]
                # Discard successful local rows as well when another rank failed.
                del local
                if memory: memory.cleanup(before=True, statuses=statuses)
                cleanup = _rollback_grpo(model, objective_callback, optimizer, buffers,
                                         on_attempt_abort, memory)
                cleanups = parallel.control_step(step, attempt, "cleanup", cleanup) if parallel else [cleanup]
                if memory: memory.cleanup(before=False, statuses=cleanups)
                fatals = [peer for peer in statuses + cleanups if peer["state"] == "FATAL"]
                if not fatals and attempt < retry_limit: continue
                detail = fatals[0] if fatals else errors[0]
                if memory:
                    memory.logger.write(event="retry_failed" if attempt else "step_failed", phase=detail.get("phase"),
                        **memory.context, rank_results=statuses, cleanup_results=cleanups, reason=detail.get("error"))
                exception_type = ValueError if detail.get("error_type") == "ValueError" else RuntimeError
                raise exception_type("GRPO step failed: " + str(detail))
            # All peers acknowledged OK. Nothing below this boundary is retried,
            # including gradient synchronization, clipping, optimizer and scheduler.
            learning_rate = optimizer.param_groups[0]["lr"]
            rows, loss_value, prefixes = [row for _, row in local["rows"]], local["loss"], local["prefixes"]
            parallel_metrics = None
            if parallel:
                stage("gradient_sync")
                if memory:
                    memory.phase("gradient_sync", group_id=None, group_ordinal=None, window_ordinal=None, branch_ordinal=None)
                    parallel.attempt_peak_allocated_bytes = memory.peaks["peak_allocated_bytes"]
                    parallel.attempt_peak_reserved_bytes = memory.peaks["peak_reserved_bytes"]
                parallel.local_phase_seconds = {"forward_seconds": local["forward_seconds"],
                    "backward_seconds": local["backward_seconds"], "queue_wait_seconds": queue_wait_seconds,
                    **getattr(all_groups, "window_io", {})}
                parallel.local_memory = memory.logger.memory() if memory else None
                parallel.local_offload_metrics = memory.offload_metrics() if memory else None
                rows, loss_value, prefixes, parallel_metrics = parallel.finish_window(
                    model, parameters, window, local["rows"], loss_value, prefixes, local["seconds"])
                stage("gradient_sync_complete", completed_groups_in_update=len(window))
            if baseline is not None and not local["baseline"].get("skipped"):
                attribution = local.pop("attribution")
                if attribution is not None:
                    if parallel:
                        from RL.online_parallel import reduce_attribution
                        reduce_attribution(parallel, parameters, attribution)
                    summary = attribution.summary()
                    local["baseline"]["gradient_attribution"] = summary
                    del attribution
                baseline_metrics = local["baseline"]
                peers = parallel.gather(baseline_metrics) if parallel else [baseline_metrics]
                baseline_metrics["rank_compute_seconds"] = [
                    peer["source_compute_seconds"].get("baseline", 0.) for peer in peers]
                baseline_metrics["loss"] = sum(peer["loss"] for peer in peers)
                baseline_metrics["token_count"] = sum(peer["token_count"] for peer in peers)
                loss_value = loss_value * baseline_metrics["dycon_weight"] + baseline_metrics["loss"]
            for parameter in parameters:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError("Nonfinite GRPO gradient; optimizer update rejected")
            if on_gradients: on_gradients()
            norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
            if not torch.isfinite(norm): raise ValueError("Nonfinite GRPO gradient norm")
            optimizer_metrics = gradient_signals(parameters, norm, grad_clip)
            before_parameters = parameter_sample(parameters, state["global_step"] + 1)
            stage("optimizer_update")
            if memory: memory.phase("optimizer_update", group_id=None, group_ordinal=None, window_ordinal=None, branch_ordinal=None)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            optimizer_metrics.update(update_signals(parameters, before_parameters))
            if parallel: parallel.after_step(model)
            next_count = consumed() + len(window)
            state["global_step"] += 1
            if baseline is not None:
                state["baseline_cursor"] = local["baseline_cursor"]
            if not execution_plan:
                state["group_cursor"] += len(window)
                state["consumed_groups"] += len(window)
            branches = sum(group.group_size for group in window)
            tokens = sum(int(group.loss_mask.sum()) for group in window)
            state["consumed_branches"] += branches; state["consumed_tokens"] += tokens
            if prepared: state.update(all_groups.cursor(next_count))
            if hasattr(all_groups, "commit_window"): all_groups.commit_window(window)
            from RL.grpo.metrics import summarize_update
            metrics = {**summarize_update(rows), **state, "loss": loss_value,
                "group_count": len(window), "branch_count": branches, "token_count": tokens,
                "gradient_norm": float(norm), "learning_rate": learning_rate,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.monotonic() - step_started, "queue_wait_seconds": queue_wait_seconds,
                "forward_seconds": local["forward_seconds"],
                "backward_seconds": local["backward_seconds"], "failed_attempt_seconds": failed_seconds,
                "successful_attempt_seconds": local["seconds"], "step_attempt": attempt,
                "evaluated_at_global_step": state["global_step"] - 1, **optimizer_metrics}
            if parallel and getattr(parallel, "physical_execution", None):
                metrics["physical_execution"] = dict(parallel.physical_execution)
            if device.type == "cuda":
                torch.cuda.synchronize(device); free, total = torch.cuda.mem_get_info(device)
                metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(device), free_bytes=free, total_bytes=total)
            if memory:
                memory.committed(global_step=state["global_step"], window=window, attempt_id=attempt)
                metrics.update(memory.peaks)
                metrics["offload_decision"] = memory.last_decision
                metrics.update(memory.logger.memory())
                metrics.update(memory.offload_metrics())
            if execution_plan:
                metrics.update(parent_group_count=len(window), execution_chunk_count=len(window),
                    parent_branch_count=32, completed_parent_groups_in_update=(
                        len(window) if window[0].metadata["execution_chunk"]["half_index"] == 1 else 0),
                    branch_loss_weight=1 / (16 * len(window)), metrics_schema_version=3)
                metrics.update(getattr(all_groups, "window_io", {}))
            if baseline is not None:
                metrics["offline_baseline"] = local["baseline"]
            if prefixes: metrics["prefix_reuse"] = prefixes
            if parallel_metrics:
                metrics["parallel"] = parallel_metrics
                for key in ("gradient_sync_seconds", "rank_critical_path_seconds"):
                    if key in parallel_metrics:
                        metrics[key] = parallel_metrics[key]
            from RL.grpo.signals import throughput
            metrics.update(throughput(metrics))
            if on_update: on_update(dict(state), metrics)
    except BaseException as exc:
        stage("exception")
        if memory:
            memory.logger.write(event="terminal_failure", phase=memory.phase_name,
                **memory.context, error=f"{type(exc).__name__}: {exc}"[:16384],
                **memory.logger.memory(), offload=memory.last_decision)
        optimizer.zero_grad(set_to_none=True)
        if hasattr(objective_callback, "drain_prefix_metrics"): objective_callback.drain_prefix_metrics()
        if hasattr(objective_callback, "close_group"): objective_callback.close_group()
        raise
    finally:
        if hasattr(windows, "close"): windows.close()
    return state
