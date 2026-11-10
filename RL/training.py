"""Loss-independent single-device optimizer loop with token-weighted accumulation."""

import itertools
import random
import time

import numpy as np
import torch

from core.io import hash_json
from RL.supervision import prepare_batch


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
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG topology differs from the current single-device runtime")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_and_scheduler(model, settings):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate, betas=(0.9, 0.999),
                                  weight_decay=settings.weight_decay)
    schedule = (lambda step: max(0.0, 1 - step / settings.max_updates)) if settings.scheduler == "linear" else (
        lambda step: 1.0)
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
        optimizer.step()
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
                   "gradient_norm": float(norm), "seconds": time.monotonic() - started, **memory}
        metrics.update(forward_seconds=forward_seconds, backward_seconds=backward_seconds)
        metrics.update(input_tokens=sum(selection.input_tokens for selection in selections),
                       available_supervised_tokens=available_tokens,
                       supervision_fraction=total_tokens / available_tokens,
                       selection_sha256=hash_json([selection.sha256 for selection in selections]))
        if on_update:
            on_update(dict(state), metrics)
    return state


def train_grpo(model, groups, objective_callback, optimizer, scheduler, *,
               group_batch_size=4, max_epochs=1, grad_clip=1.0, state=None,
               max_updates=None, on_update=None, on_gradients=None, on_window_start=None):
    """Consume frozen groups once, with group-preserving windows and resumable cursors."""
    if max_epochs != 1:
        raise ValueError("offline GRPO supports exactly one epoch")
    from RL.grpo.groups import iter_group_batches
    state = dict(state or {"global_step": 0, "group_cursor": 0,
                           "consumed_groups": 0, "consumed_branches": 0,
                           "consumed_tokens": 0})
    if state["group_cursor"] != state["consumed_groups"] or state["group_cursor"] < 0:
        raise ValueError("Invalid GRPO resume group cursor")
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if not expected or len(actual) != len(expected) or {id(p) for p in actual} != expected:
        raise ValueError("Optimizer must contain each trainable adapter parameter exactly once")
    all_groups = list(groups)
    expected_step = (state["group_cursor"] + group_batch_size - 1) // group_batch_size
    if (state["global_step"] != expected_step or
            (state["group_cursor"] != len(all_groups) and state["group_cursor"] % group_batch_size)):
        raise ValueError("Resume cursor is not an optimizer update boundary")
    if state["group_cursor"] > len(all_groups):
        raise ValueError("Resume cursor is beyond the frozen dataset")
    skipped = all_groups[:state["group_cursor"]]
    if state["consumed_branches"] != sum(g.group_size for g in skipped):
        raise ValueError("Resume branch cursor mismatch")
    if state["consumed_tokens"] != sum(int(g.loss_mask.sum()) for g in skipped):
        raise ValueError("Resume token cursor mismatch")
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    windows = iter(iter_group_batches(all_groups[state["group_cursor"]:], group_batch_size))
    while max_updates is None or state["global_step"] < max_updates:
        window = next(windows, None)
        if window is None:
            break
        if on_window_start:
            on_window_start()
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic(); forward_seconds = backward_seconds = 0.0
        loss_value = 0.0; group_metrics = []; learning_rate = optimizer.param_groups[0]["lr"]
        for group in window:
            if hasattr(objective_callback, "iter_results"):
                results = objective_callback.iter_results(model, group)
            else:
                results = (objective_callback(model, item) for item in (group,))
            group_rows = []
            while True:
                phase_started = time.monotonic()
                result = next(results, None)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                forward_seconds += time.monotonic() - phase_started
                if result is None:
                    break
                loss = result.loss
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("Nonfinite GRPO objective; optimizer update rejected")
                loss_value += float(loss.detach()) / len(window)
                started_backward = time.monotonic()
                (loss / len(window)).backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                backward_seconds += time.monotonic() - started_backward
                group_rows.append(result.metrics)
                del loss, result
            if not group_rows:
                raise ValueError("Objective returned no group contribution")
            group_metrics.append(group_rows)
        for parameter in actual:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite GRPO gradient; optimizer update rejected")
        if on_gradients:
            on_gradients()
        norm = torch.nn.utils.clip_grad_norm_(actual, grad_clip, error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite GRPO gradient norm")
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        branches = sum(group.group_size for group in window)
        tokens = sum(int(group.loss_mask.sum()) for group in window)
        state["global_step"] += 1; state["group_cursor"] += len(window)
        state["consumed_groups"] += len(window); state["consumed_branches"] += branches
        state["consumed_tokens"] += tokens
        from RL.grpo.metrics import summarize_update
        metrics = {**summarize_update(group_metrics), **state,
                   "loss": loss_value, "group_count": len(window), "branch_count": branches,
                   "token_count": tokens, "gradient_norm": float(norm),
                   "learning_rate": learning_rate, "next_learning_rate": optimizer.param_groups[0]["lr"],
                   "seconds": time.monotonic() - started, "forward_seconds": forward_seconds,
                   "backward_seconds": backward_seconds}
        if device.type == "cuda":
            torch.cuda.synchronize(); free, total = torch.cuda.mem_get_info()
            metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_bytes=free, total_bytes=total)
        if on_update:
            on_update(dict(state), metrics)
    return state
