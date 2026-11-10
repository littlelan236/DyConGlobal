"""Loss-independent single-device optimizer loop with token-weighted accumulation."""

import itertools
import random
import time

import numpy as np
import torch

from RL.validation import validate_batch


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
          state=None, on_update=None, on_gradients=None):
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
        counts = [validate_batch(batch, vocab_size=vocab_size, max_length=max_length) for batch in window]
        total_tokens = sum(counts)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        total_loss = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch, expected_count in zip(window, counts, strict=True):
            loss_sum, count = loss_callback(model, {key: value.to(device) for key, value in batch.items()})
            if count != expected_count or loss_sum.ndim != 0 or not torch.isfinite(loss_sum):
                raise ValueError("Nonfinite loss or invalid loss callback supervision contract")
            total_loss += float(loss_sum.detach())
            (loss_sum / total_tokens).backward()
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
        if on_update:
            on_update(dict(state), metrics)
    return state
