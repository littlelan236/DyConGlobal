"""Group-owned prefix graph, independent branch leaves, and one boundary VJP."""
from contextlib import contextmanager
import time
import sys

import torch

from RL.qwen_boundary import segment, validate_model
from .logprobs import hidden_target_logprobs


EXECUTION_CONTRACT = {"mode": "boundary_grad", "schema": "qwen35_boundary_v1",
    "implementation": "functional_layer_cache_v1", "gradient_accumulation_dtype": "float32",
    "checkpoint": "explicit_state_nonreentrant_layer", "transformers": "5.17.0"}


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class PrefixContext:
    def __init__(self, model, prompt, *, offloader=None):
        self.roots = self.leaves = ()
        self.gradients = []
        self.closed = False
        self.finished = False
        self.device = next(model.parameters()).device
        self.offloader = offloader
        self.metrics = {"prefix_length": len(prompt), "branch_count": 0,
            "prefix_forward_calls": 0, "prefix_backward_calls": 0,
            "branch_backward_calls": 0, "max_active_branches": 0,
            "source_bytes": 0, "branch_leaf_peak_bytes": 0,
            "boundary_gradient_bytes": 0, "boundary_gradient_dtype": "float32",
            "timing_method": "synchronized_wall_clock"}
        try:
            text = validate_model(model)
            self.keys = tuple(key for index, layer in enumerate(text.layers) for key in
                ((f"layer.{index}.key", f"layer.{index}.value") if layer.block_type == "full_attention" else
                 (f"layer.{index}.conv", f"layer.{index}.recurrent"))) + ("last_hidden",)
            with self.phase("prefix_forward"):
                hidden, states = segment(model, torch.tensor([prompt], device=self.device),
                                        metrics=self.metrics, offloader=self.offloader)
                self.roots = (*states, hidden[:, -1:].clone())
                if len(self.roots) != len(self.keys):
                    raise RuntimeError("Boundary schema/root cardinality mismatch")
                self.versions = tuple(t._version for t in self.roots)
                self.gradients = [None] * len(self.roots)
            self.metrics.update(prefix_forward_calls=1, source_bytes=tensor_bytes(self.roots),
                                source_dtypes=sorted({str(t.dtype) for t in self.roots}))
        except BaseException:
            self.close()
            raise

    @contextmanager
    def phase(self, name):
        def sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            if sys.exc_info()[0] is None:
                sync()
            key = name + "_seconds"
            self.metrics[key] = self.metrics.get(key, 0.) + time.perf_counter() - started
            if self.device.type == "cuda":
                for suffix, observe in (("allocated", torch.cuda.max_memory_allocated),
                                        ("reserved", torch.cuda.max_memory_reserved)):
                    memory_key = name + "_observed_peak_" + suffix + "_bytes"
                    self.metrics[memory_key] = max(self.metrics.get(memory_key, 0), observe(self.device))

    def branch_logprobs(self, model, completion, block_size):
        if self.closed or self.finished or self.leaves:
            raise RuntimeError("Invalid prefix/branch lifecycle")
        self.check_source()
        with self.phase("state_copy"):
            self.leaves = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in self.roots)
        self.metrics["max_active_branches"] = 1
        self.metrics["branch_leaf_peak_bytes"] = max(self.metrics["branch_leaf_peak_bytes"], tensor_bytes(self.leaves))
        with self.phase("suffix_forward"):
            tokens = torch.tensor([completion], device=self.device)
            if len(completion) > 1:
                hidden, _ = segment(model, tokens[:, :-1], self.leaves[:-1], self.metrics["prefix_length"],
                                    self.metrics, offloader=self.offloader)
                hidden = torch.cat((self.leaves[-1], hidden), dim=1)
            else:
                hidden = self.leaves[-1]
            # Existing projection expects labels one position to the right.
            labels = torch.cat((tokens.new_full((1, 1), -100), tokens), dim=1)
            return hidden_target_logprobs(model, hidden, labels, block_size)

    def after_branch_backward(self):
        if not self.leaves:
            raise RuntimeError("No active branch")
        with self.phase("boundary_accumulate"):
            for index, (root, leaf) in enumerate(zip(self.roots, self.leaves, strict=True)):
                grad = leaf.grad
                if grad is None:
                    continue
                if (grad.shape != root.shape or grad.device != root.device
                        or grad.dtype != root.dtype or not torch.isfinite(grad).all()):
                    raise ValueError(f"Nonfinite or inconsistent branch boundary gradient: {self.keys[index]}")
                value = grad.detach().float()
                if self.gradients[index] is None:
                    self.gradients[index] = value.clone()
                else:
                    self.gradients[index].add_(value)
                leaf.grad = None
            self.leaves = ()
        self.check_source()
        self.metrics["branch_count"] += 1
        self.metrics["branch_backward_calls"] += 1
        self.metrics["boundary_gradient_bytes"] = tensor_bytes(self.gradients)
        if self.device.type == "cuda":
            self.metrics["allocated_after_branch_bytes"] = torch.cuda.memory_allocated(self.device)

    def check_source(self):
        if tuple(t._version for t in self.roots) != self.versions:
            raise RuntimeError("Read-only prefix boundary was modified")

    def finish(self):
        if self.closed or self.finished or self.leaves:
            raise RuntimeError("Invalid prefix finish lifecycle")
        self.check_source()
        pairs = [(root, grad) for root, grad in zip(self.roots, self.gradients, strict=True) if grad is not None]
        if any(grad.grad_fn is not None or not torch.isfinite(grad).all() for _, grad in pairs):
            raise ValueError("Nonfinite accumulated boundary gradient")
        with self.phase("prefix_backward"):
            if pairs:
                torch.autograd.backward([r for r, _ in pairs], [g.to(r.dtype) for r, g in pairs])
        self.metrics["prefix_backward_calls"] = 1
        self.finished = True

    def close(self):
        for leaf in self.leaves:
            leaf.grad = None
        self.leaves = self.roots = ()
        self.gradients.clear()
        self.closed = True
