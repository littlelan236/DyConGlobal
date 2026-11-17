"""Physical branch microbatches with group-normalized objective contributions."""
from .logprobs import model_target_logprobs
from .objective import grpo_objective


class GroupObjective:
    def __init__(self, reference_values, settings, block_size, offloader=None):
        self.reference_values = reference_values
        self.settings = settings
        self.block_size = block_size
        self.offloader = offloader
        self.prefix = None
        self.group_metadata = {}
        self.active_reference = None
        self.reset()

    def reset(self):
        self.tokens = self.blocks = 0
        self.sequence_lengths = []
        self.prefix_metrics = []

    def drain_prefix_metrics(self):
        rows, self.prefix_metrics = self.prefix_metrics, []
        return rows

    def begin_group(self, model, group):
        if self.prefix is not None:
            raise RuntimeError("Previous prefix group was not closed")
        self.group_metadata = {"parent_group_id": group.group_id,
            "branch_ids": list(group.branch_ids), "prefix_length": len(group.prompt_token_ids[0]),
            **group.metadata.get("execution_chunk", {})}
        if "physical_task" in group.metadata:
            self.group_metadata["physical_task"] = dict(group.metadata["physical_task"])
        self.active_reference = (self.reference_values.for_group(group)
                                 if hasattr(self.reference_values, "for_group") else None)
        mode = getattr(getattr(self.settings, "prefix_reuse", None), "mode", "disabled")
        if mode == "boundary_grad":
            from .prefix import PrefixContext
            if any(p != group.prompt_token_ids[0] for p in group.prompt_token_ids):
                raise ValueError("GRPO group prompt tokens are not identical")
            self.prefix = PrefixContext(model, group.prompt_token_ids[0], offloader=self.offloader)

    def after_branch_backward(self):
        if self.prefix is not None:
            self.prefix.after_branch_backward()

    def record_branch_backward(self, seconds):
        if self.prefix is not None:
            metrics = self.prefix.metrics
            metrics["suffix_backward_seconds"] = metrics.get("suffix_backward_seconds", 0.) + seconds

    def finish_group(self):
        if self.prefix is not None:
            self.prefix.finish()
            self.prefix_metrics.append({**self.prefix.metrics, **self.group_metadata,
                "timing_method": "synchronized_segments"})

    def close_group(self):
        if self.prefix is not None:
            self.prefix.close()
            if self.prefix.device.type == "cuda" and self.prefix_metrics:
                import torch
                self.prefix_metrics[-1]["allocated_after_group_bytes"] = torch.cuda.memory_allocated(self.prefix.device)
            self.prefix = None
        self.active_reference = None
        self.group_metadata = {}

    def iter_results(self, model, group):
        if (getattr(getattr(self.settings, "prefix_reuse", None), "mode", "disabled") == "boundary_grad"
                and self.prefix is None):
            raise RuntimeError("boundary_grad requires explicit begin_group/finish_group/close_group")
        device = next(model.parameters()).device
        for index, branch_id in enumerate(group.branch_ids):
            yield self._branch_result(model, group, index, branch_id, device)

    def _branch_result(self, model, group, index, branch_id, device):
        if self.prefix is not None:
            current = self.prefix.branch_logprobs(model, group.completion_token_ids[index], self.block_size)
        else:
            current = model_target_logprobs(model, group.branch_batch(index, device), self.block_size)
        reference = (self.active_reference if self.active_reference is not None
                     else self.reference_values[group.group_id])[branch_id].to(device)
        if current.shape != reference.shape:
            raise ValueError("Reference cache target shape mismatch")
        import torch
        diagnostics = None
        diagnostic_settings = getattr(self.settings, "token_diagnostics", None)
        if diagnostic_settings is not None and diagnostic_settings.enabled:
            from .token_diagnostics import branch_diagnostics
            diagnostics = branch_diagnostics(current, reference, group, index, diagnostic_settings)
            if any(v for k, v in diagnostics["counts"].items()
                   if k.startswith(("current_", "reference_"))
                   and k.endswith(("_nan_tokens", "_posinf_tokens", "_neginf_tokens"))):
                import json
                raise ValueError("Nonfinite GRPO target diagnostics: " + json.dumps(diagnostics, allow_nan=False))
        result = grpo_objective(current[None], reference[None], [group.advantages[index]],
            torch.ones_like(current[None], dtype=torch.bool),
            clip_epsilon=self.settings.clip_epsilon, kl_beta=self.settings.kl_beta,
            ratio_guard_enabled=self.settings.ratio_guard.enabled,
            ratio_guard_threshold=self.settings.ratio_guard.threshold)
        if diagnostics is not None:
            from .token_diagnostics import compare_branch_scores
            compare_branch_scores(model, group, index, current, reference, diagnostics,
                                  diagnostic_settings, self.block_size)
            result.metrics["_token_diagnostics"] = diagnostics
        result.metrics["branch_id"] = branch_id
        result.metrics["native_branch_ordinal"] = group.metadata.get("physical_task", group.metadata.get("execution_chunk", {})).get(
            "native_branch_ordinals", list(range(group.group_size)))[index]
        result.loss = result.loss / group.group_size
        result.group_count = 0
        self.tokens += len(current)
        self.blocks += __import__("math").ceil(len(current) / (self.block_size or len(current)))
        self.sequence_lengths.append(len(group.prompt_token_ids[index]) + len(group.completion_token_ids[index]))
        return result
