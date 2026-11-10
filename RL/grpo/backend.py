"""Physical branch microbatches with group-normalized objective contributions."""
from .logprobs import model_target_logprobs
from .objective import grpo_objective


class GroupObjective:
    def __init__(self, reference_values, settings, block_size):
        self.reference_values = reference_values
        self.settings = settings
        self.block_size = block_size
        self.reset()

    def reset(self):
        self.tokens = self.blocks = 0
        self.sequence_lengths = []

    def iter_results(self, model, group):
        device = next(model.parameters()).device
        for index, branch_id in enumerate(group.branch_ids):
            batch = group.branch_batch(index, device)
            current = model_target_logprobs(model, batch, self.block_size)
            reference = self.reference_values[group.group_id][branch_id].to(device)
            if current.shape != reference.shape:
                raise ValueError("Reference cache target shape mismatch")
            import torch
            result = grpo_objective(current[None], reference[None], [group.advantages[index]],
                torch.ones_like(current[None], dtype=torch.bool),
                clip_epsilon=self.settings.clip_epsilon, kl_beta=self.settings.kl_beta)
            result.loss = result.loss / group.group_size
            result.group_count = 0
            self.tokens += len(current)
            self.blocks += __import__("math").ceil(len(current) / (self.block_size or len(current)))
            self.sequence_lengths.append(batch["input_ids"].shape[1])
            yield result
