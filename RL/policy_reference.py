"""Frozen initial-policy scoring, independent of rollout and training grouping."""
from contextlib import contextmanager
from pathlib import Path


def computation(config):
    objective = getattr(config, "objective", None)
    return objective.kl.computation if objective is not None else "legacy"


def dycon_scorer(config, model, directory):
    if not config.grpo.kl_beta or computation(config) != "recompute":
        return None
    return InitialPolicyScorer(model, Path(directory) / "grpo/reference/adapter")


class InitialPolicyScorer:
    """Temporarily install S0 only while no policy autograd graph is live."""

    def __init__(self, model, adapter, *, block_size=16):
        from peft.utils.save_and_load import load_peft_weights
        self.model = model
        self.block_size = block_size or 16
        self.reference = load_peft_weights(str(adapter), device="cpu")

    @contextmanager
    def frozen(self):
        import torch
        from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
        current = {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model).items()}
        training = self.model.training
        try:
            set_peft_model_state_dict(self.model, self.reference)
            self.model.eval()
            with torch.no_grad():
                yield
        finally:
            set_peft_model_state_dict(self.model, current)
            self.model.train(training)

    def __call__(self, tensors, *, block_size=None):
        from RL.online_runtime import online_selected_logprobs
        with self.frozen():
            values, _ = online_selected_logprobs(self.model, tensors, block_size=block_size or self.block_size)
            return values.detach()

    def score_group(self, group, *, block_size, reuse_prefix=False):
        """Finish all reference work before constructing the policy prefix graph."""
        from RL.grpo.logprobs import model_target_logprobs
        import torch
        result = {}
        with self.frozen():
            prefix = None
            try:
                if reuse_prefix:
                    from RL.grpo.prefix import PrefixContext
                    if any(p != group.prompt_token_ids[0] for p in group.prompt_token_ids):
                        raise ValueError("Reference prefix requires identical group prompts")
                    prefix = PrefixContext(self.model, group.prompt_token_ids[0])
                device = next(self.model.parameters()).device
                for i, branch in enumerate(group.branch_ids):
                    values = (prefix.branch_reference_logprobs(self.model, group.completion_token_ids[i], block_size)
                              if prefix is not None else
                              model_target_logprobs(self.model, group.branch_batch(i, device), block_size))
                    if not torch.isfinite(values).all():
                        raise ValueError("Nonfinite frozen KL reference logprobs")
                    result[branch] = values.detach().float().cpu()
            finally:
                if prefix is not None:
                    prefix.close()
        return result
