"""Immutable online policy publication and initial-policy KL evaluation."""
from pathlib import Path
import tempfile

from RL.checkpoints import export_adapter, verify_adapter, restore_checkpoint


def publish_adapter(model, destination, contract):
    destination = Path(destination)
    if destination.exists():
        verify_adapter(destination, contract, verify_owner=False)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".online-export-", dir=destination.parent) as scratch:
        stage = Path(scratch) / "adapter"
        export_adapter(model, stage, contract)
        stage.rename(destination)
    return destination


def restore_policy_checkpoint(model, path, optimizer, scheduler, contract, adapter_contract, *, verify=True):
    """Restore policy tensors as well as optimizer/RNG at replay boundaries."""
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    from RL.checkpoints import verify_checkpoint
    if verify:
        verify_checkpoint(path)
    verify_adapter(Path(path) / "adapter", adapter_contract, verify_owner=False)
    state = restore_checkpoint(path, optimizer, scheduler, contract, adapter_contract, verify=verify, online=True)
    set_peft_model_state_dict(model, load_peft_weights(str(Path(path) / "adapter"), device="cpu"))
    return state


class InitialPolicyScorer:
    """Score S0 on exact inputs without another base-model allocation.

    Each call occurs before the row's autograd forward. Adapter tensors are
    restored in finally; no training graph remains alive across a swap.
    """
    def __init__(self, model, adapter):
        from peft.utils.save_and_load import load_peft_weights
        self.model = model
        self.reference = load_peft_weights(str(adapter), device="cpu")

    def __call__(self, tensors):
        import torch
        from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
        from RL.online_runtime import online_selected_logprobs
        current = {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model).items()}
        training = self.model.training
        try:
            set_peft_model_state_dict(self.model, self.reference)
            self.model.eval()
            with torch.no_grad():
                values, _ = online_selected_logprobs(self.model, tensors)
                return values.detach()
        finally:
            set_peft_model_state_dict(self.model, current)
            self.model.train(training)
