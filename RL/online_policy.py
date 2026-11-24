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


# Historical import path retained for existing callers.
from RL.policy_reference import InitialPolicyScorer
