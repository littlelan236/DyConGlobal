import copy
import pytest
from RL.offline_baseline_train import extension_restore_contract


def contracts():
    old = {"schedule": {"epochs": 3, "max_updates": 168, "tasks": 111},
           "input": {"config": {"epochs": 3}, "manifest_sha256": "dataset"},
           "objective": {"kind": "offline_baseline_prefix_trace_v1", "settings": {"epochs": 3}},
           "training": {"max_updates": 168, "scheduler": "constant", "learning_rate": 5e-5}}
    new = copy.deepcopy(old)
    new["schedule"].update(epochs=5, max_updates=280)
    new["input"]["config"]["epochs"] = 5
    new["objective"]["settings"]["epochs"] = 5
    new["training"]["max_updates"] = 280
    return old, new


def test_extension_returns_original_restore_contract_without_mutating_inputs():
    old, new = contracts()
    snapshot = copy.deepcopy(new)
    assert extension_restore_contract(old, new, {"global_step": 168}) == old
    assert new == snapshot


@pytest.mark.parametrize("change", ["lr", "dataset", "linear", "shrink", "past_end"])
def test_extension_rejects_other_changes(change):
    old, new = contracts()
    state = {"global_step": 168}
    if change == "lr":
        new["training"]["learning_rate"] = 1e-4
    elif change == "dataset":
        new["input"]["manifest_sha256"] = "changed"
    elif change == "linear":
        old["training"]["scheduler"] = new["training"]["scheduler"] = "linear"
    elif change == "shrink":
        new["schedule"]["epochs"] = 2
    else:
        state["global_step"] = 169
    with pytest.raises(ValueError):
        extension_restore_contract(old, new, state)
