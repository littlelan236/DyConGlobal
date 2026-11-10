import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from RL.grpo.artifacts import contract_sha256, verify_objective_contract, write_objective_contract
from RL.grpo.reference import ReferenceIdentity, ReferenceLogprobCache, cache_key


def test_objective_contract_and_reference_cache_are_identity_bound(tmp_path):
    contract = {"objective_version": "offline_grpo_k3_v1", "dataset_id": "d"}
    path = tmp_path / "contract.json"
    write_objective_contract(path, contract)
    assert verify_objective_contract(path, contract)["sha256"] == contract_sha256(contract)
    identity = ReferenceIdentity("m", "t", "a")
    key = cache_key(identity, dataset_receipt_sha256="d" * 64, group_id="g", branch_id="b",
                    mask_policy="native_completion_reasoning_action_v1")
    cache = ReferenceLogprobCache(tmp_path / "cache", identity)
    cache.put(key, torch.tensor([[1.0]]), torch.tensor([[True]]))
    values, mask = cache.get(key)
    assert values.item() == 1 and mask.item()
    with pytest.raises(ValueError):
        ReferenceLogprobCache(tmp_path / "cache", ReferenceIdentity("other", "t", "a")).get(key)
