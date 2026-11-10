"""Dataset-wide GRPO preflight without a split artifact."""
from pathlib import Path

import pytest
import yaml

from core.io import sha256
from RL.config import Config, load, schedule_contract
from RL.grpo.data import prepare
from RL.grpo.contract import validate_input
from test_rl_dataset import build, token_source


def smoke_config():
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    return config


def test_config_and_contract_bind_dataset_without_partition():
    config = smoke_config()
    assert "split_id" not in config.algorithm.model_dump()
    assert "split_id" not in schedule_contract(config)["algorithm"]
    raw = config.model_dump()
    raw["algorithm"]["split_id"] = "obsolete"
    with pytest.raises(ValueError, match="Extra inputs"):
        Config.model_validate(raw)


def test_synthetic_preflight_covers_every_eligible_group_and_freezes_order():
    config = smoke_config()
    groups, proof = prepare(config)
    assert len(groups) == proof["total_groups"] == proof["group_count"] == 5
    assert proof["group_coverage"] == 1
    assert proof["eligible_group_ids"] == [g["group"]["group_id"] for g in groups]
    assert not any("split" in key for key in proof)
    assert prepare(config)[1] == proof
    config.grpo.shuffle_seed += 1
    changed = prepare(config)[1]
    assert proof["eligible_groups_sha256"] == changed["eligible_groups_sha256"]
    assert proof["group_order_sha256"] != changed["group_order_sha256"]


def test_native_dataset_preflight_does_not_require_partition_manifest(tmp_path):
    dataset = build(token_source(tmp_path))
    config = smoke_config()
    config.algorithm.dataset_id = "unit-dataset"
    config.algorithm.dataset_root = str(tmp_path / "out")
    config.training.max_updates = 1
    assert not (tmp_path / "out/RL/splits").exists()
    groups, proof = prepare(config)
    assert len(groups) == proof["total_groups"] == proof["group_count"] == 1
    assert len(groups[0]["members"]) == proof["branch_count"] == 6
    assert proof["dataset_receipt_sha256"] == sha256(dataset / "receipt.json")
    assert proof["eligible_group_ids"] == [groups[0]["group"]["group_id"]]
    assert not any("split" in key for key in proof)


@pytest.mark.parametrize("model", [None, "unverified-path", {}, {"fingerprint": "other"}])
def test_native_dataset_requires_verified_model_identity(model):
    with pytest.raises(ValueError, match="fingerprint"):
        validate_input({"fingerprint": "expected"}, {"synthetic": False,
            "dataset_model": model, "tokenizer_fingerprint": "native-tokenizer"})


def test_native_dataset_accepts_matching_model_and_requires_tokenizer():
    base = {"fingerprint": "expected"}
    proof = {"synthetic": False, "dataset_model": base, "tokenizer_fingerprint": "native-tokenizer"}
    validate_input(base, proof)
    proof["tokenizer_fingerprint"] = None
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        validate_input(base, proof)
