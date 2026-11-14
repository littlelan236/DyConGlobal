"""Repeated startup audits can be omitted without silently re-enabling them."""
from pathlib import Path

import pytest
from test_rl_dataset import build, token_source

from core.io import read_json, write_json
from RL.artifacts import base_identity
from RL.config import Config, load, schedule_contract
from RL.grpo.data import prepare, tensor_groups


def config_for(dataset):
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    config.algorithm.dataset_id = dataset.name
    config.algorithm.dataset_root = str(dataset.parents[2])
    config.training.max_updates = 1
    return config


def test_audits_default_off_and_are_strict_and_bound_to_contract():
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    assert not any(config.preflight.model_dump().values())
    assert schedule_contract(config)["preflight"] == config.preflight.model_dump()
    for field in ("verify_identity", "audit_native_tokens"):
        raw = config.model_dump()
        raw["preflight"][field] = "false"
        with pytest.raises(ValueError):
            Config.model_validate(raw)


def test_native_audit_off_skips_ledger_and_collation_hashes_but_on_rejects(tmp_path, monkeypatch):
    fixture = token_source(tmp_path)
    dataset = build(fixture)
    path = fixture[0]["nas"] / "requests/1.json"
    value = read_json(path)
    value["request_id"] = "previously-unverified-request"
    write_json(path, value)
    config = config_for(dataset)
    raw, proof = prepare(config)
    assert proof["preflight"] == config.preflight.model_dump()
    assert all(row["audit_enabled"] is False for row in raw[0]["members"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Token hash recomputed during unchecked collation")
    with monkeypatch.context() as m:
        m.setattr("RL.dataset.schema.hash_json", forbidden)
        assert list(tensor_groups(raw))[0].group_size == 6
    config.preflight.audit_native_tokens = True
    with pytest.raises(ValueError, match="checksum"):
        prepare(config)


def test_identity_audit_off_skips_dataset_inventory_but_on_rejects(tmp_path):
    dataset = build(token_source(tmp_path))
    (dataset / "extra-not-in-receipt.txt").write_text("unregistered")
    config = config_for(dataset)
    _, proof = prepare(config)
    assert proof["group_count"] == 1
    config.preflight.verify_identity = True
    with pytest.raises(ValueError, match="inventory/hash"):
        prepare(config)


def test_disabled_audits_keep_same_prompt_and_length_checks(tmp_path):
    fixture = token_source(tmp_path)
    dataset = build(fixture)
    path = fixture[0]["nas"] / "requests/1.json"
    value = read_json(path)
    value["prompt_token_ids"][-1] += 1
    write_json(path, value)
    with pytest.raises(ValueError, match="prompt"):
        prepare(config_for(dataset))


def model_fixture(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", dict(model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"], text_config=dict(
            model_type="qwen3_5_text", hidden_size=4096, num_hidden_layers=32, vocab_size=248320)))
    write_json(model / "tokenizer.json", {})
    write_json(model / "tokenizer_config.json", {})
    (model / "model.safetensors").write_bytes(b"original")
    return model


def test_model_identity_off_reuses_registration_without_content_hashing(tmp_path, monkeypatch):
    model = model_fixture(tmp_path)
    original = base_identity(model, tmp_path, True)
    (model / "model.safetensors").write_bytes(b"modified")
    def forbidden(*args, **kwargs):
        raise AssertionError("Deep model hash recomputed")
    with monkeypatch.context() as m:
        m.setattr("core.model_identity.describe", forbidden)
        m.setattr("core.model_identity.sha256", forbidden)
        assert base_identity(model, tmp_path, False)["fingerprint"] == original["fingerprint"]
    assert base_identity(model, tmp_path, True)["fingerprint"] != original["fingerprint"]


def test_unchecked_model_requires_registered_unambiguous_identity(tmp_path):
    model = model_fixture(tmp_path)
    with pytest.raises(ValueError, match="registered model identity"):
        base_identity(model, tmp_path, False)


def test_service_passes_default_and_enabled_identity_switch(tmp_path, monkeypatch):
    from RL.service import _preflight
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/lora-smoke.yaml", root)
    calls = []
    monkeypatch.setattr("RL.service.memory_plan", lambda c: {})
    monkeypatch.setattr("RL.service.base_identity", lambda *args: calls.append(args[2]) or {"fingerprint": "test"})
    _preflight(config, tmp_path, None)
    config.preflight.verify_identity = True
    _preflight(config, tmp_path, None)
    assert calls == [False, True]
