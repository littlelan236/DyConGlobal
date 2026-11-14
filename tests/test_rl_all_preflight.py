"""Default startup prepares artifacts once and runs no optional audit passes."""
from pathlib import Path

import numpy as np
import pytest

from core.io import read_json, write_json
from RL.config import Config, load
from RL.service import _preflight, _select_gpu

ROOT = Path(__file__).parents[1]


def config():
    return load(ROOT / "RL/experiments/grpo-offline-k3-smoke.yaml", ROOT)[0]


def forbidden(*args, **kwargs):
    raise AssertionError("Disabled precheck was executed")


def test_all_preflight_switches_are_strict_and_default_off():
    c = config()
    assert set(c.preflight.model_dump()) == {
        "verify_identity", "audit_native_tokens", "validate_dataset", "verify_probabilities",
        "verify_checkpoint", "check_resources", "verify_model_state", "run_diagnostics"}
    assert not any(c.preflight.model_dump().values())
    for name in c.preflight.model_dump():
        raw = c.model_dump()
        raw["preflight"][name] = "false"
        with pytest.raises(ValueError):
            Config.model_validate(raw)


def test_default_preflight_does_not_scan_dataset_or_check_resources(monkeypatch):
    c = config()
    monkeypatch.setattr("RL.service.base_identity", lambda *args: {"fingerprint": "declared"})
    monkeypatch.setattr("RL.grpo.data.prepare", forbidden)
    monkeypatch.setattr("RL.service.memory_plan", forbidden)
    assert _preflight(c, ROOT, None) == {"fingerprint": "declared"}


def test_explicit_dataset_precheck_is_available_without_torch(monkeypatch):
    c = config()
    c.preflight.validate_dataset = True
    monkeypatch.setattr("RL.service.base_identity", lambda *args: {"fingerprint": "declared"})
    assert _preflight(c, ROOT, None)["grpo_dataset"]["group_count"] == 5


def test_default_gpu_selection_does_not_run_admission_query(monkeypatch):
    monkeypatch.setattr("RL.service.subprocess.check_output", forbidden)
    assert _select_gpu(config())["resource_precheck"] == "disabled"


def test_resource_checks_can_be_enabled_explicitly(tmp_path, monkeypatch):
    from RL.resources import memory_plan
    c = config()
    c.model.path = str(tmp_path)
    write_json(tmp_path / "config.json", {"text_config": {
        "hidden_size": 32, "num_hidden_layers": 2, "vocab_size": 64}})
    c.training.memory.activation_offload = "checkpoint_boundary"
    c.training.memory.host_budget_gib = .000001
    assert memory_plan(c)["capacity_verified"] is False
    c.preflight.check_resources = True
    with pytest.raises(ValueError, match="exceeds per-rank host budget"):
        memory_plan(c)
    monkeypatch.setattr("RL.service.subprocess.check_output", lambda *args, **kwargs: "0, unit, GPU, 600, 10000")
    with pytest.raises(RuntimeError, match="not idle"):
        _select_gpu(c)


def test_source_checkpoint_audit_can_be_disabled_and_enabled(tmp_path):
    from RL.artifacts import commit_receipt
    from RL.service import _source
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_json(adapter / "identity.json", {"contract": {}})
    commit_receipt(tmp_path, "lora-attempt-v1")
    (tmp_path / "unregistered").write_text("changes inventory")
    c = config()
    c.run.start.mode = "warm_start"
    c.run.start.checkpoint = str(adapter)
    assert _source(c)["adapter"] == str(adapter)
    c.preflight.verify_checkpoint = True
    with pytest.raises(ValueError, match="inventory/hash"):
        _source(c)


def test_prepare_advantages_does_not_collate_unused_batches(monkeypatch):
    from RL.grpo.data import prepare
    monkeypatch.setattr("RL.dataset.loader.collate_group", forbidden)
    raw, proof = prepare(config())
    assert len(raw) == proof["group_count"] == 5
    assert all(g["frozen_advantages"] for g in raw)


def test_training_prepares_groups_once_and_keeps_required_artifacts(tmp_path, monkeypatch):
    import test_grpo_lifecycle as lifecycle

    import RL.grpo.data as data
    from RL.artifacts import verify_receipt
    real_load = lifecycle.load
    real_train = lifecycle._train_grpo
    real_prepare = data.prepare
    calls = []

    def unchecked_load(*args):
        c, source = real_load(*args)
        c.preflight = type(c.preflight)()
        return c, source

    def counted_prepare(*args):
        calls.append(1)
        return real_prepare(*args)

    def unchecked_train(c, *args, **kwargs):
        c.preflight = type(c.preflight)()
        return real_train(c, *args, **kwargs)

    monkeypatch.setattr(lifecycle, "load", unchecked_load)
    monkeypatch.setattr(lifecycle, "_train_grpo", unchecked_train)
    monkeypatch.setattr(data, "prepare", counted_prepare)
    monkeypatch.setattr("RL.modeling.frozen_checksums", forbidden)
    monkeypatch.setattr("RL.grpo.reference.diagnose_initial_current", forbidden)
    _, directory, _, result = lifecycle.run_tiny(tmp_path, "unchecked-required-artifacts")
    assert result["status"] == "completed" and len(calls) == 1
    for name in ["input-identity.json", "selection.json", "frozen-groups.json", "grpo-contract.json",
                 "grpo/reference.json", "grpo/reference-logprobs.pt", "adapter/identity.json", "metrics.jsonl"]:
        assert (directory / name).is_file()
    assert read_json(directory / "validation.json")["frozen_base_unchanged"] is None
    verify_receipt(directory, "lora-attempt-v1")


def test_unchecked_external_import_skips_audits_but_writes_verifiable_cache(tmp_path, monkeypatch):
    from test_grpo_external_reference import BASE, configuration, import_cache, make_source, select

    from RL.checkpoints import adapter_contract
    from RL.config import schedule_contract
    from RL.grpo.contract import validate_reference_files
    from tests.lora_helpers import model_and_targets
    c = configuration(tmp_path)
    c.preflight = type(c.preflight)()
    make_source(tmp_path, c)
    select(c, tmp_path)
    monkeypatch.setattr("RL.dataset.artifacts.verify_receipt", forbidden)
    monkeypatch.setattr("RL.dataset.probability.ProbabilityStream.validate_dataset_rows", forbidden)
    monkeypatch.setattr("RL.grpo.reference.adapter_compatibility", forbidden)
    monkeypatch.setattr("RL.grpo.reference.validate_scoring_identity", forbidden)
    values, ref, directory, groups, proof = import_cache(tmp_path, c)
    assert len(values) == len(groups)
    assert ref["probability_binding"]["training_initial_adapter"]["kind"] == "unchecked"
    model, targets = model_and_targets()
    validate_reference_files(directory, {"grpo": ref, "input": proof,
        "adapter": adapter_contract(BASE, targets, c.lora), "schedule": schedule_contract(c)})


def test_probability_value_hashes_are_optional_and_numerical_checks_remain(tmp_path):
    from RL.dataset.materialize import ShardWriter, read_array
    writer = ShardWriter(tmp_path, "<f4", 1024)
    ref = writer.put([-1., -2.])
    writer.close()
    (tmp_path / ref["shard"]).write_bytes(np.asarray([-3., -4.], dtype="<f4").tobytes())
    assert read_array(tmp_path, ref, verify=False).tolist() == [-3., -4.]
    with pytest.raises(ValueError, match="checksum"):
        read_array(tmp_path, ref, verify=True)
