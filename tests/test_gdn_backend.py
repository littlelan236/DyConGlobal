"""Explicit backend selection, lifecycle, and compatibility regression tests."""
import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from RL import gdn_backend as backend
from RL.artifacts import resume_contract_matches
from RL.config import Config, schedule_contract

ROOT = Path(__file__).parents[1]


@pytest.fixture
def clean_binding():
    backend.reset_for_tests()
    yield
    backend.reset_for_tests()


def config_values():
    return yaml.safe_load((ROOT / "RL/experiments/lora-smoke.yaml").read_text())


def test_default_reference_retains_historical_schedule():
    values = config_values()
    config = Config.model_validate(values)
    assert config.model.gdn_backend == "reference_pytorch"
    assert "gdn_backend" not in schedule_contract(config)["model"]
    config.model.gdn_backend = "fla_chunk"
    assert schedule_contract(config)["model"]["gdn_backend"] == "fla_chunk"


@pytest.mark.parametrize("name", ["auto", "fla", "fla_chunk_fp32_inputs", "", None])
def test_invalid_backend_rejected(name):
    values = config_values()
    values["model"]["gdn_backend"] = name
    with pytest.raises(ValueError):
        Config.model_validate(values)


@pytest.mark.parametrize("selected", ["fla_chunk"])
def test_fp32_fla_rejected_before_training(selected):
    values = config_values()
    values["model"].update(dtype="float32", gdn_backend=selected)
    with pytest.raises(ValueError, match="BF16"):
        Config.model_validate(values)


def test_config_and_backend_module_do_not_import_training_stack():
    source = (
        "import sys; import RL.config; import RL.gdn_backend; import RL.service; "
        "assert 'torch' not in sys.modules; "
        "assert not any(n == 'fla' or n.startswith('fla.') for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_real_state_gradients_and_observations(clean_binding):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    reference = inspect.unwrap(qwen.torch_chunk_gated_delta_rule)
    conv = inspect.unwrap(qwen.causal_conv1d_fn)
    binding = backend.install_gdn_backend("reference_pytorch")
    identity = copy.deepcopy(binding.execution_contract())
    assert qwen.causal_conv1d_fn is conv
    assert binding is backend.install_gdn_backend("reference_pytorch", identity)
    torch.manual_seed(913)
    original = [torch.randn(1, 2, 2, 16) for _ in range(3)]
    original += [-torch.rand(1, 2, 2), torch.rand(1, 2, 2),
                 torch.randn(1, 2, 16, 16)]
    cotangents = (torch.randn_like(original[2]), torch.randn_like(original[-1]))
    outputs = []
    for function in (reference, qwen.torch_chunk_gated_delta_rule):
        inputs = [value.clone().requires_grad_() for value in original]
        o, ht = function(*inputs[:5], initial_state=inputs[5],
                         output_final_state=True, use_qk_l2norm_in_kernel=True)
        gradients = torch.autograd.grad(
            (o * cotangents[0]).sum() + (ht * cotangents[1]).sum(), inputs)
        outputs.append((o.detach(), ht.detach(), *(g.detach() for g in gradients)))
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert bool(outputs[1][-1].abs().sum() > 0)
    assert binding.observations()["calls"]["chunk"] == 1
    assert binding.observations()["calls"]["chunk_with_initial_state"] == 1
    assert binding.execution_contract() == identity
    json.dumps(identity, allow_nan=False)


def test_backend_switch_rejected(clean_binding):
    backend.install_gdn_backend("reference_pytorch")
    with pytest.raises(backend.GDNBackendError, match="Cannot switch"):
        backend.install_gdn_backend(SimpleNamespace(gdn_backend="fla_chunk", dtype="bfloat16"))


def test_identity_failure_does_not_modify_module(clean_binding):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    originals = {name: getattr(qwen, name) for name in
                 ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")}
    with pytest.raises(backend.GDNBackendError, match="identity"):
        backend.install_gdn_backend("reference_pytorch", {"version": -1})
    assert backend.active_binding() is None
    assert all(getattr(qwen, name) is value for name, value in originals.items())


def test_reset_restores_original_symbols(clean_binding):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
             "causal_conv1d_fn", "causal_conv1d_update")
    originals = {name: getattr(qwen, name) for name in names}
    backend.install_gdn_backend("reference_pytorch")
    backend.reset_for_tests()
    assert all(getattr(qwen, name) is value for name, value in originals.items())


@pytest.mark.parametrize("selected", ["fla_chunk"])
def test_missing_fla_fails_without_fallback(clean_binding, monkeypatch, selected):
    real_import = backend.importlib.import_module
    def missing(name, *args, **kwargs):
        if name == "fla.ops.gated_delta_rule":
            raise ImportError("injected absent FLA")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(backend.GDNBackendError, match="cannot be imported"):
        backend.install_gdn_backend(SimpleNamespace(gdn_backend=selected, dtype="bfloat16"))
    assert backend.active_binding() is None


def test_source_change_rejected(clean_binding, monkeypatch):
    monkeypatch.setattr(backend, "QWEN_SOURCE_SHA256", "0" * 64)
    with pytest.raises(backend.GDNBackendError, match="source hash"):
        backend.install_gdn_backend("reference_pytorch")


def test_incompatible_fla_signature_rejected():
    with pytest.raises(backend.GDNBackendError, match="arguments"):
        backend._fla_adapter(lambda q, k, v: None)


def test_legacy_resume_only_ignores_unused_optional_packages():
    expected = {
        "attention": {"execution_backend": {
            "causal_conv1d": "reference_pytorch", "gated_delta_rule": "reference_pytorch"}},
        "dependencies": {"torch": "2.13.0", "flash-linear-attention": None},
        "schedule": {"seed": 42}}
    actual = copy.deepcopy(expected)
    actual["dependencies"]["flash-linear-attention"] = "0.6.0"
    actual["gdn_execution"] = {"resolved_backend": "reference_pytorch",
                               "conv_backend": "reference_pytorch"}
    before = copy.deepcopy(expected)
    assert resume_contract_matches(expected, actual)
    assert expected == before
    actual["dependencies"]["torch"] = "different"
    assert not resume_contract_matches(expected, actual)
    actual["dependencies"]["torch"] = "2.13.0"
    actual["gdn_execution"]["resolved_backend"] = "fla_chunk"
    assert not resume_contract_matches(expected, actual)
