"""Opt-in BF16 FLA forward/backward checks, without reference comparisons."""
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DYCON_FLA_GPU_TEST") != "1",
    reason="BF16 FLA GPU checks require explicit opt-in",
)


@pytest.mark.parametrize("length", [1, 65])
@pytest.mark.parametrize("path", ["both", "output_only", "state_only", "no_final_state"])
def test_bf16_forward_and_state_gradients(length, path):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from RL.gdn_backend import install_gdn_backend, reset_for_tests

    reset_for_tests()
    torch.manual_seed(1701)
    torch.use_deterministic_algorithms(True)
    try:
        binding = install_gdn_backend(SimpleNamespace(gdn_backend="fla_chunk", dtype="bfloat16"))
        shape = (1, length, 2, 128)
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
                   for _ in range(3)]
        g = (-torch.rand(shape[:3], device="cuda", dtype=torch.float32)).requires_grad_()
        beta = torch.rand(shape[:3], device="cuda", dtype=torch.bfloat16).requires_grad_()
        h0 = (torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.float32) * .01).requires_grad_()
        leaves = (q, k, v, g, beta, h0)
        snapshots = [(value._version, value.detach().clone()) for value in leaves]
        output, state = qwen.torch_chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=h0,
            output_final_state=path != "no_final_state", use_qk_l2norm_in_kernel=True,
        )
        assert output.dtype == torch.bfloat16
        assert output.shape == v.shape
        assert bool(torch.isfinite(output).all())
        loss = None if path == "state_only" else (output.float() * torch.randn_like(output).float()).sum()
        if path in {"both", "state_only"}:
            assert state is not None and state.shape == h0.shape
            assert state.dtype == torch.float32 and bool(torch.isfinite(state).all())
            assert state.untyped_storage().data_ptr() != h0.untyped_storage().data_ptr()
            state_loss = (state * torch.randn_like(state)).sum()
            loss = state_loss if loss is None else loss + state_loss
        if path == "no_final_state":
            assert state is None
        gradients = torch.autograd.grad(loss, leaves)
        torch.cuda.synchronize()
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        assert bool(gradients[-1].abs().sum() > 0), "initial-state gradient must reach the prefix"
        assert all(bool(value.abs().sum() > 0) for value in gradients[1:3])
        if path != "state_only":
            assert bool(gradients[0].abs().sum() > 0)
        for value, (version, snapshot) in zip(leaves, snapshots, strict=True):
            assert value._version == version and torch.equal(value, snapshot)
        assert binding.resolved_backend == "fla_chunk"
        assert binding.observations()["calls"]["chunk_with_initial_state"] == 1
    finally:
        reset_for_tests()
