"""Real-model bf16 diagnostics and numerical acceptance, never a training launcher."""

import os
from pathlib import Path

import pytest

from core.io import write_json

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("DYCON_LORA_LONG_GPU_TEST") != "1",
                                reason="Real bf16 numerical audit is opt-in")


def test_real_bf16_mlp_projection_diagnostics():
    from RL.config import load
    from RL.service import _select_gpu
    config, _ = load(ROOT / "RL/experiments/lora-smoke.yaml", ROOT)
    _select_gpu(config)
    import torch

    from RL.config import Memory
    from RL.memory import LossBackend, PositionChunkedMLP, attention_context, text_backbone
    from RL.modeling import inject, load_base
    from RL.training import seed_everything
    from RL.validation import causal_ce, fixture_batches
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(config.run.seed)
    model, tokenizer = load_base(config)
    model, _ = inject(model, config.lora)
    model.eval()
    batch = {k: v.cuda() for k, v in fixture_batches(tokenizer, 2048)[0][0].items()}
    layers = text_backbone(model).layers
    originals = [layer.mlp for layer in layers]
    results = []
    torch.backends.cuda.preferred_blas_library("cublaslt")
    for reduced in [True, False]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (reduced, reduced)
        for chunk in [0, 256, 512, 1024, 2048]:
            for layer, mlp in zip(layers, originals, strict=True):
                layer.mlp = PositionChunkedMLP(mlp, chunk) if chunk else mlp
            with torch.no_grad(), attention_context("flash_only"):
                loss, count = causal_ce(model, batch)
                value = float(loss) / count
            record = dict(reduced_precision_reduction=reduced, chunk=chunk, loss=value)
            results.append(record)
            print("BF16 MLP", record, flush=True)
    for layer, mlp in zip(layers, originals, strict=True):
        layer.mlp = mlp
    with torch.no_grad(), attention_context("flash_only"):
        loss, count = LossBackend(Memory(loss_chunk_tokens=512))(model, batch)
        print("BF16 CE", float(loss) / count, flush=True)
    work = ROOT / "temp/lora-long-context-20261110"
    write_json(work / "bf16-mlp-diagnostics.json", dict(results=results))
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in results)
    precise = [row["loss"] for row in results if not row["reduced_precision_reduction"]]
    assert max(precise) - min(precise) < 1e-6
    # Exercise actual training backward with every adapter, using the production
    # fp32 accumulation policy. Gradients are normalized by the same token count.
    from RL.modeling import enable_checkpointing, frozen_checksums
    enable_checkpointing(model)
    model.train()
    frozen = frozen_checksums(model)
    with attention_context("flash_only"):
        dense_loss, count = causal_ce(model, batch)
        (dense_loss / count).backward()
    expected = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    errors = []
    mode_results = {}
    offload_metrics = None
    for mode in ["ce", "mlp", "combined", "offload"]:
        model.zero_grad(set_to_none=True)
        for layer, mlp in zip(layers, originals, strict=True):
            layer.mlp = PositionChunkedMLP(mlp, 512) if mode != "ce" else mlp
        if mode == "offload":
            from RL.memory import BoundaryOffload
            offloader = BoundaryOffload(model, 64 * 2**30)
            for layer in layers:
                layer._gradient_checkpointing_func = offloader.checkpoint
        with attention_context("flash_only"):
            chunk_loss, other_count = LossBackend(Memory(loss_chunk_tokens=512 if mode != "mlp" else 0,
                                                         mlp_chunk_tokens=512 if mode != "ce" else 0))(model, batch)
            (chunk_loss / other_count).backward()
        assert count == other_count
        torch.testing.assert_close(chunk_loss, dense_loss, atol=0.01, rtol=1e-6)
        rows = []
        max_difference = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad:
                gradient = p.grad.detach().cpu()
                difference = float((gradient - expected[name]).abs().max())
                max_difference = max(max_difference, difference)
                rows.append(dict(name=name, max_abs_difference=difference,
                                 reference_norm=float(expected[name].norm()), actual_norm=float(gradient.norm())))
                try:
                    torch.testing.assert_close(gradient, expected[name], atol=2e-5, rtol=2e-3)
                except AssertionError as error:
                    errors.append(f"{mode} {name}: {str(error)[:350]}")
            else:
                assert p.grad is None
        mode_results[mode] = dict(loss=float(chunk_loss.detach()) / count, max_abs_difference=max_difference, gradients=rows)
        print("BF16 gradient mode", mode, "max abs difference", max_difference, "first", rows[1], flush=True)
        write_json(work / "bf16-gradient-diagnostics.json", mode_results)
        if mode == "offload":
            torch.cuda.synchronize()
            offload_metrics = offloader.metrics()
            assert offload_metrics["offload_d2h_bytes"] == offload_metrics["offload_h2d_bytes"] > 0
            assert offload_metrics["offload_live_bytes"] == 0
            assert all(p.device.type == "cuda" for p in model.parameters())
    assert frozen_checksums(model) == frozen
    assert not errors, "\n".join(errors[:8])
    write_json(work / "bf16-equivalence.json", dict(status="passed", length=2048,
               dense_loss=float(dense_loss.detach()) / count, chunk_loss=float(chunk_loss.detach()) / count,
               adapter_gradient_max_abs_difference=max_difference, gradient_atol=2e-5, gradient_rtol=2e-3,
               frozen_base_unchanged=True, offload=offload_metrics))
    print("BF16 all-adapter gradients passed; max difference", max_difference, flush=True)
