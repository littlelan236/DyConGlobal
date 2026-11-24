"""Small BF16/FLA mixed-gradient acceptance; no production run or dataset loading."""
import os
from types import SimpleNamespace as NS

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("DYCON_KL_REFERENCE_GPU") != "1",
                               reason="CUDA frozen KL acceptance is opt-in")


@pytest.mark.parametrize("shared_host", [False, True])
def test_fla_dycon_prefix_and_baseline_trace_share_frozen_scorer(tmp_path, shared_host):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from tests.lora_helpers import raw_model, LORA
    from tests.test_grpo_prefix import group
    from RL.gdn_backend import install_gdn_backend, reset_for_tests
    from RL.modeling import inject
    from RL.config import Grpo, Training
    from RL.policy_reference import InitialPolicyScorer
    from RL.grpo.backend import GroupObjective
    from RL.online_runtime import (OnlineRow, OnlineRolloutBatch, SourceMass, TorchOnlineUpdater)
    from RL.grpo.logprobs import model_target_logprobs

    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, True)
    cfg = raw_model().config
    cfg.text_config.hidden_size = 256
    cfg.text_config.intermediate_size = 384
    cfg.text_config.head_dim = 128
    cfg.text_config.linear_key_head_dim = 128
    cfg.text_config.linear_value_head_dim = 128
    cfg._attn_implementation = "sdpa"
    reset_for_tests()
    binding = install_gdn_backend(NS(gdn_backend="fla_chunk", dtype="bfloat16"))
    try:
        model, _ = inject(Qwen3_5ForConditionalGeneration(cfg).to(device="cuda", dtype=torch.bfloat16), LORA)
        model.train()
        offloader = None
        if shared_host:
            from RL.config import Memory
            from RL.memory import configure_checkpointing
            memory = Memory(activation_offload="checkpoint_boundary", host_budget_gib=None,
                shared_trace_host_budget_gib=8., offload_host_reserve_gib=1.,
                host_offload={"enabled": True, "allocator_budget_gib": 8., "reclaim_policy": "after_use"})
            offloader = configure_checkpointing(model,
                NS(training=NS(memory=memory, checkpoint_policy="layer"),
                   run=NS(attempt_id="test-mixed-shared-" + tmp_path.name)), None)
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "lora_" in name:
                    value.uniform_(-.05, .05)
        model.save_pretrained(tmp_path / "initial")
        scorer = InitialPolicyScorer(model, tmp_path / "initial")
        row = group(65, 3)
        shared_reference = scorer.score_group(row, block_size=16, reuse_prefix=True)
        full_reference = scorer.score_group(row, block_size=16, reuse_prefix=False)
        for key in shared_reference:
            torch.testing.assert_close(shared_reference[key], full_reference[key], atol=.02, rtol=.002)
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "lora_B" in name:
                    value.add_(.01)
        denominator = {row.group_id: full_reference}
        objective = GroupObjective(denominator, Grpo(kl_beta=.02, k3_cap=1.,
            prefix_reuse={"mode": "boundary_grad"}), 16, reference_scorer=scorer, offloader=offloader)
        objective.begin_group(model, row)
        try:
            for result in objective.iter_results(model, row):
                assert torch.isfinite(result.loss)
                (.5 * result.loss).backward()
                objective.after_branch_backward()
            objective.finish_group()
        finally:
            objective.close_group()
        dycon_gradients = [p.grad.clone() if p.grad is not None else None for p in model.parameters() if p.requires_grad]
        settings = Training(max_updates=1, gradient_accumulation_steps=1, learning_rate=5e-5,
            weight_decay=0., grad_clip=1., scheduler="constant", checkpoint_every_updates=1)
        # Two adjacent completions form one baseline trace; history stays in the graph.
        baseline = (
            OnlineRow("b0", "task", "baseline", "task", (2,3,4), (4,), (-4.,), None,
                      1., "trajectory", 3, 0, True),
            OnlineRow("b1", "task", "baseline", "task", (2,3,4,5,6), (5,6), (-4.,-4.), None,
                      1., "trajectory", 3, 1, True))
        updater = TorchOnlineUpdater(model, settings, source_mass=SourceMass(0.,1.),
            kl_beta=.02, k3_cap=1., reference_scorer=scorer, kl_reference_mode="recompute",
            baseline_training_mode="prefix_trace", max_trace_input_tokens=None,
            max_trace_completion_tokens=4096, activation_offloader=offloader)
        _, metrics, _ = updater.accumulate(
            OnlineRolloutBatch(0, ("task",), "S0", "E0", baseline),
            [1/6, 1/3], SourceMass(0.,1.), 0, attribution=None)
        assert metrics["original_row_count"] == 2 and metrics["row_count"] == 1
        combined = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in combined)
        assert any(not torch.equal(g, old) for g, old in zip(combined, dycon_gradients) if old is not None)
        assert binding.observations()["calls"].get("chunk_with_initial_state", 0) > 0
        assert model.training
        if offloader is not None:
            assert offloader.trace_admission is not None
            assert offloader.host_reserve_bytes == 0
            assert offloader.failed_trace is None and offloader.live_bytes == 0
            assert offloader.metrics()["offload_d2h_bytes"] > 0
            offloader.close()
    finally:
        reset_for_tests()
