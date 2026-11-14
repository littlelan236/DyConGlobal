"""Opt-in GDN raw FULL tests against FP16 full-vocabulary logits."""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif("DYCON_REPLAY_TEST_GPU" not in os.environ, reason="Explicit idle GPU required")


def test_gdn_raw_full_mixed_prefill_replay(tmp_path):
    import sys
    from hidden.backends.vllm.engine import configure_environment, gpu_reservation, verify_runtime

    root = Path(__file__).resolve().parents[1]
    gpu = int(os.environ["DYCON_REPLAY_TEST_GPU"])
    configure_environment({"project_root": str(root), "serving": {"python": ".venvs/easysteer/bin/python"},
                           "hidden": {"vllm": {"python": sys.executable, "gpus": [gpu]}}})
    verify_runtime(root)
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, PreTrainedTokenizerFast
    from RL.dataset.replay.validation import target_logprobs
    from RL.dataset.replay.worker import sampling_options
    from vllm import LLM, SamplingParams

    torch.manual_seed(1731)
    with gpu_reservation(root, [gpu]):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
            vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=128, linear_value_head_dim=128,
            layer_types=["linear_attention", "full_attention"], max_position_embeddings=1024,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )).half().eval()
        checkpoint = tmp_path / "tiny-gdn-fp16"
        model.save_pretrained(checkpoint)
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", bos_token="t1",
                               eos_token="t2", pad_token="t0").save_pretrained(checkpoint)
        prompts = [[1] + [5, 6, 7, 8] * 67, [1] + [9, 10, 11] * 49]
        completions = [[2, 7, 2, 63, 4, 0, 2], [3, 8], [63, 0, 2, 9]]
        rows = [(prompts[0], completions[0]), (prompts[0], completions[1]),
                (prompts[1], completions[2]), (prompts[0], completions[2])]
        oracle = []
        model = model.cuda()
        with torch.inference_mode():
            for prompt, completion in rows:
                logits = model(torch.tensor([prompt + completion], device="cuda"), use_cache=False).logits[0, len(prompt)-1:-1].float()
                oracle.append(logits.log_softmax(-1).gather(1, torch.tensor(completion, device="cuda")[:, None]).flatten().cpu().numpy())
        del model
        torch.cuda.empty_cache()
        results = []
        for prefix, mode in ((False, "NONE"), (True, "NONE"), (True, "FULL")):
            engine = LLM(model=str(checkpoint), dtype="half", generation_config="vllm", language_model_only=True,
                         enable_trace_replay=True, logprobs_mode="raw_logprobs", async_scheduling=False,
                         enforce_eager=mode == "NONE", compilation_config={"cudagraph_mode": mode,
                             "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128]},
                         enable_prefix_caching=prefix, enable_chunked_prefill=True, mamba_cache_mode="align",
                         max_model_len=512, max_num_seqs=4, max_num_batched_tokens=128,
                         kv_cache_memory_bytes=64*1024**2, attention_config={"backend": "TRITON_ATTN"},
                         gdn_prefill_backend="triton", additional_config={"raw_gdn_fullgraph": True},
                         worker_extension_cls="RL.dataset.replay.observation.ReplayObservation", seed=1731)
            try:
                engine.collective_rpc("replay_observe_start")
                params = [SamplingParams(**sampling_options({"completion_token_ids": c, "completion_count": len(c)})) for _, c in rows]
                # Publish one prefix, then mix a cache hit and a chunked prefill.
                first = engine.generate([{"prompt_token_ids": rows[0][0]}], [params[0]], use_tqdm=False)[0]
                actual = [target_logprobs(first, rows[0][1])]
                llm = engine.llm_engine
                pending = {}
                for i in range(1, len(rows)):
                    request_id = f"gdn-{i}"
                    llm.add_request(request_id, {"prompt_token_ids": rows[i][0]}, params[i])
                    pending[request_id] = i
                by_id = {}
                while llm.has_unfinished_requests():
                    for output in llm.step():
                        if output.finished:
                            i = pending[output.request_id]
                            by_id[i] = target_logprobs(output, rows[i][1])
                actual.extend(by_id[i] for i in range(1, len(rows)))
                for value, expected in zip(actual, oracle):
                    np.testing.assert_allclose(value, expected, atol=.005, rtol=0)
                results.append(actual)
                observation = engine.collective_rpc("replay_observe_snapshot")[0]
                assert observation["effective_mode"] == mode
                if mode == "FULL":
                    assert observation["actual_full_dispatch"].get("FULL", 0) > 0
                    assert len(observation["actual_full_shape_counts"]) >= 2
                llm.add_request("abort-gdn", {"prompt_token_ids": prompts[1]}, params[2])
                llm.step()
                llm.abort_request(["abort-gdn"])
                engine.reset_prefix_cache()
                after = engine.generate([{"prompt_token_ids": prompts[1]}], [params[2]], use_tqdm=False)[0]
                np.testing.assert_allclose(target_logprobs(after, completions[2]), oracle[2], atol=.005, rtol=0)
            finally:
                shutdown = getattr(engine.llm_engine, "shutdown", None) or engine.llm_engine.engine_core.shutdown
                shutdown()
        for actual in results[1:]:
            for value, eager in zip(actual, results[0]):
                np.testing.assert_allclose(value, eager, atol=.005, rtol=0)
