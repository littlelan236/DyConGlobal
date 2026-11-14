"""Opt-in raw FP16 oracle, EOS and cache/graph integration in pinned vLLM."""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif("DYCON_REPLAY_TEST_GPU" not in os.environ, reason="Explicit idle GPU required")


def test_fp16_raw_trace_matches_full_vocabulary_oracle_with_cache_and_graph(tmp_path):
    import sys
    from hidden.backends.vllm.engine import configure_environment, gpu_reservation, verify_runtime
    root = Path(__file__).resolve().parents[1]; gpu = int(os.environ["DYCON_REPLAY_TEST_GPU"])
    configure_environment({"project_root": str(root), "serving": {"python": ".venvs/easysteer/bin/python"},
                           "hidden": {"vllm": {"python": sys.executable, "gpus": [gpu]}}})
    verify_runtime(root)
    with gpu_reservation(root, [gpu]):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from vllm import LLM, SamplingParams
        from RL.dataset.replay.validation import target_logprobs
        from RL.dataset.replay.worker import sampling_options
        torch.manual_seed(1729)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
                                  num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                  max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)).half()
        checkpoint = tmp_path / "tiny-raw-fp16"; model.save_pretrained(checkpoint)
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", bos_token="t1", eos_token="t2", pad_token="t0").save_pretrained(checkpoint)
        prompt = [1] + [5, 6, 7, 8] * 8  # non-block boundary and multiple prefill chunks
        completions = [[2, 7, 2], [63, 4], [0, 2, 9, 5]]
        model = model.cuda().eval(); oracle = []
        with torch.no_grad():
            for completion in completions:
                tokens = torch.tensor([prompt + completion], device="cuda")
                logits = model(tokens).logits[0, len(prompt)-1:-1].float()
                oracle.append(torch.log_softmax(logits, -1).gather(1, torch.tensor(completion, device="cuda")[:, None]).flatten().cpu().tolist())
        del model; torch.cuda.empty_cache()
        results = []
        for prefix, mode in ((False, "NONE"), (True, "FULL")):
            engine = LLM(model=str(checkpoint), dtype="half", generation_config="vllm", language_model_only=True,
                         enable_trace_replay=True, logprobs_mode="raw_logprobs", async_scheduling=False,
                         enforce_eager=mode == "NONE", compilation_config={"cudagraph_mode": mode, "cudagraph_capture_sizes": [1, 2, 4, 8]},
                         enable_prefix_caching=prefix, enable_chunked_prefill=True,
                         max_model_len=128, max_num_seqs=4, max_num_batched_tokens=32,
                         kv_cache_memory_bytes=8*1024**2,
                         attention_config={"backend": "TRITON_ATTN"},
                         worker_extension_cls="RL.dataset.replay.observation.ReplayObservation", seed=1729)
            try:
                engine.collective_rpc("replay_observe_start")
                params = [SamplingParams(**sampling_options({"completion_token_ids": c, "completion_count": len(c)})) for c in completions]
                values = []
                # Sequential first request publishes real prefix state for subsequent requests.
                for completion, param in zip(completions, params):
                    output = engine.generate([{"prompt_token_ids": prompt}], [param], use_tqdm=False)[0]
                    values.append(target_logprobs(output, completion))
                for actual, expected in zip(values, oracle): np.testing.assert_allclose(actual, expected, atol=.002, rtol=0)
                results.append(values)
                observation = engine.collective_rpc("replay_observe_snapshot")
                assert observation[0]["effective_mode"] == mode
                if mode == "FULL": assert observation[0]["actual_full_dispatch"].get("FULL", 0) > 0
                # Abort after partial prefill; new IDs must still bind to exact targets.
                llm = engine.llm_engine
                llm.add_request("abort-raw", {"prompt_token_ids": prompt}, params[2]); llm.step(); llm.abort_request(["abort-raw"])
            finally:
                shutdown = getattr(engine.llm_engine, "shutdown", None) or engine.llm_engine.engine_core.shutdown
                shutdown()
        for left, right in zip(*results): np.testing.assert_allclose(left, right, atol=.002, rtol=0)
