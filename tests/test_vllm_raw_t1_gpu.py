"""Opt-in pinned-runtime raw observer tests and real model acceptance."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

pytestmark = pytest.mark.skipif(os.environ.get("DYCON_ONLINE_RAW_GPU") != "1",
                              reason="explicit GPU acceptance opt-in required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_raw_observer_pre_grammar_oracle_rng_and_single_target(dtype):
    from vllm import SamplingParams
    from vllm.v1.worker.gpu.sample.raw_t1 import RawT1Observer, RAW_T1_KEY
    observer = RawT1Observer(4)
    observer.add_request(1, SamplingParams(logprobs=0, extra_args={RAW_T1_KEY: True}))
    logits = torch.tensor([[1., 4., 2., -3.], [4., 1., -1., 2.]], device="cuda", dtype=dtype)
    original = logits.clone()
    batch = SimpleNamespace(idx_mapping_np=np.array([1, 3]), num_draft_tokens=0)
    rng_before = torch.cuda.get_rng_state()
    state = observer.prepare(logits, batch)
    assert torch.equal(logits, original)
    # The highest raw probability is grammar-disallowed. Observer must retain
    # its finite mass in the full-vocabulary denominator.
    logits[0, 1] = -torch.inf
    logits[0].div_(.6)
    sampled = torch.tensor([0, 1], device="cuda")
    result = observer.finish(state, sampled)
    oracle = original.float().log_softmax(-1)[0, 0]
    torch.testing.assert_close(result.logprobs[0, 0], oracle, atol=2e-6, rtol=2e-6)
    assert result.logprobs.shape == (2, 1)
    assert torch.isfinite(original.float().log_softmax(-1)[0, 1])
    assert torch.equal(rng_before, torch.cuda.get_rng_state())
    observer.add_request(1, SamplingParams())
    assert observer.prepare(logits, batch) is None


def test_real_qwen_raw_capture_seed_grammar_and_three_repeat_generation(tmp_path, monkeypatch):
    if os.environ.get("DYCON_ONLINE_RAW_MODEL_TEST") != "1":
        pytest.skip("real model acceptance opt-in required")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.worker.gpu.sample.raw_t1 import RAW_T1_KEY
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("DYCON_RAW_T1_ENABLED", "1")
    monkeypatch.setenv("VLLM_PLUGINS", "")
    import shutil
    if shutil.which("ninja") is None:
        candidates = sorted(Path.home().glob(".cache/uv/archive-v0/*/ninja-*.data/scripts/ninja"))
        assert candidates, "Managed serving requires the pinned ninja build helper"
        monkeypatch.setenv("PATH", str(candidates[-1].parent) + os.pathsep + os.environ.get("PATH", ""))
    model = os.environ.get("DYCON_ONLINE_RAW_MODEL", "/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B")
    llm = LLM(model=model, dtype="float16", tensor_parallel_size=1, max_model_len=2048,
              max_num_seqs=8, max_num_batched_tokens=2048, gpu_memory_utilization=.55,
              enable_prefix_caching=True, compilation_config={"cudagraph_mode": "FULL"},
              trust_remote_code=False, seed=17)
    prompts = ["Explain why the sky is blue in one sentence.", "Count from one to ten.",
               "Write a simple Python function returning the square of a number.",
               "What is the capital of France?"]
    metrics = []
    evidence = Path(os.environ.get("DYCON_ONLINE_RAW_EVIDENCE", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    outputs = {}
    # Warm both paths, excluding compilation from the repeated generation test.
    for enabled in (False, True):
        extra = {RAW_T1_KEY: True} if enabled else None
        llm.generate(prompts[:1], SamplingParams(seed=17, max_tokens=8, logprobs=0 if enabled else None,
                                                extra_args=extra), use_tqdm=False)
    for repeat in range(3):
        for enabled in (False, True):
            params = SamplingParams(temperature=.6, top_p=.95, top_k=20, seed=123,
                max_tokens=96, ignore_eos=True, logprobs=0 if enabled else None,
                extra_args={RAW_T1_KEY: True} if enabled else None)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            tokens = [list(item.outputs[0].token_ids) for item in result]
            outputs[enabled] = tokens
            if enabled:
                assert outputs[False] == tokens
                for item in result:
                    completion = item.outputs[0]
                    assert len(completion.logprobs) == len(completion.token_ids)
                    assert all(set(entry) == {token} for token, entry in zip(completion.token_ids, completion.logprobs))
                    assert all(np.isfinite(entry[token].logprob) for token, entry in zip(completion.token_ids, completion.logprobs))
            metrics.append({"repeat": repeat, "raw_enabled": enabled, "seconds": duration,
                            "completion_tokens": sum(map(len, tokens)),
                            "tokens_per_second": sum(map(len, tokens)) / duration})
    samples = [{"prompt_token_ids": item.prompt_token_ids,
                "completion_token_ids": list(item.outputs[0].token_ids),
                "raw_logprobs": [float(entry[token].logprob) for token, entry in
                    zip(item.outputs[0].token_ids, item.outputs[0].logprobs)]} for item in result[:2]]
    (evidence / "online-samples.json").write_text(json.dumps({"model": model, "dtype": "float16", "samples": samples}))
    for enabled in (False, True):
        params = SamplingParams(temperature=.6, seed=12, max_tokens=16,
            structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
            logprobs=0 if enabled else None, extra_args={RAW_T1_KEY: True} if enabled else None)
        result = llm.generate(["Answer yes or no: Is water wet?"], params, use_tqdm=False)
        if not enabled:
            grammar_ids = list(result[0].outputs[0].token_ids)
        else:
            assert grammar_ids == list(result[0].outputs[0].token_ids)
            assert all(np.isfinite(next(iter(e.values())).logprob) for e in result[0].outputs[0].logprobs)
    (evidence / "generation-pilot.json").write_text(json.dumps({"scope": "generation only; rollout/writer/NAS pilot required separately",
        "model": model, "dtype": "float16", "full_cuda_graph": True,
        "seed_tokens_equal": True, "grammar_seed_tokens_equal": True, "metrics": metrics}, indent=2))
