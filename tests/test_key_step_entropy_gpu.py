"""Opt-in real pinned sampler acceptance; no benchmark data are collected."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DYCON_KEY_STEP_GPU") != "1", reason="Explicit GPU acceptance required"
)


def test_real_qwen_eight_way_entropy_preserves_seeded_tokens(tmp_path):
    os.environ["DYCON_RAW_T1_ENABLED"] = "1"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # isolated local worker test callbacks
    # Native sampler warmup compiles a FlashInfer helper before the observer.
    ninja = sorted(Path.home().glob(".cache/uv/archive-v0/*/ninja-*.data/scripts/ninja"))
    if ninja:
        os.environ["PATH"] = str(ninja[-1].parent) + os.pathsep + os.environ["PATH"]
    pytest.importorskip("torch")
    from vllm import LLM, SamplingParams

    from core.io import write_json

    llm = LLM(
        model="/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B",
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.35,
        enable_prefix_caching=False,
        enforce_eager=True,
        worker_extension_cls="runtime.serving.capture_worker.CaptureWorkerExtension",
        seed=17,
    )
    prompts = [
        "Write one sentence about " + x
        for x in ["trees", "water", "space", "math", "music", "rain", "code", "light"]
    ]
    raw = {"dycon_raw_t1_sampled_token_v1": True}
    params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=123,
        max_tokens=16,
        ignore_eos=True,
        logprobs=0,
        extra_args=raw,
    )
    before = llm.generate(prompts, params, use_tqdm=False)
    assert llm.collective_rpc("configure_generation_entropy") == [True]
    after = llm.generate(prompts, params, use_tqdm=False)
    assert [list(x.outputs[0].token_ids) for x in before] == [list(x.outputs[0].token_ids) for x in after]

    # LLM returns public IDs; worker records the engine's randomized native IDs.
    def native_ids(worker):
        observer = worker._dycon_entropy
        return sorted(
            set(observer.values) | {n for names, *_ in observer.pending for n in names if n is not None}
        )

    def exact_native_id(public):
        matches = [x for x in llm.collective_rpc(native_ids)[0] if x.startswith(public + "-")]
        assert len(matches) == 1
        return matches[0]

    warmup = llm.generate(
        ["warmup"], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False
    )[0]
    warmup_stat = llm.collective_rpc(
        "generation_entropy", args=(exact_native_id(warmup.request_id), list(warmup.outputs[0].token_ids))
    )[0]
    assert warmup_stat["entropy_count"] == 1 and warmup_stat["entropy_mean"] == 0
    for a, b in zip(before, after, strict=True):
        # Native floating point kernels can differ slightly between batches.
        pa = [
            entry[token].logprob
            for token, entry in zip(a.outputs[0].token_ids, a.outputs[0].logprobs, strict=True)
        ]
        pb = [
            entry[token].logprob
            for token, entry in zip(b.outputs[0].token_ids, b.outputs[0].logprobs, strict=True)
        ]
        assert pa == pytest.approx(pb, abs=5e-3, rel=1e-3)
    stats = []
    for output in after:
        ids = list(output.outputs[0].token_ids)
        row = llm.collective_rpc("generation_entropy", args=(exact_native_id(output.request_id), ids))[0]
        assert row["entropy_count"] == len(ids) == 16
        assert 0 <= row["entropy_mean"] <= 16
        stats.append(row)
    from vllm.sampling_params import StructuredOutputsParams

    grammar = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=123,
        max_tokens=8,
        logprobs=0,
        extra_args=raw,
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
    )
    result = llm.generate(["Answer yes or no: Is water wet?"], grammar, use_tqdm=False)
    grammar_stat = llm.collective_rpc(
        "generation_entropy",
        args=(exact_native_id(result[0].request_id), list(result[0].outputs[0].token_ids)),
    )[0]
    assert grammar_stat["entropy_count"] == len(result[0].outputs[0].token_ids)
    evidence = Path(os.environ.get("DYCON_KEY_STEP_EVIDENCE", str(tmp_path)))
    write_json(
        evidence / "entropy-gpu.json",
        {
            "seed_tokens_equal": True,
            "branches": stats,
            "grammar": grammar_stat,
            "greedy_warmup": warmup_stat,
            "raw_t1_aligned_and_close": True,
        },
    )
