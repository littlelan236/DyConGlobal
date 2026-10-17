# Eager vLLM validation — 2026-10-17

## Completed checks

- DyConGlobal suite: **54 passed, 4 skipped** in the runner environment (optional dependencies/GPU SDK unavailable there).
- Isolated vLLM adapter + native SDK checks: **24 passed**.
- HF hidden/enrichment regression checks in the analysis environment: **8 passed**.
- Original serving environment's TACT and capture SDK CPU tests: **108 passed**.
- Ruff checks on changed source files: passed.
- The serving environment's `verify_easysteer_runtime.py`: passed in `.venvs/easysteer`.
- The same verifier correctly rejected `.venvs/latent-vllm`, preventing accidental use of the serving environment.
- Legacy `resolve(config.yaml)` output remained exactly equal to the pre-extension implementation when `hidden.vllm` is absent.
- Real Qwen3.5-9B engine on GPU4 loaded vLLM 0.29.0 in eager mode with chunked prefill and APC. Logs show 3 capture hooks and native GDN prefill. The initial run reached inference and captured the requested sites.

## Observed real engine behavior

With 129, 2049 and 4097-token inputs (layers 0, 15 and 31), one mixed window took about **0.73 s cold**. A repeated hot window reported **5,280 cached tokens out of 6,275 prompt tokens** and about **0.14 s**. This confirms that the adapter is using native APC rather than a custom cache. Eager capture forwards and request labels were present.

The first run used BF16 and passed all runtime integrity checks but failed the provisional batch/cache numerical consistency threshold: some mixed/cache paths had cosine about 0.992 and relative L2 about 0.128. A subsequent no-cache run showed exact equality for the shortest sample but up to cosine 0.986 and relative L2 0.166 on the longest sample after batch/singleton comparison. These differences are retained as evidence that backend/batch numerical behavior must be audited for the intended estimator; thresholds must not be loosened to declare equivalence.

FP16 follow-up runs passed the unchanged smoke thresholds (cosine >= 0.999, relative L2 <= 0.05):

| Model / sites | TP | Worst cosine | Worst relative L2 | Result |
|---|---:|---:|---:|---|
| Qwen3.5-9B, all 32 HF-compatible sites | 1 | 0.999291 | 0.037843 | pass |
| Qwen3.5-27B, sites 0/31/63 | 1 | 0.999637 | 0.026962 | pass |
| Qwen3.6-27B, sites 0/31/63 | 1 | 0.999338 | 0.036390 | pass |
| Qwen3.6-35B-A3B, sites 0/19/39, native Triton MoE | 2 | 0.997515 | 0.071310 | numerical threshold failed; capture/cache/labels operational |

On Qwen3.5-9B FP16, the three-site mixed window took approximately 0.57 s cold and 0.14 s hot, with 5,280/6,275 prompt tokens reported cached. Qwen3.5-27B and Qwen3.6-27B each reported 5,488 cached tokens on the repeated window, with cold/hot times approximately 1.78/0.31 s and 1.80/0.32 s. These are small diagnostic measurements, including capture and CPU export, not HF speedup benchmarks.

An actual six-sample, mixed-marker subset from a migrated training rollout was captured through the original `core.cli hidden` dispatch into the isolated environment. It wrote NPZ/index/receipt and both manifest entries successfully. The original sample IDs, task ID, run ID and trajectory checksum were retained. Capture ID `vllm-eager-validation-9b` is diagnostic; it does not replace any original product. A completed all-32-site diagnostic capture uses ID `vllm-eager-validation-9b-final` and includes sample-to-native-request-ID mappings, target positions/token IDs, model fingerprint and inference dtype.

HF comparison at identical tokens/checkpoint/dtype on Qwen3.5-9B sites 0/15/31 had worst cosine 0.999601 and relative L2 0.028273. Across all 32 sites, the worst cosine was 0.998276 and relative L2 0.058840. These are observed differences, not a claim that the two backends meet the same numerical threshold. The all-site HF audit is saved as `hidden/evaluation/outputs/vllm-eager-9b-fp16-all-sites-final/hf-comparison.json`.

The MoE TP=2 path successfully exported labelled replicated-owner states using the native TP assembly, and the hot window reported 4,224 cached tokens out of 6,275 with approximately 0.17 s versus 1.50 s cold. Its numerical smoke failed without changing tolerances. The initial automatic FlashInfer MoE selection spent several minutes compiling unused quantized variants; only that diagnostic process group was stopped. The completed MoE diagnostic explicitly used `hidden.vllm.moe_backend=triton`, max_num_batched_tokens=2048 and GPUs 4/6. No serving or baseline process was stopped.

The pinned vLLM runtime rejects `VLLM_BATCH_INVARIANT=1` for Qwen GDN. The extraction launcher therefore forces it off and records ordinary eager execution. Deterministic batch-invariant mode is not currently available for these Qwen3.5 models. This is a known limitation, not a silent fallback.

## Interpretation and gate

The eager code path, GPU isolation, native batch scheduling, chunked prefill, APC hit reporting, target-token protection and manifest binding are implemented and tested. The three tested dense FP16 models pass the smoke numerical policy; the MoE FP16 model and BF16 diagnostic do not. Existing default HF extraction and its global BF16 default remain unchanged. Select FP16 explicitly for the validated dense vLLM path and use separate capture IDs; do not silently merge BF16/HF/vLLM products as if they were numerically identical. The pinned GDN implementation does not provide batch-invariant execution. Model/site-specific validation remains necessary beyond the tested lengths/sites, and MoE is not numerically qualified under the current policy.

All diagnostic logs and reports are under `state/vllm-eager-*` and `hidden/evaluation/outputs/`; they are ignored by the repository and do not affect baseline rollout or steering processes.
