# vLLM eager latent extraction

This is the only offline hidden-state backend. It uses the project's pinned EasySteer/vLLM 0.29.0 runtime in an independent `.venvs/latent-vllm` environment. It does not patch the serving environment, vendor sources, model weights or rollout code.

Production extraction uses `dycon hidden DATASET_ID CAPTURE_ID`, which delegates
to the isolated interpreter. Reuse this entry point; do not create temporary
extraction scripts or directly call workers. See [ENTRYPOINTS.md](ENTRYPOINTS.md)
for the historical queue launcher and online/offline workflow boundaries.

## Reused implementation

vLLM owns continuous batching, chunked prefill, Automatic Prefix Caching (APC), hybrid recurrent-state restoration and KV allocation/eviction. The adapter passes bounded windows of token-ID prompts to `LLM.generate`; this window is not a custom GPU batch scheduler. The original input order (step order within each rollout) is retained. A window completes and is drained before another is submitted; there is no custom asynchronous scheduler.

Capture uses existing `tact_states` RPCs, `SelectSpec(prompt_positions=[-1])`, request/position/token labels, the native residual composition, SDK `fetch_pages` and `CaptureResult`. The SDK public `capture()` helper currently excludes `tact_states`; only its RPC lifecycle is adapted here. No new model hooks, row-geometry producer, cache manager or tensor-parallel assembly are implemented.

APC leaves the final input token for local computation in this pinned release (`max_cache_hit_length = request.num_tokens - 1`). Native capture admission accepts this final-token selector without disabling cache reads. Cache-elided targets, dropped rows, incomplete sites, wrong positions/token IDs and nonfinite values are errors. One output token triggers prefill; the generated token is discarded and is never used as the latent anchor. Exact token-ID prompts are never truncated or re-templated.

## Sites

For an N-block model, public site indices 0..N-2 are complete decoder outputs and index N-1 is final norm. Native `tact_states` also exposes the final pre-norm block output, so public index N-1 maps to native index N. The final token of a marker is used if it consists of multiple token IDs.

Only eager is implemented: `enforce_eager=True`, asynchronous scheduling disabled, no speculative configuration, no torch.compile or CUDA Graph capture. Kernel JIT compilation is still part of ordinary eager execution. The pinned runtime does not support batch-invariant mode for Qwen GDN, so extraction explicitly disables that mode rather than bypassing its compatibility checks.

## Runtime setup

From the project root, prepare a new environment once:

```bash
.venvs/runner/bin/python -m hidden.scripts.prepare_vllm_runtime
```

This verifies the original serving runtime, then copies it using `cp -a --reflink=auto` (copy-on-write or independent copies, never hardlinks). It refuses an existing target. Use the copied environment's `bin/python`; copied console-script shebangs are not used. All pinned vLLM Python source hashes are checked before model loading. Capture SDK and project implementation hashes enter capture identity.

The environment has its own vLLM, Triton and FlashInfer caches. The launcher sets PATH to include its own tools and the existing CUDA toolkit. No package installation into `.venvs/easysteer` is performed.

## Configuration and execution

The only editable configuration is still `config.yaml`. `hidden.vllm` contains the required isolated replay settings. All settings can be placed under `hidden.vllm` or supplied with `--set`. Defaults:

| Field | Default |
|---|---|
| python | `.venvs/latent-vllm/bin/python` |
| gpus | `[]`, must be set explicitly |
| tensor_parallel_size | 1 |
| max_num_seqs | 16 |
| max_num_batched_tokens | 4096 |
| max_model_len | 32768 |
| max_inflight_requests | 32 |
| gpu_memory_utilization | 0.8 |
| enable_chunked_prefill | true |
| enable_prefix_caching | true |
| allow_shared_gpus | false; diagnostic-only override |
| min_free_gpu_bytes | 0; required when sharing a GPU |
| moe_backend | auto (`triton` explicitly selects native Triton MoE kernels) |
| capture_budget_bytes | 268435456 |
| fetch_bytes | 16777216 |

Use a new capture ID. The dataset's checkpoint fingerprint must match the configured checkpoint; configure `model.path`, `model.fingerprint`, `model.served_name` and `hidden.layers` as usual. Real Qwen validation passed in FP16; BF16 showed appreciably larger batch-dependent differences. Current remote hidden dtype is FP16. For example, after those model settings and a free GPU scope have been confirmed:

```bash
.venvs/runner/bin/python -m core.cli \
  --set hidden.adapter=vllm_eager \
  --set hidden.dtype=float16 \
  --set 'hidden.vllm.gpus=[4]' \
  --set hidden.vllm.tensor_parallel_size=1 \
  --set 'hidden.layers=[0,15,31]' \
  hidden DATASET_ID NEW_CAPTURE_ID
```

The CLI delegates to the independent interpreter. Explicit physical GPU indices must match TP size. Before importing torch/vLLM, the child checks existing compute processes and acquires extraction GPU locks. Occupied GPUs cause an error; no existing process is terminated. PP/DP/context/sequence/expert parallel capture restrictions are enforced by the native runtime; only ordinary TP is supported. GPU locks coordinate extraction instances; other software must still use separately allocated GPUs.

`max_num_batched_tokens` controls each engine scheduling iteration; `max_inflight_requests` bounds submitted samples and captured CPU data. Capture overflow fails clearly and does not retry silently. Reduce the submission window or increase its capture budget. APC hits are best-effort and require exact token prefixes; history reformatting can reduce reuse.

On SM120 GPUs, the automatic MoE choice may trigger a lengthy FlashInfer C++ compilation, including unused quantized variants. `hidden.vllm.moe_backend=triton` selects vLLM's native Triton MoE implementation explicitly, without replacing kernels or modifying the serving environment. This choice is stored in the capture identity.

## Products and recovery

`hidden.extract.extract_hidden` uses the vLLM batch API. It retains per-sample NPZ files, checksummed partial-result sidecars, final index/receipt, and bounded batch flushing. A failed window does not commit a final receipt; subsequent invocation with identical inputs/runtime/config reuses verified completed samples. Runtime/code/config changes require a new capture ID.

vLLM receipts additionally contain runtime/source fingerprints, inference dtype, engine options and window timing/cache metrics. Window metrics retain sample-to-native-request-ID mappings and target position/token labels. Index entries contain target token ID and absolute position. Both `enrichment/manifest.json` and `hidden/manifest.json` receive `native_captures` entries linking task ID, run ID, rollout ID, trajectory hash, explicit marker text and receipt. Existing migrated records and active extraction job records are preserved under the shared manifest lock. Capture output locations stay under `hidden/outputs/`; diagnostic results stay under `hidden/evaluation/outputs/`.

## Validation tools

CPU contract tests: `tests/test_vllm_hidden.py`. Native SDK CPU tests: `tests/test_vllm_native_capture.py` (run with the isolated interpreter). These cover layer mapping, request labels, failures, GPU admission, batched resume, manifest preservation and old config/snapshot compatibility.

`python -m hidden.evaluation.smoke_vllm --output NEW_DIRECTORY` accepts `--config`/`--set`. It uses 129-, 2049- and 4097-token mixed-anchor prompts, checks cold/hot APC, reverse input order and singleton requests, verifies real cache hits, and checks cosine >= 0.999 and relative L2 <= 0.05. Failures remain failures and are retained in diagnostic reports. The default GPU admission rejects existing compute processes; a shared-GPU diagnostic must explicitly set `allow_shared_gpus=true`, a positive `min_free_gpu_bytes`, and a bounded `kv_cache_memory_bytes`.

Small GPU smoke tests are diagnostic evidence only; they do not establish whole-corpus speedup or bitwise equality across models.

### 2026-10-19 validation

The replay comparison used four events from the same request in the complete online
capture batch `capture-verified200-20261018-143317`. The source tensors came from
the NAS publication and the prefixes came from the matching request ledger. With
Qwen3.5-9B FP16, vLLM 0.29.0 eager, TP=1 and GPU 0 shared under an explicit
12 GB free-memory floor, all cold/warm/reordered comparisons passed:

| Samples | Worst cosine | Worst relative L2 | Report |
|---:|---:|---:|---|
| 4 | 0.9999958551 | 0.0028796251 | `hidden/evaluation/outputs/replay-online-smoke-20261019-4/report.json` |

The APC smoke used the same checkpoint and shared-GPU admission. The first mixed
window took 0.945 s; the repeated hot window took 0.229 s and reported 5,280 of
6,275 prompt tokens cached (84.1%). Its report is
`hidden/evaluation/outputs/smoke-online-cache-20261019/report.json`.
