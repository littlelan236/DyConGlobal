# Long-context memory backends

完整训练 YAML、CLI 和恢复操作见[训练接口使用指南](TRAINING.md)。

The supported entry remains `.venvs/rl/bin/dycon lora-train --training-config YAML`.
This implements single-GPU synthetic engineering validation, with complete causal
attention, all-linear LoRA coverage and update-boundary recovery. Accelerate/DDP is
deferred; FSDP, ZeRO, context parallel and higher-order gradients are unsupported.

The additive `training.memory` settings default to the original dense path:

| Field | Default | Contract |
|---|---|---|
| `loss_chunk_tokens` | `0` | Positive enables frozen-head CE before vocabulary projection. Only shifted supervised positions are projected; blocks are recomputed in backward. |
| `mlp_chunk_tokens` | `0` | Positive enables token-position MLP recomputation, preserving every PEFT Parameter object and export path. Dropout must remain zero. |
| `exact_length` | `false` | Trim a single right-padded sample to its true length. Left padding, holes, supervised padding and multiple samples are rejected. |
| `attention_policy` | `auto` | `flash_only` restricts full attention to Flash SDPA in both forward and backward. A dense four-dimensional mask or unavailable kernel fails. |
| `activation_offload` | `none` | `checkpoint_boundary` moves only rank-three checkpoint input activations through pinned CPU memory. Weights stay on GPU. |
| `host_budget_gib` | `64.0` | Hard per-rank pinned activation budget. Preflight checks available RAM and all local ranks, retaining a host reserve. |
| `profile_first_update` | `false` | Collect actual full-attention operators, CUDA kernel names, masks and observed transfer/compute overlap. The first update includes profiler overhead. |

`training.checkpoint_policy` selects `layer` (production default), `selective_mlp`
(only the chunked MLP recomputes), or `layer_group` (registered Qwen3.5 text backend
with non-reentrant groups). `training.checkpoint_group_layers` defaults to four.
The group backend retains the original model tree, position/mask helpers and PEFT
paths; it supports text only and rejects KV caches. It is not an attention patch.
Any Transformers/PEFT upgrade, compilation or parameter sharding needs new tests.

## Configuration examples

The following is a fragment of the standalone training YAML; retain all required
run/model/lora/runtime/training/validation fields shown in [TRAINING.md](TRAINING.md).
It illustrates CE/MLP chunking with one right-padded sample per microbatch:

```yaml
training:
  checkpoint_policy: layer
  checkpoint_group_layers: 4
  memory:
    loss_chunk_tokens: 512
    mlp_chunk_tokens: 512
    exact_length: true
    attention_policy: flash_only
    activation_offload: none
    host_budget_gib: 64.0
    profile_first_update: false
  supervision:
    mode: uniform
    max_targets_per_sample: 32
```

`loss_chunk_tokens` bounds vocabulary projection positions per block; sparse
supervision determines which positions reach those blocks. `mlp_chunk_tokens`
chunks every input position inside MLPs and does not sparsify attention. Their
block sizes are independent of the linear-attention reference kernel's internal
64-token chunks. Sparse selection never shortens valid backbone history.

| Desired policy | YAML change | Required combination |
|---|---|---|
| Ordinary layer checkpointing | `checkpoint_policy: layer` | Default; may combine with CE/MLP chunks |
| MLP-only recomputation | `checkpoint_policy: selective_mlp` | `mlp_chunk_tokens > 0`; attention graphs remain resident |
| Group consecutive layers | `checkpoint_policy: layer_group` | Positive `checkpoint_group_layers`; text only, no KV cache |
| Offload checkpoint boundaries | `memory.activation_offload: checkpoint_boundary` | `layer` or `layer_group`, explicit host budget |

Throughput and memory presets default to dry-run:

```bash
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-long-throughput.yaml
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-long-memory.yaml
```

Copy a preset to a task YAML and edit that file to choose lengths/policies. Execute
an actual smoke using `.venvs/rl/bin/dycon` only after setting a new attempt ID and
`run.mode: smoke`. There are no CLI memory-policy overrides. Changing memory or
checkpoint policy invalidates exact resume; use an explicit adapter warm-start for
a new strategy. `exact_length` only trims right padding, never valid input after
the last selected target. CPU offload moves boundary activations, not model weights.

## Numerical policy and capacity

The supported fixture configuration limit is 262144, with no claim of that capacity.
Lengths above 2048 require chunked CE, exact-length microbatch one and Flash full
attention. Linear attention is separately identified: the pinned environment uses
Transformers' reference PyTorch gated-delta and causal-convolution paths. That
fallback is explicitly recorded, and is not described as optimized linear attention.

Real 9B numerical auditing found shape-dependent bf16 GEMM differences with default
CUDA reduction settings. The training service explicitly selects cuBLASLt and
disables bf16 reduced-precision reduction **and split-K**. With these settings,
dense and chunked 2K loss agree and all adapter gradient differences satisfy the
predeclared `atol=2e-5, rtol=2e-3` check (observed maximum about `7.3e-7`). Baseline
performance comparisons must use the same settings. This policy, implementation
hashes, dependencies and memory settings are part of the resume contract. Old
checkpoints with a different policy require explicit adapter warm-start.

`dry-run` validates and prints the resolved strategy, host availability/budgets,
temporary logits size and dense-mask fallback estimate without importing Torch or
claiming that the sequence fits. It does not silently change `run.mode`. All actual
training uses a fresh attempt ID, including after any failed/OOM attempt.

Each update records projected tokens/CE blocks, MLP forward/backward blocks, actual
sequence lengths, category gradient coverage, GPU allocated/reserved peaks, host
RSS/high-water/locked observations and synchronized forward/backward wall times.
With offload it also records D2H/H2D bytes, CUDA-event copy times, effective bandwidth,
live/pinned activation peaks and host copy wall times. The profiler audit separately
reports measured copy/compute overlap; asynchronous APIs do not establish overlap.
`VmLck` may be zero for CUDA-pinned allocations; use the explicit pinned accounting.
Profiler event timings include intervening current-stream work and first-use costs.

Layer checkpointing executes MLP chunks twice forward (original and layer recompute)
and once backward. This is explicit in the counters and is a measured tradeoff.
Selective MLP checkpointing avoids the extra forward but retains attention graphs;
its much larger measured memory footprint rules it out as a long-context default.
CPU offload stays opt-in when ordinary layer checkpointing already meets the target.

Run the small-model checks without GPUs:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venvs/rl/bin/python -m pytest -q \
  tests/test_lora_memory.py tests/test_lora_config.py tests/test_lora_training.py \
  tests/test_lora_checkpoint.py tests/test_lora_entrypoint.py tests/test_lora_supervision.py
```

Run real GPU suites in separate processes so the direct numerical audit releases
its CUDA allocator before the CLI's physical-GPU idle check:

```bash
DYCON_LORA_LONG_GPU_TEST=1 .venvs/rl/bin/python -m pytest -q -s tests/test_lora_bf16_gpu.py
DYCON_LORA_LONG_GPU_TEST=1 .venvs/rl/bin/python -m pytest -q -s tests/test_lora_long_context_gpu.py
```

These tests call the formal CLI in fresh child processes, freeze YAML and tolerances,
verify receipts, test checkpoint resume and independently reload exported adapters.
The capacity ladder stops before moving higher if measured reserved-memory margin
is below 15 GiB. 64K/128K exploration is opt-in via `DYCON_LORA_EXTENDED_CAPACITY=1`;
262144 is not part of automatic acceptance. See the validation report for measured
lengths, memory and throughput. Synthetic results do not measure business quality.
