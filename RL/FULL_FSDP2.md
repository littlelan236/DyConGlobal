# Full language-policy RL with FSDP2

`dycon rl-train --training-config YAML` uses the existing RL service, GPU locks,
attempt lifecycle and distributed worker supervisor. Schema 1 remains LoRA;
schema 2 selects `tuning.mode: full`. The legacy `lora-train` command rejects full
configurations. Help, configuration parsing and training dry-runs do not load Torch.

The scope includes embedding, every decoder parameter (including GDN non-Linear
parameters), final norm and lm_head. Shared parameters are optimized once.
Vision remains frozen in the immutable base files and is included in HF exports.
Unknown module ownership fails explicitly.

## Supported configuration

See the complete, machine-specific short acceptance configurations:

- [pause after update 1](experiments/full-9b-fsdp2-acceptance-pause.yaml)
- [resume through update 2](experiments/full-9b-fsdp2-acceptance-resume.yaml)
- [FLA pause after update 1](experiments/full-9b-fla-fsdp2-acceptance-pause.yaml)
- [FLA resume through update 2](experiments/full-9b-fla-fsdp2-acceptance-resume.yaml)
- [FLA selected native capacity measurement](experiments/full-9b-fla-fsdp2-native-capacity.yaml)

The pause/resume examples use a synthetic fixture, BF16 computation, FP32 master weights, FP32 AdamW
moments and FP32 gradient reduction. They are engineering acceptance examples,
not a production dataset or benchmark-quality training recipe. The learning rate
is explicitly 1e-6. Select idle GPUs and new attempt IDs for subsequent runs.

Required schema 2 fields:

```yaml
tuning: {mode: full, scope: language_model, freeze_vision: true}
execution: {backend: fsdp2, sharding: full_shard, topology_policy: exact}
optimizer: {name: adamw, master_weights_dtype: float32, states_dtype: float32}
```

BF16 matmul numerics are fixed to the existing defaults:
`bf16_allow_reduced_precision_reduction: false`, `bf16_allow_split_k: true`.
The full loader configures deterministic algorithms, disables TF32 and selects
cuBLASLt through the same numerical setup used by probes.

The initial milestone supports dense Qwen3.5, prepared offline GRPO, complete
branches, explicit `reference_pytorch` or BF16 `fla_chunk`, disabled prefix reuse, layer checkpointing and
sequence lengths up to 2048 in the historical default configuration.
Managed decoder checkpoint activation offload is opt-in (see below). Full-policy 8x16 uses the same frozen two-pass execution plan as LoRA on 2/4/8 ranks, with complete 32-branch parents split into two 16-branch updates; prefix reuse remains disabled for FSDP2. `training.warmup_updates` is optimizer-mode agnostic and defaults to zero. Parameter/
optimizer CPU offload and automatic OOM retry remain disabled. Megatron, shared-prefix execution,
streaming/append resume and topology migration are rejected. Actual production
length capacity requires separate measurements; inputs are never silently truncated.

Select FLA with `model.gdn_backend: fla_chunk` and `model.dtype: bfloat16`.
The existing pinned FLA/core 0.6.0 adapter changes only GDN chunk execution.
Convolution, normalization, the FSDP2 strategy and optimizer precision keep their
existing implementations. This is runtime acceptance, not cross-backend numerical
equivalence or training-quality acceptance. Historical failed numerical audits
remain applicable; see [FLA backend](FLA_BACKEND.md).

For bounded native capacity measurements, `algorithm.group_ids: [ID, ...]`
explicitly selects complete parents in a full-policy task YAML. The shared
preparation path retains every branch and exact token coordinates; unknown or
duplicate IDs fail. The selection is frozen in the schedule and input proof.
The empty default keeps the existing dataset-wide behavior and historical
schedule identity. This selection currently requires full-policy prepared training.
Length limits still reject overlong selected parents without truncating them.

### Two-pass 8x16 execution

Full-policy training reuses the frozen `distinct_parent_two_pass_v1` execution
plan and complete-parent scoring from [8x16](DATA_PARALLEL_8X16.md). Configure
`group_batch_size: 8`, `parallel_branch_chunk_size: 16`,
`parallel_branch_schedule: distinct_parent_two_pass_v1` and
`prefix_reuse: {mode: disabled}`. Use `dycon rl-train` with 2, 4 or 8 physical
GPUs and `runtime.topology_policy: exact`. FSDP2 topology migration remains
unsupported; selecting fewer GPUs for a fresh attempt retains the logical plan.
Each rank processes its assigned parent halves in synchronized 16-branch rounds.
Every branch has weight `1/(16*K)` for an actual K-parent window, with one
world-size compensation for FSDP's mean reduction. Empty slots run a locally
loaded branch with zero loss, preserving collectives without counting extra
branches. The optimizer advances once per complete window. The execution cursor
tracks consumed halves separately from completed parents, including each tail.
Shared-prefix graph reuse is a separate optimization and remains unavailable in
the full-policy executor.

### Shared learning-rate warmup

`training.warmup_updates` applies identically to LoRA and full training, measured
in successful optimizer updates, not branches or microsteps. It defaults to zero
and must not exceed `max_updates`. With N warmup updates, update u (1-based,
u<=N) uses `learning_rate*u/N`. `scheduler: constant` then keeps the base rate.
With `linear`, the remaining updates decay linearly: update u>N uses
`learning_rate*(max_updates-u+1)/(max_updates-N+1)`; the scheduler reaches zero
after the final update. Zero warmup preserves the historical linear schedule.
Scheduler state resumes at the saved optimizer step; exact resume must retain
warmup configuration. Warm start resets the optimizer and warmup. Omitting the
field or setting it to zero preserves old schedule-contract identities.

## Execution and numerical contract

Parameters are built on meta, explicitly converted to FP32 metadata, sharded by
decoder block and root, then populated directly from bounded safetensor slices.
There is no full CPU model or per-rank full GPU model during loading. Root owns
embedding/head together to preserve tying. AdamW excludes embedding, norm and
bias from weight decay and covers every unique trainable parameter exactly once.

Each logical window has one frozen branch microstep table. Every rank executes
the same number of forwards/backwards; missing assignments use a zero-contribution
branch. Branch losses retain parent/branch means, with exactly one world-size
compensation for FSDP averaging. The old LoRA manual gradient reduction is bypassed.
Global clipping reduces shard squared norms. The differentiable chunked output
projection updates lm_head as well as the entire decoder.
Finite zero gradients are valid when ratio guards or clipping saturate the whole
window; missing gradients and nonfinite norms fail explicitly.

Fresh training freezes raw T=1 target logprobs before any update. Reference
identity includes base/tokenizer, parameter inventory and initial shard hashes.
External raw probability streams must match the initial base scoring identity;
warm_start builds a new model-forward anchor. Resume retains the original anchor.

## Persistence, export and inspection

`rl-full-checkpoint-v1` stores local FP32 weights/moments in bounded tensor files,
plus rank RNG, scheduler, optimizer groups, logical cursor and complete contracts.
All ranks finish staging before the primary commits the receipt and atomically
publishes the directory. A failed/incomplete staging directory is not resumable.
Same world size, backend, schedule, model and numerical environment are required.
Warm start loads policy weights and resets optimizer/scheduler/cursor/reference.

Full FLA checkpoints additionally freeze `gdn_execution` in the training and
initial-anchor identities. Workers compare the actual binding (including pinned
Qwen/FLA source digests, versions and state precision semantics) before loading.
Every rank writes sealed `gdn.json` identity/call evidence; artifact verification
rejects missing evidence or zero chunk calls. Reference-to-FLA exact resume fails.
Historical reference checkpoints remain readable. An explicit warm start may
import weights under a new backend and builds a new anchor/optimizer identity.
Metrics record all-parameter gradient validation, every local FP32 master/moment
finite check, and per-rank GDN call observations. Probe binds the checkpoint
execution identity; HF exports carry it in `full-policy-export.json` so independent
reload callers can install the same binding before constructing Transformers.

Independent reload must call `install_gdn_backend(config.model,
expected_identity=metadata['contract']['gdn_execution'])` before standard HF
`from_pretrained`. Ordinary HF checkpoint files do not install the project
adapter automatically.

```bash
.venvs/runner/bin/dycon rl-status ATTEMPT --storage-root ROOT
.venvs/runner/bin/dycon rl-advantages ATTEMPT --storage-root ROOT --group-ordinal 0 --limit 3
.venvs/runner/bin/dycon artifact verify ATTEMPT --scope rl-full-attempt --storage-root ROOT
.venvs/runner/bin/dycon rl-export --checkpoint CHECKPOINT --base-model BASE --output NEW_DIRECTORY --dtype bfloat16
.venvs/runner/bin/dycon rl-probe --checkpoint CHECKPOINT --group-id GROUP --gpus 5 6 --output REPORT.json
```

`rl-tensorboard` reuses the existing synchronizer and lease management. Per-rank
memory and sampled parameter deltas are in `metrics.jsonl`; `policy_changed`
records bounded sampling evidence, not a full-tensor equality proof. Checkpoint
and export receipts record all file sizes/hashes. Export streams shards on CPU
and restores the frozen vision weights from the fingerprint-matched base model.
It never gathers the whole model on a rank. Probe uses the original frozen anchor.

## Validation

CPU tests cover all-parameter gradients against the conditional-model oracle,
tied weights, unequal parent/branch weights and zero contributions at 1/2/4
simulated ranks, Adam updates, external reference identity and disk exhaustion.
Opt-in NCCL acceptance uses the formal runner CLI and compares real FSDP2 worlds,
pause/resume, optimizer/RNG state, warm start, export and Transformers reload.

```bash
DYCON_FULL_FSDP2_GPU=1 DYCON_FULL_GPUS=5,6,7 \
  .venvs/rl/bin/python -m pytest -q tests/test_full_fsdp2_gpu.py --basetemp=temp/NEW_TEST_DIRECTORY
```

Four-rank NCCL acceptance requires four idle GPUs and remains deferred by the
operator. Existing GPU0–4 training must be preserved. Numerical smoke acceptance
does not establish benchmark improvement or long-context production capacity.

Real 9B short-fixture measurements on GPUs 5/6: 8,953,803,264 language parameters
and 456,010,480 frozen visual parameters. Update 1 took 9.05 seconds of computation,
with peak allocated/reserved memory 70.56/80.08 GiB per rank. After exact restore,
update 2 took 5.39 seconds with 80.04/93.11 GiB per rank. Host process RSS was about
2.4 GiB per rank. The second update is close to device capacity; these measurements
do not establish capacity at the configured 128-token upper bound or at 2048 tokens.
Each FP32 weight+moment checkpoint is about 100.07 GiB. On this shared server,
writing, fsync and hashing take substantially longer than the training computation.
The second checkpoint took 1052.96 seconds to commit. The standard BF16 HF export
contains 775 safetensors files and 19,306,208,736 weight bytes (about 17.98 GiB).

The real 9B acceptance completed two updates across pause/exact resume, followed
by successful `rl-full-attempt` artifact verification and a fixed complete-parent
probe at step 2. Standard `Qwen3_5ForConditionalGeneration.from_pretrained` loaded
the exported model independently on GPU7. Conditional-logit target logprobs for
three branches / 12 target tokens matched the two-rank FSDP2 probe with maximum
absolute error 0.0 (predeclared BF16 tolerance: atol=0.05, rtol=0.01). This checks
the fixed fixture only. Independent reload and comparison took 15.74 seconds;
peak GPU allocated/reserved memory was 17.58/17.61 GiB and host peak RSS was
11.59 GiB. Training CPU offload and OOM retry remained disabled. GPUs 5–7 were
released afterward; existing GPU0–4 training continued.

Final two-rank tiny NCCL tests also cover a sealed native dataset with unequal
2/3/4-branch parents and unequal completion lengths. Loss, preclip gradient norm
and final parameters match world 1 within the predeclared FP32 tolerances.
Rank-local restore failure terminates peer workers and leaves no attempt receipt.

TensorBoard service startup was not accepted on this server because the existing
`.venvs/analysis` environment is missing. The neutral aliases reuse that existing
dependency requirement. Status, frozen-advantage queries and scalar parsing can
be used independently of the TensorBoard service.

## Subsequent FLA full-policy runtime acceptance

The FLA pause/resume examples completed two 9B updates on GPUs5/6, followed by
deep artifact verification, BF16 export and a complete-parent probe. Independent
Transformers reload on GPU7 matched all 12 fixed target logprobs exactly. Tiny
FLA NCCL tests additionally compare worlds 1/2, unequal native parent sizes,
FP32 moments and bit-exact pause/resume. These checks retain the historical
cross-backend numerical discrepancy; they do not establish numerical equivalence.

The selected native example completed three full 32-branch parents with maximum
sequence lengths 1800, 1552 and 1532, in frozen shuffle order. It consumed 96
branches and 14,141 targets without truncation. Updates took 24.481, 24.119 and
23.769 seconds, including gradient/optimizer checks. Peak allocated memory was
70.563 GiB on the first update and 80.035 GiB thereafter; reserved memory reached
93.443 GiB. The first update creates Adam state and its memory profile differs
from subsequent updates. Saving the final checkpoint took 997.882 seconds; the
formal training CLI took 1339.900 seconds, and deep artifact verification passed.

The allocator emitted eight expandable-segment mapping OOM warnings when device
free memory fell to approximately 6–8 MiB. The task completed without a project
OOM retry, but allocator-internal recovery is distinct from that configuration.
These samples establish successful runtime with tight memory, not stable
production capacity or capacity at exactly 2048 tokens. No standalone NCCL
profile or controlled reference/FLA throughput comparison was performed.

At the user's request, historical test checkpoints and the new test payloads
were removed after their verification consumers finished; the HF export was
also removed after independent reload. Original receipts, JSON metadata, metrics
and deletion ledgers remain under `temp/full-fla-fsdp2-20261117/`. Removed
checkpoints cannot be resumed or deeply verified again. GPUs5–7 were released.

## Full cross-backend gradient diagnostic

`dycon rl-gradient-compare --comparison-config YAML [--dry-run]` compares a
complete prepared parent from an explicit fresh BF16 full-policy training YAML.
The comparison YAML has `schema_version: 1`, `training_config`, `group_id`,
`output` (a new directory below project `temp/`) and optional `modes`:
`fixed_cotangent`, `shared_reference_grpo` (both by default). See
`experiments/full-9b-gradient-comparison.yaml`.

The formal command owns the usual GPU locks and worker supervisor. Backend
phases run in fresh workers at the same topology. Full FP32 parameter shard
digests must match before comparison. The reference backend supplies one shared
initial-policy logprob anchor. Fixed cotangents are `-advantage / target_count`,
with the standard world-size/parent/branch loss weighting; actual GRPO reuses
`window_update` and the same shared reference, preserving ratio/clipping effects.
Both compare full FP32 reduced gradients before clipping. There is no optimizer,
weight update, sequence truncation or CPU offload. Vision remains frozen.

FP64 reductions cover every gradient element and report reference/actual norms,
difference norm, relative L2, cosine and maximum absolute error for all language
parameters, each tensor and parameter categories. Relative L2 means
`norm(actual-reference)/norm(reference)`; a zero reference norm with a nonzero
difference is explicitly undefined. No cross-backend equivalence threshold is
implied by successful completion. Temporary reference gradient tensors are
removed only after all modes finish; the final receipt seals remaining reports
and the cleanup ledger. An earlier phase receipt records those deleted tensor
hashes and cannot subsequently be deeply verified as a standalone phase.

Real 9B measurement on 2026-11-18: the complete 32-branch native parent
`g-80eae3da202857131e049cc0c0b00ee7002f6c2e` (maximum length 1532, 4485 targets)
covered 8,953,803,264 language parameters / 427 unique tensors on GPUs5/6.
Full-gradient relative L2 was 10.423867% for fixed cotangents and 13.848474%
for actual shared-reference GRPO; cosines were 0.994870390 and 0.991747274.
Maximum target logprob difference was 0.170940876. These are measured numerical
discrepancies, not numerical-equivalence acceptance. Fixed cotangents still
traverse each backend's own forward graph; the result does not isolate the
backward kernel. Neither the two relative errors nor historical LoRA errors
can be directly subtracted to assign a cause. The evidence and sealed report
are under `temp/full-gradient-comparison-20261118/native32-v1/`.


## Managed activation offload

Full training reuses `BoundaryOffload` and the scoped `ManagedHostPool`; FSDP2
sharding, optimizer, anchor and checkpoint interfaces are unchanged. Hooks wrap
only decoder non-reentrant checkpoint input saves. Parameters, all-gather buffers,
current-layer workspaces and output-head checkpoint inputs are not offloaded.
Every completed branch drains released leases; necessary small native pinned
cache remains available. Long layers can reclaim completed leases at allocations.

```yaml
training:
  warmup_updates: 500 # first update starts at 1/500 of learning_rate; shared by LoRA/full
  memory:
    activation_offload: memory_pressure # checkpoint_boundary / auto / none also supported
    host_budget_gib: 8.0                # logical live activations per rank
    host_offload:
      enabled: true
      reclaim_policy: after_use
      allocator_budget_gib: 8.0         # actual pinned allocation per process
      cache_limit_gib: 0.0
      reserve_gib: 8.0
      reserve_fraction: 0.10
    memory_pressure:
      device_headroom_gib: 8.0
    oom_retry: {enabled: false}
```

Auto retains globally coordinated length thresholds; memory_pressure makes local
per-tensor driver-free decisions. Actual allocator capacity and shared-host reserve
are checked before allocation; this is cooperative admission, not an OS reservation.
Budgets must cover all concurrent ranks and jobs. Allocation refusal is terminal;
FSDP2 collective-bearing backward is never retried locally.

Lengths above 2048 require managed host mode, BF16, exact-length batches, chunked
output loss and Flash-only attention. Admission does not prove capacity; measure
each backend and length. Reference and FLA both use the same boundary hook.
The schedule contract freezes the selected memory policy and allocator version;
exact resume must retain them. Historical no-offload contracts remain unchanged.
Per-rank update metrics expose transfers, pool state, native host allocation and
selection decisions; `lora-memory` reads the existing memory event log.
