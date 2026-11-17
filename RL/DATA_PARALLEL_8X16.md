# Experimental GRPO 8-GPU distinct-parent 8x16

Physical-topology exception: `runtime.topology_policy: preserve_updates` can execute this same frozen two-pass plan on 2/4/8 GPUs, or split a whole-parent plan across eight GPUs without adopting two-pass semantics. Default exact mode retains the original restrictions below. See [topology switching](TOPOLOGY_SWITCH.md) for configuration, RNG restrictions and validation boundaries.

The default `exact` implementation of this explicit opt-in mode uses the existing `dycon lora-train --training-config
YAML` lifecycle, GPU locks, eight supervised workers, preparation, checkpoint
publication and artifact verification. It supports native step streaming and
indexed prepared input, on one node with eight distinct physical GPUs.

```yaml
runtime:
  gpu: null
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
grpo:
  group_batch_size: 8
  parallel_branch_chunk_size: 16
  parallel_branch_schedule: distinct_parent_two_pass_v1
  branch_partition_seed: 42
  prefix_reuse:
    mode: boundary_grad
```

Merge these fields into a complete task YAML, use a new attempt ID, and retain
`training.checkpoint_policy: layer`. `runtime.data_preparation_async: true`
selects native streaming for fresh input without a prepared binding; the normal
prepared route is also supported. Run runner-environment dry-run before an
actual `.venvs/rl/bin/dycon lora-train` invocation. Do not wrap it in torchrun.

In default `exact` mode, eight GPUs without splitting and splitting on 1/2/4 GPUs
are rejected. `preserve_updates` permits the original two-pass plan on 2/4/8
physical GPUs and whole-parent execution on 2/4/8 GPUs without converting plans.
Other chunk sizes,
non-boundary execution, fewer than eight eligible parents or an eligible parent
with other than 32 branches are rejected. Required structural checks happen
before model loading regardless of optional preflight flags.

## Frozen scoring and execution

Complete 32-branch native parents remain the scoring and anchor identity unit.
The existing implementation computes eligibility, retained/ranks and advantages
once. Halves preserve scores, advantages, masks, references and branch IDs;
there is no half-level reranking, cutoff, normalization or zero-sum correction.
Random half advantages can have nonzero sum.

`sha256_ordinal_permutation_v1` sorts native ordinals by the digest of partition
algorithm, partition seed, stable parent ID and ordinal. The first 16 form half
0 and the rest half 1; execution within each half uses sorted native ordinals.
This avoids process hash randomization and traversal-order dependence.

`distinct_parent_two_pass_v1` executes all half-0 chunks and then all half-1
chunks. Each round independently uses the existing prefix-length candidate-pool
ordering, with a frozen SHA256-derived round seed. Every regular window has
eight distinct parent IDs. Round tails remain separate, with no padding,
resampling or discarded branch. Different parent IDs need not have unique prompt
text. Parent prompt deduplication is not introduced.

For G parents, preparation freezes 2G chunks and `2*ceil(G/8)` update windows.
For G=6773, there are 1694 updates, 13546 chunks and 216736 branches, with two
5-parent / 80-branch tails. The partition, parent metadata, windows, world size,
assignment and loss protocol are hashed into a distinct execution identity.
Ranks read that identity; they do not partition using mutable runtime settings.

## Objective and cursor

For an actual K-parent window, the loss is the mean of K selected 16-branch
means; each branch first uses its completion-token mean. Branch weight is
`1/(16*K)`, including tails. FP32 SUM all-reduce combines already normalized
contributions without dividing by world size. Gradient presence, unused
parameters, AdamW decay, clipping and replica fingerprint checks still apply.
Idle ranks participate in synchronization.

At fixed parameters, the average of a parent's two half means equals its full
mean. Actual updates occur between halves; this is a new optimizer trajectory,
not exact-equivalent 4x32 training. The same trajectory budget does not guarantee
identical tail weighting or improved downstream quality.

`execution_window_cursor_v1` binds committed windows, round, schedule hash,
consumed half identity hash and branch/token counts. Checkpoints publish only
after a complete optimizer update with cleared gradients; some parents may
still have an outstanding half. Progress distinguishes planned parents/chunks,
consumed chunks, seen distinct parents and completed parents. Legacy
`group_cursor` / `consumed_groups` explicitly project completed parents only.
No prefix graph or boundary-gradient buffer is saved.

Exact resume retains all-rank RNG, adapter, optimizer, scheduler, mapping and
reduction protocol, with the same world size. Physical cards may change.
Old 1/2/4-rank or v10 checkpoints cannot resume directly as 8x16. Any supported
new-stage adapter warm-start resets optimizer/cursor and binds the actual new
reference anchor; streaming warm-start and 8x16 append-resume are unsupported.
No silent fresh fallback is allowed.

With `runtime.topology_policy: preserve_updates`, an existing two-pass checkpoint
can instead resume on 2/4/8 GPUs. Each rank executes its assigned original halves
before the same single optimizer update; rounds, weights and incomplete-parent
cursor remain unchanged. The plan's historical `world_size: 8` remains a logical
identity. Actual worker count is saved separately, including the saved rank RNG
cardinality. This policy rejects training RNG consumption. A whole-parent
checkpoint can use eight GPUs under this policy while retaining its whole-parent
schedule; it does not become an 8x16 checkpoint. See [operation](TOPOLOGY_SWITCH.md)
and [validation](TOPOLOGY_SWITCH_VALIDATION.md).

## IO, telemetry and validation boundary

Streaming freezes complete parent metadata, then reads only this rank's 16
native completions/reference arrays per half. Other ranks' payloads are metadata
placeholders. Parents are revisited in the second round, so prompt/native IO is
repeated. Prepared input retains complete indexed parent artifacts; only selected
completions are read during execution. Its reference chunks are memory-mapped
and selected targets copied. Integrity verification may read sealed bytes in
either path. Source fingerprints and access to unconsumed inputs remain required.

Prefix telemetry is explicitly drained per update attempt, including failures,
and carries parent ID, half, native ordinals, round/window and rank. Status and
TensorBoard distinguish parent/chunk counters. Elapsed update time and rank
critical path are separate from summed synchronized segment times. Suffix work
is reduced per rank, while prefix graphs and boundary buffer shapes are not
halved; a short fixture cannot prove capacity for long v10 prefixes.

CPU validation uses `tests/test_grpo_execution.py`, actual 8-process Gloo in
`tests/test_grpo_parallel.py`, real tiny Qwen checkpoint/resume, and historical
1/2/4-rank regression tests. GPU acceptance and measured performance remain
pending while the eight cards have existing workloads. This mode remains
experimental; there is no measured speedup or quality claim.

GPU acceptance must use fresh attempts under
`temp/data-parallel-8x16-acceptance/`, the formal CLI, Qwen3.5-9B, all eight idle
cards, tails/cross-round/pause-resume, rank fingerprints and artifact verify.
Separately test representative v10 long-prefix memory. Repeat the same-data,
same-total-branch-budget 4x32 versus distinct-parent 8x16 comparison twice with
short/long prefixes; report median/P90, branches/s, tails, total/critical-path
time, sync, IO/reference, allocated/reserved memory, startup/checkpoint cost and
failure rate. The suggested 10% throughput gate is unverified. Training quality
requires separate common-evaluation and multi-seed comparisons.

The opt-in public-CLI tests are prepared in `tests/test_grpo_execution_gpu.py`:

```bash
DYCON_GRPO_8X16_GPU=1 DYCON_GRPO_GPUS=0,1,2,3,4,5,6,7 \
DYCON_8X16_EVIDENCE="$PWD/temp/data-parallel-8x16-acceptance/gpu-NEW-ID" \
  .venvs/rl/bin/python -m pytest -q -s tests/test_grpo_execution_gpu.py
```

Run only after all selected cards are idle. Choose a fresh evidence directory.
This covers both data routes, all four boundaries of a nine-parent epoch,
all-rank RNG and artifact verification. Set `DYCON_8X16_PERFORMANCE=1` for two
repeats of matched 512-branch 4x32/8x16 plans at 64- and 4096-token synthetic
prefixes. These use an engineered external raw-probability fixture and do not
establish training quality or v10's actual long-context capacity. The separately
prepared `v10data-8x16-capacity-PENDING.yaml` has not been executed; it is a fresh
stage on the full historical source, with a one-update pause and a fresh anchor,
not a continuation of the old optimizer.

## 8×16 validation status (2026-11-14)

The explicit experimental [8-GPU distinct-parent protocol](DATA_PARALLEL_8X16.md) has passed CPU/Gloo regression: **426 passed, 12 skipped**, including real eight-process synchronization and tiny-Qwen prepared/streaming resume. See [validation details and pending acceptance](DATA_PARALLEL_8X16_VALIDATION.md). The skips do not count as GPU acceptance; real Qwen3.5-9B eight-GPU correctness, long-prefix capacity, throughput and quality remain pending.
