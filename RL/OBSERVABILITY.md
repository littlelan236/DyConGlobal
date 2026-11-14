# RL progress, TensorBoard and frozen advantages

Streaming recovery reports a verified checkpoint independently of a failed parent's top-level receipt or missing attempt contract. The checkpoint's own receipt, frozen metadata and committed cursor determine recoverability; inflight/prepared windows and later metrics do not change the resume point. Runner checks do not load Torch or prove actual anchor tensors/CUDA execution. See [streaming recovery](STREAMING_RESUME.md).

Use the project CLI from `/home/wangtianyi/DyConGlobal`. Queries run in
`.venvs/runner` without Torch. Managed visualization and legacy index workers use
`.venvs/analysis` with `tensorboard==2.20.0` and `ijson==3.5.1`; they never load a
model. Training continues through the existing `.venvs/rl/bin/dycon lora-train`.

## Current training

```bash
.venvs/runner/bin/dycon lora-status --active
.venvs/runner/bin/dycon lora-status grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon lora-status grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --watch 5 --format json
.venvs/runner/bin/dycon lora-tensorboard grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --action start --port 16006
```

On the local PC, keep this forwarding connection open while browsing:

```text
ssh -N -L 16006:127.0.0.1:16006 wangtianyi@10.249.187.9
http://127.0.0.1:16006
```

The server only binds `127.0.0.1`. Start is idempotent for the same source and
port; occupied ports are errors. A running attempt uses a training lifecycle:
the service stops when the attempt becomes terminal or its training process exits.
For a non-running attempt, start creates a query lease of 30 minutes by default;
override it with `--lease-seconds N`, and renew an active lease with
`--action renew --lease-seconds N`. The synchronizer and TensorBoard both stop
when the lease expires. Use `--action status` to inspect both processes
and event freshness, and `--action stop` to stop only their exact registered
PID/starttime/argv identities, wait for exit and publish a stop receipt. A failed
worker never stops or restarts training. `--action rebuild` explicitly discards
the old synchronization cursor, creates a fresh generation and restarts the owned
visualization processes; use it after reviewing a source replacement/conflict.

## Progress semantics

`status` preserves the raw attempt status, including preparation stages. Its
`observed_liveness` is independent: an exited process with a nonterminal attempt
state is `not_running_with_stale_state`, not an invented success or failure.
Old process discovery binds the exact training-config argument to its attempt
ID, physical output root and process creation time. The live record for new
training also records proc starttime, which prevents PID reuse mistakes.

`last_completed_update` and `committed_consumed_groups` come from complete metric
lines after optimizer updates. `inflight_update` and
`live.completed_groups_in_update` describe work whose parameters have not yet
been committed. `recoverable_checkpoint_step` comes from a published checkpoint
receipt with a matching state hash and contract. Checkpoint publication can lag
the latest update. `execution_stop_target` is the validation pause boundary;
`max_updates` is the complete frozen schedule target, including append segments.

The latest and recent metrics contain scalars only. Status reads an expanding
tail, not the full frozen token JSON. JSONL parsing accepts complete newline
terminated objects, holds a partial tail for the next read, reports malformed
complete lines and conflicting steps, and has a 64 MiB per-line resource limit.
Unknown arrays such as attention masks are discarded by the streaming parser.
Watch mode preserves its incremental cursor and prints one JSON object per line.

ETA uses the last ten completed updates produced by this attempt, excluding the
inherited prefix. It reports sample count, mean, median, remaining updates and
confidence; times exclude startup and may exclude checkpoint overhead. Seconds
and bytes are the raw metric units; memory chart units are GiB (2**30 bytes).
No stale-metric timeout automatically declares a hung CUDA kernel or restarts it.

New GRPO training exposes window/group forward/backward, prefix phases where
available, optimizer update, checkpoint and finalization boundaries. Group and
branch identity are never inferred from a partially processed gradient tensor.
`contribution_ordinal` counts objective contributions, and `branch_ordinal` stays
null when the backend cannot prove a native branch mapping. Old running trainers
report `stage=unknown`. Heartbeats every ten seconds indicate process liveness;
stage timestamps indicate when a boundary changed. Diagnostic writes are
throttled to two seconds, except committed/checkpoint/error boundaries, and fail
as warnings without changing optimizer, cursor or random-number behavior.

## Curves and recovery

Metrics are replayed into a single independent run per attempt. Horizontal step
is the lineage's `global_step`, with `progress/inherited` identifying the parent
checkpoint prefix. The cursor records the parent checkpoint and inherited step.
Resume/append attempts never share event directories even when steps overlap.

`train/loss` is the total objective; `train/policy_loss` and
`train/k3_penalty_mean` are separate, the latter before multiplication by beta.
A policy-ratio trio is also recorded: `train/ratio_mean` is the valid completion
token mean, while `train/policy_ratio_positive_advantage` and
`train/policy_ratio_negative_advantage` partition that same ratio by positive and
negative branch advantage. A missing sign in an update is omitted from TensorBoard
rather than emitted as a misleading zero.
A lower training objective does not establish a higher benchmark success rate.
The UI provides step, relative and wall-time axes and its own smoothing control;
use smoothing zero to view raw values. No smoothing is applied to source data.
Only complete optimizer updates create loss points, even if an update takes many
minutes. New metrics include `observed_at` and the actual producing attempt ID;
inherited metric lines are copied unchanged. Legacy metrics without timestamps
have `historical_wall_time_estimated=true`; their event wall time is zero rather
than a fabricated per-step time derived from file mtime.

The synchronization/flush/reload intervals are two seconds. The normal local
filesystem acceptance target is less than ten seconds after an update. A single
writer lock and `(global_step, source-line SHA256)` detect duplicate/conflicting
rows. A crash can occur between event and cursor publication: every restart
replays history into a new generation, then atomically replaces the visible
logdir; it does not append to the interrupted generation. Source inode,
size, first-line and last-committed-line fingerprints detect truncation,
replacement and committed-prefix changes; such errors remain visible in status
until explicit rebuild. Old generations are excluded from the active run.

## Frozen group/branch advantages

```bash
.venvs/runner/bin/dycon lora-advantages grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --build-index
.venvs/runner/bin/dycon lora-advantages grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --index-status
.venvs/runner/bin/dycon lora-advantages grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --group-ordinal 0
.venvs/runner/bin/dycon lora-advantages observability-gpu1-resume-20261112-v1 --storage-root /data3/wangtianyi/DyConGlobal --group-ordinal 0 --format csv
```

No group filter returns twenty group summaries and a `next_offset`. An explicit
group returns its branches, up to 256 rows; truncation reports total, returned and
next offset. `--offset/--limit` page summaries without a group filter, and native
branches with a group filter. CSV requires an explicit group/branch filter.
Branch IDs reverse-resolve through the index. All ordinals and ranks are zero
based. Group ordinal follows the frozen training schedule, with schedule,
segment and segment-local identities for prepared artifacts. Native branch order
is preserved. Lower historical score is better; equal scores share a rank,
and the separate tie-break order uses native branch order. Frozen ranking
diagnostics take precedence over deriving this ordering from scores. Excluded groups
return their reason and null advantage, never a fabricated zero.

Each result binds the attempt contract, dataset receipt, policy/config hash and
frozen order. `frozen_prepared_artifact` reads the actual score/advantage shard;
metadata index and score receipts are verified without opening token arrays.
`PreparedReader.metadata()` is the shared query API; `raw()` and explicit deep
artifact verification retain their existing full-token behavior. Bounded
`--token-preview 0..128` is available for prepared artifacts and defaults to zero.
Legacy indexes deliberately contain no tokens and reject token preview.

`frozen_groups_json` means the original saved frozen advantages. Old records
without saved values return `provenance=unavailable` and null advantages; current
math is insufficient evidence of the producer's historical implementation, so
it is never silently used to manufacture an as-used advantage.

Legacy `--build-index` is explicit and returns an owned worker with byte/group
progress. It parses the complete JSON using the locked streaming parser, spools
one group's canonical token representation temporarily to disk for the original
order/content hash, and retains no token arrays in Python or in the final SQLite
cache. It validates group count/order, score/retained/advantage/token order hash,
source hash, and start/end file identity. A temporary database is published only
after validation; concurrent builds share a lock. Queries never start a 59 GiB
scan. Ready matching indexes are reused. Unsealed sources are clearly marked
`locally_fingerprinted_unsealed`; sealed sources additionally verify their
frozen-JSON receipt hash. Query results verify the published index checksum.

## Mutable state and verification

All source binding, live heartbeat, managed service records, event generations
and legacy indexes live under
`runtime/rl-observability/<canonical-root-attempt-contract-key>/`. Status and
advantage queries never update attempts or checkpoints. No mutable file is added
to sealed attempts/prepared artifacts; `artifact verify --scope lora-attempt`
therefore remains valid across visualization start/status/stop.

Tests: `tests/test_rl_observability.py` (runner/analysis),
`tests/test_rl_observability_training.py` (rl), and the existing prepared,
lifecycle, service and advantage regression suites. See
`OBSERVABILITY_VALIDATION.md` for this deployment's evidence and process state.

## Memory and OOM recovery events

As of 2026-11-14, event/summary behavior has CPU regression coverage and the
formal CLI reads legacy v10 without mutation. That attempt has no new-format
memory event files; zero files/buckets cannot establish its memory behavior.
Production GPU telemetry and actual transfers await GPU availability. See
[OOM validation](OOM_RECOVERY_VALIDATION.md) for the deployment evidence.

Each rank appends `memory-events.jsonl` under its attempt directory (rank zero at
`memory-events.jsonl`, other workers under `distributed/rank-N/`). Events include
step/group lengths, committed versus inflight step, phase, allocator decision,
allocated/reserved/peak/free bytes, host budget, and OOM cleanup/retry status.
`metrics.jsonl` remains committed-update-only. A retry keeps the same window and
cursor and advances the optimizer once after a successful attempt.

The one device-wide sampler resets peaks only after sampling them, aggregates
nonoverlapping phase intervals into attempt peaks, and excludes peer waiting from
compute phase time. Existing prefix/update callbacks do not independently reset
those counters. Branch samples are aggregated by group and phase. Attention-mask
records and profiler state are scoped to an attempt, including failed attempts.
The sampler uses existing compute synchronization points; it adds no per-layer
synchronization. Transfer timings are reported separately from wall time.

Use the Torch-free, read-only summary through the ordinary CLI:

```bash
.venvs/runner/bin/dycon lora-memory ATTEMPT_ID --storage-root /data3/wangtianyi/DyConGlobal --limit 100
.venvs/runner/bin/dycon lora-memory ATTEMPT_ID --storage-root /data3/wangtianyi/DyConGlobal --rank 1 --phase prefix_backward
```

It groups samples by rank, sequence-length interval, phase, offload mode and
outcome (`committed`, `CUDA_OOM`, peer-discarded, fatal, or uncommitted). It streams
bounded records and supports bounded output. An incomplete trailing line from an
active writer is ignored explicitly. The original OOM text is preserved: the
failed allocation is absent from allocated peaks, and reserved-minus-allocated is
not evidence of a reusable contiguous allocation or a leak. CPU state-machine and
hook tests do not establish actual CUDA savings or pinned-transfer behavior.


## 8x16 parent and execution progress

[8x16 mode](DATA_PARALLEL_8X16.md) writes metrics schema 3. Status and TensorBoard
expose `planned_parent_groups`, `planned_execution_chunks`,
`consumed_execution_chunks`, `distinct_parent_groups_seen`,
`completed_parent_groups` and `execution_window_cursor` separately. The legacy
group cursor/consumed counters project only fully committed parents, so they can
remain zero throughout round 0. Per-update `parent_group_count` and
`execution_chunk_count` are K, and branch count is 16K. Parent branch count is
32; loss is a mean over selected halves at the current parameter state.
`parallel.rank_critical_path_seconds` and end-to-end `seconds` are elapsed times;
summing synchronized segment durations across ranks is aggregate compute, not
wall time. Prefix rows carry exact execution identities and are update-scoped.
