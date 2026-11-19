# Dynamic Adam moment residency

`training.memory.adam_moment_offload` is opt-in, initially integrated with
full-language FSDP2, independent of reference/FLA GDN and the 8x16 schedule.
LoRA explicitly rejects enabling this option. Disabled configurations retain
Torch AdamW and their old schedule identity.

```yaml
training:
  memory:
    activation_offload: memory_pressure
    host_offload:
      enabled: true
      reclaim_policy: after_use
      allocator_budget_gib: 24.0
      cache_limit_gib: 0.0
      reserve_gib: 16.0
      reserve_fraction: 0.08
    adam_moment_offload:
      mode: memory_pressure
      backend: gpu_streamed_fp32_v1
      host_state_budget_gib: 16.0
      node_state_budget_gib: 64.0
      tile_pair_mib: 64
      gpu_scratch_budget_mib: 256
      pinned_staging_budget_mib: 512
      low_watermark_gib: 12.0
      high_watermark_gib: 20.0
      restore_stable_windows: 3
      min_residency_windows: 3
      max_restore_gib_per_window: 2.0
```

Launch through `.venvs/rl/bin/dycon rl-train --training-config YAML` as usual.
The existing dry-run reports budgets and capabilities. Budgets are upper limits,
not a preallocation request. Per-rank moments can span GPU and CPU. Parameters
and gradients remain FP32 GPU shards; compute precision and GDN are unchanged.

Each m/v tile independently owns its storage. Pressure checks run before the
first window (including lazy state and gradient costs), before every branch,
and before Adam. CUDA cache is trimmed only at a pressure boundary to measure
driver free space. A conservative linear workspace estimate plus reserve is
raised using observed branch peaks; it cannot guarantee arbitrary sequence
lengths or predict every native workspace. Allocator cached bytes are recorded
but are not credited as guaranteed contiguous capacity. There is no CUDA OOM
retry. Distributed phase failures abort the attempt before cursor/scheduler
commit; partial parameter updates are terminal and cannot be retried in memory.

CPU moments are durable state, distinct from reusable activation caches. A
single pinned slot and GPU scratch stream their FP32 values through Torch
2.13.0's non-fused/non-foreach functional AdamW. The functional import is
version-checked; this backend does not implement another Adam formula. Every
tile uses the same parameter step, committed once per successful parameter.
None gradients do not advance a parameter. Unsupported optimizer modes are
rejected. Parameter updates use `no_grad` local shard views and preserve FSDP
ownership/version tracking; no `.data` mutation is used.

With S GiB of CPU-authoritative moments, every update transfers at least S GiB
H2D plus S GiB D2H, separately from initial eviction/final restoration. Ordinary
CPU backing uses independent anonymous mappings (8192 process limit), not tens
of GiB of locked pages. Every transfer records and waits for its stream event
before source release or slot reuse. The first backend is deliberately serial.

After stable short windows, only tiles that satisfy minimum residency and leave
the next window's workspace, gradient and high watermark can return to GPU.
Completed ownership transfer releases CPU mappings. Escaping reader views keep
the mapping and its budget charge alive until the last reader leaves. Required
CPU moments may persist across updates; deleting them would corrupt Adam.

Pinned staging uses the existing activation pool and process allocator budget;
`pinned_staging_budget_mib` is a sublimit, not another independent allocation
allowance. Necessary native pinned caches remain enabled. Large managed blocks
use the existing event-safe, size-separated reclamation policy. Node admission
in `state/rl-host-admission/` checks PID/starttime owners, rank and node moment
limits, MemAvailable and cgroup ancestor limits under a short lock. Only pending
reservations plus candidate increments are subtracted from availability; charged
memory is already represented there. Both pageable moments and new pinned
requests join this ledger. Existing older workers and external processes do not
participate: this is cooperative admission, not an OS hard reservation.

`ranks[].adam_moment_offload` reports authoritative CPU/GPU state, mapped and
pending storage, eviction/restore versus per-step traffic, peak/scratch bytes,
forecast decisions and cumulative phase timings. CUDA event times and host wall
times are separate. `adam_enqueue` is CPU launch time; `adam_gpu` is measured
kernel time; transfer wait includes queueing while transfer_gpu measures copy
events. These overlapping/cumulative phases must not be summed as an update's
wall duration. Full update wall duration remains `compute_seconds`.

Production full checkpoints serialize current authoritative tiles directly,
and restore incrementally without assembling complete GPU moments. Standard
optimizer state_dict/load_state_dict remain available for small interoperability
fixtures; they materialize CPU state and should not be used for large export.
The checkpoint contract records backend/layout/arithmetic/policy identity, not
transient residency. Same-backend resume is checked with full-state tiny-model
tests. Tolerance equivalence is the public promise; test-level bitwise matches
on one backend/hardware do not imply universal bitwise portability.

A standard-AdamW checkpoint cannot silently enter the new backend. Explicit
full/FSDP2 resume may add this hash-bound start control:

```yaml
run:
  start:
    mode: resume
    checkpoint: /absolute/verified/full/checkpoint
    optimizer_backend_migration:
      source_schedule_sha256: <hash_json of source state.json contract.schedule>
      target_schedule_sha256: <hash_json of schedule_contract(target Config)>
      reason: Explicit FP32 Adam residency migration
```

Only Adam residency strategy may differ in this transition. Data, world size,
LR/warmup, clipping, activation policy and all mathematical settings must match.
The source artifact is immutable; the new checkpoint records migration lineage.
Other backend transitions are rejected.

For bounded capacity/performance tests without large checkpoint export, set
`validation.skip_checkpoint_export: true` and an explicit `stop_after_update`
below the configured epoch update count. Full GRPO windows still commit in
memory, metrics/progress and an attempt receipt are written, but no model or
optimizer checkpoint is written. Such attempts explicitly report
`validation_only: true`, `resumable: false`, `final_checkpoint: null`; they cannot
serve as resume sources. The default remains normal checkpoint saving.

For expensive length ladders, `validation.capacity_microsteps: 1` additionally
runs only that many canonical microsteps of the first window, with complete
native branches/masks and fully initialized Adam state. It requires streamed
Adam and no-export validation. It does **not** call optimizer.step, advance the
scheduler/cursor, or claim a complete GRPO update. `capacity.json` records actual
branch lengths, full-state validation and memory; the attempt marks
`capacity_only: true`, and no update metric is emitted. Use separate complete
native 8x16 windows for update acceptance. These execution/output controls do
not change the mathematical schedule identity.

Tests: `tests/test_optimizer_offload.py`; opt-in GPU tests use
`DYCON_ADAM_OFFLOAD_GPU=1 DYCON_ADAM_GPUS=4,5`, and 8x16 tests use
`DYCON_ADAM_EXECUTION_GPU=1 DYCON_ADAM_EXECUTION_GPUS=4,5`. Select idle devices.
Real 9B length and throughput claims require separate complete native-branch
evidence; the tiny fixtures establish correctness, not 80K capacity.
