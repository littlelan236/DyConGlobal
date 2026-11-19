# Host offload budget and memory pressure policy

`training.memory.host_offload` is opt-in, with process-wide pinned admission.
The validated runtime is exactly Torch `2.13.0+cu130`. It keeps native pinned
caching/rounding for blocks up to 1 MiB. Every project activation block is page
aligned and at least 1 MiB + 4 KiB, so it bypasses native caching when released.
Small activation tensors incur this minimum charge, included before allocation.
The cutoff is a per-block limit, NOT a total native-cache limit; it also changes
caching of other large pinned consumers. Model loading and other consumers are
included in `host_allocated_bytes`. No native extension or private clear API is used.

`reclaim_policy: after_use` is the default: graph leases must be released and every
D2H/H2D event complete before a block is freed. Finalizers only enqueue reclamation.
Allocation and existing branch/group control points poll without synchronizing;
the safe window boundary synchronizes when blocks or native deferred frees remain,
then drains native events through a cacheable 4 KiB tick. Events created by native
free() itself are also completed before final draining. Only a pool with neither
blocks nor deferred frees skips device synchronization. Native scalar/staging caches survive these calls.
`bounded_reuse` explicitly retains at most `cache_limit_gib` total idle project
bytes across sizes; host pressure overrides retention. `after_use` uses an effective
idle limit of zero even if an older config specifies a nonzero cache limit.

The pool refuses allocation before the real charge plus drain scratch and current
process pinned allocation exceed `allocator_budget_gib`. Other concurrent native
users are checked but not intercepted: this is not an all-thread allocator hard
cap. Host/cgroup availability and all local ranks are checked cooperatively, with
an absolute/fractional reserve even if optional preflight is false. Missing APIs,
unsupported runtimes, CUDA graph capture, failed cleanup or budget exhaustion fail
explicitly. No global host cache clear or unbounded CPU fallback is used.

`activation_offload: memory_pressure` is a separate opt-in policy. At each checkpoint
boundary it compares driver-free GPU bytes against candidate tensor bytes plus the
configured compute and H2D retry headroom. It does not credit `reserved - allocated`,
predict tensor release, or use sequence length. Ranks may choose different boundaries;
the existing Gloo control envelope, gradient collective, commit, and retry protocol
remain global. A retry increases headroom by a finite increment and is reset at the
next window. The policy identity is `scoped_host_reclaim_v2` and is included in the
memory contract, resolved configuration, events, and metrics.

Per-rank events include CPU RSS/HWM, PSS diagnostic samples, host active/allocated and
peak values, native cache estimate, pool reserved/idle/peak values, cache reclamation,
GPU allocated/reserved/free, logical live/peak, D2H/H2D bytes, selected/retained
boundary counts and reasons, cgroup capability, and the host-global available-memory
sample. Host-global values are never summed across ranks. Existing `none`,
`checkpoint_boundary`, and length-based `auto` behavior remains compatible.

Metrics distinguish leased, event-pending and idle project bytes. Reclaim events
report dropped pool bytes and conservative net native allocation decrease; the
latter can include concurrent-user noise and first scratch-cache allocation.

The CPU acceptance suite uses a deterministic allocator adapter and Gloo ranks; it
does not claim real pinned capacity. GPU evidence uses fresh processes, short/long/
short windows, real asynchronous copies, a controlled pressure input, and a tiny
two-rank NCCL data collective. The formal 9B CLI acceptance is opt-in and must use a
new attempt and an explicitly generated native-token engineering fixture; historical
production workloads are not modified or interrupted.

## Configuration and restore

Merge this section into an existing validated LoRA training YAML; the model,
dataset, algorithm and optimizer sections retain their existing values:

```yaml
training:
  memory:
    activation_offload: memory_pressure
    host_budget_gib: 2.0  # Existing logical live-byte limit, still enforced.
    host_offload:
      enabled: true
      allocator_budget_gib: 2.0  # Process-wide pinned allocation, per rank.
      reclaim_policy: after_use # Or explicitly bounded_reuse.
      cache_limit_gib: 0.0       # Project idle cache only; native small cache survives.
      reserve_gib: 8.0
      reserve_fraction: 0.10  # Reserve is max(absolute, fraction of host RAM).
    memory_pressure:
      device_headroom_gib: 8.0
      retry_headroom_increment_gib: 4.0
      max_retries: 2          # Allowed range: 0..3; initial try is additional.
    oom_retry:
      enabled: true
      max_retries: 1         # Legacy policy setting; new policy uses field above.
    memory_telemetry:
      enabled: true
      granularity: phase
```

These are configuration defaults and an example, not a certified capacity for a
9B long-context workload. Admission accounts for every local rank even with
`preflight.check_resources: false`. Native allocator options (1 MiB cutoff/rounding, no background event thread,
cudaHostAlloc backend) are merged before
CUDA initialization; conflicts and unsupported Torch versions fail explicitly.
The process budget includes unrelated pinned allocations and one 4 KiB event-drain
scratch allocation. Ordinary Python RAM, model weights and checkpoints also require
headroom; the pinned budget does not cap process RSS or reserve the whole machine.

For bounded legacy `auto`, keep its thresholds and enable only `host_offload`.
For the historical allocator behavior, leave `host_offload.enabled: false` and use
an existing policy. `host_budget_gib` continues to limit logical live activations;
it has not become an RSS or native-cache limit. Full-parameter/FSDP2 execution
rejects the managed host mode and `memory_pressure`.

Retry feedback is local to one optimizer window and resets before the next;
there is no hidden cross-update pressure history to restore. Opt-in settings and
the policy version enter the existing exact-resume contract. Changing the policy,
budget, cache or retry settings requires contract validation and cannot silently
resume an incompatible historical checkpoint. Use an explicitly selected warm
start/new attempt if changing the contract, with its normal optimizer/RNG reset
semantics. Rollback means restoring the prior source/configuration and starting
an admissible attempt; never edit frozen configuration, receipts or checkpoints.

## Reproducing bounded acceptance

Run from the project root after checking available RAM and the selected GPUs:

```bash
timeout 240s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venvs/rl/bin/python -m pytest -q tests/test_host_offload.py tests/test_lora_oom_recovery.py tests/test_lora_memory.py tests/test_lora_config.py tests/test_full_policy.py tests/test_grpo_prefix.py
timeout 300s env OMP_NUM_THREADS=1 DYCON_HOST_OFFLOAD_GPU=1 DYCON_HOST_OFFLOAD_GPUS=4,5 DYCON_HOST_OFFLOAD_EVIDENCE=temp/NEW-EVIDENCE .venvs/rl/bin/python -m pytest -q -s tests/test_host_offload_gpu.py
timeout 5400s env OMP_NUM_THREADS=1 DYCON_HOST_OFFLOAD_9B=1 DYCON_HOST_OFFLOAD_GPUS=6 DYCON_HOST_OFFLOAD_EVIDENCE=temp/NEW-9B-EVIDENCE .venvs/rl/bin/python -m pytest -q -s tests/test_host_offload_cli_gpu.py
```

Use a fresh evidence directory for each invocation. The 9B test executes the
formal CLI in fresh processes, freezes its acceptance specification first, uses
six complete engineering groups per run, verifies receipts, compares three
repetitions of A/B/C/D plus M (none with host governance), and checks exact pause/resume. Ample GPU capacity makes C
retain all boundaries; actual partial copies are established separately by the
controlled-pressure GPU fixture. Neither fixture certifies historical long
sequences, four-card production capacity, or naturally occurring CUDA OOM.


## Migration from v1

This replaces global `pinned_max_cached_size_mb:0` and zero rounding. Both allocator
environment aliases must agree with the new options; old zero-cache overrides are
rejected before CUDA initialization. `scoped_host_reclaim_v2`, backend identity and
cutoff enter the memory contract. V1 opt-in checkpoints cannot silently exact-resume
as V2; use the existing explicit migration/warm-start rules or retain the old source.
Legacy disabled-host configurations keep the prior default memory contract.

Acceptance includes retained-graph repeated unpack, cross-stream backward, 20
short/long/short rounds, actual partial GPU copies, strict two-rank retry, formal
9B A/B/C/D/M and exact pause/resume. Functional and performance verdicts are separate;
the test fails if policy overhead exceeds 10% or incremental host governance
monitoring exceeds 3%. Historical long sequences/four-card capacity are separate.
