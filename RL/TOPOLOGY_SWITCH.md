# Resume GRPO on 2, 4 or 8 GPUs while preserving optimizer windows

Use the existing `dycon lora-train --training-config YAML` interface and a new
attempt ID. The opt-in runtime policy changes the physical assignment of work;
it preserves the source's logical training protocol, frozen windows, advantages,
initial anchor, optimizer/scheduler and committed cursor.

```yaml
run:
  attempt_id: YOUR_NEW_ATTEMPT_ID
  mode: dry-run
  # Retain the source seed and all other frozen training fields.
  start:
    mode: resume
    checkpoint: /ABSOLUTE/ATTEMPT/checkpoints/step-NNNNNN
runtime:
  gpu: null
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]  # Or two/four explicit physical devices.
  topology_policy: preserve_updates
  # Retain the source output/input settings, except an authorized output move.
```

This is a fragment to merge into the source's complete training YAML. Keep the
original algorithm, global group batch, schedule, prefix settings, probability
source, preflight switches, memory policy and original total update count.
For streaming checkpoints retain `data_preparation_async: true` and
`prepared_data: null`; their unconsumed native ledgers/probabilities must remain
available. Prepared checkpoints retain their immutable prepared/reference input.

Run runner-environment dry-run first:

```bash
.venvs/runner/bin/dycon lora-train --training-config PATH_TO_NEW_YAML
```

Dry-run checks the source topology/protocol and existing metadata contracts; its
`physical_execution` describes the target and `training_rng_validation` explicitly
remains pending. Change only the new YAML's `run.mode` to `smoke` for actual
execution through `.venvs/rl/bin/dycon lora-train --training-config ...`.
Do not wrap the CLI in torchrun. Select idle GPUs or first pause the source at a
complete checkpoint and confirm its process/resource cleanup.

## Supported plans and weights

Both directions, 2↔4↔8, support indexed prepared and native streaming checkpoints.
The default `topology_policy: exact` retains the old same-world-size admission.
Once portability is adopted, retain `preserve_updates` on subsequent resumes,
including same-card-count resumes.

**Whole-parent plan:** the original parent windows remain unchanged. Physical
tasks contain at most 16 native branches; tasks are assigned in frozen parent /
native-branch order using `task_ordinal % current_world_size`. A parent with N
branches in a K-parent window retains branch weight `1/(K*N)`, including
non-32-branch parents and tails. All halves finish at the same parameters before
one update. Eight cards do not turn this into the distinct-parent two-pass plan.

**Distinct-parent two-pass plan:** reuse the original sealed half partition,
two rounds, windows, tail weights `1/(16*K)`, and execution-window cursor. On
two/four GPUs each rank executes multiple original half tasks before the same
single update. Incomplete parents at a round-0 checkpoint remain incomplete;
already committed halves are never replayed. The historical execution plan's
`world_size: 8` remains part of its immutable identity, not the current physical
worker count.

Gradients use the existing FP32 SUM reduction, followed by one global clipping,
AdamW step and scheduler step. Never average again by physical world size.
Idle ranks participate in synchronization. Whole-parent rows are reconstructed
by native branch ordinal; missing/duplicate/misassigned contributions reject the
update. Objective metrics retain parent weighting rather than averaging chunks.

Each split parent task recomputes its own prefix with the existing backend;
only parameter gradients cross devices. No prefix graph or boundary buffer is
migrated. More GPUs can increase prefix computation, so speedup must be measured.
Portable whole-parent streaming currently preserves the existing full-parent
CPU token loading; reference arrays are imported only for locally used parents.
This is not a claim of minimal per-rank CPU IO or memory.

## Checkpoint identity and RNG

Historical checkpoints and their receipts are never modified. New attempts keep
the original logical `contract.parallel` and original training/input contracts.
The target is recorded separately in:

- `topology-transition.json`, copied into new checkpoints and covered by receipts;
- `state.json.physical_execution`, covered by checkpoint inventory hashes;
- committed metric `physical_execution`, alongside the actual `parallel.world_size`.

The transition records the source checkpoint receipt and source/target physical
topology. A later resume resolves the actual number of saved rank RNG states
using `physical_execution`, not the historical logical contract world size.

This first version accepts only **training computations that do not consume RNG**.
Model audit rejects nonzero dropout/known stochastic attention settings. Every
physical forward/backward task snapshots Python, NumPy, CPU Torch and visible
CUDA RNG and rejects the entire optimizer window if any advances. An arbitrary
stochastic training callback is therefore unsupported. Existing exact-mode
stochastic resume retains its original behavior.

The source rank RNG states remain saved. On a topology change, existing rank
states are retained where possible; additional ranks inherit bookkeeping states
cyclically. This does not represent a preserved stochastic trajectory: correctness
depends on the enforced no-training-RNG contract. CUDA-visible-device cardinality
must still match saved per-worker state. Supporting dropout/random training
would require a separately versioned logical-task RNG protocol.

The guarantee is the same objective, optimizer windows and restored optimizer
state. FP32/BF16 summation and prefix VJP order can differ; bitwise-identical
parameters, gradients and RNG streams across GPU counts are not promised.

Warm-start, append-resume and unindexed legacy non-streaming migration are not
part of this policy. Resume requires an unfinished schedule and a complete
checkpoint with cleared gradients. Missing source data does not trigger replay
or a silent fresh restart. Nonphysical schedule/anchor/model mismatches remain
errors; opting in does not bypass their compatibility checks.

## Validation

`tests/test_grpo_topology.py` exercises real CPU/Gloo 2/4/8-process switches,
independent unsplit objective/gradient oracles, Adam moments and scheduler,
non-32 parents, tails, and real tiny Qwen+PEFT prepared/streaming checkpoint
cycles for both whole-parent and two-pass plans. It verifies unchanged logical
contracts, cursors and source checkpoint bytes and validates new receipts.

Run the CPU suite with:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venvs/rl/bin/python -m pytest -q tests/test_grpo_topology.py
```

The separate GPU acceptance is explicitly opt-in and needs eight selected idle
GPUs, a local model and the identity fixture described by
`tests/test_grpo_execution_gpu.py`. Set `DYCON_8X16_MODEL` to the actual local
Qwen3.5-9B directory; the fixture's historical model default may not exist on
this server. After checking those prerequisites, run:

```bash
DYCON_GRPO_TOPOLOGY_GPU=1 DYCON_GRPO_GPUS=0,1,2,3,4,5,6,7 \
  .venvs/rl/bin/python -m pytest -q tests/test_grpo_topology_gpu.py
```

Optionally set `DYCON_GRPO_TOPOLOGY_EVIDENCE` to a new persistent evidence
directory. This test starts real training through the CLI; it is not a dry-run.

Real Qwen3.5-9B/NCCL cross-topology correctness, BF16 numerical tolerances,
long-context GPU capacity, throughput and downstream quality require separate
GPU acceptance. CPU/Gloo success does not establish those results.

See `TOPOLOGY_SWITCH_VALIDATION.md` for the recorded implementation checks.
