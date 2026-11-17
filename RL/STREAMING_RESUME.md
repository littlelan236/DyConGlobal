# Native streaming GRPO checkpoint recovery

Opt-in exception: `runtime.topology_policy: preserve_updates` supports 2/4/8-GPU bidirectional resume while preserving the original logical windows, anchor and optimizer state. Default exact mode retains same-world-size recovery. Training RNG consumption is rejected in the portable mode; see [topology switching](TOPOLOGY_SWITCH.md).

Use the existing `.venvs/rl/bin/dycon lora-train --training-config TASK.yaml`
entrypoint. Streaming applies to fresh training with
`runtime.data_preparation_async: true` and `prepared_data: null`, or to exact
resume from a supported streaming checkpoint. The parent contract selects the
data execution mode. An async flag cannot convert a prepared checkpoint into a
streaming checkpoint. Streaming warm-start, append-resume, format conversion and
world-size changes in the default exact mode are rejected. See [acceptance evidence](STREAMING_RESUME_VALIDATION.md)
for the tested single/two-GPU short native fixtures and the read-only v10 checks.

## Choose the recovery path

| Start and parent contract | Execution path |
|---|---|
| fresh, async=true, prepared_data=null | Native streaming with an explicit external raw probability source |
| resume, supported streaming contract | Streaming; requires async=true and prepared_data=null |
| resume, prepared contract | Prepared, even when async=true |
| resume, neither streaming nor prepared contract | Existing legacy non-streaming recovery |

Streaming source selection is `probability_artifact` or `dataset_probability`;
it does not silently use `model_forward`. Unknown streaming versions and
conflicting execution settings fail. `run.mode` selects dry-run versus actual
execution; `run.start.mode: resume` selects the checkpoint state. In GRPO,
`run.mode: smoke` runs the configured dataset through the real training path.

## Configure and run

Copy the parent's resolved training settings into a new task YAML and change only
the attempt, start/pause, physical device and output controls allowed by the
existing schedule contract. For example, retaining the original complete epoch:

```yaml
run:
  attempt_id: NEW_UNIQUE_ATTEMPT_ID
  mode: dry-run
  start:
    mode: resume
    checkpoint: /ABSOLUTE/PARENT/checkpoints/step-000060
runtime:
  data_preparation_async: true
  gpu: null
  gpus: [0, 1]
  output_root: /ABSOLUTE/ISOLATED_OUTPUT
prepared_data: null
training:
  max_updates: 1694
validation:
  stop_after_update: 61
```

This is a partial example, not a replacement for the parent's full YAML. Preserve
the original seed, LoRA, optimizer, scheduler, mathematical settings, preflight
switches, reference selection and parallel execution contract. Do not lower
`max_updates` to the pause target. The pause target must exceed the checkpoint
step, and a completed epoch cannot be resumed exactly.

Each invocation needs a new attempt ID, including a dry-run followed by actual
training. Use separate task YAMLs and output logs; do not reuse an executed
example ID. The checkpoint path names a committed `step-NNNNNN/` directory,
not its adapter export or an implicit latest checkpoint.

```bash
.venvs/runner/bin/dycon lora-train --training-config TASK-DRYRUN.yaml
# Use another new attempt ID and mode: smoke for actual updates:
.venvs/rl/bin/dycon lora-train --training-config TASK-TRAIN.yaml
.venvs/runner/bin/dycon lora-status NEW_UNIQUE_ATTEMPT_ID --storage-root /ABSOLUTE/ISOLATED_OUTPUT
.venvs/runner/bin/dycon artifact verify NEW_UNIQUE_ATTEMPT_ID --scope lora-attempt --storage-root /ABSOLUTE/ISOLATED_OUTPUT
```

Runner preflight imports neither Torch nor a model. It verifies checkpoint
receipt/JSON/file identities, the frozen plan, committed cursor, original source
binding, source metadata and dataset metadata. Its result explicitly reports
`tensor_validation: pending_training_model`. Actual training validates the
original adapter tensors using PEFT's saved/runtime name and dtype mapping,
restores the current checkpoint adapter, then restores optimizer, scheduler and
the corresponding rank RNG before any producer or forward starts. An independent
Torch-only snapshot check can validate the original tensor hashes without loading
the 9B base, but that check does not establish successful training.

## Evidence and publication

New streaming checkpoints contain these recovery files in the normal receipt:

```text
state.json
training-state.pt
adapter/...
grpo/metrics-prefix.jsonl
grpo/stream-plan.json
grpo/source-provenance.json
grpo/anchor-tensors.json
grpo/reference/adapter/...
grpo/stream-resume.json
```

`stream-resume.json` schema 1 binds controlled relative recovery paths, byte
counts, hashes and the original reference identity. Internal symlinks, path
traversal, missing files and checksum mismatches fail. A partial or damaged new
bundle never falls back to another directory. Publication retains the existing
staging, receipt, fsync and atomic rename lifecycle. Recovery copies the original
anchor adapter and compact metadata at checkpoint times; it does not copy a
producer queue or build a full reference-probability cache. Disk overhead is the
anchor adapter plus plan/provenance metadata per checkpoint, in addition to the
existing current adapter and training payload.

The mathematical `contract.streaming.version` remains 1. Recovery metadata does
not broaden `resume_contract_matches`: historical project source hashes remain
provenance, while schedule, model, targets, dependency versions, attention,
determinism, reference and parallel contracts remain enforced.

## Read-only historical compatibility

A historical checkpoint without recovery files may read fixed auxiliary paths
from its own `ATTEMPT/checkpoints/step-N` parent. It must provide the original
plan, source provenance, reference adapter and `initial-adapter.json`. The
checkpoint's own receipt, state and payload authorize recovery; a failed parent
does not need a top-level receipt or intact attempt progress fields. Neither the
old directory nor its receipts are modified. Missing or inconsistent original
evidence fails rather than reconstructing a shuffle or substituting today's
adapter as the anchor.

The new attempt's `initial-adapter.json` describes its current starting adapter.
The inherited original anchor is separately recorded in `grpo/anchor-tensors.json`
and `reference_identity.adapter_hash`. New checkpoints carry all recovery
metadata and can be recovered after the original attempt is removed. The old
checkpoint itself still depends on its historical auxiliary evidence.

## External inputs and progress

Recovery metadata is self-contained; training inputs are not. Native streaming
continues to require the sealed dataset, exact native token ledgers, selected raw
probability arrays and base model. Native `branches.jsonl` references external
`native_evidence.root/ledger_ref`; copying the dataset alone is insufficient.
Keep the original dataset/source absolute paths. Automatic path relocation,
source migration and CUDA RNG device-list remapping are not implemented.

| Recovery input | What must remain accessible |
|---|---|
| New checkpoint | Current adapter/training state, frozen recovery bundle and receipt; original parent attempt is unnecessary |
| Legacy streaming checkpoint | Its checkpoint plus fixed historical parent plan/provenance/reference/initial-adapter evidence |
| Sealed dataset | Original receipt and frozen group/branch metadata |
| Native token payloads | Referenced `native_evidence.root/ledger_ref` request JSON files for unconsumed groups |
| Raw reference probabilities | Selected artifact/stream metadata, receipts and unconsumed value files |
| Base model/tokenizer | Original compatible registered model files and identities |

A native ledger is a historical model request JSON containing exact prompt and
completion token IDs. It is separate from raw reference logprob arrays and
hidden activation tensors. Copying `groups.jsonl`/`branches.jsonl` alone leaves
those token payloads behind. For a canonical probability artifact, original
upstream probability sources may be offline once receipt-local copies exist;
the selected canonical artifact itself must still be available to streaming.
Use the frozen unconsumed group set to inventory dependencies. This feature
does not automatically copy data, rewrite paths or widen the frozen audit flags.

Metadata may be scanned, but native and probability payloads of consumed groups
are not reread. Frozen advantages and order are inherited directly from the plan.
For each unconsumed window, token hashes, lengths, masks and probability bindings,
checksums and finite float32 targets are mandatory even when optional historical
audit flags are false. Source and dataset metadata identity is checked during
cleanup on completion, pause and exception. A pause is not evidence that every
external payload was audited.

The two-window producer restarts from `state.group_cursor`. Prefetched or inflight
windows do not advance the checkpoint cursor. Each rank restores its own saved
RNG, and CUDA visible-device counts must match exactly. Group assignment is the
existing group-index-within-window rule; the historical assignment string does
not redefine that algorithm. Queue and producer errors propagate to training;
the distributed supervisor terminates failed peers.

For the historical v10 step-60 target the expected initial cursor is 240 groups,
7680 branches and 1492544 completion tokens. Metrics beyond that checkpoint do not
authorize skipping later updates. A successful short GPU fixture or runner
dry-run does not prove the original long contexts fit GPU memory. The original
OOM remains possible, and a full 1634-update continuation requires an explicit
execution request. v10 validation on 2026-11-14 did not execute update 61.

## Inspect progress and failures

Use `lora-status` to distinguish `last_completed_update` from
`recoverable_checkpoint_step`. Only the published checkpoint determines the
restart state; later metric lines and inflight/prefetched groups cannot advance
it. A failed parent may still have a valid checkpoint. Missing evidence,
changed source/token/probability content, wrong rank RNG topology or a changed
contract fails recovery; there is no fallback to fresh or automatic re-scoring.
Successful `artifact verify` establishes artifact consistency, while actual
forward/update acceptance and long-context GPU capacity are separate checks.


## 8x16 execution cursor

The `state.group_cursor` producer restart rule above applies to whole-parent
schedules. [8x16 mode](DATA_PARALLEL_8X16.md) restarts from
`consumed_execution_chunks`, validated against immutable complete window
boundaries, the schedule hash, round, selected native branch identities and
consumed-half prefix hash. `group_cursor` and `consumed_groups` explicitly
project `completed_parent_groups`; they do not count halves. A valid checkpoint
may have seen a parent whose other half is unconsumed. Prefetch never commits
this cursor. The existing self-contained recovery bundle includes the extended
stream plan and all-rank RNG; external inputs are still required. Exact resume
requires the same 8-rank execution/reduction protocol and partition, while the
physical GPU indexes may change. Old v10/1/2/4-rank checkpoints cannot directly
switch protocols. Streaming warm-start/append restrictions remain in force.

With `runtime.topology_policy: preserve_updates`, physical world size can change
among 2/4/8 while the source protocol, frozen windows and committed cursor remain
unchanged. A round-0 checkpoint retains all outstanding halves and resumes from
the execution-window cursor. Keep the original async/prepared binding, complete
schedule length and unconsumed external inputs. Use a new attempt ID and first
validate through the runner CLI. The real v10 step-60 eight-device dry-run passed
metadata validation on 2026-11-15; model tensor and training RNG validation still
require actual execution. See [topology switching](TOPOLOGY_SWITCH.md) and
[evidence and pending GPU acceptance](TOPOLOGY_SWITCH_VALIDATION.md).
