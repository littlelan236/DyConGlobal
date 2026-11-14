# Prepared GRPO data and append continuation

Prepared 与 native streaming 是两条由冻结 contract 区分的数据路径。async=true 不会把 prepared resume 转为 streaming。Streaming fresh/resume 不要求预先发布全量 prepared/reference 产物，但持续依赖未消费的外部 native ledgers 和 raw probabilities；不支持 append_resume。详见 [流式恢复](STREAMING_RESUME.md)。本页“不访问外部来源”的 exact resume 描述仅适用于 prepared 路径。

All training uses `dycon lora-train --training-config YAML` and `RL.service.run`.
New GRPO attempts using the prepared path publish prepared and reference artifacts before updates. They
do not write `frozen-groups.json`. CE and legacy checkpoint schemas keep their
existing paths. No existing attempt, dataset, checkpoint or receipt is rewritten.

## Prepare and inspect

```bash
.venvs/runner/bin/dycon lora-prepare --training-config TRAINING.yaml \
  --prepared-id PREPARED_ID --shard-groups 16
.venvs/runner/bin/dycon artifact show PREPARED_ID --scope rl-prepared --storage-root ROOT
.venvs/runner/bin/dycon artifact verify PREPARED_ID --scope rl-prepared --storage-root ROOT
```

Preparation is CPU-only and does not import Torch or load model weights. It scans
native groups once and uses the same eligibility/advantage implementation as
training preflight. New GRPO configurations default to
`grpo.shuffle_algorithm: prefix_length_pool128_v1`: seeded compact group IDs are
partitioned into candidate pools of up to 128, sorted by prompt token length,
formed into complete `group_batch_size` windows, then the windows are seeded and
shuffled. Equal-length groups retain their seeded order. A final partial window
is retained; no group is dropped or duplicated. Group contents, branch order,
advantages and reference values remain intact. The exact output order is frozen
in `schedule.json`; `max_updates` must equal the planned update count.

For four-GPU one-group-per-card execution, this groups four similar prompt
lengths into each optimizer update to reduce prefix-backward stragglers. Pool
size 128 is a fixed schedule contract value, not an automatic memory/batch size.

Set `grpo.shuffle_algorithm: stable_random_v1` to explicitly create a new
random-order schedule. Artifacts created before this field existed retain their
recorded `stable_random_v1` order and continue to resume without rebuilding. An
append segment freezes its own named shuffle algorithm and remains a strict child
of the parent's schedule.
The output includes a `prepared_data` binding; copy that mapping to the training
YAML to explicitly reuse this release. Without a binding, a fresh invocation
publishes an artifact named from its unique attempt ID. There is no implicit
search for a compatible or latest artifact.

Artifacts use the existing root/publication/locking/receipt machinery:

| Scope | Directory | Payload |
|---|---|---|
| `rl-prepared` | `ROOT/RL/prepared/ID` | manifest, schedule, dataset binding, SQLite inventory and sealed binary shards |
| `rl-reference` | `ROOT/RL/references/ID` | manifest, initial adapter snapshot, per-group logprob chunks and source provenance |
| `rl-extension` | `ROOT/RL/extensions/ID` | sealed parent checkpoint, prepared release and constant scheduler extension request |

Each prepared group stores one prompt in little-endian int64, branch completions
in int64, and scores/advantages in float64. SQLite indexes store stable group and
branch IDs, evidence metadata, offsets, counts and content hashes. Native target
masks have the existing `all native completion targets` semantics and reconstruct
at window load time. Padding and tensors are not persisted. This replaces the
illustrative JSONL indexes in the design with a standard indexed store; binary
arrays, stable identities, shard receipts and parent references retain the same
responsibilities. New reference chunks contain one group each; adapter weights
are stored once per reference artifact, not repeated in every chunk.

`show` returns counts, hashes, segment count and parent bindings. `verify` also
walks every prepared group to validate bounds, branch order and native token
hashes. Resume verifies sealed inventories and checks token/offset/content hashes
when loading the needed windows; it does not collate consumed groups.

## Exact resume

```yaml
run:
  attempt_id: NEW_UNIQUE_ATTEMPT
  mode: smoke
  seed: 42
  start:
    mode: resume
    checkpoint: /ABSOLUTE/PARENT/checkpoints/step-000001
```

Keep the same complete training configuration, dataset receipt, seed, advantage
policy, batch size, model/LoRA/dependencies and reference
selection. The checkpoint pins prepared/reference manifest and receipt hashes,
schedule ID/hash, segment/ordinal, last consumed ID, cumulative consumed-ID hash,
and group/branch/token counters. Mandatory checks run even with optional preflight
checks off. Optimizer, scheduler, adapter and Python/NumPy/CPU/CUDA RNG restore
from the complete checkpoint. Legacy source-code fingerprints remain in old
receipts for audit but are not a code-version compatibility gate. Dataset,
prepared schedule, checkpoint, adapter, model/tokenizer and reference content
hashes remain enforced. No native ledger or external probability source is
opened and no shuffle is repeated. An exact resume after epoch completion fails.
An exact resume inside a previously appended segment keeps the same extension
lineage and does not create another segment.

## Append complete new groups

Publish a new immutable dataset with the existing dataset build interface. It may
contain just new complete groups or unchanged parent groups plus new groups.
Parent metadata, branch IDs, token hashes, scores and retained membership must be
identical. Mutating a parent group or reusing one of its branch IDs fails. An
append is not a way to change advantage policy or add branches to an old group.

Create an extension training YAML with the new dataset ID, a fresh attempt ID,
the same mathematical training settings, and the new total `max_updates`.
The total is the **sum** of each segment's ceiling division by group batch size.
For five old groups plus three new groups and batch size two it is `3 + 2 = 5`.
Set the start fields and choose a new extension ID:

```yaml
run:
  start:
    mode: append_resume
    checkpoint: /ABSOLUTE/PARENT/checkpoints/step-000001
    extension_manifest: /ABSOLUTE/ROOT/RL/extensions/EXTENSION_ID/manifest.json
training:
  scheduler: constant
  max_updates: 5
```

Retain all other required fields from the parent configuration. Then:

```bash
.venvs/runner/bin/dycon lora-prepare --training-config EXTENSION.yaml \
  --prepared-id CHILD_PREPARED_ID --parent-checkpoint /ABSOLUTE/PARENT/checkpoints/step-000001 \
  --extension-id EXTENSION_ID --append-seed 19
.venvs/rl/bin/dycon lora-train --training-config EXTENSION.yaml
.venvs/runner/bin/dycon artifact verify EXTENSION_ID --scope rl-extension --storage-root ROOT
```

The request validates the parent's complete checkpoint and consumed prefix first.
Parent segments are an exact prefix of the child schedule; only new IDs are
shuffled using `append_seed`. Parent and child partial windows never merge.
Training finishes the old remainder before the added segment. Completed parent
epochs may be extended. Constant schedules inherit optimizer/scheduler state and
global update counts; linear extensions fail explicitly because no linear
extension rule is implemented.

Reference values for old groups remain in their sealed parent artifact. New
groups use the **original** initial-policy adapter snapshot, temporarily loaded
and then restored to the checkpoint adapter before optimizer/RNG restoration.
External sources must provide complete new-branch coverage with the same raw T=1
anchor, dtype rule, tokenizer, adapter and scoring identity. Probability metadata
may describe the whole new dataset, but only new branch vectors are imported.
Run `rl-dataset probability-plan`, the existing missing-only replay, and
`probability-build` first when values are missing; training never launches a
second scoring worker or treats partial coverage as a complete anchor.

The report records `extension_of_attempt`, parent checkpoint, extension request,
new group/branch/token counts, old/new update totals, prepared cache hits, source/native
scan counts, materialized groups, reference reused/new counts, segment count,
verification seconds and bytes read. Model loading, reference preparation and
per-update training times are reported separately. Byte counts represent local
verification reads and omit operating-system page-cache effects.

## Migration, failures and rollback

A checkpoint with a streaming contract uses the
[native streaming recovery path](STREAMING_RESUME.md); absence of prepared
metadata alone does not identify a legacy non-streaming checkpoint. Only
checkpoints with neither prepared nor streaming contracts use the old
non-streaming path. No path silently converts a checkpoint into another format. Historical
source-code identity metadata is retained but does not block exact resume. For explicit
read-only import of a committed legacy attempt:

```bash
.venvs/runner/bin/dycon lora-prepare --training-config MATCHING.yaml \
  --prepared-id MIGRATED_ID --source-attempt /ABSOLUTE/LEGACY_ATTEMPT
```

This reads the old JSON once and produces a new release with `source_attempt`.
The original group order/content hash must match. It does not rewrite the old
checkpoint contract or add append support to it; use explicit adapter warm-start
for a new stage. Warm-start freezes a new initial policy and cannot reuse
base-policy probabilities unless they match that policy's adapter.

An interrupted writer leaves a hidden `.ID.staging` directory, never a consumable
published ID. Retry with new prepared/reference/extension and attempt IDs; these
scopes intentionally do not offer in-place recovery. Preserve stages for
diagnosis. An orphan sealed prepared artifact from a failed extension request
remains valid but cannot authorize append without a sealed request. A failed
reference import prevents updates and leaves the parent unchanged. Never edit a
receipt or reuse an incomplete/failed attempt ID. Back up and restore only this
change's code diff for rollback, preserve all artifacts, and restore the source
implementation matching the checkpoint you intend to resume. Restoring an older
reader does not make it capable of consuming the new schema.

CPU/tiny-Qwen and short native-token 9B smoke tests establish format, cursor and
state inheritance behavior. They do not establish long-context GPU capacity or
business-model quality. Test evidence and timing limits are recorded separately.


## 8x16 indexed execution

The [8x16 protocol](DATA_PARALLEL_8X16.md) retains complete indexed 32-branch
parent artifacts and their original IDs, scores, advantages and token storage.
The preparation service adds a separate frozen execution plan and binds its
hash into the published schedule. Readers validate this identity and native
parent metadata before model loading. During execution, a rank reads only its
selected 16 completions; remote-rank groups use metadata placeholders. Reference
parent chunks are memory-mapped, and only selected target arrays are validated
and copied for the active half. Receipt verification can still scan all sealed
bytes for integrity. No source dataset or old prepared artifact is rewritten.
The new schedule supports exact resume; append execution is explicitly rejected.
