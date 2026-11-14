# Canonical raw probability artifacts

Use `dycon rl-dataset probability-plan/build/check` to align online and offline
completion logprobs. These commands do not score a model. Only the existing
`replay-logprobs` lifecycle can start the isolated native trace worker.

## Storage and immutable identities

`ROOT/RL/probabilities/ID/` is scope `rl-probability`. It contains schema-3
`probability.json`, `branches.jsonl`, `selection.json`, `targets.jsonl`,
`missing.jsonl`, `coverage.json`, `sources.json`, `provenance/`, bounded
`shards/part-*.f4`, config snapshots, `summary.json` and `receipt.json`.
`ROOT/RL/probability-plans/ID/` is scope `rl-probability-plan`; `plan.json` binds
the dataset receipt/inventory, selection, source receipts, policy, implementation
and per-branch coverage. Large branch lists are JSONL; CLI output is a summary.

Both scopes reuse `RL.dataset.artifacts.directory/publication/seal`: lock,
stage, compatible explicit recovery, receipt and atomic rename. A committed ID
cannot be overwritten. Changed source receipts, selection, policy, configuration
snapshot or implementation require new IDs. Recovery rebuilds canonical shards
from the same verified inputs; replay recovery reuses its sealed branch journal.
Stages are never consumer artifacts. No source artifact, ledger, dataset flag,
replay or checkpoint is rewritten by alignment.

The canonical manifest binds `dataset.id`, the actual enclosing
`dataset.receipt_sha256`, and branch/group inventory hashes. The older online
stream's `dataset_binding` remains its content hash; the explicit adapter checks
that binding and does not reinterpret it as the enclosing receipt hash.

The stream is `raw_t1_output_tokens`, semantics `raw_t1_output_tokens_v1`, raw
contract `{temperature: 1.0, normalization: full_vocabulary,
logits_stage: pre_transform, targets: completion_tokens}`, dtype `<f4`. Each
completion token retains one scalar in its original coordinate system, including
tokens outside a downstream loss mask. Repackaging preserves original FP32 bytes.
It performs no averaging, normalization, truncation, interpolation or model call.

## Source configuration and single-source migration

Start with [the strict configuration example](configs/probability-example.yaml).
Paths resolve relative to the project root; physical roots must resolve to safe
absolute non-root directories. `sources` explicitly select:

| kind | Required binding | Adapter |
|---|---|---|
| `dataset_stream` | dataset ID/root and explicit `stream` | Receipt-covered online schema 2; old sampled-token semantics validated as equivalent raw targets |
| `replay` | replay ID/root | Raw schema 2, including native producer model/runtime/policy metadata |
| `probability` | canonical ID/root | Schema 3, preserving branch origin and parent provenance |

The default role is `existing`. For online-only or offline-only migration, set
the corresponding source and run `probability-build`; a plan is optional when
there is no `backfill` source. A missing target fails unless `allow_partial: true`
was explicitly configured. A historical dataset with no online stream can be
declared as `dataset_stream`; it contributes no values. A dataset that declares
available values but loses their index is corrupt and fails.

One scoring function is allowed per artifact. `scoring_function_identity` binds
the checkpoint fingerprint (whose producer must deeply inventory model/config/
tokenizer/weights), tokenizer, adapter, effective model overrides and raw contract.
`execution_cohorts` separately bind collector compute dtype, FP32 reduction,
runtime/backend/version/engine/graph/cache options and implementation evidence.
FP32 storage does not imply FP32 model compute. Source compute dtypes must agree.
Unknown/incomplete identities fail. Multiple cohorts require their exact hashes
in `cohort_policy.allowed_execution_cohorts`; this cannot authorize a different
checkpoint, adapter, tokenizer or dtype. Plan/check summaries expose known cohort
hashes; undeclared cohorts are rejected, rather than inferred from a directory.

For a planned but not yet produced replay, declare the approved stable cohort
hashes from factual runtime evidence before freezing the plan. Timing,
queue counters and dispatch measurements do not define the cohort hash; original
runtime bytes, including those measurements, remain in source provenance. Graph
and cache configuration and effective execution modes remain cohort factors.
If no existing source supplies a scoring identity, provide
`expected_scoring_identity` with all legacy scoring fields (checkpoint/tokenizer/
adapter/collector_dtype/effective_model_overrides/runtime/implementation). This
is an expected contract, not evidence that an execution already occurred.

## Missing-only replay and merge

Add at most one source with `kind: replay`, `role: backfill`, and the fresh replay
ID/root it will use. Planning does not require that artifact to exist. Defaults
select every branch in the dataset loader's complete native groups;
`group_ids` freezes a subset of whole groups. Probability availability never
changes token-group eligibility, reward, mask or ordering.

```bash
.venvs/runner/bin/dycon rl-dataset probability-plan --probability-config PROBABILITY.yaml
```

The returned `plan_sha256` is SHA256 of `plan.json`; the separately returned
receipt hash identifies its publication. Add this binding to the probability
configuration and to the replay configuration:

```yaml
# In PROBABILITY.yaml, leave the other resolved fields unchanged:
coverage_plan:
  id: COVERAGE_PLAN_ID
  root: /ABSOLUTE/PLAN/ROOT
  sha256: REPLACE_WITH_RETURNED_64_HEX_PLAN_SHA256

# In REPLAY.yaml, alongside the existing schema-2 dataset/runtime/storage:
selection:
  coverage_plan:
    id: COVERAGE_PLAN_ID
    root: /ABSOLUTE/PLAN/ROOT
    sha256: REPLACE_WITH_RETURNED_64_HEX_PLAN_SHA256
```

`selection.coverage_plan` and `selection.group_ids` are mutually exclusive. The
replay ID/output root must equal the plan's declared backfill. Preparation verifies
the frozen dataset, source receipts, implementation and model/dtype before scoring.
The existing bounded producer/prefetch/continuous scheduler/async writer retains
group locality but submits only missing branches. With 31 of 32 branches present,
only the remaining native P/C trace enters the engine. Replay publication requires
exactly the frozen missing set and binds the plan ID/hash; partial group output
does not claim the whole dataset is probability-ready. Cancellation/detach/status
and recovery use the existing replay commands, without an alternate launcher.

```bash
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml --dry-run
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml
.venvs/runner/bin/dycon rl-dataset probability-build --probability-config PROBABILITY.yaml
.venvs/runner/bin/dycon rl-dataset probability-check PROBABILITY_ID --storage-root ROOT --with-values
.venvs/runner/bin/dycon artifact verify PROBABILITY_ID --scope rl-probability --storage-root ROOT
.venvs/runner/bin/dycon artifact verify COVERAGE_PLAN_ID --scope rl-probability-plan --storage-root PLAN_ROOT
```

With no missing branches, replay returns `no_missing` before GPU reservation or
worker startup; build directly exports the existing values without requiring an
empty replay artifact. Build preserves frozen existing values and accepts the
backfill only for missing coordinates. Wrong plan, missing/extra/duplicate output,
changed receipts/tokens and conflicting values fail. Duplicate existing sources
require `deduplicate_identical: true`, matching raw/model contracts and identical
value hashes; all source proofs remain in `source_ref`.

Coverage states are `valid`, `absent`, `recorded_failed` and `nonfinite`.
`allowed_missing_reasons` defaults to `[absent]`; recorded collection failures
require an explicit `recorded_failed` entry. Corruption/conflicts fail, rather than
becoming missing work. Legal `-inf` is copied and reported; it is not backfilled.
NaN, +inf, positive logprobs, wrong length/hash and unsafe locators fail.
`coverage_status: complete` means all selected branches have vectors;
`finite_coverage_complete` separately states whether every value is finite.

## Portable reading and training

Every value locator belongs to the new receipt. Shards are copied, not symlinked.
Provenance includes original receipt bytes, original manifests/adapter identity
metadata, parent IDs and original value hashes. Ordinary `probability-check`,
`artifact verify` and `ProbabilityStream` reading do not reopen original
upstream source NAS. Prepared/legacy-cache training resume uses the checkpoint
cache. Native streaming resume still requires the selected canonical artifact
and unconsumed native token ledgers, but canonical probability reads need not
reopen the artifact's original upstream sources. Explicit
`probability-check --sources` audits original source receipts
and reports `source_audit_status: incomplete` for unavailable/changed sources,
separately from local verification.

```yaml
grpo:
  reference_cache:
    source: probability_artifact
    artifact:
      scope: rl-probability
      root: /ABSOLUTE/STORAGE/ROOT
      id: PROBABILITY_ID
    dtype_compatibility: fp16_source_bf16_training
```

The shared reader and existing cache importer consume schema 3 without per-origin
paths. All training-eligible branches must have finite values. A partial artifact
is accepted only when its missing branches fall outside the training eligibility
already frozen from tokens/rewards. Import never removes a group to hide missing
probabilities and never falls back to scoring. Local checkpoint provenance binds
the canonical receipt and cohorts. Prepared/legacy-cache resume uses the
frozen cache; native streaming inherits its source binding and reads remaining
vectors from the selected artifact without importing a full cache. The same
anchor serves ratio and K3, current remains live, and the objective is unchanged.
Checkpoint recovery metadata does not embed all external training inputs; see
[streaming resume](../STREAMING_RESUME.md).
Old `rl-replay`, `dataset_probability`, cache schemas and checkpoint resume remain
supported; processed probabilities cannot be relabeled as raw.

## Acceptance boundary

The 2026-11-11 combined CPU regression completed with **348 passed, 6 skipped**.
The permanent [alignment acceptance report](../PROBABILITY_ALIGNMENT_VALIDATION.md)
records the exact command, covered scenarios, source-offline evidence and real-model
acceptance still to be performed.

`tests/test_rl_probability_artifacts.py` exercises actual online sidecars/dataset
publication, frozen plans, the formal replay dispatcher with CPU trace engines,
continuous scheduling and async writing, 32/31+1 and interleaved groups, merge,
training import, source-offline verification and local cache resume. It also tests
identity/dtype/cohort/value/receipt rejection, explicit duplicate handling,
partial eligibility, zero-missing workers, locks and interrupted recovery.

These are CPU format/identity/coverage tests. They do not certify numerical
agreement, performance or production readiness of real GPU scoring. Real-model
acceptance needs separately authorized fixed-token paired measurements and
thresholds declared before measurement. The worker retains requested/effective
Qwen GDN graph checks and actual FULL-dispatch evidence; alignment does not switch
graph modes to bypass runtime requirements.
