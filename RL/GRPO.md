# Offline GRPO with historical native groups

Native step streaming supports complete-checkpoint exact resume on one GPU or the original data-parallel world size. It inherits the frozen plan, advantages, original reference anchor and committed cursor; it still reads unconsumed external native/probability inputs. Mandatory recovery/payload identity checks apply independently of optional historical audits. See [streaming recovery](STREAMING_RESUME.md) for the YAML workflow, historical v10 compatibility and boundaries. The local-cache-only resume descriptions below apply to prepared/legacy cache paths.

Shared-prefix execution and explicit MIG selection are documented in
[SHARED_PREFIX.md](SHARED_PREFIX.md). Enable it with
`grpo.prefix_reuse.mode: boundary_grad`; the default complete-branch backend is
unchanged. The backend runs the common prompt once per group, executes branches
with independent boundary copies, accumulates boundary gradients and performs
one prefix backward at group end. It uses the existing BF16 GRPO training path,
checkpointing, checkpoint/resume and artifact verification contracts.

Canonical online/offline unions are supported through `reference_cache.artifact.scope: rl-probability`.
See [probability alignment, missing-only backfill and migration](dataset/PROBABILITY_ARTIFACTS.md).
Prepared/legacy-cache import freezes the final receipt and execution cohorts
in the shared cache; its resume uses that checkpoint-local cache. Native
streaming binds the same immutable source identity and reads remaining values
from the selected artifact. Neither resume path merges or scores sources.
Legacy `rl-replay` and `dataset_probability` remain supported.

Use the existing `dycon lora-train --training-config YAML` service. The algorithm is
`offline_grpo_k3_v1`; no rollout, resampling or new launcher is involved. Training
keeps Qwen3.5-9B bf16 and all-text-linear LoRA, with single-GPU or existing
[1/2/4-GPU whole-group or opt-in 8-GPU distinct-parent 8x16 execution](DATA_PARALLEL.md).

New GRPO prepared schedules default to deterministic `prefix_length_pool128_v1`:
shuffle eligible complete groups by seed, sort each 128-group candidate pool by
prompt length, make complete optimizer windows from adjacent groups, and shuffle
the windows. This balances four-group windows for four-GPU, one-group-per-card
execution while preserving each native group and all branches. Existing sealed
schedules and exact resumes keep their original order. Schedule hashes, append
behavior and the explicit legacy option are documented in
[prepared data](PREPARED_DATA.md).

## Data and advantages

New attempts may bind an immutable indexed prepared artifact. `resume` verifies
its manifest, segment prefix and consumed ID hash before loading only the current
window; it does not rescan the native ledger or probability source. Explicit
`append_resume` uses a strict schedule-prefix extension and a new reference
manifest for complete new groups. See [prepared data operations](PREPARED_DATA.md).

`algorithm` binds `name`, `dataset_root`, `dataset_id`, and `require_data_ready: true`.
The entire sealed dataset is scanned. Every eligible complete decision-step group
is used once, preserving native branch order and multiplicity. There is no
train/validation/test partition or split manifest. Committed receipt metadata and
explicit source bindings remain required. Repeating model/dataset content checks
and native request/token audits is optional and defaults to off:

```yaml
preflight:
  verify_identity: false
  audit_native_tokens: false
  validate_dataset: false
  verify_probabilities: false
  verify_checkpoint: false
  check_resources: false
  verify_model_state: false
  run_diagnostics: false
```

`verify_identity: true` recomputes model/tokenizer content fingerprints and audits
the dataset receipt against its full file inventory. When false, training reuses a
previously registered model identity matching current file names/sizes and reads
the committed dataset identity without rescanning file contents. No registration
or ambiguous matching registrations requires explicit identity verification.

`audit_native_tokens: true` checks native request ledger/inventory hashes, request
identity fields and prompt/completion token hashes. When false, training still
reads the original token IDs and checks structure, token lengths, target spans,
same-group prompt equality and the sequence limit, but skips repeated provenance
and token-content hashing in prepared/legacy paths. Native streaming always checks
the consumed token content hashes against frozen metadata. This setting overrides dataset `audit.enabled` during
training, including collation. Missing data does not trigger reconstruction.

All eight switches default to false and are frozen in the resolved configuration,
input and schedule contracts. They apply to real training and dry-run:

| Switch | Optional work when enabled |
|---|---|
| `verify_identity` | Model/tokenizer and dataset content verification; model/dataset identity comparison |
| `audit_native_tokens` | Request/ledger/inventory identity and token-content audits |
| `validate_dataset` | Separate complete eligibility/order/update-count pre-scan before model loading |
| `verify_probabilities` | Probability source receipt/content, per-record and value hashes, source scoring compatibility, and extra complete dataset-binding scan |
| `verify_checkpoint` | Source checkpoint/adapter/reference audits before restoring actual state |
| `check_resources` | GPU/MIG idle admission and predicted host/offload budget checks |
| `verify_model_state` | Initial adapter base-equivalence audit, explicit prefix model precheck, and frozen-base checksum comparison |
| `run_diagnostics` | Initial CE probe and optional external-anchor current-model diagnostic forwards |

For fresh prepared/legacy-cache training with all switches off, startup
prepares the registered base identity and begins model loading. The training service reads the native tokens once, computes the
requested advantages/eligibility/order and creates the required input, selection,
frozen-groups, reference cache/provenance, adapter and checkpoint contracts. The
old preliminary full dataset preparation is skipped. Advantage preparation does
not collate and discard temporary padded batches; actual training tensor batches
are built afterward. External probability import directly indexes the chosen
source and reads required vectors, without re-auditing its files or rescanning
all dataset rows. Unchecked initial-adapter compatibility is recorded explicitly
as `kind: unchecked`, never as a successful zero-increment proof.

Fresh dry-run with defaults prepares configuration/identity metadata and returns
`dataset: null`; it does not load a model or scan training data. Streaming resume
always validates checkpoint recovery evidence, frozen plan/cursor and
dataset/source metadata, even with optional preflight switches off. It does not
read consumed token/probability payloads or certify successful CUDA training. Enable
`validate_dataset` to request the full dataset precheck. Explicit probability or
resume checks can also require preparing their dataset input. Output hashes,
identity metadata and receipts are still generated because later checkpoints and
artifact readers consume them. Restore still loads the requested adapter,
optimizer, scheduler, RNG and cursor; it never silently substitutes another state.

Configuration parsing, missing required inputs, data structures needed to compute
advantages/batches, runtime shape/dtype and nonfinite loss/gradient errors remain
part of executing the selected training algorithm. Standalone dataset/artifact
verification APIs keep their existing strict defaults. Newly published
checkpoints still use their normal complete-output writer and receipts.

Missing scores/actions, incomplete groups, equal scores or an approximately tied
top-K boundary exclude a group with a recorded reason. Set
`grpo.boundary_tie_sigma: 0.0` to disable boundary-based exclusion, including exact
ties at K/K+1. Equal boundary scores then use frozen native branch order to break
ties; all-scores-equal and invalid-score groups remain excluded. Positive values
retain the gap <= tolerance * population-standard-deviation rule (default 0.05).
A corrupt identity or
inconsistent prompt fails. Exceeding `validation.max_sequence_length` fails without
truncation. Dataset model identity must match the training base, including tokenizer
files; the native tokenizer fingerprint is also bound into the objective contract.

Historical estimator scores are remaining reasoning workload in `log1p_tokens`,
not task-success probabilities. Lower is better. The pure advantage policy uses
the planned continuous positive top-K weights and zero-sum negative remainder;
it never exponentiates scores. `top_k: 4` uses K=4 at N=32 and
`min(N-1,max(1,ceil(N/8)))` for other group sizes. Explicit other K values must satisfy
1 <= K < N. Alternatively set `top_k: null` and `top_k_fraction` between 0 and 1.
All advantages and the seeded group order are frozen before optimizer updates.

### Consensus branches in the best-scoring half

For consensus-aware training, use this explicit policy in the same training YAML:

```yaml
grpo:
  advantage_policy: consensus_top_half_zero_sum_v1
  top_k: null
  top_k_fraction: 0.5
  boundary_tie_sigma: 0.0
```

Rank **all** branches by ascending estimator score, breaking ties by frozen native
branch order. At N=32, take the first 16; for other N use ceil(N/2), capped at N-1.
Only the intersection with `retained: true` receives positive advantages.
`selected` identifies the executed representative and does not enter this rule.
Every branch needs an explicit boolean `retained`; missing/malformed flags fail.

The existing score conversion uses the global cutoff score `b`, best score `s_min`
and population standard deviation `sigma`: `scale=max(b-s_min, 0.1*sigma, 1e-6)`
(0.1 is `score_scale_floor`), and `w_i=0.5+0.5*(b-s_i)/scale` for positive members.
Normalize these weights by their mean. If M branches are positive, each of the
other N-M branches receives `-M/(N-M)`, so positive mean is 1 and group sum is 0.
All N branches remain in the training group. No qualifying consensus branch skips
the group with `no_consensus_in_top_half`; all-equal and invalid scores retain
their exclusions. Boundary ties never exclude groups in this policy.

Policy/version, settings, retained membership, scores, frozen advantages and group
order are bound to the input/checkpoint contracts. Policy or membership changes
reject exact resume. Omitted `advantage_policy` retains the legacy top-K behavior.

## Configuration and one epoch

The two templates are:

- `RL/experiments/grpo-offline-k3-smoke.yaml`: defaults to dry-run and explicitly
  uses the built-in synthetic token fixture (five groups, three branches each).
- `RL/experiments/grpo-offline-k3-task.yaml`: defaults to dry-run and requires an
  actual dataset ID and a new attempt ID before use.

`grpo.group_batch_size` counts complete groups per optimizer update. The final
partial window uses its actual group count. Physical microbatches are one complete
native branch at a time, preserving its full prompt and completion. The reduction
is branch token mean, group branch mean, then update group mean. Token counts are
diagnostics and do not replace group weights. `gradient_accumulation_steps` must
be 1 and supervision must be `labels` (all native completion targets).

`max_epochs` is exactly 1. `training.max_updates` must equal
`ceil(eligible_groups / group_batch_size)`; a mismatch reports the required count.
This keeps the scheduler's complete schedule explicit. The loader never repeats
data to meet an update target. `validation.stop_after_update` can pause at an
earlier complete update boundary. Completion of the epoch rejects exact resume.
A supported non-streaming adapter `warm_start` starts a new stage with a new
attempt ID; streaming checkpoints cannot be used for warm-start or append-resume.

For an actual synthetic engineering smoke, copy the smoke YAML to an explicit task
configuration, set a unique attempt ID, an idle physical `runtime.gpu`, and
`run.mode: smoke`, then use the RL environment:

```bash
.venvs/runner/bin/dycon lora-train --training-config RL/experiments/grpo-offline-k3-smoke.yaml
.venvs/rl/bin/dycon lora-train --training-config PATH_TO_EXECUTION_CONFIG.yaml
.venvs/runner/bin/dycon artifact verify ATTEMPT_ID --scope lora-attempt --storage-root ROOT
```

With `preflight.validate_dataset: true`, dry-run scans eligibility, freezes the
input identity and estimates updates without Torch, model loading, GPU allocation
or creating an attempt. `verify_identity: true` additionally recomputes model
content identity and may update the existing model-identity registry. Defaults
skip these optional scans.

## Probability objective and reference

For each native completion target, `ratio=exp(logp_current-logp_initial)`, and the
policy surrogate is `min(ratio*A, clip(ratio,0.9,1.1)*A)`. The penalty is
`expm1(delta)-delta`, where `delta=logp_initial-logp_current`; beta is fixed at 0.02.
`clip_epsilon=0.1`, `reference=initial_policy`, and `probability_mode=raw_t1` are
the supported settings. Advantages/reference are detached, current logprobs retain
gradients, and ratio/K3/reductions use fp32. Nonfinite ratios, deltas, K3, loss or
gradients reject the update, without clamping away overflow.

Projection shares the existing backbone and causal target coordinates with CE.
Its frozen output head computes the full vocabulary normalization for selected
targets in token blocks, recomputing blocks during backward. It does not retain
full-sequence vocabulary logits or compute full-vocabulary KL. Processed replay
estimates are never the denominator. K3 is a penalty on fixed historical targets,
not an unbiased estimate of current-policy KL.

For prepared/legacy-cache training, `grpo/reference/adapter/` snapshots the initial adapter (including a warm-start
adapter). `grpo/reference.json` binds initial adapter tensor fingerprints,
model/tokenizer, dataset receipt, group/branch IDs, prompt/completion hashes,
target-mask hashes, objective version and raw probability mode.
`grpo/reference-logprobs.pt` contains only detached fp32 target logprobs.
Every committed checkpoint carries these files with receipt hashes. Resume verifies
and copies the original cache/snapshot; it never rebuilds a reference from the
updated adapter. Missing or mismatched cache/snapshot is a hard resume failure.

Native streaming checkpoints instead carry the original adapter snapshot,
`stream-plan.json`, `source-provenance.json`, `anchor-tensors.json` and
`stream-resume.json` under `grpo/`, alongside current adapter/training state and
the metrics prefix. They do not contain a full `reference-logprobs.pt` cache.
They continue to require the unconsumed external native/probability inputs.
See [streaming checkpoint layout and legacy compatibility](STREAMING_RESUME.md).

### Importing an external shared anchor

The training YAML remains `schema_version: 1`; source artifact, cache manifest
and binding contract versions are independent of the objective version.

`reference: initial_policy` describes the selected policy; `reference_cache`
chooses its source. Omitting `reference_cache`, or setting only
`source: model_forward`, retains the original forward/cache schema and schedule
contract. External modes require explicit dtype compatibility and never fall back
to model-forward on missing data:

```yaml
grpo:
  reference: initial_policy
  probability_mode: raw_t1
  reference_cache:
    source: probability_artifact
    artifact:
      scope: rl-probability
      root: /data3/wangtianyi/DyConGlobal
      id: REPLACE_WITH_COMMITTED_RAW_ID
    dtype_compatibility: fp16_source_bf16_training
    initial_diagnostic_branches: 2
```

This example binds the canonical schema-3 artifact. Prepare it before training
using [coverage planning and canonical merge](dataset/PROBABILITY_ARTIFACTS.md).
Legacy raw replay artifacts remain directly readable with `scope: rl-replay`.
The canonical importer records the final receipt and every execution cohort in
local provenance; it does not follow per-branch source paths. The producer must
use one scoring function and matching collector compute dtype, with exact allowed
cohort hashes for multiple runtimes. The training dtype policy cannot relax those
merge constraints. A partial artifact is usable only if all frozen training-eligible
branches are covered and finite; probability coverage never changes eligibility.

For a dataset-bound online stream, first enable `capture.logprobs.enabled` for a new rollout and build the dataset through the existing `rl-dataset build` entrypoint; see [online raw collection](../docs/ONLINE-RAW-LOGPROBS-CN.md). Verify raw coverage separately from token `data_ready`: unavailable or failed sidecars cannot satisfy a selected eligible branch, and an explicitly selected stream never silently falls back to model-forward. This collector currently records unsteered base-model scores without an adapter; training performs fresh/warm-start compatibility checks. Replace the entire `reference_cache` object:

```yaml
reference_cache:
  source: dataset_probability
  stream: raw_t1_output_tokens
  dtype_compatibility: fp16_source_bf16_training
  initial_diagnostic_branches: 2
```

The neutral reader is `RL.dataset.probability.open_probability_stream`. Its source
schema is documented in [PROBABILITY_STREAM.md](PROBABILITY_STREAM.md). It shares
dataset receipt, safe path and float32 shard readers. An artifact may cover extra
valid dataset groups; every selected eligible branch must be present exactly once,
with matching request, token hashes, target spans, scoring identity and finite
float32 values. Import does not change eligibility, advantages or ordering.
Staging, processed probabilities and arbitrary scoring temperatures are rejected.
Sources carry computation facts, without training roles or objective parameters.

Fresh LoRA is accepted as base-equivalent only after checking actual zero B
weights, bias, modules-to-save, active adapter and initialization configuration.
A warm-start requires the exact scoring adapter tensor fingerprint; no-adapter
base scores cannot substitute for it. Checkpoint/tokenizer identities and empty
model-function overrides (including quantization/rope/steering overrides) are
required regardless of dtype compatibility. `same_dtype` requires identical
scoring/training dtype; `fp16_source_bf16_training` explicitly permits float16
scoring and bfloat16 training. These rules do not require initial ratio=1 or K3=0.

External dry-run verifies the committed source, raw semantics, compatible scoring
metadata and selected target vectors without importing Torch or loading a model.
Actual fresh LoRA zero-increment / warm-start adapter tensor compatibility is
verified during model loading and import, before optimizer creation. Dry-run does
not certify those loaded-state checks or GPU capacity.

External imports write cache manifest schema 2 and a self-contained
`grpo/source-provenance.json`. `probability_binding` separates the source scoring
identity from the training initial-adapter proof. The training snapshot proves
compatibility; it does not claim that BF16 training recomputed FP16 source scores.
`role_binding` assigns the same cache hash to `ratio_denominator` and
`k3_reference`. Current logprobs always come from the live trainable model. No
second cache is made and source artifacts are unchanged. The algorithm remains
`offline_grpo_k3_v1`.

Prepared/legacy-cache resume verifies and copies parent-local cache and
provenance without opening the external probability source. Streaming resume
inherits the original source binding and opens that source for remaining
payloads. Source selection, dtype rule, role and content identity cannot change
on resume. Legacy schema-1 caches keep their original validation path. Project
source fingerprints remain audit metadata; model, dependency, schedule, data,
reference and numerical compatibility checks remain enforced.
An interrupted import has no manifest commit marker and cannot be used as a cache;
use a new attempt ID, preserving the failed attempt for diagnosis.

`initial_diagnostic_branches` defaults to 0. Set a small positive count to record
initial-current delta/ratio quantiles and K3 across representative branches in
`grpo/initial-current-diagnostic.json`. This optional current-model forward does
not recalibrate the anchor. `grpo/reference-preparation.json` reports cache import
or model-forward preparation time separately from optimizer update telemetry.
Raw T=1 scores of trajectories sampled at T=0.6/top-p/top-k are raw model scores
on historical tokens, not exact historical processed-behavior importance weights.

## Checkpoints and metrics

The existing adapter/optimizer/scheduler/RNG checkpoint mechanism is reused.
Contracts include dataset receipt, eligible group IDs/order, advantage policy,
reference snapshot/cache identity and objective/probability mode. Counters include
`group_cursor`, `consumed_groups`, `consumed_branches`, `consumed_tokens`, and
`global_step`. The resume cursor must be an optimizer update boundary. A checkpoint
also carries its metric prefix so an interrupted source attempt need not finish
before resuming.

Attempt receipts include objective/reference identity and consumed counters.
`frozen-groups.json`, `input-identity.json`, `selection.json`, `grpo-contract.json`,
`metrics.jsonl`, `validation.json`, and `grpo-report.json` make the result reviewable.
The artifact verifier checks that the report is reconstructed from the committed
contract, metrics and final cursor. Metrics retain gradient coverage, clipping,
time, CUDA/host memory and projection/MLP/offload telemetry, plus:

- group coverage, exclusion reasons and advantage distribution;
- ratio mean/min/max and 0/25/50/75/100% quantiles;
- clip fraction, `k3_penalty_mean`, and positive/negative branch logprob changes;
- actual consumed group/branch/token counters and partial-window group count.

## Fixed checkpoint preference probe

Use `dycon lora-probe` to rescore the same explicit complete parent groups across
checkpoints. This is separate from the positive/negative logprob changes emitted
by training: those describe the current, changing batch at the model state used
by that update. A fixed probe holds inputs, frozen advantages and the original
shared probability anchor constant and scores the explicitly selected checkpoint
in evaluation mode. It requires additional forwards and writes an independent
JSON report instead of TensorBoard events.

For branch `b`, let `L_b` be the mean raw T=1 logprob over native completion
targets and `R_b` its mean original-reference logprob. Within each complete group:

- `preference_gap = mean(L_b | A_b > 0) - mean(L_b | A_b < 0)`.
- `preference_gap_change` subtracts the same gap computed from `R_b`.
- `advantage_logprob_alignment = mean(A_b * (L_b - R_b))` over all branches.

Aggregate metrics average group values equally; absent sign comparisons are
`null` and excluded from the corresponding mean, with support counts reported.
The 8x16 protocol is evaluated as full 32-branch parents, preserving original
advantages without recentering within execution halves. This diagnostic measures
preference on a fixed training subset, not held-out or downstream success.

```bash
.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000100 \
  --group-id GROUP_A --group-id GROUP_B \
  --gpu 0 --output reports/rl-probe/step100.json
```

For a later checkpoint, use `--baseline-report reports/rl-probe/step100.json`
instead of the group arguments to reuse the fixed ordered IDs and obtain
current-minus-baseline deltas. Incompatible tokens, advantages, anchor or scoring
protocol are rejected. See [FIXED_PROBE.md](FIXED_PROBE.md) for report fields,
dry-run, checkpoint-owner configuration, source retention and GPU requirements.
This command is independent of startup `preflight.run_diagnostics`.

## Engineering acceptance

```bash
PYTHONPATH=.:tests .venvs/rl/bin/python -m pytest -q tests/test_grpo_*.py
```

Tests use explicit synthetic data and tiny real Qwen+LoRA models. They cover dense
versus chunked logprobs/objectives/LoRA gradients, clipping directions and K3,
one-epoch/tail normalization, exact pause/resume, warm-start reference freezing,
identity/tamper rejection, Torch-free dry-run and public artifact publication.
Real historical dataset/GPU capacity and benchmark quality are Phase 4, a separate
acceptance step; engineering smoke results do not establish those outcomes.

Opt-in public-CLI GPU acceptance (requires an idle selected GPU and the local
Qwen3.5-9B checkpoint) compares model-forward and external same-dtype values,
then performs pause/resume after moving the source out of its configured path:

```bash
DYCON_GRPO_EXTERNAL_GPU=1 DYCON_GRPO_GPU=0 \
  DYCON_GRPO_EVIDENCE=/ABSOLUTE_TASK_DIRECTORY \
  .venvs/rl/bin/python -m pytest -q tests/test_grpo_external_gpu.py
```

It creates explicit engineering fixture artifacts and new training attempt IDs
through the existing CLI; `gpu-acceptance.json` records the evidence and timings.
This same-dtype check is distinct from real FP16 collector/BF16 training acceptance.
Implementation results and measurement scope are in
[GRPO_BINDING_VALIDATION.md](GRPO_BINDING_VALIDATION.md).

## Progress and frozen advantage inspection

Use `dycon lora-status` for committed group/update progress and receipt-backed
checkpoints; `--active` discovers training attempts, and `--watch` follows them.
`dycon lora-tensorboard ATTEMPT_ID --action start` replays historical metrics and
follows complete new updates in a managed TensorBoard service. Total loss,
policy loss and unweighted K3 remain separate curves. Services use loopback and
SSH forwarding; stopping visualization leaves training running.

`dycon lora-advantages ATTEMPT_ID --group-ordinal 0` queries the first group in
the frozen training schedule. `--group-id` and `--branch-id` support identity
filters; `--offset/--limit` page results, and `--format json/csv` supports scripts.
All three commands accept `--storage-root ROOT` and run from `.venvs/runner`.

Prepared queries verify metadata/score shards without opening token arrays by
default. Legacy `frozen-groups.json` requires explicit `--build-index` with
`--index-status` progress. Results preserve native branch order, report saved
advantage provenance, and return unavailable/null when as-used values cannot be
proven. Group ordinals follow the frozen schedule, including append segments;
inflight groups remain uncommitted until their optimizer update completes.

Existing training remains observable without a restart; new stage detail is
available to newly started GRPO processes. Events and indexes stay outside sealed
artifacts. See [RL observability](OBSERVABILITY.md) for commands, SSH forwarding,
resume inheritance and limits, and [validation](OBSERVABILITY_VALIDATION.md) for
the tested current/legacy and GPU fresh/resume paths.


## Per-token K3 hard cap (2026-11-17)

The standalone training YAML supports an optional cap on each completion
token's K3, before beta multiplication and the existing branch/group means:

```yaml
grpo:
  kl_beta: 0.02
  k3_cap: 1.0
  ratio_guard:
    enabled: false
```

`k3_cap: null` (the default) preserves the uncapped objective and legacy
schedule contract. A configured cap must be finite, positive and representable
within the float32 upper range. The objective uses
`K_used = min(expm1(-log_ratio) + log_ratio, k3_cap)`.
Only K3 is capped: policy surrogate, PPO clipping, native target masks, original
token denominators and branch/group weights stay the same. A token strictly
above the cap contributes the constant cap and has zero direct K3 gradient;
its policy contribution is retained. Equality retains the original derivative.
This is a hard loss cap, not a gradient-only clipping rule or a straight-through
estimator. It does not restore extreme-token probabilities through K3.

For numerical stability, saturated tokens are removed from the differentiable
inverse-ratio exponentiation before it can overflow. Finite extreme negative
log-ratios can therefore produce a finite capped objective and gradients even
if uncapped K3 overflows. Nonfinite input logprob differences, policy ratios,
or final objective values are still rejected. If ratio guard is independently
enabled, guard exclusion happens first and excluded tokens do not count toward
cap saturation.

`k3_penalty_mean` is the actual capped objective mean before beta. Existing
`k3_penalty_max` and `k3_cap_1/5/10_*` diagnostics still describe the uncapped
K3 after any guard, so historical comparisons remain meaningful. New metrics:
`k3_cap`, `k3_cap_clipped_tokens`, `k3_cap_eligible_tokens`,
`k3_cap_clipped_fraction`, `k3_raw_penalty_mean`,
`k3_raw_nonfinite_tokens`, and `k3_capped_penalty_max`.
Raw mean/max are null when unavailable due to overflow; the nonfinite count
retains that evidence. These metrics also appear in the existing observability
and TensorBoard interfaces under `drift/k3/`.

An enabled cap enters the frozen schedule/checkpoint contract. Exact resume
must preserve its value; enabling, disabling or changing it requires a new
fresh attempt or another explicitly supported start mode. The existing
`dycon lora-train --training-config YAML` remains the only training entrypoint,
including whole-parent, distinct-parent two-pass and portable 2/4/8-GPU paths.
No running training or historical frozen configuration is changed by adding
this option. The example beta is a configurable value, not an experiment result.

## Per-token extreme-ratio guard (2026-11-12)

The guard defaults to disabled for existing configurations. Enable it in the
standalone training YAML:

```yaml
grpo:
  ratio_guard:
    enabled: true
    threshold: 0.001
```

The threshold must be finite and strictly between zero and one. Enabled guard
settings enter the frozen schedule contract; disabled settings are omitted for
legacy contract compatibility. When enabled, the objective excludes a completion
target when its detached, unclipped
current/reference ratio is strictly below the configured threshold (default `1e-3`). Both its policy and K3
contributions and their direct logprob gradients are zero. The original
per-branch target denominator and branch/group weights are retained, so the
other targets are not rescaled. A fully excluded branch contributes a
differentiable zero; it does not remove the group or advance the optimizer early.
Targets remain in the causal context: later valid losses can still backpropagate
through that context and shared model parameters.

The mask is applied to the log-ratio before K3 exponentiation, preventing
overflow/NaN from extreme inverse ratios. Raw ratio diagnostics still include
excluded targets. Metrics add `ratio_guard_threshold`,
`ratio_guard_excluded_tokens`, `ratio_guard_eligible_tokens` and the token-count
fraction `ratio_guard_excluded_fraction` at update level.

This changes the training objective. Existing process memory is unaffected;
ordinary exact resume retains its objective/numerical contract checks and must
not bypass a changed objective or source binding. Project source fingerprint
changes alone are not a resume gate. Use an explicitly configured new training attempt
(for example a supported adapter warm-start) for the changed objective.
No running training is restarted by this change.


## Experimental distinct-parent 8x16 protocol

See [8x16 execution](DATA_PARALLEL_8X16.md) for the explicit configuration.
Eight different parent IDs contribute 16 native branches each per full update;
32-branch scoring and frozen advantages remain unchanged. Both prepared and
native streaming publish the same versioned execution mapping. There are
`2*ceil(G/8)` updates, with one independent tail per round. The branch weight is
`1/(16*K)` for the actual K-parent window. This changes optimization order;
it does not promise equivalence to 4x32 or better downstream quality.
Execution cursor schema 1 tracks committed halves and completed parents
separately; legacy group counters project completed parents only.
