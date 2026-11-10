# Offline GRPO with historical native groups

Use the existing `dycon lora-train --training-config YAML` service. The algorithm is
`offline_grpo_k3_v1`; no rollout, resampling or new launcher is involved. Training
keeps the Qwen3.5-9B bf16, single-GPU, all-text-linear LoRA boundary of the CE service.

## Data and advantages

`algorithm` binds `name`, `dataset_root`, `dataset_id`, and `require_data_ready: true`.
The entire sealed dataset is scanned. Every eligible complete decision-step group
is used once, preserving native branch order and multiplicity. There is no
train/validation/test partition or split manifest. Receipt checks remain mandatory.
Actual prompt tokens must match across members; native token checks are performed
before training, even if dataset source audit was disabled during construction.

Missing scores/actions, incomplete groups, equal scores or an approximately tied
top-K boundary exclude a group with a recorded reason. A corrupt identity or
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
earlier complete update boundary. Completion of the epoch rejects resume; a new
stage requires explicit adapter `warm_start` and a new attempt ID.

For an actual synthetic engineering smoke, copy the smoke YAML to an explicit task
configuration, set a unique attempt ID, an idle physical `runtime.gpu`, and
`run.mode: smoke`, then use the RL environment:

```bash
.venvs/runner/bin/dycon lora-train --training-config RL/experiments/grpo-offline-k3-smoke.yaml
.venvs/rl/bin/dycon lora-train --training-config PATH_TO_EXECUTION_CONFIG.yaml
.venvs/runner/bin/dycon artifact verify ATTEMPT_ID --scope lora-attempt --storage-root ROOT
```

Dry-run scans eligibility, freezes the input identity and estimates updates without
Torch, model loading, GPU allocation or creating an attempt. It computes the base
content identity and may update the existing model-identity registry.

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

`grpo/reference/adapter/` snapshots the initial adapter (including a warm-start
adapter). `grpo/reference.json` binds initial adapter tensor fingerprints,
model/tokenizer, dataset receipt, group/branch IDs, prompt/completion hashes,
target-mask hashes, objective version and raw probability mode.
`grpo/reference-logprobs.pt` contains only detached fp32 target logprobs.
Every committed checkpoint carries these files with receipt hashes. Resume verifies
and copies the original cache/snapshot; it never rebuilds a reference from the
updated adapter. Missing or mismatched cache/snapshot is a hard resume failure.

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
