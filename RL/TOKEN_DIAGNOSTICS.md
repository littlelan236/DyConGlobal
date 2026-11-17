# GRPO token K3 diagnostics

Enable explicitly in the existing standalone training YAML. Use the formal
`dycon lora-train --training-config YAML` entrypoint and a new attempt ID;
resume may enable these observational controls without changing its schedule.

```yaml
grpo:
  token_diagnostics:
    enabled: true
    k3_threshold: 10.0
    max_tokens_per_branch: 8
    context_tokens: 12
    compare_full_branch: false
    compare_base: false
```

Defaults disable all diagnostics. `metrics.jsonl` committed updates gain
`token_diagnostics`: native target count, explicit NaN/positive-infinity/
negative-infinity counts for current/reference/ratio/K3, FP32 probability/ratio
zero counts, threshold exceedance and truncation counts, and bounded outliers.
Each outlier binds group/branch/native ordinal, frozen advantage, prompt and
completion lengths, zero-based completion offset and sequence position, target
token ID, context token IDs, current/reference logprob, FP64 probability and
ratio/K3 recomputation. FP64 numbers are diagnostic only; the FP32 objective
and its reductions are unchanged. Nonfinite target evidence is serialized as
strings in a descriptive error before the existing rejection of the update.

Context IDs are raw model tokens and can decode to source task content. Records
are bounded per branch (1–32 outliers, 0–32 context tokens on each side).
The default threshold is K3 > 10, independently of the ratio guard. Existing
guarded objectives retain their original masking and denominators.

Optional `compare_full_branch` recomputes a complete native branch for branches
with recorded outliers and records logprob differences from the training path.
Optional `compare_base` temporarily disables LoRA and recomputes the base model
in the training dtype, comparing it with the stored reference. Both add
read-only `no_grad` forwards, consume memory/time and use the same target scorer;
they do not recompute or replace the reference used by the objective. Do not
interpret the BF16 base control as an exact reproduction of FP16 vLLM replay.

Disabled runs preserve the legacy schedule contract. Diagnostics are excluded
from that mathematical contract and remain in the resolved attempt config.
Tests cover bounded positions, nonfinite serialization, FP32 underflow,
group aggregation, schedule compatibility, and loss/gradient/RNG invariance.

## Online evidence with a hard K3 cap

For the cap of 1.0, add this fragment to the training YAML (set `kl_beta`
separately to the chosen positive coefficient):

```yaml
grpo:
  k3_cap: 1.0
  ratio_guard:
    enabled: false
  token_diagnostics:
    enabled: true
    k3_threshold: 1.0
    max_tokens_per_branch: 8
    context_tokens: 12
    compare_full_branch: false
    compare_base: false
```

Diagnostics consume detached current logprobs from the actual training forward
and the reference logprobs used by that objective, before K3 capping. With both
comparison flags false, there is no additional model forward or reference replay.
CPU transfer, FP64 arithmetic, sorting, and JSON writing still have a cost.

The threshold applies to raw FP64 K3, not capped K3. Thus an extreme token still
has its original `k3_fp64` and `ratio_fp64` recorded when the objective uses 1.0.
The FP32 objective and FP64 diagnostic can differ at the threshold due to
rounding. FP32 raw overflow can remain finite in FP64; FP64 overflow is recorded
as the JSON string `"inf"`. This is evidence from the training scoring path,
not an independent replay.

Read `token_diagnostics.outliers` in each committed `metrics.jsonl` record.
The enclosing `global_step` identifies the committed update;
`evaluated_at_global_step` identifies the model state used to score it.
Token and context are saved as IDs. Decode those IDs with the run's tokenizer
to inspect text; decoding does not require loading model weights or replaying
the training forward.

Above this threshold, the largest raw outliers per branch are retained (nonfinite
candidates first). The limit of 8 is configurable up to 32; it is not a promise
to save every exceeding token. Check `threshold_exceeded_tokens` and
`truncated_outlier_tokens` to distinguish counts from retained identities.
Only committed updates enter this history; a failed update does not produce a
committed metrics record. These switches do not retroactively recover evidence
for runs that had diagnostics disabled.
