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
