# Sparse causal CE supervision

完整 CLI/YAML、训练恢复和可运行函数示例见[训练接口使用指南](TRAINING.md)。
本文保留稀疏监督的坐标、算法与 loss/恢复契约。

The formal entry is `.venvs/rl/bin/dycon lora-train --training-config YAML`.
This adds supervision selection to the existing single-device training core.
Backbone inputs, causal attention, differentiable recurrent/convolution history,
all-linear text LoRA targets, and frozen embeddings/base/output head are preserved.
There is no KL/reference/reward/RL objective or distributed implementation here.

## Configuration and coordinates

```yaml
training:
  memory:
    loss_chunk_tokens: 16
  supervision:
    mode: explicit_mask  # labels | explicit_mask | uniform
    max_targets_per_sample: null
```

Omitting `supervision` preserves `labels` behavior. Sparse modes require positive
`loss_chunk_tokens`; configuration fails rather than enabling chunking implicitly.
`uniform` requires a strictly positive integer `max_targets_per_sample`. Other modes
require null. Unknown fields, bool/float limits and conflicting masks are rejected.

Python batches retain `input_ids`, `attention_mask`, and `labels`, with an optional
`loss_mask: BoolTensor[B,L]`. The coordinates refer to original **target tokens**:
`loss_mask[b,j]` selects `head(hidden[b,j-1]) -> labels[b,j]`. Position zero must be
false. Every selected label and its preceding predictor must be accessible and
nonpadding; selected labels cannot be `-100`. Different labels and input IDs are
allowed. Selection does not infer EOS, text segments or agent-step boundaries.
The input provider expresses its final agent step by setting the corresponding
raw target coordinates in `loss_mask`.

`explicit_mask` requires a mask on every batch; `labels` and `uniform` reject any
external mask. No broadcasting, floating weights, 4D block packing or second
coordinate-list input is accepted. Every sample must select at least one target.
Exact length still requires one contiguous right-padded sample and trims padding
only, even when the last selected target occurs near the beginning.

`uniform_valid_targets_v1` sorts eligible raw target coordinates. It takes all if
N <= K, the last if K=1, otherwise index `i*(N-1)//(K-1)` for i=0..K-1. It uses no
RNG and makes no claim of unbiased estimation of full-token CE.

## Python API and loss contract

```python
from RL.config import Memory, Supervision
from RL.memory import LossBackend
from RL.training import train

# model, batches, optimizer and scheduler are provided by the caller.
# Required train keywords, shown here explicitly instead of training_arguments:
state = train(model, batches, LossBackend(Memory(loss_chunk_tokens=16)),
              optimizer, scheduler, supervision=Supervision(mode="explicit_mask"),
              max_updates=4, gradient_accumulation_steps=2, grad_clip=1.0,
              vocab_size=model.config.text_config.vocab_size, max_length=128)
```

For `explicit_mask`, each original batch must include the bool mask. For `uniform`,
omit `loss_mask` and pass `Supervision(mode="uniform", max_targets_per_sample=8)`.
For the default labels behavior, omit both the external mask and `supervision`.
The CLI explicit preset uses the fixture's generated mask; custom mask files are
not a `lora-train` option. Custom batches use the Python core API above.

`RL.supervision.prepare_batch` exposes raw coordinates, original availability,
selected count, effective labels and a content identity. It copies labels and never
retains hidden/logit graphs. `selection.model_batch()` exposes the three existing
callback fields. The training loop resolves an entire accumulation window once,
before any forward/backward, and callbacks consume its effective labels. Direct
LossBackend calls reject unresolved `loss_mask`; prepare it explicitly first.
The compatibility `validate_batch` API still returns an integer, now the selected
count when given a supervision policy. Existing three-field custom callbacks keep
their `(loss_sum, selected_count)` signature.

The reused `_ChunkedCE` gathers selected predictor hidden states before vocabulary
projection and scatters their gradients through autograd. The window objective is
the sum of selected CE divided by the total selected targets, including variable
counts and partial windows. It never rescales by available/selected. Zero-sample
selection and invalid later batches fail before the window's first backward.

Only direct gradients at the final hidden boundary can be zero for unselected
positions. Earlier prefix embeddings/attention/recurrent state remain connected.
Layer, selective MLP and layer-group checkpointing retain their existing semantics.
No external cache or cross-forward recurrent state is used.

## Identity, metrics and recovery

The CLI's synthetic input provider owns `fixture_row_uniform_endpoints_v1` for
explicit smoke: row index r selects at most r+1 evenly spaced valid raw targets.
This is an engineering fixture, not a business-data policy. Mask content, tokens,
labels, attention, coordinate shape, effective lengths and algorithm are hashed
before trimming. `input-identity.json` and `selection.json` are bound to attempt
and checkpoint contracts and artifact receipts; the resolved config freezes the
supervision mode/K. Checkpoints retain the complete input selection identity.

Each update records `input_tokens`, `available_supervised_tokens`,
`supervised_tokens`, `supervision_fraction`, `selection_sha256`, plus existing
memory/time/gradient metrics. LossBackend telemetry records actual forward
`projected_tokens` and `ce_chunks`; backward projection recomputation is excluded
from those forward counts. Dense labels telemetry can project padded input slots.
`consumed_tokens` remains cumulative selected causal targets. Resume validates this
cursor using the same selection resolver, not original label counts.

Resume rejects changes to mode/K, mask content, algorithm, raw input coordinates,
schedule/source/dependencies and numerical policy. Legacy complete checkpoints
without supervision identity require explicit `warm_start`; no legacy contract is
silently upgraded. Adapter warm-start keeps base/target/LoRA identity, starts new
optimizer/scheduler/RNG and zero cursors, and permits a new supervision policy.

## Running and acceptance boundary

The `lora-sparse-labels.yaml`, `lora-sparse-explicit.yaml` and
`lora-sparse-uniform.yaml` presets default to dry-run. Dry-run imports no Torch,
loads no tokenizer or GPU, creates no attempt, and reports unknown actual selected
counts/hashes as null. Smoke freezes actual identities after encoding.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venvs/rl/bin/python -m pytest -q \
  tests/test_lora_supervision.py tests/test_lora_memory.py \
  tests/test_lora_config.py tests/test_lora_training.py \
  tests/test_lora_checkpoint.py tests/test_lora_entrypoint.py
```

The tiny fixtures use real Qwen3.5 GatedDeltaNet/full attention and real PEFT,
including pure linear attention, lengths 63/64/65/127/128, nonzero LoRA B, dense
selected references, hidden/all-LoRA gradients, full prefix dependency, actual
projection block counts, bf16/copy hooks/MLP/checkpoint combinations, selected
accumulation and separate-process exact resume. CPU fp32 gradient tolerance is
fixed per test at atol up to 7e-7 / rtol up to 5e-4; the combined CPU bf16 fixture
uses atol=0.005 / rtol=0.07. This proves the installed CPU reference paths only.

This stage runs no new real sparse GPU test or capacity benchmark. Historical dense
24K acceptance does not establish sparse GPU acceptance or 130K capacity. Switching
to FLA/causal-conv kernels requires new gradient/state-boundary/dtype/checkpoint
acceptance and a new environment identity.
