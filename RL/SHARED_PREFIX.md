# Shared-Prefix GRPO

Offline GRPO can reuse the identical prompt prefix inside each group while preserving the original group-weighted objective and LoRA gradient flow. Enable it explicitly:

```yaml
grpo:
  prefix_reuse:
    mode: boundary_grad
```

The default mode keeps complete-branch execution. Use a new attempt ID for every invocation. The BF16 example is `RL/experiments/grpo-shared-prefix.yaml`.

## Execution

At group start, the backend executes the common prompt once and retains one read-only source boundary. The boundary contains full-attention K/V state, linear-attention convolution/recurrent state, and the final normalized prompt hidden state in stable layer order.

Each branch receives independent `detach().clone()` leaves. Native Qwen decoder, rotary, convolution, gated-delta and output-projection operators consume those leaves through a functional cache facade. Source tensors are version-checked and never modified. The first completion target uses the final prompt hidden state; subsequent targets use the correct absolute positions and causal state. Completion length one is supported without a suffix backbone pass.

Branch execution keeps the existing reduction: token mean, branch mean, then group mean. Branch backward accumulates direct adapter gradients and detached boundary gradients. Boundary gradients are accumulated in FP32. After the last branch, one multi-output autograd backward sends the accumulated gradients into the original prefix graph. The group source, leaves, temporary hidden states, losses and branch graphs are then released. Adapter gradients remain available for the rest of the optimizer window.

The training loop has explicit `begin_group`, `after_branch_backward`, `finish_group` and `close_group` lifecycle hooks. Exceptions and nonfinite gradients clear the partial window and prevent an optimizer update. Prefix graphs, branch leaves and boundary-gradient buffers are transient and are not written to checkpoints.

## Checkpointing and metrics

The backend uses existing non-reentrant layer checkpointing. Boundary tensors are explicit checkpoint inputs, and recomputation creates a fresh functional cache facade. Cached suffix SDPA uses a symbolic lower-right causal bias, preserving absolute positions without constructing a dense attention mask.

Each update records prefix length, branch count, root and branch backward counts, source/leaf/gradient logical bytes and dtypes, prefix/suffix/state-copy/gradient timings, branch and group allocated-memory observations, and layer execution counts. Existing loss, ratio, K3, clipping, gradient norm and consumed-token metrics remain unchanged.

The feature is covered by CPU and tiny-Qwen tests for variable completion lengths, one-token targets, GDN chunk boundaries, 2/3/4/32 branches, unequal group windows, gradient accumulation, checkpoint recomputation, failure cleanup, unused boundary states and exact resume. The formal GPU path uses the same `dycon lora-train` service and BF16 model configuration.

## MIG devices

`runtime.gpu` accepts either a physical GPU index or one explicit MIG UUID:

```yaml
runtime:
  gpu: MIG-654b3ac2-156b-5fc6-aed8-4ac7394bd99c
```

MIG admission validates the selected instance, reads its capacity and current usage, sets `CUDA_VISIBLE_DEVICES` to that UUID, and records the instance identity in the training environment. MIG instances are selected individually and are not combined or repartitioned.

Run through the formal interfaces:

```bash
.venvs/runner/bin/dycon lora-train --training-config RL/experiments/grpo-shared-prefix.yaml
.venvs/rl/bin/dycon lora-train --training-config PATH_TO_TRAINING_CONFIG.yaml
.venvs/runner/bin/dycon artifact verify ATTEMPT_ID \
  --scope lora-attempt --storage-root ROOT
```


## Opt-in 8x16 execution

In [8x16 mode](DATA_PARALLEL_8X16.md), the prefix lifecycle belongs to one
execution half, while eligibility, retained/ranks, advantages and the reference
anchor belong to its complete 32-branch parent. Each half recomputes the prefix
in a different update. The objective uses token mean, 16-branch mean, and actual
window-parent mean; no half is re-ranked or re-centered. Prefix rows are drained
after each local update attempt and cleared on failure, independently of profiler
callbacks. Rows carry parent/half, native ordinals, round/window and rank.
Shrinking 32 suffixes to 16 does not halve prefix graphs or boundary buffers and
does not establish capacity for v10's longest prompts.
