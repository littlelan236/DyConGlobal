# Qwen3.5-9B LoRA foundation

This module provides a single GPU bf16 LoRA engineering training core in the original
DyConGlobal checkout. It does not implement RL, a reward, GRPO/PPO, a business dataset,
benchmark evaluation, vLLM adapter serving, quantization, or distributed training.

Use `.venvs/rl`, independently locked by `locks/rl.in` / `locks/rl.txt`. On this server
the environment and its UV cache live under `/data3/wangtianyi/DyConGlobal/RL/`;
`.venvs/rl` links to that environment. The project is installed editable from
`/home/wangtianyi/DyConGlobal`. Existing serving/runner dependencies are not upgraded.

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/rl/bin/dycon lora-train --training-config RL/experiments/lora-smoke.yaml
.venvs/runner/bin/dycon artifact verify ATTEMPT_ID \
  --scope lora-attempt --storage-root /data3/wangtianyi/DyConGlobal
```

The standalone training YAML is an explicit configuration exception. Paths resolve
against the project root selected by the normal CLI root configuration. Root rollout
`config.yaml` is read without importing its serving/model settings into training.
`--set` and `--run-id` are rejected. YAML rejects duplicate keys, unknown fields,
nonfinite numbers, invalid startup combinations and unsupported settings.

Set `run.mode: dry-run` in a new task configuration for deep model/tokenizer content
identity, path, source receipt and output-conflict validation without loading Torch,
occupying a GPU or creating a training attempt. It can publish the model content
fingerprint in the existing `runtime/model-identities/` registry. Exact target paths
are resolved only by an actual model load. `run.mode: smoke` uses the packaged,
handwritten synthetic fixture `RL/fixtures/causal-text-smoke-v1.json`; deterministic
repetition constructs the configured test length. It is engineering-generated token
evidence, not historical native-token evidence or a business training set. No user
batch is automatically truncated. The supported fixture length is 8–2048; the fixed
prompt must also fit. Successful capacity at one length is not a guarantee for longer
contexts.

All three start modes use the same `dycon lora-train` command and
`RL.service.run(root_config, training_config)` Python service:

| `run.start.mode` | `run.start.checkpoint` | State behavior |
|---|---|---|
| `fresh` | `null` | New adapter, optimizer/scheduler/RNG and zero counters |
| `resume` | A committed `checkpoints/step-NNNNNN/` directory | Restore adapter, optimizer/scheduler, Python/NumPy/CPU/CUDA RNG, step and input/token cursor |
| `warm_start` | This module's final or step `adapter/` directory | Trainable source adapter, new optimizer/scheduler/RNG and zero counters |

Always choose a new attempt ID. Existing failed or partial directories are also
immutable and cannot be overwritten. A missing/corrupt/incompatible source fails;
there is no fallback to fresh training. Resume preserves the original **total**
`training.max_updates`, learning rate, complete schedule, seed, input identity,
model/tokenizer content identity, source code/dependencies and full LoRA targets.
Only attempt metadata, device/output location, start reference and pause control
may differ; the frozen configurations retain those differences. A completed
schedule requires `warm_start` for more training. Warm-start may choose new training
hyperparameters and fixture length, but the base, targets and LoRA configuration
must match. Third-party/encoder/merged adapters are not accepted.

For a 2+2 resume check, use `training.max_updates: 4` throughout. First run with
`validation.stop_after_update: 2`, then choose a new ID, set `start.mode: resume`,
point at its committed `step-000002/`, and set `stop_after_update: null`. Pause
changes only execution duration, not the schedule. Source directories remain read-only.

The loader is Transformers `Qwen3_5ForConditionalGeneration`, retaining its causal
LM output head while using text inputs. LoRA targets every `nn.Linear` inside
`model.base_model.language_model`, using complete paths. It covers MLP, full and
linear attention. Vision, embeddings, output head and all original weights are
frozen. Backbone-resident extra modules are included; every target's actual execution,
gradient and update status is recorded. No MTP-name or layer-count filter is used.
The exact injected list and rank × (in + out) parameter total must reconcile.
Training uses SDPA, gradient checkpointing with non-reentrant autograd, and the
installed Transformers linear-attention implementation recorded in `environment.json`.
The current environment uses the reference PyTorch linear-attention and causal-conv1d
paths; optimized optional kernels are not installed. Deterministic Torch algorithms,
deterministic cuDNN without benchmarking, and a fixed cuBLAS workspace are enabled
and included in the resume contract. Acceptance tolerances are not relaxed after a failure.
GPU selection happens before importing the training stack. An occupied configured GPU
fails without switching cards, offloading or stopping other services.

`RL.training.train` accepts a model, batch iterable, controlled loss callback,
optimizer/scheduler and update callbacks. The current CLI uses only the engineering
causal CE in `RL.validation`. Labels use `-100` for prompt/padding; shift happens once.
Accumulation weights by supervised token count, including a final partial window.
Invalid batches, nonfinite losses/gradients and zero effective gradients fail before
an optimizer update. The optimizer contains only the trainable adapter parameters.

Outputs live in `runtime.output_root/RL/checkpoints/ATTEMPT_ID/`. Files include the
frozen source/resolved YAML, attempt/environment/model identity, complete target and
parameter inventories, per-update metrics, update-boundary checkpoints, standard
PEFT adapter export, validation, compact output probes and report. No base weights
or merged model are saved. Checkpoints publish a complete same-filesystem staging
directory by atomic rename, with the receipt committed last inside it. Restore uses
the restricted `torch.load(weights_only=True)` after receipt/hash verification.
Failures retain `attempt.json`, traceback and already committed checkpoints, without
a success receipt. `paused` is distinct from `completed`.

`artifact show/verify --scope lora-attempt` uses the shared CLI and root registry;
explicit `--storage-root` selects the physical root. Alternatively register a
`paths.artifact_roots.lora-attempt` / `paths.artifact_scan_roots` entry in the root
configuration when desired. LoRA path resolution remains local to this module;
the global `owned_path` policy is unchanged. Verification checks file integrity,
required outputs, source references and state, not benchmark quality or output
equivalence. Output equivalence is checked by model tests.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venvs/rl/bin/python -m pytest -q \
  tests/test_lora_config.py tests/test_lora_training.py \
  tests/test_lora_checkpoint.py tests/test_lora_entrypoint.py

DYCON_LORA_GPU_TEST=1 .venvs/rl/bin/python -m pytest -q -s \
  tests/test_lora_qwen35_integration.py
```

CPU tests use a real tiny Qwen3.5 with both attention types and real PEFT, without
downloads. GPU acceptance invokes the formal CLI in fresh, sequential processes:
512-token continuous 4 updates, 2-update pause + 2-update resume, independent adapter
reload + warm-start update, and a 1024-token capacity update. Its tolerance specification
is written before any update. Adapter atol=2e-5/rtol=2e-3, logits atol=0.125/rtol=0.01,
loss atol=0.05/rtol=0.01. These are bf16 engineering tolerances, not claims of bitwise
reproduction across devices or kernels. The test records content-bound receipts,
per-step memory/time/tokens/gradient norms, before/after frozen tensor samples,
per-projection gradients/updates and GPU memory after child exit. CPU tests additionally
compare every original base parameter exactly. Test configs, logs and reports are in
`temp/lora-foundation-20261110/acceptance-TIMESTAMP/`; a successful suite writes
`latest-acceptance.json`. CUDA unavailability or skipped GPU tests do not satisfy
real-model acceptance.

The inspected 9B model has 248 targets: 96 MLP, 32 full-attention and 120
linear-attention projections, giving 43,278,336 trainable parameters at rank 16.
Transformers instantiates 9,409,813,744 original parameters (including frozen vision).
The checkpoint contains 15 `mtp.*` states that the conditional-generation loader does
not instantiate; their complete names are recorded in the checkpoint inventory.
They are outside the actual text backbone and therefore are not LoRA targets. The
current backbone has no extra/MTP target modules. Stored checkpoint element counts
must not be substituted for these actual loaded/trainable counts.

[Validation report](VALIDATION.md): 40 CPU checks, full real-GPU acceptance and 95 existing-function regression tests passed on 2026-11-10. The continuous/resumed final adapter difference was 0.0.
