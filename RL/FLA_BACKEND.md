# BF16 FLA GDN backend

Select the BF16 FLA chunk operator explicitly in the existing training YAML:

```yaml
model:
  dtype: bfloat16
  gdn_backend: fla_chunk
```

Use the same `dycon lora-train --training-config YAML` entrypoint. The default
backend remains `reference_pytorch`. The FP32-input experiment is not supported
in the main project. BF16 Q/K/V are passed directly to FLA; the existing FP32
recurrent state, gate values and boundary-gradient accumulation retain their
original precision contracts.

Full language-policy training uses `dycon rl-train` with schema 2 and the same
explicit BF16 backend selection. It retains complete branches, fixed FSDP2 world
size, layer checkpointing, FP32 masters/moments/reduction and the <=2048 input
limit. Prefix reuse, CPU offload, OOM retry and context parallelism remain outside
that path. Full checkpoints/probes preserve the actual GDN identity and sealed
per-rank call evidence; see [full FSDP2](FULL_FSDP2.md).

## Environment

The base `.venvs/rl` environment uses `locks/rl.in` / `locks/rl.txt`.
Install the optional pinned BF16 FLA dependencies into that environment:

```bash
uv pip install --python .venvs/rl/bin/python --require-hashes --no-deps -r locks/rl-fla.txt
```

The adapter requires `flash-linear-attention==0.6.0`, `fla-core==0.6.0`,
`triton==3.7.1`, `einops==0.8.1`, and Transformers 5.17.0. It verifies the Qwen
and FLA source identities before training. An incompatible requested backend
fails explicitly. Reference selection unwraps the optional-package dispatch
and therefore continues to use the pinned PyTorch implementation even when
FLA is installed.

## Behavior and checkpoints

`RL/gdn_backend.py` installs the binding before model construction. Only the
GDN chunk operator changes; recurrent single-token execution and causal
convolution use reference implementations. The chunk adapter preserves Qwen's
gate/beta semantics, the single scale factor, chunk size 64, differentiable
initial/final state, and `[B,HV,K,V]` state layout. No state is detached.

Training artifacts record configured/resolved backend, versions, source hashes,
state semantics, and observed calls. Workers must agree on the backend identity.
Resume requires the same backend contract. Historical reference artifacts remain
readable; reference-to-FLA exact resume is not supported.

## Runtime checks

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venvs/rl/bin/python -m pytest -q tests/test_gdn_backend.py
DYCON_FLA_GPU_TEST=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venvs/rl/bin/python -m pytest -q tests/test_gdn_backend_gpu.py
```

The GPU suite checks BF16 outputs, finite gradients, initial-state gradient flow,
final-state-only backward, and input preservation. It does not compare numerical
results with another backend.

`RL/experiments/grpo-fla-smoke.yaml` is the configuration example. For a real
smoke run, copy it to a task-specific configuration, choose a fresh attempt ID
and output root, set `run.mode: smoke`, and use the formal training CLI from the
main project. Its short built-in GRPO fixture executes two optimizer updates;
this fixture is separate from production SWE-bench groups.

```bash
.venvs/rl/bin/python -m core.cli lora-train --training-config YOUR_TASK_CONFIG.yaml
```

On `10.249.188.230`, the integration check completed 104 CPU regression tests,
8 GPU runtime tests, and the real Qwen3.5-9B smoke attempt
`fla-bf16-main-20261117-a` with two committed updates (5 synthetic groups,
15 branches, 60 target tokens). The final step-2 checkpoint was saved and the
formal artifact verifier checked all 88 files successfully. The first-update
profiler confirmed FLA autograd forward/backward and CUDA kernels. Evidence and
the actual task configuration are under
`temp/fla-bf16-main-integration-20261117/` in the main project.

## Acceptance scope

For the earlier 2026-11-17 LoRA integration, the user explicitly selected runtime acceptance:
the backend must execute training; cross-backend numerical agreement is not a
merge requirement. Earlier isolated worktree audits did not pass the full-model
numerical thresholds. This integration does not claim numerical equivalence,
long-context throughput, or multi-GPU acceptance.

The subsequent full-policy FSDP2 acceptance uses the same runtime criterion.
Tiny real NCCL tests compare FLA worlds 1/2, unequal parent and branch counts,
tail windows, full parameters and FP32 moments; they also verify exact
pause/resume, rank-local failure, export, probe and independent HF reload.
The real 9B short fixture completes pause/resume, deep artifact verification,
BF16 HF export and a fixed three-branch probe. Independent Transformers reload
on GPU7 matches all 12 target logprobs exactly. These full-policy checks do not
establish cross-backend numerical equivalence or training quality.

Test checkpoint payloads and the HF export are deliberately deleted after their
consumers finish. The metadata archive and cleanup ledger are under
`temp/full-fla-fsdp2-20261117/`; removed checkpoints cannot be resumed or deeply
verified again. Keep fresh attempt IDs when reproducing the examples.
