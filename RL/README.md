# Qwen3.5-9B LoRA foundation

GRPO checkpoint resume can explicitly opt into `runtime.topology_policy:
preserve_updates` for bidirectional 2/4/8-GPU execution. It retains the source
logical windows, weights, frozen advantages, anchor, optimizer/scheduler and
committed cursor, while changing physical task assignment. Training computations
must not consume RNG. Use the same CLI, a new attempt ID and a dry-run first;
default `exact` restrictions remain unchanged. See [configuration and limits](TOPOLOGY_SWITCH.md)
and [validation evidence](TOPOLOGY_SWITCH_VALIDATION.md). CPU/Gloo tests passed;
real Qwen/NCCL cross-topology GPU acceptance remains pending.

Fixed checkpoint preference diagnostics are available through `dycon lora-probe`:
explicit checkpoint and fixed complete group IDs, JSON reports and optional baseline
comparison, with no TensorBoard or training-state writes. See [FIXED_PROBE.md](FIXED_PROBE.md).

Native streaming GRPO supports exact checkpoint resume through the same training CLI, retaining the original anchor and each rank's optimizer/scheduler/RNG. Unlike prepared cache reuse, streaming requires unconsumed native ledgers and raw probability sources to remain accessible. See [streaming recovery](STREAMING_RESUME.md) for complete recovery bundles, read-only historical compatibility, strict paths/topology and validation boundaries.

All optional startup audits and diagnostics default to off in `preflight`:
`verify_identity`, `audit_native_tokens`, `validate_dataset`, `verify_probabilities`,
`verify_checkpoint`, `check_resources`, `verify_model_state`, and `run_diagnostics`.
Fresh training prepares data, frozen advantages, source contracts and the shared
probability anchor through the existing training service. Streaming resume
instead inherits its verified frozen plan/anchor and reads only unconsumed payloads. It skips
the separate dataset pre-scan. Dry-run with defaults reports configuration and
identity preparation; enable checks explicitly for an audited dry-run. With model
content verification off, startup reuses an existing registered identity.
See [GRPO startup checks](GRPO.md#data-and-advantages) for each switch and artifact
preparation versus runtime requirements.

Shared-prefix GRPO, explicit MIG UUID selection and BF16 training configuration
are documented in [SHARED_PREFIX.md](SHARED_PREFIX.md). Existing complete-branch
training remains the default; `boundary_grad` is an explicit opt-in mode.

训练接口的配置、运行与代码示例见 **[训练接口使用指南](TRAINING.md)**：

| 使用情境 | 文档 |
|---|---|
| 2/4/8 卡双向 checkpoint 恢复、逻辑更新保持、RNG 限制 | [TOPOLOGY_SWITCH.md](TOPOLOGY_SWITCH.md) |
| 跨卡数 CPU/Gloo 验证、v10 八卡 dry-run 与待完成 GPU 验收 | [TOPOLOGY_SWITCH_VALIDATION.md](TOPOLOGY_SWITCH_VALIDATION.md) |
| Native streaming resume、旧 checkpoint 兼容、外部数据与路径/拓扑限制 | [STREAMING_RESUME.md](STREAMING_RESUME.md) |
| 单卡/双卡短 native 数据验收与 v10 只读验证范围 | [STREAMING_RESUME_VALIDATION.md](STREAMING_RESUME_VALIDATION.md) |
| CLI、完整 YAML、fresh/resume/warm_start、Python 服务与训练核心 | [TRAINING.md](TRAINING.md) |
| 外部 `loss_mask`、确定性 uniform、监督坐标与计数 | [SPARSE_SUPERVISION.md](SPARSE_SUPERVISION.md) |
| CE/MLP 分块、checkpoint policy、exact-length 与 offload | [LONG_CONTEXT.md](LONG_CONTEXT.md) |
| OOM 自适应卸载、整步重试、CPU 证据与待完成 GPU 验收 | [OOM_RECOVERY_VALIDATION.md](OOM_RECOVERY_VALIDATION.md) |
| 离线数据集、token cache 和 raw T=1 概率 stream/replay | [dataset/README.md](dataset/README.md) |
| 离线 GRPO、外部 shared anchor、一次 epoch 与精确恢复 | [GRPO.md](GRPO.md) |
| 训练进度、TensorBoard 实时 loss、冻结 advantage 查询 | [OBSERVABILITY.md](OBSERVABILITY.md) |
| 固定 group checkpoint 偏好评分、JSON 报告及历史改善对比 | [FIXED_PROBE.md](FIXED_PROBE.md) |
| 可观测性测试与 GPU fresh/resume 验收 | [OBSERVABILITY_VALIDATION.md](OBSERVABILITY_VALIDATION.md) |
| GRPO 原有实现与 GPU 验收 | [GRPO_VALIDATION.md](GRPO_VALIDATION.md) |
| 外部 anchor 配置、联调及 GPU fresh/pause/resume 验收 | [GRPO_BINDING_VALIDATION.md](GRPO_BINDING_VALIDATION.md) |
| 在线/离线中性 raw stream receipt、身份和数组合同 | [PROBABILITY_STREAM.md](PROBABILITY_STREAM.md) |
| 统一概率 schema 3、coverage plan、只补缺失分支和迁移 | [dataset/PROBABILITY_ARTIFACTS.md](dataset/PROBABILITY_ARTIFACTS.md) |
| 概率对齐 CPU 验收、可移植性和真实模型待验收范围 | [PROBABILITY_ALIGNMENT_VALIDATION.md](PROBABILITY_ALIGNMENT_VALIDATION.md) |
| Rollout 在线 raw 概率开关、sidecar、发布与验证范围 | [ONLINE-RAW-LOGPROBS-CN.md](../docs/ONLINE-RAW-LOGPROBS-CN.md) |

This module provides bf16 LoRA training in the original DyConGlobal checkout. It
supports synthetic causal CE and the versioned offline `offline_grpo_k3_v1`
objective. Offline GRPO supports synchronous whole-group data parallelism on 1, 2
or 4 GPUs by default, with an opt-in distinct-parent 8x16 mode on 8 GPUs;
`preserve_updates` additionally remaps either logical schedule across 2/4/8 GPUs.
See [DATA_PARALLEL.md](DATA_PARALLEL.md). Benchmark evaluation, vLLM
adapter serving and quantization remain separate.
The separate [dataset module](dataset/README.md) constructs offline native-token step groups
and exposes neutral raw T=1 online streams and offline native-vLLM replay artifacts.
Legacy processed artifacts remain diagnostic inputs.

Dataset operations use `.venvs/runner/bin/dycon rl-dataset`; the command manages its own
isolated `.venvs/latent-vllm` worker for inference. Dataset/replay YAMLs own explicit source
bindings, physical roots and replay resources. Raw replay uses a fixed T=1
pre-transform scoring contract, independently of root or historical sampling
temperature/top-p/top-k. Scoring checkpoint and dtype come from frozen sources. Dataset, cache, replay, canonical probability and coverage-plan artifacts live under
`ROOT/RL/{datasets,cache,replays,probabilities,probability-plans}/ID`
and use the unified `artifact show/verify` interface. See the [dataset workflow](dataset/README.md#workflow)
for build/check/materialize/replay commands and explicit recovery.

For mixed historical/new probability coverage, freeze `probability-plan`, bind it
to the existing replay selector, then publish `probability-build`. Reuse existing
online bytes and score only missing native traces; no missing branches means no
GPU worker. Single-source migrations use the same writer. The final artifact is
self-contained and preserves each branch's origin, source receipt and execution
cohort. Ordinary verification and prepared training resume do not require the original source NAS. Native streaming still requires its selected probability artifact and native ledgers.
See [the canonical workflow](dataset/PROBABILITY_ARTIFACTS.md).

The dataset Python loader returns groups, original target-coordinate masks and optional
`behavior_logprobs`. The versioned `offline_grpo_k3_v1` objective now consumes these
sealed groups through the same `lora-train` entrypoint. It uses raw T=1 ratios against
a frozen initial policy, token K3 on native completion targets, frozen advantages and
exactly one epoch. `grpo.reference_cache.source` defaults to `model_forward`;
`probability_artifact` imports a committed explicit `rl-probability` schema-3
ID/root/scope or a legacy raw `rl-replay` artifact, and
`dataset_probability` selects an explicit dataset stream. External sources require
`same_dtype` or `fp16_source_bf16_training`, complete eligible-branch coverage and
matching scoring checkpoint/tokenizer/adapter identities. Processed values and
nonfinite selected targets are rejected; import never silently recomputes a cache.
The generic loader's `allow_replay_estimates` option is separate from this strict
training-source contract.

See [offline GRPO](GRPO.md) for score semantics, group weighting, reference/cache
identity, one-epoch cursors, complete-checkpoint resume and acceptance commands.
Prepared indexed groups, exact artifact reuse, append-only schedule segments,
reference ownership and recovery are documented in [PREPARED_DATA.md](PREPARED_DATA.md).
Prepared/legacy cache manifests/provenance are checkpoint-local: their resume uses the frozen
parent cache without opening the external source. A warm-start imports only scores
from its actual initial adapter; fresh base scores require a proven zero-increment
LoRA initialization. Source selection, dtype compatibility and the shared cache hash
are bound in the training contract. See [external-anchor acceptance](GRPO_BINDING_VALIDATION.md)
for the same-dtype synthetic GPU result and remaining real FP16/BF16 acceptance.

Use `.venvs/rl`, independently locked by `locks/rl.in` / `locks/rl.txt`. On this server
the environment and its UV cache live under `/data3/wangtianyi/DyConGlobal/RL/`;
`.venvs/rl` links to that environment. The project is installed editable from
`/home/wangtianyi/DyConGlobal`. Existing serving/runner dependencies are not upgraded.

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon lora-train --training-config RL/experiments/lora-sparse-uniform.yaml
# The preset above defaults to dry-run. For actual synthetic training, copy a task
# YAML, set a fresh attempt_id and run.mode: smoke, then use .venvs/rl/bin/dycon.
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
batch is automatically truncated. The fixture configuration range is 8–262144; the
fixed prompt must fit. Lengths above 2048 require the long-context settings described
below. The schema limit does not establish device capacity, and successful capacity
at one length is not a guarantee for longer contexts.

Long-context settings and their causal/gradient contracts are documented in
[`LONG_CONTEXT.md`](LONG_CONTEXT.md). The additive memory policy supports chunked CE,
position-chunked MLPs, exact-length microbatches, Flash-only full attention, explicit
activation offload budgets and three checkpoint schedules. Lengths above 2048 require
chunked CE, exact length and Flash full attention. Run the supplied
`RL/experiments/lora-long-throughput.yaml` (throughput oriented) or
`RL/experiments/lora-long-memory.yaml` (memory oriented) with its attempt ID changed
and `run.mode: dry-run` before selecting a real fresh attempt. Dry-run reports estimates
and does not certify capacity.

Sparse target supervision (`labels`, `explicit_mask`, deterministic `uniform`) is
documented in [`SPARSE_SUPERVISION.md`](SPARSE_SUPERVISION.md). It selects original
causal targets before vocabulary projection while preserving full input histories.
The `lora-sparse-*.yaml` presets default to dry-run. New sparse acceptance covers CPU
tiny Qwen3.5/PEFT and runner dry-run; no new sparse GPU capacity is certified.

All four start modes use the same `dycon lora-train` command and
`RL.service.run(root_config, training_config)` Python service:

| `run.start.mode` | `run.start.checkpoint` | State behavior |
|---|---|---|
| `fresh` | `null` | New adapter, optimizer/scheduler/RNG and zero counters |
| `resume` | A committed `checkpoints/step-NNNNNN/` directory | Restore adapter, optimizer/scheduler, Python/NumPy/CPU/CUDA RNG, step and input/token cursor |
| `append_resume` | A prepared checkpoint plus a sealed extension manifest (unsupported for streaming) | Restore parent state and consume unchanged old segments before complete new groups; constant scheduler only |
| `warm_start` | A supported non-streaming final or step `adapter/` directory | Trainable source adapter, new optimizer/scheduler/RNG and zero counters |

Always choose a new attempt ID. Existing failed or partial directories are also
immutable and cannot be overwritten. A missing/corrupt/incompatible source fails;
there is no fallback to fresh training. Resume preserves the original **total**
`training.max_updates`, learning rate, complete schedule, seed, input identity,
model/tokenizer content identity, dependency versions and full LoRA targets.
Project source fingerprints remain audit metadata and do not by themselves
reject resume; all other frozen contracts remain enforced.
Only attempt metadata, device/output location, start reference and pause control
may differ; the frozen configurations retain those differences. A completed
schedule cannot be extended by exact resume. Supported non-streaming adapter
warm-starts begin a new stage; streaming checkpoints do not support warm-start
or append-resume. Warm-start may choose new training
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
optimizer/scheduler and update callbacks for CE. The CLI selects synthetic causal CE
when `algorithm` is absent, and `RL.training.train_grpo` via the same service when
`algorithm.name=offline_grpo_k3_v1` is configured. Labels use `-100` for prompt/padding;
shift happens once. CE accumulation weights by supervised token count, including
a final partial window; GRPO uses branch token → group branch → update group means.
Invalid batches, nonfinite losses/gradients and zero effective gradients fail before
an optimizer update. The optimizer contains only the trainable adapter parameters.
See the [Python API examples](TRAINING.md#4-python-服务与训练核心) for concrete batch,
callback and selected-loss calls. Low-level `train` does not publish attempts or
checkpoints automatically; `RL.service.run` manages the formal lifecycle.

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

Long-context acceptance and measured capacity are reported in
[`LONG_CONTEXT_VALIDATION.md`](LONG_CONTEXT_VALIDATION.md). The initial 2K–24K ladder
uses two updates per fresh attempt, then separate-process complete-checkpoint resume
and adapter reload. It records explicit source/kernels, temporary-logit block size,
peak memory, host/offload transfers, throughput and remaining device margin. Only the
last completed and fully verified lengths count as accepted capacity.

[Validation report](VALIDATION.md): 40 CPU checks, full real-GPU acceptance and 95 existing-function regression tests passed on 2026-11-10. The continuous/resumed final adapter difference was 0.0.

## RL observability

2026-11-14 OOM work adds allocator configuration, step-wide adaptive boundary
offload, one coordinated same-window retry before gradient collectives, and
per-rank memory events queried with `dycon lora-memory`. Existing streaming
checkpoint resume remains supported. CPU regressions and the v11 configuration
dry-run pass; GPU validation is deferred until GPU0/1 are free, with existing
workloads preserved. See [OOM recovery validation](OOM_RECOVERY_VALIDATION.md)
for exact evidence and limits; historical GPU results do not validate this path.

Use the runner CLI to inspect current and future attempts without loading a model:

```bash
.venvs/runner/bin/dycon lora-status --active
.venvs/runner/bin/dycon lora-status ATTEMPT_ID --storage-root ROOT
.venvs/runner/bin/dycon lora-tensorboard ATTEMPT_ID --storage-root ROOT --action start --port 16006
.venvs/runner/bin/dycon lora-advantages ATTEMPT_ID --storage-root ROOT --group-ordinal 0
```

Replace the placeholders with an attempt ID and physical storage root. Status
reports committed updates, inflight work, recoverable checkpoints and available
stage/heartbeat information. TensorBoard shows loss after complete optimizer
updates and is accessed through SSH forwarding; `--action stop` stops the owned
visualization processes. Advantage queries expose frozen group/branch values,
native ordering and provenance, with explicit indexing for legacy JSON.

Mutable events, heartbeats and indexes stay outside sealed training artifacts.
Existing training does not need to restart; its unrecorded stage detail remains
unknown. See [progress, curves and frozen advantages](OBSERVABILITY.md) for the
full workflow and [validation](OBSERVABILITY_VALIDATION.md) for tested behavior.


The opt-in [8-GPU distinct-parent 8x16 protocol](DATA_PARALLEL_8X16.md) shares the
existing training/preparation/checkpoint service. It preserves full-parent
scoring and executes two complementary halves in different updates. It remains
experimental pending Qwen3.5-9B GPU capacity, resume and measured throughput
acceptance; CPU/Gloo results do not establish a GPU speedup or training quality.

## 8×16 validation status (2026-11-14)

The explicit experimental [8-GPU distinct-parent protocol](DATA_PARALLEL_8X16.md) has passed CPU/Gloo regression: **426 passed, 12 skipped**, including real eight-process synchronization and tiny-Qwen prepared/streaming resume. See [validation details and pending acceptance](DATA_PARALLEL_8X16_VALIDATION.md). The skips do not count as GPU acceptance; real Qwen3.5-9B eight-GPU correctness, long-prefix capacity, throughput and quality remain pending.

Full merged checkpoints for rollout: `.venvs/rl/bin/dycon lora-export --checkpoint PATH --base-model PATH --output PATH`; see [LoRA serving export](EXPORT.md). Uses CPU only and registers the model for the existing run/serving interfaces.
