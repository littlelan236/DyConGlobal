# RL 训练手册

`.venvs/rl/bin/dycon rl-train --training-config YAML` 通过同一服务运行 LoRA 或全参 FSDP2。现有 LoRA 配置也可使用 `lora-train`。使用 `.venvs/runner/bin/dycon` 预检和查询，实际训练使用 RL 环境。

RL 的 KL 和 PPO 裁剪参数统一在顶层配置：

```yaml
objective:
  kl:
    computation: recompute  # legacy retains historical behavior
    beta: 0.02
    k3_cap: 1.0
  ppo:
    clip_epsilon: 0.1
```

`beta: 0.0` 关闭 KL；各训练模式的 PPO 概率分母保持原有定义。旧配置仍可读取，重复填写时参数必须一致。普通离线 DyCon 当前只接受 `clip_epsilon: 0.1`。在线 bootstrap 和离线 baseline-only 可以省略 `training.max_updates`，由 epoch 和任务数推导。baseline-only 可省略两路来源权重。

| 任务 | 入口 |
| --- | --- |
| 构建原生 token 数据和 raw T=1 概率 | [数据与概率手册](dataset/README.md) |
| 选择全参 FSDP2、FLA、多卡策略或内存后端 | [执行与资源手册](EXECUTION.md) |
| 选择监督、训练、恢复和诊断 | 本手册 |

所有可选 `preflight` 审计默认关闭，包括 `verify_identity`、`audit_native_tokens`、`validate_dataset`、`verify_probabilities`、`verify_checkpoint`、`check_resources`、`verify_model_state` 和 `run_diagnostics`。按任务需要显式启用。基础 dry-run 校验配置和身份准备。流式恢复继承冻结计划与 anchor，仅按游标访问后续输入。关闭模型内容校验时，启动使用已登记身份。

## 详细参考

- [部署与启动](#部署与启动) · [安装、依赖与主机路径](#安装依赖与主机路径)、[配置预检和首次训练](#配置预检和首次训练)、[LoRA 训练与产物身份](#lora-训练与产物身份)
- [训练目标](#训练目标) · [稀疏因果交叉熵](#稀疏因果交叉熵)、[离线 GRPO 分组、共享锚点与目标](#离线-grpo-分组共享锚点与目标)
- [训练恢复](#训练恢复) · [启停、检查点与加载模式](#启停检查点与加载模式)、[恢复时的显式配置修订](#恢复时的显式配置修订)、[原生流式恢复](#原生流式恢复)
- [接口与诊断](#接口与诊断) · [监督位置和 Python API](#监督位置和-python-api)、[实时进度、TensorBoard 与已冻结 advantage](#实时进度tensorboard-与已冻结-advantage)、[固定 group 偏好探针](#固定-group-偏好探针)、[Token K3 诊断](#token-k3-诊断)、[LoRA 权重导出](#lora-权重导出)

## 部署与启动

### 安装、依赖与主机路径

Environment installation does not start a run. Use the same checkout and formal
`dycon lora-train --training-config YAML` entrypoint for training.

#### Install the environments

On Linux, install Git, `flock` (util-linux), and uv first. The validated deployment
used uv 0.13.8. Python is pinned by the repository's `.python-version` to 3.12.15;
uv can download that interpreter if it is unavailable. Configure `HTTPS_PROXY`
and `HTTP_PROXY` in the shell when needed; keep credentials and host proxy
addresses outside tracked configuration.

From the checkout root:

```bash
bash runtime/scripts/setup_rl.sh --with-runner
```

This installs runner, then RL, using their checked-in locks. To install just one:

```bash
bash runtime/scripts/setup_runner.sh
bash runtime/scripts/setup_rl.sh
```

The scripts work from any checkout path. Locks contain dependency packages;
the current checkout is registered separately with `--no-deps -e .` after the
locked build tools are installed. Root `egg-info` metadata from earlier editable installs is refreshed
through the locked build backend so it cannot shadow the current dependencies.
Existing `.venvs/runner` / `.venvs/rl` links to external storage are supported.
An existing environment with a different
Python version is rejected rather than replaced. For an exact clean deployment,
start with new environment directories; install does not remove unrelated
packages from an existing environment.

Installers share a per-user setup lock and run sequentially to avoid duplicate
downloads and cache-lock timeouts. Defaults are `UV_LOCK_TIMEOUT=900` and
`UV_HTTP_TIMEOUT=90`; environment variables can override both. `UV_BIN` selects
another uv executable. No proxy or cache location is hard-coded into locks.

RL installation enforces the hashes in `locks/rl.txt`. Both environments finish
with `uv pip check`, an ijson import and training CLI help. PEFT is pinned to
0.21.2 in both `locks/rl.in` and the package's RL extra, matching the validated
training environment. `ijson==3.5.1` is included for metric reads. TensorBoard
and Matplotlib retain their separate analysis/runner ownership; FLA and
causal-conv kernels are not installed implicitly.

#### Update the RL lock without upgrading the training stack

Edit `locks/rl.in` and matching package extras in `pyproject.toml`, then compile
with the existing lock as a version constraint:

```bash
uv pip compile --python .venvs/rl/bin/python --generate-hashes \
  --constraint locks/rl.txt locks/rl.in --output-file locks/rl.txt
git diff -- locks/rl.in locks/rl.txt pyproject.toml
```

Review package versions and hashes before applying the result. Do not add
`--upgrade` for deployment fixes. Deliberate dependency upgrades require a
separate compatibility review and revised constraints. The runner lock must not
contain an editable checkout or a host's absolute project path.

#### Verify a new installation

Run the relevant existing CPU regressions in the freshly installed environments:

```bash
CUDA_VISIBLE_DEVICES='' .venvs/rl/bin/python -m pytest -q \
  tests/test_grpo_objective.py tests/test_rl_training_signals.py \
  tests/test_grpo_token_diagnostics.py
CUDA_VISIBLE_DEVICES='' .venvs/runner/bin/python -m pytest -q \
  tests/test_capture_background_recovery.py tests/test_capture_quota_writer_guard.py
.venvs/runner/bin/dycon config-check
```

These are CPU tests with temporary synthetic fixtures, not the production RL
run. `pip check` alone cannot detect undeclared imports; the metric-reader tests
exercise the ijson dependency. Configuration schema validation and CLI help do
not establish model/data availability or GPU training correctness.

#### Prepare host paths before training

Use the project's existing root `config.yaml` and explicit task training YAML;
do not add implicit config overlays. Keep host edits and dated execution YAMLs
local. Resolve and verify the training YAML's model path, dataset/replay roots,
native-ledger absolute paths and writable output root. A copied dataset receipt
does not by itself establish that every referenced native ledger is readable.

On 10.249.189.84 the copied input mirror initially lives at
`/home/wangtianyi/rl-migration-inputs/VM/wangtianyi/DyConGlobal`. Source artifacts
may retain `/VM/...` absolute references, so changing only dataset/replay roots
does not relocate all inputs. Arrange the required mount/path mapping with the
administrator, or use an explicitly supported artifact relocation workflow;
do not rewrite frozen receipts or create mappings over existing directories.

For a new training YAML, inspect current `lora-train --help`, enable the
documented data/resource preflight checks and run the formal dry-run first.
Confirm the checkpoint/start mode, fresh attempt ID, GPU availability and
physical output location before an explicitly authorized real launch. Dry-run
does not prove a successful GPU optimizer update. See [training](#配置预检和首次训练),
[GRPO](#离线-grpo-分组共享锚点与目标) and [streaming resume](#原生流式恢复).

### 配置预检和首次训练

#### 选择配置并先执行 dry-run

使用独立 attempt ID 和 `run.mode: dry-run` 核对配置。设置 `run.mode: smoke` 会执行训练。长上下文任务应先用代表性的 native 长度核验显存峰值和实际卸载。配置示例见[内存策略](EXECUTION.md#显存与主存)。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon lora-train --help

# 三种监督方式的预设均默认 dry-run，可直接用于配置预检。
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-sparse-labels.yaml
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-sparse-explicit.yaml
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-sparse-uniform.yaml
```

实际使用时复制一个配置到 `RL/experiments/` 下，填写新的 `run.attempt_id`、本机模型
路径和输出根目录，再执行同一命令。`--training-config` 是训练子命令的配置参数；
不支持 `--set`、`--run-id` 或命令行覆盖训练超参数。路径相对项目根目录解析。

以下是完整可解析的短序列配置，可保存为 `RL/experiments/lora-my-task.yaml`。
模型和输出路径为当前服务器约定，换环境时需要修改：

```yaml
schema_version: 1
run:
  attempt_id: lora-my-task-001
  mode: dry-run
  seed: 42
  start:
    mode: fresh
    checkpoint: null
model:
  path: /VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B
  dtype: bfloat16
  attention_backend: sdpa
  gradient_checkpointing: true
lora:
  rank: 16
  alpha: 32
  dropout: 0.0
  target_scope: text_backbone_all_linear
runtime:
  gpu: 0
  output_root: /data3/wangtianyi/DyConGlobal
training:
  max_updates: 4
  gradient_accumulation_steps: 2
  learning_rate: 0.0001
  weight_decay: 0.0
  grad_clip: 1.0
  scheduler: constant
  checkpoint_every_updates: 2
  checkpoint_policy: layer
  checkpoint_group_layers: 4
  memory:
    loss_chunk_tokens: 16
    mlp_chunk_tokens: 32
    exact_length: true
    attention_policy: auto
    activation_offload: none
    host_budget_gib: 64.0
    profile_first_update: false
  supervision:
    mode: uniform
    max_targets_per_sample: 8
validation:
  fixture_id: causal-text-smoke-v1
  max_sequence_length: 128
  stop_after_update: null
```

```bash
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-my-task.yaml
```

dry-run 返回 `status: validated`、解析后的配置、监督规则和显存/主存估算；不加载
Torch、tokenizer 或模型，不占 GPU，不创建训练 attempt。它会读取模型/tokenizer
文件计算内容身份，可能更新 `runtime/model-identities/`。由于尚未编码输入，实际
selected count 和 selection hash 为 null。dry-run 通过不代表显存容量已经验收。

#### 1.1 为 GRPO 选择冻结概率来源

复制 `RL/experiments/grpo-offline-k3-task.yaml`，填写已封存 dataset 的显式 ID/root、
新 attempt ID 和空闲物理 GPU；`training.max_updates` 必须等于全部 eligible groups
的一次 epoch 更新数。下表中的配置位于 `grpo.reference_cache`：

| `source` | 必需字段 | 用途 |
|---|---|---|
| `model_forward`，缺省 | 无外部字段 | 对训练起点模型前向构建 cache，保留旧 YAML 行为 |
| `probability_artifact` | `artifact.scope: rl-probability`（或旧 raw `rl-replay`）、绝对 `root`、`id`、`dtype_compatibility` | 导入统一在线/离线/混合 raw 产物，兼容旧离线来源 |
| `dataset_probability` | 明确 `stream`、`dtype_compatibility` | 导入 dataset receipt 绑定的在线 raw stream |

例如在 task YAML 的现有 `grpo` 中加入：

```yaml
reference_cache:
  source: probability_artifact
  artifact:
    scope: rl-probability
    root: /data3/wangtianyi/DyConGlobal
    id: REPLACE_WITH_COMMITTED_RAW_ID
  dtype_compatibility: fp16_source_bf16_training
  initial_diagnostic_branches: 2
```

统一产物由 `rl-dataset probability-plan`、既有 missing-only `replay-logprobs`
和 `probability-build` 在训练前生成；online-only/offline-only 导出也使用同一 build。
先按[概率对齐流程](dataset/README.md#coverage-plan缺失补评与可携带封存)准备封存 ID，训练只绑定最终
产物，不按 branch 拼接来源或启动补评。旧 raw `rl-replay` 可继续直接导入。
schema 3 将 scoring function 和实际 execution cohorts 分开记录，合并要求采集
compute dtype 相同；训练的 FP16→BF16 选项不授权来源内部混合 dtype。
所有 eligible branches 必须有限且齐全；partial 产物只有缺失部分完全位于既定
训练资格之外时可用，不会因缺概率静默删组。

改用在线来源时，替换整个对象为
`{source: dataset_probability, stream: raw_t1_output_tokens, dtype_compatibility: fp16_source_bf16_training}`。
这些是合并到现有 task YAML 的片段，产物 ID 为占位符。随后按顺序执行：

```bash
.venvs/runner/bin/dycon lora-train --training-config RL/experiments/MY_GRPO_TASK.yaml
# 确认 dry-run 通过后，使用新 attempt ID，并将同一配置的 run.mode 改为 smoke。
.venvs/rl/bin/dycon lora-train --training-config RL/experiments/MY_GRPO_TASK.yaml
.venvs/rl/bin/dycon artifact verify NEW_ATTEMPT_ID \
  --scope lora-attempt --storage-root /data3/wangtianyi/DyConGlobal
```

外部 dry-run 验证 source receipt、raw 语义、身份和 eligible branch 数值；不加载
Torch/模型。实际训练会进一步检查 fresh LoRA 的零增量或 warm-start adapter 的
实际 tensor identity。外部来源缺失或错误会失败，不回退前向、不删除训练分支。
`same_dtype` 要求采集和训练 dtype 相同；`fp16_source_bf16_training` 显式接受
float16 采集、bfloat16 训练，但不放宽 checkpoint/tokenizer/adapter 校验。

ratio 分母与 K3 共用一份 detached float32 anchor，current 由实时模型计算。
可选 `initial_diagnostic_branches` 默认 0；正数只诊断少量初始 current 的 delta、
ratio 分位数和 K3，不修改 anchor，也不要求跨精度初始 ratio=1/K3=0。
raw T=1 评分不等于原始 T=0.6/top-p/top-k 的 processed 行为概率。

完整合同见 [GRPO 训练合同](#离线-grpo-分组共享锚点与目标)，源接口见
[概率流合同](dataset/README.md#raw-概率流的身份与读取接口)，203 项 CPU/联调及真实 Qwen
同精度合成 GPU 验收见 [训练验证范围](#验证范围)。
真实 FP16 collector/BF16 历史数据小样本验收仍需指定封存来源。
本次 schema 3 对齐的 348 项 CPU 联合回归及独立验收边界见
[概率验证范围](dataset/README.md#验证范围)。

### LoRA 训练与产物身份

This module provides bf16 LoRA training in the original DyConGlobal checkout. It
supports synthetic causal CE and the versioned offline `offline_grpo_k3_v1`
objective. Offline GRPO supports synchronous whole-group data parallelism on 1, 2
or 4 GPUs by default, with an opt-in distinct-parent 8x16 mode on 8 GPUs;
`preserve_updates` additionally remaps either logical schedule across 2/4/8 GPUs.
See [完整父组数据并行](EXECUTION.md#完整父组数据并行). Benchmark evaluation, vLLM
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
and use the unified `artifact show/verify` interface. See the [dataset workflow](dataset/README.md#数据集创建物化和-replay)
for build/check/materialize/replay commands and explicit recovery.

For mixed historical/new probability coverage, freeze `probability-plan`, bind it
to the existing replay selector, then publish `probability-build`. Reuse existing
online bytes and score only missing native traces; no missing branches means no
GPU worker. Single-source migrations use the same writer. The final artifact is
self-contained and preserves each branch's origin, source receipt and execution
cohort. Ordinary verification and prepared training resume do not require the original source NAS. Native streaming still requires its selected probability artifact and native ledgers.
See [the canonical workflow](dataset/README.md#coverage-plan缺失补评与可携带封存).

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

See [offline GRPO](#离线-grpo-分组共享锚点与目标) for score semantics, group weighting, reference/cache
identity, one-epoch cursors, complete-checkpoint resume and acceptance commands.
Prepared indexed groups, exact artifact reuse, append-only schedule segments,
reference ownership and recovery are documented in [准备不可变输入](dataset/README.md#准备不可变输入与追加续训).
Prepared/legacy cache manifests/provenance are checkpoint-local: their resume uses the frozen
parent cache without opening the external source. A warm-start imports only scores
from its actual initial adapter; fresh base scores require a proven zero-increment
LoRA initialization. Source selection, dtype compatibility and the shared cache hash
are bound in the training contract. See [external-anchor acceptance](#验证范围)
for the same-dtype synthetic GPU result and remaining real FP16/BF16 acceptance.

For a new server or checkout, use the maintained [environment setup and deployment checks](#安装依赖与主机路径).

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
[`长上下文执行`](EXECUTION.md#长上下文和激活卸载). The additive memory policy supports chunked CE,
position-chunked MLPs, exact-length microbatches, Flash-only full attention, explicit
activation offload budgets and three checkpoint schedules. Lengths above 2048 require
chunked CE, exact length and Flash full attention. Run the supplied
`RL/experiments/lora-long-throughput.yaml` (throughput oriented) or
`RL/experiments/lora-long-memory.yaml` (memory oriented) with its attempt ID changed
and `run.mode: dry-run` before selecting a real fresh attempt. Dry-run reports estimates
and does not certify capacity.

Sparse target supervision (`labels`, `explicit_mask`, deterministic `uniform`) is
documented in [`稀疏监督`](#稀疏因果交叉熵). It selects original
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
See the [Python API examples](#python-服务与训练核心) for concrete batch,
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
[`执行验证范围`](EXECUTION.md#验证范围). The initial 2K–24K ladder
uses two updates per fresh attempt, then separate-process complete-checkpoint resume
and adapter reload. It records explicit source/kernels, temporary-logit block size,
peak memory, host/offload transfers, throughput and remaining device margin. Only the
last completed and fully verified lengths count as accepted capacity.

[Validation report](#验证范围): 40 CPU checks, full real-GPU acceptance and 95 existing-function regression tests passed on 2026-11-10. The continuous/resumed final adapter difference was 0.0.

## 训练目标

### 稀疏因果交叉熵

完整 CLI/YAML、训练恢复和可运行函数示例见[训练接口使用指南](#配置预检和首次训练)。
本文保留稀疏监督的坐标、算法与 loss/恢复契约。

The formal entry is `.venvs/rl/bin/dycon lora-train --training-config YAML`.
This adds supervision selection to the existing single-device training core.
Backbone inputs, causal attention, differentiable recurrent/convolution history,
all-linear text LoRA targets, and frozen embeddings/base/output head are preserved.
There is no KL/reference/reward/RL objective or distributed implementation here.

#### Configuration and coordinates

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

#### Python API and loss contract

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

#### Identity, metrics and recovery

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

#### Running and acceptance boundary

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

### 离线 GRPO 分组、共享锚点与目标

Native step streaming supports complete-checkpoint exact resume on one GPU or the original data-parallel world size. It inherits the frozen plan, advantages, original reference anchor and committed cursor; it still reads unconsumed external native/probability inputs. Mandatory recovery/payload identity checks apply independently of optional historical audits. See [streaming recovery](#原生流式恢复) for the YAML workflow, historical v10 compatibility and boundaries. The local-cache-only resume descriptions below apply to prepared/legacy cache paths.

Shared-prefix execution and explicit MIG selection are documented in
[共享前缀执行](EXECUTION.md#共享前缀执行). Enable it with
`grpo.prefix_reuse.mode: boundary_grad`; the default complete-branch backend is
unchanged. The backend runs the common prompt once per group, executes branches
with independent boundary copies, accumulates boundary gradients and performs
one prefix backward at group end. It uses the existing BF16 GRPO training path,
checkpointing, checkpoint/resume and artifact verification contracts.

Canonical online/offline unions are supported through `reference_cache.artifact.scope: rl-probability`.
See [probability alignment, missing-only backfill and migration](dataset/README.md#coverage-plan缺失补评与可携带封存).
Prepared/legacy-cache import freezes the final receipt and execution cohorts
in the shared cache; its resume uses that checkpoint-local cache. Native
streaming binds the same immutable source identity and reads remaining values
from the selected artifact. Neither resume path merges or scores sources.
Legacy `rl-replay` and `dataset_probability` remain supported.

Use the existing `dycon lora-train --training-config YAML` service. The algorithm is
`offline_grpo_k3_v1`; no rollout, resampling or new launcher is involved. Training
keeps Qwen3.5-9B bf16 and all-text-linear LoRA, with single-GPU or existing
[1/2/4-GPU whole-group or opt-in 8-GPU distinct-parent 8x16 execution](EXECUTION.md#完整父组数据并行).

New GRPO prepared schedules default to deterministic `prefix_length_pool128_v1`:
shuffle eligible complete groups by seed, sort each 128-group candidate pool by
prompt length, make complete optimizer windows from adjacent groups, and shuffle
the windows. This balances four-group windows for four-GPU, one-group-per-card
execution while preserving each native group and all branches. Existing sealed
schedules and exact resumes keep their original order. Schedule hashes, append
behavior and the explicit legacy option are documented in
[prepared data](dataset/README.md#准备不可变输入与追加续训).

#### Data and advantages

New attempts may bind an immutable indexed prepared artifact. `resume` verifies
its manifest, segment prefix and consumed ID hash before loading only the current
window; it does not rescan the native ledger or probability source. Explicit
`append_resume` uses a strict schedule-prefix extension and a new reference
manifest for complete new groups. See [prepared data operations](dataset/README.md#准备不可变输入与追加续训).

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

##### Consensus branches in the best-scoring half

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

Set `grpo.advantage_policy: consensus_top_half_binary_v1` with the same
top-half settings to keep the native step groups, branch order, eligibility and
positive membership above while mapping each positive advantage to `+1` and
each negative advantage to `-1`. The group sum need not be zero. This policy
uses the same source dataset and reference probabilities and freezes a new
training plan under its own policy version.

##### 同 action 组内固定 top-half

使用 [grpo-same-action-top-half.yaml](experiments/grpo-same-action-top-half.yaml) 或在训练 YAML 中设置下面的策略。

```yaml
grpo:
  advantage_policy: same_action_top_half_v1
  top_k: null
  top_k_fraction: 0.5
  boundary_tie_sigma: 0.0
```

每个原始 decision 内按 dataset 中的 canonical `action_hash` 分组，历史 `retained` 和 `selected` 标记不影响组成员。组内按 `historical_score` 升序排列，取前 `ceil(n/2)` 条为 `+1`，其余为 `-1`。部分同分按原生分支顺序处理。单分支、整组同分或分数无效的组跳过。原始 step 必须具备完整的原生证据和 action 身份。

源 `rl-dataset` 和覆盖所需分支的概率产物可直接复用。启用策略后由训练入口生成新的 prepared 或流式计划，也可用 `dycon lora-prepare` 事先封存 prepared。每个 optimizer update 包含最多 `grpo.group_batch_size` 个 action 组，`training.max_updates` 按一个 epoch 的新组数计算。窗口有 K 个组、某组有 N 条分支时，该组每条分支的权重为 `1/(K*N)`。奇数组的 advantage 和为 1。

LoRA 可用 `runtime.topology_policy: preserve_updates` 在 2/4/8 卡执行，并保留逻辑更新窗口。全参 FSDP2 使用 prepared 和固定卡数恢复。两条路径均拒绝 `distinct_parent_two_pass_v1`，该策略也不接受 `append_resume`。从旧策略切换时可新开训练或使用显式 warm start。精确 resume 使用已冻结的新策略组、优势和顺序。可通过 `dycon lora-advantages` 查看 LoRA 训练组的原 decision ID 和 action hash。

#### Configuration and one epoch

The task and smoke templates are:

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

#### Probability objective and reference

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
See [streaming checkpoint layout and legacy compatibility](#原生流式恢复).

##### Importing an external shared anchor

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
using [coverage planning and canonical merge](dataset/README.md#coverage-plan缺失补评与可携带封存).
Legacy raw replay artifacts remain directly readable with `scope: rl-replay`.
The canonical importer records the final receipt and every execution cohort in
local provenance; it does not follow per-branch source paths. The producer must
use one scoring function and matching collector compute dtype, with exact allowed
cohort hashes for multiple runtimes. The training dtype policy cannot relax those
merge constraints. A partial artifact is usable only if all frozen training-eligible
branches are covered and finite; probability coverage never changes eligibility.

For a dataset-bound online stream, first enable `capture.logprobs.enabled` for a new rollout and build the dataset through the existing `rl-dataset build` entrypoint; see [online raw collection](../docs/CAPTURE-CN.md#在线-raw-t1-采集与-sidecar). Verify raw coverage separately from token `data_ready`: unavailable or failed sidecars cannot satisfy a selected eligible branch, and an explicitly selected stream never silently falls back to model-forward. This collector currently records unsteered base-model scores without an adapter; training performs fresh/warm-start compatibility checks. Replace the entire `reference_cache` object:

```yaml
reference_cache:
  source: dataset_probability
  stream: raw_t1_output_tokens
  dtype_compatibility: fp16_source_bf16_training
  initial_diagnostic_branches: 2
```

The neutral reader is `RL.dataset.probability.open_probability_stream`. Its source
schema is documented in [概率流合同](dataset/README.md#raw-概率流的身份与读取接口). It shares
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

#### Checkpoints and metrics

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


#### Per-token K3 hard cap

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

#### Per-token extreme-ratio guard

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

### Offline baseline task reward normalization

Offline baseline-only and mixed DyCon + baseline training default to
`offline_baseline.advantage_mode: direct`. To enable `task_grpo`, prepare the
specific dataset once, then set `offline_baseline.advantage_mode: task_grpo`
and point `offline_baseline.dataset` to that directory.

```bash
.venvs/runner/bin/dycon rl-prepare-baseline-stats --dataset PATH
```

The command writes `task-reward-stats.json` beside the dataset manifest. For
each task, every trajectory in that dataset contributes its reward once,
independently of its run, length, or whether it is sampled in an update.
The policy advantage is `(reward - task_mean) / (task_population_std + 1e-8)`.
Tasks whose dataset rewards are all identical are excluded from baseline
sampling and KL. For baseline-only training, set `training.max_updates` to
`epochs * ceil(nonzero_variance_tasks / tasks_per_update)`. Mixed training
retains the DyCon schedule.

New imports store the raw reward in their manifests. Existing data needs a
one-time shard scan. Exact resume binds the dataset snapshot and statistics;
use a new attempt or an adapter warm start when changing mode or dataset.

## 训练恢复

### 启停、检查点与加载模式

#### 开始、暂停与恢复

要实际执行该配置的 synthetic 训练，将 YAML 的 `run.mode` 改为 `smoke`，再运行：

```bash
.venvs/rl/bin/dycon lora-train \
  --training-config RL/experiments/lora-my-task.yaml
```

训练会检查 YAML 指定的物理 GPU。卡被占用时失败，不自动换卡。不要使用环境变量
替代 `runtime.gpu` 的物理卡选择。重复调用必须使用新的 attempt ID，包括前一次
失败、OOM 或仅留下部分目录的情况；已有目录不会被覆盖。

`run.mode` 决定预检/执行，`run.start.mode` 决定训练状态来源，两者含义不同：

| `run.start.mode` | `checkpoint` 指向 | 状态与使用情境 |
|---|---|---|
| `fresh` | `null` | 新 adapter，optimizer/scheduler/RNG 和计数从头开始 |
| `resume` | 已提交的 `checkpoints/step-NNNNNN/` | 恢复完整训练状态，按父 contract 继续原 prepared/legacy/streaming 路径 |
| `append_resume` | prepared checkpoint 与封存 extension | 保留原 segment 前缀，仅追加完整新组；streaming 不支持 |
| `warm_start` | 支持的非 streaming 来源导出的 `adapter/` | 加载 adapter，开始新的训练阶段并重新计数 |

例如先暂停在第 2 次更新、随后完成总共 4 次更新：

1. 配置 A 保持 `training.max_updates: 4`，设置 `run.start.mode: fresh`、
   `checkpoint: null`、`validation.stop_after_update: 2`，执行到状态 `paused`。
2. 复制配置 A 为 B；只更换 attempt ID，设置下面的启动字段和
   `validation.stop_after_update: null`。其余 schedule、输入、监督模式/K、seed 等保持。

```yaml
run:
  attempt_id: lora-my-task-resume-001
  mode: smoke
  seed: 42
  start:
    mode: resume
    checkpoint: /data3/wangtianyi/DyConGlobal/RL/checkpoints/lora-my-task-001/checkpoints/step-000002
```

上面是替换 `run` 的配置片段，需要与完整配置其余部分一起使用。对 B 先做 dry-run，
再使用另一新 attempt ID、改回 smoke 执行同一 CLI。resume 的 `max_updates`
仍为 **4**，不能写成剩余的 2。`run.mode: smoke` 是项目的实际执行模式名；
GRPO 输入由配置的数据集决定，不会因此切换为 CE synthetic fixture。

GRPO 的数据执行路径由启动方式与父 checkpoint contract 共同决定：

| 条件 | 数据路径与约束 |
|---|---|
| fresh + async=true + prepared_data=null | native streaming；显式绑定外部 raw 概率源 |
| resume + 父 streaming contract | streaming；必须 async=true、prepared_data=null，继承原 plan/anchor/cursor |
| resume + 父 prepared contract | 保持 prepared，async=true 也不会转换路径 |
| resume + 父无 prepared/streaming contract | 保持旧非 streaming 恢复路径 |

Streaming checkpoint 不接受 warm_start/append_resume，也不支持改变 world size
或自动重写 dataset/source 绝对路径。新 checkpoint 内保存完整恢复元数据；
旧 streaming checkpoint 只从自身固定父 attempt 读取辅助证据，父 attempt
可以是 failed 且没有顶层 receipt。所有格式均以已提交 checkpoint 为恢复点，
不以预取进度或 checkpoint 之后的 metrics 代替。新格式损坏不会回退旧路径。

Runner dry-run 校验元数据而不导入 Torch；streaming 的实际 anchor tensor
验证、各 rank 状态/RNG 恢复在训练环境完成后才允许 forward。保持 CUDA
可见设备数与保存时一致。完整配置和 CLI 操作见 [流式恢复](#原生流式恢复)，
已通过的真实短数据 GPU 测试与未运行的 v10 更新见
[验收记录](#验证范围)。

对于支持的非 streaming adapter 来源，如果要改变监督方式、学习率或训练总步数，
使用新 attempt 和 `warm_start`，例如：

```yaml
run:
  attempt_id: lora-my-task-next-stage-001
  mode: smoke
  seed: 42
  start:
    mode: warm_start
    checkpoint: /data3/wangtianyi/DyConGlobal/RL/checkpoints/lora-my-task-001/adapter
```

warm-start 的 base/tokenizer、LoRA 配置和目标列表必须兼容；允许重新配置 schedule、
输入长度和监督策略。resume 还严格绑定输入内容、mask、选择算法、依赖和
数值策略。缺少新监督 contract 的历史 checkpoint 需显式 warm-start。不存在自动选取
latest、损坏后改为 fresh 或第三方 adapter 兜底。

Prepared/legacy cache 路径的 GRPO resume 使用父 checkpoint 的冻结 cache、source provenance 和角色绑定，
即使外部源离线也不重新导入；source/dtype 规则变化会明确拒绝。源码指纹作为历史
审计元数据保留，但不再作为 checkpoint/准备产物的代码版本兼容门槛。
warm-start 重置 epoch 和 cache：选择外部源时必须匹配该起点 adapter；需要重新
前向时显式选择 `source: model_forward`。完成 epoch 后不能用 resume 增加更新数。

### 恢复时的显式配置修订

Use the same formal `dycon lora-train --training-config YAML` entrypoint.
`run.start.config_migration: null` retains the existing exact-resume admission.
An explicit revision is bound to both source and target schedule hashes:

```yaml
run:
  start:
    mode: resume
    checkpoint: /ABSOLUTE/ATTEMPT/checkpoints/step-NNNNNN
    config_migration:
      source_schedule_sha256: SOURCE_HASH
      target_schedule_sha256: TARGET_HASH
      reason: Explicitly requested configuration revision
```

Compute these hashes with the existing `core.io.hash_json` and
`RL.config.schedule_contract` helpers. The source is the complete checkpoint's
`state.json.contract.schedule`; the target is the parsed new training config.
No checkpoint or frozen manifest is edited. A mismatched hash rejects admission.

This is a general revision envelope, not a two-field memory/topology whitelist.
The full new configuration is parsed and its changes recorded. Compatible future
execution settings (memory allocation/reclamation, clipping, checkpoint cadence,
learning rate, weight decay, scheduler settings and diagnostics) may be revised.
Optimizer moments, optimizer step, scheduler completed step, parameters, RNG and
cursors are loaded first. Revised optimizer settings are then explicitly applied;
loading old optimizer groups must not silently undo the requested learning rate.

Configuration permission is distinct from state compatibility. A changed model
or adapter tensor inventory/backend must pass existing tensor and execution
identity checks. Changing a frozen dataset/selection, advantage, anchor, logical
batch/window, accumulation or complete schedule requires separately defined
replanning/reset semantics and is rejected here. This interface never interprets
an incompatible cursor as progress through a new plan, or silently warm-starts.
Full-policy FSDP2 migration remains unsupported by this LoRA transition path.

`config-transition.json` records source receipt, both schedule hashes, reason,
source state, schedule/configuration differences and the previous transition
hash. It travels with new checkpoints. Old source bytes remain immutable.
Default exact-mode behavior and source receipt/tensor verification stay enabled.
The guarantee is state-preserving continuation with revised future settings,
not an unchanged optimization trajectory or bitwise cross-topology identity.

`runtime.topology_policy: preserve_updates` now additionally admits a single
worker whole-parent checkpoint as source for 2/4/8 workers. A source without a
distributed contract retains that original logical contract; new physical worker
metadata is recorded separately. Its single RNG state initializes the target
bookkeeping states; every forward/backward task must still prove no RNG use.
The same logical parents, branch weights, optimizer windows and anchor survive.

Validation: `tests/test_rl_config_migration.py` exercises hash-bound admission,
immutable/backend rejection, optimizer settings after moment restoration, and
real tiny-Qwen prepared/streaming single-to-two Gloo checkpoint round trips.
Existing topology tests cover 2/4/8-worker remapping.

### 原生流式恢复

Opt-in exception: `runtime.topology_policy: preserve_updates` supports 2/4/8-GPU bidirectional resume while preserving the original logical windows, anchor and optimizer state. Default exact mode retains same-world-size recovery. Training RNG consumption is rejected in the portable mode; see [topology switching](EXECUTION.md#保留优化窗口的跨拓扑恢复).

Use the existing `.venvs/rl/bin/dycon lora-train --training-config TASK.yaml`
entrypoint. Streaming applies to fresh training with
`runtime.data_preparation_async: true` and `prepared_data: null`, or to exact
resume from a supported streaming checkpoint. The parent contract selects the
data execution mode. An async flag cannot convert a prepared checkpoint into a
streaming checkpoint. Streaming warm-start, append-resume, format conversion and
world-size changes in the default exact mode are rejected. See [acceptance evidence](#验证范围)
for the tested single/two-GPU short native fixtures and the read-only v10 checks.

#### Choose the recovery path

| Start and parent contract | Execution path |
|---|---|
| fresh, async=true, prepared_data=null | Native streaming with an explicit external raw probability source |
| resume, supported streaming contract | Streaming; requires async=true and prepared_data=null |
| resume, prepared contract | Prepared, even when async=true |
| resume, neither streaming nor prepared contract | Existing legacy non-streaming recovery |

Streaming source selection is `probability_artifact` or `dataset_probability`;
it does not silently use `model_forward`. Unknown streaming versions and
conflicting execution settings fail. `run.mode` selects dry-run versus actual
execution; `run.start.mode: resume` selects the checkpoint state. In GRPO,
`run.mode: smoke` runs the configured dataset through the real training path.

#### Configure and run

Copy the parent's resolved training settings into a new task YAML and change only
the attempt, start/pause, physical device and output controls allowed by the
existing schedule contract. For example, retaining the original complete epoch:

```yaml
run:
  attempt_id: NEW_UNIQUE_ATTEMPT_ID
  mode: dry-run
  start:
    mode: resume
    checkpoint: /ABSOLUTE/PARENT/checkpoints/step-000060
runtime:
  data_preparation_async: true
  gpu: null
  gpus: [0, 1]
  output_root: /ABSOLUTE/ISOLATED_OUTPUT
prepared_data: null
training:
  max_updates: 1694
validation:
  stop_after_update: 61
```

This is a partial example, not a replacement for the parent's full YAML. Preserve
the original seed, LoRA, optimizer, scheduler, mathematical settings, preflight
switches, reference selection and parallel execution contract. Do not lower
`max_updates` to the pause target. The pause target must exceed the checkpoint
step, and a completed epoch cannot be resumed exactly.

Each invocation needs a new attempt ID, including a dry-run followed by actual
training. Use separate task YAMLs and output logs; do not reuse an executed
example ID. The checkpoint path names a committed `step-NNNNNN/` directory,
not its adapter export or an implicit latest checkpoint.

```bash
.venvs/runner/bin/dycon lora-train --training-config TASK-DRYRUN.yaml
# Use another new attempt ID and mode: smoke for actual updates:
.venvs/rl/bin/dycon lora-train --training-config TASK-TRAIN.yaml
.venvs/runner/bin/dycon lora-status NEW_UNIQUE_ATTEMPT_ID --storage-root /ABSOLUTE/ISOLATED_OUTPUT
.venvs/runner/bin/dycon artifact verify NEW_UNIQUE_ATTEMPT_ID --scope lora-attempt --storage-root /ABSOLUTE/ISOLATED_OUTPUT
```

Runner preflight imports neither Torch nor a model. It verifies checkpoint
receipt/JSON/file identities, the frozen plan, committed cursor, original source
binding, source metadata and dataset metadata. Its result explicitly reports
`tensor_validation: pending_training_model`. Actual training validates the
original adapter tensors using PEFT's saved/runtime name and dtype mapping,
restores the current checkpoint adapter, then restores optimizer, scheduler and
the corresponding rank RNG before any producer or forward starts. An independent
Torch-only snapshot check can validate the original tensor hashes without loading
the 9B base, but that check does not establish successful training.

#### Evidence and publication

New streaming checkpoints contain these recovery files in the normal receipt:

```text
state.json
training-state.pt
adapter/...
grpo/metrics-prefix.jsonl
grpo/stream-plan.json
grpo/source-provenance.json
grpo/anchor-tensors.json
grpo/reference/adapter/...
grpo/stream-resume.json
```

`stream-resume.json` schema 1 binds controlled relative recovery paths, byte
counts, hashes and the original reference identity. Internal symlinks, path
traversal, missing files and checksum mismatches fail. A partial or damaged new
bundle never falls back to another directory. Publication retains the existing
staging, receipt, fsync and atomic rename lifecycle. Recovery copies the original
anchor adapter and compact metadata at checkpoint times; it does not copy a
producer queue or build a full reference-probability cache. Disk overhead is the
anchor adapter plus plan/provenance metadata per checkpoint, in addition to the
existing current adapter and training payload.

The mathematical `contract.streaming.version` remains 1. Recovery metadata does
not broaden `resume_contract_matches`: historical project source hashes remain
provenance, while schedule, model, targets, dependency versions, attention,
determinism, reference and parallel contracts remain enforced.

#### Read-only historical compatibility

A historical checkpoint without recovery files may read fixed auxiliary paths
from its own `ATTEMPT/checkpoints/step-N` parent. It must provide the original
plan, source provenance, reference adapter and `initial-adapter.json`. The
checkpoint's own receipt, state and payload authorize recovery; a failed parent
does not need a top-level receipt or intact attempt progress fields. Neither the
old directory nor its receipts are modified. Missing or inconsistent original
evidence fails rather than reconstructing a shuffle or substituting today's
adapter as the anchor.

The new attempt's `initial-adapter.json` describes its current starting adapter.
The inherited original anchor is separately recorded in `grpo/anchor-tensors.json`
and `reference_identity.adapter_hash`. New checkpoints carry all recovery
metadata and can be recovered after the original attempt is removed. The old
checkpoint itself still depends on its historical auxiliary evidence.

#### External inputs and progress

Recovery metadata is self-contained; training inputs are not. Native streaming
continues to require the sealed dataset, exact native token ledgers, selected raw
probability arrays and base model. Native `branches.jsonl` references external
`native_evidence.root/ledger_ref`; copying the dataset alone is insufficient.
Keep the original dataset/source absolute paths. Automatic path relocation,
source migration and CUDA RNG device-list remapping are not implemented.

| Recovery input | What must remain accessible |
|---|---|
| New checkpoint | Current adapter/training state, frozen recovery bundle and receipt; original parent attempt is unnecessary |
| Legacy streaming checkpoint | Its checkpoint plus fixed historical parent plan/provenance/reference/initial-adapter evidence |
| Sealed dataset | Original receipt and frozen group/branch metadata |
| Native token payloads | Referenced `native_evidence.root/ledger_ref` request JSON files for unconsumed groups |
| Raw reference probabilities | Selected artifact/stream metadata, receipts and unconsumed value files |
| Base model/tokenizer | Original compatible registered model files and identities |

A native ledger is a historical model request JSON containing exact prompt and
completion token IDs. It is separate from raw reference logprob arrays and
hidden activation tensors. Copying `groups.jsonl`/`branches.jsonl` alone leaves
those token payloads behind. For a canonical probability artifact, original
upstream probability sources may be offline once receipt-local copies exist;
the selected canonical artifact itself must still be available to streaming.
Use the frozen unconsumed group set to inventory dependencies. This feature
does not automatically copy data, rewrite paths or widen the frozen audit flags.

Metadata may be scanned, but native and probability payloads of consumed groups
are not reread. Frozen advantages and order are inherited directly from the plan.
For each unconsumed window, token hashes, lengths, masks and probability bindings,
checksums and finite float32 targets are mandatory even when optional historical
audit flags are false. Source and dataset metadata identity is checked during
cleanup on completion, pause and exception. A pause is not evidence that every
external payload was audited.

The two-window producer restarts from `state.group_cursor`. Prefetched or inflight
windows do not advance the checkpoint cursor. Each rank restores its own saved
RNG, and CUDA visible-device counts must match exactly. Group assignment is the
existing group-index-within-window rule; the historical assignment string does
not redefine that algorithm. Queue and producer errors propagate to training;
the distributed supervisor terminates failed peers.

For the historical v10 step-60 target the expected initial cursor is 240 groups,
7680 branches and 1492544 completion tokens. Metrics beyond that checkpoint do not
authorize skipping later updates. A successful short GPU fixture or runner
dry-run does not prove the original long contexts fit GPU memory. The original
OOM remains possible, and a full 1634-update continuation requires an explicit
execution request. v10 validation on 2026-11-14 did not execute update 61.

#### Inspect progress and failures

Use `lora-status` to distinguish `last_completed_update` from
`recoverable_checkpoint_step`. Only the published checkpoint determines the
restart state; later metric lines and inflight/prefetched groups cannot advance
it. A failed parent may still have a valid checkpoint. Missing evidence,
changed source/token/probability content, wrong rank RNG topology or a changed
contract fails recovery; there is no fallback to fresh or automatic re-scoring.
Successful `artifact verify` establishes artifact consistency, while actual
forward/update acceptance and long-context GPU capacity are separate checks.


#### 8x16 execution cursor

The `state.group_cursor` producer restart rule above applies to whole-parent
schedules. [8x16 mode](EXECUTION.md#两轮-distinct-parent-816) restarts from
`consumed_execution_chunks`, validated against immutable complete window
boundaries, the schedule hash, round, selected native branch identities and
consumed-half prefix hash. `group_cursor` and `consumed_groups` explicitly
project `completed_parent_groups`; they do not count halves. A valid checkpoint
may have seen a parent whose other half is unconsumed. Prefetch never commits
this cursor. The existing self-contained recovery bundle includes the extended
stream plan and all-rank RNG; external inputs are still required. Exact resume
requires the same 8-rank execution/reduction protocol and partition, while the
physical GPU indexes may change. Old v10/1/2/4-rank checkpoints cannot directly
switch protocols. Streaming warm-start/append restrictions remain in force.

With `runtime.topology_policy: preserve_updates`, physical world size can change
among 2/4/8 while the source protocol, frozen windows and committed cursor remain
unchanged. A round-0 checkpoint retains all outstanding halves and resumes from
the execution-window cursor. Keep the original async/prepared binding, complete
schedule length and unconsumed external inputs. Use a new attempt ID and first
validate through the runner CLI. The real v10 step-60 eight-device dry-run passed
metadata validation on 2026-11-15; model tensor and training RNG validation still
require actual execution. See [topology switching](EXECUTION.md#保留优化窗口的跨拓扑恢复) and
[evidence and pending GPU acceptance](EXECUTION.md#验证范围).

## 接口与诊断

### 监督位置和 Python API

#### 选择监督位置

| 模式 | `max_targets_per_sample` | batch 中的 `loss_mask` | 行为 |
|---|---|---|---|
| `labels`，默认 | `null` | 必须不提供 | 所有有效 causal labels |
| `explicit_mask` | `null` | 每个 batch 必须提供 bool mask | 精确采用调用方选中的 target |
| `uniform` | 正整数 K | 必须不提供 | 每个样本至多 K 个确定性均匀位置 |

`explicit_mask` 与 `uniform` 的正式配置都要求 `training.memory.loss_chunk_tokens > 0`，
确保在 vocabulary projection 前选点。`uniform` 的 K=1 选择最后一个合法 target；
其他情况按整数下取整选取有序首尾及中间位置，算法详见[稀疏监督说明](#稀疏因果交叉熵)。
CLI explicit smoke 的 mask 由内置 fixture 生成，不代表按文本自动识别最后一个 agent step。

Python batch 坐标为 `[B,L]` 中的原始 **target token**：

```text
input_ids: [10,   11,   12,   13,   14]
labels:    [-100, -100, 12,   13,   14]
loss_mask: [false,false,true, false,true]
实际计算：head(hidden[1]) 预测 labels[2]，head(hidden[3]) 预测 labels[4]
```

整个五 token 输入仍送入 backbone。mask 不传入模型、不修改 attention、不截短有效
历史；causal shift 只做一次。位置 0 不可选；选中 label 不能为 `-100`，target 和
前一个 predictor 位置都必须非 padding。每个样本至少选择一个 target。接口允许
label 不同于对应 input ID，不自动补 EOS。

#### Python 服务与训练核心

要从 Python 调用与 CLI 一致的预检/训练流程，使用公开服务：

```python
from RL.service import run

result = run(
    {"project_root": "/home/wangtianyi/DyConGlobal"},
    "RL/experiments/lora-sparse-uniform.yaml",  # 默认 dry-run
)
print(result["status"])
```

服务读取 YAML，并管理身份、锁、attempt/checkpoint、adapter 和 receipt。GPU smoke
应在 `.venvs/rl` 的新进程中调用，让物理 GPU 选择发生在 CUDA 初始化之前。

自定义 batch 调用训练核心时，调用方先准备已注入 LoRA、冻结基座的模型，再传入
optimizer/scheduler；下面的函数可直接用于这样的模型。它不负责模型加载和文件发布：

```python
import torch
from RL.config import Memory, Supervision, Training
from RL.memory import LossBackend
from RL.training import optimizer_and_scheduler, train

def train_selected_targets(model):
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 12, 13, 14]], dtype=torch.long),
        "loss_mask": torch.tensor([[False, False, True, False, True]]),
    }
    settings = Training(
        max_updates=1, gradient_accumulation_steps=1,
        learning_rate=0.0001, weight_decay=0.0, grad_clip=1.0,
        scheduler="constant", checkpoint_every_updates=1,
        memory=Memory(loss_chunk_tokens=16),
        supervision=Supervision(mode="explicit_mask"),
    )
    optimizer, scheduler = optimizer_and_scheduler(model, settings)
    metrics = []
    state = train(
        model, [batch], LossBackend(settings.memory), optimizer, scheduler,
        max_updates=settings.max_updates,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        grad_clip=settings.grad_clip,
        vocab_size=model.config.text_config.vocab_size, max_length=5,
        supervision=settings.supervision,
        on_update=lambda state, row: metrics.append(row),
    )
    return state, metrics
```

`train` 不会自动读取 `Training` 中的配置，必须像示例一样显式传递相应参数；单独调用
时 `checkpoint_every_updates` 也不会自动保存 checkpoint。需要工程正式产物时使用
`RL.service.run`；该低层 API 的 checkpoint/receipt 生命周期由调用方负责。

训练核心在窗口任何 forward/backward 之前解析全部 batch，复制 effective labels，
按窗口实际选中 target 总数归一化：`sum(loss_sum) / sum(selected_count)`。原始 labels
不被原地改写。末尾不足一个 accumulation window 时仍按实际数量更新。

自定义 `loss_callback(model, prepared_batch)` 返回标量 `loss_sum` 和整数
`selected_count`。收到的 batch 只有 `input_ids`、`attention_mask`、effective `labels`；
未选位置已经是 `-100`。回调应返回 **sum**，不预先求 mean、不再次选点或再次 shift
已经对齐过的数据。输出计数必须与窗口解析结果相符。`RL.validation.causal_ce` 可以
作为既有三字段回调；生产稀疏 CE 使用 `LossBackend`。

如只需一次 loss，先解析原始 batch，再调用后端（假设上例的 model/batch 已准备）：

```python
from RL.config import Memory, Supervision
from RL.memory import LossBackend
from RL.supervision import prepare_batch

def selected_loss(model, batch):
    selection = prepare_batch(
        batch, vocab_size=model.config.text_config.vocab_size,
        max_length=batch["input_ids"].shape[1],
        supervision=Supervision(mode="explicit_mask"),
    )
    device = next(model.parameters()).device
    prepared = {key: value.to(device) for key, value in selection.model_batch().items()}
    loss_sum, count = LossBackend(Memory(loss_chunk_tokens=16))(model, prepared)
    return loss_sum, count, selection
```

`selection` 提供 `coordinates`、`available_count`、`selected_count`、`input_tokens`、
`identity` 和 `sha256`。不要把含原始 `loss_mask` 的 batch 直接交给 `LossBackend`，
也不要把选择后的 batch 再按 explicit 模式重新选择。`selection.model_batch()` 用于
loss；原始输入身份和选点摘要保留在 `selection` 中。

### 实时进度、TensorBoard 与已冻结 advantage

The opt-in [host budget and pressure policy](EXECUTION.md#主存预算固定页分配器和压力策略) records
process pinned active/allocated bytes, pool idle/reserved bytes, cache reclaim
costs, CPU RSS/HWM, GPU boundary choices and host/cgroup reserve samples in each
rank's memory events. `lora-memory` preserves missing historical fields as null;
TensorBoard exposes process metrics separately from host-global MemAvailable.
Host-global available RAM must never be summed across ranks. PSS is sampled by
the bounded acceptance fixture rather than scanning smaps in training hooks.

Streaming recovery reports a verified checkpoint independently of a failed parent's top-level receipt or missing attempt contract. The checkpoint's own receipt, frozen metadata and committed cursor determine recoverability; inflight/prepared windows and later metrics do not change the resume point. Runner checks do not load Torch or prove actual anchor tensors/CUDA execution. See [streaming recovery](#原生流式恢复).

Use the project CLI from `/home/wangtianyi/DyConGlobal`. Queries run in
`.venvs/runner` without Torch. Managed visualization and legacy index workers use
`.venvs/analysis` with `tensorboard==2.20.0` and `ijson==3.5.1`; they never load a
model. Training continues through the existing `.venvs/rl/bin/dycon lora-train`.

#### Current training

```bash
.venvs/runner/bin/dycon lora-status --active
.venvs/runner/bin/dycon lora-status grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon lora-status grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --watch 5 --format json
.venvs/runner/bin/dycon lora-tensorboard grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --action start --port 16006
```

On the local PC, keep this forwarding connection open while browsing:

```text
ssh -N -L 16006:127.0.0.1:16006 wangtianyi@10.249.187.9
http://127.0.0.1:16006
```

The server only binds `127.0.0.1`. Start is idempotent for the same source and
port; occupied ports are errors. A running attempt uses a training lifecycle:
the service stops when the attempt becomes terminal or its training process exits.
For a non-running attempt, start creates a query lease of 30 minutes by default;
override it with `--lease-seconds N`, and renew an active lease with
`--action renew --lease-seconds N`. The synchronizer and TensorBoard both stop
when the lease expires. Use `--action status` to inspect both processes
and event freshness, and `--action stop` to stop only their exact registered
PID/starttime/argv identities, wait for exit and publish a stop receipt. A failed
worker never stops or restarts training. `--action rebuild` explicitly discards
the old synchronization cursor, creates a fresh generation and restarts the owned
visualization processes; use it after reviewing a source replacement/conflict.

#### Progress semantics

`status` preserves the raw attempt status, including preparation stages. Its
`observed_liveness` is independent: an exited process with a nonterminal attempt
state is `not_running_with_stale_state`, not an invented success or failure.
Old process discovery binds the exact training-config argument to its attempt
ID, physical output root and process creation time. The live record for new
training also records proc starttime, which prevents PID reuse mistakes.

`last_completed_update` and `committed_consumed_groups` come from complete metric
lines after optimizer updates. `inflight_update` and
`live.completed_groups_in_update` describe work whose parameters have not yet
been committed. `recoverable_checkpoint_step` comes from a published checkpoint
receipt with a matching state hash and contract. Checkpoint publication can lag
the latest update. `execution_stop_target` is the validation pause boundary;
`max_updates` is the complete frozen schedule target, including append segments.

The latest and recent metrics contain scalars only. Status reads an expanding
tail, not the full frozen token JSON. JSONL parsing accepts complete newline
terminated objects, holds a partial tail for the next read, reports malformed
complete lines and conflicting steps, and has a 64 MiB per-line resource limit.
Unknown arrays such as attention masks are discarded by the streaming parser.
Watch mode preserves its incremental cursor and prints one JSON object per line.

ETA uses the last ten completed updates produced by this attempt, excluding the
inherited prefix. It reports sample count, mean, median, remaining updates and
confidence; times exclude startup and may exclude checkpoint overhead. Seconds
and bytes are the raw metric units; memory chart units are GiB (2**30 bytes).
No stale-metric timeout automatically declares a hung CUDA kernel or restarts it.

New GRPO training exposes window/group forward/backward, prefix phases where
available, optimizer update, checkpoint and finalization boundaries. Group and
branch identity are never inferred from a partially processed gradient tensor.
`contribution_ordinal` counts objective contributions, and `branch_ordinal` stays
null when the backend cannot prove a native branch mapping. Old running trainers
report `stage=unknown`. Heartbeats every ten seconds indicate process liveness;
stage timestamps indicate when a boundary changed. Diagnostic writes are
throttled to two seconds, except committed/checkpoint/error boundaries, and fail
as warnings without changing optimizer, cursor or random-number behavior.

#### Curves and recovery

Metrics are replayed into a single independent run per attempt. Horizontal step
is the lineage's `global_step`, with `progress/inherited` identifying the parent
checkpoint prefix. The cursor records the parent checkpoint and inherited step.
Resume/append attempts never share event directories even when steps overlap.

`train/loss` is the total objective; `train/policy_loss` and
`train/k3_penalty_mean` are separate, the latter before multiplication by beta.
A policy-ratio trio is also recorded: `train/ratio_mean` is the valid completion
token mean, while `train/policy_ratio_positive_advantage` and
`train/policy_ratio_negative_advantage` partition that same ratio by positive and
negative branch advantage. A missing sign in an update is omitted from TensorBoard
rather than emitted as a misleading zero.
A lower training objective does not establish a higher benchmark success rate.
The UI provides step, relative and wall-time axes and its own smoothing control;
use smoothing zero to view raw values. No smoothing is applied to source data.
Only complete optimizer updates create loss points, even if an update takes many
minutes. New metrics include `observed_at` and the actual producing attempt ID;
inherited metric lines are copied unchanged. Legacy metrics without timestamps
have `historical_wall_time_estimated=true`; their event wall time is zero rather
than a fabricated per-step time derived from file mtime.

The synchronization/flush/reload intervals are two seconds. The normal local
filesystem acceptance target is less than ten seconds after an update. A single
writer lock and `(global_step, source-line SHA256)` detect duplicate/conflicting
rows. A crash can occur between event and cursor publication: every restart
replays history into a new generation, then atomically replaces the visible
logdir; it does not append to the interrupted generation. Source inode,
size, first-line and last-committed-line fingerprints detect truncation,
replacement and committed-prefix changes; such errors remain visible in status
until explicit rebuild. Old generations are excluded from the active run.

#### Fixed probe reports alongside live monitoring

`lora-probe` complements the existing training monitor by evaluating an explicit
checkpoint on fixed complete parent groups. It requires extra model forwards,
runs in `.venvs/rl`, and writes JSON outside the sealed training artifacts.

Its JSON also includes `token_changes` and `groups[].token_changes`: signed and
absolute logprob-change means and quantiles, the largest 1%/5%/10% tokens' share
of total absolute change, and token fractions with relative probability changes
strictly exceeding 10%/50% (increase, decrease and either direction). Each section
splits all/positive/negative/zero-advantage tokens and pools actual target tokens
with equal token weights. These diagnostics reuse existing probe forwards, remain
outside TensorBoard, and expose no token text, positions or individual probability
pairs. They are relative to the training anchor, including when a baseline report
is supplied. Existing reports need a fresh evaluation to populate these fields;
they remain compatible baselines for the original scalar comparisons. See
[fixed probe definitions](#token-变化分布与集中度) for formulas,
rounding, empty partitions and the difference from branch/group-equal metrics.

| Interface | Input and purpose | Model forward / output |
|---|---|---|
| `lora-status` / `lora-tensorboard` | Observe recorded metrics from changing training batches | No evaluation forward; status / live curves |
| `lora-advantages` | Inspect frozen group IDs, advantages and provenance | No model forward; group / branch records |
| `lora-probe` | Compare the same groups at explicit checkpoints | Additional teacher-forced forwards; independent JSON reports |

```bash
.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000100 \
  --group-id GROUP_A --group-id GROUP_B \
  --gpu 0 --output reports/rl-probe/step100.json

.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000200 \
  --baseline-report reports/rl-probe/step100.json \
  --gpu 0 --output reports/rl-probe/step200.json
```

The second command reuses the first report's ordered IDs. Read
`aggregate.preference_gap_change` for improvement relative to the original anchor
and `comparison.aggregate_delta.preference_gap` for current-minus-baseline
checkpoint change. Check `support.comparable_group_count`; missing signs produce
`null` comparisons. Positive values indicate movement toward the frozen positive
advantage branches on this subset, not a measured downstream success increase.

Probe reports never enter the TensorBoard synchronizer and do not update
`metrics.jsonl`, checkpoints or training cursors. Choose an idle GPU explicitly;
the shared GPU lock prevents use of a device owned by training. `--dry-run` works
in the runner environment without loading a model or creating a report.
See [固定 group 探针](#固定-group-偏好探针) for exact reduction, compatibility checks,
source retention and validation scope, or [训练配置和预检](#配置预检和首次训练) for the workflow.

#### Live learning signals and chart folders

The realtime signal producer and TensorBoard projection are both required.
Restarting a visualization alone reloads chart mappings; an already running
trainer keeps its loaded Python objective and optimizer code. New producer-only
statistics appear after the next normal training launch or checkpoint resume.
No automatic training restart is performed by monitoring commands.

| TensorBoard folder | Signals and interpretation |
|---|---|
| `loss/` | Total and policy loss; retain original `train/*` aliases |
| `learning/logprob/`, `learning/ratio/` | Positive / negative advantage branch changes and ratios |
| `learning/preference/` | Group preference change, advantage alignment, fraction of positive-gap groups |
| `learning/support/` | Comparable groups and positive / negative target counts |
| `policy/active/` | Policy surrogate tokens still carrying a non-flat direct objective term |
| `policy/clipping/` | Interval exceedance and sign-dependent saturation, reported separately |
| `policy/guard/` | Excluded tokens / objective weight, sign partitions, fully excluded branches / groups |
| `drift/ratio/`, `drift/log_ratio/` | Ratio bounds / tails and log-ratio min, p01, p50, p99, max |
| `drift/k3/` | Guarded K3 objective mean, max and cap comparisons; shown even when KL beta is zero |
| `optimizer/gradient/` | Preclip norm, threshold, estimated clip scale and parameter gradient coverage |
| `optimizer/update/` | Actual adapter update on a deterministic parameter sample, at step 1 and every 10 steps |
| `optimizer/learning_rate/` | Current and next learning rate |
| `throughput/`, `timing/` | Committed update / group / branch / target rates, retry fraction, queue / synchronization timings |
| `progress/` | Existing committed counters and model step at which forward metrics were evaluated |

`learning/preference/gap` is the group mean of positive-minus-negative
**logprob changes relative to the original anchor**. In the independent fixed
probe JSON, the corresponding anchor-relative quantity is
`preference_gap_change`; the probe's `preference_gap` is an absolute difference
between current positive / negative branch mean logprobs. Training batch
composition changes every update, so neither live curve is an update-before/after
measurement on fixed inputs. Forward diagnostics emitted at committed step `s`
were evaluated at model step `s-1`; `progress/evaluated_at_global_step` records this.

For nonzero advantages, saturation means positive advantage with `r > 1+epsilon`
or negative advantage with `r < 1-epsilon`, excluding guard-rejected tokens.
Active means neither rejected nor saturated and advantage nonzero. These describe
the policy term's direct token coverage, not the norm of the complete parameter
gradient; K3 can still contribute when enabled. The old `clip_fraction` counts
both directions outside the interval and is not a saturation fraction.

`objective_fraction` uses token mean → branch mean → group mean with original
target denominators. `advantage_weighted_fraction` additionally weights by `|A|`.
Positive / negative `token_fraction` fields and log-ratio quantiles use actual
token counts. Missing signs or zero total advantage magnitude yield no scalar
point rather than a fabricated zero. In 8x16 mode these are execution-window
statistics on the scheduled halves; they do not combine different model states
into a complete-parent evaluation.

Gradient coverage counts trainable parameter elements whose gradient exists,
including zero-valued gradients; it does not claim every element has a nonzero
gradient. Update sampling reads at most 64 fixed elements per trainable tensor,
uses no RNG and makes no complete adapter copy. Relative update norm is omitted
when the sample's before-norm is zero. No extra model forward is added.

##### Existing attempts and visualization refresh

On an older running trainer, the updated reader can expose already recorded
positive / negative logprob changes, guard totals, K3 drift, next learning rate,
timings and throughput derived from stored counts / seconds. It cannot recreate
per-token tails, active / saturated fractions, group preference or sampled
parameter changes if those were never recorded. Missing historical values stay
absent. This also applies to metrics inherited from a parent attempt after resume.

After updating monitoring code, rebuild an already started TensorBoard service
using its current port and storage root:

```bash
.venvs/runner/bin/dycon lora-tensorboard ATTEMPT_ID --storage-root ROOT \
  --action rebuild --port 16006
```

The formal command restarts only its owned visualization processes and replays
existing metrics into a fresh event generation. Keep the existing SSH forwarding
and refresh the browser after it reports ready. Newly emitted signals then append
as training progresses; producer-only fields require a trainer that loaded the
new code. Semantic/data-invariant charts and independent probe forwards are not
part of these realtime additions.

#### Frozen group/branch advantages

Use `--group-id GROUP_ID` or `--group-ordinal N` to select a frozen group. Add `--branch-id BRANCH_ID` to select one branch by identity.

```bash
.venvs/runner/bin/dycon lora-advantages grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --build-index
.venvs/runner/bin/dycon lora-advantages grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --index-status
.venvs/runner/bin/dycon lora-advantages grpo-swe169-consensus-half-prefix-bf16-epoch1-gpu0-20261111-v3 --storage-root /data2/wangtianyi/DyConGlobal --group-ordinal 0
.venvs/runner/bin/dycon lora-advantages observability-gpu1-resume-20261112-v1 --storage-root /data3/wangtianyi/DyConGlobal --group-ordinal 0 --format csv
```

No group filter returns twenty group summaries and a `next_offset`. An explicit
group returns its branches, up to 256 rows; truncation reports total, returned and
next offset. `--offset/--limit` page summaries without a group filter, and native
branches with a group filter. CSV requires an explicit group/branch filter.
Branch IDs reverse-resolve through the index. All ordinals and ranks are zero
based. Group ordinal follows the frozen training schedule, with schedule,
segment and segment-local identities for prepared artifacts. Native branch order
is preserved. Lower historical score is better; equal scores share a rank,
and the separate tie-break order uses native branch order. Frozen ranking
diagnostics take precedence over deriving this ordering from scores. Excluded groups
return their reason and null advantage, never a fabricated zero.

Each result binds the attempt contract, dataset receipt, policy/config hash and
frozen order. `frozen_prepared_artifact` reads the actual score/advantage shard;
metadata index and score receipts are verified without opening token arrays.
`PreparedReader.metadata()` is the shared query API; `raw()` and explicit deep
artifact verification retain their existing full-token behavior. Bounded
`--token-preview 0..128` is available for prepared artifacts and defaults to zero.
Legacy indexes deliberately contain no tokens and reject token preview.

`frozen_groups_json` means the original saved frozen advantages. Old records
without saved values return `provenance=unavailable` and null advantages; current
math is insufficient evidence of the producer's historical implementation, so
it is never silently used to manufacture an as-used advantage.

Legacy `--build-index` is explicit and returns an owned worker with byte/group
progress. It parses the complete JSON using the locked streaming parser, spools
one group's canonical token representation temporarily to disk for the original
order/content hash, and retains no token arrays in Python or in the final SQLite
cache. It validates group count/order, score/retained/advantage/token order hash,
source hash, and start/end file identity. A temporary database is published only
after validation; concurrent builds share a lock. Queries never start a 59 GiB
scan. Ready matching indexes are reused. Unsealed sources are clearly marked
`locally_fingerprinted_unsealed`; sealed sources additionally verify their
frozen-JSON receipt hash. Query results verify the published index checksum.

#### Mutable state and verification

All source binding, live heartbeat, managed service records, event generations
and legacy indexes live under
`runtime/rl-observability/<canonical-root-attempt-contract-key>/`. Status and
advantage queries never update attempts or checkpoints. No mutable file is added
to sealed attempts/prepared artifacts; `artifact verify --scope lora-attempt`
therefore remains valid across visualization start/status/stop.

Tests: `tests/test_rl_observability.py` (runner/analysis),
`tests/test_rl_observability_training.py` (rl), and the existing prepared,
lifecycle, service and advantage regression suites. See
[训练验证范围](#验证范围) for this deployment's evidence and process state.

#### Memory and OOM recovery events

As of 2026-11-14, event/summary behavior has CPU regression coverage and the
formal CLI reads legacy v10 without mutation. That attempt has no new-format
memory event files; zero files/buckets cannot establish its memory behavior.
Production GPU telemetry and actual transfers await GPU availability. See
[OOM validation](EXECUTION.md#显存与主存) for the deployment evidence.

Each rank appends `memory-events.jsonl` under its attempt directory (rank zero at
`memory-events.jsonl`, other workers under `distributed/rank-N/`). Events include
step/group lengths, committed versus inflight step, phase, allocator decision,
allocated/reserved/peak/free bytes, host budget, and OOM cleanup/retry status.
`metrics.jsonl` remains committed-update-only. A retry keeps the same window and
cursor and advances the optimizer once after a successful attempt.

The one device-wide sampler resets peaks only after sampling them, aggregates
nonoverlapping phase intervals into attempt peaks, and excludes peer waiting from
compute phase time. Existing prefix/update callbacks do not independently reset
those counters. Branch samples are aggregated by group and phase. Attention-mask
records and profiler state are scoped to an attempt, including failed attempts.
The sampler uses existing compute synchronization points; it adds no per-layer
synchronization. Transfer timings are reported separately from wall time.

Use the Torch-free, read-only summary through the ordinary CLI:

```bash
.venvs/runner/bin/dycon lora-memory ATTEMPT_ID --storage-root /data3/wangtianyi/DyConGlobal --limit 100
.venvs/runner/bin/dycon lora-memory ATTEMPT_ID --storage-root /data3/wangtianyi/DyConGlobal --rank 1 --phase prefix_backward
```

It groups samples by rank, sequence-length interval, phase, offload mode and
outcome (`committed`, `CUDA_OOM`, peer-discarded, fatal, or uncommitted). It streams
bounded records and supports bounded output. An incomplete trailing line from an
active writer is ignored explicitly. The original OOM text is preserved: the
failed allocation is absent from allocated peaks, and reserved-minus-allocated is
not evidence of a reusable contiguous allocation or a leak. CPU state-machine and
hook tests do not establish actual CUDA savings or pinned-transfer behavior.


#### 8x16 parent and execution progress

[8x16 mode](EXECUTION.md#两轮-distinct-parent-816) writes metrics schema 3. Status and TensorBoard
expose `planned_parent_groups`, `planned_execution_chunks`,
`consumed_execution_chunks`, `distinct_parent_groups_seen`,
`completed_parent_groups` and `execution_window_cursor` separately. The legacy
group cursor/consumed counters project only fully committed parents, so they can
remain zero throughout round 0. Per-update `parent_group_count` and
`execution_chunk_count` are K, and branch count is 16K. Parent branch count is
32; loss is a mean over selected halves at the current parameter state.
`parallel.rank_critical_path_seconds` and end-to-end `seconds` are elapsed times;
summing synchronized segment durations across ranks is aggregate compute, not
wall time. Prefix rows carry exact execution identities and are update-scoped.

### 固定 group 偏好探针

`dycon lora-probe` 在显式 checkpoint 上，对固定、完整的 GRPO parent groups
进行 teacher-forced raw T=1 评分。每个分支一次完整 forward；不做 rollout、
backward、optimizer 恢复或更新，不写训练游标和 TensorBoard。

在项目根目录用 RL 环境执行正式 CLI（runner 环境可执行 `--help` / `--dry-run`）：

```bash
.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000100 \
  --group-id GROUP_A --group-id GROUP_B \
  --gpu 0 --output reports/rl-probe/step100.json

.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000200 \
  --baseline-report reports/rl-probe/step100.json \
  --gpu 0 --output reports/rl-probe/step200.json
```

第二条命令继承前一报告的有序 group ID 清单。也可再次传入完全相同的
`--group-id`。不自动选择最新 checkpoint，不随机抽样，不重算优势。
`--gpu` 是物理 GPU 编号或 MIG UUID：命令使用训练共用的 GPU 锁，并检查
设备空闲。应选择独立空闲设备；不会自动暂停、重启或抢占训练。

加 `--dry-run` 可检查 checkpoint receipt、所属 attempt 的
`resolved-config.json` 与冻结 schedule、group ID 和输出路径；无需 `--gpu`，
不加载模型，不创建报告。它不代表已验证选中 payload 或完成数值比较。

#### 与训练实时指标的区别

训练期正负优势分支的 logprob 变化及偏好差使用当步 batch，随训练进度更换样本；
固定 probe 在不同 checkpoint 上重评同一份 group 清单，因此能排除样本更换造成的
曲线变化。两者都使用原始 anchor 与冻结优势，但固定 probe 需要额外 forward，
通过独立 CLI 随时调用，报告不进入 TensorBoard。

选取 ID 时可先通过 `lora-advantages ATTEMPT_ID --storage-root ROOT --group-ordinal 0`
查看实际 group 标识及冻结正负分支。不要将 group ordinal 或 8x16 半组 ID 当成
`--group-id`。选定后，在后续 checkpoint 中用 `--baseline-report` 继承清单。
应选择同时包含正负优势分支的 groups，且持续保留其源数据。

元数据预检示例（将路径和 ID 替换为实际值）：

```bash
.venvs/runner/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000100 \
  --group-id GROUP_A --group-id GROUP_B \
  --output reports/rl-probe/step100.json --dry-run
```

#### 统计口径

对每个分支，先求真实 completion target 的平均 logprob。随后在每个
group 内按冻结优势符号划分正负分支，分别对分支等权平均；最后对 group
等权平均。不会让长 completion 或分支数较多的 group 支配整体结果。

| JSON 字段 | 含义 |
|---|---|
| `positive_logprob_mean` / `negative_logprob_mean` | 正/负优势分支的平均 logprob |
| `positive_logprob_change` / `negative_logprob_change` | 相对训练原始 anchor 的 logprob 变化 |
| `preference_gap` | 正优势 logprob 均值减负优势 logprob 均值 |
| `reference_preference_gap` | 原始 anchor 在相同分支上的偏好差 |
| `preference_gap_change` | 当前偏好差减 anchor 偏好差；越大表示相对 anchor 越偏向正优势分支 |
| `advantage_logprob_alignment` | 分支平均的 `advantage * logprob_change` |
| `pairwise_preference_rate` | group 内正负分支两两比较平均 logprob 的胜率，平局计 0.5 |
| `improved_group_fraction` | `preference_gap_change > 0` 的可比较 group 比例 |

报告包含 `aggregate`、`support`、逐 group / branch 明细、checkpoint step
与 receipt 哈希、probe 身份和评分配置。缺少正/负符号时相关比较为 `null`；
`support.comparable_group_count` 给出有效比较数量。优势为零的分支不进入
正负偏好差，但保留在总体分支统计及明细中。

`comparison.aggregate_delta` 和 `comparison.group_delta` 都是
**当前报告减历史报告**。必须保持有序 group ID、token、mask、冻结优势、
参考概率字节、base/anchor 身份以及评分协议、dtype、分块配置和依赖版本一致。
不兼容时明确报错，不生成可误读的改善幅度。

例如：anchor 偏好差为 `0.10`，step100 为 `0.25`，step200 为 `0.32`。
step200 的 `aggregate.preference_gap_change` 为 `0.22`；以 step100 为基线时，
`comparison.aggregate_delta.preference_gap` 为 `0.07`。前者始终相对训练原始
anchor，后者相对指定历史报告；传入基线报告不会更换训练 anchor。

命令 stdout 返回完成状态、报告路径、整体指标及比较摘要；`--output` 文件保存
完整 JSON，包括 `groups[].branches`、`probe_identity` 和 `probe_sha256`。

#### Token 变化分布与集中度

每次正常 probe 默认在 JSON 顶层 `token_changes` 及
`groups[].token_changes` 中输出以下统计，不需要新增 CLI 参数。
它们复用现有评分得到的 target logprob，不增加模型 forward，不进入 TensorBoard，
也不输出逐 token 数组、位置、文本或逐 token 前后概率。

定义 `d_i = log(p_current_i) - log(p_anchor_i)`，使用自然对数；
概率指同一固定上下文下、raw T=1 的真实 completion target 概率。
所有变化始终相对训练原始 anchor；`--baseline-report` 不会改变这一参照。

每个统计区都有 `all`、`positive`、`negative`、`zero` 四个分区，分别包含
全部 token、正优势分支 token、负优势分支 token、零优势分支 token。
分区由冻结的分支优势决定。顶层合并所有选中 group 的实际 token 后计算，
group 内也直接合并 token；**每个 token 等权**，不平均分支或 group 的分位数。
因此长分支权重更高，与原有 `aggregate` 中分支等权再 group 等权的指标口径不同。

以下字段相对于 `token_changes.all`（或相应 group / 优势分区）：

| 字段 | 含义 |
|---|---|
| `token_count` | 实际参与统计的 completion target token 数 |
| `logprob_change_mean` | `mean(d_i)`，带方向，可能正负抵消 |
| `abs_logprob_change_mean` | `mean(abs(d_i))`，反映平均变化幅度 |
| `logprob_change_quantiles` | 带方向 `d_i` 的分位数 |
| `abs_logprob_change_quantiles` | `abs(d_i)` 的分位数 |
| `top_abs_change.top_1pct.abs_change_share` | 按 `abs(d_i)` 降序排列，最大 1% token 占全部 `sum(abs(d_i))` 的比例 |
| `top_abs_change.top_5pct` / `top_10pct` | 同上，分别取最大 5% / 10% token |
| `relative_probability_change.gt_10pct.either_fraction` | `abs(p_current_i / p_anchor_i - 1) > 0.10` 的 token 比例 |
| `relative_probability_change.gt_50pct.either_fraction` | 同上，阈值为 0.50 |
| `relative_probability_change.gt_10pct.increase_fraction` / `decrease_fraction` | 分别为相对概率上升 / 下降超过 10% 的比例；`gt_50pct` 同样提供方向拆分 |

分位点为 `p00/p01/p05/p10/p25/p50/p75/p90/p95/p99/p100`，使用线性插值。
Top k 为 `ceil(token_count * 百分比)`，非空时至少取 1 个；每项的
`token_count` 给出实际 k，小样本的实际比例可能高于名义百分比。
空分区的均值、分位数和比例为 `null`，计数为 0；全部变化为零时集中度
`abs_change_share` 为 `null`，非空分区的阈值占比为 0。

阈值是严格“大于”，相对概率变化不是百分点变化。例如从 0.20 变为 0.22
是相对上升 10%，不是上升 10 个百分点；恰好达到阈值不计入。
实现直接比较 `d_i > log(1+t)` 或 `d_i < log(1-t)`，避免指数溢出。
各概率只在固定的 teacher-forced 上下文下成立，不代表自由生成的轨迹概率。

例如所有 token 变化同样大，最大 1% token 的变化贡献率约为 1%；
若仅 1% token 发生变化，该贡献率可达 100%。结合带方向均值、绝对均值
和分位数，可以区分“普遍小幅变化”“少量 token 大幅变化”和“正负抵消”。

该统计区使用独立 `version: 1`，是报告 schema 1 的新增字段，不改变评分协议、
`probe_sha256` 或原有偏好指标。旧报告仍可作为 `--baseline-report` 继承分组并比较
原有指标；`comparison` 继续只比较原有标量指标，不对分位数或集中度做隐含差分。
历史报告仅保留分支均值，无法补算 token 分布；需重新运行 probe 并输出到新路径。
JSON 只保存汇总，评估期间在 CPU 保留选中 token 的 double 精度变化值，
空间随选中 target token 数线性增长，并进行精确排序和分位数计算。

#### 实现验证范围

固定 probe 的 19 项专项测试已通过，覆盖真实 tiny-Qwen 前向与普通 logits 对照、
重复评估、跨 checkpoint JSON 比较、runner CLI dry-run、冻结 payload 与身份校验、
8x16 完整 parent 读取以及源文件不变检查。新增分布统计还覆盖均匀与稀疏变化、
正负抵消、token 池化与符号拆分、Top k 取整、空分区及全零变化、严格阈值与极端值、
JSON 有限值、无额外 forward、重复评估和旧报告兼容性。基础版此前通过相关训练、
目标函数、prepared / streaming 恢复回归共 87 项；本次未重跑该整套训练回归。
本次新增统计未重新执行完整 Qwen3.5-9B GPU probe；CPU 专项测试中的 GPU
准入使用替身，实际评分使用小型真实模型。入口与源码见 `RL/probe.py`、
`RL/probe_data.py`、`RL/probe_metrics.py`，测试见 `tests/test_rl_probe.py`。

操作流程见 [训练接口指南](#配置预检和首次训练)，实时监控与独立报告的分工见
[可观测性文档](#实时进度tensorboard-与已冻结-advantage)，目标函数及 anchor 语义见 [GRPO 训练合同](#离线-grpo-分组共享锚点与目标)。

#### 按 group 训练位置绘图

仅评测显式 `--group-id` 或 `--baseline-report` 继承的固定清单，不自动评测全部历史 groups。
新评测可用 `--plot-output` 同时生成 PNG / SVG / PDF；图表复用当前 JSON 指标，不增加 forward：

```bash
.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000400 \
  --group-id GROUP_A --group-id GROUP_B --gpu 2 \
  --output reports/rl-probe/step400-new.json \
  --plot-output reports/rl-probe/step400-new.png
```

已有报告可通过正式绘图接口直接绘图，无需 GPU 或重新评分：

```bash
.venvs/runner/bin/dycon lora-probe-plot \
  --report reports/rl-probe/step400-new.json \
  --output reports/rl-probe/step400-chart.svg
```

输出图表及同名 `.data.json`，后者保留完整 group ID、零基 ordinal、坐标、
训练状态、参与更新的全部 step、逐组指标和源报告 SHA256，便于定位和复核。
输出拒绝覆盖且不能写入训练 attempt / 密封 artifact / `RL/` 树。
绘图需要当前命令所在 Python 环境安装 matplotlib，使用 headless Agg 后端。
若 JSON 已完成而绘图失败，CLI 明确报告已保存的 JSON，可用 `lora-probe-plot` 重试。

**横轴中的点是 group，但位置与训练 step 呈线性比例**：取冻结逻辑训练计划中
该完整 parent 首次参与 optimizer update 的 step（从 1 开始）。例如 step 20 到
40 的距离为 step 20 到 30 的两倍；同一 step 的多个 groups 重合，不做水平抖动。
稀疏选组不按样本序号等距排列，不连接或插值没有评测的 groups。图上标出
checkpoint step 的竖线，小样本标注 group ordinal；完整 ID 见配套数据文件。

所有点均由**当前所分析 checkpoint**评分，纵轴变化均相对原始训练 anchor；
横轴上的历史 step 仅表示 group 在训练中的位置，不是该历史模型的评分结果。
五个面板依次为：正负优势分支平均 Δlogprob、偏好差变化、token 绝对变化幅度
（均值/中位数/P95）、概率相对变化超过 10%/50% 的 token 占比、Top 1%/5%/10%
变化 token 的绝对变化贡献率。前两项按分支等权，后三项在每组内按 token 等权。

默认只绘制 checkpoint 时已经完整训练的选中 groups。需要对照未训练组时，
绘图命令加 `--include-untrained`；评分命令则加 `--plot-include-untrained`。
未训练组在其**计划首次训练 step** 处以叉号显示，部分训练 parent 用菱形显示，
不会冒充已完整训练。8x16 两次半组更新记录首次和末次 step；默认须两个半组
都已提交才算已训练。prepared append 的各 segment 使用实际冻结 window
（包括不足一个 batch 的末尾），跨 GPU 拓扑恢复仍按原始逻辑 window 定位。

新 JSON 新增 `training_schedule`，不改变评分身份或原有 baseline 兼容性，
且可离线复用此坐标绘图。旧 JSON 无此字段时，绘图读取其原 checkpoint，
验证 receipt、step 和冻结计划来补充坐标，不改写原报告；因此旧报告的源
checkpoint、所属 resolved config 和 schedule evidence 必须可读。
旧报告若无 token 分布，前两图正常绘制，其余面板明确标注缺失。

##### 图内数值报告与解读

图表在原五个面板下方默认附带逐组结果表（正负优势变化、偏好差、优势对齐、token
数量、绝对变化、Top 1% 贡献及超过 10%/50% 的概率变化占比）、各训练状态的
分组均值/中位数与改善组数，以及正/负/all 分区的 token 加权汇总。
同时写明样本规模、原始 anchor、teacher-forced T=1、横轴语义、加权口径差异和
训练池诊断边界；有历史基线时展示源报告 aggregate 比较结果，注明其完整源报告范围。
这些数据从实际 JSON 动态汇总，不写死某次 step400 结果；`.data.json` 的 `summary`
保留相同数值。图中逐组表与汇总只包含当前实际显示的 groups。

全部源 groups 均显示且源报告提供 token 分布时，还附上精确池化的 median / P95 /
P99 绝对变化、Top 1%/5%/10% 贡献率及文字解读。过滤后的子集无法从旧报告的
分组汇总还原精确池化分位数/集中度，因此明确标注不可用；不会对 group 分位数
求平均或挪用未显示 groups 的总体分布。部分 group 缺少 token 统计时，对应 cohort
的 token 汇总标记 N/A，不静默只汇总部分 group。接口参数和评分行为保持一致。

#### 数据与资源边界

- 使用 checkpoint 原始共享 anchor，支持 prepared、native streaming 和 legacy。
  8x16 训练也按完整 parent group 评分；不接受半组 ID。
- prepared 复用密封 token / reference artifacts；streaming 需要选中 groups
  的原始 native ledgers 和概率源仍可读，包括训练已消费的 group。
  legacy 需要所属 attempt 的 `frozen-groups.json`，逐组扫描并验证完整
  冻结内容哈希；旧文件很大时这一步可能耗时较长，不会整体读入内存。
  现有 checkpoint / artifact 完整性校验仍会读取关联文件计算校验和；
  额外模型 forward 仅覆盖指定 groups，但启动校验 IO 不保证仅限这些 groups。
- checkpoint 必须位于保留 `resolved-config.json` 的所属 attempt 下。
  模型路径使用该配置，base 身份遵循项目已有注册与文件元数据检查。
- 完整分支评分使用与训练共用的 native target scorer、MLP 分块和 attention
  设置；投影未配置分块时使用 256 token 的有界块，不启用训练专用 prefix
  backward 路径。长上下文仍需显存；OOM 直接报错，不截断或更改样本。
- JSON 写入显式新路径，原子发布且禁止覆盖；输出应放在
  `reports/rl-probe/` 等独立目录，不能放入已密封 attempt / artifact。
- probe 是**训练样本固定子集上的偏好诊断**。优势来自既有训练语义；
  它不能证明 held-out 泛化或下游成功率，绝对分支概率也受文本固有难度影响。
  应主要观察同一 probe 的跨 checkpoint 变化。

### Token K3 诊断

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

#### Online evidence with a hard K3 cap

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

### LoRA 权重导出

`dycon lora-export` verifies a complete project-owned update checkpoint and its
adapter/base binding, then publishes a full model for the existing rollout API.
Run it from `.venvs/rl` (Torch and safetensors); help remains available in runner.

```bash
.venvs/rl/bin/dycon lora-export --checkpoint /absolute/checkpoints/step-000450 \
  --base-model /absolute/models/Qwen3.5-9B --output /absolute/models/new-serving-model \
  --dtype float16
```

There is no implicit latest selection. Only the project's plain, fixed-rank,
bias-free all-linear LoRA export is accepted; DoRA, RS-LoRA and extra saved
modules are rejected. All frozen adapter tensor names/ranks must match the base
shard index. The operation uses CPU FP32 matrix products and addition, then
rounds once to the requested serving dtype. It preserves every base tensor,
including unloaded MTP states, and copies tokenizer/template/config assets.
FP16 exported weights keep ordinary FP16 serving for both experimental arms;
this is a merged inference representation, not bitwise BF16 training forwards.

Publication uses a same-filesystem staging directory and refuses overwrite.
`lora-export-receipt.json` records the exact source checkpoint/adapter hashes,
base identity, compute/output dtype and output file hashes. The final model
is registered through `core.model_identity.register`; put the returned path and
fingerprint into root config before creating a run through `dycon run create`.
Export uses no GPU and does not start or modify training or rollout.

`--dry-run` verifies source identities and reports the plan without importing
Torch or writing a model. It does update the existing base identity registry.
The export receipt remains in the model directory. Training resume must continue
using the original complete checkpoint, not the merged serving directory.

## 验证范围

既有工程验证包括 LoRA 的短 native/synthetic 单卡与双卡 fresh/resume、同 dtype 外部 anchor，以及历史 v10 只读核验。真实训练仍应核对目标 checkpoint 的 dataset、token、optimizer、RNG 和 anchor 身份。源模型训练、下游 benchmark 质量和真实 FP16→BF16 概率偏差分别验证。

### Baseline epoch extension

For a constant-scheduler baseline-only checkpoint, set run.start.mode=resume and run.start.extend_baseline_epochs=true. Increase offline_baseline.epochs to the cumulative target and training.max_updates accordingly (or omit it). Only epoch/update targets may change; dataset, objective, optimizer settings and topology remain unchanged. Optimizer, scheduler, RNG, global step and baseline sampling cursor are restored. A completed 3-epoch/168-step run extended to 5 epochs resumes at step 168 and stops at 280.

Frozen initial-policy KL can be recomputed during LoRA training independently of grouping; see [implementation and compatibility](../docs/unsorted/frozen-kl-recompute.md). The PPO denominator remains separately configured by each algorithm.
