# RL 执行与资源手册

先选 LoRA 或全参 FSDP2，再选择 GPU 拓扑、分支调度和内存策略。训练入口与生命周期见[训练手册](README.md)，原生 GRPO 输入见[数据与概率手册](dataset/README.md)。

| 模式 | 执行约束 |
| --- | --- |
| LoRA 完整父组 | 1/2/4 卡，可用完整分支或显式开启 `boundary_grad` |
| LoRA 8×16 | 默认 exact 八卡，完整 32 分支父组评分，两个逻辑半组分别消费 |
| LoRA 跨拓扑继续 | 显式 `preserve_updates` 支持 2/4/8 卡，保留原逻辑更新，要求训练不消耗 RNG |
| 全参 FSDP2 | 固定物理拓扑，可用 reference 或 BF16 FLA，8×16 可选 2/4/8 卡 |

## 详细参考

- [全参训练](#全参训练) · [全参 FSDP2 配置、执行及检查点](#全参-fsdp2-配置执行及检查点)、[FLA 后端、依赖及数值边界](#fla-后端依赖及数值边界)、[全梯度对比](#全梯度对比)
- [多卡执行](#多卡执行) · [完整父组数据并行](#完整父组数据并行)、[两轮 distinct parent 8×16](#两轮-distinct-parent-816)、[共享前缀执行](#共享前缀执行)、[保留优化窗口的跨拓扑恢复](#保留优化窗口的跨拓扑恢复)
- [显存与主存](#显存与主存) · [长上下文和激活卸载](#长上下文和激活卸载)、[主存预算、固定页分配器和压力策略](#主存预算固定页分配器和压力策略)、[全参 FSDP2 的受控激活卸载](#全参-fsdp2-的受控激活卸载)、[全参 Adam moment 动态驻留](#全参-adam-moment-动态驻留)

## 全参训练

### 全参 FSDP2 配置、执行及检查点

`dycon rl-train --training-config YAML` uses the existing RL service, GPU locks,
attempt lifecycle and distributed worker supervisor. Schema 1 remains LoRA;
schema 2 selects `tuning.mode: full`. The legacy `lora-train` command rejects full
configurations. Help, configuration parsing and training dry-runs do not load Torch.

The scope includes embedding, every decoder parameter (including GDN non-Linear
parameters), final norm and lm_head. Shared parameters are optimized once.
Vision remains frozen in the immutable base files and is included in HF exports.
Unknown module ownership fails explicitly.

#### Supported configuration

See the complete, machine-specific short acceptance configurations:

- [pause after update 1](experiments/full-9b-fsdp2-acceptance-pause.yaml)
- [resume through update 2](experiments/full-9b-fsdp2-acceptance-resume.yaml)
- [FLA pause after update 1](experiments/full-9b-fla-fsdp2-acceptance-pause.yaml)
- [FLA resume through update 2](experiments/full-9b-fla-fsdp2-acceptance-resume.yaml)
- [FLA selected native capacity measurement](experiments/full-9b-fla-fsdp2-native-capacity.yaml)

The pause/resume examples use a synthetic fixture, BF16 computation, FP32 master weights, FP32 AdamW
moments and FP32 gradient reduction. They are engineering acceptance examples,
not a production dataset or benchmark-quality training recipe. The learning rate
is explicitly 1e-6. Select idle GPUs and new attempt IDs for subsequent runs.

Required schema 2 fields:

```yaml
tuning: {mode: full, scope: language_model, freeze_vision: true}
execution: {backend: fsdp2, sharding: full_shard, topology_policy: exact}
optimizer: {name: adamw, master_weights_dtype: float32, states_dtype: float32}
```

BF16 matmul numerics are fixed to the existing defaults:
`bf16_allow_reduced_precision_reduction: false`, `bf16_allow_split_k: true`.
The full loader configures deterministic algorithms, disables TF32 and selects
cuBLASLt through the same numerical setup used by probes.

The initial milestone supports dense Qwen3.5, prepared offline GRPO, complete
branches, explicit `reference_pytorch` or BF16 `fla_chunk`, disabled prefix reuse, layer checkpointing and
sequence lengths up to 2048 in the historical default configuration.
Full-policy 8x16 uses the same frozen two-pass execution plan as LoRA on 2/4/8 ranks, with complete 32-branch parents split into two 16-branch updates. `training.warmup_updates` is optimizer-mode agnostic and defaults to zero. The executor requires prepared data, exact topology, layer checkpointing and disabled prefix reuse, MLP chunking and OOM retry. It preserves complete native branches and rejects inputs beyond the configured length limit. See [decoder checkpoint activation offload](#全参-fsdp2-的受控激活卸载) and [dynamic Adam moment residency](#全参-adam-moment-动态驻留) for host-memory options.

Select FLA with `model.gdn_backend: fla_chunk` and `model.dtype: bfloat16`.
The existing pinned FLA/core 0.6.0 adapter changes only GDN chunk execution.
Convolution, normalization, the FSDP2 strategy and optimizer precision keep their
existing implementations. This is runtime acceptance, not cross-backend numerical
equivalence or training-quality acceptance. Historical failed numerical audits
remain applicable; see [FLA backend](#fla-后端依赖及数值边界).

For bounded native capacity measurements, `algorithm.group_ids: [ID, ...]`
explicitly selects complete parents in a full-policy task YAML. The shared
preparation path retains every branch and exact token coordinates; unknown or
duplicate IDs fail. The selection is frozen in the schedule and input proof.
The empty default keeps the existing dataset-wide behavior and historical
schedule identity. This selection currently requires full-policy prepared training.
Length limits still reject overlong selected parents without truncating them.

##### Two-pass 8x16 execution

Full-policy training reuses the frozen `distinct_parent_two_pass_v1` execution
plan and complete-parent scoring from [8x16](#两轮-distinct-parent-816). Configure
`group_batch_size: 8`, `parallel_branch_chunk_size: 16`,
`parallel_branch_schedule: distinct_parent_two_pass_v1` and
`prefix_reuse: {mode: disabled}`. Use `dycon rl-train` with 2, 4 or 8 physical
GPUs and `runtime.topology_policy: exact`. FSDP2 topology migration remains
unsupported; selecting fewer GPUs for a fresh attempt retains the logical plan.
Each rank processes its assigned parent halves in synchronized 16-branch rounds.
Every branch has weight `1/(16*K)` for an actual K-parent window, with one
world-size compensation for FSDP's mean reduction. Empty slots run a locally
loaded branch with zero loss, preserving collectives without counting extra
branches. The optimizer advances once per complete window. The execution cursor
tracks consumed halves separately from completed parents, including each tail.
Shared-prefix graph reuse is a separate optimization and remains unavailable in
the full-policy executor.

##### Shared learning-rate warmup

`training.warmup_updates` applies identically to LoRA and full training, measured
in successful optimizer updates, not branches or microsteps. It defaults to zero
and must not exceed `max_updates`. With N warmup updates, update u (1-based,
u<=N) uses `learning_rate*u/N`. `scheduler: constant` then keeps the base rate.
With `linear`, the remaining updates decay linearly: update u>N uses
`learning_rate*(max_updates-u+1)/(max_updates-N+1)`; the scheduler reaches zero
after the final update. Zero warmup preserves the historical linear schedule.
Scheduler state resumes at the saved optimizer step; exact resume must retain
warmup configuration. Warm start resets the optimizer and warmup. Omitting the
field or setting it to zero preserves old schedule-contract identities.

#### Execution and numerical contract

Parameters are built on meta, explicitly converted to FP32 metadata, sharded by
decoder block and root, then populated directly from bounded safetensor slices.
There is no full CPU model or per-rank full GPU model during loading. Root owns
embedding/head together to preserve tying. AdamW excludes embedding, norm and
bias from weight decay and covers every unique trainable parameter exactly once.

Each logical window has one frozen branch microstep table. Every rank executes
the same number of forwards/backwards; missing assignments use a zero-contribution
branch. Branch losses retain parent/branch means, with exactly one world-size
compensation for FSDP averaging. The old LoRA manual gradient reduction is bypassed.
Global clipping reduces shard squared norms. The differentiable chunked output
projection updates lm_head as well as the entire decoder.
Finite zero gradients are valid when ratio guards or clipping saturate the whole
window; missing gradients and nonfinite norms fail explicitly.

Fresh training freezes raw T=1 target logprobs before any update. Reference
identity includes base/tokenizer, parameter inventory and initial shard hashes.
External raw probability streams must match the initial base scoring identity;
warm_start builds a new model-forward anchor. Resume retains the original anchor.

#### Persistence, export and inspection

`rl-full-checkpoint-v1` stores local FP32 weights/moments in bounded tensor files,
plus rank RNG, scheduler, optimizer groups, logical cursor and complete contracts.
All ranks finish staging before the primary commits the receipt and atomically
publishes the directory. A failed/incomplete staging directory is not resumable.
Same world size, backend, schedule, model and numerical environment are required.
Warm start loads policy weights and resets optimizer/scheduler/cursor/reference.

Full FLA checkpoints additionally freeze `gdn_execution` in the training and
initial-anchor identities. Workers compare the actual binding (including pinned
Qwen/FLA source digests, versions and state precision semantics) before loading.
Every rank writes sealed `gdn.json` identity/call evidence; artifact verification
rejects missing evidence or zero chunk calls. Reference-to-FLA exact resume fails.
Historical reference checkpoints remain readable. An explicit warm start may
import weights under a new backend and builds a new anchor/optimizer identity.
Metrics record all-parameter gradient validation, every local FP32 master/moment
finite check, and per-rank GDN call observations. Probe binds the checkpoint
execution identity; HF exports carry it in `full-policy-export.json` so independent
reload callers can install the same binding before constructing Transformers.

Independent reload must call `install_gdn_backend(config.model,
expected_identity=metadata['contract']['gdn_execution'])` before standard HF
`from_pretrained`. Ordinary HF checkpoint files do not install the project
adapter automatically.

```bash
.venvs/runner/bin/dycon rl-status ATTEMPT --storage-root ROOT
.venvs/runner/bin/dycon rl-advantages ATTEMPT --storage-root ROOT --group-ordinal 0 --limit 3
.venvs/runner/bin/dycon artifact verify ATTEMPT --scope rl-full-attempt --storage-root ROOT
.venvs/runner/bin/dycon rl-export --checkpoint CHECKPOINT --base-model BASE --output NEW_DIRECTORY --dtype bfloat16
.venvs/runner/bin/dycon rl-probe --checkpoint CHECKPOINT --group-id GROUP --gpus 5 6 --output REPORT.json
```

`rl-tensorboard` reuses the existing synchronizer and lease management. Per-rank
memory and sampled parameter deltas are in `metrics.jsonl`; `policy_changed`
records bounded sampling evidence, not a full-tensor equality proof. Checkpoint
and export receipts record all file sizes/hashes. Export streams shards on CPU
and restores the frozen vision weights from the fingerprint-matched base model.
It never gathers the whole model on a rank. Probe uses the original frozen anchor.

### FLA 后端、依赖及数值边界

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
size, layer checkpointing, and FP32 masters, moments and reduction. The
[activation settings](#全参-fsdp2-的受控激活卸载) give requirements for
lengths above 2048. Full checkpoints and probes preserve actual GDN identity
and sealed per-rank call evidence.

#### Environment

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

#### Behavior and checkpoints

`RL/gdn_backend.py` installs the binding before model construction. Only the
GDN chunk operator changes; recurrent single-token execution and causal
convolution use reference implementations. The chunk adapter preserves Qwen's
gate/beta semantics, the single scale factor, chunk size 64, differentiable
initial/final state, and `[B,HV,K,V]` state layout. No state is detached.

Training artifacts record configured/resolved backend, versions, source hashes,
state semantics, and observed calls. Workers must agree on the backend identity.
Resume requires the same backend contract. Historical reference artifacts remain
readable; reference-to-FLA exact resume is not supported.

#### Runtime checks

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

### 全梯度对比

#### Full cross-backend gradient diagnostic

`dycon rl-gradient-compare --comparison-config YAML [--dry-run]` compares a
complete prepared parent from an explicit fresh BF16 full-policy training YAML.
The comparison YAML has `schema_version: 1`, `training_config`, `group_id`,
`output` (a new directory below project `temp/`) and optional `modes`:
`fixed_cotangent`, `shared_reference_grpo` (both by default). See
`experiments/full-9b-gradient-comparison.yaml`.

The formal command owns the usual GPU locks and worker supervisor. Backend
phases run in fresh workers at the same topology. Full FP32 parameter shard
digests must match before comparison. The reference backend supplies one shared
initial-policy logprob anchor. Fixed cotangents are `-advantage / target_count`,
with the standard world-size/parent/branch loss weighting; actual GRPO reuses
`window_update` and the same shared reference, preserving ratio/clipping effects.
Both compare full FP32 reduced gradients before clipping. There is no optimizer,
weight update, sequence truncation or CPU offload. Vision remains frozen.

FP64 reductions cover every gradient element and report reference/actual norms,
difference norm, relative L2, cosine and maximum absolute error for all language
parameters, each tensor and parameter categories. Relative L2 means
`norm(actual-reference)/norm(reference)`; a zero reference norm with a nonzero
difference is explicitly undefined. No cross-backend equivalence threshold is
implied by successful completion. Temporary reference gradient tensors are
removed only after all modes finish; the final receipt seals remaining reports
and the cleanup ledger. An earlier phase receipt records those deleted tensor
hashes and cannot subsequently be deeply verified as a standalone phase.

## 多卡执行

### 完整父组数据并行

新增显式 `runtime.topology_policy: preserve_updates`：完整 parent 与原 two-pass 计划均可在 2/4/8 张物理 GPU 间双向恢复，保持原 update windows、权重、anchor 和 optimizer 状态。默认 exact 模式仍遵守本文的原卡数限制。详见 [跨卡数恢复](#保留优化窗口的跨拓扑恢复)。

默认 `exact` 策略下，`dycon lora-train --training-config YAML` 支持单机 1、2、4 张物理 GPU 的同步
LoRA 数据并行。每张 GPU 保留完整 base 与 adapter，计算单位是完整 group。
本功能适用于现有 `offline_grpo_k3_v1`，包括 `prefix_reuse.mode: boundary_grad`；
单条轨迹的容量仍由单卡显存决定。

#### 配置与启动

旧单卡配置 `runtime.gpu: 0` 保持兼容。多卡时，用 `runtime.gpus` 替换
`runtime.gpu`，或将 `gpu` 显式设为 `null`：

```yaml
runtime:
  gpu: null
  gpus: [1, 2]             # 也支持 [1]、[0, 1, 2, 3]
  distributed_timeout_seconds: 3600
  output_root: /data2/wangtianyi/DyConGlobal
grpo:
  group_batch_size: 4     # 全局每个 optimizer update 的 group 数
  prefix_reuse:
    mode: boundary_grad
training:
  gradient_accumulation_steps: 1
```

示例只展示相关字段，其他训练、数据、reference 配置保留原完整 YAML。
默认 exact 完整父组模式的 GPU 列表长度为 1、2、4，不能重复；`gpu` 与 `gpus` 必须二选一。
多卡暂不接受 MIG UUID 或 causal CE 配置。通信超时涵盖初始化、rank 0
准备 reference 期间的等待、训练同步和最终收尾；长时间 reference 构建需要
相应增大该配置，超时范围为 30–86400 秒。

1. 在 `RL/experiments/` 中建立任务配置，选择新 attempt ID 与空闲 GPU。
2. 设置 `run.mode: dry-run`，通过 `.venvs/runner/bin/dycon lora-train
   --training-config PATH.yaml` 检查配置。dry-run 不导入 Torch、不加载模型；
   `memory_plan.local_world_size` 来自显式 GPU 列表。
3. 设置 `run.mode: smoke`，通过 `.venvs/rl/bin/dycon lora-train
   --training-config PATH.yaml` 执行实际训练。沿用项目现有 mode 命名。

service 获取全部 GPU 锁并内部启动 workers。每个 worker 只看到自己的一张 GPU。
不要额外套用 `torchrun`，也不要启动多份独立训练命令来模拟数据并行。

#### 一次更新的含义

每步仍为全局 4 个 group、每组 32 条分支，共 128 条。GPU 数不放大全局 batch：

| GPU 数 | 各 rank 在常规窗口中计算的 group 序号 | 全局每步分支数 |
|---|---|---|
| 1 | rank 0: 0,1,2,3 | 128 |
| 2 | rank 0: 0,2；rank 1: 1,3 | 128 |
| 4 | rank 0/1/2/3: 各自对应一个 group | 128 |

窗口、group 顺序、组内分支、冻结 advantage 和 reference 都遵循冻结的 prepared
artifact 或 native streaming plan/source binding。每卡执行原来的 `begin_group → branch backward → finish_group →
close_group`，保留组内共享前缀前向和合并前缀反向。

损失仍按 token mean → branch mean → **全局实际 group mean** 聚合：

```text
L = (1/G) Σ_g (1/N_g) Σ_b mean_t(loss_gbt)
```

各卡只计算自己 group 的上述贡献，全部完成前缀反向后，以固定参数顺序将
LoRA 梯度转换为 FP32、分桶 SUM all-reduce。此处不再除以 GPU 数。
随后各副本对相同梯度裁剪，分别执行相同的 AdamW/scheduler step，形成一次逻辑更新。
每步校验所有副本 adapter 指纹相同，再推进全局 cursor 和发布指标。

尾窗口按实际 group 数 G 归一化，全部 group 必须消费；无 group 的 rank
仍参与同步。全局均未使用的参数保持 `grad=None`，避免意外触发 AdamW
weight decay。共享前缀边界与计算图完全保留在所属 GPU，不跨卡传送。

#### 产物、恢复和异常

- rank 0 唯一发布全局 metrics、checkpoint 和 receipt。Prepared 路径由统一
  preparation 服务创建 indexed prepared/reference artifact，各 rank 读取同一
  不可变产物。Native streaming fresh/resume 无需全量 prepared/reference cache；
  各 rank 使用同一冻结 plan、advantage、原 reference anchor 和 source binding，
  从已提交 cursor 准备剩余窗口。async=true 不会把 prepared resume 转为 streaming。
- checkpoint 保存一份相同的 adapter/optimizer/scheduler、全局 cursor，以及
  **每个 rank** 的 Python/NumPy/CPU/CUDA RNG。只在完整更新且梯度清空后发布。
- 默认 `exact` 的 `resume` 要求相同 world size 和执行协议；可换同数量的物理 GPU，但必须
  保持每个 rank 的 CUDA 可见设备数与保存时一致，不做 RNG 设备列表重映射。
  各 rank 在启动 producer/forward 前恢复 optimizer/scheduler 和自己的 RNG。
  旧单卡 checkpoint 与多卡 checkpoint 不能直接互换。
- `append_resume` 仅支持原 prepared 路径，同样要求原 world size。Streaming
  checkpoint 只支持 exact resume，不支持 warm_start/append_resume。其他支持
  的非 streaming adapter warm-start 仍重置 optimizer/cursor 并重新绑定 anchor；
  不能用它冒充继续旧步数或保留旧 initial-policy anchor。
- 单卡与多卡数学目标一致；浮点累加顺序不同，不承诺跨 GPU 数逐位相同。
  BF16 adapter 对照验收固定使用 `atol=2e-4, rtol=2e-3`；这只描述跨设备
  归约顺序造成的数值差异，不放宽 loss、梯度有限性或全局 group 计数检查。
- 非有限梯度、漏算/重复 group、副本不一致或 worker 异常会终止当前任务。
  失败窗口不发布可恢复 checkpoint，从最近完整 checkpoint 重新执行。
  supervisor 负责停止、回收其他 workers 并释放全部锁；通信也有显式超时。
  SIGTERM/SIGINT 触发清理，SIGKILL 由操作系统强制终止，无法执行 Python 清理。

`metrics.parallel` 包含 world size、梯度同步时间和每卡 group/branch/token 数、
计算时间、显存峰值。顶层 loss/ratio/K3 和 `prefix_reuse` 汇总全局贡献；
分位数从实际 token 数据重建。现有 profiler、forward/backward 分段和顶层
显存指标对应 rank 0，在 `parallel.local_telemetry_rank` 明确标注。

查询及核验继续使用 `lora-status`、`lora-tensorboard` 和
`artifact verify ATTEMPT_ID --scope lora-attempt --storage-root ROOT`。

### 两轮 distinct parent 8×16

Full-policy FSDP2 now consumes the same frozen two-pass plan through
`dycon rl-train` on 2/4/8 ranks. It uses disabled prefix reuse, exact physical
topology, branch-round collectives and the FSDP mean-reduction compensation;
the LoRA-specific execution details below retain their original meaning.
See [full-policy configuration](#two-pass-8x16-execution).
The shared `training.warmup_updates` option is independent of execution and
parameter tuning mode.

Physical-topology exception: `runtime.topology_policy: preserve_updates` can execute this same frozen two-pass plan on 2/4/8 GPUs, or split a whole-parent plan across eight GPUs without adopting two-pass semantics. Default exact mode retains the original restrictions below. See [topology switching](#保留优化窗口的跨拓扑恢复) for configuration, RNG restrictions and validation boundaries.

The default `exact` implementation of this explicit opt-in mode uses the existing `dycon lora-train --training-config
YAML` lifecycle, GPU locks, eight supervised workers, preparation, checkpoint
publication and artifact verification. It supports native step streaming and
indexed prepared input, on one node with eight distinct physical GPUs.

```yaml
runtime:
  gpu: null
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
grpo:
  group_batch_size: 8
  parallel_branch_chunk_size: 16
  parallel_branch_schedule: distinct_parent_two_pass_v1
  branch_partition_seed: 42
  prefix_reuse:
    mode: boundary_grad
```

Merge these fields into a complete task YAML, use a new attempt ID, and retain
`training.checkpoint_policy: layer`. `runtime.data_preparation_async: true`
selects native streaming for fresh input without a prepared binding; the normal
prepared route is also supported. Run runner-environment dry-run before an
actual `.venvs/rl/bin/dycon lora-train` invocation. Do not wrap it in torchrun.

In default `exact` mode, eight GPUs without splitting and splitting on 1/2/4 GPUs
are rejected. `preserve_updates` permits the original two-pass plan on 2/4/8
physical GPUs and whole-parent execution on 2/4/8 GPUs without converting plans.
Other chunk sizes,
non-boundary execution, fewer than eight eligible parents or an eligible parent
with other than 32 branches are rejected. Required structural checks happen
before model loading regardless of optional preflight flags.

#### Frozen scoring and execution

Complete 32-branch native parents remain the scoring and anchor identity unit.
The existing implementation computes eligibility, retained/ranks and advantages
once. Halves preserve scores, advantages, masks, references and branch IDs;
there is no half-level reranking, cutoff, normalization or zero-sum correction.
Random half advantages can have nonzero sum.

`sha256_ordinal_permutation_v1` sorts native ordinals by the digest of partition
algorithm, partition seed, stable parent ID and ordinal. The first 16 form half
0 and the rest half 1; execution within each half uses sorted native ordinals.
This avoids process hash randomization and traversal-order dependence.

`distinct_parent_two_pass_v1` executes all half-0 chunks and then all half-1
chunks. Each round independently uses the existing prefix-length candidate-pool
ordering, with a frozen SHA256-derived round seed. Every regular window has
eight distinct parent IDs. Round tails remain separate, with no padding,
resampling or discarded branch. Different parent IDs need not have unique prompt
text. Parent prompt deduplication is not introduced.

For G parents, preparation freezes 2G chunks and `2*ceil(G/8)` update windows.
For G=6773, there are 1694 updates, 13546 chunks and 216736 branches, with two
5-parent / 80-branch tails. The partition, parent metadata, windows, world size,
assignment and loss protocol are hashed into a distinct execution identity.
Ranks read that identity; they do not partition using mutable runtime settings.

#### Objective and cursor

For an actual K-parent window, the loss is the mean of K selected 16-branch
means; each branch first uses its completion-token mean. Branch weight is
`1/(16*K)`, including tails. FP32 SUM all-reduce combines already normalized
contributions without dividing by world size. Gradient presence, unused
parameters, AdamW decay, clipping and replica fingerprint checks still apply.
Idle ranks participate in synchronization.

At fixed parameters, the average of a parent's two half means equals its full
mean. Actual updates occur between halves; this is a new optimizer trajectory,
not exact-equivalent 4x32 training. The same trajectory budget does not guarantee
identical tail weighting or improved downstream quality.

`execution_window_cursor_v1` binds committed windows, round, schedule hash,
consumed half identity hash and branch/token counts. Checkpoints publish only
after a complete optimizer update with cleared gradients; some parents may
still have an outstanding half. Progress distinguishes planned parents/chunks,
consumed chunks, seen distinct parents and completed parents. Legacy
`group_cursor` / `consumed_groups` explicitly project completed parents only.
No prefix graph or boundary-gradient buffer is saved.

Exact resume retains all-rank RNG, adapter, optimizer, scheduler, mapping and
reduction protocol, with the same world size. Physical cards may change.
Old 1/2/4-rank or v10 checkpoints cannot resume directly as 8x16. Any supported
new-stage adapter warm-start resets optimizer/cursor and binds the actual new
reference anchor; streaming warm-start and 8x16 append-resume are unsupported.
No silent fresh fallback is allowed.

With `runtime.topology_policy: preserve_updates`, an existing two-pass checkpoint
can instead resume on 2/4/8 GPUs. Each rank executes its assigned original halves
before the same single optimizer update; rounds, weights and incomplete-parent
cursor remain unchanged. The plan's historical `world_size: 8` remains a logical
identity. Actual worker count is saved separately, including the saved rank RNG
cardinality. This policy rejects training RNG consumption. A whole-parent
checkpoint can use eight GPUs under this policy while retaining its whole-parent
schedule; it does not become an 8x16 checkpoint. See [operation](#保留优化窗口的跨拓扑恢复)
and [validation](#验证范围).

#### IO, telemetry and validation boundary

Streaming freezes complete parent metadata, then reads only this rank's 16
native completions/reference arrays per half. Other ranks' payloads are metadata
placeholders. Parents are revisited in the second round, so prompt/native IO is
repeated. Prepared input retains complete indexed parent artifacts; only selected
completions are read during execution. Its reference chunks are memory-mapped
and selected targets copied. Integrity verification may read sealed bytes in
either path. Source fingerprints and access to unconsumed inputs remain required.

Prefix telemetry is explicitly drained per update attempt, including failures,
and carries parent ID, half, native ordinals, round/window and rank. Status and
TensorBoard distinguish parent/chunk counters. Elapsed update time and rank
critical path are separate from summed synchronized segment times. Suffix work
is reduced per rank, while prefix graphs and boundary buffer shapes are not
halved; a short fixture cannot prove capacity for long v10 prefixes.

CPU validation uses `tests/test_grpo_execution.py`, actual 8-process Gloo in
`tests/test_grpo_parallel.py`, real tiny Qwen checkpoint/resume, and historical
1/2/4-rank regression tests. GPU acceptance and measured performance remain
pending while the eight cards have existing workloads. This mode remains
experimental; there is no measured speedup or quality claim.

GPU acceptance must use fresh attempts under
`temp/data-parallel-8x16-acceptance/`, the formal CLI, Qwen3.5-9B, all eight idle
cards, tails/cross-round/pause-resume, rank fingerprints and artifact verify.
Separately test representative v10 long-prefix memory. Repeat the same-data,
same-total-branch-budget 4x32 versus distinct-parent 8x16 comparison twice with
short/long prefixes; report median/P90, branches/s, tails, total/critical-path
time, sync, IO/reference, allocated/reserved memory, startup/checkpoint cost and
failure rate. The suggested 10% throughput gate is unverified. Training quality
requires separate common-evaluation and multi-seed comparisons.

The opt-in public-CLI tests are prepared in `tests/test_grpo_execution_gpu.py`:

```bash
DYCON_GRPO_8X16_GPU=1 DYCON_GRPO_GPUS=0,1,2,3,4,5,6,7 \
DYCON_8X16_EVIDENCE="$PWD/temp/data-parallel-8x16-acceptance/gpu-NEW-ID" \
  .venvs/rl/bin/python -m pytest -q -s tests/test_grpo_execution_gpu.py
```

Run only after all selected cards are idle. Choose a fresh evidence directory.
This covers both data routes, all four boundaries of a nine-parent epoch,
all-rank RNG and artifact verification. Set `DYCON_8X16_PERFORMANCE=1` for two
repeats of matched 512-branch 4x32/8x16 plans at 64- and 4096-token synthetic
prefixes. These use an engineered external raw-probability fixture and do not
establish training quality or v10's actual long-context capacity. The separately
prepared `v10data-8x16-capacity-PENDING.yaml` has not been executed; it is a fresh
stage on the full historical source, with a one-update pause and a fresh anchor,
not a continuation of the old optimizer.

### 共享前缀执行

Offline GRPO can reuse the identical prompt prefix inside each group while preserving the original group-weighted objective and LoRA gradient flow. Enable it explicitly:

```yaml
grpo:
  prefix_reuse:
    mode: boundary_grad
```

The default mode keeps complete-branch execution. Use a new attempt ID for every invocation. The BF16 example is `RL/experiments/grpo-shared-prefix.yaml`.

#### Execution

At group start, the backend executes the common prompt once and retains one read-only source boundary. The boundary contains full-attention K/V state, linear-attention convolution/recurrent state, and the final normalized prompt hidden state in stable layer order.

Each branch receives independent `detach().clone()` leaves. Native Qwen decoder, rotary, convolution, gated-delta and output-projection operators consume those leaves through a functional cache facade. Source tensors are version-checked and never modified. The first completion target uses the final prompt hidden state; subsequent targets use the correct absolute positions and causal state. Completion length one is supported without a suffix backbone pass.

Branch execution keeps the existing reduction: token mean, branch mean, then group mean. Branch backward accumulates direct adapter gradients and detached boundary gradients. Boundary gradients are accumulated in FP32. After the last branch, one multi-output autograd backward sends the accumulated gradients into the original prefix graph. The group source, leaves, temporary hidden states, losses and branch graphs are then released. Adapter gradients remain available for the rest of the optimizer window.

The training loop has explicit `begin_group`, `after_branch_backward`, `finish_group` and `close_group` lifecycle hooks. Exceptions and nonfinite gradients clear the partial window and prevent an optimizer update. Prefix graphs, branch leaves and boundary-gradient buffers are transient and are not written to checkpoints.

#### Checkpointing and metrics

The backend uses existing non-reentrant layer checkpointing. Boundary tensors are explicit checkpoint inputs, and recomputation creates a fresh functional cache facade. Cached suffix SDPA uses a symbolic lower-right causal bias, preserving absolute positions without constructing a dense attention mask.

Each update records prefix length, branch count, root and branch backward counts, source/leaf/gradient logical bytes and dtypes, prefix/suffix/state-copy/gradient timings, branch and group allocated-memory observations, and layer execution counts. Existing loss, ratio, K3, clipping, gradient norm and consumed-token metrics remain unchanged.

The feature is covered by CPU and tiny-Qwen tests for variable completion lengths, one-token targets, GDN chunk boundaries, 2/3/4/32 branches, unequal group windows, gradient accumulation, checkpoint recomputation, failure cleanup, unused boundary states and exact resume. The formal GPU path uses the same `dycon lora-train` service and BF16 model configuration.

#### MIG devices

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


#### Opt-in 8x16 execution

In [8x16 mode](#两轮-distinct-parent-816), the prefix lifecycle belongs to one
execution half, while eligibility, retained/ranks, advantages and the reference
anchor belong to its complete 32-branch parent. Each half recomputes the prefix
in a different update. The objective uses token mean, 16-branch mean, and actual
window-parent mean; no half is re-ranked or re-centered. Prefix rows are drained
after each local update attempt and cleared on failure, independently of profiler
callbacks. Rows carry parent/half, native ordinals, round/window and rank.
Shrinking 32 suffixes to 16 does not halve prefix graphs or boundary buffers and
does not establish capacity for v10's longest prompts.

### 保留优化窗口的跨拓扑恢复

Use the existing `dycon lora-train --training-config YAML` interface and a new
attempt ID. The opt-in runtime policy changes the physical assignment of work;
it preserves the source's logical training protocol, frozen windows, advantages,
initial anchor, optimizer/scheduler and committed cursor.

```yaml
run:
  attempt_id: YOUR_NEW_ATTEMPT_ID
  mode: dry-run
  # Retain the source seed and all other frozen training fields.
  start:
    mode: resume
    checkpoint: /ABSOLUTE/ATTEMPT/checkpoints/step-NNNNNN
runtime:
  gpu: null
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]  # Or two/four explicit physical devices.
  topology_policy: preserve_updates
  # Retain the source output/input settings, except an authorized output move.
```

This is a fragment to merge into the source's complete training YAML. Keep the
original algorithm, global group batch, schedule, prefix settings, probability
source, preflight switches, memory policy and original total update count.
For streaming checkpoints retain `data_preparation_async: true` and
`prepared_data: null`; their unconsumed native ledgers/probabilities must remain
available. Prepared checkpoints retain their immutable prepared/reference input.

Run runner-environment dry-run first:

```bash
.venvs/runner/bin/dycon lora-train --training-config PATH_TO_NEW_YAML
```

Dry-run checks the source topology/protocol and existing metadata contracts; its
`physical_execution` describes the target and `training_rng_validation` explicitly
remains pending. Change only the new YAML's `run.mode` to `smoke` for actual
execution through `.venvs/rl/bin/dycon lora-train --training-config ...`.
Do not wrap the CLI in torchrun. Select idle GPUs or first pause the source at a
complete checkpoint and confirm its process/resource cleanup.

#### Supported plans and weights

Both directions, 2↔4↔8, support indexed prepared and native streaming checkpoints.
The default `topology_policy: exact` retains the old same-world-size admission.
Once portability is adopted, retain `preserve_updates` on subsequent resumes,
including same-card-count resumes.

**Whole-parent plan:** the original parent windows remain unchanged. Physical
tasks contain at most 16 native branches; tasks are assigned in frozen parent /
native-branch order using `task_ordinal % current_world_size`. A parent with N
branches in a K-parent window retains branch weight `1/(K*N)`, including
non-32-branch parents and tails. All halves finish at the same parameters before
one update. Eight cards do not turn this into the distinct-parent two-pass plan.

**Distinct-parent two-pass plan:** reuse the original sealed half partition,
two rounds, windows, tail weights `1/(16*K)`, and execution-window cursor. On
two/four GPUs each rank executes multiple original half tasks before the same
single update. Incomplete parents at a round-0 checkpoint remain incomplete;
already committed halves are never replayed. The historical execution plan's
`world_size: 8` remains part of its immutable identity, not the current physical
worker count.

Gradients use the existing FP32 SUM reduction, followed by one global clipping,
AdamW step and scheduler step. Never average again by physical world size.
Idle ranks participate in synchronization. Whole-parent rows are reconstructed
by native branch ordinal; missing/duplicate/misassigned contributions reject the
update. Objective metrics retain parent weighting rather than averaging chunks.

Each split parent task recomputes its own prefix with the existing backend;
only parameter gradients cross devices. No prefix graph or boundary buffer is
migrated. More GPUs can increase prefix computation, so speedup must be measured.
Portable whole-parent streaming currently preserves the existing full-parent
CPU token loading; reference arrays are imported only for locally used parents.
This is not a claim of minimal per-rank CPU IO or memory.

#### Checkpoint identity and RNG

Historical checkpoints and their receipts are never modified. New attempts keep
the original logical `contract.parallel` and original training/input contracts.
The target is recorded separately in:

- `topology-transition.json`, copied into new checkpoints and covered by receipts;
- `state.json.physical_execution`, covered by checkpoint inventory hashes;
- committed metric `physical_execution`, alongside the actual `parallel.world_size`.

The transition records the source checkpoint receipt and source/target physical
topology. A later resume resolves the actual number of saved rank RNG states
using `physical_execution`, not the historical logical contract world size.

This first version accepts only **training computations that do not consume RNG**.
Model audit rejects nonzero dropout/known stochastic attention settings. Every
physical forward/backward task snapshots Python, NumPy, CPU Torch and visible
CUDA RNG and rejects the entire optimizer window if any advances. An arbitrary
stochastic training callback is therefore unsupported. Existing exact-mode
stochastic resume retains its original behavior.

The source rank RNG states remain saved. On a topology change, existing rank
states are retained where possible; additional ranks inherit bookkeeping states
cyclically. This does not represent a preserved stochastic trajectory: correctness
depends on the enforced no-training-RNG contract. CUDA-visible-device cardinality
must still match saved per-worker state. Supporting dropout/random training
would require a separately versioned logical-task RNG protocol.

The guarantee is the same objective, optimizer windows and restored optimizer
state. FP32/BF16 summation and prefix VJP order can differ; bitwise-identical
parameters, gradients and RNG streams across GPU counts are not promised.

Warm-start, append-resume and unindexed legacy non-streaming migration are not
part of this policy. Resume requires an unfinished schedule and a complete
checkpoint with cleared gradients. Missing source data does not trigger replay
or a silent fresh restart. Nonphysical schedule/anchor/model mismatches remain
errors; opting in does not bypass their compatibility checks.

## 显存与主存

### 长上下文和激活卸载

完整训练 YAML、CLI 和恢复操作见[训练接口使用指南](README.md#配置预检和首次训练)。

The supported entry remains `.venvs/rl/bin/dycon lora-train --training-config YAML`.
This implements single-GPU synthetic engineering validation, with complete causal
attention, all-linear LoRA coverage and update-boundary recovery. Accelerate/DDP is
deferred; FSDP, ZeRO, context parallel and higher-order gradients are unsupported.

The additive `training.memory` settings default to the original dense path:

| Field | Default | Contract |
|---|---|---|
| `loss_chunk_tokens` | `0` | Positive enables frozen-head CE before vocabulary projection. Only shifted supervised positions are projected; blocks are recomputed in backward. |
| `mlp_chunk_tokens` | `0` | Positive enables token-position MLP recomputation, preserving every PEFT Parameter object and export path. Dropout must remain zero. |
| `exact_length` | `false` | Trim a single right-padded sample to its true length. Left padding, holes, supervised padding and multiple samples are rejected. |
| `attention_policy` | `auto` | `flash_only` restricts full attention to Flash SDPA in both forward and backward. A dense four-dimensional mask or unavailable kernel fails. |
| `activation_offload` | `none` | `checkpoint_boundary` moves only rank-three checkpoint input activations through pinned CPU memory. Weights stay on GPU. |
| `host_budget_gib` | `64.0` | Per-rank logical live activation bytes only; excludes native rounding, idle cache, RSS and other pinned users. Optional legacy preflight checks all local ranks. Actual pinned allocation admission requires opt-in `host_offload`. |
| `profile_first_update` | `false` | Collect actual full-attention operators, CUDA kernel names, masks and observed transfer/compute overlap. The first update includes profiler overhead. |

`training.checkpoint_policy` selects `layer` (production default), `selective_mlp`
(only the chunked MLP recomputes), or `layer_group` (registered Qwen3.5 text backend
with non-reentrant groups). `training.checkpoint_group_layers` defaults to four.
The group backend retains the original model tree, position/mask helpers and PEFT
paths; it supports text only and rejects KV caches. It is not an attention patch.
Any Transformers/PEFT upgrade, compilation or parameter sharding needs new tests.

#### Configuration examples

The following is a fragment of the standalone training YAML; retain all required
run/model/lora/runtime/training/validation fields shown in [训练配置和预检](README.md#配置预检和首次训练).
It illustrates CE/MLP chunking with one right-padded sample per microbatch:

```yaml
training:
  checkpoint_policy: layer
  checkpoint_group_layers: 4
  memory:
    loss_chunk_tokens: 512
    mlp_chunk_tokens: 512
    exact_length: true
    attention_policy: flash_only
    activation_offload: none
    host_budget_gib: 64.0
    profile_first_update: false
  supervision:
    mode: uniform
    max_targets_per_sample: 32
```

`loss_chunk_tokens` bounds vocabulary projection positions per block; sparse
supervision determines which positions reach those blocks. `mlp_chunk_tokens`
chunks every input position inside MLPs and does not sparsify attention. Their
block sizes are independent of the linear-attention reference kernel's internal
64-token chunks. Sparse selection never shortens valid backbone history.

| Desired policy | YAML change | Required combination |
|---|---|---|
| Ordinary layer checkpointing | `checkpoint_policy: layer` | Default; may combine with CE/MLP chunks |
| MLP-only recomputation | `checkpoint_policy: selective_mlp` | `mlp_chunk_tokens > 0`; attention graphs remain resident |
| Group consecutive layers | `checkpoint_policy: layer_group` | Positive `checkpoint_group_layers`; text only, no KV cache |
| Offload checkpoint boundaries | `memory.activation_offload: checkpoint_boundary` | `layer` or `layer_group`, explicit host budget |

Throughput and memory presets default to dry-run:

```bash
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-long-throughput.yaml
.venvs/runner/bin/dycon lora-train \
  --training-config RL/experiments/lora-long-memory.yaml
```

Copy a preset to a task YAML and edit that file to choose lengths/policies. Execute
an actual smoke using `.venvs/rl/bin/dycon` only after setting a new attempt ID and
`run.mode: smoke`. There are no CLI memory-policy overrides. Changing memory or
checkpoint policy invalidates exact resume; use an explicit adapter warm-start for
a new strategy. `exact_length` only trims right padding, never valid input after
the last selected target. CPU offload moves boundary activations, not model weights.

#### Numerical policy and capacity

The supported fixture configuration limit is 262144, with no claim of that capacity.
Lengths above 2048 require chunked CE, exact-length microbatch one and Flash full
attention. Linear attention is separately identified: the pinned environment uses
Transformers' reference PyTorch gated-delta and causal-convolution paths. That
fallback is explicitly recorded, and is not described as optimized linear attention.

Real 9B numerical auditing found shape-dependent bf16 GEMM differences with default
CUDA reduction settings. The training service explicitly selects cuBLASLt and
disables bf16 reduced-precision reduction **and split-K**. With these settings,
dense and chunked 2K loss agree and all adapter gradient differences satisfy the
predeclared `atol=2e-5, rtol=2e-3` check (observed maximum about `7.3e-7`). Baseline
performance comparisons must use the same settings. This policy, implementation
hashes, dependencies and memory settings are part of the resume contract. Old
checkpoints with a different policy require explicit adapter warm-start.

`dry-run` validates and prints the resolved strategy, host availability/budgets,
temporary logits size and dense-mask fallback estimate without importing Torch or
claiming that the sequence fits. It does not silently change `run.mode`. All actual
training uses a fresh attempt ID, including after any failed/OOM attempt.

Each update records projected tokens/CE blocks, MLP forward/backward blocks, actual
sequence lengths, category gradient coverage, GPU allocated/reserved peaks, host
RSS/high-water/locked observations and synchronized forward/backward wall times.
With offload it also records D2H/H2D bytes, CUDA-event copy times, effective bandwidth,
live/pinned activation peaks and host copy wall times. The profiler audit separately
reports measured copy/compute overlap; asynchronous APIs do not establish overlap.
`VmLck` may be zero for CUDA-pinned allocations; use the explicit pinned accounting.
Profiler event timings include intervening current-stream work and first-use costs.

Layer checkpointing executes MLP chunks twice forward (original and layer recompute)
and once backward. This is explicit in the counters and is a measured tradeoff.
Selective MLP checkpointing avoids the extra forward but retains attention graphs;
its much larger measured memory footprint rules it out as a long-context default.
CPU offload stays opt-in when ordinary layer checkpointing already meets the target.

Run the small-model checks without GPUs:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venvs/rl/bin/python -m pytest -q \
  tests/test_lora_memory.py tests/test_lora_config.py tests/test_lora_training.py \
  tests/test_lora_checkpoint.py tests/test_lora_entrypoint.py tests/test_lora_supervision.py
```

Run real GPU suites in separate processes so the direct numerical audit releases
its CUDA allocator before the CLI's physical-GPU idle check:

```bash
DYCON_LORA_LONG_GPU_TEST=1 .venvs/rl/bin/python -m pytest -q -s tests/test_lora_bf16_gpu.py
DYCON_LORA_LONG_GPU_TEST=1 .venvs/rl/bin/python -m pytest -q -s tests/test_lora_long_context_gpu.py
```

These tests call the formal CLI in fresh child processes, freeze YAML and tolerances,
verify receipts, test checkpoint resume and independently reload exported adapters.
The capacity ladder stops before moving higher if measured reserved-memory margin
is below 15 GiB. 64K/128K exploration is opt-in via `DYCON_LORA_EXTENDED_CAPACITY=1`;
262144 is not part of automatic acceptance. See the validation report for measured
lengths, memory and throughput. Synthetic results do not measure business quality.

#### Adaptive activation offload

The opt-in `host_offload` pool and `memory_pressure` policy are documented in
[`主存预算和压力策略`](#主存预算固定页分配器和压力策略). They add actual process-wide
pinned allocation admission, bounded idle-cache reuse, host/cgroup reserve checks,
and per-tensor GPU pressure selection. Existing `none`, `checkpoint_boundary`, and
length-based `auto` contracts retain their meanings.

`training.memory.activation_offload: auto` enables boundary offload only when a
window reaches the configured prompt or sequence threshold. `oom_retry` permits
one same-window retry after a local CUDA allocation failure. The policy never
moves model weights or changes attention semantics; host budget exhaustion is a
failure rather than a silent fallback.

#### OOM recovery configuration

Implementation and CPU regressions pass; the v11 fresh configuration validates
through the formal CLI. GPU acceptance is deferred until GPU0/1 are free, keeping
existing workloads. Actual allocator effects, pinned transfers, memory savings,
the original step-64 workload and natural CUDA OOM recovery remain unverified.
See [OOM validation and evidence](#显存与主存).

The production training interface remains `dycon lora-train --training-config YAML`.
The following fields are strictly validated and included in resolved configuration:

```yaml
runtime:
  cuda_allocator:
    expandable_segments: true
training:
  memory:
    activation_offload: auto  # none / checkpoint_boundary / auto
    host_budget_gib: 64.0     # per rank
    auto_offload:
      prompt_tokens_threshold: 49152
      sequence_tokens_threshold: 49152
      device_headroom_gib: 8.0
    oom_retry:
      enabled: true
      max_retries: 1
    memory_telemetry:
      enabled: true
      granularity: phase     # phase or step
```

The two thresholds are conservative candidates, not validated capacity limits.
Headroom is recorded for analysis; no capacity formula or free-memory-only predictor
is claimed. `auto` uses the current yielded window and ORs local risk across Gloo
ranks before any forward. An OOM retry enables boundary offload globally for that
attempt only. The next step reevaluates its initial policy.

The native CUDA allocator receives merged `PYTORCH_ALLOC_CONF` settings before
CUDA initialization in both the parent and spawned workers. The legacy alias is
merged with conflict detection. Other settings, including bracketed size classes,
are retained. `expandable_segments` with a non-native backend is rejected. Logs
separate requested options, actual backend, and support that still needs device
validation. PyTorch allocator warnings remain in the training stderr log.

Retry is available only for offline GRPO and layer/layer-group checkpointing;
shared-prefix execution continues to require the pinned layer schedule. The
shared-prefix `segment` path explicitly receives the same `BoundaryOffload`
instance as the ordinary decoder path. Local custom-autograd recomputation keeps
its existing `local_saved_tensors` isolation. Host budget exhaustion is FATAL.
Offload moves checkpoint input activations; prefix state, weights, optimizer state,
and attention semantics are unchanged. Offload cannot eliminate every temporary
projection allocation.

### 主存预算、固定页分配器和压力策略

`training.memory.host_offload` is opt-in, with process-wide pinned admission.
The validated runtime is exactly Torch `2.13.0+cu130`. It keeps native pinned
caching/rounding for blocks up to 1 MiB. Every project activation block is page
aligned and at least 1 MiB + 4 KiB, so it bypasses native caching when released.
Small activation tensors incur this minimum charge, included before allocation.
The cutoff is a per-block limit, NOT a total native-cache limit; it also changes
caching of other large pinned consumers. Model loading and other consumers are
included in `host_allocated_bytes`. No native extension or private clear API is used.

`reclaim_policy: after_use` is the default: graph leases must be released and every
D2H/H2D event complete before a block is freed. Finalizers only enqueue reclamation.
Allocation and existing branch/group control points poll without synchronizing;
the safe window boundary synchronizes when blocks or native deferred frees remain,
then drains native events through a cacheable 4 KiB tick. Events created by native
free() itself are also completed before final draining. Only a pool with neither
blocks nor deferred frees skips device synchronization. Native scalar/staging caches survive these calls.
`bounded_reuse` explicitly retains at most `cache_limit_gib` total idle project
bytes across sizes; host pressure overrides retention. `after_use` uses an effective
idle limit of zero even if an older config specifies a nonzero cache limit.

The pool refuses allocation before the real charge plus drain scratch and current
process pinned allocation exceed `allocator_budget_gib`. Other concurrent native
users are checked but not intercepted: this is not an all-thread allocator hard
cap. Host/cgroup availability and all local ranks are checked cooperatively, with
an absolute/fractional reserve even if optional preflight is false. Missing APIs,
unsupported runtimes, CUDA graph capture, failed cleanup or budget exhaustion fail
explicitly. No global host cache clear or unbounded CPU fallback is used.

`activation_offload: memory_pressure` is a separate opt-in policy. At each checkpoint
boundary it compares driver-free GPU bytes against candidate tensor bytes plus the
configured compute and H2D retry headroom. It does not credit `reserved - allocated`,
predict tensor release, or use sequence length. Ranks may choose different boundaries;
the existing Gloo control envelope, gradient collective, commit, and retry protocol
remain global. A retry increases headroom by a finite increment and is reset at the
next window. The policy identity is `scoped_host_reclaim_v2` and is included in the
memory contract, resolved configuration, events, and metrics.

Per-rank events include CPU RSS/HWM, PSS diagnostic samples, host active/allocated and
peak values, native cache estimate, pool reserved/idle/peak values, cache reclamation,
GPU allocated/reserved/free, logical live/peak, D2H/H2D bytes, selected/retained
boundary counts and reasons, cgroup capability, and the host-global available-memory
sample. Host-global values are never summed across ranks. Existing `none`,
`checkpoint_boundary`, and length-based `auto` behavior remains compatible.

Metrics distinguish leased, event-pending and idle project bytes. Reclaim events
report dropped pool bytes and conservative net native allocation decrease; the
latter can include concurrent-user noise and first scratch-cache allocation.

The CPU acceptance suite uses a deterministic allocator adapter and Gloo ranks; it
does not claim real pinned capacity. GPU evidence uses fresh processes, short/long/
short windows, real asynchronous copies, a controlled pressure input, and a tiny
two-rank NCCL data collective. The formal 9B CLI acceptance is opt-in and must use a
new attempt and an explicitly generated native-token engineering fixture; historical
production workloads are not modified or interrupted.

#### Configuration and restore

Merge this section into an existing validated LoRA training YAML; the model,
dataset, algorithm and optimizer sections retain their existing values:

```yaml
training:
  memory:
    activation_offload: memory_pressure
    host_budget_gib: 2.0  # Existing logical live-byte limit, still enforced.
    host_offload:
      enabled: true
      allocator_budget_gib: 2.0  # Process-wide pinned allocation, per rank.
      reclaim_policy: after_use # Or explicitly bounded_reuse.
      cache_limit_gib: 0.0       # Project idle cache only; native small cache survives.
      reserve_gib: 8.0
      reserve_fraction: 0.10  # Reserve is max(absolute, fraction of host RAM).
    memory_pressure:
      device_headroom_gib: 8.0
      retry_headroom_increment_gib: 4.0
      max_retries: 2          # Allowed range: 0..3; initial try is additional.
    oom_retry:
      enabled: true
      max_retries: 1         # Legacy policy setting; new policy uses field above.
    memory_telemetry:
      enabled: true
      granularity: phase
```

These are configuration defaults and an example, not a certified capacity for a
9B long-context workload. Admission accounts for every local rank even with
`preflight.check_resources: false`. Native allocator options (1 MiB cutoff/rounding, no background event thread,
cudaHostAlloc backend) are merged before
CUDA initialization; conflicts and unsupported Torch versions fail explicitly.
The process budget includes unrelated pinned allocations and one 4 KiB event-drain
scratch allocation. Ordinary Python RAM, model weights and checkpoints also require
headroom; the pinned budget does not cap process RSS or reserve the whole machine.

For bounded legacy `auto`, keep its thresholds and enable only `host_offload`.
For the historical allocator behavior, leave `host_offload.enabled: false` and use
an existing policy. `host_budget_gib` continues to limit logical live activations;
it has not become an RSS or native-cache limit. For full-policy FSDP2, see the
[managed activation settings](#全参-fsdp2-的受控激活卸载).

Retry feedback is local to one optimizer window and resets before the next;
there is no hidden cross-update pressure history to restore. Opt-in settings and
the policy version enter the existing exact-resume contract. Changing the policy,
budget, cache or retry settings requires contract validation and cannot silently
resume an incompatible historical checkpoint. Use an explicitly selected warm
start/new attempt if changing the contract, with its normal optimizer/RNG reset
semantics. Rollback means restoring the prior source/configuration and starting
an admissible attempt; never edit frozen configuration, receipts or checkpoints.


#### Migration from v1

This replaces global `pinned_max_cached_size_mb:0` and zero rounding. Both allocator
environment aliases must agree with the new options; old zero-cache overrides are
rejected before CUDA initialization. `scoped_host_reclaim_v2`, backend identity and
cutoff enter the memory contract. V1 opt-in checkpoints cannot silently exact-resume
as V2; use the existing explicit migration/warm-start rules or retain the old source.
Legacy disabled-host configurations keep the prior default memory contract.

Acceptance includes retained-graph repeated unpack, cross-stream backward, 20
short/long/short rounds, actual partial GPU copies, strict two-rank retry, formal
9B A/B/C/D/M and exact pause/resume. Functional and performance verdicts are separate;
the test fails if policy overhead exceeds 10% or incremental host governance
monitoring exceeds 3%. Historical long sequences/four-card capacity are separate.

### 全参 FSDP2 的受控激活卸载

#### Managed activation offload

Full training reuses `BoundaryOffload` and the scoped `ManagedHostPool`; FSDP2
sharding, optimizer, anchor and checkpoint interfaces are unchanged. Hooks wrap
only decoder non-reentrant checkpoint input saves. Parameters, all-gather buffers,
current-layer workspaces and output-head checkpoint inputs are not offloaded.
Every completed branch drains released leases; necessary small native pinned
cache remains available. Long layers can reclaim completed leases at allocations.

```yaml
training:
  warmup_updates: 500 # first update starts at 1/500 of learning_rate; shared by LoRA/full
  memory:
    activation_offload: memory_pressure # checkpoint_boundary / auto / none also supported
    host_budget_gib: 8.0                # logical live activations per rank
    host_offload:
      enabled: true
      reclaim_policy: after_use
      allocator_budget_gib: 8.0         # actual pinned allocation per process
      cache_limit_gib: 0.0
      reserve_gib: 8.0
      reserve_fraction: 0.10
    memory_pressure:
      device_headroom_gib: 8.0
    oom_retry: {enabled: false}
```

Auto retains globally coordinated length thresholds; memory_pressure makes local
per-tensor driver-free decisions. Actual allocator capacity and shared-host reserve
are checked before allocation; this is cooperative admission, not an OS reservation.
Budgets must cover all concurrent ranks and jobs. Allocation refusal is terminal;
FSDP2 collective-bearing backward is never retried locally.

Lengths above 2048 require managed host mode, BF16, exact-length batches, chunked
output loss and Flash-only attention. Admission does not prove capacity; measure
each backend and length. Reference and FLA both use the same boundary hook.
The schedule contract freezes the selected memory policy and allocator version;
exact resume must retain them. Historical no-offload contracts remain unchanged.
Per-rank update metrics expose transfers, pool state, native host allocation and
selection decisions; `lora-memory` reads the existing memory event log.

### 全参 Adam moment 动态驻留

`training.memory.adam_moment_offload` is opt-in, initially integrated with
full-language FSDP2, independent of reference/FLA GDN and the 8x16 schedule.
LoRA explicitly rejects enabling this option. Disabled configurations retain
Torch AdamW and their old schedule identity.

```yaml
training:
  memory:
    activation_offload: memory_pressure
    host_offload:
      enabled: true
      reclaim_policy: after_use
      allocator_budget_gib: 24.0
      cache_limit_gib: 0.0
      reserve_gib: 16.0
      reserve_fraction: 0.08
    adam_moment_offload:
      mode: memory_pressure
      backend: gpu_streamed_fp32_v1
      host_state_budget_gib: 16.0
      node_state_budget_gib: 64.0
      tile_pair_mib: 64
      gpu_scratch_budget_mib: 256
      pinned_staging_budget_mib: 512
      low_watermark_gib: 12.0
      high_watermark_gib: 20.0
      restore_stable_windows: 3
      min_residency_windows: 3
      max_restore_gib_per_window: 2.0
```

Launch through `.venvs/rl/bin/dycon rl-train --training-config YAML` as usual.
The existing dry-run reports budgets and capabilities. Budgets are upper limits,
not a preallocation request. Per-rank moments can span GPU and CPU. Parameters
and gradients remain FP32 GPU shards; compute precision and GDN are unchanged.

Each m/v tile independently owns its storage. Pressure checks run before the
first window (including lazy state and gradient costs), before every branch,
and before Adam. CUDA cache is trimmed only at a pressure boundary to measure
driver free space. A conservative linear workspace estimate plus reserve is
raised using observed branch peaks; it cannot guarantee arbitrary sequence
lengths or predict every native workspace. Allocator cached bytes are recorded
but are not credited as guaranteed contiguous capacity. There is no CUDA OOM
retry. Distributed phase failures abort the attempt before cursor/scheduler
commit; partial parameter updates are terminal and cannot be retried in memory.

CPU moments are durable state, distinct from reusable activation caches. A
single pinned slot and GPU scratch stream their FP32 values through Torch
2.13.0's non-fused/non-foreach functional AdamW. The functional import is
version-checked; this backend does not implement another Adam formula. Every
tile uses the same parameter step, committed once per successful parameter.
None gradients do not advance a parameter. Unsupported optimizer modes are
rejected. Parameter updates use `no_grad` local shard views and preserve FSDP
ownership/version tracking; no `.data` mutation is used.

With S GiB of CPU-authoritative moments, every update transfers at least S GiB
H2D plus S GiB D2H, separately from initial eviction/final restoration. Ordinary
CPU backing uses independent anonymous mappings (8192 process limit), not tens
of GiB of locked pages. Every transfer records and waits for its stream event
before source release or slot reuse. The first backend is deliberately serial.

After stable short windows, only tiles that satisfy minimum residency and leave
the next window's workspace, gradient and high watermark can return to GPU.
Completed ownership transfer releases CPU mappings. Escaping reader views keep
the mapping and its budget charge alive until the last reader leaves. Required
CPU moments may persist across updates; deleting them would corrupt Adam.

Pinned staging uses the existing activation pool and process allocator budget;
`pinned_staging_budget_mib` is a sublimit, not another independent allocation
allowance. Necessary native pinned caches remain enabled. Large managed blocks
use the existing event-safe, size-separated reclamation policy. Node admission
in `state/rl-host-admission/` checks PID/starttime owners, rank and node moment
limits, MemAvailable and cgroup ancestor limits under a short lock. Only pending
reservations plus candidate increments are subtracted from availability; charged
memory is already represented there. Both pageable moments and new pinned
requests join this ledger. Existing older workers and external processes do not
participate: this is cooperative admission, not an OS hard reservation.

`ranks[].adam_moment_offload` reports authoritative CPU/GPU state, mapped and
pending storage, eviction/restore versus per-step traffic, peak/scratch bytes,
forecast decisions and cumulative phase timings. CUDA event times and host wall
times are separate. `adam_enqueue` is CPU launch time; `adam_gpu` is measured
kernel time; transfer wait includes queueing while transfer_gpu measures copy
events. These overlapping/cumulative phases must not be summed as an update's
wall duration. Full update wall duration remains `compute_seconds`.

Production full checkpoints serialize current authoritative tiles directly,
and restore incrementally without assembling complete GPU moments. Standard
optimizer state_dict/load_state_dict remain available for small interoperability
fixtures; they materialize CPU state and should not be used for large export.
The checkpoint contract records backend/layout/arithmetic/policy identity, not
transient residency. Same-backend resume is checked with full-state tiny-model
tests. Tolerance equivalence is the public promise; test-level bitwise matches
on one backend/hardware do not imply universal bitwise portability.

A standard-AdamW checkpoint cannot silently enter the new backend. Explicit
full/FSDP2 resume may add this hash-bound start control:

```yaml
run:
  start:
    mode: resume
    checkpoint: /absolute/verified/full/checkpoint
    optimizer_backend_migration:
      source_schedule_sha256: <hash_json of source state.json contract.schedule>
      target_schedule_sha256: <hash_json of schedule_contract(target Config)>
      reason: Explicit FP32 Adam residency migration
```

Only Adam residency strategy may differ in this transition. Data, world size,
LR/warmup, clipping, activation policy and all mathematical settings must match.
The source artifact is immutable; the new checkpoint records migration lineage.
Other backend transitions are rejected.

For bounded capacity/performance tests without large checkpoint export, set
`validation.skip_checkpoint_export: true` and an explicit `stop_after_update`
below the configured epoch update count. Full GRPO windows still commit in
memory, metrics/progress and an attempt receipt are written, but no model or
optimizer checkpoint is written. Such attempts explicitly report
`validation_only: true`, `resumable: false`, `final_checkpoint: null`; they cannot
serve as resume sources. The default remains normal checkpoint saving.

For expensive length ladders, `validation.capacity_microsteps: 1` additionally
runs only that many canonical microsteps of the first window, with complete
native branches/masks and fully initialized Adam state. It requires streamed
Adam and no-export validation. It does **not** call optimizer.step, advance the
scheduler/cursor, or claim a complete GRPO update. `capacity.json` records actual
branch lengths, full-state validation and memory; the attempt marks
`capacity_only: true`, and no update metric is emitted. Use separate complete
native 8x16 windows for update acceptance. These execution/output controls do
not change the mathematical schedule identity.

Tests: `tests/test_optimizer_offload.py`; opt-in GPU tests use
`DYCON_ADAM_OFFLOAD_GPU=1 DYCON_ADAM_GPUS=4,5`, and 8x16 tests use
`DYCON_ADAM_EXECUTION_GPU=1 DYCON_ADAM_EXECUTION_GPUS=4,5`. Select idle devices.
Real 9B length and throughput claims require separate complete native-branch
evidence; the tiny fixtures establish correctness, not 80K capacity.

## 验证范围

8×16 和保留窗口的跨拓扑恢复已有 CPU/Gloo 验证。真实 Qwen/NCCL 跨拓扑更新、长前缀容量、吞吐和训练质量需要独立验证。Dry-run 的 metadata 验证与实际 tensor、RNG 和更新验证是不同阶段。

历史全参 BF16 FLA 运行验证覆盖固定拓扑下的 fresh/resume、导出与 probe。以相同初始权重和完整 native parent 进行的 reference/FLA 全梯度测量观察到数值差异，比较接口输出误差指标供使用方判定。

主存卸载和 OOM 验收应分别核对 native pinned admission、实际迁移、显存峰值及是否只提交一次完整更新。Full FSDP2 的 collective backward 禁止本地重试。LoRA 的受控窗口重试必须保持分支、anchor、游标和 optimizer 状态合同。
