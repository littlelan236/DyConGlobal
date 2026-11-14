# 训练接口使用指南

流式 GRPO 现在支持单卡和原 world size 的完整 checkpoint resume，继续使用本页的正式 CLI。启动方式和父 contract 共同选择 streaming/prepared 路径；保留父配置的完整 max_updates，通过 stop_after_update 暂停。配置、旧 checkpoint 只读兼容、外部 native/probability 依赖及迁移限制见 [流式恢复](STREAMING_RESUME.md)。runner dry-run 会明确标注尚待实际 tensor 验证。

本指南说明 `RL/` 中单卡 Qwen3.5-9B LoRA causal CE 与离线 GRPO 接口。命令从项目根目录
`/home/wangtianyi/DyConGlobal` 执行。正式训练使用 `.venvs/rl`；仅做 dry-run、
查看帮助或核验产物可使用不依赖 Torch 的 `.venvs/runner`。

| 需要完成的工作 | 使用接口 |
|---|---|
| 校验配置、模型内容身份、输出路径 | `dycon lora-train`，YAML 设置 `run.mode: dry-run` |
| synthetic CE smoke、新训练、恢复或 adapter 接续训练 | 同一命令，YAML 设置 `run.mode: smoke` |
| 在 Python 中执行同样的产物管理流程 | `RL.service.run(root_config, training_config)` |
| 为已有模型传入自定义 batch / loss | `RL.training.train(...)`，由调用方管理加载与产物 |
| 只解析监督位置或调用分块 CE | `prepare_batch(...)` / `LossBackend(...)` |
| GRPO 绑定外部 raw anchor、一次 epoch 和完整 checkpoint 恢复 | [GRPO.md](GRPO.md)，同一 `lora-train` 入口 |
| 查看训练进度、loss 曲线和冻结 advantage | `lora-status` / `lora-tensorboard` / `lora-advantages`，见 [可观测性](OBSERVABILITY.md) |
| 查询每 rank 显存阶段、卸载决策与 OOM 重试结果 | `lora-memory ATTEMPT_ID --storage-root ROOT`，见 [OOM 验证记录](OOM_RECOVERY_VALIDATION.md) |
| 构建离线 step groups、缓存和概率 replay | [数据集模块](dataset/README.md)，独立于 CE smoke |
| Training progress, loss curves and branch advantages | [RL observability guide](OBSERVABILITY.md) |

CLI 默认读取 synthetic fixture；`offline_grpo_k3_v1` 配置则绑定一个 sealed
`rl-dataset`，通过同一 `lora-train` 入口执行离线 GRPO。设置显式的
`algorithm.name`, `dataset_id` 和 `grpo` block；group 是调度单元，
`max_epochs` 固定为 1，checkpoint contract 会绑定 dataset receipt、reference、
advantage policy、完整 eligible group 列表、group order 和 probability mode。
训练使用指定 dataset 中全部符合条件的完整 group，不划分训练集/验证集/测试集。

离线 GRPO 的完整配置、固定历史 K3 语义、reference cache、一次 epoch 游标、
暂停恢复和验收命令见 [GRPO.md](GRPO.md)。GRPO 按 branch token mean →
group branch mean → update group mean 聚合；CE 的 token 加权累积规则保持兼容。

## 训练进度、实时 loss 与冻结 advantage 查询

这些功能通过 `.venvs/runner` 的正式 CLI 使用，不加载模型；TensorBoard 同步器和
旧 JSON 索引 worker 由 CLI 在 `.venvs/analysis` 中管理。将 `ATTEMPT_ID` 替换为训练
attempt ID，`ROOT` 替换为其实际存储根目录，例如 `/data2/wangtianyi/DyConGlobal`。

```bash
# 发现活动训练；无需记住完整 attempt ID。
.venvs/runner/bin/dycon lora-status --active
# 以 JSONL 持续查看进度；去掉 --watch 可查询一次快照。
.venvs/runner/bin/dycon lora-status ATTEMPT_ID --storage-root ROOT --watch 5 --format json
# 显式启动可视化；重放已有 metrics，并持续同步完整 optimizer update。
.venvs/runner/bin/dycon lora-tensorboard ATTEMPT_ID --storage-root ROOT --action start --port 16006
# 查询冻结训练顺序中的第 0 组，保持组内 native branch 顺序。
.venvs/runner/bin/dycon lora-advantages ATTEMPT_ID --storage-root ROOT --group-ordinal 0
```

在本地电脑另开终端保持以下转发连接，浏览器访问 `http://127.0.0.1:16006`：

```bash
ssh -N -L 16006:127.0.0.1:16006 wangtianyi@10.249.187.9
```

训练进度区分已完成 update、当前 inflight 工作和已发布 receipt 的可恢复 checkpoint。
loss 曲线按完整 optimizer update 刷新；总 loss、policy loss 与未乘 beta 的 K3 分别显示。
旧进程无需重启，缺少的阶段埋点返回 `unknown`。GRPO advantage 查询显示实际保存的冻结
值及 provenance；证据不足时返回 unavailable/null，不用动态训练信号代替。

旧格式 `frozen-groups.json` 首次查询前需要显式构建外部索引，普通查询不会隐式扫描全文件：

```bash
.venvs/runner/bin/dycon lora-advantages ATTEMPT_ID --storage-root ROOT --build-index
.venvs/runner/bin/dycon lora-advantages ATTEMPT_ID --storage-root ROOT --index-status
.venvs/runner/bin/dycon lora-tensorboard ATTEMPT_ID --storage-root ROOT --action status
.venvs/runner/bin/dycon lora-tensorboard ATTEMPT_ID --storage-root ROOT --action stop
```

停止可视化不停止训练。events、心跳和索引位于独立 runtime 目录，不写入 sealed attempt
或 checkpoint。分页、CSV、resume 继承曲线、历史时间与 token 预览边界见
[可观测性专题](OBSERVABILITY.md)，测试证据见 [验收记录](OBSERVABILITY_VALIDATION.md)。

## 1. 选择配置并先执行 dry-run

### 本次 OOM 配置与执行状态（2026-11-14）

已准备正式 fresh 配置
[`grpo-data-parallel-gpu01-pool128-kl-off-step-stream-fresh-20261113-v11-oom.yaml`](experiments/grpo-data-parallel-gpu01-pool128-kl-off-step-stream-fresh-20261113-v11-oom.yaml)。
它保留物理 GPU `[0,1]`、KL=0、原 seed/shuffle、4×32 group/branch 语义、
1694 updates、有界异步预取及 BF16 Split-K 修复；新增 allocator、自动卸载、
一次整步重试与每 rank 阶段遥测。

该 YAML 的 `run.mode: smoke` 是项目实际执行枚举，直接调用会启动正式训练。
仅校验时使用独立 attempt ID 的 `run.mode: dry-run` 副本；本次已校验副本保存在
`temp/rl_oom_recovery_20261114/v11-dry-run.yaml`。参数说明见
[长上下文配置](LONG_CONTEXT.md#oom-recovery-configuration-2026-11-14)。
49152 是候选长度阈值，8 GiB headroom 仅用于记录分析，均不构成容量保证。

按用户选择保留现有 GPU 任务，待 GPU 空闲后再进行正式训练验收。当前没有启动
v11，也没有额外 GPU smoke 或显存耗尽程序。GPU 空闲后重新核对资源，通过本页
同一 `lora-train` 入口执行正式 YAML，并检查原第 64 步顺序成功提交、实际卸载
传输和后续长数据。若训练未自然发生 CUDA OOM，真实 GPU 恢复仍应标为未实证。
测试结果、证据路径和完整验收清单见 [OOM 验证记录](OOM_RECOVERY_VALIDATION.md)。

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

## 1.1 为 GRPO 选择冻结概率来源

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
先按[概率对齐流程](dataset/PROBABILITY_ARTIFACTS.md)准备封存 ID，训练只绑定最终
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

完整合同见 [GRPO.md](GRPO.md)，源接口见
[PROBABILITY_STREAM.md](PROBABILITY_STREAM.md)，203 项 CPU/联调及真实 Qwen
同精度合成 GPU 验收见 [GRPO_BINDING_VALIDATION.md](GRPO_BINDING_VALIDATION.md)。
真实 FP16 collector/BF16 历史数据小样本验收仍需指定封存来源。
本次 schema 3 对齐的 348 项 CPU 联合回归及独立验收边界见
[PROBABILITY_ALIGNMENT_VALIDATION.md](PROBABILITY_ALIGNMENT_VALIDATION.md)。

## 2. 开始、暂停与恢复

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
可见设备数与保存时一致。完整配置和 CLI 操作见 [流式恢复](STREAMING_RESUME.md)，
已通过的真实短数据 GPU 测试与未运行的 v10 更新见
[验收记录](STREAMING_RESUME_VALIDATION.md)。

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

## 3. 选择监督位置

| 模式 | `max_targets_per_sample` | batch 中的 `loss_mask` | 行为 |
|---|---|---|---|
| `labels`，默认 | `null` | 必须不提供 | 所有有效 causal labels |
| `explicit_mask` | `null` | 每个 batch 必须提供 bool mask | 精确采用调用方选中的 target |
| `uniform` | 正整数 K | 必须不提供 | 每个样本至多 K 个确定性均匀位置 |

`explicit_mask` 与 `uniform` 的正式配置都要求 `training.memory.loss_chunk_tokens > 0`，
确保在 vocabulary projection 前选点。`uniform` 的 K=1 选择最后一个合法 target；
其他情况按整数下取整选取有序首尾及中间位置，算法详见[稀疏监督说明](SPARSE_SUPERVISION.md)。
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

## 4. Python 服务与训练核心

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

## 5. 长上下文参数、输出与检查

分块 CE、MLP、checkpoint policy 和 offload 的配置组合见
[长上下文接口](LONG_CONTEXT.md#configuration-examples)。大于 2048 的 synthetic
长度要求分块 CE、`exact_length: true` 和 `attention_policy: flash_only`。
exact-length 当前只支持 microbatch=1 的连续右 padding；它只去除 padding，不按
最后监督位置截断。schema 允许长度 8–262144，但这不代表设备容量。

输出位置为 `runtime.output_root/RL/checkpoints/ATTEMPT_ID/`：

| 文件/目录 | 用途 |
|---|---|
| `training-config.yaml`、`resolved-config.json` | 原始及解析配置 |
| `attempt.json`、`environment.json` | 状态、来源、依赖与 kernel 身份 |
| `input-identity.json`、`selection.json` | 实际输入和裁剪前监督身份 |
| `metrics.jsonl` | 每次更新的 loss、计数、耗时、显存和梯度指标 |
| `checkpoints/step-NNNNNN/` | 可精确 resume 的完整状态及 adapter |
| `adapter/` | 可用于 warm-start 的 PEFT 导出；不含完整 optimizer 状态 |
| `validation.json`、`report.md`、`receipt.json` | 验证结果、摘要与内容核验凭据 |

```bash
.venvs/runner/bin/dycon artifact show lora-my-task-001 \
  --scope lora-attempt --storage-root /data3/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon artifact verify lora-my-task-001 \
  --scope lora-attempt --storage-root /data3/wangtianyi/DyConGlobal
```

核验成功说明产物和身份一致，不说明模型质量。`consumed_tokens` 是累计实际监督
target 数；单次 `supervised_tokens` 是 CE 分母，`available_supervised_tokens` 是选点前
可用 target 数，`input_tokens` 是 backbone 非 padding 输入数。分块路径的
`projected_tokens` 等于选中数，`ce_chunks` 统计 forward 投影块，不重复计 backward
重算。没有选中 target、无效 mask、非有限 loss/gradient 等错误会拒绝更新。

稀疏接口已通过 CPU tiny Qwen3.5/PEFT 和 runner dry-run 验收；新增真实 sparse GPU
及容量测试未执行。历史 dense 长度结果见[验收报告](LONG_CONTEXT_VALIDATION.md)。
| ??????????? loss ????? branch advantage | [RL ??????](OBSERVABILITY.md) |

## 8×16 validation status (2026-11-14)

The explicit experimental [8-GPU distinct-parent protocol](DATA_PARALLEL_8X16.md) has passed CPU/Gloo regression: **426 passed, 12 skipped**, including real eight-process synchronization and tiny-Qwen prepared/streaming resume. See [validation details and pending acceptance](DATA_PARALLEL_8X16_VALIDATION.md). The skips do not count as GPU acceptance; real Qwen3.5-9B eight-GPU correctness, long-prefix capacity, throughput and quality remain pending.
