# difficultyEstimator 可配置训练使用说明

编辑独立训练配置 `difficulty/training/config.yaml`，然后在远程工程目录中执行：

```bash
cd /home/wangtianyi/DyConGlobal
source difficulty/scripts/train.sh
# 也可以指定另一份训练配置
source difficulty/scripts/train.sh /absolute/path/to/training.yaml
```

启动脚本使用 analysis 环境，调用正式入口 `dycon difficulty-train --training-config YAML`。脚本在子 shell 中运行，保留调用者的工作目录、shell 选项、环境变量和信号处理设置（trap）。训练在前台运行，按 `Ctrl+C` 可以取消。

## 数据与模型配置

训练配置采用 `schema_version: 2`，主要字段如下：

| 配置字段 | 作用 |
|---|---|
| `data.dataset_id` | 指定训练数据集 |
| `data.split.id` | 指定已有的冻结数据划分 |
| `data.split.strategy`、`seed`、`validation_fraction`、`test_fraction` | 按策略、随机种子和比例生成划分；不能与已有划分 ID 同时配置 |
| `data.targets` | 指定到提交时的剩余工作量回归目标 |
| `model.kind` | 选择 `ridge`、`ffn_mlp` 或 `ridge_plus_ffn` |
| `features.public_site` | 选择输入的大模型 hidden 特征层，保持单层输入 |
| `model.mlp.hidden_dims` | 配置 MLP 各隐藏层维度，列表长度即隐藏层数 |
| `model.mlp.activation`、`normalization`、`dropout` | 配置激活函数、输入归一化方式和 dropout 比例 |
| `training.optimizer` | 配置 MLP 优化器及其参数，支持 AdamW、Adam 和 SGD |

配置会拒绝不兼容的参数组合。Ridge 使用闭式求解，通过 `model.ridge` 配置参数，不使用迭代训练的 `training` 或 MLP 网络结构选项。

Ridge+MLP 必须在 `model.ridge_publications` 中为每个目标指定真实的已发布 Ridge manifest。训练期间 Ridge 基础模型保持冻结，仅训练 MLP 残差部分，不重新拟合 Ridge。

## 运行模式与产物

`run.attempt_id: auto` 会为每次运行生成新的、不可覆盖的训练尝试 ID。如果手动指定 ID，其输出目录必须尚不存在，包括未完成的尝试目录。

相对配置路径以工程根目录为基准解析。`run.mode` 支持 `dry-run`、`check-data` 和 `fit`：前两者分别检查配置与数据来源、进一步检查特征数据，均不执行拟合；`fit` 执行训练或拟合，并在启用 TensorBoard 时自动启动可视化服务。

产物保存在 `difficulty/checkpoints/ATTEMPT_ID/`，其中 `ATTEMPT_ID` 是启动输出中的实际训练尝试 ID。

## 在本地 PC 访问 TensorBoard

TensorBoard 默认监听远程服务器的 `127.0.0.1:6006`。在本地 PC 的 PowerShell 中执行：

```powershell
ssh -N -L 16006:127.0.0.1:6006 wangtianyi@10.249.187.9
```

保持该 SSH 连接运行，在本地浏览器打开 `http://localhost:16006`。`16006` 是本地端口，`6006` 是远程端口；如果修改训练 YAML 中的 TensorBoard 端口，需要同步修改 SSH 命令中的远程端口。本地端口被占用时，可以改用其他本地端口，并同步修改浏览器地址。

MLP 按 epoch 记录 evaluation 模式下的训练集损失 `loss/train` 和验证集损失 `loss/val`；训练模式下的 minibatch 优化损失单独记录为 `loss/train_optimization`。Ridge 没有 epoch，其曲线横轴为候选参数组合的试验编号。

事件文件保存在本次 attempt 目录中，写入器关闭后才完成产物校验凭证（receipt）的封存。TensorBoard 的可变进程日志保存在 `state/difficulty-training/`。

## 查看和停止服务

在工程目录执行以下命令，将 `ATTEMPT_ID` 替换为实际训练尝试 ID：

```bash
.venvs/analysis/bin/dycon difficulty-tensorboard ATTEMPT_ID --action status
.venvs/analysis/bin/dycon difficulty-tensorboard ATTEMPT_ID --action stop
```

如果 `tensorboard.keep_after_training` 为 `true`，训练成功后会保留 TensorBoard，方便继续查看曲线；训练失败时会清理本次启动的服务。上述命令只操作已登记、进程启动身份与 TensorBoard 命令行均匹配的进程。

## 兼容性与使用边界

兼容适配层仍接受旧版 MLP YAML。新的可配置模型 checkpoint 使用独立的架构标识；旧版 v1 checkpoint 按原来的两个隐藏层结构加载。

本入口生成离线训练 checkpoint，不自动将模型发布为在线 predictor。

## 完整配置示例：使用工程内已有的数据集与划分

以下 dataset 和 split 已于 **2026-10-26** 在远程工程中核对存在：

- 数据集：`resolved-migration331-fraction2-20261024`，位于 `difficulty/datasets/resolved-migration331-fraction2-20261024/`，包含 106 个 task、106 条 rollout 和 6104 个样本。
- 划分：`resolved-migration331-fraction2-event70-15-15-20261024`，位于上述数据集的 `splits/` 下，训练/验证/测试样本数分别为 4272/916/916。
- **字段名是 `data.split.id`**。本文所说的 split ID 在 YAML 中不写成顶层 `split_id`；它必须属于 `data.dataset_id` 指定的数据集。

下面是一份完整的两层 MLP 配置示例。训练参数仅用于展示写法，不代表最优超参数；配置中的 `dry-run` 会检查来源而不训练，准备好后改为 `fit`。

```yaml
schema_version: 2
run:
  attempt_id: auto
  mode: dry-run
data:
  dataset_id: resolved-migration331-fraction2-20261024
  expected_receipt_sha256: 93af67c4bebf44ee3c2b5b18e02a878f5b4d825558e436a347f67b739a9b39b2
  split:
    id: resolved-migration331-fraction2-event70-15-15-20261024
  targets:
    - submission_remaining_steps_fraction
    - submission_remaining_reasoning_tokens_fraction
features:
  public_site: 3
model:
  kind: ffn_mlp
  mlp:
    hidden_dims: [256, 128]
    activation: gelu
    normalization: layernorm
    dropout: 0.1
training:
  epochs: 100
  batch_size: 256
  seed: 42
  device: cpu
  torch_threads: 4
  loss: mse
  optimizer:
    kind: adamw
    learning_rate: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
    eps: 1.0e-8
  early_stopping:
    patience: 20
    min_delta: 0.0
  scheduler:
    enabled: true
    kind: reduce_on_plateau
    factor: 0.5
    patience: 5
    threshold: 0.0001
    min_lr: 1.0e-6
tensorboard:
  enabled: true
  host: 127.0.0.1
  port: 6006
  flush_secs: 5
  reload_interval: 5
  keep_after_training: true
```

此示例中的 receipt SHA 只对应上述数据集的 `receipt.json`。更换数据集时应改成新数据集的实际 SHA，或写 `null` 表示不预先固定期望 SHA；入口仍会解析并记录实际 SHA。

## 各字段可选值、示例与约束

以下说明面向新的独立训练配置 schema 2。默认值指当前代码在**省略字段时**的值，不一定等于现有 `config.yaml` 中显式写入的值。所有层级都拒绝未知字段；YAML 中布尔值用 `true`/`false`，空值用 `null`，列表使用 YAML 列表格式。

### 运行、数据与特征

| 完整字段名 | 示例或可选值 | 默认值与约束 |
|---|---|---|
| `schema_version` | `2` | 当前仅支持 `2` |
| `run.attempt_id` | `auto`、`mlp-fraction-trial-20261026` | 默认 `auto`；显式 ID 长度 1–151，以英文字母或数字开头，其余仅允许英文字母、数字、`_`、`.`、`-`；目录必须不存在 |
| `run.mode` | `dry-run`、`check-data`、`fit` | 默认 `fit`；只有 `fit` 自动拉起已启用的 TensorBoard |
| `data.dataset_id` | `resolved-migration331-fraction2-20261024` | 必填；必须是已有训练数据集 ID，而不是目录路径；ID 字符规则同上 |
| `data.expected_receipt_sha256` | 完整示例中的 SHA 或 `null` | 默认 `null`；非空时必须是 64 位小写十六进制，且与选定数据集的 receipt 一致 |
| `data.split.id` | `resolved-migration331-fraction2-event70-15-15-20261024` | 默认 `null`；非空时读取该数据集已有划分，不能同时填写下列新建划分参数 |
| `data.split.strategy` | `event_random_v1`、`task_random_v1` | 未指定已有 ID 时默认 `event_random_v1`；前者按样本划分，允许 task 跨集合；后者按 task 分组划分 |
| `data.split.seed` | `42`、`123` | 未指定已有 ID 时默认 `42`；整数，范围 `0 <= seed < 2^32` |
| `data.split.validation_fraction` | `0.15`、`0.2` | 未指定已有 ID 时默认 `0.15`；严格大于 0、小于 1 |
| `data.split.test_fraction` | `0.15`、`0.2` | 未指定已有 ID 时默认 `0.15`；严格大于 0、小于 1，且与 validation 比例之和小于 1 |
| `data.targets` | 如 `[submission_remaining_steps_fraction]` | 必填、非空、不重复；可选目标见下一表，数据集必须提供对应标签 |
| `features.public_site` | `3` | 默认 `3`；整数 `0–31`，选择输入的单个 public hidden site，不是 MLP 的隐藏层数 |

支持的全部回归目标：

| `data.targets` 可选值 | 标签含义与变换 |
|---|---|
| `submission_remaining_steps_linear` | 到提交时的剩余 action 步数，保持原数值 |
| `submission_remaining_steps_log1p` | 对上述步数做 `log1p` 变换 |
| `submission_remaining_reasoning_tokens_linear` | 到提交时的剩余 reasoning token 数，保持原数值 |
| `submission_remaining_reasoning_tokens_log1p` | 对上述 token 数做 `log1p` 变换 |
| `submission_remaining_steps_fraction` | 按所在完整轨迹的总 action 步数归一化，使用比例标签、identity 变换 |
| `submission_remaining_reasoning_tokens_fraction` | 按所在完整轨迹的总 reasoning token 数归一化，使用比例标签、identity 变换 |

fraction 目标需要数据集中存在 fraction 标签，不能仅切换目标名就把 raw 数据集变为归一化数据集。上面的 `fraction2` 数据集可用于示例中的两个 fraction 目标。

### 模型结构与 Ridge 参数

| 完整字段名 | 示例或可选值 | 默认值与约束 |
|---|---|---|
| `model.kind` | `ridge`、`ffn_mlp`、`ridge_plus_ffn` | 默认 `ffn_mlp` |
| `model.mlp.hidden_dims` | `[256]`、`[1024, 512]`、`[512, 256, 128]` | 默认 `[1024, 1024]`；非空正整数列表；输出标量层自动添加，不写入该列表 |
| `model.mlp.activation` | `gelu`、`relu`、`silu` | 默认 `gelu` |
| `model.mlp.normalization` | `layernorm`、`none` | 默认 `layernorm`；指输入归一化，不是在每个隐藏层后追加归一化 |
| `model.mlp.dropout` | `0.0`、`0.1`、`0.2` | 默认 `0.1`；范围 `[0, 1)` |
| `model.ridge.layers` | `[3]`、`[3, 18]` | 默认 `null`，此时使用 `[features.public_site]`；显式列表非空、无重复，元素为整数 `0–31`；各层分别搜索，不拼接 |
| `model.ridge.alphas` | `[0.1, 1.0, 10.0]` | 默认 `[1.0]`；非空、无重复的有限正数列表 |
| `model.ridge.standardize` | `true` | 当前仅接受 `true`，标准化参数只用训练集拟合 |
| `model.ridge.blas_threads` | `4` | 默认 `4`；整数 `1–32` |
| `model.ridge_publications` | `{目标名: manifest路径}` | 仅 `ridge_plus_ffn` 必填；键集合必须与 `data.targets` 完全相同，每个值是非空 manifest 路径 |

三种模型的配置块不能混用：`ridge` 删除 `model.mlp`、`model.ridge_publications` 和整个 `training` 块；`ffn_mlp` 不接受 `model.ridge` 或 `model.ridge_publications`；`ridge_plus_ffn` 使用 `model.mlp` 与 `model.ridge_publications`，不接受用于重新拟合的 `model.ridge`。

### 训练、优化器、早停与调度器

以下字段适用于 `ffn_mlp` 和 `ridge_plus_ffn`。

| 完整字段名 | 示例或可选值 | 默认值与约束 |
|---|---|---|
| `training.epochs` | `100`、`1000` | 默认 `100`；正整数，表示最大 epoch 数，早停可能提前结束 |
| `training.batch_size` | `64`、`256` | 默认 `256`；正整数 |
| `training.seed` | `42` | 默认 `42`；整数 `0 <= seed < 2^32`；不同于数据划分 seed |
| `training.device` | `auto`、`cpu`、`cuda`、`cuda:0`、`cuda:1` | 默认 `auto`；自动选择可用 CUDA，否则 CPU；显式指定不可用 CUDA 时失败 |
| `training.torch_threads` | `4` | 默认 `4`；整数 `1–32`，控制 PyTorch CPU 线程数 |
| `training.loss` | `mse` | 当前仅支持 `mse` |
| `training.optimizer.kind` | `adamw`、`adam`、`sgd` | 默认 `adamw` |
| `training.optimizer.learning_rate` | `0.001`、`0.01` | 默认 `0.001`；有限正数 |
| `training.optimizer.weight_decay` | `0.0001`、`0.0` | 默认 `0.0001`；有限非负数 |
| `training.optimizer.betas` | `[0.9, 0.999]`、`null` | 默认 `null`，Adam/AdamW 实际使用 `[0.9, 0.999]`；显式值为两个 `[0,1)` 数值；SGD 不接受非空值 |
| `training.optimizer.eps` | `1.0e-8`、`null` | 默认 `null`，Adam/AdamW 实际使用 `1.0e-8`；显式值为有限正数；SGD 不接受非空值 |
| `training.optimizer.momentum` | `0.9`、`null` | 默认 `null`，SGD 实际使用 `0.0`；显式值范围 `[0,1)`；Adam/AdamW 不接受非空值 |
| `training.early_stopping.patience` | `20`、`100` | 默认 `100`；正整数，验证损失连续指定 epoch 未达到改善条件时早停 |
| `training.early_stopping.min_delta` | `0.0`、`0.0001` | 默认 `0.0`；有限非负数，早停所要求的最小损失改善量 |
| `training.scheduler.enabled` | `true`、`false` | 默认 `false` |
| `training.scheduler.kind` | `reduce_on_plateau` | 当前仅支持此值，监控 validation loss |
| `training.scheduler.factor` | `0.5` | 默认 `0.5`；严格大于 0、小于 1，降低学习率时的乘数 |
| `training.scheduler.patience` | `0`、`5` | 默认 `5`；非负整数，含义由 ReduceLROnPlateau 的 plateau 规则决定，与早停 patience 分开配置 |
| `training.scheduler.threshold` | `0.0001` | 默认 `0.0001`；有限非负数，判断改善的阈值，当前使用调度器默认的相对阈值模式 |
| `training.scheduler.min_lr` | `1.0e-6` | 默认 `1.0e-6`；有限非负数，学习率下限 |

YAML 科学计数法建议使用 `1.0e-8` 而非 `1e-8`，保证按数值而非字符串解析。

### TensorBoard

| 完整字段名 | 示例或可选值 | 默认值与约束 |
|---|---|---|
| `tensorboard.enabled` | `true`、`false` | 默认 `true`；关闭时不启动服务、不写 TensorBoard 事件，原训练记录仍保留 |
| `tensorboard.host` | `127.0.0.1`、`0.0.0.0` | 默认 `127.0.0.1`；前者通过 SSH 隧道访问，后者监听所有 IPv4 接口；能否直连取决于网络与防火墙 |
| `tensorboard.port` | `6006`、`6007` | 默认 `6006`；整数 `1024–65535`，必须未被占用 |
| `tensorboard.flush_secs` | `5` | 默认 `5`；整数 `1–60`，writer 的自动刷新间隔；当前实现还会在每个 epoch/trial 后主动 flush |
| `tensorboard.reload_interval` | `5` | 默认 `5`；整数 `1–60`，TensorBoard 重载数据的间隔，单位秒 |
| `tensorboard.keep_after_training` | `true`、`false` | 默认 `true`；成功后是否保留本次服务，失败或中断仍执行清理 |

## 常见配置切换示例

以下是**替换片段**，不是可以独立运行的完整文件。请在完整配置中替换同名块，避免重复键。

### 改为按 task 重新划分

删除原 `data.split.id`，把 `data.split` 替换为：

```yaml
data:
  split:
    strategy: task_random_v1
    seed: 123
    validation_fraction: 0.15
    test_fraction: 0.15
```

保留原 `data.dataset_id`、`data.targets` 等字段。这里训练比例为剩余的 70%，实际计数受分组和取整影响。

### 改为三层 MLP

```yaml
model:
  kind: ffn_mlp
  mlp:
    hidden_dims: [512, 256, 128]
    activation: silu
    normalization: layernorm
    dropout: 0.1
```

### 改用 Adam 或 SGD

将 `training.optimizer` 分别替换为以下一种；其他训练字段保持不变：

```yaml
training:
  optimizer:
    kind: adam
    learning_rate: 0.001
    weight_decay: 0.0
    betas: [0.9, 0.999]
    eps: 1.0e-8
```

```yaml
training:
  optimizer:
    kind: sgd
    learning_rate: 0.01
    weight_decay: 0.0001
    momentum: 0.9
```

### 改用 Ridge 网格搜索

将 `model` 替换为下方内容，并删除整个 `training` 块；沿用完整示例的数据、目标、运行与 TensorBoard 配置：

```yaml
model:
  kind: ridge
  ridge:
    layers: [3, 18]
    alphas: [0.1, 1.0, 10.0]
    standardize: true
    blas_threads: 4
```

这里每个目标分别搜索 2 个输入层 × 3 个 alpha，而不是拼接 2 层特征。

### 指定 Ridge+MLP 的基础模型

以下 manifest 于 2026-10-26 核对存在，声明的目标是 `submission_remaining_reasoning_tokens_log1p`、site 是 `3`、fit scope 是 `train`：

```yaml
data:
  targets:
    - submission_remaining_reasoning_tokens_log1p
features:
  public_site: 3
model:
  kind: ridge_plus_ffn
  mlp:
    hidden_dims: [256, 128]
    activation: gelu
    normalization: layernorm
    dropout: 0.1
  ridge_publications:
    submission_remaining_reasoning_tokens_log1p: runtime/parallel-artifacts/ridge-resolved-migration331-event70-15-15-reasoning-log1p-20261024/manifest.json
```

这是**真实路径和对应字段的示例，不保证与任意 dataset/split 组合兼容**。入口还会核验文件 SHA、输入宽度、模型 fingerprint、目标、site，以及基础 Ridge 的训练样本是否与当前 validation/test 重叠；存在文件并不意味着可跳过这些校验。增加第二个 target 时，必须同时提供对应的第二个 manifest。

例如，工程还存在 site 18 的 `runtime/parallel-artifacts/ridge-resolved-migration331-reasoning-log1p-20261024/manifest.json`；不能在 `public_site: 3` 下直接引用它。log1p 的 manifest 也不能绑定到 fraction 目标。
