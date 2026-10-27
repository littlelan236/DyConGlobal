# 剩余工作量估计与数据接口

2026-10-21：difficulty 新产物独立于来源 run，统一使用 `dycon`。一个训练集可通过
`difficulty-dataset compose NEW_ID --source-dataset A --source-dataset B` 引用多个批次，来源 receipt/hash 递归核验。
`difficulty-dataset resolved ID --source-run RUN --index-id INDEX` 冻结成功轨迹训练集。

```text
difficulty/datasets/DATASET_ID/{dataset.json,rollouts.json,samples.jsonl,source-evidence.json,summary.json,receipt.json}
difficulty/datasets/DATASET_ID/splits/SPLIT_ID/{split.json,receipt.json}
difficulty/checkpoints/ATTEMPT_ID/{attempt.json,experiment.json,models/,metrics.json,trials.jsonl,receipt.json,...}
difficulty/predictions/PREDICTION_ID/...
difficulty/evaluations/EVALUATION_ID/...
difficulty/cache/CACHE_HASH/...
```

online/offline index 和 token join 也位于 datasets/ID，但仍保留各自 schema，不能当作已有 rollouts/samples 的训练 selection。
dataset receipt 不包含后增的 splits 子目录；每次新划分/训练用独立 ID。显式 split 先由
`dycon difficulty-split DATASET_ID SPLIT_ID --strategy event_random_v1` 创建；旧调用未指定 split_id 时
通过同一 service materialize 一个确定性的 auto-ID。各模型训练尝试引用已冻结分配。

模型 NPZ 与同名 JSON 绑定模型结构、hidden model fingerprint、输入特征/site/维度、目标变换、scaler、fit scope 和 dataset/split receipt。
`dycon artifact verify ID --scope difficulty-dataset/difficulty-attempt/difficulty-prediction/difficulty-evaluation`
校验其来源与文件；split 使用 `--scope difficulty-split --dataset-id DATASET_ID`。
旧 run 产物保持原样，旧 index 使用显式 `runs/RUN/difficulty/datasets/ID` 路径；旧 checkpoint 可用原 experiment 路径发布。
`--run-id` 仍可读取冻结配置，但不决定 difficulty 输出目录。runtime 部署副本仍在 runtime/parallel-artifacts/。

MLP 复用同一 dataset/split/attempt 生命周期：

```bash
.venvs/analysis/bin/dycon difficulty-mlp ATTEMPT_ID \
  --dataset-id DATASET_ID --split-id SPLIT_ID --mode dry-run/check-data/fit
.venvs/analysis/bin/dycon artifact verify ATTEMPT_ID --scope difficulty-attempt
```

Ridge+MLP 使用同一套 attempt/receipt/epoch-history 接口：

```bash
.venvs/analysis/bin/dycon difficulty-ridge-mlp ATTEMPT_ID \
  --dataset-id DATASET_ID --split-id SPLIT_ID --mode dry-run/check-data/fit
```

`difficulty/training/config.yaml` 的 `ridge_mlp_probe.ridge_publications` 为两个 log1p 目标分别绑定已发布 Ridge manifest。入口验证 manifest、发布包文件 SHA、public site、hidden width、模型 fingerprint、target 及训练 split；Ridge 必须是已训练的 `fit_scope=train` 或 `train_validation` 数值 checkpoint，且其训练样本不得出现在当前 validation/test。Ridge 参数保持冻结，Ridge+MLP 只训练零初始化的 FFN residual，最终预测为 `Ridge(x) + residual(x)`。每个 target 的 `models/<target>/ridge-publication/` 保存发布元信息和 payload 快照，因此 attempt 可独立复核。

MLP 与 Ridge+MLP 使用配置的 public site（当前为 site 3），训练两个独立的 submission log1p 目标。`fit` 按 validation MSE 选择 checkpoint，再评估 train/validation/test；`history/<target>.jsonl` 保存逐 epoch train/eval loss、MSE/R²/MAE/RMSE 和 learning rate，`progress.json` 保存运行状态。MLP 是离线 attempt 产物，现有 predictor/evaluate schema 不自动消费它。

MLP 默认使用 `event_random_v1`：同一 task 的样本可以进入不同 split；如需 task holdout，必须显式创建并传入 `--split-id`。回归任务没有分类 accuracy，history 同时保存 train/validation/test 的 MSE、R²、MAE、RMSE；`difficulty.training.mlp_visualize.plot_attempt` 生成 loss/R² 曲线。

MLP 专用参数不放在根 `config.yaml`，统一维护于 `difficulty/training/config.yaml`。该文件包含模型、数据划分和 scheduler 配置；每个 attempt 的 `attempt.json` 保存完整训练配置及 SHA，每个 target 的 `models/<target>/config.json` 保存参数快照。scheduler 默认使用 `ReduceLROnPlateau`，监控 validation loss；每个 epoch 的 `learning_rate` 写入 history，并绘制在学习率曲线中。


同 action、不同 reasoning 的评分不稳定性评测使用 `dycon difficulty-evaluate-parallel RUN_ID EVALUATION_ID`，
实现与已有并行轨迹可视化同在 `difficulty/evaluation/` 并复用轨迹对齐。
支持平均两两差、标准差/极差、log1p 倍数分歧率和相对 Q，以及去重/原采样两种口径和 task/repo CI。
参数、分组、汇总与产物见 [使用说明](../docs/ENTRYPOINTS.md#7-ridge可视化和一致性指标)。

并行解码的在线评分可使用 `dycon difficulty-plot-parallel RUN_ID PLOT_ID --rollout-id ROLLOUT_ID`
绘制实际选中轨迹的逐步预测折线及全部候选分支散点，并导出 PNG/SVG、评分 CSV 和来源 receipt。
无需加载 checkpoint、hidden tensor 或启动 GPU；具体选择、单位、缺失评分与噪音解释见
[并行轨迹评分可视化](../docs/ENTRYPOINTS.md#7-ridge可视化和一致性指标)。

`dycon difficulty-ridge EXPERIMENT_ID` 在 analysis 环境消费冻结的 submission 选择，执行单层 Ridge × alpha 搜索、验证 R²选优及参数/可视化报告保存。支持原 task 两划分，以及 event/task 两种三划分：`event_random_v1` 允许同 task 的 step 跨分区，`task_random_v1` 将同 task 的全部轨迹和 step 放在同一分区；三划分均在 train＋validation 重拟合冠军后评估 test。这是“到该次提交”的独立标签视图，允许已提交但未成功或未评测的轨迹，原成功标签不改。完整协议见 [实验指南](../docs/ENTRYPOINTS.md#7-ridge可视化和一致性指标)。

2026-10-19 按实际代码更新。正式命令见 [任务入口表](../docs/ENTRYPOINTS.md)。已有索引、评分和 hidden 提取入口时直接复用，不另写临时脚本拼接数据或重定义标签。

## Ridge 实验与可视化

在工程根目录执行，使用不同的新实验 ID；已有目录（含未完成目录）拒绝覆盖：

```bash
.venvs/analysis/bin/dycon difficulty-ridge SPLIT_CHECK_ID --mode dry-run
.venvs/analysis/bin/dycon difficulty-ridge DATA_CHECK_ID --mode check-data
.venvs/analysis/bin/dycon difficulty-ridge EXPERIMENT_ID
```

`dry-run` 核验来源并保存划分；`check-data` 另读回全部 hidden、核验哈希/有限值并生成缓存；默认 `fit` 执行拟合、选优、指标与图表。正式配置在唯一 `config.yaml` 的 `difficulty.ridge_probe`：238 个 task、14,384 个共同可用 close 样本，190 task 训练、48 task 验证；32 个 public layers × 13 个 alpha × 2 个目标，共 832 个候选。来源清单旧 `split=train` 由本次 `split.json` 覆盖。标准化和 Ridge 参数只在训练集拟合，验证集分别为两个目标选择最佳 layer/alpha，最高验证 R²是模型选择分数。

画图库包名为 **matplotlib**，代码使用 `import matplotlib.pyplot as plt`，没有单独的 `plt` 包。runner 和 analysis 环境已安装 `matplotlib 3.11.2`；runner 依赖固定在 `locks/runner.in`、`locks/runner.txt`，analysis 的依赖声明包含 matplotlib。完整实验推荐 analysis，runner 也可执行不依赖 torch 的 Ridge/可视化测试。SSH 无显示器时使用 `Agg` 后端保存 PNG，无需桌面窗口。

```bash
.venvs/runner/bin/python -m pytest -q tests/test_ridge_probe.py
.venvs/analysis/bin/python -m pytest -q tests/test_ridge_probe.py tests/test_difficulty_estimators.py
```

fit 产物在 `difficulty/checkpoints/EXPERIMENT_ID/`：

| 文件 | 内容 |
|---|---|
| `attempt.json`、兼容 `experiment.json` | 实际配置、模型输入契约、dataset/split receipt；split.json 在 datasets/DATASET_ID/splits/SPLIT_ID/ |
| `trials.jsonl`、`models/` | 所有组合的训练/验证指标及可无 pickle 加载的 NPZ 参数 |
| `selected.json`、`validation-predictions.jsonl` | 两个目标的最优组合、验证逐样本预测 |
| `metrics.json`、`report.md` | 验证指标、训练均值基线、repo/source/task 分组结果 |
| `plots/*-grid.png` | 每目标一张 layer × alpha 验证 R²热图 |
| `plots/*-validation.png` | 每目标一张验证真值与预测散点图 |
| `plots/*-trajectories.png` | 每目标一张短/中/长验证轨迹比较图 |
| `receipt.json` | 完成模式与产物 SHA；必须区分数据检查与实际 fit |

每个目标输出 3 张图，均采用该目标的拟合空间；原单位指标单独保存在 metrics 中。`difficulty.ridge_probe.targets` 支持以下显式目标，默认仍为 tokens（log1p）和 steps（linear）：

| 目标 ID | 标签与变换 |
|---|---|
| `submission_remaining_reasoning_tokens_log1p` | `log1p(remaining_reasoning_tokens_to_submission)`，默认 |
| `submission_remaining_steps_linear` | `remaining_steps_to_submission`，默认 |
| `submission_remaining_steps_log1p` | `log1p(remaining_steps_to_submission)` |
| `submission_remaining_reasoning_tokens_linear` | `remaining_reasoning_tokens_to_submission` |

通过根配置或允许的 `--set difficulty.ridge_probe...` 指定 targets 与 alpha 网格，再运行 `dycon difficulty-ridge ATTEMPT_ID --dataset-id DATASET_ID --split-id SPLIT_ID --mode dry-run/fit`。log1p 预测以 `expm1` 还原，linear 保持原值；不截断负预测。不同拟合空间的 R² 不直接用于判断变换优劣，应比较相同单位的 `test_raw_units`。本目标扩展用于 Ridge 训练/报告，不扩展现有 predictor 发布及在线目标映射。

此前 `ridge-grid-real-debug-20261019` 是 2 层 × 3 alpha × 2 目标的 12 候选调试，只证明入口与报告流程，不是 832 候选正式搜索。独立新任务测试不属于当前选择验证集。

## 正式数据与评分命令

```bash
.venvs/analysis/bin/dycon dataset-index DATASET_ID CAPTURE_ID INDEX_ID
.venvs/analysis/bin/dycon token-join DATASET_ID CAPTURE_ID JOIN_ID
.venvs/analysis/bin/dycon hidden-token-join-audit JOIN_ID
.venvs/analysis/bin/dycon online-index RUN_ID ONLINE_INDEX_ID
.venvs/analysis/bin/dycon difficulty-evaluate INDEX_ID PREDICTIONS.json SCORE_ID
```

| 来源 | 产物 | consumer 边界 |
|---|---|---|
| `dataset-index` | schema 1 index.json；label_shard 和 hidden_path | 训练 loader、difficulty scorer、action-index |
| `token-join` | schema 2 rows.jsonl + receipt.json | exact token-hidden 审计；没有上述消费者所需的 index.json |
| `online-index` | schema 3 index.json；labels、task partition、NAS shard locator | 训练 loader、action-index；当前 difficulty scorer 不支持 |

索引冻结构建时的标签。在线索引若在评测前建立，resolved 未确定的行保持 null；评测后用新 index ID 重建，旧索引不会自动更新。

## 模型与目标

`difficulty.models.estimators.build_estimator(kind, **kwargs)` 提供：

- `ridge`：可选标准化、含不惩罚截距项的闭式 Ridge。
- `ffn_mlp`：LayerNorm → 两个 Linear/GELU/Dropout block → scalar；AdamW/MSE。
- `ridge_plus_ffn`：Ridge 加 FFN 残差；末层零初始化，初始组合等于 Ridge。

hidden `[samples, selected_layers, hidden_size]` 展平为 `[samples, selected_layers * hidden_size]`，不硬编码层数/宽度。schema 3 按数字 site 顺序组装特征。

| 目标 | 变换 |
|---|---|
| `remaining_reasoning_tokens_log1p` | `log1p(remaining_reasoning_tokens)` |
| `remaining_steps_log1p` | `log1p(remaining_steps)` |
| `remaining_steps_linear` | 原始 remaining steps |

`transform_target`/`inverse_target` 提供变换与逆变换。`train_estimators` 返回模型对象、transformed-space MAE/RMSE/R²、逆变换后的 validation predictions/targets；不自动输出 scorer 所需的 sample_id/target 预测清单或保存 checkpoint。预测格式见 [INPUTS.md](../docs/ENTRYPOINTS.md#4-显式输入格式)。

## Python API 和划分限制

`difficulty.training.train.load_training_data(cfg, index_id, target, split)` 排除缺失标签，返回 `(x, y_transformed, sample_ids)`。`train_estimators(cfg, index_id, target, kinds=..., **kwargs)` 读取 train/val、拟合并返回结果；kwargs 以 estimator 名为键，各参数见构造函数。

- schema 3 按 task partition 选择 train/val；未 resolved 或 censored 的标签在索引阶段置 null。2026-10-19 观察的 task 全为 train，不能将无 val 的现状称作完成验证。
- schema 1 按排序 rollout ID 约取 80%/20%；同 rollout 不跨组，但同 task 多条 rollout 可能跨组。该分支仅区分 train/val/其余全部，没有独立 test split。
- 正式 MLP/Ridge+MLP attempt 支持 `device: auto|cpu|cuda[:N]`，在 analysis 环境使用 PyTorch GPU；Ridge 基础模型仍为 NumPy 数值 checkpoint，训练期间保持冻结。

**没有旁路 `dycon difficulty-train`。** MLP 使用正式 `difficulty-mlp`，Ridge+MLP 使用正式 `difficulty-ridge-mlp`，通过统一 attempt/receipt/verifier 生成离线 checkpoint 和评测。不能以临时训练脚本绕行。`difficulty.estimator=null` 不自动选择模型；动态在线难度控制未实现。

已有数值契约测试 `tests/test_difficulty_estimators.py`；小型合成测试不等于实际数据训练或跨任务泛化验证。

## 来源运行与轨迹选择

新建 online index / resolved dataset 必须先确定来源 run，再以 `--rollout-id` 或 `--rollout-list` 筛选该 run 内轨迹；不再查询 Catalog 的 rollout 列表。共享实现是 `difficulty.datasets.selection.select_run(root, source_run, rollout_ids, rollout_list)`。跨 run 数据通过独立冻结数据集后 compose。已有 receipt 绑定的数据集继续独立验证，不依赖实时 Catalog 或其他 run 的评测状态。命令示例见 `docs/ENTRYPOINTS.md`。


## Trajectory fraction labels

Create a new resolved dataset with `--label-normalization trajectory_fraction` to derive causal, per-trajectory labels without rewriting raw labels. For a trajectory with `T` assistant steps and current close step `s`, the step target is `(T-s)/T`; the reasoning target is `remaining_reasoning_tokens_after / sum(finalized_reasoning_block_lengths)`. Fraction targets are `submission_remaining_steps_fraction` and `submission_remaining_reasoning_tokens_fraction`; train them through the existing Ridge, MLP, or Ridge+MLP commands. Existing datasets and checkpoints remain immutable.


## Configurable training and TensorBoard

Edit standalone `difficulty/training/config.yaml`, then run `source difficulty/scripts/train.sh` (or pass a YAML path). The source wrapper delegates to `dycon difficulty-train` in the analysis environment and preserves the caller shell via a subshell. Schema 2 selects dataset/split/targets, Ridge, MLP or Ridge+MLP, input site, MLP hidden dimensions/activation/normalization/dropout, optimizer, scheduler and device. Ridge remains closed-form; Ridge+MLP requires one verified published Ridge manifest per target. Every run creates a new immutable attempt.

TensorBoard binds to remote loopback port 6006 by default. On the local PC run `ssh -N -L 16006:127.0.0.1:6006 wangtianyi@10.249.187.9`, then open `http://localhost:16006`. MLP events expose evaluation-mode train/validation MSE per epoch and separate optimization loss; Ridge events use candidate-trial steps. Events are included in the attempt receipt. Check or stop the managed service with `dycon difficulty-tensorboard ATTEMPT_ID --action status|stop`.


## 重复 rollout 预测与真实剩余量对照（2026-10-27）

使用 `dycon difficulty-plot-trajectories RUN_ID PLOT_ID --attempt-id RIDGE_ATTEMPT --dataset-id RESOLVED_DATASET --target submission_remaining_reasoning_tokens_log1p`。
按 canonical repetition 统计成功次数；每条成功轨迹绘制预测与 native 真值双面板，同 task 同行汇总到自包含 HTML。缺失证据保留断线及说明图。
完整口径与统一 artifact 校验见 [轨迹对照说明](../docs/TRAJECTORY-COMPARISON-CN.md)。


### 预测与真值独立覆盖（2026-10-27）

在 `difficulty-plot-trajectories` 中显式添加 `--repair-id REPAIR_ID`，即可对全部可验证的 repaired canonical close hidden 预测，独立于真值是否可用。
预测、真值分别统计覆盖数；canonical-identity-only 预测点单列，不冒充 native token 标签。输出 `coverage-audit.json`，旧训练数据集和旧图保持不可变。
