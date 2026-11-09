# Dataset Ridge 任务留出评估与全量轨迹可视化

`difficulty-ridge-holdout` 对一个已发布、具备训练资格的 difficulty dataset 完成来源校验、两层划分、Ridge 网格搜索、冠军重拟合、全量外层 test 预测与 NMSE/R² 统计，并可为全部 test 轨迹生成对照图和自包含 HTML。正式执行使用 analysis 环境，不启动 rollout、GPU 推理或 hidden 重放。

## 入口与配置

在项目根目录执行；每次使用新的 evaluation ID，包括上次失败或仅检查数据的情况：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/analysis/bin/dycon difficulty-ridge-holdout EVALUATION_ID \
  --workflow-config difficulty/training/experiments/ridge-holdout.yaml
.venvs/analysis/bin/dycon artifact verify EVALUATION_ID --scope difficulty-evaluation
```

随工程提供的 [完整配置](../difficulty/training/experiments/ridge-holdout.yaml) 使用 `skillsbench-positive77-terminal-20261102`；[smoke 配置](../difficulty/training/experiments/ridge-holdout-smoke.yaml) 使用同一数据集，限制为 2 层 × 3 alpha。新实验应明确选择 dataset，并调整配置及 evaluation ID。

```yaml
schema_version: 1
run:
  mode: fit
data:
  dataset_id: skillsbench-positive77-terminal-20261102
  expected_receipt_sha256: null
  targets:
    - submission_remaining_reasoning_tokens_log1p
    - submission_remaining_steps_linear
outer_split:
  strategy: task_random_v1
  seed: 42
  test_fraction: 0.20
inner_split:
  strategy: event_random_v1
  seed: 42
  validation_fraction: 0.20
model:
  kind: ridge
  ridge:
    layers: auto
    alphas: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0]
    standardize: true
    blas_threads: 4
evaluation:
  nmse_normalization: target_energy
visualization:
  enabled: true
  export_formats: [png, svg]
```

`expected_receipt_sha256` 可指定预期 dataset receipt；为 null 时仍验证并冻结实际 SHA。online index 和仅供可视化的 publication 不能代替训练 dataset。参数经严格 schema 校验；不使用 rollout 根配置中的训练默认值填补搜索网格。

`run.mode` 在 YAML 中配置，当前命令没有 `--mode` 参数：

| mode | 行为 |
|---|---|
| `dry-run` | 校验配置、dataset 来源与特征契约，冻结两层划分及配置；不读取 hidden 建缓存，不拟合 |
| `check-data` | 另核验 hidden/cache；不拟合 |
| `fit` | 执行搜索、重拟合、test 预测、指标与可选绘图 |

三种模式均生成独立 evaluation 和 attempt，receipt 明确记录真实模式。`layers: auto` 展开 dataset 契约中的全部合法 public site，也可使用合法整数列表。每次候选仅输入一层，不拼接；层索引从 0 开始，public/native 映射与宽度来自冻结契约。每个 target 独立按 inner validation 拟合空间 pooled R² 选优；容差 1e-10 内优先较大 alpha，再较小 public layer。

Python 调用方使用同一服务，`cfg` 必须是项目解析出的冻结配置：

```python
from core.config import resolve
from difficulty.evaluation.holdout import run

cfg = resolve("config.yaml")
summary = run(cfg, "NEW_EVALUATION_ID",
              workflow_config="difficulty/training/experiments/ridge-holdout.yaml")
```

## 两层划分与拟合边界

| cohort | 构成与用途 |
|---|---|
| outer_train | 外层训练 tasks 的全部 dataset events；用于内层划分和最终重拟合 |
| inner_train | outer_train 内分配的训练 events；只在这里拟合候选 scaler 与 Ridge |
| inner_validation | outer_train 内分配的验证 events；用于选择 layer/alpha |
| outer_test | 外层留出 tasks 的全部 dataset events 和 rollout/repetition；只用于最终评估与绘图 |

外层使用 benchmark-aware task identity，同一 task 跨 source run、repetition 和 rollout 始终属于同一外层角色。SkillsBench 身份包含 revision；SWE-bench 包含 repo 与 instance_id。随机外层 test task 数为 `min(G-1, max(1, round(G*test_fraction)))`，采用 Python ties-to-even 取整；少于两个 task group 时拒绝。

内层按 source_kind 分层做确定性的 event 二划分。单个 event 的小分层留在 train；其余分层将 validation 数限制在 1 至 n−1，全局 train 或 validation 为空时拒绝。内层 validation 比例的分母仅为 outer_train events。同 task、同轨迹的不同 events 可跨内层两组，因此内层分数存在轨迹相关性。

选定冠军后，在全部 outer_train（inner_train + inner_validation）上重新拟合 scaler 和 Ridge，再对 outer_test 预测。outer_test 不参与候选标准化、拟合、选优或重拟合。跨 task 泛化以外层 test 为准。

冻结策略名为 `task_holdout_event_v1`：`sample_ids` 仅含内层 train/validation，外层 test 保存为独立 `outer_test_sample_ids`，并记录 outer task assignment、分配计数、实际比例和样本顺序 hash。创建、加载及 artifact verifier 核验完整分配与外层 task/轨迹/sample 隔离。

首版拒绝 WebArena：现有实现要求按 intent_template_id 分组以避免 template 泄漏，当前协议尚未定义其与内层 event 划分的兼容方式。详见[特征与划分契约](DIFFICULTY-FEATURE-CONTRACT-CN.md)。

## 显式外层 task manifest

在 workflow YAML 中替换 `outer_split`，此时不填写随机 seed/test_fraction：

```yaml
outer_split:
  strategy: task_manifest_v1
  task_partitions: path/to/outer-tasks.json
```

JSON 格式如下；实际文件必须恰好覆盖 dataset 的全部 task，task_key 与 instance_id 从冻结 `rollouts.json` 读取：

```json
{
  "task_assignments": [
    {"task_key": "DATASET_TRAIN_TASK_KEY", "instance_id": "TRAIN_TASK_ID", "partition": "train"},
    {"task_key": "DATASET_TEST_TASK_KEY", "instance_id": "TEST_TASK_ID", "partition": "test"}
  ]
}
```

仅允许 train/test；未知、重复、遗漏、冲突身份和空分区均拒绝。这与独立 `difficulty-split --strategy task_manifest_v1` 的三路 task 协议不同：本工作流仍在 outer_train 内按 event 做二划分，不从 outer_train 再划出 task validation。Catalog partition 不能直接代替冻结 split。

## 指标、覆盖与图表

主报告使用原单位 `test_raw_units`，同时提供拟合空间 `test_transformed`。log1p 目标通过 expm1 逆变换；linear/fraction 保持其定义。fraction 仅使用 dataset 已发布的相应标签，不在预测时用完整轨迹真值分母临时转换 raw 目标。

```text
SSE = Σ(y − prediction)²
NMSE_energy = SSE / Σ(y²)
NMSE_variance = SSE / Σ((y − mean(y))²)
R² = 1 − NMSE_variance
```

默认主 NMSE 为 energy；`evaluation.nmse_normalization: variance` 改变主字段选择，两种完整字段始终保留。输出 pooled、by_task、by_rollout、等权 macro_task/macro_rollout，以及仅以 outer_train 标签均值形成的基线。macro 记录有效/未定义组数，与 pooled 分开报告。

主指标只计算有限真值与预测的配对，并记录预期、有效与排除数量及分母。N=0、单点、零能量、零方差、非有限预测、算术/逆变换溢出记录原因，未定义值写 JSON null；合法负 R² 保留。任一预期 test event 失败则不写全量成功 receipt。

图表只读取 attempt 的权威 `test-predictions.jsonl`，不再次推理。cohort 是 dataset 内全部 outer_test rollout/repetition 及其已发布 close events，不使用 source run 的默认 resolved 筛选，不加入 dataset 外的轨迹或补造 hidden/标签。positive_reward 数据集中 `resolved=false` 的部分成功轨迹仍被保留；其真值终点是实际 agent terminal，不表示最终成功。

默认每个 target、每条 test 轨迹导出 PNG/SVG，横轴为 agent_step，原单位纵轴与 log1p 目标的拟合空间面板对照；标注 task、source run、repetition、rollout、最优 public/native site、宽度、alpha、checkpoint SHA 和轨迹指标。预测不裁负、不强制单调，缺失点保留断线。自包含 HTML 支持 task/轨迹筛选，并为每条预期轨迹显示覆盖和状态。

设置 `visualization.enabled: false` 跳过 Ridge 搜索图、轨迹图与 HTML，且不加载 matplotlib；划分、搜索、重拟合、预测、指标、CSV 和 coverage audit 仍执行，summary 标记 `skipped_by_config`。

## 产物与恢复

```text
difficulty/datasets/DATASET_ID/splits/SPLIT_ID/
  split.json, receipt.json
difficulty/checkpoints/ATTEMPT_ID/
  attempt.json, selected.json, models/, trials.jsonl, metrics.json
  test-predictions.jsonl, receipt.json, ...
difficulty/evaluations/EVALUATION_ID/
  workflow.json, resolved-config.json, training-config.yaml
  cohort.json, coverage.json, metrics.json
  by-task.csv, by-rollout.csv, predictions.jsonl, prediction-source.json
  report.md, receipt.json
  plots/, plot-manifest.json, index.html  # visualization.enabled=true
```

输出目录受锁保护，已有目录（含未完成目录）拒绝覆盖。workflow 记录阶段状态和子产物 ID；实际 attempt ID 由工作流派生，可从返回摘要读取。`training-config.yaml` 是 receipt 绑定的 schema-2 快照；checkpoint 仍由现有训练 service 管理并在完成后封存。evaluation 引用 dataset、split、attempt receipt 与 selected-refit checkpoint，复制预测时核对字节 SHA。检查和失败模式的产物与实际 fit 明确区分。

已完成训练而绘图失败时，保留诊断产物，不向封存 attempt 追加图表。使用新 evaluation ID 和相同有效配置正式重建评测：

```bash
.venvs/analysis/bin/dycon difficulty-ridge-holdout NEW_EVALUATION_ID \
  --workflow-config difficulty/training/experiments/ridge-holdout.yaml \
  --reuse-attempt-id COMPLETED_ATTEMPT_ID
```

复用仅支持 fit，且 dataset/split、targets、特征契约、完整有效训练配置和模型 receipt 必须匹配；不能通过修改可视化开关改变被复用 attempt 的有效配置。无需手工编辑 attempt 文件。

## 2026-11-09 验收记录与结果解读

使用 `skillsbench-positive77-terminal-20261102`：43 个 task、77 条 rollout、2,572 个 close events；dataset receipt SHA256 为 `3b13f468e92b3aba937cb934a70a8e688365374da2d35b8e71fe6820d10b56c0`。此处为冻结实验记录，后续实验应使用自己的实际 cohort 和 receipt。

外层为 34 个 train tasks、9 个 test tasks；内层为 1,657 个 train events、414 个 validation events。外层 test 为 18 条轨迹，每个目标 501 个 events，其中 6 条轨迹 `resolved=false`，仍按 positive_reward 资格纳入。

smoke（2 层 × 3 alpha × 2 目标）通过后，完整运行 32 层 × 13 alpha × 2 目标，共 832 个候选，使用 4 个 BLAS 线程。全部预期 test events 均完成预测，生成 36 PNG、36 SVG 和完整 HTML 索引。

| 目标 | 最优 public/native site | alpha | 内层 validation 拟合空间 R² | 外层 test 拟合空间 R² | 外层 test 原单位 R² | 原单位 NMSE_energy |
|---|---:|---:|---:|---:|---:|---:|
| 剩余 reasoning tokens，log1p 拟合 | 31 | 100 | 0.816624 | 0.660584 | -0.814806 | 1.036428 |
| 剩余 steps，linear 拟合 | 28 | 100 | 0.847154 | 0.135362 | 0.135362 | 0.512605 |

这里的 site 是单个输入 hidden 层，不是网络叠层数量；两目标宽度均为 5120。内层允许轨迹相关性，且 tokens 的拟合空间和原单位空间不同，不能把 validation/log1p 的高分当成原单位跨任务泛化效果。原单位 tokens R² 为负，低于 test 均值基线；本次 outer_train 均值基线 R² 也为负（-0.028572），但优于该 tokens 模型。steps 有有限的泛化能力。功能与覆盖验收通过不意味着模型质量已达到应用要求。

| 产物 | ID |
|---|---|
| smoke evaluation | `skillsbench77-holdout-smoke-20261109` |
| 完整网格 evaluation | `skillsbench77-holdout-full-20261109` |
| 最终版式 evaluation | `skillsbench77-holdout-full-plots-20261109` |
| 完整网格 attempt | `skillsbench77-holdout-full-20261109-ridge-427dfe7355243cb5` |
| dataset-owned split | `skillsbench77-holdout-full-20261109-split-46d38ca629bbb4c4` |

最终版式通过 `--reuse-attempt-id` 重建，预测、指标、coverage、cohort 和两份分组 CSV 与完整训练 evaluation 的文件 SHA 一致；没有重复拟合。最终 evaluation receipt SHA256 为 `c654ee43fdce07dbdb921e69cd4a402ecad58f1809bac458c1060e9da052ee04`。

[最终轨迹 HTML](../difficulty/evaluations/skillsbench77-holdout-full-plots-20261109/index.html) · [完整指标](../difficulty/evaluations/skillsbench77-holdout-full-plots-20261109/metrics.json) · [覆盖审计](../difficulty/evaluations/skillsbench77-holdout-full-plots-20261109/coverage.json)

新增验收和相关回归共 91 项通过，包括隔离、确定性、test 标签不影响模型参数、candidate/refit scaler 范围、checkpoint 数值重放、可视化开关一致性、跨 run compose/fraction、失败恢复和来源校验。smoke、完整网格和最终 evaluation 均通过正式 artifact verifier：

```bash
.venvs/analysis/bin/dycon artifact verify skillsbench77-holdout-full-plots-20261109 \
  --scope difficulty-evaluation
```

测试文件为 `tests/test_ridge_holdout.py`、`test_ridge_probe.py`、`test_ridge_task_threeway.py`、`test_difficulty_feature_contract.py`、`test_difficulty_artifacts.py`、`test_ridge_target_transforms.py`、`test_explicit_trajectory_view.py`、`test_trajectory_sources.py`、`test_fraction_labels.py`、`test_trajectory_comparison.py`、`test_training_fit_provenance.py`（均在 `tests/` 下）。当次日志保存在 `temp/difficulty_holdout_implementation_20261109/regression-postplots.log`；历史测试记录不代替修改后的重新验证。

Standalone 训练入口仍为 `dycon difficulty-train --training-config YAML`，attempt ID 写在 YAML 的 `run.attempt_id`。一般训练说明见[训练 USAGE](../difficulty/training/USAGE.md)，统一入口规则见 [ENTRYPOINTS](ENTRYPOINTS.md)。
