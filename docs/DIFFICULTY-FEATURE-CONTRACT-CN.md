# Difficulty 特征、划分与训练契约

## Hidden 特征身份

Ridge loader 从冻结数据集的 `storage_shapes` 推导 hidden width 与 native site。一个数据集中的 rollout 必须满足同一特征契约；宽度、site 布局或模型 fingerprint 不一致时拒绝混合。当前 Qwen3.6-27B capture 使用 32 个 native sites（0–31）、每 site width 5120。旧 4096 维格式使用 sites 0–30 与 32；仅在数据集明确属于该旧格式时按其冻结映射读取。

每个样本绑定 canonical rollout、trajectory SHA、close anchor、native site、storage shape 和源文件 SHA。loader 校验 receipt、tensor shape、finite 值及模型 fingerprint。Cache 按 site 写入 float32 矩阵，其身份包含 dataset receipt、样本顺序、width 和 native site contract；改变来源或特征布局需新建 dataset/cache，不能沿用旧 checkpoint。

## 评测资格与标签终点

`difficulty-dataset resolved` 默认采用 `resolved_only`：样本须绑定 completed evaluation、被选 patch 的精确 subject 与 `resolved=true`。Submission 不是成功标签。

Workspace evaluation 可显式使用 `--eligibility positive_reward` 纳入 completed 官方评测中 finite 且大于 0 的 numeric reward。此模式仍校验 evaluation report、subject、attempt 和 reward；它不把部分得分改写为 `resolved=true`。resolved-only 标签从 close 到成功提交；positive-reward 标签从 close 到该轨迹实际结束。两种终点不能混作同一目标。

标签来自已 finalize 的 native exact suffix；未评测、无效、截尾或缺失的值不补造。positive-reward 数据集在 receipt 中冻结资格策略，loader 会重新核验绑定的官方评测报告。

## Task manifest 划分

当 catalog partitions 必须约束 task holdout 时，使用 `task_manifest_v1`：

```bash
.venvs/analysis/bin/dycon difficulty-split DATASET_ID SPLIT_ID \
  --strategy task_manifest_v1 --seed 42 --validation-fraction 0.2 \
  --task-partitions catalog/partitions/SPLIT_NAME/split.json
```

Manifest 中每项使用 `task_key`、`instance_id` 和 `train|val|test` partition。样本 task 必须在 manifest 中且身份一致；同 task 的全部 rollout、repetition 和 close samples 进入同一 split。`val` 映射到 validation。若 manifest 只有 train/test，validation 从 train tasks 按 seed 和比例确定；test 保持 holdout。无样本的分区、缺任务、冲突 task identity 或相同 trajectory 跨 split 均拒绝。未出现在 dataset 中的 test task 记为 `external_test_tasks`。

Split 是 dataset 下的独立不可变产物，不改写 dataset receipt。训练配置引用已冻结 split ID 与 receipt；Catalog partition 不可直接冒充训练 split。

## Ridge 训练

Standalone schema-2 YAML 绑定 dataset ID/receipt、split、targets、public site、Ridge layers、alphas 和训练模式：

```bash
.venvs/analysis/bin/dycon difficulty-train \
  --training-config difficulty/training/experiments/EXPERIMENT.yaml
```

配置以 strict schema 校验。Ridge 标准化与拟合只使用 train；validation 选择 layer/alpha。`refit_train_validation: true` 时选定模型在 train+validation 重拟合，test 仍只用于最终评估。Attempt receipt 冻结配置、dataset/split 身份、特征契约、模型参数和产物 SHA。更改 dataset、split、native site、width 或 target 定义时使用新 attempt。

一般训练流程和其他模型入口见 [difficulty README](../difficulty/README.md)；`dycon difficulty-split --help` 与 `dycon difficulty-train --help` 给出当前 CLI 参数。


## Dataset Ridge 外层任务留出评估

完整配置、两层协议、指标口径、失败恢复与 2026-11-09 真实验收记录见[任务留出评估专题](DIFFICULTY-RIDGE-HOLDOUT-CN.md)。

正式入口（analysis 环境）一次完成来源校验、外层 task train/test、内层 event train/validation、
每目标独立 layer × alpha 搜索、outer_train 重拟合、全量 outer_test 预测及 NMSE/R²：

```bash
.venvs/analysis/bin/dycon difficulty-ridge-holdout EVALUATION_ID \
  --workflow-config difficulty/training/experiments/ridge-holdout.yaml
.venvs/analysis/bin/dycon artifact verify EVALUATION_ID --scope difficulty-evaluation
```

编辑该独立 workflow YAML 指定已发布 dataset。`run.mode: dry-run|check-data|fit` 分别对应只检查来源与划分、
再检查 hidden/cache、实际拟合。`model.ridge.layers: auto` 展开 dataset 契约的全部合法单层；
默认 13 个严格正 alpha（10^-4 至 10^8），不拼接层。配置、完整网格和特征契约冻结在 evaluation 中。
外层默认 seed=42、test_fraction=0.2；内层默认 seed=42、validation_fraction=0.2，其分母仅为 outer_train。
内层同轨迹相关性允许存在；跨任务泛化只看 outer_test。外层 WebArena 当前因 template 泄漏约束明确拒绝。

`outer_split.strategy: task_manifest_v1` 使用 `outer_split.task_partitions: FILE.json`；
此时不要填写随机 seed/test_fraction。JSON 为 `{"task_assignments":[{"task_key":"DATASET_TASK_KEY",
"instance_id":"TASK_ID","partition":"train"}]}`，须恰好覆盖所有 dataset task，partition 只允许 train/test。
task_key 从冻结 rollouts.json 读取，SkillsBench 包含 revision；未知、遗漏、重复和冲突身份均拒绝。

`visualization.enabled: false` 跳过 Ridge 搜索图、轨迹图和 HTML，且不加载 matplotlib；
预测、指标及 coverage audit 仍完整生成。默认 true，每个 test rollout/repetition、每个 target 均导出 PNG/SVG，
HTML 自包含并支持筛选。图只读取权威预测，无二次推理；positive_reward 中 resolved=false 的部分成功轨迹也保留。
fraction 仅使用 dataset 已发布标签，不临时用完整轨迹真值归一化 raw 预测。

产物：`difficulty/evaluations/EVALUATION_ID/` 的 workflow、resolved-config、schema-2 training-config、
cohort、coverage、metrics、by-task/by-rollout CSV、predictions、report，以及可选 plots/index.html。
split 属于原 dataset，attempt 仍由统一 training service 管理并独立封存。
主指标为原单位 pooled NMSE_energy=SSE/sum(y²)、NMSE_variance=SSE/sum((y-mean(y))²)、R²；
另报拟合空间、分组与等权 macro、outer_train 均值基线。有限配对计数及未定义原因显式记录，
零能量、零方差、单点、非有限预测和逆变换溢出不冒充零误差。预测不裁负、不平滑、不强制单调。

每次使用新 evaluation ID，含未完成目录也拒绝覆盖。若训练完成后绘图失败，保留诊断且不写 evaluation 成功 receipt；
使用新 ID 和 `--reuse-attempt-id ATTEMPT_ID` 正式恢复，dataset/split、特征契约、完整有效训练配置与 receipt 必须相同。
关闭可视化不能用于改变已有 attempt 的有效训练配置。任一 test event 失败则标记 partial、无成功 receipt。
Python 调用方使用同一 `difficulty.evaluation.holdout.run` service。

Standalone 训练仍为 `dycon difficulty-train --training-config CONFIG.yaml`，attempt ID 写在 YAML 的
`run.attempt_id`，不是 CLI 位置参数。Standalone `model.ridge.plots_enabled` 默认 true，以保持旧入口行为。


### 嵌套协议与原 task manifest 的区别

`task_holdout_event_v1` 的外层 manifest 仅接受 train/test，须恰好覆盖 dataset 全部 task；
内层在 outer_train 中按 event 二划分，允许共享 task/轨迹。
它与上文独立 `difficulty-split --strategy task_manifest_v1` 的 task validation 协议不同，
不接受多余的 external_test_tasks，也不将 outer_test 写成内层第三个 role。
