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
.venvs/analysis/bin/dycon difficulty-train ATTEMPT_ID \
  --training-config difficulty/training/experiments/EXPERIMENT.yaml
```

配置以 strict schema 校验。Ridge 标准化与拟合只使用 train；validation 选择 layer/alpha。`refit_train_validation: true` 时选定模型在 train+validation 重拟合，test 仍只用于最终评估。Attempt receipt 冻结配置、dataset/split 身份、特征契约、模型参数和产物 SHA。更改 dataset、split、native site、width 或 target 定义时使用新 attempt。

一般训练流程和其他模型入口见 [difficulty README](../difficulty/README.md)；`dycon difficulty-split --help` 与 `dycon difficulty-train --help` 给出当前 CLI 参数。
