# 剩余工作量估计与数据接口

同 action、不同 reasoning 的评分不稳定性评测使用 `dycon difficulty-evaluate-parallel RUN_ID EVALUATION_ID`，
实现与已有并行轨迹可视化同在 `difficulty/evaluation/` 并复用轨迹对齐。
支持平均两两差、标准差/极差、log1p 倍数分歧率和相对 Q，以及去重/原采样两种口径和 task/repo CI。
参数、分组、汇总与产物见 [使用说明](../docs/parallel-score-instability.md)。

并行解码的在线评分可使用 `dycon difficulty-plot-parallel RUN_ID PLOT_ID --rollout-id ROLLOUT_ID`
绘制实际选中轨迹的逐步预测折线及全部候选分支散点，并导出 PNG/SVG、评分 CSV 和来源 receipt。
无需加载 checkpoint、hidden tensor 或启动 GPU；具体选择、单位、缺失评分与噪音解释见
[并行轨迹评分可视化](../docs/parallel-score-visualization.md)。

新增正式 `dycon difficulty-ridge EXPERIMENT_ID`：在 analysis 环境消费冻结的 submission 选择，执行 task 分组的单层 Ridge × alpha 搜索、验证R²选优和参数/报告保存。`--mode dry-run` 只核验/划分，`--mode check-data` 另检查真实hidden及缓存，默认fit。见 [实验入口与语义](../docs/ridge-submission-pilot.md)。这是到该次提交的独立标签视图，允许已提交但未成功/未评测的轨迹，原成功标签不改。

2026-10-19 按实际代码更新。正式命令见 [任务入口表](../docs/ENTRYPOINTS.md)。已有索引、评分和 hidden 提取入口时直接复用，不另写临时脚本拼接数据或重定义标签。

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

`transform_target`/`inverse_target` 提供变换与逆变换。`train_estimators` 返回模型对象、transformed-space MAE/RMSE/R²、逆变换后的 validation predictions/targets；不自动输出 scorer 所需的 sample_id/target 预测清单或保存 checkpoint。预测格式见 [INPUTS.md](../docs/INPUTS.md)。

## Python API 和划分限制

`difficulty.training.train.load_training_data(cfg, index_id, target, split)` 排除缺失标签，返回 `(x, y_transformed, sample_ids)`。`train_estimators(cfg, index_id, target, kinds=..., **kwargs)` 读取 train/val、拟合并返回结果；kwargs 以 estimator 名为键，各参数见构造函数。

- schema 3 按 task partition 选择 train/val；未 resolved 或 censored 的标签在索引阶段置 null。2026-10-19 观察的 task 全为 train，不能将无 val 的现状称作完成验证。
- schema 1 按排序 rollout ID 约取 80%/20%；同 rollout 不跨组，但同 task 多条 rollout 可能跨组。该分支仅区分 train/val/其余全部，没有独立 test split。
- FFN 当前使用 CPU tensors/模块，没有 device/GPU 参数；Ridge 为 NumPy 实现。

**无通用 `dycon difficulty-train`，原训练模块没有 main/argparse。** 本次单层Ridge pilot使用上述正式 difficulty-ridge；FFN/残差训练及通用checkpoint工作流仍未扩展。不能以临时训练脚本绕行。`difficulty.estimator=null` 不自动选择模型；动态在线难度控制未实现。

已有数值契约测试 `tests/test_difficulty_estimators.py`；小型合成测试不等于实际数据训练或跨任务泛化验证。
