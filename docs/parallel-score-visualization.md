# 并行轨迹剩余难度估计可视化

对“同 action、不同 reasoning”的波动进行定量评测，使用相邻的
`dycon difficulty-evaluate-parallel RUN_ID EVALUATION_ID` 入口；
平均两两差、std/range、倍数分歧率与相对 Q 见[一致性评测说明](parallel-score-instability.md)。

统一入口 `dycon difficulty-plot-parallel` 从已发布的并行 rollout 中选择一条最终轨迹，绘制每一步的选中分支评分折线及所有候选分支评分散点。适用于 Ridge、MLP 或其它估计器，直接使用当时保存的分数，不重新加载 checkpoint 或重算预测。

## 快速使用

在远程工程根目录执行；使用 analysis 环境，不需要 GPU、Docker 或模型服务：

```bash
cd /home/wangtianyi/DyConGlobal

# 先列出 run 内的轨迹、task、attempt、终止原因和评测状态
.venvs/analysis/bin/dycon difficulty-plot-parallel \
  parallel-ridge-event-b4-w12-disjoint31-20261019 --list

# 按 task 选择（仅在该 task 在本 run 中只有一个 attempt 时可省略 attempt）
.venvs/analysis/bin/dycon difficulty-plot-parallel \
  parallel-ridge-event-b4-w12-disjoint31-20261019 noise-sympy-18199 \
  --task-id sympy__sympy-18199 --attempt-id attempt-000

# 用 rollout ID 精确选择；只绘制 step 40..49（含两端）
.venvs/analysis/bin/dycon difficulty-plot-parallel \
  parallel-ridge-event-b4-w12-disjoint31-20261019 noise-sympy-18199-tail \
  --rollout-id r-dca23db3a050d75689bb93711d83e394 \
  --step-start 40 --step-end 49
```

`noise-sympy-18199` 等为自行指定的新产物 ID，已有 ID 拒绝覆盖。`--list` 仅需 RUN_ID，不接 PLOT_ID 或筛选参数。它列出 catalog 中的全部 attempts，不保证每项均已具备完整发布产物。无法唯一匹配 task 时，命令列出匹配的 attempt/rollout ID 并停止，不默认选最新或成功轨迹。被 quarantine/excluded 的轨迹拒绝分析。

等价的模块脚本入口（所有参数仍由同一个 CLI 处理）：

```bash
.venvs/analysis/bin/python -m difficulty.evaluation.scripts.plot_parallel \
  RUN_ID PLOT_ID --rollout-id ROLLOUT_ID
```

## 图的含义

- 横轴：最终 canonical trajectory 的实际 agent step，从 0 开始，不按 decision UUID 或文件名排序。
- 蓝色折线：每一步实际选中且进入最终轨迹的分支评分。使用保存的 selected 标记，不重新选择最小分支；fallback 时也保留实际选择。
- 橙色圆点：该步所有有有限评分且 eligible 的分支，包含选中分支；其圆点可能被蓝线上的点覆盖。各分支共享真实 step 横坐标，不加随机 jitter。
- 橙色叉号：有有限评分但不属于 eligible 的分支，例如不满足输出格式/长度要求的候选。
- 橙色底部竖线：分支评分缺失；它是坐标轴标记，不表示分数为 0。
- 蓝色底部叉号及折线断点：实际 step 无可用的选中评分，包括非每步分支策略下没有并行决策的 step。缺失值不填 0、不插值、不跨断点连线。

纵轴始终使用当时估计器的原始输出单位。目标完整名称写在图上；`*_log1p` 为 log1p 空间，`*_steps_linear` 为 step 原单位，其它目标显示 native score。负预测保留，不做截断、平滑、指数逆变换或第二次取 log。PNG 适合预览，SVG 适合放大长轨迹或排版。

各 step 的分支都基于选中轨迹在该步的同一个历史前缀；散点不是四条独立延续到终点的轨迹。分支离散程度可以辅助检查估计稳定性，但包含 reasoning/action 差异，不能直接视为纯测量噪音或置信区间。蓝线是“实际轨迹上的预测”，不是剩余量真值；最后一步的预测不强制归零。submitted、resolved 和 limits exceeded 等终止状态也不混同。

最低分选择会使蓝线自然贴近候选散点的下沿，这是选择机制产生的偏差，不能仅凭蓝线低于其它候选断言其误差更小。

## 输入与对齐

命令复用现有模块，不引入第二套 rollout、hidden 读取或训练流程：

1. `baseline/manifest.json` 和现有 `select_trajectories/read_actions` 选择并验证 canonical trajectory SHA，获取实际 assistant step。
2. `core.config.load_snapshot` 读取来源 run 的冻结配置，以其中的 NAS/export 路径定位数据。当前 `config.yaml` 用于确定 project root，不用它当前的 checkpoint 替换历史估计器。
3. 查找唯一匹配 rollout ID 的已发布 `sessions/*/finalization.json`，核验 canonical/model 身份。
4. `hidden.artifacts.read_branch_index` 校验 branch receipt、决策及已有 classification revision；它不加载 hidden tensor。无需事先创建 online-index 或 action-classify 产物。
5. 按选中候选的 native request ID、finalization 中的 message index/agent step 及 branch index 对齐，仅将真正进入轨迹的 action response 接入折线。

FormatError 修正请求不增加实际 action step；没有赢家的终止 decision 不伪造额外一步。这些排除项记录在 `summary.json`。缺少 finalization、未发布/被修改的 receipt、目标/估计器身份不一致、重复或无法绑定的 decision 会明确失败；应先通过原 capture 发布/恢复流程补全证据。此工具不读取活跃 spool，也不自行修复历史产物。

## 产物

保存到 `difficulty/evaluation/outputs/PLOT_ID/`，该目录已被工程 `.gitignore` 忽略；代码和文档保持版本管理。

| 文件 | 内容 |
|---|---|
| `trajectory.png`、`trajectory.svg` | 同一张折线＋候选散点图 |
| `steps.csv` | 选中分支、选中评分、全部有限分支评分的极差/总体标准差、有限/缺失/eligible 数量、fallback 原因 |
| `branches.csv` | 每个实际 step 的每个分支、score、selected、status/reason、native request/sample ID |
| `scores.json` | 与两张 CSV 对应的结构化数据；缺失值为 null |
| `selection.json` | catalog revision、选中 canonical trajectory 及 SHA、run/task/model/评测身份 |
| `summary.json` | 来源、目标、评分覆盖、绘制范围、排除的重试/终止 decision |
| `receipt.json` | 输入 snapshot/finalization/branch receipt、代码和全部输出文件的 SHA256 |

`steps.csv` 的 spread/std 使用全部有限候选分数，包含有分数但不 eligible 的候选；用 `branches.csv` 的 status 可另做 eligible-only 统计。所有计数除 `canonical_steps` 和排除项外均针对绘制范围。CSV 缺失值为空，不是零。输出先写入临时目录，成功后原子发布；失败不会留下可误认成已完成的 PLOT_ID 目录。

## 代码与测试

- 编排与校验：`difficulty/evaluation/parallel.py`。
- 画图：复用现有绘图模块 `difficulty/evaluation/visualize.py`。
- CLI：`core/cli.py`；薄脚本 `difficulty/evaluation/scripts/plot_parallel.py`。

```bash
.venvs/analysis/bin/python -m pytest -q tests/test_parallel_visualization.py
```

测试覆盖跨文件顺序、非零赢家、fallback 不重选最小值、稀疏策略断点、无评分分支、FormatError/终止请求隔离、多 attempt 歧义、篡改证据拒绝、真实无显示器 PNG/SVG 输出与 receipt。
