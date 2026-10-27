# 分支 action 统计、熵与轨迹进度曲线

2026-10-26。正式入口：`.venvs/analysis/bin/dycon action-branch-stats`。
实现：`action/analyzers/branches.py`、`action/evaluation/branch_histograms.py`、
`action/evaluation/branch_curves.py`。读取现有 canonical trajectory 和 parallel decision；
不运行模型，不执行轨迹命令，不改变 estimator 或 rollout 配置。

## 1. 轨迹选择与运行

最小绘图单位是 `(source_run_id, rollout_id)`。使用 `--rollout-id` 精确选择轨迹，
或用 `--rollout-list` 绑定清单选择多条；不再提供 `--task-id` 绘图参数。
同一 task 的不同 repetition/attempt 只作为不同 rollout 处理，不按 task 合并或平均。

```bash
cd /home/wangtianyi/DyConGlobal

# 指定一条轨迹，绘制原始 step，不归一化、不跨轨迹平均。
.venvs/analysis/bin/dycon action-branch-stats NEW_SINGLE_ID \
  --source-run RUN_A --rollout-id ROLLOUT_ID --line-mode per-trajectory

# 某个 run 的全部 eligible/final 轨迹：每条独立一张图。
.venvs/analysis/bin/dycon action-branch-stats NEW_BATCH_ID \
  --source-run RUN_A --line-mode per-trajectory

# 仅绘制该 run 的 Submitted 轨迹；每张图只显示 Shannon 熵。
.venvs/analysis/bin/dycon action-branch-stats NEW_ENTROPY_ID \
  --source-run RUN_A --termination-reason Submitted \
  --line-mode per-trajectory --line-metric entropy --format png

# 同一张轨迹图同时显示多个指标，重复 --line-metric。
.venvs/analysis/bin/dycon action-branch-stats NEW_SELECTED_METRICS \
  --source-run RUN_A --rollout-list bindings.json --line-mode per-trajectory \
  --line-metric normalized_entropy --line-metric winner_uncertainty

# 保留 run 平均图，单 run 和跨 run 都按 0–100% 归一化进度展示。
.venvs/analysis/bin/dycon action-branch-stats NEW_RUN_AVERAGES \
  --source-run RUN_A --source-run RUN_B --rollout-list bindings.json --line-mode both

.venvs/analysis/bin/dycon action-branch-stats NEW_BATCH_ID --check
```

多 run 的 binding 清单：

```json
[
  {"source_run_id": "RUN_A", "rollout_id": "R1", "cohort": "a"},
  {"source_run_id": "RUN_B", "rollout_id": "R1", "cohort": "b"}
]
```

可以附加 `trajectory_sha256` 并验证真实字节。单 run 也接受 JSON ID 数组。
多个 run 中同名 rollout 用 source run 区分；`--rollout-id` 只用于单个 source run。
省略 rollout 选择时复用 difficulty/action 的 eligible/final selection；进行中的重采样 run
可能拒绝冻结默认全集，此时显式提供已经发布的轨迹 ID。
沿用原有字节级 duplicate/alias 规则，实际 canonical 轨迹与别名记录在 `selection.json`。
`--termination-reason Submitted` 只是可选状态筛选，不代表 harness resolved。

| 参数 | 默认值 / 可选值 | 含义 |
|---|---|---|
| `--source-run` | 创建分析必需，可重复 | 指定来源 run |
| `--rollout-id` / `--rollout-list` | 互斥 | 直接指定一条/多条轨迹 |
| `--line-mode` | `both`；`per-trajectory/per-run/all-runs/both/none` | 逐轨迹独立图、run 平均图或禁用折线 |
| `--line-metric` | 可重复 | 选择图中的指标面板，不改变统计计算和导出 |
| `--branch-scope` | `both`；`all/retained/both` | 全部分支和历史 estimator 保留分支 |
| `--progress-points` | `101`，范围2–1001 | run 平均图的进度插值网格；逐轨迹图不使用插值 |
| `--termination-reason` | 不额外筛选 | 可选精确终态筛选 |
| `--format` | PNG 和 SVG，可重复 | 同一张图可同时保存两种格式 |
| `--dpi` | `180`，范围50–600 | 图像分辨率 |
| `--check` | 既有分析 ID | 校验输入、原始数值、图像清单与 receipt，不覆盖产物 |

`retained` 直接使用历史 `retained_branch_ids`，不重新计算阈值或分数。
没有此名单的非 consensus run 只能用 `--branch-scope all`。
缺失原始分支证据会明确报错，不把丢失的 action 当成无 action。

## 2. 每个 step 的符号与计算

每个分支的一整组**有序工具调用**算一个 action，复用
`core.action_identity.canonical_action`。保留命令字符串内的空格、换行、引号和调用顺序；
忽略 reasoning、call ID 和 JSON 包装层键顺序。无可比 action 的分支单独计数，
不伪造为额外 action 类别。

以下符号在 `all` 和 `retained` 两个集合内分别定义：

- `N`：有可比 action 的分支数，即 `valid_actions_SCOPE`。
- `K`：不同 action 的种类数，即 `action_types_SCOPE`。
- `n_i`：第 i 种 action 的分支票数；各组票数见 `group_sizes_SCOPE`。
- `p_i = n_i / N`：第 i 种 action 的分支比例。
- `w`：历史系统实际选中的 canonical action，绝不在 all 集合内重新选最大票组。
- `n_w`：统计集合内对 w 的支持分支数，即 `winner_count_SCOPE`。
- `q = n_w / N`：实际 winner 支持率。

| 字段前缀 | 计算 | 含义 |
|---|---|---|
| `entropy` | `H = -sum(p_i * ln(p_i))` | 同时反映 action 种类与票数分散程度 |
| `normalized_entropy` | `H / ln(K)`，K=1 时为 0 | 已出现 action 之间的相对均匀度 |
| `effective_action_count` | `E = exp(H)` | 与当前分布等熵的均匀 action 数 |
| `winner_support` | `q = n_w / N` | 对真实 winner 的支持比例 |
| `winner_uncertainty` | `U = 1-q` | 真实 winner 缺少支持的程度 |
| `winner_surprisal` | `I = -ln(q)` | 真实 winner 的自信息/惊奇度 |

自然对数，H 和 I 的单位为 nats。每个字段分别加 `_all` 或 `_retained` 后缀。
第五类“筛选影响”不是新的熵：`delta_METRIC = METRIC_retained - METRIC_all`。
输出 `delta_entropy`、`delta_normalized_entropy`、`delta_effective_action_count`、
`delta_winner_support`、`delta_winner_uncertainty` 和 `delta_winner_surprisal`；仅 `both` 有定义。

边界：

- `N=0`：K 和有效分支数为 0；H、归一化 H、E、q、U、I 为 null，而不是伪造 0。
- `K=1, N>0`：H=0、归一化 H=0、E=1。
- 实际 winner 不可识别：支持数及 q/U/I 为 null。
- `q=0`：U=1；数学上 I 为正无穷。JSON/CSV 的 I 用 null，并以
  `winner_surprisal_status_SCOPE=positive_infinity` 明确区分；不裁剪为有限数。
- 无穷 I 不进入有限均值、插值和折线；summary 单列 `positive_infinity_steps`。
  有一侧 I 不有限时，`delta_winner_surprisal` 也为 null。

注意：`H/ln(K)` 只衡量相对均匀度，不会把“两种各半”和“32种各一票”区分开。
不能仅凭它判定探索规模；应同时看 H、E、K、N 和 q。
summary 先逐 step 计算再取平均，例如 `mean(E)` 通常不等于 `exp(mean(H))`。

## 3. 直方图：问题（1）

默认始终生成选择范围内的四张直方图（`both` 模式）：

- `plots/action_types_all.*`
- `plots/action_types_retained.*`
- `plots/winner_count_all.*`
- `plots/winner_count_retained.*`

横轴为 action 种类数或真实 winner 支持分支数，纵轴为 step 数；
整数分箱、柱顶标注 step 数。每个 canonical parallel step 等权，包含提交 step。
0 是合法的整数分箱；无法识别 winner 的 step 以 N/A 单列，不用 0 代替。
多 run 一起分析时，这四图是选定 steps 的合并分布；各 run 数字另列 summary。
连续的熵指标不塞入这些整数分箱，保存在逐 step 明细、汇总和折线图中。

## 4. 折线绘图

### 每条轨迹独立一张图

使用 `--line-mode per-trajectory`。图文件为：

```text
plots/per-trajectory/SOURCE_RUN_ID/ROLLOUT_ID.png
plots/per-trajectory/SOURCE_RUN_ID/ROLLOUT_ID.svg
plots/trajectory-index.json
```

每张图只包含这一条轨迹的 canonical assistant steps，横轴从1开始，
不归一化、不插值、不跨轨迹平均。即使两条轨迹属于同一个 task，也分别成图。
标题标注 source run、rollout ID；instance ID 只是说明信息，不作为分组键。

默认将以下五个指标画在同一张多面板图中，all 实线、retained 虚线：

| `--line-metric` 名称 | 指标 |
|---|---|
| `entropy` | Shannon 熵 H，nats |
| `normalized_entropy` | 归一化熵 H/ln(K)，表示相对均匀度 |
| `action_types` | 有可比 action 的不同种类数 K |
| `effective_action_count` | 有效 action 数 / Perplexity，exp(H) |
| `winner_uncertainty` | Winner 不确定性，1−q |

`--line-metric entropy` 为每条轨迹生成仅含熵的一张图；重复该参数可选择多个面板。
还支持 winner_support、winner_surprisal、winner_count、valid_actions 和六种 delta 指标，
完整名称见 CLI `--help`。delta 指标要求 `--branch-scope both`；不可用组合会报错。
指定指标仅影响折线面板，所有逐 step 统计仍会保存。

`plots/trajectory-index.json` 一条记录对应一条轨迹，保存来源、SHA、parallel step数、
指标和图文件名。某条已选轨迹没有 parallel decision 时，仍生成独立的“指标不可用”说明图，
状态为 `no_parallel_steps`，不补造成零熵。整个选择都没有可统计 step 时继续明确报错。

`curves/per-trajectory.csv` 保存实际绘图的**原始值**：source_run_id、rollout_id、instance_id、
metric、scope、axis、x、value。`axis=assistant_step`，x为原始 step；没有 mean 或 task 平均值。

### 单 run 与跨 run 平均曲线

原 `per-run`、`all-runs`、`both` 功能保留，聚合对象为该 run 内显式选择的轨迹。
每条轨迹独立使用 `x=t/(L−1)` 归一化到0–100%，先插值到默认101点网格，
再在每个 run 内按轨迹等权取平均。L=1只在0%有观测，不外推、不跨缺失值插值。

`per-run` 写 `plots/per-run/RUN_ID.*`，`all-runs` 写 `plots/all-runs-normalized.*`；
默认 `both` 同时生成这两种图。跨 run 图只比较声明的来源，不自动纳入其他实验。
`--line-metric` 同样可控制平均图面板；不提供时保留原来的全部可用指标。

run 平均图的实际数值写入 `curves/per-run.csv` 和 `curves/all-runs-normalized.csv`，
其中 axis 为 normalized_progress，x 为0..1，图中显示0..100%；
`curves/raw-steps.csv` 为附加原始 step 聚合审计视图。
这些平均曲线具有 mean、贡献轨迹数、标准差及标准误，与逐轨迹的 value 导出明确区分。

## 5. 产物与校验

产物写 `action/outputs/ANALYSIS_ID/`，必须使用全新 ID：

```text
analysis.json / selection.json / summary.json
source-evidence/ / source-config/ / code/
trajectories.json / trajectories.csv
decisions.jsonl / steps.jsonl / steps.csv / histogram_counts.csv
curves/per-trajectory.csv                         # per-trajectory 模式
plots/per-trajectory/RUN_ID/ROLLOUT_ID.png|svg
plots/trajectory-index.json
curves/per-run.csv / all-runs-normalized.csv / raw-steps.csv   # run 平均模式
plots/per-run/RUN_ID.png|svg / plots/all-runs-normalized.png|svg
plots/action_types_*.png|svg / plots/winner_count_*.png|svg
branch-receipt.json
```

统计 schema v2 不变，曲线约定更新为 `curve_schema_version=4`，冻结 line_mode、line_metrics 和轨迹身份。
`--check` 验证来源、逐步统计及文件 SHA，重新从冻结 steps 计算曲线 CSV；
逐轨迹模式还核验清单对所选轨迹的覆盖和每张图的存在性，不只检查一张代表图。
历史版本（包括已经生成的旧单 task 图）仅保留 receipt 校验兼容，不再提供按 task 选择的新绘图入口。
代码升级后的 current_implementation_matches_snapshot=false 不等于旧分析损坏。

## 6. 当前功能验收

- `consensus32-all-rollout-entropy-v2-20261026`：全量86条 canonical 轨迹、2703个parallel steps，每轨迹独立熵图；其中10条无parallel step，保存独立不可用说明图。
- `consensus32-single-trajectory-panels-v2-20261026`：直接指定一条 rollout，17 steps，默认五指标独立图。
- 两组真实产物均通过来源、数值、逐轨迹覆盖和图表文件校验。
- 相关计算/坐标/同task多轨迹分离/指标选择及既有action可视化回归合计36项测试通过。
