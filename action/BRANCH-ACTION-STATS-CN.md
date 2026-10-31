# 分支 action 统计、熵与轨迹进度曲线

## 2026-10-28：统一批量入口，逐指标独立成图

在同一条 `action-branch-stats` 命令中使用 `--plot-layout separate`，即可为所有选中指标分别生成独立图片；重复 `--line-metric` 选择子集，省略时使用当前 line-mode 的默认指标集合。该选项保留原有指标定义和进度算法，不重跑模型。

```bash
.venvs/analysis/bin/dycon action-branch-stats NEW_ID \
  --source-run verified80-parallel-reasoning-log1p-20261021 \
  --branch-scope all --line-mode per-run --plot-layout separate \
  --progress-points 101 --format png --format svg
.venvs/analysis/bin/dycon action-branch-stats NEW_ID --check
```

单 run 图在 `plots/per-run/RUN_ID/METRIC.png|svg`；逐轨迹图在 `plots/per-trajectory/RUN_ID/ROLLOUT_ID/METRIC.png|svg`；跨 run 图在 `plots/all-runs-normalized/METRIC.png|svg`。同一指标的 scope/run 对照可共用一张图，各种指标彼此分图。`plots/plot-index.json` 记录全部应生成图片，CLI 返回清单路径、figure_count 和 image_file_count，`--check` 校验全部文件覆盖。

separate 模式额外支持 `--line-metric branches`，直接读取既有 `branches_all` 和已记录的 `branches_retained`；其中 all 是生成分支 inventory 数量，retained 才是历史实际保留集合的长度。旧版本分析未启用 `branch_count_curves` 时继续按旧列集合重算，旧多面板产物及校验保持。

**历史 run 口径：** `verified80-parallel-reasoning-log1p-20261021` 是 4 分支 argmin run，没有 consensus retained 集合，也没有动态熵规则；必须使用 `--branch-scope all`。可画真实 action 熵、有效 action 分支数和生成分支数，不能把生成数、有效 action 数或最终执行代表数冒充动态 top-k。动态 H/k/N 指标及 metric-set 预设仍是计划功能，本次没有宣称实现。

不指定 `--plot-layout` 时维持现有 panels 布局，避免改变原命令行为；需要一量一图时显式传 separate。

更新：2026-10-27。正式入口：`.venvs/analysis/bin/dycon action-branch-stats`。
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

### 快速选择绘图模式（2026-10-27 补充）

| `--line-mode` | 每张图包含的数据 | 横轴 | 输出位置 |
|---|---|---|---|
| `per-trajectory` | 一条 rollout 的原始值；每轨迹一图 | 原始 assistant step，从1开始 | `plots/per-trajectory/RUN_ID/ROLLOUT_ID.*` |
| `per-run` | 一个 run 内所选轨迹的等权平均；每 run 一图 | 0–100%进度 | `plots/per-run/RUN_ID.*` |
| `all-runs` | 各来源 run 的轨迹均值在同图比较 | 0–100%进度 | `plots/all-runs-normalized.*` |
| `both`（默认） | 同时生成 per-run 和 all-runs；不自动逐条绘图 | 0–100%进度 | 上述两类平均图 |
| `none` | 不绘折线；仍计算统计、导出曲线数值并生成整数直方图 | 不适用 | 统计与数值导出仍保留 |

逐轨迹默认五个面板；run 平均图默认展示全部可用指标。两者都可重复使用 `--line-metric`
只选部分面板，例如 entropy、normalized_entropy、effective_action_count、winner_uncertainty。
这些字段中的 normalized_entropy 是纵轴指标 H/ln(K)，与横轴的轨迹长度归一化是两回事。

```bash
# 一个 run 的全部默认 eligible/final 轨迹平均；不限定 Submitted。
.venvs/analysis/bin/dycon action-branch-stats NEW_RUN_MEAN_ID \
  --source-run RUN_ID --branch-scope both --line-mode per-run \
  --progress-points 101 --format png --format svg

# 若只平均指定轨迹子集，在上述命令上额外传入 --rollout-id 或 --rollout-list。
# 若只平均 Submitted 轨迹，额外传入 --termination-reason Submitted。
# 只有逐条独立绘制时，才使用 --line-mode per-trajectory。
.venvs/analysis/bin/dycon action-branch-stats NEW_RUN_MEAN_ID --check
```

“全部轨迹”指该来源 run 的默认 eligible/final canonical selection，仍遵守原有发布、
重采样 final 和 alias 规则，不表示无条件扫描所有历史文件。
所有平均都先逐轨迹归一化/插值，再在每个 run、每个进度点按有效轨迹等权平均。
没有 parallel step 或该位置没有有限指标值的轨迹不补零参与平均，
逐点实际分母见 `curves/per-run.csv:n_trajectories`；`total_trajectories` 是选择总数。

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

### 最新单 run 全部轨迹平均示例（2026-10-27）

来源：`parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025`。
已生成并通过 `--check` 的分析 ID 为 `consensus32-all-trajectory-average-20261027`。
选择全部86条轨迹、2703个parallel steps；其中10条无parallel step，不补零，
H_all曲线的每个进度点有76条有效轨迹参与平均。未额外限定Submitted，
因此与早先只选40条Submitted轨迹的图不是同一数据范围。

远程产物：

```text
action/outputs/consensus32-all-trajectory-average-20261027/
  plots/per-run/parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025.png
  plots/per-run/parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025.svg
  curves/per-run.csv
  analysis.json / selection.json / summary.json / branch-receipt.json
```

该示例使用101点、包含0%和100%端点的进度网格；实线为全部分支，虚线为历史estimator保留分支。
`curves/per-run.csv` 中x为0..1，图中显示0..100%，mean是对应进度的轨迹均值。
完整口径和已选轨迹应以本分析自身的冻结metadata与selection为准。

## 7. 同一历史候选池的 A/B/C 离线映射与 HTML（2026-10-28）

`action-branch-stats` 对 consensus 来源默认离线计算全部 A/B/C 映射的保留分支数，
每个量、每种映射独立成图，并将全部图片汇总为内嵌 HTML。
只需传入新分析 ID 和来源 run；默认使用 `per-run` 与 `separate`，不必额外启用开关。
不改变源 run 或实际 winner，不启动模型。

```bash
.venvs/analysis/bin/dycon action-branch-stats NEW_MAPPING_ID \
  --source-run parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025
.venvs/analysis/bin/dycon action-branch-stats NEW_MAPPING_ID --check
```

保留 `--entropy-mapping-comparison` 作为显式启用选项；使用
`--no-entropy-mapping-comparison` 可关闭离线映射与 HTML，恢复原有绘图模式。
不指定布局时，关闭映射后恢复 `both/panels` 默认值；要使用 per-trajectory、none、
all-runs 等既有模式，请同时关闭映射。`--check` 始终读取冻结设置，不允许传入启停开关。
Python `analyze_branches()` 同样自动解析默认值。
非 consensus 来源无法重放共识候选池，默认继续使用原分支统计，不伪造映射；显式要求映射则报错。

一次命令为三种生产映射分别生成 H、保留数量 k、筛选前 winner 支持率、筛选后
winner 支持率、winner 改变率和 score cutoff 六张独立图，共18张映射图。
不再绘制候选池数量 N，数值仍保留用于审计。所选历史实际指标图及整数直方图也汇总至 `index.html`。
winner 基准为 estimator top-k 之前候选池的共识 action：先对全池应用同一生产投票及
平票规则，再对每种动态映射筛选后的集合重放共识。支持率始终统计筛选前 winner 的票数，
分母分别为筛选前/后的分支数；改变率按 action 身份比较，逐 step 为 0/1。
无可确定 winner 时用 null，不补0；run 平均使用原有逐轨迹插值/等权规则。
HTML 内嵌全部图片，可单文件离线打开。PNG/SVG 原图仍分别保存。

候选池按 source run 冻结配置和原始 decision 重放校验后，使用最终状态 eligible、
有限 score、canonical action 可比较的候选；不使用筛选后集合反推熵。若有 loop escape，
使用其定案后的候选池。复用 `rollout.entropy.entropy_retention`，不复制公式或重打分。
每 step 先计算整数 k，按 `(score, branch_id)` 精确保留 k 条，cutoff 为最后一条 score；
再复用既有进度插值及轨迹等权平均。不得先平均 H 再代入非线性映射。

m 和 r₀ 来自各源 run 的冻结共识配置。`N<m` 的生产返回 k=0；N=0 的 H 为 null。
三种映射的 H 和筛选前支持率相同，分别展示用于按规则查阅。均值 k 可为小数。
所有图标注为同一历史池上的离线反事实对照，不声称这些是对应策略真实执行后的轨迹、
效果或 resolved rate。缺失必要原始候选证据会拒绝分析，不把未知 action 补成新类别。

新增产物：

```text
index.html
entropy-mappings/steps.jsonl / steps.csv
entropy-mappings/per-run.csv / raw-steps.csv
entropy-mappings/plot-index.json / gallery-index.json
plots/entropy-mappings/RUN_ID/RETENTION_RULE/METRIC.png|svg
```

`analysis.json` 显式冻结 feature version、规则和 counterfactual 解释；decision record
冻结候选池与 score。`--check` 从 SHA 绑定的原始 decision 重新校验输入，再重算所有
映射数值、曲线、图片清单和内嵌 HTML 内容。旧分析不启用此 feature，仍按旧路径校验。


## 固定比例 winner 对照（mapping v4）

`action-branch-stats` 默认在 A/B/C 三种熵映射中分别新增
`winner_changed_vs_fixed_ratio` 图，并加入内嵌 HTML 与 CSV。
同一个 post-loop-escape 候选池按 `(score, branch_id)` 排序，以固定比例
0.5（至少 m 条）的共识 action 为基准，直接比较各熵映射 winner 的
action 身份，逐 step 不同记 1，相同记 0，任一 winner 缺失记 null。
沿用逐轨迹进度归一化、插值与等权平均。这不是原有两条改变率相减，
也不是重新执行 agent 后的实际效果。旧分析保持冻结版本和原文件。
