# 离线分析手册

从冻结 run 和正式采集产物建立不可变分析。以下三部分覆盖 action 分类与分支统计、剩余工作量估计、共识候选的 steering 数据集与可视化。分析命令使用 `.venvs/analysis/bin/dycon`，具体选项见子命令帮助。

## 详细参考

- [Action 分析](#action-分析) · [正式轨迹分类与共享索引](#正式轨迹分类与共享索引)、[标签和统计单位](#标签和统计单位)、[分支统计、进度曲线和历史候选比较](#分支统计进度曲线和历史候选比较)、[提交审计](#提交审计)、[收尾复查长度](#收尾复查长度)
- [Difficulty 估计](#difficulty-估计) · [估计目标与离线工作流](#估计目标与离线工作流)、[特征身份、标签和任务划分](#特征身份标签和任务划分)、[训练配置与 TensorBoard](#训练配置与-tensorboard)、[Ridge 外层任务留出评估](#ridge-外层任务留出评估)、[外部磁盘训练和已采纳测试集](#外部磁盘训练和已采纳测试集)、[已采纳轨迹的终点标签](#已采纳轨迹的终点标签)、[冻结估计器的漂移评估](#冻结估计器的漂移评估)、[轨迹级预测与真值对照](#轨迹级预测与真值对照)
- [Steering 分析](#steering-分析) · [共识数据集及 LDA、t-SNE、UMAP](#共识数据集及-ldat-sneumap)、[数据集、缓存与可视化产物](#数据集缓存与可视化产物)

## Action 分析

### 正式轨迹分类与共享索引

Action 新产物与 difficulty 一样，独立于任何 source run，统一写入项目级
action/outputs/ANALYSIS_ID/；共享 hidden 索引写入 action/datasets/INDEX_ID/。
无需建立 analysis run，来源 run 只提供轨迹与证据。即使配置来自全局 --run-id，产物位置也不改变。

#### 最小操作

从项目根目录执行，analysis ID 必须全新。分类命令会原子生成 labels、统计和两张默认热力图：

    .venvs/analysis/bin/dycon action-classify ANALYSIS_ID --source-run RUN_A --rollout-id ROLLOUT_A
    .venvs/analysis/bin/dycon action-check ANALYSIS_ID
    .venvs/analysis/bin/dycon artifact verify ANALYSIS_ID --scope action-analysis
    .venvs/analysis/bin/dycon artifact show ANALYSIS_ID --scope action-analysis

同时分析多个 run：重复 --source-run，用 JSON binding 清单明确每条轨迹属于哪个 run。
下列字段名是严格校验的；trajectory_sha256 与 cohort 可省略，实际 SHA 总会冻结：

    [
      {"source_run_id":"RUN_A", "rollout_id":"R1", "trajectory_sha256":"ACTUAL_SHA", "cohort":"baseline"},
      {"source_run_id":"RUN_B", "rollout_id":"R1", "trajectory_sha256":"ACTUAL_SHA", "cohort":"parallel"}
    ]

    .venvs/analysis/bin/dycon action-classify ANALYSIS_ID --source-run RUN_A --source-run RUN_B --rollout-list bindings.json

省略筛选时冻结每个 source run 中全部 eligible 轨迹。单个 run 的 --rollout-list 兼容 JSON ID 列表；
多个 run 的清单必须含 source_run_id，--rollout-id 只接受单个 source run。
相同 run/rollout 重复请求拒绝；相同 bytes、同 task 的副本只计一次并记录 alias；
相同 bytes 绑定不同 task，或 model/cohort/classifier cwd 冲突时拒绝，避免不独立样本被重复统计。

#### 最小分析单元

    action/outputs/ANALYSIS_ID/
      analysis.json / selection.json / resolved-config.json
      source-evidence/RUN_ID.json
      code/...                              # 分类/统计/绘图实现快照
      labels.jsonl / summary.json / transition.json / examples.json
      plots/progress-share.png
      plots/transition-probability.png
      plots/plots.json
      receipt.json
      plots/views/PLOT_ID/...                # 后增独立 view，不改 parent receipt

analysis.json 是本单元完整冻结配置，明确 source run、rollout ID、路径、SHA、分类 cwd、taxonomy、
配置/实现/package 身份和统计口径。labels 的唯一身份是 source_run_id + rollout_id + step_id。
source-evidence 保存选中的 run evidence 子集；后来 harness 更新原 evidence 不会改变旧分析，
但被绑定的 trajectory bytes 改变会使完整校验失败。读取不依赖实时 Catalog、enrichment 或 hidden。

#### 热力图与不可变视图

progress-share 的列是相对进度五阶段，行是现有七类，默认 micro 为该阶段类别数/全部 action 数；
summary 同时保存按轨迹等权的 macro、计数与分母，以及各 source run/cohort 分组矩阵。
transition-probability 的行是当前 action、列是下一 action，按行归一化；不跨轨迹连接，末步没有下一步。
空阶段/空行用 null 和 N/A 表示，颜色范围固定 0–1。所有 assistant message 均是 observation unit，
无 bash、多个 tool call 等形态仍按既有 taxonomy 标 OTHER。

    # 默认图在分类时已产生；此命令只校验并返回它们，重复调用不改 receipt
    .venvs/analysis/bin/dycon action-plot ANALYSIS_ID
    # 新绘图配置使用独立 view ID，支持 macro 与 PNG/SVG
    .venvs/analysis/bin/dycon action-plot ANALYSIS_ID --plot-id macro-v1 --aggregation macro --format png --format svg

绘图只读冻结 labels/summary，不再打开来源 run 或 trajectory。view 自己保存 receipt，绑定 parent receipt SHA；
原分析 receipt 不包含后加 views（与 difficulty dataset/splits 的关系一致）。完整 action-check 会连同 views 校验。
修改参数必须使用新 analysis/view ID，不提供覆盖开关。实现快照被 hash 保护；当前安装代码与快照是否一致单独报告，
正常代码升级不会仅因版本不同就判定历史 receipt 损坏。

#### 历史读取

旧 schema-1 产物保持原位；项目级旧产物按 ID 读取，run 内旧产物必须显式给路径，禁止隐式 fallback：

    .venvs/analysis/bin/dycon action-check runs/OLD_RUN/action/outputs/OLD_ID

不向旧目录补文件、不迁移、不重签 receipt；旧格式需重新分类为新 ID 才能生成新图。

Action 分为两个互补层：action-classify 直接读取 canonical rollout trajectory 并生成互斥 action 类别；action-index 用于复用 difficulty hidden 样本的共享索引。两者都不重复提取 hidden，也不执行 trajectory 中的 command。

#### 轨迹 action 分类

完整规则和产物见 [标签和统计单位](#标签和统计单位)。

    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN --rollout-id ROLLOUT_ID
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN --rollout-list ROLLOUTS.json
    .venvs/analysis/bin/dycon action-check ACTION_ANALYSIS_ID

该入口按显式 source run 和 rollout 清单选择 canonical 轨迹，验证 trajectory SHA，直接解析 assistant/tool-call 记录。它不依赖 enrichment、token trace、hidden 或 GPU；已有统一入口时不得另建临时读取/分类脚本。输出在项目级 action/outputs/ACTION_ANALYSIS_ID/，同一 ID 不覆盖。

#### Hidden 共享索引

    .venvs/analysis/bin/dycon action-index SHARED_INDEX_ID ACTION_INDEX_ID

core.cli 调用 action.datasets.index.create_action_index，读取 difficulty/datasets/SHARED_INDEX_ID/index.json，记录源索引 SHA，写入 action/datasets/ACTION_INDEX_ID/index.json。它不是 trajectory action 分类的前置步骤。

action 复用已有 hidden 与 task 分区，不重复提取 latent，不向当前 prefix 泄漏未来动作。当前 taxonomy 只分析可观察 shell/tool 行为；action 训练、taxonomy 学习和动作评测尚未实现。完整任务链见 [运行手册](MINIMUM-RUN-CN.md#进度状态字段与重试投影)。

### 标签和统计单位

#### 统计单位与唯一标签

一个 assistant message 是一个 action observation unit。历史数据中一个 assistant message 通常对应一次 bash tool call；多个 tool call、非 bash tool 或 malformed arguments 统一标为 OTHER，并写入 uncertainty。一个 bash 调用内部可以有多个 shell command，但只产生一个 label；同时保留 components、rules 和 compound。

主类别固定优先级：

    DELIVER > EDIT > ENV > RUN > CLEAN > INSPECT > OTHER

| 类别 | 目的 | 可观察判别 |
|---|---|---|
| INSPECT | 获取信息而不改变持久状态 | 读文件、搜索、查看 diff、git status/log/show、版本查询 |
| EDIT | 修改代码、测试、配置、补丁或工作树 | 写重定向/heredoc、Python 写文件、sed -i、patch、cp/mv/touch/mkdir、git restore/checkout |
| ENV | 准备依赖或运行环境 | pip/uv/apt/conda 安装或升级、权限和工具链准备 |
| RUN | 执行程序或验证 | Python、pytest、pylint、mypy、ruff check、诊断/构建执行 |
| CLEAN | 删除临时产物或清理状态 | rm/rmdir/unlink、git clean |
| DELIVER | 形成或提交最终结果 | submission marker、git diff 重定向到 patch、git add/commit/format-patch |
| OTHER | 无法安全静态判定 | shell AST 错误、工具形态未知、动态副作用未支持 |

RUN 有意合并复现和验证：历史 rollout 中二者会使用相同的 Python/pytest 命令，命令本身不足以互斥区分它们。网络请求、find -delete/-exec 等潜在副作用也进入 OTHER，而不是误归为只读或修改。

复合命令仍只有一个主类别。例如写入脚本后运行脚本归 EDIT；安装依赖后测试归 ENV；删除临时文件后测试归 RUN；marker 加 patch 输出归 DELIVER。具体子命令保留在 rules。

#### 阶段定义

对每条 trajectory 的 step s、总 step 数 T 使用：

    phase_bin = min(4, floor(5 * s / max(1, T - 1)))

这是事后位置描述，不是在线分类输入，也不能表示模型当时知道未来。0 为前段，4 为末段。

#### 产物

action/outputs/ACTION_ANALYSIS_ID/包含：

- analysis.json：source runs、轨迹选择、taxonomy、有效配置及实现身份。
- selection.json / source-evidence/：run evidence revision/SHA、冻结子集、trajectory bindings、duplicate aliases。
- resolved-config.json：本次分析使用的配置快照。
- labels.jsonl：每个 assistant step 一行，唯一主标签。
- summary.json：类别、阶段、cohort、compound 和 uncertainty 统计。
- examples.json：各类代表命令，供人工抽查。
- code/ / receipt.json：源码快照、配置、输入选择和产物 SHA；action-check 重新验证全部内容。
- transition.json / plots/：阶段占比与相邻 action 转移矩阵、默认图及独立 view receipt，详见 README。

同一 canonical trajectory 被多个 rollout 副本引用时，selection 会保留一个 source 并记录 aliases；不同 task 绑定相同 bytes 会直接拒绝。action 分析不改变 Catalog 或来源 run，也不修改已有 action-index 样本视图。

### 分支统计、进度曲线和历史候选比较

#### 批量绘图与逐指标独立输出

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

#### 轨迹选择与运行

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

##### 快速选择绘图模式

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

#### 每个 step 的符号与计算

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

#### 直方图：问题（1）

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

#### 折线绘图

##### 每条轨迹独立一张图

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

##### 单 run 与跨 run 平均曲线

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

#### 产物与校验

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


#### 同一历史候选池的 A/B/C 离线映射与 HTML

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


#### 固定比例 winner 对照（mapping v4）

`action-branch-stats` 默认在 A/B/C 三种熵映射中分别新增
`winner_changed_vs_fixed_ratio` 图，并加入内嵌 HTML 与 CSV。
同一个 post-loop-escape 候选池按 `(score, branch_id)` 排序，以固定比例
0.5（至少 m 条）的共识 action 为基准，直接比较各熵映射 winner 的
action 身份，逐 step 不同记 1，相同记 0，任一 winner 缺失记 null。
沿用逐轨迹进度归一化、插值与等权平均。这不是原有两条改变率相减，
也不是重新执行 agent 后的实际效果。旧分析保持冻结版本和原文件。

### 提交审计

#### 运行入口与产物

实现为 `action/analyzers/submissions.py`，通过统一 CLI 调用；
共享 `action/datasets/trajectories.py` 的 SHA 绑定 canonical reader，不另实现轨迹加载器。
纯 CPU/只读来源数据，不请求 LLM、不启动 rollout/harness/GPU，不执行轨迹中的命令。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/analysis/bin/dycon action-submission-audit --list
.venvs/analysis/bin/dycon action-submission-audit submitted-patch-80tasks-gt30-final-20261026 \
  --task-count 80 --min-submitted 30
.venvs/analysis/bin/dycon action-submission-audit submitted-patch-80tasks-gt30-final-20261026 --check
```

输出为 `action/outputs/ANALYSIS_ID/`，同 ID 拒绝覆盖：

- `scope.json`：全部 17 个候选及排除理由；`audit.json`：范围、时间、全部输入文件 SHA 和策略。
- `summary.json` / `REPORT.md`：汇总、分 run 分母与未知样本。
- `trajectories.jsonl`：302 条逐轨迹 patch、最新/全部历史评测、原因和来源路径；CSV 为轻量表。
- `inputs/`：冻结 plan/tasks/evidence/评测报告；`code/`：实现快照；`audit-receipt.json`：输出哈希。
- `review-samples.jsonl`：8 条真实 canonical 轨迹的逐 step 抽样阅读材料，覆盖 5 个 run、
  有效/空/非空无效提交和137步长轨迹；不是随机代表性样本，也不是已验证的复查标签。

本审计使用专属 `--check`，不是 action 七分类分析，不传给 `action-check` 或 `artifact --scope action-analysis`。
`--check` 验证产物与输入 SHA、逐轨迹唯一性及汇总重算；来源 run 后续更新会报告快照已过期，
应使用新 analysis ID 重跑，不修改旧报告。

### 收尾复查长度

#### 实现位置

| 文件 | 职责 |
|---|---|
| action/analyzers/review_effects.py | shell/Python静态写入意图、helper内容与生命周期 |
| action/analyzers/review_phases.py | 候选解状态、两通道证据、复查episode和unknown |
| action/analyzers/review_length.py | 父产物验证、canonical reader、原子发布与专属校验 |
| action/evaluation/review_statistics.py | 总体、分run、分评测组的分母与分位数 |
| tests/test_action_review_length.py | 特征、状态、提交、产物、防覆盖回归测试 |
| core/cli.py | action-review-length统一入口 |

复用原有action分类结果与action.datasets.trajectories.read_messages，不另建canonical读取器。
不执行轨迹命令、不调用LLM、不启动rollout/harness/GPU、不重新评测patch。

#### 范围与可复现入口

沿用此前确认的同一组80题、submitted严格大于30的5个run。观测单位是选中canonical
轨迹中的assistant message，不累加未选parallel分支；已登记submitted attempt全部保留。
重评run只提供评测证据，不重复计算轨迹。

输入固定为：

- 提交审计：submitted-patch-80tasks-gt30-final-20261026，302条submitted。
- action父分析：review-length-submitted302-actions-20261026，302条、14,235个assistant step。
- 最终结果：review-length-submitted302-final-20261026。

创建分析与检查：

    cd /home/wangtianyi/DyConGlobal
    .venvs/analysis/bin/dycon action-review-length NEW_ANALYSIS_ID \
      --audit-id submitted-patch-80tasks-gt30-final-20261026 \
      --action-id review-length-submitted302-actions-20261026
    .venvs/analysis/bin/dycon action-review-length NEW_ANALYSIS_ID --check

新ID拒绝覆盖；父审计与父action必须覆盖完全相同的轨迹集合，不允许缺条、额外轨迹或alias。
父action分类通过原有action-classify和302条显式binding完成。重现已有复查分析不依赖
探索阶段的temp/extra30导出文件。

复查分析有专属--check，不传给仅接受七分类产物的action-check。
父分析仍可使用action-check review-length-submitted302-actions-20261026校验。

#### 当前规则

方法版本：review-episodes-v1。规则与代码随分析冻结。

1. 优先从最终patch获得交付路径；无标准diff时，从轨迹中明确的工作区git diff回退。
   git show和跨commit比较不能充当工作区变更证据。
2. 追踪直接写入、字面量Python路径、轨迹生成后执行的helper，以及restore/checkout/stash等状态变化。
   /tmp中的helper若实际修改目标实现，仍重置候选解；文件名包含test不能单独证明属于验证。
3. 临时脚本/fixture综合创建、内容、非交付对象、调用和后续清理判定。
   最终交付路径优先于临时文件启发式。已修复清理root helper、repro*.py时误重置的问题。
4. 形成候选解后，操作、目标对象、核查语义中至少两个通道支持时进入复查。
   不把首次check关键词或最后EDIT后的整个后缀直接当复查。
5. VERIFY是核查行为，VERIFY_SETUP是为核查创建/修正临时验证、调整测试运行方式或环境。
   再次修改交付内容或恢复候选状态会结束当前episode，之后重新识别。
6. PACKAGE、CLEANUP、SUBMIT_RETRY独立标注；根据canonical exit/实际最终提交确定终点。
   修改+验证混合step作为候选状态变化，不计严格复查。重复操作按实际开销保留。
7. 未检测到候选变化、末段修改无法解释或终止无法确认时，主长度保留null。
   明确没有复查、后续只有打包/提交可为有效0；unknown绝不补0。

主指标review_steps为最终episode内VERIFY/VERIFY_SETUP的assistant步数。
review_span从最终复查起点到实际终止提交之前，包含穿插的打包、清理和失败提交重试。
同时输出宽松敏感性口径review_steps_permissive、所有episode核查步数、返工次数、
候选状态稳定位置、复查起点、逐步证据、unknown原因和警告。

strict/permissive不是统计置信区间，也不是被证明包含真值的上下界。本版统计step，
没有用字符数或累计prompt token冒充复查token长度。

静态分析无法完整模拟任意shell/Python：动态执行可能隐藏副作用，成功returncode不证明
sed真的改了内容。缺少逐step文件hash时，不能可靠区分所有no-op和实际内容变化。
因此结果明确标为规则估计，尚非独立人工标注验证的心理阶段真值。


#### 产物与验证

目录：action/outputs/review-length-submitted302-final-20261026/。

- analysis.json：规则设置、父receipt、代码SHA、依赖版本、时间、口径。
- selection.json：逐轨迹身份、trajectory/patch路径和SHA。
- trajectories.jsonl与CSV：逐轨迹指标、episode、unknown和评测组。
- steps.jsonl：14,235条逐步特征、阶段、原message_index与证据。
  reasoning/observation保存片段、原字符数与hash；判定使用完整canonical内容。
- summary.json与REPORT.md：总体、各run、评测组、分位数、分母和警告子集。
- code/与receipt.json：实现快照与不可变输出清单。

专属--check验证输出hash、父receipt及完整性、身份唯一性、步数覆盖、
逐step标签与净步数一致、汇总重算、canonical trajectory/patch内容。

## Difficulty 估计

### 估计目标与离线工作流

Parallel capture 的 visualization-only publication 可由
`difficulty-plot-trajectories --rollout-id ... --publication-id ... --attempt-id ...`
和 `difficulty-plot-parallel --rollout-id ... --publication-id ...` 直接读取。
不需要训练 dataset，且这些 publication 不具有训练资格；轨迹预测仍需已有兼容
checkpoint。详细 coverage 和 truth 语义见 `docs/CAPTURE-CN.md` 与
[轨迹级预测与真值对照](#轨迹级预测与真值对照)。

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
参数、分组、汇总与产物见 [使用说明](MINIMUM-RUN-CN.md#进度状态字段与重试投影)。

并行解码的在线评分可使用 `dycon difficulty-plot-parallel RUN_ID PLOT_ID --rollout-id ROLLOUT_ID`
绘制实际选中轨迹的逐步预测折线及全部候选分支散点，并导出 PNG/SVG、评分 CSV 和来源 receipt。
无需加载 checkpoint、hidden tensor 或启动 GPU；具体选择、单位、缺失评分与噪音解释见
[并行轨迹评分可视化](MINIMUM-RUN-CN.md#进度状态字段与重试投影)。

`dycon difficulty-ridge EXPERIMENT_ID` 在 analysis 环境消费冻结的 submission 选择，执行单层 Ridge × alpha 搜索、验证 R²选优及参数/可视化报告保存。支持原 task 两划分，以及 event/task 两种三划分：`event_random_v1` 允许同 task 的 step 跨分区，`task_random_v1` 将同 task 的全部轨迹和 step 放在同一分区；三划分均在 train＋validation 重拟合冠军后评估 test。这是“到该次提交”的独立标签视图，允许已提交但未成功或未评测的轨迹，原成功标签不改。完整协议见 [实验指南](MINIMUM-RUN-CN.md#进度状态字段与重试投影)。

2026-10-19 按实际代码更新。正式命令见 [任务入口表](MINIMUM-RUN-CN.md#进度状态字段与重试投影)。已有索引、评分和 hidden 提取入口时直接复用，不另写临时脚本拼接数据或重定义标签。

#### Ridge 实验与可视化

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

#### 正式数据与评分命令

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

#### 模型与目标

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

`transform_target`/`inverse_target` 提供变换与逆变换。`train_estimators` 返回模型对象、transformed-space MAE/RMSE/R²、逆变换后的 validation predictions/targets；不自动输出 scorer 所需的 sample_id/target 预测清单或保存 checkpoint。预测格式见 [INPUTS.md](MINIMUM-RUN-CN.md#进度状态字段与重试投影)。

#### Python API 和划分限制

`difficulty.training.train.load_training_data(cfg, index_id, target, split)` 排除缺失标签，返回 `(x, y_transformed, sample_ids)`。`train_estimators(cfg, index_id, target, kinds=..., **kwargs)` 读取 train/val、拟合并返回结果；kwargs 以 estimator 名为键，各参数见构造函数。

- schema 3 按 task partition 选择 train/val；未 resolved 或 censored 的标签在索引阶段置 null。2026-10-19 观察的 task 全为 train，不能将无 val 的现状称作完成验证。
- schema 1 按排序 rollout ID 约取 80%/20%；同 rollout 不跨组，但同 task 多条 rollout 可能跨组。该分支仅区分 train/val/其余全部，没有独立 test split。
- 正式 MLP/Ridge+MLP attempt 支持 `device: auto|cpu|cuda[:N]`，在 analysis 环境使用 PyTorch GPU；Ridge 基础模型仍为 NumPy 数值 checkpoint，训练期间保持冻结。

**没有旁路 `dycon difficulty-train`。** MLP 使用正式 `difficulty-mlp`，Ridge+MLP 使用正式 `difficulty-ridge-mlp`，通过统一 attempt/receipt/verifier 生成离线 checkpoint 和评测。不能以临时训练脚本绕行。`difficulty.estimator=null` 不自动选择模型；动态在线难度控制未实现。

已有数值契约测试 `tests/test_difficulty_estimators.py`；小型合成测试不等于实际数据训练或跨任务泛化验证。

#### 来源运行与轨迹选择

新建 online index / resolved dataset 必须先确定来源 run，再以 `--rollout-id` 或 `--rollout-list` 筛选该 run 内轨迹；不再查询 Catalog 的 rollout 列表。共享实现是 `difficulty.datasets.selection.select_run(root, source_run, rollout_ids, rollout_list)`。跨 run 数据通过独立冻结数据集后 compose。已有 receipt 绑定的数据集继续独立验证，不依赖实时 Catalog 或其他 run 的评测状态。索引用法见[采集与发布手册](CAPTURE-CN.md)。


#### Trajectory fraction labels

Create a new resolved dataset with `--label-normalization trajectory_fraction` to derive causal, per-trajectory labels without rewriting raw labels. For a trajectory with `T` assistant steps and current close step `s`, the step target is `(T-s)/T`; the reasoning target is `remaining_reasoning_tokens_after / sum(finalized_reasoning_block_lengths)`. Fraction targets are `submission_remaining_steps_fraction` and `submission_remaining_reasoning_tokens_fraction`; train them through the existing Ridge, MLP, or Ridge+MLP commands. Existing datasets and checkpoints remain immutable.


#### Configurable training and TensorBoard

Edit standalone `difficulty/training/config.yaml`, then run `source difficulty/scripts/train.sh` (or pass a YAML path). The source wrapper delegates to `dycon difficulty-train` in the analysis environment and preserves the caller shell via a subshell. Schema 2 selects dataset/split/targets, Ridge, MLP or Ridge+MLP, input site, MLP hidden dimensions/activation/normalization/dropout, optimizer, scheduler and device. Ridge remains closed-form; Ridge+MLP requires one verified published Ridge manifest per target. Every run creates a new immutable attempt.

TensorBoard binds to remote loopback port 6006 by default. On the local PC run `ssh -N -L 16006:127.0.0.1:6006 wangtianyi@10.249.187.9`, then open `http://localhost:16006`. MLP events expose evaluation-mode train/validation MSE per epoch and separate optimization loss; Ridge events use candidate-trial steps. Events are included in the attempt receipt. Check or stop the managed service with `dycon difficulty-tensorboard ATTEMPT_ID --action status|stop`.


#### 重复 rollout 预测与真实剩余量对照

使用 `dycon difficulty-plot-trajectories RUN_ID PLOT_ID --attempt-id RIDGE_ATTEMPT --dataset-id RESOLVED_DATASET --target submission_remaining_reasoning_tokens_log1p`。
按 canonical repetition 统计成功次数；每条成功轨迹绘制预测与 native 真值双面板，同 task 同行汇总到自包含 HTML。缺失证据保留断线及说明图。
完整口径与统一 artifact 校验见 [轨迹对照说明](#轨迹级预测与真值对照)。


##### 预测与真值独立覆盖

在 `difficulty-plot-trajectories` 中显式添加 `--repair-id REPAIR_ID`，即可对全部可验证的 repaired canonical close hidden 预测，独立于真值是否可用。
预测、真值分别统计覆盖数；canonical-identity-only 预测点单列，不冒充 native token 标签。输出 `coverage-audit.json`，旧训练数据集和旧图保持不可变。

5120 维 capture 的 hidden/site 契约、task manifest 划分和 Ridge refit 边界见[Difficulty 特征与训练契约](#特征身份标签和任务划分)。


#### Positive-reward terminal trajectory selection

For SkillsBench workspace evaluations, `capture-trace RUN PUBLICATION --eligibility positive_reward --session-id SESSION` and `difficulty-dataset resolved DATASET --source-run RUN --index-id INDEX --rollout-list FILE --eligibility positive_reward` explicitly include completed official evaluations whose numeric `rewards.reward` is finite and greater than zero. Defaults remain `resolved_only`. Use the normal `online-index` between these commands; its success-only labels stay null for unresolved trajectories. Positive-reward selection consumes the existing exact finalized suffix labels (`remaining_steps` and `remaining_reasoning_tokens_after`) and preserves resolved=false for partial credit. Targets describe the actual terminal endpoint, not eventual success. Canonical AgentCompleted, complete native capture, exact evaluation subject/report hashes and canonical action alignment are still required. A post-agent export failure can be included only if those preserved artifacts verify. The dataset receipt and loader recheck the positive evaluation report. No original run evidence or success labels are rewritten.

Use the existing `difficulty-split --strategy event_random_v1 --seed 42 --validation-fraction 0.15 --test-fraction 0.15` and schema-2 Ridge training YAML. Validation selects site and alpha; refit train+validation leaves the test set held out.


#### 自动 capture publication 与 partial 数据

非并行 think_close run 收尾会独立审计并正式发布最终 accepted/submitted canonical sessions。
Workspace/live-state 类型复用其正式 subject 资格，不要求不适用的 patch submission。
通过 `dycon capture status RUN_ID` 获取稳定 index_id/publication_id 和 `index_command`，
再用 `dycon online-index RUN_ID INDEX_ID --publication-id PUBLICATION_ID` 建立 schema-3 index。
该 index 绑定 publication receipt，并保留 coverage locator/hash、缺 hidden 数量、exact
alignment、non-action、censored 与 evaluation-ineligible 区分；不合成缺失 hidden。

`difficulty-dataset resolved ... --source-run RUN_ID --index-id INDEX_ID` 仍只吸收完成评测、
resolved=true、exact think_close、action response 且双标签有效的点。Partial trajectory
贡献其有效子集；accepted/unresolved 留在 publication 中供显式终点可视化使用，默认不
进入 resolved-only dataset。缺 native response 的前缀不获得伪造的剩余 token 标签。
轨迹可视化通过 repair receipt 保留有证据点，预测和 truth 各自缺失处断线，不插值或补零。
完整与 partial 自动 publication 均使用新 immutable ID；历史 index/dataset 不回写。

### 特征身份、标签和任务划分

#### Hidden 特征身份

Ridge loader 从冻结数据集的 `storage_shapes` 推导 hidden width 与 native site。一个数据集中的 rollout 必须满足同一特征契约；宽度、site 布局或模型 fingerprint 不一致时拒绝混合。当前 Qwen3.6-27B capture 使用 32 个 native sites（0–31）、每 site width 5120。旧 4096 维格式使用 sites 0–30 与 32；仅在数据集明确属于该旧格式时按其冻结映射读取。

每个样本绑定 canonical rollout、trajectory SHA、close anchor、native site、storage shape 和源文件 SHA。loader 校验 receipt、tensor shape、finite 值及模型 fingerprint。Cache 按 site 写入 float32 矩阵，其身份包含 dataset receipt、样本顺序、width 和 native site contract；改变来源或特征布局需新建 dataset/cache，不能沿用旧 checkpoint。

#### 评测资格与标签终点

`difficulty-dataset resolved` 默认采用 `resolved_only`：样本须绑定 completed evaluation、被选 patch 的精确 subject 与 `resolved=true`。Submission 不是成功标签。

Workspace evaluation 可显式使用 `--eligibility positive_reward` 纳入 completed 官方评测中 finite 且大于 0 的 numeric reward。此模式仍校验 evaluation report、subject、attempt 和 reward；它不把部分得分改写为 `resolved=true`。resolved-only 标签从 close 到成功提交；positive-reward 标签从 close 到该轨迹实际结束。两种终点不能混作同一目标。

标签来自已 finalize 的 native exact suffix；未评测、无效、截尾或缺失的值不补造。positive-reward 数据集在 receipt 中冻结资格策略，loader 会重新核验绑定的官方评测报告。

#### Task manifest 划分

当 catalog partitions 必须约束 task holdout 时，使用 `task_manifest_v1`：

```bash
.venvs/analysis/bin/dycon difficulty-split DATASET_ID SPLIT_ID \
  --strategy task_manifest_v1 --seed 42 --validation-fraction 0.2 \
  --task-partitions catalog/partitions/SPLIT_NAME/split.json
```

Manifest 中每项使用 `task_key`、`instance_id` 和 `train|val|test` partition。样本 task 必须在 manifest 中且身份一致；同 task 的全部 rollout、repetition 和 close samples 进入同一 split。`val` 映射到 validation。若 manifest 只有 train/test，validation 从 train tasks 按 seed 和比例确定；test 保持 holdout。无样本的分区、缺任务、冲突 task identity 或相同 trajectory 跨 split 均拒绝。未出现在 dataset 中的 test task 记为 `external_test_tasks`。

Split 是 dataset 下的独立不可变产物，不改写 dataset receipt。训练配置引用已冻结 split ID 与 receipt；Catalog partition 不可直接冒充训练 split。

#### Ridge 训练

Standalone schema-2 YAML 绑定 dataset ID/receipt、split、targets、public site、Ridge layers、alphas 和训练模式：

```bash
.venvs/analysis/bin/dycon difficulty-train \
  --training-config difficulty/training/experiments/EXPERIMENT.yaml
```

配置以 strict schema 校验。Ridge 标准化与拟合只使用 train；validation 选择 layer/alpha。`refit_train_validation: true` 时选定模型在 train+validation 重拟合，test 仍只用于最终评估。Attempt receipt 冻结配置、dataset/split 身份、特征契约、模型参数和产物 SHA。更改 dataset、split、native site、width 或 target 定义时使用新 attempt。

一般训练流程和其他模型入口见 [difficulty README](#估计目标与离线工作流)；`dycon difficulty-split --help` 与 `dycon difficulty-train --help` 给出当前 CLI 参数。

### 训练配置与 TensorBoard

编辑独立训练配置 `difficulty/training/config.yaml`，然后在远程工程目录中执行：

```bash
cd /home/wangtianyi/DyConGlobal
source difficulty/scripts/train.sh
# 也可以指定另一份训练配置
source difficulty/scripts/train.sh /absolute/path/to/training.yaml
```

启动脚本使用 analysis 环境，调用正式入口 `dycon difficulty-train --training-config YAML`。脚本在子 shell 中运行，保留调用者的工作目录、shell 选项、环境变量和信号处理设置（trap）。训练在前台运行，按 `Ctrl+C` 可以取消。

#### 数据与模型配置

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

#### 运行模式与产物

`run.attempt_id: auto` 会为每次运行生成新的、不可覆盖的训练尝试 ID。如果手动指定 ID，其输出目录必须尚不存在，包括未完成的尝试目录。

相对配置路径以工程根目录为基准解析。`run.mode` 支持 `dry-run`、`check-data` 和 `fit`：前两者分别检查配置与数据来源、进一步检查特征数据，均不执行拟合；`fit` 执行训练或拟合，并在启用 TensorBoard 时自动启动可视化服务。

产物保存在 `difficulty/checkpoints/ATTEMPT_ID/`，其中 `ATTEMPT_ID` 是启动输出中的实际训练尝试 ID。

#### 在本地 PC 访问 TensorBoard

TensorBoard 默认监听远程服务器的 `127.0.0.1:6006`。在本地 PC 的 PowerShell 中执行：

```powershell
ssh -N -L 16006:127.0.0.1:6006 wangtianyi@10.249.187.9
```

保持该 SSH 连接运行，在本地浏览器打开 `http://localhost:16006`。`16006` 是本地端口，`6006` 是远程端口；如果修改训练 YAML 中的 TensorBoard 端口，需要同步修改 SSH 命令中的远程端口。本地端口被占用时，可以改用其他本地端口，并同步修改浏览器地址。

MLP 按 epoch 记录 evaluation 模式下的训练集损失 `loss/train` 和验证集损失 `loss/val`；训练模式下的 minibatch 优化损失单独记录为 `loss/train_optimization`。Ridge 没有 epoch，其曲线横轴为候选参数组合的试验编号。

事件文件保存在本次 attempt 目录中，写入器关闭后才完成产物校验凭证（receipt）的封存。TensorBoard 的可变进程日志保存在 `state/difficulty-training/`。

#### 查看和停止服务

在工程目录执行以下命令，将 `ATTEMPT_ID` 替换为实际训练尝试 ID：

```bash
.venvs/analysis/bin/dycon difficulty-tensorboard ATTEMPT_ID --action status
.venvs/analysis/bin/dycon difficulty-tensorboard ATTEMPT_ID --action stop
```

如果 `tensorboard.keep_after_training` 为 `true`，训练成功后会保留 TensorBoard，方便继续查看曲线；训练失败时会清理本次启动的服务。上述命令只操作已登记、进程启动身份与 TensorBoard 命令行均匹配的进程。

#### 兼容性与使用边界

兼容适配层仍接受旧版 MLP YAML。新的可配置模型 checkpoint 使用独立的架构标识；旧版 v1 checkpoint 按原来的两个隐藏层结构加载。

本入口生成离线训练 checkpoint，不自动将模型发布为在线 predictor。

#### 完整配置示例：使用工程内已有的数据集与划分

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

#### 各字段可选值、示例与约束

以下说明面向新的独立训练配置 schema 2。默认值指当前代码在**省略字段时**的值，不一定等于现有 `config.yaml` 中显式写入的值。所有层级都拒绝未知字段；YAML 中布尔值用 `true`/`false`，空值用 `null`，列表使用 YAML 列表格式。

##### 运行、数据与特征

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

##### 模型结构与 Ridge 参数

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

##### 训练、优化器、早停与调度器

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

##### TensorBoard

| 完整字段名 | 示例或可选值 | 默认值与约束 |
|---|---|---|
| `tensorboard.enabled` | `true`、`false` | 默认 `true`；关闭时不启动服务、不写 TensorBoard 事件，原训练记录仍保留 |
| `tensorboard.host` | `127.0.0.1`、`0.0.0.0` | 默认 `127.0.0.1`；前者通过 SSH 隧道访问，后者监听所有 IPv4 接口；能否直连取决于网络与防火墙 |
| `tensorboard.port` | `6006`、`6007` | 默认 `6006`；整数 `1024–65535`，必须未被占用 |
| `tensorboard.flush_secs` | `5` | 默认 `5`；整数 `1–60`，writer 的自动刷新间隔；当前实现还会在每个 epoch/trial 后主动 flush |
| `tensorboard.reload_interval` | `5` | 默认 `5`；整数 `1–60`，TensorBoard 重载数据的间隔，单位秒 |
| `tensorboard.keep_after_training` | `true`、`false` | 默认 `true`；成功后是否保留本次服务，失败或中断仍执行清理 |

#### 常见配置切换示例

以下是**替换片段**，不是可以独立运行的完整文件。请在完整配置中替换同名块，避免重复键。

##### 改为按 task 重新划分

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

##### 改为三层 MLP

```yaml
model:
  kind: ffn_mlp
  mlp:
    hidden_dims: [512, 256, 128]
    activation: silu
    normalization: layernorm
    dropout: 0.1
```

##### 改用 Adam 或 SGD

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

##### 改用 Ridge 网格搜索

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

##### 指定 Ridge+MLP 的基础模型

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

### Ridge 外层任务留出评估

`difficulty-ridge-holdout` 对一个已发布、具备训练资格的 difficulty dataset 完成来源校验、两层划分、Ridge 网格搜索、冠军重拟合、全量外层 test 预测与 NMSE/R² 统计，并可为全部 test 轨迹生成对照图和自包含 HTML。正式执行使用 analysis 环境，不启动 rollout、GPU 推理或 hidden 重放。

#### 入口与配置

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

#### 两层划分与拟合边界

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

首版拒绝 WebArena：现有实现要求按 intent_template_id 分组以避免 template 泄漏，当前协议尚未定义其与内层 event 划分的兼容方式。详见[特征与划分契约](#特征身份标签和任务划分)。

#### 显式外层 task manifest

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

#### 指标、覆盖与图表

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

#### 产物与恢复

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

### 外部磁盘训练和已采纳测试集

`paths.artifact_roots.difficulty` 指定 difficulty dataset、split、checkpoint、
evaluation 的物理存储根；例如在 analysis CLI 前使用
`--set paths.artifact_roots.difficulty=/data2/wangtianyi/DyConGlobal`。
`paths.artifact_roots.difficulty_cache` 单独指定大体积特征缓存根。
正式 holdout 服务在调用期间使用该存储作用域，所有 source/run 路径仍按原项目
owned_path 规则核验；没有放宽全局路径边界，也没有改写原 run 的冻结配置。

默认行为继续使用项目原路径。相同 ID 同时出现在配置存储根和原根时拒绝歧义。
dataset-only 的构建器仍支持既有 `--storage-root`。

对已拟合模型进行固定 task 留出评估：

```bash
PYTHONPATH=. .venvs/analysis/bin/dycon \
  --set paths.artifact_roots.difficulty=/data2/wangtianyi/DyConGlobal \
  --set paths.artifact_roots.difficulty_cache=/data2/wangtianyi/DyConGlobal \
  difficulty-ridge-test NEW_EVALUATION \
  --attempt-id FIT_ATTEMPT --dataset-id ACCEPTED_DATASET \
  --task-partitions ALL_NOMINAL_TASKS.json
```

输入清单必须包含显式 train/test 身份；评估器校验原 checkpoint 的 task partition，
只对 test tasks 预测，任何 test task 与实际 fit task 重叠时立即报错。
模型不重新拟合，也不根据 test 结果重选 layer/alpha。
输出只计算 log1p 目标的拟合空间指标，分别报告 accepted 和其 resolved 子集。
每条轨迹先计算 R²，再取不加权平均、中位数和指定分位数；另保留 pooled R²。
不可定义的轨迹 R² 保留 null，并单列数量。

每次正式调用使用新的 evaluation ID；产物含逐点预测、逐轨迹指标、隔离审计
及绑定数据集和模型 SHA 的 receipt。可视化没有启用。

### 已采纳轨迹的终点标签

`dycon difficulty-dataset resolved DATASET --source-run RUN --index-id INDEX
--rollout-list ACCEPTED.json --eligibility accepted_terminal` freezes exact
action-close samples for explicitly selected, completed workspace evaluations,
including resolved=false and reward=0. The caller supplies the final accepted
canonical selection, for example from the receipt-bound capture publication.
An explicit rollout selection is required. Subject, report, canonical
AgentCompleted, native alignment and uncensored suffix checks remain enabled.

Labels describe the observed agent endpoint. Unresolved trajectories do not
receive success remainder labels. Defaults remain resolved_only;
positive_reward continues to require strictly positive official reward.

Configure `paths.artifact_roots.difficulty_dataset` as an absolute storage root
on a data disk. Both `online-index` and `difficulty-dataset` write under
`ROOT/difficulty/datasets/ID`; the selection loader uses the same resolver.
If an ID exists in both the configured root and legacy project root, resolution
rejects ambiguity. Global owned_path rules are unchanged. Model/evaluation
output storage is unchanged; this option only controls datasets and indices.

The source hidden tensors remain in their published NAS shards; dataset
construction does not make an extra copy of them. Put TMPDIR and logs on the
data disk too. Use the same storage-root configuration when consuming the new
dataset. Other training entrypoints with legacy direct paths may need their own
storage integration before use; this construction does not refit estimators.

### 冻结估计器的漂移评估

正式入口：

```bash
PYTHONPATH=. .venvs/analysis/bin/dycon difficulty-evaluate-drift EVALUATION_ID \
  --workflow-config WORKFLOW.yaml
```

该入口读取原 rollout 的冻结 `parallel.estimator_ref/estimator_sha256`，校验
导出 manifest、payload 和 predictor 实现哈希，并复用原 loader/preprocessing。
不选择 layer/alpha、不重拟合、不更新 scaler、不请求模型或补算 hidden。

workflow schema 1 必须包含 `estimator_run`、`allow_hidden_model_shift`、`groups`
和 `skipped_groups`。每个 group 包含 `name`、`dataset_id`、`sources`；每个
source 显式绑定 `run_id`、`publication_id`、`publication_receipt_sha256`、
`index_id` 和唯一 `dataset_id`。跨 checkpoint hidden 必须显式声明
`allow_hidden_model_shift: true`，报告同时保存 predictor 与数据真实模型指纹；
在线 serving 的同模型指纹限制不改变。各组内部不允许混合模型指纹。

来源必须是已封存的普通 run-end publication，按其 receipt 绑定的最终 accepted
canonical selection 取轨迹。复用 `online-index`、submission selection 与
`difficulty-dataset compose` 实现来源校验和合并。没有正式 publication 的活跃
run 不隐式暂停、发布或改选；调用方在 `skipped_groups` 中记录原因。

`difficulty-dataset resolved --eligibility submitted` 是显式的到提交终点工作量
视图，必须指定 rollout 选择，不要求 benchmark harness 成功。只允许已提交的
SWE 类 patch 轨迹，仍验证 canonical trajectory/subject、publication、native
exact action-close、标签和哈希；不产生成功标签。原 `resolved_only`、
`positive_reward` 和 `accepted_terminal` 规则保持各自定义。

每组所有有效 action-close 点合并后计算一次 pooled R²：主指标是
`log1p(remaining reasoning tokens to submission)` 与原 predictor score 的 R²；
附原 token 空间 `expm1(score)` 的 R²、MAE、RMSE、SSE、方差分母。
不裁负值，不平滑，不平均轨迹 R²，不只选择 resolved，不静默丢弃非有限配对。
读取复用 grouped tensor reader，校验 shard/member 哈希和所需 site/width。

产物位于 `difficulty/evaluations/EVALUATION_ID/`：workflow、estimator 合同、
progress、summary、各组 predictions/逐轨迹指标，以及绑定 composed datasets
与 predictor 文件的最终 receipt。未完成工作流没有成功 receipt；恢复必须保留
原失败目录并用新 evaluation ID，已有数据集只在 receipt/来源一致时复用。

这是观察到的轨迹分布上的 estimator 适用性检查，不是 task holdout；R²变化同时
受到 hidden 表示、任务/轨迹构成和真值方差变化的影响，不能单独证明因果漂移。

### 轨迹级预测与真值对照

Parallel run 的自动 visualization publication 可直接传给现有命令，无需创建
训练 dataset：

```bash
.venvs/analysis/bin/dycon difficulty-plot-trajectories RUN_ID PLOT_ID --rollout-id ROLLOUT_ID --publication-id PUBLICATION_ID --attempt-id EXISTING_FIT_ATTEMPT --target submission_remaining_reasoning_tokens_log1p
.venvs/analysis/bin/dycon difficulty-plot-parallel RUN_ID BRANCH_PLOT_ID --rollout-id ROLLOUT_ID --publication-id PUBLICATION_ID
```

轨迹预测仅使用经过 winner 和 native/tensor 校验的 actual-path hidden，truth
独立按 native suffix 计算。缺失值为 null 并断线，未评测或 unresolved 轨迹的
终点是实际 agent 终点。分支图展示 receipt 绑定的原始 scores，不重新评分，也不
将 score 当作 native truth；无兼容 checkpoint 时仍可单独生成分支图。

`difficulty-plot-trajectories` 将 run 中每个 task 的 resolved repetition 数量与指定 Ridge checkpoint 的逐步预测/实际剩余量并列展示。它不估计成功概率。

#### 选择输入

默认数据集模式沿用冻结 difficulty dataset 的 resolved selection：

```bash
.venvs/analysis/bin/dycon difficulty-plot-trajectories SOURCE_RUN PLOT_ID \
  --attempt-id RIDGE_ATTEMPT --dataset-id DATASET_ID \
  --target submission_remaining_reasoning_tokens_log1p
```

也可显式传 `--rollout-id ID` 或 SHA-bound `--rollout-list FILE`，纳入指定的 canonical terminal rollout，包括未 resolved 的正常结束样本。显式选择不能与 `--dataset-id` 混用；它用于实际终点分析，不会声称失败轨迹具有成功标签。

指定 `--repair-id` 时，预测可读取已发布 pointwise repair 中通过 receipt、canonical trajectory、model、request/response 与 think_close 身份核验的 hidden。Native-exact 与 canonical-identity-only 预测绑定分别标记；后者不能冒充精确 token 证据。原始 NAS publication 与 repair 可按现有入口合并覆盖。`--reuse-predictions-id` 仅复用身份和 SHA 一致的已验证预测。

#### 统计和曲线

- 最终成功次数按冻结的 task×repetition work item 和最终 canonical rollout 计算；缺少最终 result 的 repetition 单列 pending，不当作失败。
- 历史成功 attempt 可用于逐轨迹曲线，但与最终 repetition 成功数分开统计。completed/unresolved、未评测和 infrastructure error 也分别保留。
- 每个图以 canonical assistant step（0-based）为横轴。`think_close` 时当前 step action 尚未执行；剩余 steps 为 `T-s`。
- Reasoning-token 真值只来自已绑定 publication 的 native exact suffix。缺失/截尾未来 token 保持 unavailable，不从文本重建 exact token。
- 预测与真值独立保留：缺 hidden 时预测断开，缺完整 native 后缀时真值断开；不插值、补零、跨缺口连线或平滑。
- log1p checkpoint 预测先用 `expm1` 还原到目标原单位，再与原单位真值比较。图按轨迹展示，不跨轨迹平均或归一化长度。

曲线带有训练 split 与 `used_in_fit` 身份。如果同一 task 或轨迹样本参与 checkpoint 拟合，图只能解释为拟合诊断，不能当作 task holdout 泛化结果。

#### 压缩事件标注

每条 canonical trajectory 的 ACP 事件流会检查明确的 condenser/compression 触发事件。图中以红色竖虚线标出对应 assistant step，HTML 同 task 的各 repetition 仍在同一行；`trajectories.csv` 保留事件索引、事件类型和 step。未记录触发事件时显示“未记录”，不会根据轨迹长度或配置阈值推测压缩发生。

#### 产物和校验

PLOT_ID 在 `difficulty/evaluations/` 中保存 selection、summary、逐点 CSV、task 计数、独立 SVG 图和 HTML 报告；缺失覆盖以 unavailable/coverage 字段记录，不用零值替代。输出拒绝覆盖既有 ID。

```bash
.venvs/analysis/bin/dycon artifact verify PLOT_ID --scope difficulty-evaluation
```

并行分支在线评分图使用独立的 `difficulty-plot-parallel` 入口，语义与 trajectory remaining-work 图不同。

## Steering 分析

### 共识数据集及 LDA、t-SNE、UMAP

核对日期：2026-11-01。离线模块：steering/；正式入口：core.cli / dycon。

#### 数据契约

- actual-consensus-action-v1：候选 exact canonical action 与历史实际 winner action 相同为 positive，否则有效 action 为 negative。标签不是正确性、resolved 或 remaining-work。
- 遍历全部候选，包含 estimator filtered。selected/retained/score/loop exclusion 仅作为来源字段。
- 每个 decision/branch 只选择一个明确绑定的 generated think_close。默认不选 prompt/history、newline、startup/smoke；多 close 歧义计 excluded，不取第一条或平均。
- 从来源 run 冻结配置解析 NAS，只消费登记的 ACKed/finalized publication。核对 trajectory SHA、完整 decision、历史策略、winner、request/branch/token 身份和 receipt。
- 真正缺失计 excluded；SHA、winner/canonical 矛盾、身份错绑和不兼容特征空间属于硬错误，不降级为 retained-only。
- native site ID 来自 shape/dtype 和 model inventory，区分 decoder_output/final_norm。默认拒绝 model/tokenizer/intervention/site/dtype 差异。

#### 环境与隔离目录

在 analysis 环境安装 `locks/steering-analysis.txt` 所列依赖。实际版本冻结到 `analysis.json`。

离线 CLI 从 --config 读取项目位置和 artifact roots；capture/策略/模型身份取各来源 run 冻结快照，不用当前根配置代替历史配置。

#### 最小命令

以下均为新建示例 ID；已有不可变 ID 拒绝覆盖。ROOT 为基础目录，自动追加 steering/category/ID。

~~~bash
cd /home/wangtianyi/DyConGlobal
CLI=.venvs/analysis/bin/dycon
CONFIG=/home/wangtianyi/DyConGlobal/config.yaml
ROOT=/data3/wangtianyi/DyConGlobal

$CLI --config "$CONFIG" steering-dataset build NEW_DATASET_ID --source-run RUN_ID --all-submitted --site 3 --storage-root "$ROOT"
$CLI --config "$CONFIG" steering-dataset check NEW_DATASET_ID --dataset-root "$ROOT"
$CLI --config "$CONFIG" steering-dataset cache NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --site 3
$CLI --config "$CONFIG" steering-dataset cache NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --site 3 --check-data
$CLI --config "$CONFIG" steering-visualize NEW_ANALYSIS_ID --dataset-id NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --storage-root "$ROOT" --site 3 --fit-scope all --memory-budget-gib 32 --max-display-points 20000 --seed 42 --facet retained --format png --format svg --format pdf
$CLI --config "$CONFIG" artifact verify NEW_DATASET_ID --scope steering-dataset --storage-root "$ROOT"
$CLI --config "$CONFIG" artifact verify NEW_ANALYSIS_ID --scope steering-analysis --storage-root "$ROOT"
~~~

--source-run 可重复。三种选择三选一：单 run 重复 --rollout-id、--rollout-list bindings.json、或显式 --all-submitted。
跨 run 的 binding JSON 每条包含 source_run_id、rollout_id，可选 trajectory_sha256。重复 binding、含糊裸 ID、非法 ID、不同身份同名副本拒绝。
显式选择保留同 task 不同 repetition/attempt，不自动挑最新，不继承 estimator 的 resolved-only 规则。
不传 build --site 时冻结兼容 inventory；--site 可重复，表示 native ID 而非数组位置。cache 可只选冻结 sites 的子集。

#### 路径与校验

paths.artifact_roots 支持 steering_dataset、steering_cache、steering_analysis；未设置使用 project root。
显式 --storage-root / --dataset-root / --cache-root 优先；读取还支持 artifact_scan_roots。未登记磁盘可以显式传基础根目录。
不放宽 core.io.owned_path，不将任意外盘路径交给 difficulty dataset resolver。

- steering/datasets/ID：dataset/selection/rollouts JSON，decisions/samples/excluded JSONL，source-evidence、dependencies、summary、receipt。仅冻结元数据和 tensor locator，不复制整份 NAS。
- steering/cache/HASH：cache.json、site-ID.f32.npy、receipt.json。物理 cache root 不进入 hash；缓存缺失不代表 dataset 损坏。
- steering/outputs/ID：analysis/selection/summary、完整 points.csv、points-metadata.jsonl、transform/*.npz、projections/*.npy、plots/*、report.md、index.html、receipt。

dataset check/verify 默认核对 metadata 和绑定来源；--with-tensors 才读大张量。
cache 构建按 shard 分组，校验 shard/member SHA、shape/dtype/finite，按冻结样本次序写 float32 memmap。
缓存有空间预算、排他锁、同盘临时目录和原子发布；半成品、损坏和跨盘冲突拒绝复用，不补零。

#### LDA 和分组

使用 sklearn LinearDiscriminantAnalysis(solver=svd, n_components=1)，等先验，不附加 StandardScaler。
各 site 单独拟合，符号固定为 positive 拟合集均值更大，投影始终为 [N,1]；strip 纵轴只有类别和展示 jitter。
SVD 前估算工作内存；--memory-budget-gib 不足报错，不静默截断样本。缺类/退化记录 unavailable，全部 site 不可用时 CLI 返回 2。

--fit-scope all 是探索性监督可视化，不能把同批分离当作泛化或实际 steering 效果证据。
--fit-scope grouped --split groups.json：每个 benchmark/revision/task 只能进入一个 split，跨 run 同 task 与同 decision 分支不泄漏。
只在 train 拟合，再向全部样本应用冻结 transform。分组须完整覆盖选中任务，重复、冲突、遗漏均拒绝。

~~~json
{"groups":[{"task_key":"TASK_KEY_A","split":"train"},{"task_key":"TASK_KEY_B","split":"validation"}]}
~~~

group 也可使用完整 benchmark/revision/instance_id。分析保存拟合样本清单、分组和来源文件 SHA。
points.csv 保存全部坐标与身份；--max-display-points 只限制 strip 显示，seed、ordinal 清单和 jitter 都冻结，统计和直方图仍用全量。

#### t-SNE / UMAP 二维可视化

使用同一 dataset/cache 和统一 `dycon steering-visualize` 入口，通过 `--method tsne|umap`
选择方法；未指定 method 的新拟合仍使用 LDA。两种方法均为无监督全 cohort 探索性嵌入，
标签只用于颜色和计数，不传给拟合器。当前仅支持 `--fit-scope all`，拒绝 grouped/split，
避免把联合嵌入当作独立验证。单一标签不阻止拟合；少于 4 条或完全相同特征记录 unavailable。

~~~bash
$CLI --config "$CONFIG" steering-visualize NEW_TSNE_ID --method tsne --dataset-id NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --storage-root "$ROOT" --site 3 --pca-components 50 --perplexity 30 --max-iter 1000 --threads 4 --memory-budget-gib 32 --seed 42 --max-display-points 20000 --facet retained --format png --format svg --format pdf
$CLI --config "$CONFIG" steering-visualize NEW_UMAP_ID --method umap --dataset-id NEW_DATASET_ID --dataset-root "$ROOT" --cache-root "$ROOT" --storage-root "$ROOT" --site 3 --pca-components 50 --n-neighbors 15 --min-dist 0.1 --threads 4 --memory-budget-gib 32 --seed 42 --max-display-points 20000 --facet retained --format png --format svg --format pdf
~~~

- 默认先做不白化、无 StandardScaler 的 randomized PCA，维度为 `min(50, width, N-1)`；
  `--pca-components 0` 禁用。全部样本进入 PCA 和嵌入，不根据标签采样。
- t-SNE：sklearn Barnes-Hut，二维、Euclidean、learning_rate=auto、early_exaggeration=12、
  angle=0.5；默认 perplexity=30（必须小于 N）、max_iter=1000（至少 250）。PCA 初始化，
  输入只有一维时改用随机初始化。
- UMAP：umap-learn，二维、Euclidean、spectral 初始化、spread=1、low_memory=True；
  默认 n_neighbors=15（必须满足 2≤值<N）、min_dist=0.1（范围 [0,1]）。
  为保证固定 seed 的可复现性，UMAP 优化使用单线程；`--threads` 控制 BLAS/OpenMP，默认 4。
- 内存估算覆盖输入、PCA、稀疏邻居图和工作区；预算不足拒绝拟合。估算不是实际峰值承诺。
- `points.csv` 保存每个 site 全部 N 行、原身份和 TSNE1/TSNE2 或 UMAP1/UMAP2；
  `projections/site-ID.npy` 为 `[N,2]`。显示上限仅限制散点，不改变拟合或导出。
- `analysis.json` 冻结参数、PCA 信息、版本、seed、线程限制和诊断。
  `transform/*.npz` 对这两种方法仅保存无 pickle 的审计描述和可选 PCA 数组，
  不导出可用于新样本的非线性模型；不能像 LDA 一样调用线性 transform。
- 每种方法和 site 的坐标系独立，方向、点间距离和可见分离不能直接跨方法/seed 比较，
  也不能解释为泛化、action 正确性或实际 steering 效果。

设计参考旧仓库 `littlelan236/TACTrefractored` 的 `src/tact/research/plots.py` 与
`docs/POINT-CLOUDS.md`（commit `dea0e39a820f94586f9ed6a9c8e77f2c589ca874`）：
复用完整点身份、PNG/PDF/SVG/HTML、冻结坐标与校验的设计。该版本没有 t-SNE/UMAP
实现，因此算法通过 sklearn/umap-learn 新增，未复制旧项目的三类轴定义。

#### 冻结重绘

~~~bash
$CLI --config "$CONFIG" steering-visualize NEW_REDRAW_ID --redraw-from OLD_ANALYSIS_ID --analysis-root "$ROOT" --storage-root "$ROOT" --site 3 --seed 43 --max-display-points 5000 --format png
~~~

先校验旧 bundle，只读冻结 labels、transform 和坐标，不读 source run/NAS/cache，不调用 fit。
发布到新 analysis ID，不改变原 dataset receipt。投影或 transform 篡改会被拒绝。transform 为无 pickle NPZ，不是 steering payload。
LDA/t-SNE/UMAP 均支持；method 默认继承源 bundle。显式不同 method 或重新传入嵌入拟合参数将被拒绝。

### 数据集、缓存与可视化产物

`steering` consumes a frozen source-run selection and the corresponding ACKed
capture publication. It labels every verifiable consensus candidate by exact
canonical action agreement with the historical selected action. Estimator
retention and selected-branch status remain provenance fields; they do not
define the binary label.

The package has three immutable artifact categories:

* `steering/datasets/<ID>` stores metadata, source bindings, all candidate
  rows, excluded rows, and source checksums. Hidden tensors remain in their
  published capture location.
* `steering/cache/<HASH>` stores rebuildable per-site float32 memmaps. The
  cache identity includes the dataset receipt, sample order, site inventory,
  and feature-space metadata.
* `steering/outputs/<ID>` stores one supervised LD1 or a two-dimensional
  unsupervised t-SNE/UMAP embedding per selected site, frozen coordinates,
  fit descriptors, reports, plots, and a receipt.

The formal entrypoints are `dycon steering-dataset build/check/cache`,
`dycon steering-visualize`, and `dycon artifact verify ... --scope
steering-dataset|steering-analysis`. `steering-visualize --method lda|tsne|umap`
selects the fit method (default LDA). t-SNE/UMAP fit all selected samples without
labels, with optional PCA and explicit seeded parameters; a display cap affects
plots only. Their NPZ descriptors are not reusable out-of-sample reducers.
Visualization is exploratory; coordinates are not steering payloads and redraw
consumes frozen coordinates without refitting, even with the source data offline.
