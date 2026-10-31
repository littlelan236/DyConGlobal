# Action 熵与动态保留分支数量：可视化扩展计划

日期：2026-10-28（Asia/Shanghai）。工程：`10.249.187.9:/home/wangtianyi/DyConGlobal`。

状态：**已通过 SSH 核对当前工作树并完成设计；尚未实现可视化扩展。** 本次仅新增此计划文档，不修改运行逻辑、根配置、旧 run 或历史分析产物，不启动模型、GPU、rollout 或 check-only。

## 1. 目标与范围

在现有 `action/` 分支统计与轨迹进度可视化中，通过统一接口一次生成以下各量的独立图片：

1. 当前 canonical step 用于动态筛选的 action 熵 `decision_pool_entropy`。
2. 动态规则产生的 top-k 保留数量 `decision_keep_count`，并与实际 `retained_branch_ids` 的长度核对。
3. 作为上下文的决策池分支数 `decision_pool_count`，帮助区分熵变化与可用候选数变化。

复用正式入口 `dycon action-branch-stats` 和现有冻结分析产物体系；支持单轨迹、单 run 平均及跨 run 对照。**每个指标单独一张图，一次命令生成全部选中指标的图片**，不拼接多指标面板。新增 `--line-metric-set entropy-retention` 预设，展开为 H、k、N 三项；现有可重复 `--line-metric` 用于显式选择任意指标集合，一次调用同样批量生成各指标图片。预设与显式指标选择互斥。不指定指标参数时沿用各 line-mode 的默认指标集合，但新版本逐指标分图；历史多面板产物保持原位。

本计划中“保留分支”特指 **estimator top-k 筛选后进入共识投票的分支集合**，不是 action 种类数、winner 组大小、实际执行工具的分支数，也不是下一 step 的生成分支数量。当前架构最终只执行获选代表的 action。

## 2. 当前三种算子在哪里

三者都是对**同一个原始 Shannon action 熵**的保留数量映射，并非三种不同的熵定义。

主文件：`rollout/entropy.py`。

| 位置（核对时行号） | 职责 |
|---|---|
| 第 5 行 `RETENTION_RULES` | 三种规则标识 |
| 第 16 行 `entropy_retention()` | 统一纯函数入口 |
| 第 26–29 行 | 原始 Shannon 熵计算，单位 nats |
| 第 32–33 行 | A：`effective_count_v1` |
| 第 34–35 行 | B：`shannon_exp_keep_v1` |
| 第 36–37 行 | C：`baseline_floor_v1`（经过规则合法性检查后的 else 分支） |
| 第 8–13、38 行 | 16 ULP 整数边界吸附、向上取整和上下界约束 |

设当前最终 eligible 池大小为 `N`，各 canonical action 的频数为 `n_i`，`p_i=n_i/N`，`m=min_retained_branches`，`r₀=retention_ratio`：

```text
H = -Σ p_i ln(p_i)
A: raw_keep = exp(H)
B: raw_keep = N - (N-m) exp(-H)
C: raw_keep = N [1-(1-r₀) exp(-H)]
```

当 `N>=m`，`k=min(N,max(m,stable_ceil(raw_keep)))`；当 `N<m`，当前实现返回 `keep_count=0` 和不足/不可用状态。`N=0` 时 H 为 null。A/B 不使用 r₀；C 使用。B 是函数签名默认值，但正式创建动态 run 仍须显式冻结 `entropy.retention_rule`，不能据此推断已有 run 使用 B。

相关调用和配置：

- `rollout/consensus.py::consensus_options()`：从本 run 的冻结配置提取规则。
- `rollout/consensus.py::select_consensus()`：整理合法且有有限 estimator score 的 action 池，调用上述纯函数，按 `(score, branch_id)` 升序精确保留 k 条，记录 IDs、cutoff score 和 cutoff branch ID。分数越低越优；同分不额外扩充数量。
- `rollout/decision/coordinator.py`：调度与 loop escape 定案后调用共识选择，形成决策证据。
- `core/parallel_policy.py`：独立 run 策略的 schema、冻结和校验；正式模式名保持现有拼写 `parallel_enthophy`。
- `core/action_identity.py::canonical_action()`：统一 action 身份定义，不因本扩展改变。

## 3. 当前可视化能力与准确缺口

| 文件 / 函数 | 已有能力 | 本次计划的扩展 |
|---|---|---|
| `action/analyzers/branches.py::decision_record()` | 读取 full/embedded decision，核对 canonical winner、retained IDs 和新策略证据；已有 `decision_pool_entropy/count`、`decision_keep_count`、`retention_rule` | 冻结动态状态及必要审计字段，统一缺失字段结构 |
| 同文件 `step_statistics()` | 把上述字段复制到逐 step 结果 | 确认数值、actual retained 数和状态，保留 all/retained 原指标 |
| `action/evaluation/branch_curves.py::METRICS/metric_columns()` | 只识别 all/retained/delta 三种指标 scope | 增加独立 decision scope；目前三项 decision 数值不会进入曲线 |
| 同文件曲线构造、绘图函数 | 原始 step 单轨迹图，0–100% 归一化 run 平均图；PNG/SVG | 注册新增指标、逐指标独立绘图、计数坐标和覆盖信息 |
| `core/cli.py` | `--line-metric` 的 choices 来自 `METRICS` | 新增指标及批量预设 `--line-metric-set entropy-retention`，仍使用同一 CLI |
| `action/analyzers/branches.py::check_branch_analysis()` | SHA、来源、逐 step 重算、曲线 CSV 和逐轨迹图覆盖校验 | 按新版本校验 decision 指标与计数一致性，保持历史校验 |

现有 `entropy_all` 统计当前 inventory 中可识别 action，`entropy_retained` 统计筛选后集合；它们不保证等于动态规则实际使用的、经过 score/合法性/loop escape 处理的池熵。**不能拿 `entropy_all` 替代 `decision_pool_entropy`，也不能拿 `valid_actions_retained` 替代 k。**

当前结构：`schema=exact-action-branch-statistics-v2`、`curve_schema_version=4`。当前 `metric_columns()` 只认后缀，而 `resolve_line_metrics()` 和 `build_curve_tables()` 主要从 `rows[0]` 判断字段，混合新旧轨迹时会遗漏后续才出现的指标；需一起修正。

## 4. 数据口径与校验

### 4.1 新增可绘指标

| CLI 指标名 | 来源 | 单位 / scope |
|---|---|---|
| `decision_pool_entropy` | `decision.consensus.entropy.entropy` | nats / `decision` |
| `decision_keep_count` | `decision.consensus.entropy.keep_count` | branches / `decision` |
| `decision_pool_count` | `decision.consensus.entropy.eligible_count` | branches / `decision` |

使用同一 step 的完整决策证据或与 full receipt 绑定并通过校验的 embedded summary。保留现有 `_full_decision()` 路径解析、source config、SHA 和 canonical action 校验，不从当前根配置猜测历史参数，不重打 estimator 分数、不模拟其他规则。

新增审计字段建议：`decision_entropy_status`、`decision_entropy_stage`、`decision_actual_retained_count`、`decision_raw_keep`、`decision_cutoff_score`、`decision_cutoff_branch_id`；沿用 `retention_rule`、`representative_policy`。字段直接来源于冻结证据或 retained IDs 长度；不必全部成为绘图指标。

`decision_actual_retained_count` 的来源仅为已验证 `consensus.retained_branch_ids` 的长度。动态有效决策必须满足：

```text
decision_keep_count == consensus.retained_count == len(retained_branch_ids)
0 <= decision_keep_count <= decision_pool_count <= branch_count
status == available => m <= decision_keep_count <= N
retained_branch_ids 是 candidate_branch_ids 的子集
```

保留现有 `validate_consensus_evidence()` 作为选择重放和完整性校验入口；若缺少必要保护，应在该统一验证器补齐，而非在 action 中重复实现 A/B/C 公式。核对规则、pool IDs、H、raw_keep、k、rank 和 cutoff 的证据绑定。新分析记录本 run 冻结策略及相应身份，不引入第二套策略配置。

### 4.2 旧 run、缺失与异常

- 旧 `parallel_consensus` 没有 entropy 块：decision 三指标均为 null，明确标记 `not_dynamic`；可保留 actual retained 数作为审计字段，但不得冒充动态 k。
- 动态元数据缺失：标记 `missing_evidence`。如果冻结配置或已有证据明确宣称动态模式，却缺失必需决策元数据，应拒绝把该点当成已验证动态观测；完整性冲突继续报错。
- 原始不可用动态决策若有完整记录：保留真实 `status`；N=0 的 H 为 null，k=0 只有明确记录时才为真实 0。N<m 的 H 可能仍有效，但须标明无法形成正常筛选决策。
- 当前正式分析遍历 canonical assistant message 中的 `extra.dycon_parallel`，要求 selected branch；失败且未进入 canonical trajectory 的 decision 不会自动纳入。首版维持此范围，不把故障日志里的未提交决策插入轨迹；后续若需失败决策时间线，应另行设计。
- 为同一新分析的所有 step 显式提供相同指标键，缺失使用 null，CSV 保持矩形，禁止补 0；指标注册/导出从完整行集合或固定版本 schema 获取，不依赖首行。
- 混合新旧 run 时只对有效动态观测求均值，逐点报告实际贡献数。显式请求新增指标但全无可用动态观测时，给出清晰错误；单条无动态观测轨迹在混合分析中为每个选中指标生成独立说明图，覆盖索引仍保留。

## 5. 图形与进度设计

采用**一指标一图**：H（nats）、k（branches）、N（branches）分别输出文件，使用一致的进度口径。预设一次生成三类图片；重复 `--line-metric` 可选择任意子集或增加现有指标，均逐指标单独输出。每张图只有该指标的一套坐标轴。

同一指标的 all/retained 两种 scope 可以在该指标图中比较；跨 run 模式也可在同一指标图中比较多个 run。按“量”分图，保留所选 scope/run 的对照能力。

- `per-trajectory`：沿用原始 assistant step（从 1 开始），每 `(source_run_id, rollout_id, metric)` 一张图；每个 parallel 决策点画 marker，不插值、不平均。k/N 使用整数 tick；缺失点断线，不能画出未观测的整数筛选预算。
- `per-run/all-runs/both`：沿用 `assistant_index/(total_assistant_steps-1)` 的 0–100% 进度；L=1 仅在 0% 有观测。保持原有逐轨迹线性插值、无外推、不跨显式缺失值插值，再按轨迹等权平均的约定。
- 平均图中的 k/N 是**插值后轨迹均值**，允许小数，图例和脚注必须说明；不四舍五入后冒充某个真实 step 的整数 k。不为了新指标静默改变旧熵曲线的插值算法。
- 稀疏 parallel 观测间的线段/插值只是分析约定，不表示中间每个 assistant step 都执行过该筛选；脚注明确这一点。原始点可由逐 step 文件追溯。
- `decision` 指标与 `--branch-scope all/retained/both` 独立，只出现一条 decision 曲线，不复制成 all/retained 两条。原有 entropy all/retained 与 delta 行为保持。
- 标题/元数据标出 mode、retention_rule、代表策略；不同 run 使用各自策略。首版不在同一历史池上额外画未执行的 A/B/C 反事实曲线。
- 继续提供 `n_trajectories`、`total_trajectories`、sample_std、standard_error；新增每指标有效/不可用 step 与轨迹数，避免不同覆盖率造成误读。

## 6. 产物与版本兼容

继续使用 `action/outputs/ANALYSIS_ID/`，必须是全新 ID：

```text
analysis.json / selection.json / summary.json
decisions.jsonl / steps.jsonl / steps.csv / trajectories.json
source-config/ / source-evidence/ / code/
curves/per-trajectory.csv                         # per-trajectory 模式
curves/per-run.csv / all-runs-normalized.csv / raw-steps.csv
plots/per-trajectory/RUN_ID/ROLLOUT_ID/METRIC.png|svg
plots/per-run/RUN_ID/METRIC.png|svg
plots/all-runs-normalized/METRIC.png|svg
plots/trajectory-index.json                      # per-trajectory 模式，含各指标文件
plots/plot-index.json                            # 所有绘图模式的批量图片清单
branch-receipt.json
```

新增 curve 行使用完整指标名及 `scope=decision`，保留来源轨迹、axis、x、value/mean 等既有列；逐 step 文件保留 decision_id/step_id，便于回溯。不要给 aggregate CSV 每一行虚构单个 decision_id。

统一命令的返回结果包含 `plots/plot-index.json` 路径及生成图片数量。清单按 mode、run/rollout、metric、format 记录文件、状态和缺失原因；逐轨迹索引按指标列出文件。`--check` 检查所有选中指标和格式的覆盖，不能仅检查代表图；无数据说明图也登记。

建议新输出提升 `curve_schema_version` 至 5，冻结新增指标 schema、scope、缺失值与覆盖规则，以及 `plot_layout=one_metric_per_figure`、请求的 metric set 与展开后的指标清单；必要时添加独立 `decision_metrics_version=1`。保持统计 v2 顶层身份的前提是采用明确版本分派；若重算结构不可兼容，应显式升主 schema，不能继续宣称旧语义。

旧 v1/v2 和 curve<=4 分支按原有字段存在性与导出方式重算，不向旧 records 注入新 null 键，不重写 receipt。所有新增字段仅用于新版本分析。`--check` 必须按冻结版本选择计算路径，不能用新版默认字段集合让旧分析失败。对 v5 重算三项 decision 数值、覆盖、CSV 和图清单；保留实现快照差异提示，不把正常代码升级当成历史产物损坏。

## 7. 实施顺序

1. **数据和验证**：扩展 `branches.py` 中的新版本 records/steps，补齐动态状态、actual retained 数与一致性检查；使用现有统一验证器。补旧记录和新记录的版本分派。
2. **曲线和 CLI**：为 `METRICS` 建立显式 decision 指标描述，更新 `metric_columns()`、`resolve_line_metrics()`、原始/归一化 curve builder；消除 rows[0] 依赖；CLI 通过现有 choices 接入，并增加批量预设及其与显式指标选择的互斥校验。
3. **绘图和覆盖**：遍历选中指标，每指标独立生成图片；支持 decision scope 图例、计数轴及无数据说明图；补统一 plot-index、trajectory-index 和 summary 的指标级覆盖。原有默认指标集合也逐指标分图。
4. **校验与回归**：v5 重算、完整覆盖和篡改校验；旧版本产物校验与原始命令回归。所有测试通过后做只读来源数据的新 ID 离线分析。
5. **文档与交付**：更新 `action/README.md`、`action/BRANCH-ACTION-STATS-CN.md` 和入口说明，保存测试结果、分析 ID、实际来源和可复现命令。没有真实动态数据时如实说明，只能报告 fixture 验收，不能声称真实 run 图已完成。

实施前再次核对工作树及文件 SHA，保留当前大量未提交修改。临时材料放本地本任务子目录、远程 `temp/action_entropy_retention_visualization_20261028/`；正式功能直接扩展上述模块，不另造 launcher 或 trajectory reader。

## 8. 拟用命令（新增指标实现后才可执行）

以下新增指标和 `--line-metric-set` 预设目前尚未实现；命令是实施后的验收接口设计。

```bash
cd /home/wangtianyi/DyConGlobal
# 单条轨迹：一次命令分别生成 H、k、N 三张独立图；PNG/SVG 共六个图片文件。
.venvs/analysis/bin/dycon action-branch-stats NEW_ENTROPY_KEEP_ID \
  --source-run DYNAMIC_RUN --rollout-id CANONICAL_ROLLOUT \
  --line-mode per-trajectory --branch-scope both \
  --line-metric-set entropy-retention --format png --format svg

# run 轨迹等权平均：一次生成 H、k 两张独立图。
# 可用上述预设一次生成 H、k、N 全套；可重复 --source-run 对照不同真实规则。
.venvs/analysis/bin/dycon action-branch-stats NEW_ENTROPY_KEEP_MEAN_ID \
  --source-run DYNAMIC_RUN --line-mode per-run --progress-points 101 \
  --line-metric decision_pool_entropy --line-metric decision_keep_count

.venvs/analysis/bin/dycon action-branch-stats NEW_ENTROPY_KEEP_ID --check
```

实际 source run 和 rollout ID 必须从当前发布证据选择，不能直接复用占位符。已有 `action-plot` 用于七分类冻结视图，不作为本分支曲线扩展的入口。

## 9. 测试与完成标准

优先扩展 `tests/test_action_branches.py`；复用 `tests/test_parallel_entropy.py`、`tests/test_parallel_entropy_integration.py`、`tests/test_run_parallel_policy.py` 的真实选择 fixture。可新增专门的曲线测试文件，但不重复生产公式。

必须覆盖：

1. A/B/C 各至少一个受验证 decision：H、N、k 与 receipt 一致；同 H 不同 N/m/r₀ 时读取实际结果。
2. 三种 rule × 两种代表策略：代表策略不改变 k 的数据口径；score 同分由 branch ID 限定确切数量。
3. loop escape 后池与 inventory 不同、invalid score、不可识别 action：decision H/N 不被 all H/N 替代。
4. 旧 fixed consensus、动态/旧 run 混合且旧数据先出现、部分动态元数据缺失、完整性矛盾和全无动态数据。
5. N=0、N<m 的数据层行为及 canonical 分析边界；不能为测试强行构造有 winner 的不合法成功 decision。
6. k==retained_count==IDs 长度；IDs 重复、越界、池外和 cutoff/rule/策略身份篡改都被拒绝。
7. 同 task 多 repetition 分图、跨 run 同名 rollout 分离、无 parallel/无动态轨迹的逐指标覆盖。
8. 单轨迹原始值为整数；平均曲线允许小数；长度归一化首中末点、L=1、缺失、无外推和逐点分母正确。
9. `--line-metric` 新 choices、metric-set 预设展开及互斥校验、三种 branch scope、所有 line-mode、PNG/SVG、分析 ID 防覆盖；一次调用生成所有选中指标的独立图片，验证图片数量、文件路径、单指标坐标轴和完整清单。
10. 新 receipt 的 CSV/plot/index 篡改可检测；旧 v1/v2、curve<=4 分析通过其历史校验路径，不改旧文件。

计划中的 CPU 回归命令（本次未运行）：

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg .venvs/runner/bin/python -m pytest -q \
  tests/test_action_branches.py tests/test_parallel_entropy.py \
  tests/test_parallel_entropy_integration.py tests/test_run_parallel_policy.py
```

另在 `.venvs/analysis` 通过正式 CLI 完成新增指标 help、fixture/真实已有数据离线作图和 `--check` 验证。真实数据验收至少抽首、中、末三个 decision，对照 H/N/k、retained IDs、图点和 CSV；整批图清单必须覆盖所有选中轨迹。

完成条件：正式入口一次生成并校验决策池熵、动态保留数及候选池大小各自的独立图片，值可追溯至实际 receipt，新旧版本与选择规则回归通过；真实动态来源缺失时明确列为尚未完成的真实数据验收，不编造实验结果。

## 10. 本次核查基线

源码依据是 SSH 读取的远程**当前工作树**，不是 IDE 历史副本；行号以后可能变化。本次只运行只读源码/文档检查和 `action-branch-stats --help`，未运行实施测试或实际分析。

| 文件 | SHA-256（2026-10-28 核查） |
|---|---|
| `rollout/entropy.py` | `a84420ea7b36412cd14d8e1b74cc30f5d78a70b1e582609c7de98187c237efa7` |
| `rollout/consensus.py` | `bd64c5dfe931f06ef02c5236ece701ae45ccdff262ce8a4de642ec9be998c557` |
| `action/analyzers/branches.py` | `40ec46d3e75fa92ad3c25d599216aae06512e564e0ffa6bfe9d89ee53bb12d65` |
| `action/evaluation/branch_curves.py` | `c5dceb2e00e0f172c8439b11b035cc31e98c1c086a95f09ac6fcebf1f60252bb` |
| `core/cli.py` | `39a6a5c382bb353911e1189a470a55c6a37e12310e2f35359ee853daac768209` |
| `core/parallel_policy.py` | `1f7733f342e61b50cf339e80717d2a3d6042cb25051e96f31c917b98805b0447` |

远程交付位置：`docs/ACTION-ENTROPY-RETENTION-VISUALIZATION-PLAN-CN.md`。本地同名副本位于本任务目录；上传后逐字节校验 SHA。SSH 口令不写入文档、脚本或配置。
