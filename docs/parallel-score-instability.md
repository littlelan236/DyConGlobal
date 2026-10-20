# 并行分支剩余难度估计的一致性评测

统一命令 `dycon difficulty-evaluate-parallel` 评估“同一个历史 prefix、最终工具 action 相同，但 reasoning 不同”的评分波动。代码与[并行轨迹评分可视化](parallel-score-visualization.md)同在 `difficulty/evaluation/`，两者复用同一个 canonical trajectory／已发布分支读取器。

只读取当时保存的预测，不加载估计器/checkpoint，不读取 hidden tensor，不启动 GPU、Docker 或新 rollout。任何已成功发布对应分支记录的估计器均可使用，不限于 Ridge。

## 使用

在 `/home/wangtianyi/DyConGlobal` 使用 analysis 环境：

```bash
# 查看可选择的 task、attempt、rollout
.venvs/analysis/bin/dycon difficulty-plot-parallel RUN_ID --list

# 单条轨迹：task 有多个 attempts 时必须指定 attempt，或直接使用 rollout ID
.venvs/analysis/bin/dycon difficulty-evaluate-parallel \
  parallel-ridge-event-b4-w12-disjoint31-20261019 consistency-sympy \
  --task-id sympy__sympy-18199 --attempt-id attempt-000

# 多条明确选择的轨迹；--rollout-id 或 --task-id 均可重复使用，但两类不能混用
.venvs/analysis/bin/dycon difficulty-evaluate-parallel RUN_ID EVALUATION_ID \
  --rollout-id ROLLOUT_A --rollout-id ROLLOUT_B

# 无 task/rollout 筛选时：整个来源 run 的可分析 canonical trajectories
.venvs/analysis/bin/dycon difficulty-evaluate-parallel RUN_ID EVALUATION_ID \
  --skip-unavailable

# 更严格的原生 reasoning token 身份，不退回文本，也不重建 token
.venvs/analysis/bin/dycon difficulty-evaluate-parallel RUN_ID EVALUATION_ID \
  --rollout-id ROLLOUT_ID --reasoning-identity tokens
```

每个 EVALUATION_ID 必须是新 ID，不能与现有绘图/评分产物同名。等价的薄脚本入口为 `python -m difficulty.evaluation.scripts.evaluate_parallel`，参数仍由统一 CLI 处理。

默认 whole-run 选择保留该 run 的全部可分析 attempts，任务内先对 rollout 等权；不自动挑选成功、最新或最短轨迹。canonical 内容重复的 aliases 由现有 selector 去重并记录；quarantined/excluded 不进入默认范围。显式 task 有多个匹配时会列出候选并拒绝猜测。

`--skip-unavailable` 仅跳过缺少已发布 finalization／branch receipt／action-bound parallel data 的轨迹，保存具体 rollout ID 和原因。checksum 不匹配、错误 native/step 绑定、不同 checkpoint/target 混用等仍会报错；不会借此跳过损坏证据。全部不可用时不生成空的“成功评测”。缺失 reasoning 证据则在候选覆盖统计中单列。

## 分组与身份

比较组的身份为 `(run, rollout, 实际 step, parent prefix, action)`。只使用 **status=eligible、评分有限、action 和 reasoning 身份均可用** 的候选。未选中的合格分支全部保留；不能只用赢家进行稳定性分析。

action 身份使用有序工具调用列表中的工具名和解析后的完整 JSON 参数：忽略 call ID 与 JSON key 顺序，保留调用顺序及命令字符串的空格、换行等字节内容。不把近似命令或 AST 类别视作同一个 action；重复 JSON key、非有限 JSON 数字和非法参数会标记为身份不可用。此处只是完全相同 action 的身份分组，不替代 `dycon action-classify` 的语义分类，也不要求预先运行它。

reasoning 身份有两种显式模式：

| 模式 | 定义与边界 |
|---|---|
| `text`（默认） | 对保存的 `reasoning_content` 或 provider 的 reasoning 字符串计算 hash；不去空白、不改大小写。只代表文本身份，不声称原生 token 相同 |
| `tokens` | 复用 `hidden.artifacts.read_request` 核验 ledger/inventory，复用 `enrichment.online_trace.reasoning_blocks` 定位被评分的 close，hash 本轮 reasoning token IDs；不包含后续 action tokens，不 retokenize。需要完整 ledger 和 marker inventory，读取量较大 |

两个统计视图总是同时输出：

- **`deduplicated` 主口径**：同一个 action 内，相同 reasoning hash 的评分先求均值，再对不同 reasoning 各计一个单元。避免重复 reasoning 人为降低波动。
- **`sampling` 原采样口径**：保留每个原始候选，包括重复 reasoning，反映实际采样频率。在该视图的主指标和倍数率中，重复 reasoning 之间的近零差异也参与计算。

无论哪种视图，只有至少两种不同 reasoning 的 action 组才支持主／辅助／倍数指标。单候选组或只有重复 reasoning 的组记录为 unavailable，不能以 0 冒充稳定。

## 四类指标

以下 (s_1,\ldots,s_m\) 为当前视图的组内评分。去重视图中 m 为不同 reasoning 数，采样视图中 m 为候选数。

**主指标：平均两两评分差。**

\[
D_g=\frac{2}{m(m-1)}\sum_{i<j}|s_i-s_j|
\]

**辅助指标：组内样本标准差与极差。**

\[
\sigma_g=\sqrt{\frac{\sum_i(s_i-\bar s)^2}{m-1}},\qquad R_g=\max(s)-\min(s)
\]

标准差使用 **ddof=1**。报告的 `mean_sample_std`、`mean_range` 为组指标按下述层次汇总的均值，不是将所有评分混合后求标准差。另存所有有效组 D、标准差、极差的中位数/P90/P95/max；这些是 pooled 描述统计，并按 `unit_count` 分组，防止混淆不同候选数量的极差。

**倍数分歧率：仅对 log1p 输出有效。**

\[
P_g(K)=\frac{2}{m(m-1)}\sum_{i<j}\mathbf{1}(|s_i-s_j|\ge\log K)
\]

默认 K 为 1.25、1.5、2、3；`--multipliers 1.5 2 3` 可改为其它有限且大于 1 的阈值。`pair_rate_ge_2.0x` 表示组内候选对的 2 倍分歧率；`group_rate_ge_2.0x` 表示组内是否存在任意一对达到 2 倍分歧，随后按同样的层次平均。所有 rate 范围为 [0,1]，不是百分数。

分差 δ 对应模型隐含的 `1+剩余量` 尺度比 `exp(δ)`，不是实际剩余量的比例。保持负预测，不截断、不平滑、不先做 exp 变换。

`--score-space auto` 在 target 名以 `_log1p` 结尾时启用倍数率；其它目标仍计算 D、标准差、极差、Q，但 `multiplier_rates_available=false`，不产生无意义的倍数率字段。非标准命名而输出确为 log1p 时可显式指定 `--score-space log1p`，该声明会写入参数；`--score-space native` 强制禁用倍数率。

**相对指标 Q：同 step 内的 action 条件一致性。**

在同一步内，先分别计算：

1. 相同 action、不同 reasoning 的候选对评分差；各 action 组先平均，再对 action 组等权。
2. 不同 action、不同 reasoning 的候选对评分差；各 action 对先平均，再对 action 对等权。

两侧都排除 reasoning hash 完全相同的对；采样视图也遵守此条件。只有同时支持这两侧的 mixed step 参与 Q。两侧分别做 rollout/task 平均后相除：

\[
Q=\frac{\text{paired task-balanced same-action gap}}{\text{paired task-balanced different-action gap}}
\]

它是“均值的比值”，不是各 step 比值的平均；不使用仅支持一侧的 step 填充分子或分母。分母 <=1e-12 时 ratio=null，不加 epsilon 后强行相除。Q 越小表示同 action 更集中；Q 接近 1 表示区分有限。常数预测器的 Q 未定义，不能据零波动判为有用。没有 hard-code “合格/不合格”阈值。

## 汇总与置信区间

主指标、辅助指标与倍数率按 **action 组 → step → rollout → task** 逐层等权平均；同一个 task 的多个 attempts 不获得多个独立 task 的权重。每项指标只汇总自身有支持的数据，并输出实际支持的 task、rollout、step 数。

task 身份与聚类以 catalog 的 `task_key` 为准，`task_id` 仅作可读名称；不同 benchmark/base commit 的同名实例不会因此被误并。

`--bootstrap-samples` 默认 2000，`--seed` 默认 42；`--bootstrap-samples 0` 关闭 CI。分别提供 whole-task 与 whole-repo 的 block bootstrap 95% CI，保留任务内部 step 的时间相关性；repo bootstrap 重采样完整 repo block，再按其任务数归一，保持 task 等权定义。没有将同一步的候选对当独立样本。

单 task 或单 repo 无法构造对应聚类 CI 时为 null。Q 的 bootstrap 分子/分母共同抽样；若超过 5% 的重采样分母接近零，则不报告条件化的 CI，保存 valid_resamples。区间不是单条轨迹中任意四分支的“置信区间”，也不是预测真值误差条。

## 产物与字段

输出在 `difficulty/evaluation/outputs/EVALUATION_ID/`，该目录已有 gitignore：

| 文件 | 内容 |
|---|---|
| `summary.json`、`report.md` | 两种视图的四类指标、task/repo CI、组大小分布、覆盖统计 |
| `branches.csv/jsonl` | 实际 action step 的全部候选，score/status、action/reasoning hash、identity_status、native/sample ID |
| `groups.csv/jsonl` | 各 action 组的单元评分、分支 ID、D/std/range、倍数率、是否可用 |
| `steps.csv/jsonl` | 各 step/view 的组平均，以及 Q 两侧评分差与 step ratio |
| `tasks.csv/jsonl` | 每个任务、每种视图的指标与支持数量 |
| `selection.json`、`sources.json` | 冻结选择、canonical SHA、run snapshot/finalization/branch receipt、native 模式额外 ledger/inventory 证据 |
| `receipt.json` | 参数、代码与全部产物 hash |

例如主要结果路径：

```text
views.deduplicated.aggregate.mean_pair_gap.mean
views.deduplicated.aggregate.mean_sample_std.mean
views.deduplicated.aggregate.mean_range.mean
views.deduplicated.groups.pooled_descriptive.range.p95
views.deduplicated.aggregate.pair_rate_ge_2.0x.mean
views.deduplicated.aggregate.relative.ratio
```

`sources.json` 中原生 ledger 使用文件 SHA，inventory 的 `hash_kind=canonical_json` 使用既有 inventory 规范化 JSON hash，不能混用。CSV 里的列表/对象为 JSON 字符串，缺失值为空；JSONL/JSON 缺失为 null。产物临时目录写完后原子发布，已有 ID 拒绝覆盖。

步骤身份来自可视化已有的 finalized native-request → canonical message/actual step 绑定。旧轨迹的 query step 可能因 FormatError 重试而比实际 action step 多；`branches` 中保留 `query_step_id`，统计始终使用 `step_id`。重试响应与无赢家终止请求不会伪造额外 action step。

发现异常组后，可用相同的 run/rollout 和 `--step-start/--step-end` 调用 `difficulty-plot-parallel` 看原图。两个入口使用相同轨迹绑定规则，统计不会维护第二套 step 定义。

## 代码与验证

| 模块 | 职责 |
|---|---|
| `difficulty/evaluation/parallel.py` | 绘图和评测共用的发布证据／实际轨迹读取与对齐 |
| `difficulty/evaluation/parallel_identity.py` | action 完全相同分组、text/token reasoning 身份 |
| `difficulty/evaluation/instability.py` | 无 IO 的四类指标、层次汇总、聚类 bootstrap |
| `difficulty/evaluation/parallel_instability.py` | 选择、覆盖统计、评测产物发布 |
| `difficulty/evaluation/scripts/evaluate_parallel.py` | 统一 CLI 的薄包装 |

```bash
.venvs/analysis/bin/python -m pytest -q \
  tests/test_parallel_instability.py tests/test_parallel_visualization.py
```

测试覆盖手算值、重复 reasoning 均值去重、组大小权重、倍数阈值边界、Q 的 paired ratio、零分母、负预测、task/rollout 平衡、token/action 边界、checksum 拒绝、FormatError 计数偏移与既有绘图兼容性。

这套评测衡量条件一致性，不证明估计准确或选中 action 更有帮助。尤其 preserve_thinking 时，不同 reasoning 会影响后续行为；“同 action”不保证真实未来剩余工作量相同。模型比较仍应另看独立预测误差与 rollout 性能。
