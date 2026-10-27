# 复查长度分析：接口、规则、运行结果

日期：2026-10-26；工程根目录：/home/wangtianyi/DyConGlobal。

## 实现位置

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

## 范围与可复现入口

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

## 当前规则

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

## 最终运行结果

最终交付ID为review-length-submitted302-final-20261026。较早results、episodes-v1/v2
产物是开发/回归快照，保留原件，不混合其均值。

| Run | Submitted | 可估计 | Unknown | 净步数均值 | 中位数 | P90 | 收尾跨度均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025 | 40 | 39 | 1 | 9.21 | 10 | 15.00 | 12.64 |
| verified80-parallel-baseline-20261021 | 73 | 70 | 3 | 11.19 | 10 | 18.00 | 14.99 |
| verified80-parallel-reasoning-fraction-gpu01-w4-s16-20261025 | 61 | 56 | 5 | 11.30 | 10 | 21.00 | 14.62 |
| verified80-parallel-reasoning-log1p-20261021 | 69 | 63 | 6 | 10.98 | 10 | 19.80 | 14.59 |
| verified80-parallel-step-linear-20261021 | 59 | 53 | 6 | 10.38 | 10 | 19.60 | 13.92 |
| 合计 | **302** | **281** | **21** | **10.74** | **10** | **19.00** | **14.30** |

新增指标“中段复查后晚变更”：在 `unresolved_late_mutation` 轨迹中，已经识别出至少一个复查 episode 的轨迹数。最终总体为 **15/15**；也就是说，15条因后段修改而无法确定最终复查边界的轨迹，都存在可识别的中段复查行为。这不是最终复查长度，也不是人工心理意图标签。

覆盖率93.05%。281条可估计轨迹严格净步数合计3,017；宽松口径均值12.54。
严格净步数占轨迹长度macro均值28.64%，按step加权micro占比22.61%；
两种分母均只包含281条可估计轨迹，不把21条未知当作0。

21条unknown中：6条未检测到候选改动，15条末段存在未解析修改。
281条估计中：105条无规则警告、176条带警告。无警告子集均值7.30步，单独保留在summary中；
该筛选子集也不是人工真值集，不能用来宣称整体准确率。

按此前冻结评测分组：

| 评测组 | Submitted | 可估计 | Unknown | 净步数均值 |
|---|---:|---:|---:|---:|
| resolved | 193 | 184 | 9 | 11.17 |
| unresolved | 69 | 67 | 2 | 11.04 |
| 非空无效patch | 25 | 21 | 4 | 6.38 |
| 空patch | 8 | 2 | 6 | 9.00 |
| 未评测 | 7 | 7 | 0 | 9.86 |

空patch组仅2条可估计，缺失明显，不能把9.00直接与其他组作总体比较。
同一组80题在多个run重复出现，此处仅为规则下的描述统计，不构成因果或显著性结论。

## 代表轨迹核对

| Run / rollout | 起点step | 净步数 | 跨度 | 核对重点 |
|---|---:|---:|---:|---|
| baseline / r-d2bd1a35cb0f6baee93935b84b1b7c6d | 4 | 6 | 8 | 与此前xarray手工示例一致 |
| baseline / r-fcb0568e4597df8cf1fa20e3aec6e1c8 | 22 | 8 | 9 | docstring修改后重入，清理helper不重置 |
| step-linear / r-77edf194284e3f8dc003292c0047f73c | 129 | 6 | 7 | 128回退后核查，repro通配符清理不重置 |
| consensus / r-9c65cc040a72d3e49a9611dc267290e3 | 3 | 2 | 4 | 第一次提交失败后继续，取最终提交边界 |

## 产物与验证

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
源码升级后的快照一致性单独报告，不据此伪报旧receipt损坏。

新增tests/test_action_review_length.py的27项测试通过，覆盖直接/间接修改、复查返工、
临时验证、no-edit unknown/真实0、提交重试/exit、混合步、动态路径、失败写入、
历史diff、helper/通配符清理、分母、消息身份、atomic发布、防覆盖和拒绝篡改。
真实302条/14,235步运行及专属检查通过；代表性手工样例与回归一致。

相关action分类、绘图、提交审计和复查测试合跑：80 passed，退出码0。
运行后共享core/cli.py发生后续更新，因此全源码一致性标记可为false；已单独核对，
四个复查算法/统计模块仍与最终冻结快照一致，父输入与输出校验均通过。
