# 复查长度自动判定：新增轨迹观察与方法建议

日期：2026-10-26；工程：`/home/wangtianyi/DyConGlobal`。

## 结论

推荐以 **命令副作用/文件角色 + 复查语义证据 + 回溯状态机** 做自动化判定。
不要直接使用“第一次出现 check/verify”或“最后一次 EDIT 后的长度”。
脚本应同时输出候选边界、证据和不确定项；证据不足时返回 unknown，不能强行输出精确长度。

当前完成了新增30条轨迹的抽样阅读、正式action分类、静态副作用原型和方法对比。
原型结果是**候选分段，不是经过人工真值验证的复查长度**，不据此发布302条总体均值。

## 1. 新查看的样本

沿用之前已确认的5个run、302条submitted范围，排除上轮已读8条，每个run再选6条，共新增30条。
按空patch/非空无效patch、短/中/长轨迹和部分unresolved情况做目的性覆盖；不是随机抽样。
新增样本共 **1,792个assistant step，长度8–212步**；包含样本清单和完整SHA绑定。

通过已有统一入口执行：

```bash
.venvs/analysis/bin/dycon action-classify review-length-extra30-20261026 \
  --source-run parallel-consensus32-dynamic-w2-s64-lowreserve-t8h-20261025 \
  --source-run verified80-parallel-baseline-20261021 \
  --source-run verified80-parallel-reasoning-fraction-gpu01-w4-s16-20261025 \
  --source-run verified80-parallel-reasoning-log1p-20261021 \
  --source-run verified80-parallel-step-linear-20261021 \
  --rollout-list temp/review_length_automation_20261026/bindings.json
.venvs/analysis/bin/dycon action-check review-length-extra30-20261026
```

正式action产物在 `action/outputs/review-length-extra30-20261026/`，30条/1792步校验通过。
补充reasoning/observation时调用现有 `read_messages`，没有另写canonical轨迹读取器。
原型只消费以上导出数据及正式action标签，不执行轨迹命令，不调用LLM或GPU。

## 2. 新的实际观察

下列step ID从0开始。“副作用边界”是静态检测到最后一次候选代码状态改动的下一步，
尚不等于语义上已确定进入复查。

| 真实案例 | 最后任意EDIT的下一步 | 副作用边界候选 | 观察与方法启示 |
|---|---:|---:|---|
| consensus / django-11848，31步，`r-ced5b8d1616d4bbe025558261b5a75a8` | 24 | 12 | 11最后改`django/utils/http.py`，22–23只是创建/修正临时边界测试；用EDIT切分会漏掉较长核查过程 |
| baseline / xarray-4966，33步，`r-51daf11ed243f99a124c3dee05e7a352` | 31 | 9 | 8的`/tmp/fix.py`实际写`xarray/coding/variables.py`；30的`/tmp/test_final.py`只验证。目录在/tmp不能决定是开发还是复查 |
| fraction / pylint-6903，19步，`r-9d94df35784ba296659aaac9d41aabaa` | 16 | 7 | 6后目标代码不再改，但8、14、15创建临时验证；9–12还要找测试、绕过无关依赖错误 |
| baseline / sklearn-25931，56步，`r-61080e184d840d6154b8a9bd01a01eac` | 49 | 49 | 23起长时间核查，48又通过`/tmp/fix3.py`清理目标实现中的冗余赋值。不能因文件名fix3位于/tmp就当成验证脚本；最终复查需重置 |
| fraction / matplotlib-25775，212步，`r-0de90e1dca62be9240180071bbe263de` | 199 | 199 | 最后阶段逐一查看多个源文件diff，再执行综合验证。复查不一定以pytest开头，也可能跨多个文件 |
| step-linear / sphinx-8595，62步，`r-5a1f671f05a0a8d333baba2a33cf8807` | 55 | 21 | 20后反复创建tests目录中的验证测试/fixture，56显式删除这些临时内容；“写tests目录=交付测试”会误判 |
| step-linear / pylint-4970，116步，`r-2378b7ccfc8595850bc6c8b17aceee9c` | 114 | 63 | 62改目标实现，113还新建综合验证脚本，且其中有subprocess/dynamic path。副作用规则应给动态效果警告，不能假装完全理解 |
| consensus / pytest-5809，8步，`r-9c65cc040a72d3e49a9611dc267290e3` | 3 | 3 | 3–4查看改后代码和diff，没有跑测试；5提交失败，6才生成patch，7最终提交。只数测试会漏掉复查，第一次提交字符串也不能当终点 |
| step-linear / django-13513，44步，`r-3656b6af70a122b9f3d0ae5da81b9c3f` | 无 | unknown | 没检测到代码改动，但大量运行验证、查历史并声称修复已存在。必须单列无改动核查，不能自动记0，也不能把LLM声称当事实 |

每条轨迹的逐step命令、reasoning、observation片段、官方action标签与静态副作用均保留在原型结果中。

## 3. 三类可实现的方法

### 方法A：简单代理，仅用作基线

可以计算：

- A1：第一次verification/completion措辞到提交的跨度。
- A2：最后一次action EDIT之后到提交的跨度。
- A3：末尾连续非EDIT区间内的RUN/INSPECT步数。

优点是实现简单、容易复现；但A1会把早期诊断中的“verify bug”计入，A2/A3会被临时测试写入打断。
反过来，长时间INSPECT/RUN也可能仍在诊断，不自动说明已经进入收尾。
这些结果应命名为proxy，不直接命名为review_length真值。

### 方法B：文件角色与副作用追踪，推荐作为自动化骨架

**B1. 确定候选交付对象。**

- 离线分析可从最终patch的`diff --git`头提取交付路径，包含交付的测试和文档。
- 最终patch为空或无效时，退回轨迹中的工作区`git diff`证据，并记录来源降级。
- 只用于离线回溯，不把这种使用最终信息的方法称为实时在线检测。
- 不把`git show COMMIT`或`git diff HEAD OTHER_COMMIT`当成当前工作区变更；不能只看有没有diff头。
- 同时关注修改后又回退、未进入最终patch的生产文件；仅盯最终patch路径仍会漏掉返工。

**B2. 为文件记录角色与生命周期。**

建议区分：交付实现/交付测试文档、临时验证脚本和fixture、修改代码的helper、patch或日志输出。
判断临时性质要组合“何时创建、内容做什么、是否执行验证、是否最终清理/不在交付diff”证据。
文件名包含test、位于/tmp、位于tests目录，都不能独立定性。
若原始记录没有tracked/untracked和创建前状态，仅能标为probable temporary。

**B3. 追踪间接修改。**

复用已有shell AST和官方action标签，再补充目标路径特征：

- 直接操作：`sed -i`、重定向、cp/mv、patch/git apply、删除目标文件。
- Python：静态识别`open(...,'w'/'a'/...)`、`Path(...).write_text/write_bytes`，传播简单字符串变量。
- helper：保存轨迹中生成的脚本内容，后续`python /tmp/fix.py`沿脚本体追踪真实写入目标。
- `git checkout/restore/reset/stash/revert`可能改变候选状态，需要边界重置。
- 动态exec、嵌套shell、subprocess、无法解析的路径和未知helper不能判为无副作用，应进入unknown/needs_review。

静态写意图不保证命令实际成功或内容真的改变：正式实现还应结合returncode、前后diff/文件hash等证据。
失败的写入不能无条件重置边界，成功但no-op的sed也不能等同真实代码改动。

**B4. 从最后一次目标状态变化后寻找复查证据。**

先得到 `candidate_stable_step = last_candidate_change_step + 1`。
这只是允许最终复查开始的最早候选位置；再用方法C确定语义起点，不能直接将整个后缀都算复查。

### 方法C：带约束的逐step标签与状态机，推荐与B组合

先给每步提取多通道特征，而非单个关键词：

| 通道 | 自动特征例子 | 作用 |
|---|---|---|
| 行为 | 看改后代码/diff、运行reproducer/pytest、创建验证脚本、查询测试运行方式 | 提供操作证据 |
| 目标对象 | 命令所读写文件是否属于候选解、相关测试、临时支持文件 | 区分诊断对象和核查对象 |
| 语义 | “fix applied/works”“verify change”“regression/edge cases”等 | 支持核查意图；单独不足以判定 |
| 观察 | 测试输出、缺模块、失败提交、空diff、clean工作区 | 校验实际发生了什么；不以LLM声称代替 |
| 时间上下文 | 是否已有候选解、后续是否继续修改、相邻步骤是否构成验证链 | 避免把前期诊断当收尾 |

一个可实现的初始判据是：已有候选解，当前及后续少量步骤出现来自“行为/对象/语义”中至少两个通道的核查证据，
且没有新的交付状态修改，则从首个有证据的step进入REVIEW。
窗口大小、证据阈值和冲突权重需要由标注样例校准；不能凭经验设置后宣称准确。
这份原型仅实现“副作用候选边界 + 逐步严格/宽松证据”，尚未训练/校准完整序列模型。

状态转移建议如下：

```text
DEVELOP -- 候选已形成且出现核查证据 --> REVIEW
REVIEW  -- 创建/修正一次性验证、调整测试环境 --> REVIEW_SETUP
REVIEW / REVIEW_SETUP -- 新的交付状态改动 --> DEVELOP
REVIEW / REVIEW_SETUP -- 打包/清理 --> 保留episode，相关步单独标记
任意状态 -- 实际终止提交 --> SUBMIT_END
未知副作用/意图冲突 --> UNKNOWN，保留可解释的候选范围
```

离线回溯取最后一次有效候选状态后的最终episode，同时保留先前episode和复查后返工次数。
第一次输出提交marker后如果轨迹还继续，属于SUBMIT_RETRY，不是终止。
一次assistant step有多条子命令时可保留多标签，主长度仍每步最多1次；开发+验证混合步单列MIXED。

对“执行验证还是继续诊断”这样的难例，可选用LLM对结构化证据做辅助标签，
但必须输出step ID、支持证据和unknown，受真实写入和提交边界硬约束，冻结模型/prompt/版本并抽样人工校验。
本次未调用任何LLM judge，也未测其准确率或费用。

## 4. 本次原型对比的结果

分析原型位于远程 `temp/review_length_automation_20261026/prototype.py`，
输入为已有正式action分析的导出记录，保持探索脚本与正式功能代码分离。

- 新增样本30条；2条没有检测到交付改动，仅有“修复已存在”的声称，返回boundary unavailable，不强制计0。
- 剩余28条有副作用边界候选；其中 **14条**与“最后任意EDIT”的下一步不同。
- 这28条中 **24条**的首次verification/completion语言命中与副作用候选边界不同。
- 30条中 **13条**不能从最终提交内容取得标准diff路径，说明仅依赖最终patch会有明显缺口。
- 有候选边界的记录中，**10条**仍存在未解析动态/嵌套效果警告，不能视为全部可靠自动完成。

这些数字表示**方法之间的差异和证据缺口，不是14/24条已被证明误判，也不是准确率**。
样本本就特意覆盖无效提交与长轨迹，不能外推为全部302条的缺失率或错误率。

原型输出字段包括：

```text
terminal_submit_step          实际终止位置候选
keyword_first_hit             简单语言代理
last_any_edit_start            简单EDIT代理
path_aware_start               最后检测到目标状态变化后的候选位置
review_anchor_step            其后首个严格核查证据位置
strict_candidate_count        严格规则命中的核查步数
permissive_candidate_count    宽松地计入未确定上下文后的候选步数
uncertain_steps / warnings    不确定step与原因
step_features                 每步保留原action、命令、reasoning、observation、静态副作用
```

strict/permissive只是规则敏感性结果，**不是统计置信区间，也不是被证明包住真值的上下界**。
原型使用有限的literal AST、目录前缀和临时文件生命周期启发式，尚不能完整模拟shell/Python执行效果。

运行原型：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/analysis/bin/python temp/review_length_automation_20261026/prototype.py \
  temp/review_length_automation_20261026/extra30-enriched.jsonl \
  temp/review_length_automation_20261026/NEW_METHOD_COMPARISON_ID
```

既有结果在 `temp/review_length_automation_20261026/method-comparison-v1/`，不覆盖运行。
8项针对helper间接修改、临时测试构造、动态效果警告、git状态变化、重定向及不执行命令的测试通过。
这些测试验证代码特征行为，不证明语义分段正确。

## 5. 建议固定的长度指标

主指标建议采用 **最终复查净步数**，辅助指标采用 **收尾跨度**：

```text
review_steps = 最终episode中 VERIFY / VERIFY_SETUP 的assistant步数
review_span  = 最终提交step - 最终复查起始step
```

清理/打包/失败提交重试可进入跨度，但不进入净核查步数；终止提交本身不计入两者。
同时保留全程所有episode的复查步数、返工次数和复查重复次数。
重复读同一文件/重复跑同一测试属于真实消耗，主成本指标不自动去重；可另报去重后行为数。

没有明确目标改动但有验证行为时，另报no-edit review；没有复查证据才是0，记录不全应是unknown。
若需要token长度，在分段后累加选中assistant request的独立reasoning或completion证据，
不重复计provider/extra.response副本，不累加未选parallel分支，不用prompt累计长度冒充本步输出长度。

下一步应先对这批30条做明确的逐step/起点参考标注，固定MIXED、测试环境排障、
无改动核查及“最后一次内容变化”规则，再评估自动脚本的起点误差、step precision/recall与unknown比例。
在此之前不把探索原型升级为正式全量统计入口。


## 后续实现与最终运行

用户已授权按本计划实现和运行。正式规则入口、27项新增测试、302条submitted全量结果及局限
见[复查长度分析](REVIEW-LENGTH-CN.md)；最终结果为review-length-submitted302-final-20261026。
本文保留此前探索过程，原型差异数字不作为最终总体统计。
