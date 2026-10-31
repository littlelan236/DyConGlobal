# Action Shannon 熵动态 estimator 筛选与可配置共识代表：修改计划

日期：2026-10-27（Asia/Shanghai）。状态：功能已实施并完成不占 GPU 的单元/mock 集成验证；未启动 GPU run，未修改项目根运行配置。使用方式见 [Run 独立策略说明](ACTION-ENTROPY-RUN-POLICY-CN.md)，完整验收与既有问题见 [实施调试报告](ACTION-ENTROPY-IMPLEMENTATION-STATUS-20261027.md)。

本版依据最新要求修订：沿用当前 canonical action identity；组内代表策略及三种熵映射均可选择；新增策略参数是每个 run 的独立配置，不放入项目根 `config.yaml`。先前“新增逐字身份、强制最短代表、首版只实现 B”的提议作废。

核查工程：`/home/wangtianyi/DyConGlobal`，服务器 `10.249.187.9`。核查时 HEAD 为 `c59abd2effef9437addf4f1c035e38f1cb319036`，工作树已有大量未提交变更；以下判断以读取到的工作树源码为准，不能仅凭 HEAD 复现。本地 `source_snapshot/` 和 `source-manifest.json` 保存本次关键源码及 SHA-256。

## 1. 已确认需求与交付边界

1. 共识分组及 action 类别判定沿用当前 `core/action_identity.py::canonical_action`；reasoning 不参与 action 分类或投票。不新增逐字身份协议，也不改变现有参数 JSON 规范化行为。
2. 获胜 action 组内的实际分支选择由每个 run 的 `consensus.representative_policy` 控制，可选 `estimator_score`：按 `(estimator_score, branch_id)` 升序；或 `min_step_tokens`：按当前 step reasoning + action 生成 token 总数最少选择。
3. 新增 rollout 模式，名称严格使用用户给出的 `parallel_enthophy`。动态筛选后继续 action 共识投票，再按本 run 配置的代表策略选择实际分支；不强制新模式使用最短代表。
4. 使用原始 Shannon 熵 `H = -Σ p_i ln(p_i)`，单位 nats。**不使用现有的 `H/ln(K)`，本方案也不使用 `H/ln(N)`。**
5. 首版实现本文 A/B/C 三种熵到保留数量的映射，通过每个 run 的 `entropy.retention_rule` 选择；熵大时扩大 estimator 保留集合，熵小时缩小，尽可能不增加可调数值超参数。
6. 新增策略参数及新模式的选择通过正式 run 创建入口接收并冻结到本 run 的独立配置；不写项目根 `config.yaml`，不以修改共享根配置作为切换实验的步骤。固定比例、最低保留数等既有参数也允许在该 run 的专属输入中显式设置。
7. 后续授权的实现及 CPU 调试已经完成；不变更已有 run、快照、权重或正在运行的服务。GPU 资源紧张，按用户要求不启动 GPU run；真实 GPU 验收与效果实验尚未执行，不声称得到性能收益。

## 2. 当前代码的实际行为

| 位置 | 已核实行为 | 对修改的影响 |
|---|---|---|
| `core/config.py:311–351` | 共识配置含 `retention_ratio`、`min_retained_branches`；mode 只接受现有三种值 | 增加 run 专用策略 schema 与有效配置校验，根 YAML 不承载新增策略参数 |
| `rollout/consensus.py:13–58` | estimator **分数越小越优**；按 `(score, branch_id)` 升序，保留 `max(m, ceil(r*N))` 个 | 当前“固定阈值”实际是固定保留比例，不是固定绝对 score cutoff |
| `rollout/consensus.py:71–91` | 仅对保留集合解析 `canonical_action` 并分组 | 新模式必须在 estimator 截断前获取全部候选 action，才能计算不循环依赖的熵 |
| `rollout/consensus.py:121–150` | action 组先比票数；并列时比组内最佳 score、平均 score、最小 branch ID | 保持组间并列规则，避免把“组内最短”误用为“全局最短” |
| `rollout/consensus.py:131–134` | 获胜 action 组内目前按 `(score, branch_id)` 选代表 | 保留为 `estimator_score`，新增可选 `min_step_tokens`，均限于获胜组 |
| `core/action_identity.py:27–54` | 当前解析参数 JSON 后规范化，忽略键顺序与部分表示差异 | 完整沿用该行为，共识与熵统计调用同一身份函数 |
| `action/analyzers/branches.py:150–179` | 原始熵使用自然对数；另有归一化熵与 `exp(H)` | 沿用当前 action 身份与原始熵定义，不使用归一化熵 |
| `rollout/decision/coordinator.py:94–101, 266–371` | 当前实际调度与选枝在共享 coordinator | `rollout/parallel.py` 已是兼容包装，不能只修改旧包装文件 |
| `runtime/serving/capture.py:1069–1198` | 先收到 selected branch，再等待并核验 native receipt；随后记录真实 completion token 数 | 最短选枝需把 token 证据准备前移，不能依赖选完后才补出的字段 |
| `runtime/serving/parallel_validation.py:23–54` | 当前启动验收只识别已有选择模式 | 新模式、长度规则和动态保留集合必须能重放验证 |
| `core/cli.py::run create`、`application/execution/run_plan.py::create_run` | 正式入口支持新 run 与 frozen clone，当前没有本文拟议的专属策略输入参数；clone 覆盖有白名单 | 在统一创建入口增加 run 专属输入与快照绑定，不另建 launcher，也不能假定当前 clone 已支持任意策略覆盖 |

核查时根 `config.yaml` 的**当前选中模式是 `parallel_baseline`**；虽然配置中保留共识参数 `retention_ratio: 0.5`、`min_retained_branches: 2`，这并不代表当前所有运行都在执行共识模式。已有 run 应始终按自身冻结配置解释。

现有“共识”实际是 plurality：选票最多的 action 组获胜，不要求严格多数，也不要求获胜 action 必须有两票；只要求总可用 action 分支满足最低数量。新方案先保留此定义，避免额外引入多数比例超参数。全体 action 不同时，仍按原组间 estimator tie-break 决胜，receipt 必须标明并列，不能称为多数一致。

## 3. 沿用当前 Action Identity

### 3.1 比较对象

直接调用现有 `core/action_identity.py::canonical_action`：按 tool call 顺序提取 `function.name` 与 `function.arguments`；字符串参数先解析为 JSON 对象，再按对象键排序形成 canonical action 和 SHA-256。参数已经是 dict 时，继续按当前函数的校验和序列化处理。

- 保留工具名称、解析后的参数值、参数字符串值内部的命令字符以及多个 tool call 的顺序。
- JSON 对象键顺序、结构格式空白、解码后相同字符的转义差异不构成新的 action 类别。
- 忽略 reasoning、tool call ID、response ID、request ID、usage、传输元数据和时间。
- 不判断命令执行效果是否等价，例如 `pwd` 与 `echo "$PWD"` 仍为不同 action。
- 参数 JSON 非法、存在重复键、缺少有效工具调用等，继续按既有 `ActionIdentityUnavailable` 路径处理，不自行制造空 action 类别。
- 非 tool-call 的终结或纠错响应，继续走各适配器的原有终态/异常路径。

| 分支 action 对比 | 是否同一类 |
|---|---|
| 相同 action，不同 reasoning | 是 |
| 相同 action，不同 tool call ID | 是 |
| `ls -l` 与 `ls  -l` | 否 |
| 参数字符串 `{"a":1,"b":2}` 与 `{"b":2,"a":1}` | 是 |
| 参数字符串 `{"x":1}` 与 `{ "x": 1 }` | 是 |
| 参数字符串 `{"x":1}` 与 `{"x":1.0}` | 否 |
| 同样两条 tool call，执行顺序互换 | 否 |

共识分组、动态熵统计、loop escape 原有 action 指纹及离线 action 分析沿用现有共享身份函数，不引入新的 action identity 可选开关，不复制一套 action 解析/规范化实现。

### 3.2 分布与熵的统计集合

定义 `C_t` 为当前 step：完成生成、通过原有格式/终态校验、有有限 estimator score、经现有 loop escape 确认实际保留、并具有当前 canonical action 身份的候选集合。令 `N = |C_t|`，实际出现的 action 类别数为 `K`，每类数量为 `n_i`：

```text
p_i = n_i / N
H_t = -Σ(i=1..K) p_i * ln(p_i)
```

每条候选分支权重相同；不使用 estimator score 的 softmax 概率，也不使用模型逐 token 概率。reasoning、历史 step、历史分支槽位和实际未生成的候选均不进入当前 action 分布。

计算时机：**estimator 已评分、loop escape 已定案之后，estimator 按排名截断之前**。不能使用本轮动态筛选之后的 action 分布来反推本轮保留阈值，否则产生循环依赖。

可另记 `H_pre_loop_escape` 供诊断，但决策固定使用上述 `H_t`。`N=0` 时熵为 unavailable，不能记成 0；全部同 action 且 `N>0` 时才有 `H=0`。

身份与旧分析一致，但 `N` 所处的候选筛选阶段未必等于旧分析 `all` 的分母。因此只有候选集合完全一致时，旧 `entropy_all` 才能与新决策的 `H_t` 直接比较。

## 4. 三种直接以原始 Shannon 熵为输入的映射

沿用已有最低保留数 `m = min_retained_branches`，当前为 2；`N` 是实际候选数量，不是配置的 branch_count。先判定 `N >= m`，否则沿用 insufficient-branches 失败路径，不能用 clamp 凭空补分支。

所有方案都按 estimator score 升序保留前 `k` 个候选。计算得到的 `k` 是数量阈值；派生保留比例 `r_t=k/N` 仅用于配置衔接与审计，**不作为新的归一化熵指标**。

### 方案 A：有效 action 数决定保留数量

```text
D = exp(H)
k_A = min(N, max(m, ceil(D)))
```

新增可调数值超参数：0。`exp(H)` 直接解释为当前分布的有效 action 数；均匀的 K 类 action 对应 D=K。低熵时接近最低保留数，高熵时保留更多；所有 action 各不相同时保留全部 N 条。

局限：两类 action 各占一半时，不论 N 多大，只保留 2 条（m=2）。截断可能让这两条都来自同一类，不保证每类至少一条，也可能让投票退化成 estimator tie-break。更适合希望强力缩小投票集合的消融实验。

### 方案 B：指数衰减筛除量（建议首选）

```text
raw_keep_B = N - (N - m) * exp(-H)
k_B = min(N, max(m, ceil(raw_keep_B)))
```

新增可调数值超参数：0。等价解释：把超过最低保留数的 `(N-m)` 条候选作为可调部分，筛除量按 `exp(-H)` 衰减。

- `H=0`：保留 m 条。
- 固定 N、m 时，连续保留量对 H 的导数为 `(N-m)*exp(-H) >= 0`，向上取整后 k 单调不减。
- 所有 action 各不相同时 `H=ln(N)`；m>=2 时，取整后保留 N 条。
- 两类均衡时比方案 A 保留更宽的投票基础，不需要温度、斜率、熵分段点或上下限比例。

这里没有计算 `H/ln(K)` 或 `H/ln(N)`；`exp(-H)` 是从原始熵到筛除量的直接映射，不是现有归一化熵定义。

代价：中等熵已经可能保留较多候选；这是本建议的设计取向，不能在没有实验的情况下声称其 resolved rate 最优。

### 方案 C：复用原固定比例作为低熵下限

令 `r_0` 为本 run 冻结的 `consensus.retention_ratio`，既有默认为 0.5，可在 run 专属输入中显式设置：

```text
raw_keep_C = N * [1 - (1 - r_0) * exp(-H)]
k_C = min(N, max(m, ceil(raw_keep_C)))
```

新增可调数值超参数：0；但保留对已有 r_0 的依赖。H=0 时接近原固定比例，熵增大时逐渐放宽。

适合保守地从旧共识切换；局限是低熵时不能缩到原比例以下。例如 N=32、r_0=0.5 时，最低仍保留16条，不如 A/B 能充分体现低熵缩小集合的目标。

### 4.1 统一数值例子

以下为数学示例，不是当前 run 的实测结果。`N=32, m=2, r_0=0.5`，H 使用自然对数；表中先计算未舍入的 H，再取整 k。

| 各 action 类别的分支数 | 原始 H（nats） | A：有效类数 | B：建议 | C：原比例下限 | 原固定 0.5 |
|---|---:|---:|---:|---:|---:|
| 32 | 0.000000 | 2 | 2 | 16 | 16 |
| 31、1 | 0.139061 | 2 | 6 | 19 | 16 |
| 24、8 | 0.562335 | 2 | 15 | 23 | 16 |
| 16、16 | 0.693147 | 2 | 17 | 24 | 16 |
| 8、8、8、8 | 1.386294 | 4 | 25 | 28 | 16 |
| 8 类，各 4 条 | 2.079442 | 8 | 29 | 30 | 16 |
| 16 类，各 2 条 | 2.772589 | 16 | 31 | 31 | 16 |
| 32 类，各 1 条 | 3.465736 | 32 | 32 | 32 | 16 |

首版必须实现 A/B/C 全部三种，由每个 run 明确配置；B 只是建议优先实验的规则，不硬编码为唯一行为。首次选择 `parallel_enthophy` 必须在 run 专属输入明确给出 `entropy.retention_rule`，缺失则在创建时拒绝；clone 已有动态 run 可继承源 run 已冻结的规则并记录来源，不能从根配置或未版本化的代码默认值补出。A 用于强筛选消融，C 用于旧比例衔接对照；尚无实验性能结论。

“熵大保留更多”指相同 N、m 下保留数量单调不减；整数取整和边界饱和允许平台，不意味着任意两步不同 N 下，H 较大的一步绝对保留数一定更大。

### 4.2 Score cutoff 的准确表达

设候选按 `(score, branch_id)` 排成 `c_(1)..c_(N)`。动态截断结果为前 k 条，派生 score 阈值 `τ_t = score(c_(k))`。

不能仅判断 `score <= τ_t`，因为边界同分会多保留。正式判定使用排名或 `(score, branch_id)` 的字典序边界，receipt 同时记录 cutoff score 与 cutoff branch ID。熵越高，k 单调扩大，cutoff score 单调不减；与“低分优先”的方向一致。

边界浮点误差需按版本固定数值规则处理：例如 `exp(ln(4))` 在机器精度下可能略大于4，不能因此多保留一条。采用 `math.fsum`，仅对机器误差尺度内的整数边界做数值吸附，再 ceil；该容差作为固定数值实现协议，并非实验可调超参数。协议应覆盖 A、B、C，并用边界用例验证。

## 5. 获胜 Action 组内的可配置代表策略

### 5.1 两层选择顺序

1. 按固定比例或新模式动态 k 筛选 estimator 候选。
2. 按当前 `canonical_action` 分组，每条保留分支一票，取票数最多的 action 组。
3. action 组票数并列：沿用 `(组内最低 estimator score, 组内平均 estimator score, 组内最小 branch_id)` 决胜。
4. 仅在最终获胜组内，根据本 run 的 `consensus.representative_policy` 选择实际代表：

| 参数值 | 升序比较键 | 含义 |
|---|---|---|
| `estimator_score` | `(estimator_score, branch_id)` | 完整保留当前组内选枝机制；缺省行为 |
| `min_step_tokens` | `(当前 step 生成 token 总数 L_b, estimator_score, branch_id)` | 优先 reasoning + action 最短；长度相同时以 score、branch ID 确定性决胜 |

两种策略同时适用于 `parallel_consensus` 与 `parallel_enthophy`，与熵映射选择相互独立。其他非共识模式不使用该参数；若 run 输入对非共识模式显式设置代表策略或熵规则，应在创建校验时拒绝，避免配置看似生效。

不从 estimator 已排除的同 action 分支里“捞回”更短候选，也不从所有 action 中全局找最短。当前 step 之外的历史 token 数不进入 L_b。

示例：获胜 action A 有两条保留分支，b0 的 score=0.1、L=1200，b1 的 score=0.3、L=700。配置 `estimator_score` 时选择 b0；配置 `min_step_tokens` 时选择 b1。另一 action B 即使 L=100，也不能越过 action 投票直接获胜。上述数值仅是行为示例。

### 5.2 Token 口径与证据顺序

建议采用 native 完整输出长度：`L_b = len(native_receipt.output_token_ids)`，即本次 reasoning + action 的全部生成 token，包含模型输出的边界/格式 token，排除 prompt、历史消息、工具执行结果及后续 step。不能用 `usage.total_tokens`，也不能只用 reasoning token 或文本字符数。

`usage.completion_tokens` 只能在与 native receipt 一致后视为同一长度证据。不能额外叠加 `reasoning_tokens`，以免重复计数；不能把统一账单的全部分支 token 总量当作某一分支长度。

当前 native 长度是在 `record_decision` 已收到 winner 后补齐，直接在 selector 读取该字段会有时序错误。仅对 `min_step_tokens`，计划将现有 receipt 等待、身份绑定与 token 长度核验提取为共享的**候选证据准备阶段**，在选代表之前完成；最终写盘仍复核相同 native request/output SHA。不复制第二套匹配逻辑，也不改写已经选定的 winner。

`min_step_tokens` 下若获胜组内存在长度未知的分支，无法保证选出真正最短者：记录 `token_length_unavailable` 并按原故障收尾，不填0、不默默退回 estimator 选代表、不把未知分支直接剔除后声称最短。`estimator_score` 的选择键不读取长度，不额外增加前置长度依赖，但仍保留原有最终 native capture 校验。API、capture 和 deadline 异常继续使用既有统一处理路径。

最短策略缩短的是所选分支进入后续上下文的输出，可能降低后续 prompt 成本；所有候选已经生成，因此不能宣称会节省本轮已发生的生成 token。两种代表策略的物理费用、total_usage 均统计全部分支。

## 6. 实施步骤与代码修改点

### 阶段一：沿用身份并增加两种代表策略（已实现、CPU 验证通过）

- 保持 `core/action_identity.py` 行为不变，在线共识、在线原始熵和离线分析统一调用当前 canonical API；删除旧计划中逐字 identity、新 identity 版本和适配器原始字符串保真改造的需求。
- 在 `rollout/consensus.py` 抽出可复用的分组与代表选择逻辑，增加 `representative_policy` 枚举 `estimator_score` / `min_step_tokens`。组间 tie-break 不受该参数影响，组内比较键明确记录。
- 在 `runtime/serving/capture.py` 提取现有 native receipt 验证/长度准备能力，供 `min_step_tokens` 在选择前使用；两个 adapter 共用，遵守现有 cancellation、deadline、身份绑定和错误预算。
- 缺省代表策略为 `estimator_score`；旧 run 缺少新字段时按既有机制解释，不重写旧 snapshot，也不要求旧 receipt 补出新增 policy 字段。

阶段交付：同一候选池可按 run 配置切换两种代表策略，action 分组不变；长度相同的决胜可复现；旧行为和旧证据继续可验证。

### 阶段二：新增 parallel_enthophy 与动态数量策略（已实现、CPU 验证通过）

- 扩展 run 有效配置 schema 的 mode，允许 `parallel_enthophy`；对新模式要求 estimator、至少两分支、`fallback_policy=stop`、可用共识配置和显式有效映射规则。不向根 YAML 增加策略参数，也不要求修改根 YAML 来选择新模式。
- 增加一个纯数学保留规则模块（建议 `rollout/entropy.py`），首版完整实现 A=`effective_count_v1`、B=`shannon_exp_keep_v1`、C=`baseline_floor_v1` 三种枚举，接受 action 类别计数、N、m、需要时的 r_0，返回 H、k、cutoff 所需信息。不要导入离线分析的数据加载、绘图或 CLI 模块。
- 在 `rollout/decision/coordinator.py` 接入新模式：真实候选收集 → estimator 评分及原有合法性检查 → loop escape 定案 → 当前 canonical action 身份与 H → 按本 run 的规则动态 top-k → action 投票 → 按本 run 的代表策略选枝 → 统一持久化 → 执行所选 action。最短策略的 native 长度在选代表前准备好。
- `rollout/loop_escape.py::selection_failure` 对新模式共用候选可行性检查；不能让新 mode 误走普通 minimum-score 分支。可行性检查只负责实际可选集合与投票条件，不因尚未进入最短代表阶段而误报长度缺失；实际长度错误仍由长度证据阶段处理。保留原有 action/reasoning 历史指纹语义。
- 动态模式完全由规则推导 k，不再额外乘 `retention_ratio`。C 读取该 run 的 r_0；A/B 中该字段标记 unused；固定比例 `parallel_consensus` 继续使用其 run 冻结的 r_0，避免双重筛选。

阶段交付：三个映射 × 两个代表策略共6种组合均可选择且可复现，新 mode 真实进入共识路径，不新增 launcher、不修改 estimator target 或权重。

### 阶段三：每个 Run 的独立配置、证据与兼容（已实现、CPU 验证通过）

本次用户明确指定新增参数归属 run，因此下述设计替代旧计划的根 YAML 配置方式。复用正式 `run create` / `create_run` 与快照体系，不另建配置驱动的执行入口。

**已实现接口**：`dycon run create RUN_ID ... --parallel-config PATH`，传入该 run 专属的策略 YAML。以下字段已由 strict schema、正式 create/clone 入口及 mock 测试验证；创建仅冻结配置和任务，不启动 GPU。

例如某个 run 的专属输入 `experiment-configs/RUN_ID.parallel.yaml`：

```yaml
schema_version: 1
enabled: true
mode: parallel_enthophy
fallback_policy: stop
consensus:
  representative_policy: min_step_tokens  # 或 estimator_score
  min_retained_branches: 2
  retention_ratio: 0.5                    # C 使用；A/B 不使用
entropy:
  retention_rule: shannon_exp_keep_v1     # A: effective_count_v1；C: baseline_floor_v1
```

```bash
# 已实现；创建并冻结 run，不启动模型或 GPU。
.venvs/runner/bin/dycon run create RUN_ID --operation rollout \
  --task-id INSTANCE_ID --parallel-config experiment-configs/RUN_ID.parallel.yaml
```

每个 run 可以使用不同 policy/rule。实际 strict schema 的可选值与默认如下：

| Run 专属字段 | 可选值或约束 | 缺省/生效范围 |
|---|---|---|
| `mode` | `parallel_consensus`、`parallel_enthophy`；缺省继承基础 run 配置的已有 mode | 新动态模式在 run 专属输入中选择 |
| `enabled`、`fallback_policy` | 复用既有开关和 fallback 枚举；共识要求 enabled=true、fallback_policy=stop | 可在本 run 专属输入覆盖，避免为启用本 run 而修改根配置 |
| `consensus.representative_policy` | `estimator_score` / `min_step_tokens` | 缺省 `estimator_score`；仅两种共识模式使用 |
| `entropy.retention_rule` | `effective_count_v1` / `shannon_exp_keep_v1` / `baseline_floor_v1` | 首次启用动态模式须显式选择；clone 可继承已冻结规则；固定比例共识不接受此字段 |
| `consensus.min_retained_branches` | 整数，2<=m<=branch_count | 缺省继承该 run 基础配置；可在专属输入覆盖 |
| `consensus.retention_ratio` | 有限数，0<r_0<=1 | 缺省继承该 run 基础配置；固定比例共识及 C 使用，A/B 记录 unused |

不新增 action identity 选项。`representative_policy` 与 `retention_rule` 是相互独立的离散算法选择；不增加温度、斜率等数值超参数。继承基础配置时，所有值在 run 创建完成前解析并冻结，执行阶段不再读取共享配置以决定这两个参数。

创建及配置冻结的具体契约：

1. 复用现有配置/任务/selection 解析。普通新 run 以现有基础配置为起点；`--from-run` 则以源 run 已冻结配置为起点，随后应用本次明确指定的 run 专属策略；源 run 不变。clone 从动态模式显式切回固定比例共识时，在新有效配置中移除来源的动态专用 entropy 块并记录模式变更；若本次输入仍显式提供冲突 entropy 字段则拒绝。clone 继续动态模式则可继承来源已冻结规则。
2. 专属输入仅允许上述字段及 `schema_version`，拒绝未知字段。新增专用 `RunParallelPolicy` schema（位置可在 `core/config.py` 或正式配置模块），与根 YAML schema 区分；根 `config.yaml` 不新增这两个参数，误放到根 YAML 的新增策略键应明确拒绝，不能静默忽略。
3. 在 `application/execution/run_plan.py` 的同一正式创建事务中，白名单合并专属字段并校验完整有效配置，包括 estimator、capture/parallel 已有前置条件、mode/规则组合、m 范围与 stop fallback。允许的 run 专属覆盖不扩展成任意字段覆盖。
4. `core/cli.py` 将 `--parallel-config` 传给统一 `create_run`；frozen clone 分支也要显式支持此输入。当前 clone 白名单与两层 dotted-key 写入并不能直接处理新嵌套字段，需使用共用的已校验合并函数，不能另写旁路 launcher 或先创建后手改 snapshot。若 `--set` 同时涉及专属输入白名单中的相同字段，拒绝双重来源冲突。
5. 创建时把输入内容冻结到 `runs/RUN_ID/parallel-config.yaml`，记录输入 SHA-256、schema、来源和实际合并值；有效值进入同一 run 的 `config.yaml` / `resolved-config.json` 及 plan/hash 绑定。该 run 的 `config.yaml` 是独立快照，**不是项目根 `config.yaml`**。配置缺省值也须在新 run 创建时明确解析，避免不同进程隐式选择不同策略。
6. 后续修改/删除外部输入文件或更改根 YAML，不影响已创建 run。start、resume、分析和验收只读本 run 冻结值；不得在启动时重新合并输入文件或临时 `--set`。策略变更通过创建新 run，首版不新增对既有 run 的策略热修订。
7. 旧 run 没有新输入、字段或文件时，按旧格式读取并验证原 hash；选择机制仍为 `estimator_score`，固定保留比例和当前 action identity 不变。不向旧快照注入默认键、不改写旧 config/hash/receipt。clone 旧 run 时，可在新 run 里显式添加新策略并建立独立证据。
8. 两个并行创建/运行的 run 可分别使用不同代表策略和映射，不共享可变策略状态；仅复用其本来应共享的服务/资源约束。为两个 run 写不同输入不会改动根 YAML，也不会互相覆盖。

每轮 decision 至少保存：

| 层级 | 必需审计信息 |
|---|---|
| 算法 | representative_policy、固定的既有 canonical action 身份口径、entropy definition=`shannon_nats`、retention rule/version、run 策略输入 SHA/有效配置 hash |
| 输入 | 熵统计前/后阶段、原始生成分支数、eligible IDs、action-identity-unavailable IDs、loop-excluded IDs、N、K、每类数量及 canonical action/hash |
| 动态阈值 | 原始 H、raw_keep、m、最终 k、实际保留 IDs、派生 r_t、cutoff score 与 branch ID、数值边界协议 |
| 共识 | retained action votes、组间并列比较键、获胜 action 身份 |
| 代表选择 | 生效的 representative_policy、相应 score/ID 或 L_b/score/ID 比较键、selected branch；最短策略另存 length source、native request/ledger/output SHA |
| 失败 | 无 action、score 无效、长度缺失、候选不足、capture/deadline/cancel 等真实状态 |

- 扩展 `rollout/consensus.py::validate_consensus_evidence` 与 `runtime/serving/parallel_validation.py::validate_selection`，按该 run 冻结的 policy/rule 重放当前 canonical 分组、原始 H、k、top-k、投票和实际代表选择，检测篡改。不能以当前根配置或代码默认规则代替 run 已冻结的选择。
- 由 `record_decision` 在工具执行前调用统一校验；扩展 artifact verify 的现有验证链，不再实现另一套 selector。
- compact/embedded receipt 必须保留重放所需的 canonical action 身份、生效策略和相应比较键；最短策略须保留长度与其证据绑定。不能只在 full response 中保留，然后在 compact 中丢失关键输入。
- 扩展 `action/analyzers/branches.py` 及 branch-stats schema，增加 decision-pool 原始熵、动态保留数、映射及代表策略字段；身份仍是当前 canonical 协议。按 rule/policy 分组分析，保留历史指标名称和旧数据解释，不回写旧产物。
- 更新正式 mode/run-config/验收入口文档，明确根 YAML 与 run 快照的区别。已有统一 CLI 继续使用 `dycon`，不增设独立 rollout 脚本。

### 阶段四：验证与对照实验（CPU 已完成；GPU 实验按要求不执行）

纯逻辑、mock/native receipt 集成和全仓 CPU 回归已执行，实际结果及修改前基线复测见实施报告。真实 managed-serving 验收和完整 rollout 对照需要 GPU，本次按用户要求不执行。

| 验证项 | 必须满足的结果 |
|---|---|
| 当前 action identity | reasoning/ID、JSON 键顺序/格式空白不改变类别；命令字符串内部空白、实际参数值、tool-call 顺序等依当前函数判定；与既有结果一致 |
| Shannon 公式 | 全同 H=0；16/16 为 ln2；32种各一条为 ln32；不出现 H/lnK 或 H/lnN |
| 无效与缺失 | N=0 为 unavailable；N<m 明确失败；无效 score、length 截断和 native abort 不当成完整 action |
| 三种动态数量 | A/B/C 首版全部可选，分别重现第4节示例；边界整数不多保留；相同 N 下 H 增大时 k 单调不减；m<=k<=N；C 读取该 run 的 r_0 |
| 排名截断 | score 越小越优；边界同分按 branch ID；不因 `score<=cutoff` 超额保留 |
| 两种代表选择 | estimator_score 完整复现旧机制；min_step_tokens 长度优先、同长度再 score/ID；组外更短不越过投票；缺省为 estimator_score |
| 组合覆盖 | 固定比例共识×两种代表策略，以及三种动态规则×两种代表策略；改变代表策略不改变同候选池的投票分组/获胜 action |
| 长度时序 | 最短策略在选代表前已有 native 长度；长度缺失不伪造/静默回退；score 策略不增加前置长度依赖；receipt 定案后才执行工具 |
| loop escape | 新模式候选不足时沿用恢复原候选规则；动态阈值基于实际定案集合，验证重放一致 |
| 双 adapter | Mini/OpenHands 共用 selector、当前 canonical action identity 与 per-run 参数；异常/终态维持既有契约 |
| Run 专属隔离 | 不改根 YAML即可创建两个不同策略的 run；输入文件改动不影响已冻结 run；并发运行互不覆盖 policy/rule |
| 创建与 clone | 普通 create 和 --from-run 均冻结专属输入及有效值；根 YAML 误放新策略键、未知参数、冲突来源、无效组合在创建前拒绝 |
| 历史兼容 | 旧 fixed-ratio、canonical action、estimator 代表策略及 receipt/hash 不变；resume 不自动切换策略，不向旧快照补字段 |
| 反篡改 | 修改 run 专属输入绑定、生效 policy/rule、H、类别计数、k、候选身份、长度或 selected ID，验证应拒绝 |
| 完整费用 | total_usage 覆盖全部已生成分支，不把最短 winner 当全部生成费用 |

拟执行的正式检查入口：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/python -m pytest tests/test_parallel_consensus.py tests/test_parallel_loop_escape.py tests/test_action_branches.py
.venvs/runner/bin/python -m pytest tests/test_parallel_entropy.py tests/test_run_parallel_policy.py tests/test_parallel_entropy_integration.py
# 根 config-check 只校验基础配置；新增 run 专属输入须由正式 create 执行完整校验并冻结。
# 后续真实验收：通过 --parallel-config 创建独立新 run，再走现有 run start --check-only；不为切换策略修改根配置。
# 真实验收会占用 GPU/容器；本次按用户要求不执行。
```

对照建议：固定比例×score/最短两种代表；A/B/C 各自×score/最短两种代表，共8种配置。实现验收覆盖全部组合；真实实验可先选择其中一个小队列，逐步扩大。各组 action identity 完全相同，便于独立比较代表策略与熵映射的影响。

各组固定任务、初始 seed、branch_count、estimator artifact、generation、loop escape、agent/step/token 限制和 harness 条件，仅在各 run 的专属输入中改变代表策略/映射，创建并冻结不同新 run，不改根 YAML 或旧 snapshots。离线重放可验证同一候选池的决策差异，但完整轨迹会因所选 reasoning/context 变化而分叉，不能用离线重放代替真实最终效果。

观察原始 H 与 k 曲线、保留比例、winner 支持度、最短代表的 token 差异、完整 rollout token/时间、重试/终态分布及官方 harness resolved；Submitted 不作为 resolved。不要把尚无实验支持的收益写成结论。

## 7. 完成标准与建议落地顺序

完成标准：保持当前 action identity；`parallel_consensus` 与 `parallel_enthophy` 都能按 run 参数选择 score 或当前 step 最短代表；`parallel_enthophy` 首版支持 A/B/C 三种原始 Shannon 熵动态 top-k 映射；这些策略通过正式 run 创建入口独立配置和冻结，不写项目根 `config.yaml`；两个 adapter、loop escape、receipt、启动验收及 action 分析口径一致，旧 run 及 hash/receipt 可复现。

建议按“run 专属 schema/创建快照 → 两种代表策略与最短所需 token 证据 → 新模式与三种映射 → 兼容/receipt/分析 → 组合测试 → 正式小规模验收/对照”实施。首版就必须提供两个代表策略、三个映射的全部枚举，无须引入额外数值超参数。

最新实现遵循“现有 canonical 身份、代表策略可选、三种映射可选、参数逐 run 独立配置”，替代此前逐字身份及强制最短的设计。已完成 CPU 范围的实施与调试；未启动 GPU run，未执行真实 GPU 效果对照。完整结果见实施报告。
