# 每个 Run 独立的 Action 熵与共识代表策略

`parallel_consensus` 继续按固定比例筛选；新增 `parallel_enthophy` 使用当前 step 的**原始 Shannon 熵**动态确定 estimator top-k。两个模式均可选择获胜 action 组内的实际分支。沿用 `core/action_identity.py::canonical_action`，不改变 action 身份口径，不考虑 reasoning 来划分类别。

## 创建与冻结

为目标 run 准备独立输入，如 `experiment-configs/entropy-b-shortest.yaml`：

```yaml
schema_version: 1
enabled: true
mode: parallel_enthophy
fallback_policy: stop
consensus:
  representative_policy: min_step_tokens
  min_retained_branches: 2
  retention_ratio: 0.5
entropy:
  retention_rule: shannon_exp_keep_v1
```

通过正式入口创建；创建只冻结配置及任务，不启动 GPU、模型或容器：

```bash
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout \
  --task-id INSTANCE_ID --parallel-config experiment-configs/entropy-b-shortest.yaml
```

复制来源 run 的完整配置与任务，再为新 run 指定策略：

```bash
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout --from-run SOURCE_RUN \
  --repeat-count SOURCE_REPEAT_COUNT \
  --parallel-config experiment-configs/entropy-b-shortest.yaml
```

clone 保留 source 的 repeat_count，不同时指定 task/partition/tasks-file。不传新文件时可继承动态 source 的冻结策略；首次选择动态模式必须显式提供映射规则。切回固定比例模式时，输入中删掉 `entropy`，创建器会移除新 run 继承的动态专用块，source 不变。

输入原样冻结到 `runs/NEW_RUN/parallel-config.yaml`；有效参数进入本 run 的 `config.yaml`、`resolved-config.json` 和 plan/config hash；`parallel_policy` 绑定输入 SHA 与生效选择参数 SHA。**不向项目根 `config.yaml` 写入新模式、representative_policy 或 entropy**；根 schema 拒绝这些新增配置。

start/resume/analysis 只读本 run 快照，不重新读取外部策略文件或临时更改参数。更换策略创建新 run。旧 run 缺少新字段时仍使用原固定比例及 estimator-score 代表，不补写旧配置、hash 或 receipt。

共享模型、capture、estimator artifact、容量等仍须满足现有前提。新输入仅接受上例字段：schema_version、enabled、mode、fallback_policy、consensus 的三个参数和 entropy.retention_rule，以及下述新映射专用的 entropy.full_retention_threshold。拒绝未知字段、重复 YAML 键、null、冲突的 --set 来源及无效组合，不开放任意模型/任务/seed/资源覆盖。

## 获胜组内代表策略

| consensus.representative_policy | 获胜 action 组内升序比较键 |
|---|---|
| `estimator_score`（缺省） | `(estimator_score, branch_id)` |
| `min_step_tokens` | `(当前step生成token总数, estimator_score, branch_id)` |

先选最多票 action 组；组间并列仍按组内最低 score、平均 score、最小 branch ID 决胜。代表策略只改变获胜组中的实际分支，不改变组间规则，不捞回已过滤分支。

最短策略使用 `len(native_receipt.output_token_ids)`，包括本次 reasoning+action 的输出边界/格式 token，排除 prompt、历史、工具返回和后续 step。统一 receipt 验证在选代表前取得长度，提交时再次核验；缺失或矛盾不填0、不静默退回 score。score 策略不增加前置长度等待。费用和 total_usage 始终覆盖全部生成分支。

## 原始熵与映射

统计集合为通过原合法性检查、有有限 score、经 loop escape 最终保留并具备 canonical action 的当前候选池。设实际数量为 N，每类数量 n_i、p_i=n_i/N：`H=-sum(p_i*ln(p_i))`，单位 nats。**不使用 H/ln(K) 或 H/ln(N)**；在 estimator 排名截断前计算，不用筛选后集合反推自身阈值。N=0 的熵为 unavailable；N<m 为候选不足。

令 m 为本 run 的 min_retained_branches，r_0 为 retention_ratio：

| entropy.retention_rule | raw_keep | 特点 |
|---|---|---|
| `effective_count_v1`（A） | `exp(H)` | 有效 action 类数，筛选较强 |
| `shannon_exp_keep_v1`（B） | `N-(N-m)*exp(-H)` | 筛除量指数衰减 |
| `baseline_floor_v1`（C） | `N*(1-(1-r_0)*exp(-H))` | 原比例为低熵下限 |

最终 `k=min(N,max(m,ceil(raw_keep)))`。在整数边界吸附至多16个 machine ULP，避免 exp/log 舍入使整数保留量无故多一条。A/B 不使用 r_0，C 使用；动态模式不额外乘固定比例。固定 N、m 时三种映射均对 H 单调不减。

按 `(score, branch_id)` 升序保留前 k 条，score 越低越优；边界同分按 branch ID，不能仅用 score<=cutoff 而超额保留。两种代表与三映射可独立组合；固定比例共识另有两种代表组合。

## 证据与检查

### effective_count_full_above_v1（2026-10-31）

新增映射先完整计算 `effective_count_v1` 的最终整数分支预算
`base_keep_count=min(N,max(m,stable_ceil(exp(H))))`。仅当该预算**严格大于**
`entropy.full_retention_threshold`（默认 10、可配置正整数）时，改为保留候选池全部 N 条。
本实验 branch_count=32，完整合法池触发时保留全部 32 条；无效或 loop escape 排除的分支不会被恢复。
否则与 `effective_count_v1` 一致；候选不足仍按原规则报告不可用。
例如原 k=10 保留 10，原 k=11 改为 32。比较发生在取整、最小值和候选数量裁剪之后。

```yaml
entropy:
  retention_rule: effective_count_full_above_v1
  full_retention_threshold: 10
```

参数仅用于此映射，通过 `--parallel-config` 写入新 run 快照及策略 SHA，旧映射不接受它。
decision entropy 元数据新增 `base_keep_count`、`full_retention_threshold` 和
`full_retention_triggered`；`raw_keep` 仍为 `exp(H)`，`keep_count` 为最终预算。
协调器、loop escape 可行性检查和 full/compact evidence 重放均使用冻结阈值。

decision 记录候选池、canonical 类别、原始 H、规则、raw_keep、k、cutoff、保留 IDs、投票、代表比较键与 run 策略绑定，compact receipt 保留重放输入。执行工具前完成 native 证据与选择校验；启动验收及 artifact verify 按本 run 冻结规则重放。

action-branch-stats 增加 decision_pool_entropy、decision_keep_count、representative_policy、retention_rule，可按策略分组。旧分析的指标定义不变。策略 SHA 不包含可独立审计修订的 admission/timeout 资源项；正式离线 serving-memory/concurrency 修订不改变选择策略，完整 config hash 仍按原修订链更新。

```bash
CUDA_VISIBLE_DEVICES='' .venvs/runner/bin/python -m pytest -q \
  tests/test_parallel_entropy.py tests/test_run_parallel_policy.py \
  tests/test_parallel_entropy_integration.py tests/test_parallel_consensus.py \
  tests/test_parallel_loop_escape.py tests/test_native_receipt_contract.py
```

`run start --check-only` 会启动 managed serving，仍需要 GPU。此次实现调试不执行该入口，不启动 GPU rollout/harness run。


## Effective count 加法偏移映射（2026-11-01）

Run 独立策略新增 `effective_count_plus_n_v1`，必须显式配置非负整数 `entropy.n`：

```yaml
entropy:
  retention_rule: effective_count_plus_n_v1
  n: 6
```

先按 `effective_count_v1` 完整计算整数预算 `k_base`（沿用原始 Shannon 熵、
16 ULP 整数边界处理和 `min_retained_branches` 下限），再取
`k = min(N, k_base + n)`。`N` 为 loop escape 后、estimator top-k 前的有效分支数。
有效分支不足原下限时仍保留原有 insufficient/unavailable 行为，不用偏移量补足。
例如 N=32、全部 action 相同、下限为 2 时，n=6 保留 8；k_base=30 时保留 32。
此规则不使用 `full_retention_threshold`，切换映射的 clone 会移除不适用的旧字段。

输入和 n 随新 run 冻结；选择、loop escape 可用性检查与证据回放都使用同一个偏移量。
entropy 证据额外记录 `base_keep_count`、`n`，`raw_keep` 保留原 exp(H)。
旧 run 的快照、缺少此映射的选择行为均不补写。


冻结 clone 同时允许 `--set parallel.max_inflight_branches=CAPACITY`。
该值不得超过新 run 的 `serving.max_num_seqs`；当候选分支总数大于容量时，
仍须启用 `dynamic_branch_admission`。例如 32 个候选分支、max_num_seqs=24 时，
设置 max_inflight_branches=24，通过已有动态准入机制分批运行，不改变候选池大小。
