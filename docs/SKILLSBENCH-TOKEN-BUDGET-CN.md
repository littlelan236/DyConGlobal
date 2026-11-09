# SkillsBench/OpenHands 物理 attempt 独立 completion-token 预算

在根 `config.yaml` 的 `execution` 中设置 `sample_completion_token_budget: 正整数`，随后通过 `dycon run create` 冻结。该字段沿用旧名称以兼容冻结配置，当前含义是**每个样本物理 attempt 的 completion-token 上限**。省略或为 `null` 时不启用；仅 SkillsBench/OpenHands 支持，其他 backend 配置此字段会拒绝创建。字段类型、配置规范化和 snapshot hash 算法不变，不改写历史冻结配置。

预算绑定 `run_id/task_key/sample_key/repetition_id/attempt_id`。物理 attempt 如 `rep-000-attempt-001`，不同于 `execution/attempts/004` 的 run execution attempt。初始执行、基础设施重试、seed resample、limits retry 和 `run retry-failed` 后的正式 resume，只要分配新物理 attempt ID，就获得完整额度，不继承旧 attempt 的已用 tokens、unknown 或未结算预留。新 execution attempt 本身不会重置相同物理 attempt 的预算。

同一物理 attempt 内，所有经过 bridge 的调用共享预算，包括 `agent_decision`、`condensation` 和并行候选。只累计精确 `usage.completion_tokens`；prompt tokens 不计入此上限，压缩 prompt 不会减少已用 completion tokens。

账本位于 `runs/RUN/rollout/sample-token-budgets/SAMPLE/ATTEMPT.json`，根目录通过 `run_path(cfg, run_id)` 解析，支持冻结的物理存储根。新格式为 `schema_version: 2`、`scope: physical_attempt`，binding 包含上述五个字段；锁为同目录下 `ATTEMPT.lock`，写入沿用原子替换。未知 schema、错误 scope、binding/limit 不匹配、跨 attempt 请求证据或损坏计数均报错，不按空账本重新发放额度。

每次调用先加锁预留，实际 `max_tokens` 不超过可用额度，并同步 native request owner 约束。并发请求按“上限减已用量减未结算预留”协调；额度被占满时等待结算，不会仅因存在正常 in-flight reservation 就禁止其他调用。结算按实际 completion usage 计数并释放未用预留；API attempt ID 拒绝重复预留，同一精确 usage receipt 重复结算幂等。未经本账本预留的 API ID 不得结算或登记 unknown。

同一物理 attempt 重建 bridge 时复用原账本；未结算预留继续占用额度，不能重新发放。`usage_unknown` 阻止该 attempt 后续调用，不把未知用量计作零。重建 bridge 不代表旧预留已结束，不会自动清除预留或 unknown。已经启动但中断的历史 attempt 由统一 runner 做证据恢复或分配新物理 attempt，不重新执行其模型调用；仅预留、从未启动的 attempt 可以沿用 ID 开始执行。

旧路径 `rollout/sample-token-budgets/SAMPLE.json` 的 schema 1 sample 账本仅保留为历史证据。新实现不读取、不迁移、不清空、不删除它，也不将旧用量导入新 attempt。历史 `attempt.json` outcome 和 `resolved-config.json` 不改写；旧运行的跨 attempt 累计语义不继续实施。

达到额度后不把生成响应作为可执行 action 返回，终态为 `SampleTokenBudgetExceeded`；缺失精确 usage 且没有更具体的终止原因为 `SampleTokenBudgetUsageUnknown`。二者 disposition 为 failed，不属于 `LimitsExceeded` / `reasoning_loop`，不会换 seed 自动重采样。主动重跑仍须通过原有 `run retry-failed` 和 `run resume`，满足 capture recovery、spool、退出及清理完整性约束。

明确的服务端上下文长度拒绝归为 `ContextWindowExceededError`（上下文超限），沿用现有 accepted 终态语义；accepted 表示接纳轨迹的结束结果，不表示任务已解决或用量完整。预算启用时，该失败请求仍记 `usage_unknown`，保留 `usage.complete=false`，不将缺失 usage 填为零。后续压缩或恢复若被同一 unknown 拦截，保留上下文超限主因；若后续请求成功完成，则不将此前可恢复的拒绝作为终态。并行调用仅在失败决策的所有实际候选均明确被上下文拒绝时归此类，成功或混合失败候选不据此强制分类。证据位于本物理 attempt 的 `diagnostics/context-window-error.json` 和模型调用记录，绑定 request/API attempt。正常与 finalization 异常路径保留该 agent 状态，capture、清理和取消约束仍独立生效；历史 attempt 和冻结快照不回写。

证据包括物理 attempt 账本（API ID、role、request ID、attempt ID、预留及实际值）、物理 attempt 的 `sample-token-budget.json` 停止 receipt、模型调用的 `sample_budget`、native terminal receipts 及 `attempt.json` 中的 `sample_token_budget` 快照。预算快照和调用摘要携带 scope 与 binding，剩余额度属于该物理 attempt；运行报告可另行统计所有历史 attempt 的总模型消耗，该总量不作为单个 attempt 的预算消耗。finalization 失败时也保留预算快照。

本修改不处理服务端 262,144 context 上限。过长输入仍可能被 HTTP 400 拒绝，context preflight 和调整 `max_tokens` 是独立任务。跨网络中断不能证明实际生成量时保留 unknown，不据此宣称精确累计完整。

历史 canary 的 1,024 与 65,536 是触发验收与对照参数，不是生产阈值。MIG 与整卡的墙钟时间不可直接视为同硬件 token 预算收益；三项任务不足以代表全量 resolved 保留率。
