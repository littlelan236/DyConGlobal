# SkillsBench/OpenHands 样本级 completion-token 预算

在根 `config.yaml` 的 `execution` 中设置 `sample_completion_token_budget: 正整数`，随后通过 `dycon run create` 冻结。省略或为 `null` 时不启用；当前仅 SkillsBench/OpenHands 支持，其他 backend 配置此字段会拒绝创建。旧冻结快照保持原有哈希与行为。

预算绑定逻辑 `sample_key`，跨同一样本的 seed resample、基础设施重试和恢复保留。所有经过 bridge 的物理调用都记账，包括 `agent_decision`、`condensation` 和并行候选。压缩 prompt 不会减少累计 completion tokens。

每次调用先在 `runs/RUN/rollout/sample-token-budgets/SAMPLE.json` 加锁预留输出额度，实际 `max_tokens` 不超过可用额度，并同步 native request owner 的约束。并行候选没有可用额度时等待已预留请求结算；结算按响应中的精确 `usage.completion_tokens` 累计并释放未用额度。API attempt ID 防止重复入账。进程中断后仍有未结算预留或出现缺失 usage 时停止，不把未知用量计作零。

达到额度后不把生成响应作为可执行 action 返回，终态为 `SampleTokenBudgetExceeded`；缺失精确 usage 为 `SampleTokenBudgetUsageUnknown`。二者 disposition 为 failed，不属于 `LimitsExceeded` / `reasoning_loop`，不会换 seed 自动重采样。独立 capture 状态和 verifier 结果仍保留。

证据包含样本账本（API/role/request/attempt/预留/实际值）、物理 attempt 的 `sample-token-budget.json`、请求与 native terminal receipts 及 `attempt.json`。跨网络中断不能证明实际生成量时，账本保留 unknown；不据此宣称精确累计已完整。

本次 canary 的 1,024 与 65,536 分别为触发验收与对照参数，不是全量任务的已定标生产阈值。MIG 与整卡的墙钟时间不可直接视为同硬件 token 预算收益；三项任务也不足以代表全量 resolved 保留率。
