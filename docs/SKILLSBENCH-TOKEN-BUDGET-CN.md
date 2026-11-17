# SkillsBench/OpenHands 物理 attempt 主轨迹生成预算

在根 `config.yaml` 的 `execution` 中设置 `sample_completion_token_budget: 正整数`，通过 `dycon run create` 冻结。省略或为 `null` 时不启用；仅 SkillsBench/OpenHands 支持。字段名称、类型、配置规范化和 snapshot hash 算法不变，不改写历史冻结配置。2026-11-14 起，并行运行的该上限约束**主轨迹累计 completion tokens**，与单分支 baseline 对齐：包含已选 reasoning+action 和独立 condensation，排除 prompt 以及未选候选。它不是 context window 上限，也不是全部候选的算力预算。

预算绑定 `run_id/task_key/sample_key/repetition_id/attempt_id`。物理 attempt 如 `rep-000-attempt-001`，不同于 run execution attempt。初始执行、基础设施重试、seed resample、limits retry 和正式 `run retry-failed` 后的 resume，只要分配新物理 attempt ID，就获得完整额度，不继承旧 attempt。仅新建 execution attempt 不会重置同一个物理 attempt。

baseline 保持原有 schema 2 和逐物理调用结算，包括 condensation、单分支生成及其重试。并行配置使用 schema 3、`scope: physical_attempt`、`accounting: selected_trajectory_completion_v1`：每个物理调用的 usage 都记录在 `requests`，但 agent_decision 候选只有最终选中的一条才设置 `charged: true`；独立 condensation 仍直接扣费。`completion_tokens`、`by_role` 和剩余额度仅统计 charged 项。费用、attempt `usage` / `output_tokens`、coordinator `total_usage` 继续表示全部分支实际物理消耗，可能远大于主轨迹预算；这些字段不再用于解释主轨迹剩余额度。

账本位于 `runs/RUN/rollout/sample-token-budgets/SAMPLE/ATTEMPT.json`，通过冻结物理存储根解析；锁为同目录的 `ATTEMPT.lock`，沿用原子替换。旧 schema 1 sample 账本和 schema 2 历史并行 attempt 保留，不迁移、不重置、不将总消耗解释为已选长度。用新并行实现打开同一旧 schema 2 attempt 会拒绝；正式恢复应保留旧证据并分配新物理 attempt。禁止手改账本或历史 `attempt.json`。

并行候选都可以使用“主轨迹上限减已选累计值”作为各自的输出额度，实际 `max_tokens` 不超过该值及配置的单次上限，native request owner 同步该限制。互斥候选不会因为其他候选的预留而共享瓜分额度。生成完成先保留实际 usage；统一选择器选定后按 request/API attempt 精确绑定，仅结算选中的那条。重复结算幂等，同一逻辑请求的不同胜者、跨请求/attempt 结算、损坏计数或错误 schema/accounting 都拒绝。

未选候选缺失 usage 时保留其 unknown 证据与整体 `usage.complete=false`，但不据此阻止已知 usage 的胜者。选中响应缺失精确 usage，或独立 condensation 的 usage 未知，则阻止该 attempt 后续调用并以 `SampleTokenBudgetUsageUnknown` 停止；不填零。单分支 baseline 的 unknown 行为保持不变。

选中响应使主轨迹额度耗尽时，保留结算与停止 receipt，但不将该响应作为可执行 action 返回，终态 `SampleTokenBudgetExceeded`。若某个候选因主轨迹剩余额度限制而以 length 结束，它作为预算受限候选退出；较短的有效胜者仍可继续。所有候选均因此失败时保留预算停止原因及 receipt。单次 generation 自身的上限、reasoning loop、选择失败和实际 context window 拒绝继续分别处理。

`SampleTokenBudgetExceeded` 和 `SampleTokenBudgetUsageUnknown` 的 disposition 为 failed，不属于 `LimitsExceeded` / `reasoning_loop`，不自动换 seed 重采样。主动重跑使用正式 `run retry-failed --exit-status SampleTokenBudgetExceeded` 再 `run resume`；先停止并确认退出/清理，检查并恢复已有 capture，保留旧轨迹和账本。禁止覆盖冻结配置或手改 run 状态。

真实服务端 HTTP 400 上下文长度拒绝归为 `ContextWindowExceededError`，保留 `diagnostics/context-window-error.json` 和模型调用证据。仅失败决策的所有实际候选均明确被上下文拒绝时归此类；恢复成功或混合失败不强制分类。accepted 表示接纳轨迹结束，不表示解决任务。正常及 finalization 异常路径保留主因，capture、清理和取消约束独立生效。

同一物理 attempt 重建 bridge 复用其匹配 schema 的账本；不能以重建 bridge 清除 unknown 或重新发放额度。已经启动但中断的 attempt 由统一 runner 恢复证据或分配新 attempt，不重新执行旧调用；仅预留、从未启动的 attempt 可以沿用 ID 开始执行。

停止证据包括 `sample-token-budget.json`、attempt 账本、模型调用 `sample_budget` 和选择后的请求级 `sample_budget`、native terminal receipts、`attempt.json` 的 `sample_token_budget`。并行 ledger 的 schema/accounting 明确标识新口径；旧产物不回写。

本预算不替代 `serving.max_model_len`。实际 context window 约束单次 prompt+completion，压缩可缩短后续 prompt，但不减少已经累计的 completion。context preflight 和服务端输出余量处理属于独立约束。不同硬件与墙钟时间不能直接据此比较任务质量。
