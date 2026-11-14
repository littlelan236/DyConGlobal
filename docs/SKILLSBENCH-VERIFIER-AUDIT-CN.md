# SkillsBench verifier 审计

在线结算和离线复查共用 `skillsbench_verifier_audit_v2` 判定核心。在线评测归档原始回执、verifier 日志、CTRF 和 subject 输出后，在写入最终评测及 attempt 回执之前审计原本可接受状态的有限数值 reward < 1。证实必需评测阶段因 verifier 故障中止时，状态改为 `VerifierError`、resolved 置空，原始 reward 不变。`ValidOutcome` 和 `Review` 保留原结算。负数进入核心并返回 `Review/invalid_reward_range`，同时保留 worker 范围校验产生的 `UnsupportedRewardPolicy` 状态。布尔值、空值和非有限数不转换为零。

`VerifierError` 表示有证据证明必需阶段失败且没有有效恢复；`ValidOutcome` 表示绑定的评测契约和计分证据支持该结果，并不代表任务成功；`Review` 表示证据、归属、顺序或契约不足以作出上述判断。部分分数和其他测试通过不能覆盖已证实的必需阶段故障。不同日志流缺乏顺序时不能推定恢复；多个 verifier 调用缺乏最终调用绑定时返回 Review。

契约适配绑定冻结任务包清单及相关源码哈希。未知或变更的契约不会借用同名任务的适配器。SimPO 参考路径和提交路径重合及 PDDL 说明与语法冲突保持 Review。归档读取只读取限制大小的普通成员，不提取、执行或跟随链接；被请求的链接和路径穿越会被拒绝。保存的发布元数据用于 AgentOps 依赖兼容性判断，历史运行时参考输入本身没有逐字节证明。

新 attempt 的诊断回执保留 `verifier_zero_audit` 字段名，Decision 的 schema version 为 2。旧零分 tuple API 必须提供冻结任务身份，且只将零分的 `ValidOutcome` 映射为旧标签 `ValidZero`。已结算 attempt 读取既有回执，不重新分类；半写入的最终评测交由现有恢复流程处理。已运行 worker 是否加载新代码取决于进程加载时机，部署或重启需遵循项目正式流程。

## 离线复查

```bash
.venvs/runner/bin/dycon skillsbench audit-rewards RUN_ID --output OUTPUT_DIR
```

命令使用冻结计划和 run status 的当前 sample/attempt 投影，复查当前 accepted 且有限数值 reward < 1 的样本。在线新判为 VerifierError 的样本不属于这个 accepted 范围；比较两个入口时必须使用相同的原始证据和候选范围。命令不运行模型、容器或 verifier，也不修改历史回执。输出目录必须是新的项目拥有目录，生成 `summary.json`、`samples.jsonl`、`samples.csv` 和 `report.md`。汇总报告 schema version 为 1，逐样本 Decision schema version 为 2。

退出码：0 表示报告完整生成，包括含 Review 的报告；1 表示运行、加载或输出错误；2 表示命令参数错误；3 表示读取不完整或扫描期间来源变化。活动 run 的结果仅代表扫描时点。日志和归档只证明已保存证据的内容，不证明未保存的参考输入或科学计算结果。
