# 中断轨迹清理记录（2026-10-17）

已从 DyConGlobal 的唯一 manifest 和轨迹/补丁目录清理 96 条未完成检查点，旧工程原始数据保持不变。清理前 1,204 条，清理后 1,108 条；258 个 task 仍全部为 train，保留轨迹全部为 training。

## 清理依据

| 类型 | 数量 | 证据 |
|---|---:|---|
| 明确中断的 attempt | 64 | 原始轨迹 exit_status 为空，对应旧 attempt.json 标为 Interrupted |
| 停止或暂停 run 的未结束轨迹 | 28 | exit_status 为空、没有最终 attempt 记录，run 状态为 stopped_by_user 或 paused |
| 无进程的遗留检查点 | 4 | exit_status 为空、没有最终 attempt 记录，文件已超过一天未更新，旧工程没有对应运行进程 |

最后 4 条可以确认是未完成且已无人运行的检查点，但没有足够日志区分 SIGKILL、服务退出等具体原因，不将其标为已证实 SIGKILL。所有被清理轨迹最多只有 195 次 assistant 调用，低于配置的 250 步上限；均未提交、未经评测，不含任何 resolved 轨迹。

## 保留的轨迹

保留全部 154 条 LimitsExceeded、76 条 ContextWindowExceededError、872 条 Submitted 以及 2 条 RepeatedFormatError。

另有 4 条 ContainerCorrupted：容器实际命令为 sleep 28800，运行时间分别约 8 小时 2 分、8 小时、8 小时 31 分和 9 小时 12 分，随后报告容器已 exited。它们具备容器寿命到期的时间证据，因此按“不清理达到上限的轨迹”的要求保守保留，没有仅凭 ContainerCorrupted 就删除。

清理后互斥结果分类：resolved 321、submitted（尚未确认成功）551、unsubmitted 236。当前来源映射为 1,779 个文件对应 1,108 条唯一轨迹。成功评测标签和所有保留轨迹的记录保持不变。

## 文件与实现

唯一当前目录为 baseline/manifest.json。清理脚本为 baseline/scripts/clean_interrupted.py，默认 audit 只生成计划，apply 在确认 manifest 和数据哈希未变化后执行。

审计目录：baseline/provenance/interrupted-cleanup-20261017/。

- manifest-before.json：清理前元数据备份，不包含原始轨迹副本。
- plan.json：逐条清理理由、原始 run/attempt 证据及哈希。
- removed-trajectories.csv：96 条已清理轨迹的 task-id、run-id、attempt-id、rollout-id 和理由。
- retained-container-lifetime-evidence.json：4 条保留容器轨迹的配置命令和时间证据。
- receipt.json：清理结果、文件校验及导出哈希。
- trajectory-index.csv、source-occurrences.csv：从当前 manifest 导出的清理后只读索引。

原始迁移目录 tact-report-20261017 下的计划、回执和 CSV 保持为迁移当时的历史快照；其中的 1,204 条不是当前可用数量。运行和训练仍只读取唯一 manifest。

验证：15 项相关测试通过，涵盖上限保护、旧 sidecar 冲突保护、已提交轨迹删除拒绝、实际文件删除、保留轨迹不变及重复执行。清理后的 Catalog 全量轨迹、补丁和报告哈希校验通过，manifest revision 为 1236。所有 96 条已从 manifest 移除，轨迹和补丁文件均已删除；保留的 1,108 条元数据与清理前一致。
