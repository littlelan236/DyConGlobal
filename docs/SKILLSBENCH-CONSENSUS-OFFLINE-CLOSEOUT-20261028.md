# 两项验收收尾

日期：2026-10-28（北京时间）。

本次授权范围已完成：真实 MIG consensus 路径验收、首批两个任务的离线安装及官方 oracle/verifier 验收、资源清理复核与报告同步。

| 检查 | 结果 |
|---|---|
| Consensus run | skillsbench-consensus-mig7-offer-20261028：complete / exit 0 |
| 全 run audit | verified=true / whole_run_acceptance=true |
| Agent 任务成绩 | reward=0 / resolved=false，兼容通过不等于解题成功 |
| Consensus 数据 | 2 decisions / 4 candidates / 4 independent score recomputations / max error 0 |
| 真实投票路径 | 两次均为 1:1 平票，走冻结的 estimator_score tie policy |
| Losing tool calls | 2 个均未执行 |
| Offline oracle/verifier | dialogue-parser、offer-letter-generator 均 reward=1.0 |
| 真正网络隔离 | NetworkMode=none，仅 loopback，无联网 fallback |
| 安装范围 | 207 个锁定包、SDK/ACP import 与 CLI help，308 个快照文件校验 |
| 资源清理 | 所属进程、容器、网络均无残留 |
| 当前 runner 回归 | 1120 passed, 67 skipped，102.86 秒 |
| 当前 SkillsBench 专项 | 68 passed, 3 skipped，3.62 秒 |

模型通过 MIG-b13c6595-2246-526e-8a72-1c5f59a014e8、TP=1、KV4GiB 运行。详细参数、receipt 和限制见 SKILLSBENCH-CONSENSUS-OFFLINE-ACCEPTANCE-20261028.md。

离线预置 uv0.13.5 是登记的环境变体；不声称原始在线安装链逐字节相同。该验收不包含无网络 LLM rollout、冷启动 apt 构建、其他平台或全部 SkillsBench 任务。

收尾仅复核既有终态和证据，不重新训练 predictor、不修改旧 run/评分/receipt、不启动新的 workload。
