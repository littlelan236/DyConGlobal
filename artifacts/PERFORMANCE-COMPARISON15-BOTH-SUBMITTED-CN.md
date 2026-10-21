# Baseline / Parallel 性能对比（两侧均 Submitted 的 15 tasks）

核对时间：2026-10-20T11:24:50.707858+08:00（北京时间）。本表仅保留 Baseline 和 Parallel 最终轨迹都为 `Submitted` 的 **15 个相同 task**；不包含任一侧未提交、LimitsExceeded 或 ParallelSelectionError 的 task。

| 条目 | Baseline | Parallel（B=4） | 相对变化 |
|---|---:|---:|---:|
| submitted | 15/15（100.00%） | 15/15（100.00%） | +0.00% |
| resolved | 5/15（33.33%） | 7/15（46.67%） | +40.00% |
| 实际请求数 | 1,365 | 4,140 | +203.30% |
| prompt | 38,381,440 | 112,730,120 | +193.71% |
| completion | 774,440 | 1,498,899 | +93.55% |
| total tokens | 39,155,880 | 114,229,019 | +191.73% |
| 选择分支calls | 1,365 | 1,035 | -24.18% |
| 选择分支prompt | 38,381,440 | 28,182,530 | -26.57% |
| 选择分支completion | 774,440 | 279,144 | -63.96% |
| 选择分支total tokens | 39,155,880 | 28,461,674 | -27.31% |
| Parallel cache-miss prompt（全部分支） | 不可得 | 1,737,656 | 不可计算 |

`相对变化` = `(Parallel - Baseline) / Baseline`。`submitted` 和 `resolved` 的分母都是 15，因此这里的 resolved 只表示该共同 Submitted 子集中的 harness 结果，不是原 27-task cohort 的总体成功率。

Parallel 的 15 个 task 全部保留最终 Submitted 轨迹。该子集中的 Baseline resolved 为 5，Parallel resolved 为 7；null 结果仍保留为未完成 harness 判分，不计作 unresolved。

Parallel 的 `实际请求数`、prompt、completion、total tokens 累计全部 B=4 候选分支；选择分支四项只累计最终选中分支。Parallel cache-miss prompt 按候选分支逐项计算 `prompt_tokens - num_cached_tokens`，本子集为 **1,737,656 tokens**，总分支 prompt 为 112,730,120，缓存命中 prompt 为 110,992,464。

Baseline 轨迹没有逐请求 `num_cached_tokens`，因此 Baseline cache-miss prompt 及差值不可得；未用全批次 prefix-cache hit rate 外推。

数据来自 `comparison27-with-resolved.json` 的逐 task canonical usage 和 `cache-miss.raw.txt` 的分支级 cache receipt。原始 27-task 表仍保留在 `PERFORMANCE-COMPARISON27-CN.md`。

选择分支 completion 降幅的原因已另行核查，见 [共同 Submitted 任务的 reasoning 循环分析](REASONING-LOOPS15-CN.md)：重复输出差额约占降幅的 98.2%～98.4%；剔除 6 条已确认循环响应后，completion 仅下降 1.33%，非循环 reasoning 增加 13.97%。本表保留原始 usage，不扣除循环开销。
