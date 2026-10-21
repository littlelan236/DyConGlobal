# Baseline / Parallel 性能对比（相同 27 tasks）

核对时间：2026-10-20T11:09:17.605318+08:00（北京时间）。本表对应 parallel 最终状态为 21 Submitted + 6 LimitsExceeded 的 27 个相同 instance_id。原 31-task 中另外 4 个 ParallelSelectionError 已在两侧同时排除；表中不包含后续 36-task。

| 条目 | Baseline | Parallel（B=4） | 相对变化 |
|---|---:|---:|---:|
| submitted | 19/27（70.37%） | 21/27（77.78%） | +10.53% |
| resolved | 7/27（25.93%） | 7/27（25.93%） | 0.00% |
| 实际请求数 | 3,683 | 12,324 | +234.62% |
| prompt | 156,768,600 | 509,431,796 | +224.96% |
| completion | 2,429,142 | 6,574,155 | +170.64% |
| total tokens | 159,197,742 | 516,005,951 | +224.13% |
| 选择分支calls | 3,683 | 3,081 | -16.35% |
| 选择分支prompt | 156,768,600 | 127,357,949 | -18.76% |
| 选择分支completion | 2,429,142 | 814,519 | -66.47% |
| 选择分支total tokens | 159,197,742 | 128,172,468 | -19.49% |
| Parallel cache-miss prompt（全部分支） | 不可得 | 5,220,836 | 不可计算 |

`相对变化` = `(Parallel - Baseline) / Baseline`。对 `submitted` / `resolved` 也按计数相对变化计算；两者分母都是 27，因此其相对变化同样等于 cohort 成功率的相对变化。`submitted` 不代表 patch 有效或通过测试。`resolved` 是已有官方 harness 证据确认的成功数量。Baseline 判分 16 个（7 true、9 false、11 null），Parallel 判分 18 个（7 true、11 false、9 null）；两侧有效判分集合不同，不能据此宣称完整成功率相同。Parallel 的 9 个 null 包括 7 个 MissingPatch 和 2 个 patch apply 失败。

`实际请求数`、`prompt`、`completion`、`total tokens` 在 Parallel 一侧累计全部 B=4 候选分支，而后四项只累计最后选中轨迹。Baseline 为单分支，相应 token 字段重复列出以保持对比口径。Prompt 是请求 usage 中报告的输入长度，包含缓存命中的输入，并非实际重新 prefill 的 token 数。

`Parallel cache-miss prompt（全部分支）` 按 12,324 个分支逐个累加 `prompt_tokens - num_cached_tokens`，精确为 **5,220,836 tokens**；缓存命中 prompt 为 504,210,960。Baseline 的逐响应轨迹没有 `num_cached_tokens`，只有整次 258-task serving 的全局 prefix-cache hit rate，不能分离出本表这 27 个匹配 task 的缓存未命中量，所以 Baseline 和差值不可得，不拿全批次 hit rate 外推。

`选择分支calls` 表示逻辑模型 query 次数，而非工具执行次数：Baseline 为 model_stats.api_calls，Parallel 为去重 decision_id 数。Baseline 的 3,683 次调用中 3,681 条有返回 usage，另外 2 次上下文超限调用没有 token usage；token 统计仅使用已有 usage，不作估算。此前单独记录的工具调用为 Baseline 3,663、Parallel 3,081，不能用来替换本表的模型 calls。

主表只统计这 27 个 task 的最终 attempt。Parallel 另有 5 次 ContainerCorrupted 失败重试，未混入主表；若计入它们，实际请求数为 14,040，prompt 为 576,637,204，completion 为 14,746,777，total tokens 为 591,383,981。

数据来源：

- Baseline run：`baseline-qwen35-9b-forward-apc-coord-gpu4-7-w12-20261012`，精确匹配这 27 个 instance_id。
- Parallel run：`parallel-ridge-event-b4-w12-disjoint31-20261019`，按 progress.json 的最终 rollout_id 选择。
- Token/calls：canonical 轨迹的顶层 response.usage 或 dycon_parallel.total_usage，分别按 response ID / decision ID 去重；后者已与候选分支逐项求和核验。
- Parallel harness：`parallel-ridge-event-b4-w12-disjoint31-final27-harness-20261020` 加 `parallel-ridge-event-b4-w12-disjoint31-retry6-harness-20261020`，通过统一 prepare-evaluation / evaluate 入口执行，6 条 Docker 连接重置在单 worker 重试后全部判分完成。
- 逐任务指标：[comparison27-with-resolved.json](comparison27-with-resolved.json)。评测汇总：[final27-harness-combined-20261020.json](final27-harness-combined-20261020.json)。

Baseline 来自多轮历史配置，dtype、并发、显存比例和 seed 与当前 Parallel 不同；本表为历史实测对照。
