# Baseline / Parallel 性能对比（31 + 33 tasks）

核对时间：2026-10-20T22:44:56.182991+08:00（北京时间）。本表保留 31-task run 的全部 31 个任务，以及 36-task run 中已完成的 33 个任务，合计 64 个互不重叠 task；不排除 ParallelSelectionError。36-task run 中尚未完成的 ValueError 与 pending task 不计入 33 个已完成任务。

| 条目 | Baseline | Parallel（B=4） | 相对变化 |
|---|---:|---:|---:|
| submitted | 49/64（76.56%） | 49/64（76.56%） | +0.00% |
| resolved | 23/64（35.94%） | 18/64（28.12%） | -21.74% |
| 实际请求数 | 7,218 | 25,252 | +249.85% |
| prompt | 318,131,533 | 1,130,169,380 | +255.25% |
| completion | 5,060,294 | 14,916,616 | +194.78% |
| total tokens | 323,191,827 | 1,145,085,996 | +254.31% |
| 选择分支calls | 7,218 | 6,313 | -12.54% |
| 选择分支prompt | 318,131,533 | 282,542,345 | -11.19% |
| 选择分支completion | 5,060,294 | 1,983,542 | -60.80% |
| 选择分支total tokens | 323,191,827 | 284,525,887 | -11.96% |

`相对变化` = `(Parallel - Baseline) / Baseline`。Parallel 的前六项指标累计全部 B=4 候选；后四项只累计轨迹中实际继续的选择分支。ParallelSelectionError 保留其已生成的 decision/candidate usage，不将失败任务记为零。

Baseline termination={"Submitted": 49, "LimitsExceeded": 11, "ContextWindowExceededError": 4}；Parallel termination={"Submitted": 49, "LimitsExceeded": 9, "ParallelSelectionError": 6}。Baseline submitted=49/64、resolved=23/64；Parallel submitted=49/64、resolved=18/64。

逐 task 记录和 canonical trajectory SHA 见 `COMPARISON31PLUS33-20261020.json`。
