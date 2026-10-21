# Baseline / Parallel 对比：31 + 33 个结果，排除未提交与 Baseline reasoning 长循环

核对时间：2026-10-20T22:09:05.340907+08:00（北京时间）。从 31-task run 的 31 个结果及 36-task run 已完成的 33 个结果出发，先要求双方最终 canonical trajectory 均 Submitted，再排除 Baseline 有 reasoning 长循环的整对任务，最终保留 **35 个相同任务（12 + 23）**。

| 指标 | Baseline | Parallel（B=4） | 相对变化 |
| --- | ---: | ---: | ---: |
| 实际请求数（全分支） | 2,124 | 7,496 | +252.92% |
| prompt（全分支） | 50,644,987 | 176,493,936 | +248.49% |
| completion（全分支） | 457,039 | 2,505,328 | +448.17% |
| total tokens（全分支） | 51,102,026 | 178,999,264 | +250.28% |
| 选择分支 calls | 2,124 | 1,874 | -11.77% |
| 选择分支 prompt | 50,644,987 | 44,123,484 | -12.88% |
| 选择分支 completion | 457,039 | 454,152 | -0.63% |
| 选择分支 total tokens | 51,102,026 | 44,577,636 | -12.77% |

`相对变化 = (Parallel − Baseline) / Baseline`。前四行的 Parallel 累计所有 B=4 候选，后四行仅累计最终选中轨迹。Baseline 为单分支。这里按**整条轨迹及其匹配任务对**剔除，不是只减掉循环片段的 tokens。筛选仅限制 Baseline 的长循环；保留任务中的 Parallel 落选候选如有循环，仍按真实 usage 累计。

最终选中轨迹 calls 变化为 **-11.77%**，prompt 为 **-12.88%**，completion 为 **-0.63%**，total tokens 为 **-12.77%**。

## 筛选过程

| 批次 | 起始任务 | Parallel 未提交 | Parallel 已提交但 Baseline 未提交 | 双方提交但 Baseline 长循环 | 最终纳入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 31-task run | 31 | 10 | 6 | 3 | 12 |
| 36-task run 已完成的 33 个 | 33 | 5 | 3 | 2 | 23 |

64 个起始结果中，15 个 Parallel 未提交，剩余 49 个中有 9 个 Baseline 未提交，得到 40 个双方 Submitted 的任务；其中 5 个 Baseline 存在长循环，最后保留 35 个。排除规则按表中的顺序互斥计数。

36-task run 的当前 progress 为 28 Submitted、3 LimitsExceeded、2 ParallelSelectionError、1 ValueError，另有 2 个任务没有最终 progress 行。用户所指 33 个结果对应前面三类；其余三个为：`sympy__sympy-14248`（not_recorded）；`pydata__xarray-4966`（not_recorded）；`sphinx-doc__sphinx-9461`（ValueError）。它们未纳入起始 33，也未纳入任何主表开销。

## Baseline 循环排除证据

| 整对排除的任务 | Baseline 循环响应数 | 循环响应 reasoning tokens | 重复 32-token 覆盖率 |
| --- | ---: | ---: | ---: |
| `django__django-12125` | 1 | 81,920 | 98.53%～98.53% |
| `sympy__sympy-13091` | 3 | 245,760 | 99.32%～99.86% |
| `django__django-11333` | 2 | 163,840 | 98.45%～98.76% |
| `sympy__sympy-15875` | 1 | 81,920 | 98.68%～98.68% |
| `matplotlib__matplotlib-25775` | 1 | 81,920 | 99.83%～99.83% |

沿用上一轮 `reasoning15-analysis.json` 的检测口径：仅在单次 reasoning 内，以相同模型 tokenizer 重编码，寻找有更早非重叠匹配的连续 32-token 窗口，对后续重复窗口取覆盖并集；reasoning 至少 1,024 tokens，重复覆盖至少 1,024 tokens，且覆盖率至少 50%，才判为强循环。原 API reasoning/completion usage 是长度统计来源，文本重编码只服务于重复检测，不声称是 Baseline 原生生成 IDs。

上表 5 个任务共 8 条循环响应，全部达到 81,920 reasoning tokens 且没有产生工具调用；最高频段落与首尾摘录保存在 JSON 的 `baseline_loops` 字段。新增批次排除的是 `sympy__sympy-15875` 和 `matplotlib__matplotlib-25775`；此前三个排除任务保持一致。已复核旧批次筛选后的 12-task calls/prompt 与之前的结果完全一致。

## 分批汇总

### 31-task run：12 个匹配任务

| 指标 | Baseline | Parallel（B=4） | 相对变化 |
| --- | ---: | ---: | ---: |
| 实际请求数（全分支） | 986 | 3,124 | +216.84% |
| prompt（全分支） | 27,259,247 | 79,601,276 | +192.02% |
| completion（全分支） | 206,344 | 1,140,837 | +452.88% |
| total tokens（全分支） | 27,465,591 | 80,742,113 | +193.98% |
| 选择分支 calls | 986 | 781 | -20.79% |
| 选择分支 prompt | 27,259,247 | 19,900,319 | -27.00% |
| 选择分支 completion | 206,344 | 207,918 | +0.76% |
| 选择分支 total tokens | 27,465,591 | 20,108,237 | -26.79% |

### 新增 33 个结果：23 个匹配任务

| 指标 | Baseline | Parallel（B=4） | 相对变化 |
| --- | ---: | ---: | ---: |
| 实际请求数（全分支） | 1,138 | 4,372 | +284.18% |
| prompt（全分支） | 23,385,740 | 96,892,660 | +314.32% |
| completion（全分支） | 250,695 | 1,364,491 | +444.28% |
| total tokens（全分支） | 23,636,435 | 98,257,151 | +315.70% |
| 选择分支 calls | 1,138 | 1,093 | -3.95% |
| 选择分支 prompt | 23,385,740 | 24,223,165 | +3.58% |
| 选择分支 completion | 250,695 | 246,234 | -1.78% |
| 选择分支 total tokens | 23,636,435 | 24,469,399 | +3.52% |


## 逐任务清单

| 任务 | 批次 | B calls | P 选中 calls | B prompt | P 选中 prompt | B completion | P 选中 completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `astropy__astropy-13977` | 33 | 39 | 87 | 598,191 | 3,181,954 | 6,861 | 30,501 |
| `astropy__astropy-8707` | 31 | 63 | 88 | 1,279,355 | 2,011,731 | 12,267 | 15,385 |
| `django__django-10880` | 33 | 87 | 88 | 2,123,856 | 2,974,147 | 12,884 | 19,147 |
| `django__django-10999` | 31 | 28 | 20 | 324,654 | 192,078 | 8,652 | 5,699 |
| `django__django-11099` | 33 | 12 | 22 | 45,809 | 144,931 | 2,825 | 4,018 |
| `django__django-14580` | 33 | 72 | 59 | 1,457,846 | 1,349,209 | 11,351 | 9,707 |
| `django__django-15368` | 33 | 42 | 54 | 640,822 | 1,214,506 | 7,841 | 8,222 |
| `django__django-16315` | 33 | 83 | 111 | 2,353,388 | 3,290,446 | 12,259 | 14,504 |
| `matplotlib__matplotlib-13989` | 33 | 34 | 35 | 481,885 | 437,016 | 8,045 | 6,902 |
| `matplotlib__matplotlib-14623` | 31 | 93 | 88 | 2,937,833 | 2,269,673 | 20,915 | 19,437 |
| `matplotlib__matplotlib-24870` | 33 | 54 | 56 | 1,457,085 | 1,157,456 | 25,481 | 14,299 |
| `matplotlib__matplotlib-25479` | 33 | 70 | 87 | 2,119,349 | 2,681,080 | 17,157 | 16,882 |
| `mwaskom__seaborn-3187` | 31 | 136 | 131 | 4,819,112 | 5,546,108 | 39,694 | 44,249 |
| `pydata__xarray-4075` | 33 | 34 | 26 | 576,747 | 367,627 | 12,539 | 9,797 |
| `pydata__xarray-4629` | 33 | 16 | 12 | 134,005 | 80,376 | 7,771 | 6,397 |
| `pydata__xarray-6992` | 33 | 33 | 24 | 340,259 | 197,761 | 8,243 | 6,294 |
| `pylint-dev__pylint-4551` | 31 | 164 | 98 | 6,594,827 | 3,190,371 | 35,652 | 23,682 |
| `pylint-dev__pylint-4604` | 31 | 152 | 58 | 4,570,283 | 808,351 | 17,743 | 8,629 |
| `pylint-dev__pylint-6903` | 33 | 39 | 23 | 689,090 | 215,864 | 17,751 | 6,606 |
| `pylint-dev__pylint-7277` | 33 | 51 | 17 | 860,020 | 163,089 | 11,902 | 3,968 |
| `pytest-dev__pytest-10081` | 31 | 49 | 34 | 733,531 | 384,358 | 5,777 | 4,720 |
| `pytest-dev__pytest-10356` | 31 | 70 | 87 | 1,875,639 | 2,220,991 | 20,691 | 11,179 |
| `pytest-dev__pytest-5631` | 33 | 30 | 30 | 297,432 | 308,007 | 5,810 | 4,209 |
| `pytest-dev__pytest-7432` | 31 | 44 | 37 | 752,453 | 611,486 | 10,597 | 9,223 |
| `pytest-dev__pytest-8399` | 33 | 44 | 64 | 526,809 | 986,847 | 3,916 | 5,366 |
| `scikit-learn__scikit-learn-12682` | 33 | 106 | 59 | 3,096,835 | 1,152,790 | 15,176 | 11,605 |
| `scikit-learn__scikit-learn-13142` | 33 | 24 | 46 | 272,640 | 931,740 | 4,208 | 14,532 |
| `scikit-learn__scikit-learn-14496` | 33 | 41 | 28 | 477,152 | 259,884 | 9,361 | 5,478 |
| `scikit-learn__scikit-learn-14629` | 31 | 65 | 59 | 1,110,360 | 1,076,808 | 11,385 | 19,411 |
| `scikit-learn__scikit-learn-25102` | 33 | 79 | 50 | 2,486,372 | 1,573,213 | 26,727 | 29,555 |
| `scikit-learn__scikit-learn-26323` | 33 | 43 | 30 | 722,737 | 413,032 | 9,676 | 5,340 |
| `sphinx-doc__sphinx-8595` | 33 | 71 | 50 | 1,234,217 | 698,248 | 7,113 | 6,229 |
| `sympy__sympy-12481` | 33 | 34 | 35 | 393,194 | 443,942 | 5,798 | 6,676 |
| `sympy__sympy-18199` | 31 | 71 | 50 | 1,587,990 | 1,287,607 | 13,931 | 41,366 |
| `sympy__sympy-19346` | 31 | 51 | 31 | 673,210 | 300,757 | 9,040 | 4,938 |

## 来源与核验

- Baseline 固定使用历史 run `baseline-qwen35-9b-forward-apc-coord-gpu4-7-w12-20261012`，按 manifest task_key 精确匹配，不另挑更优或更短的 Baseline。
- Parallel 使用 `parallel-ridge-event-b4-w12-disjoint31-20261019` 和 `parallel-ridge-event-b4-w12-overlap36-20261019`。只统计各 run progress 绑定的最终 rollout_id/attempt_id；失败重试不混入。
- 从 manifest 绑定的 canonical trajectory 读取；读取时逐条校验 SHA-256。冻结的 selection/progress SHA 与 manifest revision=2432、manifest SHA 保存在 JSON。
- 只读顶层消息的 `extra.response.usage`，按 response ID 去重，包含被写在 user 消息上的格式错误回复。Parallel 全分支 usage 按 decision ID 去重，并核对四个 candidate summary 的 token 求和与赢家 usage。35 个任务两侧的 calls 均有完整响应 usage。
- 统计范围 JSON 包含原始 67 个槽位的纳入/排除原因，64 个已完成范围标为 `in_requested_completed_scope=true`。输出含汇总 CSV、35 对逐任务 CSV、67 槽位筛选 CSV；均可追溯 canonical trajectory SHA。
- Prompt 包含缓存命中的输入 tokens，不等于实际 cache-miss prefill。历史 Baseline 与 Parallel 的 dtype、seed、运行环境不同，此表为历史实测对照。

本次只分析已有证据，未修改 rollout、评测、训练或运行配置。当前未重新运行 harness；本表沿用上轮的八行 token/calls 格式，不将未评测结果计作 unresolved。
