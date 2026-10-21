# 共同 Submitted 任务：completion 下降与 reasoning 内部循环的关系

本次证据采集时间：2026-10-20T13:56:02.946619+08:00。只分析 Baseline 与 Parallel **均 Submitted 的 15 个相同任务**，沿用此前性能表绑定的最终 canonical trajectory；排除旧失败重试和其余任务。主比较包含 Baseline 的 1,365 次响应和 Parallel 的 1,035 次赢家响应；机制检查读取这些赢家对应的 4,140 个候选，无新 rollout、模型调用或 harness 运行。

结论是：**选中轨迹 completion 的大幅下降，主要对应少数超长 reasoning 循环未进入选中轨迹；没有看到“正常 reasoning 本身更省 tokens”的证据。** 原始降幅为 495,296 tokens（63.96%）。按重复片段覆盖估计，重复输出差额占降幅约 **98.2%～98.4%**；把已确认的循环响应整条剔除后，占比是 **99.24%**，剩余 completion 只下降 1.33%，而剩余 reasoning 增加 13.97%。这属于历史轨迹的记账拆分，不能将全部差额直接解释为 Ridge 的因果收益。

## 1. 原始降幅如何组成

| 指标 | Baseline | Parallel 选中分支 | 变化 |
| --- | ---: | ---: | ---: |
| completion | 774,440 | 279,144 | -63.96% |
| reasoning | 573,739 | 93,709 | -83.67% |
| 非 reasoning 输出（动作/工具格式/正文等） | 200,701 | 185,435 | -7.61% |
| 有大量重复的 reasoning 响应数 | 6 | 0 | -100.00% |

| completion 减少的组成 | Baseline − Parallel | 占总降幅 |
| --- | ---: | ---: |
| 6 条循环响应整体 | 491,520 | 99.24% |
| 其余 reasoning | -11,490 | -2.32% |
| 非 reasoning 输出 | 15,266 | 3.08% |
| 合计 | 495,296 | 100.00% |

Baseline 的 6 条循环响应每条均为 81,920 tokens、`finish_reason=length`，API 报告全部是 reasoning，且没有工具动作或正文。它们合计 491,520 tokens，只分布于 3 个任务：`django__django-11333` 两次，`django__django-12125` 一次，`sympy__sympy-13091` 三次。Parallel 选中轨迹没有这类响应。不能因为最终任务 Submitted 就漏掉这些中间失败响应：它们被序列化成顶层 user 格式纠错消息，但 `extra.response.usage` 记录了真实生成开销。

精确的记账恒等式为：

```text
774,440 − 279,144 = 491,520 − 11,490 + 15,266 = 495,296
```

其中循环响应整体含少量循环发生前的正常内容，所以 99.24% 不是“纯无效 token”的精确占比。下面用只统计文本重复片段的口径复核。

## 2. 确认是 reasoning 内部循环，而不是只因输出很长

将每条 reasoning 单独用同一个模型 tokenizer 重新编码；只比较该响应内部的重复。一个 token 位置被计入重复覆盖，要求它属于某个 32-token 连续窗口，且相同窗口在更早、与该窗口不重叠的位置已出现。对后续匹配窗口取区间并集，防止重叠重复计数。判定“强循环”的诊断阈值是：reasoning 至少 1,024 tokens、重复覆盖至少 1,024 tokens，且覆盖率至少 50%。该阈值只用于离线分析。

| 任务 | 请求序号（0 起） | message_index | reasoning tokens | 重复 32-token 覆盖率 | 单段最高重复次数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `django__django-11333` | 53 | 108 | 81,920 | 98.76% | 212 |
| `django__django-11333` | 54 | 109 | 81,920 | 98.45% | 350 |
| `django__django-12125` | 44 | 90 | 81,920 | 98.53% | 339 |
| `sympy__sympy-13091` | 47 | 96 | 81,920 | 99.86% | 629 |
| `sympy__sympy-13091` | 64 | 129 | 81,920 | 99.32% | 207 |
| `sympy__sympy-13091` | 104 | 208 | 81,920 | 99.67% | 549 |

这些响应都不是仅有少量重复术语或代码：核查全文抽样和最高频段落后，可见数百次重述同一判断，始终没有执行反复宣称要执行的动作。例如：

- `django__django-11333` 请求 54：`Let me verify that the test is actually testing the right thing...` 整段出现 **350 次**；`Wait, I think I'm confusing myself. Let me re-read the PR description again...` 出现 **348 次**。
- `sympy__sympy-13091` 请求 47：以 `Actually, I think I see the issue now. When s.__eq__(f) is called...` 开头的同一段落出现 **629 次**，反复说明同一比较协议而未产生工具调用。
- `django__django-12125` 请求 44：以 `Wait, I think I understand the issue now...` 开头的同一段落出现 **339 次**。

把覆盖率阈值改为 30%、70% 或 90%，仍只识别出同样的 6 条 Baseline、0 条赢家和 4 条落选 reasoning 循环。仅累计后续重复窗口，而不删掉整条循环响应，结果如下：

| 重复窗口长度 | Baseline 重复覆盖 tokens | Parallel 选中重复覆盖 tokens | 覆盖差 / completion 总降幅 |
| --- | ---: | ---: | ---: |
| 16 | 491,153 | 4,779 | 98.20% |
| 32 | 489,316 | 2,025 | 98.38% |
| 64 | 487,751 | 894 | 98.30% |

32-token 口径下，重复覆盖差为 487,291 tokens，占总降幅 98.38%。从 reasoning 中减去各自的重复覆盖后，Baseline 余下 **84,423** tokens，Parallel 余下 **91,684** tokens，仍增加 **8.60%**。这些是文本重复的诊断估计；被重复引用的代码或合理复述也可能被覆盖，未检测到重复的内容也不等于经过验证的“有效思考”。

token 总长度采用原 API usage。重复分析采用文本重编码，**不冒充 Baseline 原生生成 token IDs**：5,505 条响应中 5,503 条重编码长度与 API reasoning 计数一致，2 条短 1 token；6 条 Baseline 循环均完全一致。

## 3. 去掉循环后，正常部分没有更短

| 剔除已确认的循环响应后 | Baseline | Parallel 选中分支 | 变化 |
| --- | ---: | ---: | ---: |
| 响应数 | 1,359 | 1,035 | -23.84% |
| completion | 282,920 | 279,144 | -1.33% |
| reasoning | 82,219 | 93,709 | +13.97% |
| 非 reasoning 输出 | 200,701 | 185,435 | -7.61% |
| 平均 reasoning / 响应 | 60.50 | 90.54 | +49.65% |
| reasoning 中位数 / 响应 | 23 | 26 | +13.04% |
| 平均 completion / 响应 | 208.18 | 269.70 | +29.55% |

剔除循环响应后 calls 减少，但平均每次 reasoning 增加约 49.65%，总 reasoning 增加 11,490 tokens。completion 剩下的 3,776-token 小幅下降来自非 reasoning 输出减少 15,266，抵消了 reasoning 增加。`非 reasoning 输出` 通过 `completion_tokens − reasoning_tokens` 得出，包含动作、工具序列化、正文及特殊边界，不能全部叫作代码。

这也不是“所有任务都一样”：无循环时，个别任务仍减少 calls 或输出，个别任务增加。为避免只在一侧删响应，再整体排除出现循环的 3 个任务，保留两侧轨迹均未检测到 reasoning 循环的 **12 个任务**：

| 两侧均无 reasoning 循环的 12 个任务 | Baseline | Parallel 选中 | 变化 |
| --- | ---: | ---: | ---: |
| calls | 986 | 781 | -20.79% |
| completion | 206,344 | 207,918 | +0.76% |
| reasoning | 56,749 | 68,067 | +19.94% |

这个子集的 completion 增加 0.76%，reasoning 增加 19.94%。因此，现有数据不支持把总体 63.96% 的下降表述为对正常思考的普遍压缩。

## 4. 同一 prefix 下，评分器是否偏爱更短的正常 reasoning

Baseline 和 Parallel 后续工具动作及请求数不同，不能按 step 序号当成同一个问题来配对。为减少这一歧义，额外在 Parallel 已发生的每个 prefix 下比较候选：先求该组有效、非循环候选的平均 reasoning 长度，再跨组累计，与实际赢家累计长度比较。

| 同一 prefix 的比较范围 | 按候选均值累计的 reasoning | 选中 reasoning | 变化 |
| --- | ---: | ---: | ---: |
| 全部 1035 组、仅有效且非循环的候选 | 88,625.42 | 93,709 | +5.74% |
| 四候选均有效且无循环的 1029 组 | 86,720.25 | 91,212 | +5.18% |

在四候选全部有效且无循环的 1,029 组中，赢家 reasoning 比同前缀四候选均值增加 5.18%；赢家 completion 为 275,823，对照均值累计 269,719.75，增加 2.26%。因此，候选级证据也没有显示评分器在普遍挑选更短的当前 reasoning。

这只是固定已发生 prefix 的局部长度比较，不是重新执行的随机分支 rollout，也不是判断思考质量。当前评分目标是 `submission_remaining_reasoning_tokens_log1p`，在 close 处估计的是后续到提交的剩余 reasoning；它本就不是当前 reasoning 长度最小化器。

## 5. 避开循环的是哪个机制

原生 ledger SHA 和输出 token IDs 复核了共同 Submitted 范围内 5 个达到输出上限的 Parallel 候选：

| 任务 | step（0 起） | 落选 branch | reasoning tokens | 原生 close 位置 | 排除原因 | 赢家 completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `django__django-11333` | 41 | 3 | 1,042 | 1042 | FormatError | 1,023 |
| `scikit-learn__scikit-learn-14629` | 45 | 3 | 81,920 | 无 | FormatError | 862 |
| `scikit-learn__scikit-learn-14629` | 46 | 3 | 81,920 | 无 | FormatError | 812 |
| `scikit-learn__scikit-learn-14629` | 53 | 1 | 81,920 | 无 | FormatError | 500 |
| `scikit-learn__scikit-learn-14629` | 53 | 2 | 81,920 | 无 | FormatError | 500 |

`scikit-learn__scikit-learn-14629` 的 4 条落选候选才是 reasoning 内部循环，分布于 3 个 prefix，均未生成 `</think>`，没有 close hidden 和有效分数，最终都以 `FormatError` 标为 ineligible。赢家保留了可执行动作，循环输出没有进入后续上下文。

另一个长候选 `django__django-11333` step 41 在原生输出位置 1,042 已生成 close，后面才出现约 80,878 tokens 的非 reasoning 输出；尾部不断重复缓存 wrapper 的代码注释。它不计入上述 reasoning 循环。其 score 为 **6.0034**，比赢家 **6.6616** 更低，却仍因格式错误而落选，直接说明资格检查能覆盖分数排序。

代码证据（源码 SHA 与摘录见 `reasoning15-mechanism-evidence.json`）：`baseline/rollout/model.py` 的 parse 复用 `_parse_actions`；`baseline/rollout/parallel.py` 只有解析成功、分数有效且未以 length 截断的候选才进入 eligible 集合，然后才以最低 score 选赢家。因此，当前可以确认 **B=4 提供其他有效候选，格式/完整性过滤阻止这些循环输出成为赢家**；不能据此声称 Ridge 通过识别循环特征排除了它们。历史 Baseline 的 6 个循环发生在另外的 prefix，无法与这些候选一一对应。

## 6. 对实际性能的含义与证据边界

选择分支不包含循环，并不代表运行时没有生成循环。共同 Submitted 范围内，全部候选 completion 为 **1,498,899**，比 Baseline **774,440** 增加 **93.55%**；其中落选 reasoning 循环已消耗 **327,680** tokens。当前协调器等待全部候选完成才选择，这些循环仍会拖慢在线 rollout。所以上面的收益主要反映**最终轨迹保留的输出开销**，不能据此声称总生成算力或时延下降。

Baseline 是历史 BF16、seed=0、并发 6/8 的轨迹，Parallel 是 FP16、seed=42+branch、并发 12；还存在 vLLM 版本差异。任务虽相同，工具路径和随机采样不同。此处可以判断大幅下降在账目上由哪些输出构成，也可以核对同前缀候选，但不能严格分离 Ridge、资格过滤、数值精度与随机采样的因果贡献。共同 Submitted 的筛选也不代表完整任务集。

若后续要验证“评分器节省正常思考”的因果效应，应在同模型数值设置、seed 方案和共同任务上，比较同一 B=4 生成/资格过滤框架下的 Ridge 选择与有效候选随机选择，并分别报告循环率、非循环 reasoning、完整 rollout completion 及全部分支开销。本次未改解码策略或启动该实验。

## 7. 逐任务复核

`剔循环后 B−P completion` 为正表示 Parallel 更少。此表仍只包含同一批 15 个共同 Submitted 任务。

| 任务 | 原始 B completion | P 选中 completion | B 循环次数 | 剔循环后 B−P completion | B 非循环 reasoning | P 非循环 reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `astropy__astropy-8707` | 12,267 | 15,385 | 0 | -3,118 | 2,011 | 3,265 |
| `django__django-10999` | 8,652 | 5,699 | 0 | 2,953 | 2,617 | 2,590 |
| `django__django-11333` | 181,012 | 19,425 | 2 | -2,253 | 5,004 | 7,035 |
| `django__django-12125` | 112,207 | 38,199 | 1 | -7,912 | 8,168 | 14,077 |
| `matplotlib__matplotlib-14623` | 20,915 | 19,437 | 0 | 1,478 | 9,458 | 10,734 |
| `mwaskom__seaborn-3187` | 39,694 | 44,249 | 0 | -4,555 | 9,851 | 15,834 |
| `pylint-dev__pylint-4551` | 35,652 | 23,682 | 0 | 11,970 | 5,977 | 7,774 |
| `pylint-dev__pylint-4604` | 17,743 | 8,629 | 0 | 9,114 | 5,613 | 3,065 |
| `pytest-dev__pytest-10081` | 5,777 | 4,720 | 0 | 1,057 | 2,695 | 2,545 |
| `pytest-dev__pytest-10356` | 20,691 | 11,179 | 0 | 9,512 | 4,904 | 2,625 |
| `pytest-dev__pytest-7432` | 10,597 | 9,223 | 0 | 1,374 | 4,422 | 4,743 |
| `scikit-learn__scikit-learn-14629` | 11,385 | 19,411 | 0 | -8,026 | 2,910 | 9,525 |
| `sympy__sympy-13091` | 274,877 | 13,602 | 3 | 15,515 | 12,298 | 4,530 |
| `sympy__sympy-18199` | 13,931 | 41,366 | 0 | -27,435 | 3,184 | 3,901 |
| `sympy__sympy-19346` | 9,040 | 4,938 | 0 | 4,102 | 3,107 | 1,466 |

## 8. 产物与复现

- `reasoning15-analysis.json`：聚合、逐任务、敏感性、同前缀比较、循环原文摘录和段落频次。
- `reasoning15-responses.csv`：5,505 条响应的 usage、资格、重复覆盖、message/step/branch 标识；不重复累计赢家。
- `reasoning15-decisions.csv`：1,035 个同前缀候选组。
- `reasoning15-corpus.jsonl.gz` 与 `reasoning15-corpus-meta.json`：冻结的 reasoning 证据、30 条 canonical trajectory 的 SHA 和 1,035 个 decision 文件 SHA；读取时已与 manifest 的 trajectory SHA 核对。
- `reasoning15-mechanism-evidence.json`：5 条 capped 候选原生 ledger 的 SHA、close 坐标、赢家与源码摘录。
- 脚本：`extract_reasoning15.py`、`analyze_reasoning15.py`、`verify_reasoning15_mechanism.py`、`build_reasoning15_report.py`。前三个通过既有 `transport.py --script` 在远程项目根目录使用 runner Python 执行，只读取历史轨迹/decision/ledger 并写此分析产物；最后一个在本目录生成本文和逐任务 CSV。已查阅当前入口表与诊断目录，没有对应的 reasoning 重复分析入口；未重写正式 rollout、action 分类或训练流程。

本地报告目录：`parallel-ridge-integration-20261019/artifacts/`；远程同步目录：`/home/wangtianyi/DyConGlobal/artifacts/`。原性能对比表保持原始 usage 口径。
