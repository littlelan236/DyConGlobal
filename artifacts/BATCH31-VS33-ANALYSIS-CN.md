# 为什么后批 33 tasks 相对 Baseline 的收益较小

证据时间：2026-10-20T23:49:47.939589+08:00（北京时间）。主分析沿用上轮冻结的 **31 + 33 = 64 个任务**，全部终止状态均保留；只引用已发生的最终 attempts，不重新 rollout、不改评分器、不恢复或重评 patch。36-task run 其余 3 个槽位仍不在本次 33 个已完成结果中。另以此前已建立的 12+23 个“共同 Submitted 且 Baseline 无 reasoning 长循环”任务作诊断子集，绝不以它替换主表。

**目前最强的解释有两部分：成功数差额主要伴随提交格式损坏及判分缺失；token/calls 收益差异主要受任务构成和少数极端轨迹影响。没有证据表明后批换了模型，或整体剩余量预测精度突然变差。**

## 1. 先明确差的是哪项

| 指标（完整批次） | 前批 31 tasks | 后批 33 tasks |
| --- | ---: | ---: |
| Baseline → Parallel resolved | 7 → 7 | 16 → 11 |
| Baseline → Parallel 完成测试判分 | 17 → 18 | 26 → 20 |
| Baseline → Parallel Submitted | 20 → 21 | 29 → 28 |
| 选中 calls 变化 | -16.99% | -5.36% |
| 选中 prompt 变化 | -15.57% | -3.28% |
| 选中 completion 变化 | -60.29% | -61.65% |
| 选中 total tokens 变化 | -16.25% | -4.24% |
| 两侧均完成判分的同任务子集 | 12 tasks：5 → 7 | 14 tasks：8 → 8 |

后批 Parallel 自身的确认成功率其实更高：11/33=33.33%，前批是 7/31=22.58%；变差的是**相对于各自 Baseline 的增益**。后批 Baseline 本身为 16/33=48.48%，前批仅 7/31=22.58%。后批更容易提交：Baseline 29/33，前批 20/31。把这两个不相同的任务集合直接横比，不能推导并行方法在同一批任务上退化。

完整批次的 completion 降幅也没有变差：前批 −60.29%，后批 −61.65%。明显减弱的是 calls/prompt 的节省。两批全分支 completion 分别增加 249.30% 和 103.24%，所以“后批所有性能指标都差很多”不符合当前数据。

## 2. 成功数少 5 个，不能全解释成修复能力下降

后批共有 33 个 harness 请求都已被处理，但**完成测试判分的是 20 个（11 true、9 false）**，余下 13 个仍为 null。13 个包含 5 个未 Submitted 的 MissingPatch，以及如下 8 个已 Submitted 任务：

| 后批 Submitted 但 null 的任务 | Baseline resolved | 已核实原因 |
| --- | ---: | ---: |
| `django__django-11099` | True | 提交源文件/片段而非 diff；Only garbage was found in the patch input |
| `django__django-10880` | True | 提交源文件/片段而非 diff；Only garbage was found in the patch input |
| `pydata__xarray-4075` | True | 提交源文件/片段而非 diff；Only garbage was found in the patch input |
| `matplotlib__matplotlib-13989` | True | Docker ConnectionResetError，提交的是有效 diff |
| `scikit-learn__scikit-learn-26323` | True | 提交源文件/片段而非 diff；Only garbage was found in the patch input |
| `sympy__sympy-12481` | True | 提交源文件/片段而非 diff；Only garbage was found in the patch input |
| `astropy__astropy-13977` | False | 提交源文件/片段而非 diff；Only garbage was found in the patch input |
| `scikit-learn__scikit-learn-13142` | True | 提交源文件/片段而非 diff；Only garbage was found in the patch input |

因此，前面提到的“8 个已提交任务未完成判分”应进一步区分为 **7 个模型提交格式问题 + 1 个 Docker 连接重置**，不能把 8 个都称为基础设施故障或 patch 无法应用。

这 8 个 Submitted/null 中 **7 个 Baseline resolved=true**：6 个格式问题，加 1 个 Docker reset。后批完整状态转移为：

- Baseline true → Parallel true：7 个。
- Baseline true → Parallel false：1 个（`sympy__sympy-15875`）。
- Baseline true → Parallel null：8 个（上述 7 个已提交但无判分，另有 `sympy__sympy-24443` 因 ParallelSelectionError 未提交）。
- Baseline false → Parallel true：1 个（`matplotlib__matplotlib-25775`）。
- Baseline null → Parallel true：3 个；Baseline 的 null 不能当作已知失败。

所以确认成功数的差额在账面上是 **−8 −1 +1 +3 = −5**。两侧都完成判分的 14 个相同任务中，Baseline 与 Parallel **均为 8 个 resolved**，已知的 1 负、1 正相互抵消。这个诊断子集存在筛选偏差，不能替代完整 33-task 成功率，也不能把任一 null 假定为重评必过。但它直接说明：现在还不能把全部 −5 归结为“Parallel 修复错了 5 道题”。

前批没有同样的结构：3 个 Submitted/null 的任务 Baseline 也没有已确认成功，因此它们没有制造出后批这么大的已知成功数差额。

## 3. 分支选择确实能把格式正确的候选排掉

已核对相同末步 prefix 的完整四候选 decision 及文件 SHA。有如下四个直接案例：赢家输出源文件片段，落选者中却有语法/模型格式均合格的 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` 候选。

| 相同末步 prefix | 赢家：提交源码，低分 | 落选：cat patch.txt 的最低分 | 此类有效候选数 |
| --- | ---: | ---: | ---: |
| `django__django-10880` | 3.3075 | 3.3382 | 3 |
| `pydata__xarray-4075` | 1.7983 | 2.7443 | 3 |
| `scikit-learn__scikit-learn-26323` | 1.7116 | 1.7241 | 1 |
| `scikit-learn__scikit-learn-13142` | 1.5975 | 1.6665 | 1 |

例如 `pydata__xarray-4075`：三个落选候选全部要 `cat patch.txt`，分数分别约 2.7443、3.6465、3.1902；赢家 1.7983 却要 `cat /testbed/xarray/core/weighted.py | head -150 | tail -15`。前一轮工具输出已展示正常 git diff，末轮却提交了源码片段，harness 随后报 `Only garbage was found in the patch input`。

这里的证据支持的是**选择了较低分但提交行为错误的分支**，不证明落选分支执行后一定通过测试；没有执行过落选动作，也未据此更改任何评测标签。

当前目标为 `submission_remaining_reasoning_tokens_log1p`，预测的是“到提交还需要多少 reasoning”，**不是补丁正确率、提交合法性或到成功修复的价值**。close hidden 的评分发生在执行当前 action 之前；当前资格检查能验证工具调用是否可解析，却不能保证提交输出是合法 diff。多个立即结束的候选，即便一个输出正确 diff、另一个输出源码，其到提交的未来 reasoning 都可能为 0。对最低分排序不会自动解决这个目标缺口。

这些问题前批也出现，例如 `astropy__astropy-13579` 和 `sympy__sympy-18199` 的最终选择也输出源码；后批恰好更多落在 Baseline 已成功的任务上，因此相对成功数损失显得更大。当前数据不足以证明评分器在统计意义上普遍偏好非法提交，但已有具体失败选择需要修正。

## 4. calls/prompt 收益差：前批大赢家集中，后批有反向离群点

为剔除已知超长 reasoning 对 completion 的影响，单独复用既有 12+23 个匹配任务：

| 共同 Submitted、Baseline 无长循环的子集 | 前批 12 tasks | 后批 23 tasks |
| --- | ---: | ---: |
| Baseline calls 中位数 | 67.5 | 42 |
| Parallel calls 增加的任务 | 2/12 | 11/23 |
| 选中 calls | -20.79% | -3.95% |
| 选中 prompt | -27.00% | +3.58% |
| 选中 completion | +0.76% | -1.78% |
| 选中 total tokens | -26.79% | +3.52% |

后批 Baseline calls 中位数只有 42，前批为 67.5；完整批次中位数也是 70 对 136。这个指标表示历史实现使用的步骤更少，支持“后批可压缩空间较小”，但不是独立的任务难度真值。

**前批 27.00% 的 prompt 节省高度集中在两项 Pylint 任务。**

- `pylint-dev__pylint-4551`：calls 164→98，prompt 减少 3,404,456。
- `pylint-dev__pylint-4604`：calls 152→58，prompt 减少 3,761,932。

两者合计节省 7,166,388 prompt，占前批 12-task 净节省 7,358,928 的 **97.38%**。只作敏感性诊断：移除这两项后，余下 10 个任务的 prompt 节省从 27.00% 变为 **1.20%**，calls 节省从 20.79% 变为 **6.72%**。所以前批本就不代表普遍的 27% 收益。

后批新增 prompt 的主要任务如下：

| 任务（后批 23-task 子集） | Baseline → Parallel calls | 选中 prompt 增量 |
| --- | ---: | ---: |
| `astropy__astropy-13977` | 39 → 87 | +2,583,763 |
| `django__django-16315` | 83 → 111 | +937,058 |
| `django__django-10880` | 87 → 88 | +850,291 |
| `scikit-learn__scikit-learn-13142` | 24 → 46 | +659,100 |
| `django__django-15368` | 42 → 54 | +573,684 |

后批 23-task 净增 prompt 只有 837,425，其中 `astropy__astropy-13977` 单项增量达 2,583,763，其余任务合计净节省 1,746,338。只作敏感性诊断，移除该任务后余下 22 项 prompt 为 **−7.66%**、calls 为 **−8.46%**、completion 为 **−11.52%**。这是集中度证据，不是建议事后剔除该任务来报告更好成绩。

同样，prompt 累计包含整段历史上下文，较晚阶段额外多走一步往往比早期一步更昂贵；因此 calls 少 3.95% 与 prompt 多 3.58% 并不矛盾。这里的 prompt 还包含 APC 命中部分，不能直接等同新 prefill 算力或时延。

## 5. 目前不支持的解释：换了模型、后批更陌生、预测全面失灵

两批 resolved config 已核对：相同 Qwen3.5-9B fingerprint、FP16、B=4、seed=42+branch、采样参数、TP=4/GPU0–3、显存比例 0.4、max_num_seqs=48、max_tokens=81,920、同一 layer 3 Ridge checkpoint hash `dfe6b06cbeaae12a5348c6abda0cfd307e37e6d1e5d6077d8cce88736ad0b820`。保存 decision 中没有 score_error 或 fallback（仅指已写入轨迹的 decision，不覆盖最后一次 API 全失败时未保存的请求）。

核对训练数据 receipt 和 samples/split 后：前批 **0/31** task 在 Ridge 拟合集，后批 **33/33** 均出现于 train＋validation 拟合集；其中后批 9 条 Baseline 的 trajectory SHA 本身也出现在拟合来源。因此“后批因为全是未见任务所以更差”方向与证据相反。训练中见过同任务的单条旧轨迹，并不保证线上其他分支、非法提交或不同工具路径都受过训练；offline event R²也不是四候选选择质量。

另外计算了赢家沿最终 Submitted 轨迹的事后预测诊断：从每条响应的 API reasoning usage 反向累计后续到提交的 reasoning，取 log1p，与当时赢家分数比较，不重新提 hidden 或训练：

| 已 Submitted 轨迹的事后诊断 | 前批 | 后批 |
| --- | ---: | ---: |
| 轨迹数 / 已选择响应数 | 21 / 1581 | 28 / 1488 |
| log1p 剩余 reasoning R² | 0.2052 | 0.3282 |
| log1p MAE | 1.2373 | 1.0538 |

后批的这一描述性误差并没有更大。它只覆盖最后 Submitted 的路径，受选择、任务构成及提前提交影响，不能与冻结验证集 R²直接等同，更不能测量落选动作的反事实收益。不过它进一步削弱了“后批所有评分都突然不准”这个解释。

此外，前批 75 个、后批 12 个落选/候选达到 81,920 输出上限，分别累计 6,144,000 和 983,040 completion tokens（这是上限诊断，不把所有 capped 输出都断言为 reasoning 循环）。后批并没有更多超长候选来解释所谓整体效率恶化。当前没有同负载配对 wall time 数据，本报告不把 token 差异解释成吞吐差异。

## 6. 建议按证据强弱处理

1. **先处理提交契约缺口。** 在现有工具执行/提交处理链验证 marker 后输出确实是可用 diff；不合法时反馈给同一个 agent 继续修正，保留原始响应证据。四候选中有其他有效动作时，可在现有 coordinator 的资格规则里接入有明确语义的提交检查。不能只在评分后依赖 `FormatError`，因为源码输出的 bash 调用本身可以被正常解析。
2. **单独重试真正的 infra error。** `matplotlib__matplotlib-13989` 的连接重置可按现有统一 harness 入口原 patch 重试。其余 7 个源码提交问题不是重新跑 Docker 就能修复的；如需从已存在 diff 恢复，须明确建立新 patch/新评测来源绑定，不能静默替换 canonical 产物。这里未执行这些动作。
3. **再验证 Ridge 相比资格过滤和随机分支的净收益。** 同一任务/模型/采样设置下，对比普通 rollout、B=4 有效分支随机选择、B=4 Ridge；固定完整 cohort 并分别报告 submitted、合法 patch、resolved、null 原因、calls 与所有分支开销。小规模可对关键末步做合法提交和正确性验证，才有依据判断读出分数是否真的排序了更好动作。
4. **最后讨论目标调整。** 当前读出优化到提交剩余量，无法区分“正确完成”和“错误快速结束”。若要引入合法性/成功价值，应通过可复用的现有评估器接口和新实验验证，不把单纯调 score clipping、阈值或 checkpoint 当作已证实修复。未在本次修改 difficulty/。

## 证据与限制

本报告全部读取现有 artifact；未修改配置、解码、模型、训练数据、canonical trajectory 或 harness 标签。两批并非随机分配的可交换样本，后批还与 Ridge 拟合集重叠。无循环子集和共同已判分子集仅用于解释，不替代完整批次。历史 Baseline 与 Parallel 仍有 dtype/seed/runtime 差异。相同分数校准不保证同前缀动作排名正确；合法提交也不保证 patch 通过测试。

数据：`batch31-vs33-analysis.json`（分批计账/敏感性）、`batch31-vs33-diagnostic-evidence.json`（状态/patch/配置/评测绑定）、`batch31-vs33-detail-evidence.json`（usage 与末步证据）、`batch31-vs33-final-choice-evidence.json`（四候选完整末步命令与分数）、`batch31-vs33-training-overlap.json`（训练重叠）、`batch31-vs33-selected-calibration.json`（事后校准）。原始 64-task 表和统计口径不变。
