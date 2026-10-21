# Ridge baseline 训练轨迹的提交行为分析

分析对象是固定训练清单 `submission-shortest375-20261019` 的最终 238 条 submitted baseline 轨迹。该清单 README 明确规定：训练资格只要求 `submission_status=submitted`、canonical exit/termination 为 `Submitted`，**不要求 `resolved=true`**。因此本分析检查的是实际进入 Ridge 拟合输入的全部 238 条轨迹，而不是只看已通过 Harness 的轨迹。

## Patch 质量

| patch 类型 | 数量 | 占 238 条 |
|---|---:|---:|
| unified diff | 190 | 79.83% |
| 空 patch | 31 | 13.03% |
| 源码片段/其它非 diff | 17 | 7.14% |
| 非 unified diff 合计 | 48 | **20.17%** |

17 条源码片段的典型提交命令是 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat <source-file> | head/tail/...`；31 条空 patch 中 29 条最终只输出 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`，另有 1 条执行 `git diff` 但实际 patch 为空。238 条中共有 21 条最终命令属于 source-cat，其中 18 条确实产生了非 diff patch。

按数据来源：

| 来源 | 轨迹数 | 非 unified diff | 比例 |
|---|---:|---:|---:|
| `online_close` | 107 | 32 | 29.91% |
| `legacy_close` | 117 | 14 | 11.97% |
| `offline_baseline_close` | 14 | 2 | 14.29% |

## 与 Harness 标签的关系

| `resolved` 快照 | 轨迹数 | 非 unified diff | 空 patch | 源码片段 |
|---|---:|---:|---:|---:|
| `true` | 48 | 0 | 0 | 0 |
| `false` | 63 | 0 | 0 | 0 |
| `null` | 127 | 48（37.80%） | 31 | 17 |

`resolved=false` 的 63 条轨迹 patch 都能被语法识别为 unified diff，但这不代表代码正确；它们可能在 Harness 中测试失败。相反，48 条非 diff 轨迹全部属于 `resolved=null`，说明当前主要问题是未评测/未核验 submitted 数据被纳入训练，而不是已确认失败的代码 patch 被纳入训练。

## Action 习惯与循环

对 238 条轨迹统计到：32 条至少出现过 FormatError（共 86 次），24 条出现过达到 81,920 token 上限的响应（共 77 次）。这些是明显的输出控制/推理终止风险。

已有 action-loop linkage 工具覆盖其中 127 条历史 `legacy_close`/`offline_baseline_close` 轨迹：20/127（15.75%）至少出现一次“81,920 reasoning token、无工具动作”的内部 reasoning loop，共 69 个 loop episodes；4 条有 semantic revisit，共 11 次；exact action cycle 为 0 次。该 loop 统计没有覆盖 107 条 `online_close`，所以 15.75% 是部分覆盖下的下界/子集结果，不能外推到全部 238 条。

## 判断

训练数据中确实存在较多不良提交行为，**20.17% 的轨迹无法提供可应用的 unified diff**；其中 online capture 来源最明显。模型表现出两类习惯：完成任务后把源码片段当 patch 输出，以及没有变更时仍提交空 patch。另有一部分轨迹出现 FormatError 和 reasoning 截断循环。

但这不能直接说 Ridge 已经“学会了避免或产生这些 action”。当前 Ridge 目标是 `submission_remaining_reasoning_tokens_log1p`，即从 hidden close 估计到提交的剩余 reasoning tokens；训练标签没有 patch 格式、Harness 成功率或 action correctness。hidden 也位于 action 执行前，训练过程不以最终 patch 内容作为监督。因此 Ridge 可能学到“接近提交/困难较大”的状态相关性，却不会直接学到 unified-diff 生成策略。

更重要的是，48 条无效 patch 全部属于 `resolved=null`。若把这些记录与有效 submitted/已评测记录混合训练，模型可能把“未核验、空提交、源码提交、长调试轨迹”视为同一类到 submission 的剩余量，造成标签和行为分布污染。当前清单 README 也明确说明全部 238 条用于训练，没有验证集/测试集，不能据此声称泛化性能。

## 建议

1. 训练前增加 patch 资格过滤或分层：至少将空 patch、非 unified diff、`resolved=null` 单独统计；若目标是成功提交后的剩余难度，优先使用 `resolved=true` 子集并报告样本量变化。
2. 对无效 patch 不要静默删除：保留为质量审计集，单独训练/评估“提交协议错误”标签。
3. Ridge 仍只作为剩余 reasoning 难度估计器；如果要让系统避免 source-cat/empty patch，需要增加 action-validity 分类器或在 rollout 选择中加入 patch 格式/测试信号。
4. 扩展 action-loop 审计到 107 条 `online_close`，再决定是否存在来源特定的循环行为差异。

原始逐轨迹统计保存在 `ridge-training-habits.json`；训练清单定义见远程 `difficulty/datasets/submission-shortest375-20261019/README.md`。本分析未启动训练、rollout 或 Harness。
