# 修复后 200-task 首条轨迹核验

Run：`capture-repaired200-20261018-162700`。
首个完成任务：`astropy__astropy-14365`。
Rollout ID：`r-4bfa15d539ffaf671c415a4fc72d503e`。
按 attempt receipt 的 `finished_at` 取最早完成者，没有挑选另一条轨迹替代。

结论：**在线采集完整性、token 对齐、NAS 产物校验和 FP16 vLLM 数值重放均通过**。
Agent 成功提交补丁，途中有一次已恢复的工具格式错误；不能将本条称为零格式错误轨迹。

## 实际阅读结论

任务要求 QDP 文本命令不区分大小写。轨迹先定位并阅读
`astropy/io/ascii/qdp.py`，用题目中的 `read serr 1 2` 复现 ValueError，随后给正则编译增加
`re.IGNORECASE`，验证小写、原大写和合法的混合大小写命令均可读取。
原有 `test_qdp.py` 8 项测试和 pytest doctest 4 项均通过；最终提交补丁只改一行正则编译。

模型过程有两处值得区分：

- 中途先后构造了两份无效的混合大小写 QDP 输入（同一行混放两个命令，或漏写 READ），
  工具如实返回错误；模型后来改用合法输入并验证通过。其间 reasoning 有重复和概念混淆，
  属于模型推理质量问题，没有出现服务中断或全部 token 为 0 的退化。
- 第 17 次响应的工具 function name 被解析为提交命令文本，arguments 为 `{}`，触发一次
  FormatError；下一次响应正确使用 bash/command 提交成功。不能宣称本轨迹零格式错误。

另外用同一 tokenizer 解码该次原生 output token IDs，确认模型实际生成的是
`<function=echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; git diff`，随后出现无匹配的
`</parameter>`；原文没有正确的 `<function=bash>` / `<parameter=command>` 结构。
因此这次 FormatError 有原生生成格式错误的直接证据，不是 capture 修改了正确工具调用。

共 18 次模型响应，包含 17 次有效动作和 1 次可恢复的 FormatError。
16 条普通 tool observation 加 1 条最终 submission/exit，与 17 次有效动作对应。
最终提交通过 agent 的终止分支保存为 exit 消息，没有额外普通 tool 消息。

自动 audit 的 `protocol_ok=false` 使用的是“动作数等于普通 tool 消息数且零 FormatError”
的严格条件。保留该原始结果：它不等于 agent 崩溃或提交失败。人工阅读结论为执行链路正常、
发生一次已恢复的格式错误。状态为 **submitted**，未执行 SWE-bench harness，不据此标 resolved。

运行起止：2026-10-18 17:14:09–17:16:36（北京时间），约 147 秒。
时间包含模型生成、容器/工具和任务收尾，不表示 capture 的独立增量开销。

## NAS 与精确 token 对齐

| Anchor | 预期 | NAS 实有 | 缺失 |
|---|---:|---:|---:|
| reasoning newline | 94 | 94 | 0 |
| `</think>` | 18 | 18 | 0 |
| 合计 | 112 | 112 | 0 |

NAS 原始事件也为 112 个，非 reasoning 事件为 0，session 未记录导出失败。
与上一条核验的 34/49 覆盖不同，本轨迹两种 anchor 均完整，`capture_status=complete`。

核验覆盖 canonical 轨迹与 patch SHA、18 份请求 ledger、query/API attempt/native request ID
关联、prompt/output token hash、generation_index 与 absolute_position、后续 input forward、
shard/member SHA，以及全部 32 个 site、每 site 4096 维的 shape/有限值。
FormatError 那次真实响应也在 ledger 和采集对照集合中。

执行证据：106 个点来自 FULL/decode，6 个点来自 eager/mixed。
这些模式依据每个事件的实际执行证据记录，没有将 mixed batch 标成 FULL。

轨迹 SHA：`5dbb5a2b892707ec57d104d7d2c7049263d4d500eac4b0b0656d4f1215e142a4`。

## FP16 vLLM 离线重放

已完成。使用冻结快照的 Qwen3.5-9B checkpoint、FP16、TP4、全部 32 个 site 和关闭 steering
的相同条件，原生 prefix token IDs 直接作为输入，不重分词或重新生成轨迹。
通过现有 eager replay adapter 比较 cold/warm/reordered；阈值沿用 cosine ≥ 0.999、
relative L2 ≤ 0.05。此次无缺失点，因此不需要 offline-only 补提取。

独立 replay 进程共享 GPU4–7，限制 2 并发、512 batched tokens、每卡 2 GiB KV cache，
启动前检查显存余量。只等待进程退出，不循环查询线上 rollout 状态。

三组均比较 112 点 × 32 site，共 **10,752 次逐点/逐 site 比较**，全部满足既有数值门槛：

| 条件 | 最低 cosine | 最大 relative L2 | 判定 |
|---|---:|---:|---|
| cold | 0.9999729595 | 0.735486% | 通过 |
| warm | 0.9999519460 | 0.980896% | 通过 |
| reordered | 0.9999519460 | 0.980896% | 通过 |

按 anchor 和在线执行模式进一步汇总最大 relative L2（每格取全部对应点/全部 site 的最大值）：

| 分组 | 点数 | cold | warm | reordered |
|---|---:|---:|---:|---:|
| reasoning newline | 94 | 0.735486% | 0.980896% | 0.980896% |
| `</think>` | 18 | 0.575472% | 0.517284% | 0.519562% |
| FULL / decode | 106 | 0.683677% | 0.907152% | 0.907152% |
| eager / mixed | 6 | 0.735486% | 0.980896% | 0.980896% |

上述结果验证了该轨迹已保存 online hidden 与 exact-prefix vLLM replay 的一致性，
包括 FormatError 响应及 mixed/eager 事件；浮点结果不是逐位相等。
缓存与 batch/执行路径变化可带来不同的浮点误差，warm 并不保证误差比 cold 更小。
这里的 relative L2 定义为 `||online - replay||₂ / ||replay||₂`，最大值不是全体平均值。

56 个两点窗口的实际 capture 调用时间合计：cold 629.64 秒、warm 26.97 秒、
reordered 28.43 秒。每窗口先 reset prefix cache，然后依次执行三种条件；cold 窗口内
仍允许 vLLM 原生批内前缀复用，并非强制每个 prefix 都零命中。时间不用于吞吐测速，
也不能与 agent rollout 的 147 秒直接比较为 capture 增量成本。

数值任务于 2026-10-18 18:12:54（北京时间）结束，replay exit code 为 0。
原始 `result.json` 为 `capture_complete=true`、`numerical_passed=true`、
`protocol_ok=false`、`overall_passed=false`；后两个字段来自前述严格协议条件，
不代表 hidden 数值或完整性失败。保留原始判定并在 `manual-review.json` 中另记人工阅读结论。

## 证据位置

远程核验目录：
`/home/wangtianyi/DyConGlobal/hidden/reports/capture-repaired200-20261018-162700/first-rollout/`。

NAS：`/VM/wangtianyi/DyConGlobal/hidden/online/capture-repaired200-20261018-162700/`。

- `selected.json`、`audit.json`：选择依据与完整性证据；
- `reasoning-and-actions.json`：逐响应 reasoning/动作摘录；
- `paired-events.json`：112 个 online 对照点；`missing-prefixes.json` 为空；
- `replay-launch.json`、`replay.log`、`replay/`：重放条件与逐点/逐层数值证据；
- `result.json`：原始严格自动判定。人工阅读不覆盖或删除其诊断。
- `manual-review.json`：人工阅读结论、原生格式错误文本与终止动作/exit 对应说明。

本次核验没有修改正在运行的 200-task 配置、原始轨迹、NAS online tensor、manifest 状态，
也没有重新生成轨迹或使用 offline-only 数据填充 online 对照集合。
