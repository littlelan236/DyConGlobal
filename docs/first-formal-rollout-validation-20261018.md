# 正式 200-task 首条 rollout 核验

Run：`capture-verified200-20261018-143317`。
首条：`astropy__astropy-14539`，rollout ID：`r-0403e1ace71fcb0758acd02e97a77281`。
完成时间：2026-10-18 14:38:04 +08:00。

结论：请求、解析与工具执行链可用，任务未修复/提交；已有在线点的数值检验通过，
在线采集完整性不通过。总体 `overall_passed=false`。

## 触发方式

入口：[first_rollout.py](../hidden/evaluation/first_rollout.py)。

```bash
.venvs/runner/bin/python -m hidden.evaluation.first_rollout capture-verified200-20261018-143317
```

脚本用 inotify 等待正式 runner 对 `runs/<run>/progress.json` 的原子写入，
用 pidfd 感知进程退出，没有周期性状态查询。部署时首条已经完成，因此本次从既有
attempt receipt 按 `finished_at` 选择最早记录，立即执行；没有跳过 partial 而选另一条。
结果落盘后重复调用直接读取结果，不重复重放。

## 实际阅读轨迹

阅读了完整问题描述、18 轮 reasoning/command 和工具 observation。具体行为：

- 任务要求修复 FITSDiff 对相同 VLA FITS 文件误报差异。
- 第 2 轮 `cat astropy/io/fits/diff.py` 读入整份源码；下一轮 prompt 从 1,192 增至 15,233 tokens。
- 此后多轮围绕 `arr != arr` 的 `ValueError: truth value ... ambiguous` 改写打印调试语句，
  后续才查看 `_VLF` 与 `TableDataDiff._diff`，未形成源码修复或提交。
- 18 次响应均为有效工具调用，18 条 observation，FormatError=0。
  Python 调试中的非零返回值是工具真实返回，没有被写成调用格式错误。
- 最后完成的请求 prompt=24,225 tokens、output=311 tokens；下一请求触发 ContextWindowExceededError。
  原配置总上下文 32,768、输出预算 8,192，允许的输入上界约 24,576 tokens。
  最终为 unsubmitted；这是限额终止，不是进程 kill、服务崩溃或任务 resolved。

原文 SHA-256：`6f72e58224e558b608ff3501ca63ca3ce6e4344ccf1c56453cfbe06ff887c095`。
完整 reasoning/actions 摘录在核验目录的 `reasoning-and-actions.json`；原始轨迹仍保存在 baseline canonical 路径。

## NAS 与 token 对齐检验

| Anchor | 应有 | 实有 | 缺失 |
|---|---:|---:|---:|
| reasoning 换行 | 31 | 19 | 12 |
| `</think>` | 18 | 15 | 3 |
| 合计 | 49 | 34 | 15 |

应有集合由原生 prompt/output token IDs 与共同 `reasoning_targets` 计算，不通过文本重分词估计。
已有 34 点验证通过：canonical/patch SHA、18 份 request ledger 的 query/API attempt/native ID
一一关联、原生 token 序列哈希、generation_index/absolute_position、后续 input-forward 可用性、
NAS 完整 shard/member SHA、全部 32 个 site 的 4096 维形状和有限值。
这些点的实际执行证据均为 FULL/decode。

同 session 记录了 **80 次 export_backpressure**，存在明确的导出队列拒收；采集状态正确标记为 partial。
288 个已落盘原始点中，254 个位于 reasoning 区间之外，已从本次数值比较中排除。
这表明导出链存在队列压力；尚不能把队列满直接归因于 NAS 带宽或磁盘容量。

未修改这条轨迹、baseline 状态或 NAS 原始采集数据。

## 离线 vLLM 重放

沿用原生 eager replay adapter、FP16、同一 checkpoint、TP4 与全部 32 个 site。
模型身份、steering、generation 和 site 列表取自正式 run 的冻结快照。
仅离线资源使用显式覆盖：GPU4–7、2 并发、512 batched tokens、每 GPU 2 GiB KV cache；
启动时检查每 GPU 空闲至少 14 GiB。使用独立进程，不重启在线服务。

对 NAS 中 34 个对应点分别执行 cold、warm、reordered 比较，门槛固定为
cosine ≥ 0.999、relative L2 ≤ 0.05。缺失的 15 点使用同一原生 prefix 路径单独补提取到
`replay/offline-only/`，明确记录 `compared_to_online=false`，不把它们计作在线核验通过。

实际结果如下，34 点每次均核验全部 32 个 site：

| 条件 | 最低 cosine | 最大 relative L2 | 判定 |
|---|---:|---:|---|
| cold | 0.9999685239 | 0.793903% | 通过 |
| warm | 0.9999866885 | 0.515979% | 通过 |
| reordered | 0.9999836834 | 0.571570% | 通过 |

共 3,264 次逐点/逐 site 比较。数值比较范围是 34 个已有 online 对照点，
其余 15 点仅完成 offline 提取，没有 online 数值可用于核对。
结果于 15:00:59 +08:00 返回；独立 replay 进程随后退出。

原始数值报告：[result.json](../hidden/reports/capture-verified200-20261018-143317/first-rollout/result.json)、
[逐层汇总](../hidden/reports/capture-verified200-20261018-143317/first-rollout/replay/report.json)、
[cold 逐点明细](../hidden/reports/capture-verified200-20261018-143317/first-rollout/replay/cold/comparison.json)。

## 产物与代码验证

远程核验目录：
`/home/wangtianyi/DyConGlobal/hidden/reports/capture-verified200-20261018-143317/first-rollout/`。

- `selected.json`：首条选择依据与时间；`audit.json`：结构、覆盖和完整性证据。
- `paired-events.json`：34 个 NAS 对照点；`missing-prefixes.json`：15 个缺失点的 native prefix。
- `replay/`：逐点/逐层误差、缓存条件汇总和离线补提取；`result.json`：各项独立判定。

新增脚本、冻结配置重放及 GPU 共享准入测试通过；runner 全套结果为 138 passed、7 skipped。
默认离线路径仍拒绝占用 GPU，共享模式需显式 free-memory floor 和 KV 字节预算。
