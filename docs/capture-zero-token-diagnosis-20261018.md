# 在线 capture 返回 token 0：隔离诊断

## 结论与调度修正

当前 Qwen3.5-9B 的原生 EasySteer decode capture 可以正常工作。已复现的
全 0 输出由强制单 token prefill 的执行路径触发，不能归因于 GDN 整体不支持
capture，也不是 `allowed_token_ids` 没有传入。

`--long-prefill-token-threshold=1` 影响请求的整个 prefill，不只影响 `<think>`。
目前仅采集生成阶段的 reasoning 换行和 `</think>`，因此使用原生 prefill
调度，decode 采集使用原生 FULL graph。无需为这些 anchor 重写调度器。

代码调整：

- `capture.prefill_token_chunk_size` 默认设为 `0`，不传 threshold 覆盖参数。
- 移除将单 token prompt chunk 强制派发为 uniform decode 的补丁。
- decode anchors 使用 `FULL_DECODE_ONLY`；保留模型支持的正常 prefill。
- 移除 capture 请求额外设置的 APC 读取禁用，保留原生策略和调用方显式 skip。
- 拒绝已复现出错的 Qwen 单 token prefill 配置；不再全面禁止 Qwen decode capture。

## 隔离实验

所有实验使用 FP16，在 GPU4–7 上运行，单进程 TP=1。原生实验绕过 DyCon HTTP
封装，并仅在请求完成后 fetch。每组包含 capture 关闭、开启但 selector 永不
命中、采集换行、重复请求、关闭 capture 后请求、自然生成开关对照。

| 对照 | prefill threshold | 结果 |
| --- | --- | --- |
| Qwen3.5 eager，原生 capture | 原生 | 固定输出全为 198，自然生成开关一致 |
| Qwen3.5 FULL decode，原生 capture | 原生 | 通过，92 次 capture graph replay，raw logits 全部有限 |
| Qwen3.5，开启项目 general plugin | 原生 | 通过 |
| Qwen3.5，不开启项目 general plugin | 1 | 开启 capture 后 hidden/logits 全 NaN，固定输出全为 0 |
| Qwen3.5，开启项目 general plugin | 1 | 复现同样异常 |
| Qwen3.5，APC 开启，不装同步探针 | 原生 | 输出对照通过；短 prompt 没有测得缓存命中 |
| Qwen3-8B，无 GDN，FULL decode | 原生 | 通过 |
| Qwen3-8B，无 GDN，项目 plugin | 1 | 通过 |

Qwen3.5 原生 FULL/eager 的自然生成各比较了 48 个 token，capture 开关之间
完全一致。这里的 FULL 指 decode capture，不代表 GDN prefill 也执行 FULL。

## 采样处的直接证据

失败请求首个已记录的采样步：

```text
final hidden:             4096 / 4096 个元素为 NaN
raw logits:              248320 / 248320 个元素为 NaN
allowed token count:     1
allowed token ID:        198
processed logits:        198 为 NaN，其余 248319 个值为 -inf
sampled token:           0
```

因此不是 hidden 全零，而是非有限前向结果进入 sampler。正常有限 logits 下，
同样的候选限制保留一个有限 logit，最终返回 198。NaN 输入下不能依赖正常的
采样约束保证；未约束的失败请求还会退化为重复 token 1023。

源码中，GDN builder 使用 `decode_threshold=1` 划分单 token chunk；正常 prefill
会根据 `prefill_has_initial_state` 将新序列的初始状态置零，decode 快路径直接
读取并更新 conv/recurrent cache。该执行差异与单 token prefill 的触发条件一致。
本轮尚未逐 kernel 标记第一次产生 NaN 的指令，不能把某个 dummy cache 槽位或
RecoverSSM 当成已证实原因；本次配置的 `use_kda_recoverssm` 为 false。

## 证据与复跑

远程工程 `/home/wangtianyi/DyConGlobal/state/` 中保留以下报告：

```text
capture-diag-20261018-085955-native-{full,eager}/report.json
capture-diag-20261018-090036-{dense-full,dycon-dispatch}/report.json
capture-diag-20261018-090325-{patch-only,chunk-only,native-apc-no-probe,dense-patch-chunk}/report.json
```

可复用诊断入口：

```bash
.venvs/easysteer/bin/python -m hidden.evaluation.diagnose_capture_tokens \
  --model /VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B \
  --gpus 4 --language-model-only --out state/capture-diag-new
```

添加 `--prefill-chunk 1` 可隔离复现旧触发条件；`--no-probe` 可排除同步探针掩盖
竞态的可能。诊断程序是独立入口，不给正常 rollout 安装 logits 探针。

相关 CPU／原生 SDK 回归为 22 passed。完整 runner suite 另有 7 个数据 scope
选择相关失败；未为本次 capture 修复修改 baseline 选择规则。

## 完整 HTTP 服务复核

`state/online-decode-regression-20261018-091010/` 保存了修正后的 GPU4–7、
TP=4、FP16 服务结果，完整 controller/plugin/异步导出均参与运行：

- 固定 token 对照正确，自然生成的 48-token 开关对照完全一致。
- 导出 77 条换行／结束标记样本，所有层数值有限，逐条 token 位置校验通过。
- 重复长 prompt 实际命中 1,056 个缓存 token。
- 四个 worker 均记录 107 次 capture graph replay，无 dropped token。
- `eager_capture_forwards` 为 `[2, 0, 0, 0]`，严格零 eager 验收仍失败。

原始 `report.json` 保持失败结果；`postmortem.json` 记录退出后对所有样本和
receipt 的独立检查，不能将本轮结果表述为完整 plan 全部验收通过。

## 两次 eager 的阶段定位

新增请求阶段边界记录后，在
`state/online-eager-phase-20261018-092024/phase-execution.json` 复现了同样的
107 次 FULL replay 和 2 次 eager。每个阶段结束才读取一次计数，不进行运行状态轮询。

| 阶段 | FULL replay 增量 | owner rank 的 eager 增量 |
| --- | ---: | ---: |
| engine warmup | 0 | 0 |
| capture 关闭的对照 | 0 | 0 |
| 短请求固定换行 decode | 15 | 0 |
| 自然生成 decode | 47 | 0 |
| 结束标记 decode | 15 | 0 |
| 长 prompt prefill 与短请求 decode 并发 | 15 | 2 |
| 长 prompt 缓存命中后的请求 | 15 | 0 |

长请求 prompt 为 1,465 tokens，短请求为 25 tokens，batch token 预算为 512。
混合 batch 无法匹配只支持 uniform decode 的 FULL graph；若该批还有生成阶段
的待采集行，当前 native capture 路径用普通 forward 执行 hook。后续纯 decode
继续 replay，回退不具有持续性。阶段审计尚未记录两批的逐 token 坐标或 GPU 时间。

`eager_capture_forwards` 在模型 forward hook 有待导出的采集行时才加一；graph
录制阶段不计入。目标 hidden 是 TP replicated，只有 owner rank 安装对应采集
hook，因此计数 `[2,0,0,0]` 不代表其他 rank 的同批模型计算仍在 FULL graph 中。

不能用 `2 / (107 + 2)` 推算耗时损失：两种计数的覆盖范围不同，batch 大小与
计算量也不同。混合 prefill 本身就不支持当前的 decode FULL graph；capture
额外的 hook、绕过编译路径与导出开销仍需同工作负载稳态 A/B 才能量化。
目前证据支持局部回退，不能据此声称整体性能下降某个确定百分比。

## 后续原生执行路径实测更正

在关闭 capture/steering 的当前 0.29.0 和未修改的上游 0.30.0 中，Qwen3.5
默认的混合 batch 实际使用 PIECEWISE CUDA Graph，并未跳过编译。当前工程
显式 FULL_DECODE_ONLY 会关闭该部分 graph 覆盖；capture 采集 hook 的 raw eager
又是额外的执行路径选择。因此，“不支持 mixed FULL”不能推导为“原生必然 raw
eager”，也不能据此认为当前 capture 对混合 batch 没有额外开销。

上游 0.30.0 的 Qwen3-8B 配合 Triton attention 已实际跑通 mixed FULL graph，
而 Qwen3.5/GDN 配合同一后端仍使用 PIECEWISE。完整的 6 组对照、原始 batch
记录和可复跑入口见[原生 vLLM 混合 batch 实测](native-vllm-mixed-batch-validation-20261018.md)。
