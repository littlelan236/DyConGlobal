# 原生 vLLM 混合 batch 执行路径实测

## 已确认的结论

关闭 capture 和 steering 后，当前机器上的 Qwen3.5-9B 原生默认路径是：
纯 decode 使用 FULL CUDA Graph；prefill 和混合 batch 使用 PIECEWISE CUDA Graph。
“混合 batch 无法使用当前 GDN 的 FULL graph”不能推导出“原生一定使用 raw eager”。

本机自动选择的 FlashAttention 后端也有 uniform-batch 限制，普通 attention 的
Qwen3-8B 在该后端下同样使用 FULL_AND_PIECEWISE。因此，应按模型中所有计算层、
attention 后端、vLLM 版本与 graph 配置共同判断，而不能归为整个 Qwen 家族的
数学固有属性，也不是所有 vLLM 模型的通用限制。

## 环境和隔离

- GPU4–7，各进程 TP=1，FP16；原生 APC 和 chunked prefill 开启。
- `max_num_batched_tokens=512`，短、长请求一起提交，每条输出 32 tokens。
- `VLLM_PLUGINS` 为空，无 DyCon HTTP/plugin、无 hidden capture、无 steering。
- 当前 EasySteer/vLLM 0.29.0 的关闭扩展路径，与未改动的上游 vLLM 0.30.0 分别测试。
- 上游 0.30.0 直接从已有 wheel 缓存加载；按 wheel RECORD 校验 2,681 个 Python
  源文件，全部匹配。未覆盖当前工程的 vLLM 安装或修改 runtime pin。
- 初始化后安装观测包装，记录真实调度的请求阶段、query 长度、派发的 graph 模式、
  `skip_compiled`，以及 `aot_compiled_fn` 的实际调用。包装不改变调度与 forward 参数。

## 实测结果

| Runtime / 模型 | 请求的 graph 配置 | 实际配置 | 冷混合 batch | warm 混合 batch | 纯 decode |
| --- | --- | --- | --- | --- | --- |
| 当前 0.29.0 / Qwen3.5-9B | 默认 | FULL_AND_PIECEWISE | 5 次 PIECEWISE | 1 次 PIECEWISE | 每轮 31 次 FULL |
| 当前 0.29.0 / Qwen3.5-9B | FULL_DECODE_ONLY | FULL_DECODE_ONLY | 5 次 NONE | 1 次 NONE | 每轮 31 次 FULL |
| 上游 0.30.0 / Qwen3.5-9B | FULL | 自动降为 FULL_AND_PIECEWISE | 5 次 PIECEWISE | 1 次 PIECEWISE | 每轮 31 次 FULL |
| 上游 0.30.0 / Qwen3-8B | FULL，默认 FlashAttention | 自动降为 FULL_AND_PIECEWISE | 3 次 PIECEWISE | 1 次 PIECEWISE | 每轮 31 次 FULL |
| 上游 0.30.0 / Qwen3.5-9B | FULL，Triton attention | 自动降为 FULL_AND_PIECEWISE | 5 次 PIECEWISE | 1 次 PIECEWISE | 每轮 31 次 FULL |
| 上游 0.30.0 / Qwen3-8B | FULL，Triton attention | FULL | 3 次 FULL | 1 次 FULL | 每轮 31 次 FULL |

所有实验均实际观察到混合 batch，冷／热输出限制检查全部通过。

最后两项使用相同上游 runtime、相同 Triton attention 后端和 FULL 请求配置。
Qwen3-8B 的纯 prefill 也实际使用 FULL。Qwen3.5 的日志明确指出
`GDNAttentionBackend` 支持级别为 `UNIFORM_BATCH`，因而自动降级。这说明当前
Qwen3.5/GDN 实现的限制确实存在，但不会限制其他模型使用 mixed FULL graph。

Qwen3.5 的一批真实记录：

```json
{
  "is_prefilling": [false, true],
  "query_lengths": [1, 511],
  "cg_mode": "PIECEWISE",
  "need_eager": false,
  "forward_contexts": [{"cg_mode": "PIECEWISE", "skip_compiled": false}],
  "compiled_calls": ["language_model.model"]
}
```

显式 FULL_DECODE_ONLY 的对应记录是 `cg_mode=NONE`，但 `skip_compiled=false`，
且仍调用 `language_model.model` 的 AOT 编译函数。因此，它是“没有 CUDA Graph 的
已编译前向”，不能与 capture 路径的 `skip_compiled=true` raw eager 混为一谈。

## 对当前 DyConGlobal 的含义

当前 capture 启动配置显式选择 FULL_DECODE_ONLY，确实改变了原生默认的 graph
覆盖范围：即使不需要采集的 prefill，也失去了默认 PIECEWISE graph 路径。
此外，当混合 batch 包含待采集的 generation 行时，当前 EasySteer capture
扩展会再绕过编译路径执行 hook。这两点都不能表述为原生调度不可避免的损失。

更合适的基础配置方向是保留 FULL_AND_PIECEWISE，让纯 decode 使用采集 FULL
variant，无采集需求的 prefill/mixed 保留原生优化；混合 batch 中的 hidden
采集仍需要单独处理与验收。本次实验只验证原生行为，未据此修改生产 serving 配置。

## 复跑与证据

```bash
.venvs/easysteer/bin/python -m hidden.evaluation.diagnose_native_dispatch \
  --model /VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B \
  --gpus 4 --graph-mode AUTO --output state/native-dispatch-new
```

上游测试添加 `--runtime-root /home/wangtianyi/.cache/uv/archive-v0/FFydlu_RpDnA032n`。
该路径是本机已校验的 0.30.0 wheel 缓存，不应照搬到其他主机。

原始报告位于远程工程：

```text
state/native-dispatch-20261018-094024-pinned-auto/report.json
state/native-dispatch-20261018-094024-pinned-decode-only/report.json
state/native-dispatch-20261018-094024-upstream-qwen-gdn/report.json
state/native-dispatch-20261018-094024-upstream-dense/report.json
state/native-dispatch-20261018-094339-upstream-gdn-triton/report.json
state/native-dispatch-20261018-094339-upstream-dense-triton/report.json
state/native-dispatch-summary-20261018.json
```

耗时包含观测包装和首次 graph 初始化，本轮不是吞吐 A/B，不能用其耗时估计
capture 的稳态性能损失。
