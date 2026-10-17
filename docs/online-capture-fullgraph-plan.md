# DyConGlobal 在线 FULL-Graph Hidden Capture 方案

本方案直接实施第二阶段：在线 rollout 使用 EasySteer 已有 capture graph、vLLM 原生 scheduler 和 KV/APC/GDN 状态管理，复用 `<think>` 之前的安全 cache block，只重新计算包含目标 token 的 suffix，并采集 `<think>`、`</think>` 的多层或全层 hidden。

普通 rollout 默认完全关闭 capture；只在 `capture.enabled=true` 时启用。

## 1. 现有服务与改造原因

当前结构是：

```text
rollout runner --OpenAI HTTP--> ServingManager 启动的 vLLM API 子进程
                                      ├─ scheduler / APC / GDN state
                                      ├─ EasySteer steering hooks
                                      └─ EasySteer capture worker RPC
```

EasySteer 已实现 `CaptureSession`、hidden/tact state hooks、residual 组合、request/position/token labels、TP 组装、FULL capture graph、固定 GPU buffer 和分页导出。问题是 HTTP rollout runner 不能直接调用进程内 `LLM.llm_engine.collective_rpc()`。

因此新增的是同一 vLLM engine 内的 capture control plane，而不是新的 scheduler、KV manager 或 hidden hook：

```text
普通 OpenAI data plane
capture control plane: /dycon/capture/session/{start,stop,status,results}
                         ↓
                    同一个 EngineCore
```

不把 rollout runner 和 engine 合并进一个 Python 进程，保持当前进程隔离和故障边界。

## 2. 配置开关

唯一 `config.yaml` 增加：

```yaml
capture:
  enabled: false
  backend: easysteer
  mode: online
  graph_mode: FULL
  anchors: [think_open, think_close]
  layers: all
  dtype: bfloat16
  prefix_cache_policy: safe_boundary
  capture_stage: post_steering
  failure_policy: record_and_continue
  export:
    enabled: true
    root: hidden/online
    format: npz
    queue_size: 256
    flush_interval_ms: 100
```

`enabled=false` 必须保持原普通 rollout 配置 hash、请求格式、steering、容器和 manifest 行为。`enabled=true` 必须使用 managed pinned EasySteer server、V2 runner、`enforce_eager=false` 和 `graph_mode=FULL`。`layers=all` 表示 decoder sites 0..N-1 加 final norm site N；也可配置显式 layer 列表。

## 3. 请求绑定

capture rollout 的每次请求携带：

```json
{
  "dycon_capture": {
    "session_id": "cap-...",
    "task_id": "...",
    "run_id": "...",
    "rollout_id": "r-...",
    "step_id": 12,
    "anchors": ["think_open", "think_close"]
  }
}
```

server 将它与 vLLM 内部 request ID 绑定。不能用 HTTP 顺序、响应顺序或 batch 顺序推断轨迹对应关系。

## 4. Qwen anchor 语义

四个当前 Qwen checkpoint 的 prompt 末尾均为：

```text
<|im_start|>assistant
<think>\n
```

其中 `<think>` token 是 `248068`，换行 token 是 `198`。所以 `<think>` 是 prompt 的 `-2`，不能选择 `-1`：

```python
SelectSpec(prompt_positions=[-2])
```

server admission 必须校验：当前 marker 属于本次 assistant generation、`prompt[-2]==248068`、`prompt[-1]==198`。模板不匹配时记录 `think_open_unavailable`，不能静默采集最后一个 token。

`</think>` 必须作为后续 forward 的 input 才能得到 close-token-as-input hidden。若 close 后继续生成 tool call，通常可捕获；若 response 在 close 后直接结束，记录 `no_followup_forward`，不能用前一个 token 替代。

## 5. FULL capture graph

启用 capture 时使用：

```text
enforce_eager=false
cudagraph_mode=FULL
VLLM_USE_V2_MODEL_RUNNER=1
```

复用 EasySteer `CaptureSession`、`prepare_capture_graph`、`CaptureGraphState`、固定 buffer、`fetch_pages` 和 TP label assembly：

```text
capture idle                 → 保持普通 FULL/compiled graph
需要 capture 且 batch 合法   → 独立 FULL capture graph replay
需要 capture 但 batch 不合法 → 当前 batch 回退 eager
不需要 capture               → 原 rollout 路径
```

每次 session 记录真实 `graph_replays`、`eager_capture_forwards` 和 fallback reason。固定 layer 集合并预热，避免运行中反复重录 graph。FULL prefill 是否命中取决于 attention backend、chunked prefill、TP、batch geometry 和显存，必须用真实模型测试确认。

## 6. safe-boundary APC

当前 `prompt_positions=[-2]` 的保守策略可能跳过该请求的 prefix-cache read。第二阶段将目标边界传给 vLLM request：

```json
{
  "capture_target_position": 1477,
  "capture_cache_boundary": 1476
}
```

cache lookup 使用：

```python
max_cache_hit_length = min(
    request.num_tokens - 1,
    request.capture_cache_boundary,
)
```

语义是：

```text
允许复用：<think> 之前的完整安全 cache blocks
必须执行：安全边界到目标 token 的 suffix
```

不自行读取、拼接、释放 KV，也不自行恢复 GDN state。继续使用 vLLM block coordinator、block 对齐、hybrid cache groups、GDN state、eviction、preemption 和 chunked prefill。由于 Qwen3.5/3.6 是混合 attention/GDN，第一版允许向前扩展到最近安全 block；只有各 cache group 状态一致后才缩短边界。

该修改放在独立 EasySteer/vLLM runtime，通过 request schema、request processor 和 cache lookup 传递，普通 serving 默认不加载。

## 7. 不重复实现的边界

继续复用：

| 功能 | 实现 |
|---|---|
| batch 调度 | vLLM scheduler |
| KV/GDN cache | vLLM cache manager |
| hidden hook/residual | EasySteer capture session |
| FULL graph | EasySteer capture graph |
| TP/labels/paging | EasySteer capture serialization/fetch |
| steering graph | EasySteer steering hooks |

DyConGlobal 只新增 control endpoint、metadata 绑定、anchor validator、cache boundary 字段、异步 export queue 和 online manifest。禁止新增 scheduler、padding、KV manager、GDN 恢复或通过 HTTP 顺序绑定 request。

## 8. 在线产物

```text
hidden/online/<capture_session_id>/
  tensors/*.npz
  index.json
  receipt.json
  metrics.json
```

每个 sample 记录 `sample_id/task_id/run_id/rollout_id/step_id`、anchor 文本和 token ID、绝对位置、内部 request ID、trajectory SHA-256、model fingerprint、dtype、graph execution、cache hit tokens、cache boundary、hidden path。receipt 记录 serving/rollout config hash、runtime hash、TP、anchor 定义、graph replay/fallback、队列积压和失败原因。

`enrichment/manifest.json` 与 `hidden/manifest.json` 在同一锁内增加 `online_captures`，不覆盖历史迁移产品和离线 capture。steering 同时开启时默认记录 `capture_stage=post_steering`，要求 steering 使用 `in_graph`。

GPU capture buffer 通过 non-blocking D2H copy 进入 pinned staging buffer，再异步进入有界 export queue；不在模型 forward、decode scheduler 或 capture RPC 中同步压缩和写盘。后台 writer 负责 NPZ 压缩、临时文件写入、原子 rename、sidecar checksum 和 index 更新。capture control endpoint 只等待数据成功进入 staging/queue，不等待磁盘完成。

每个 sample 的状态分为 `captured`、`queued`、`persisted`、`failed`。只有 tensor、sidecar 和 index 都完成原子提交后，manifest 才记录 `persisted`；rollout receipt 可以先记录 `queued`，服务退出时由 recovery worker 扫描 queue journal 并补交或标记失败。队列满时默认使用 `record_and_continue`：丢弃尚未进入队列的 latent 并记录 `export_backpressure`，不阻塞 rollout；严格实验可以配置 `block_rollout`，但这不是默认策略。

## 9. 实施阶段

### A. Capture control plane

1. 在 managed server 增加 start/stop/status/results endpoint。
2. 将 HTTP metadata 绑定内部 request ID。
3. 调用现有 EasySteer capture RPC，不修改 scheduler。
4. 增加 session 清理、结果分页和失败 receipt。

### B. FULL prefill `<think>`

1. 校验真实 prompt 的 `[-2]=248068`、`[-1]=198`。
2. 少量层启用 FULL prefill capture，确认 `graph_replays` 增加。
3. 逐步扩展到全层，测量 graph buffer 和 KV capacity。
4. 验证 capture 开关对 logits、tool call 和 rollout 输出没有改变。

### C. safe-boundary APC

1. 扩展 request schema 和 request processor。
2. 在原生 cache lookup 限制最大命中长度。
3. 验证 dense/MoE、TP=1/2、chunked prefill、冷/热 cache。
4. 检查 `<think>` 没有被 cache skip，且各 GDN/cache group 状态一致。

### D. `</think>` 和 steering

1. 验证 close token 后续 decode input 捕获。
2. 记录无 follow-up forward 的样本。
3. 验证 post-steering capture 和 steering `in_graph`。
4. 逐步开启全层与双 anchor。

## 10. 测试和性能验收

单元测试覆盖 `<think>` 位置、历史 marker 排除、close follow-up、metadata 绑定、boundary 合法性和 manifest。EasySteer/vLLM 集成测试覆盖 FULL prefill/decode、graph replay、eager fallback、APC、chunked prefill、不同长度混批、TP 和 MoE。

性能比较：

```text
A 普通 rollout
B FULL capture、无 boundary APC
C FULL capture + safe-boundary APC
D C + steering
```

记录 wall time、prefill/decode tokens/s、graph replay、eager fallback、cache hit tokens、峰值显存、KV capacity、capture complete rate 和 export backlog。

进入全量采集的门槛：四模型 `<think>` token 正确；close unavailable 可解释；目标 batch 真实 FULL replay；safe boundary 不跳过目标；rollout 输出与 capture 关闭时一致；steering 协议不变；异常可回滚；普通 rollout 回归全部通过。

## 11. 回滚

```yaml
capture:
  enabled: false
```

回滚必须停止 session、移除 capture hooks、释放 graph buffer、清理 export queue、保留失败 receipt，并恢复普通请求。cache boundary 有独立开关，关闭后回到原生 vLLM cache lookup。任何 capture 失败不得终止其他 rollout 或 steering 进程。
