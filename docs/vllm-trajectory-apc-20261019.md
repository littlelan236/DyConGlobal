# 离线 trajectory APC 重放

更新日期：2026-10-19。正式 hidden、在线产物对照和首条轨迹验收统一使用现有 vLLM eager adapter 的连续 APC 实现。

## 行为与唯一接口

- 正式提取仍使用 `.venvs/analysis/bin/dycon hidden DATASET_ID NEW_CAPTURE_ID`。历史 queue launcher 仍按 `ENTRYPOINTS.md` 的批次范围运行。
- 在线产物对照仍使用 `.venvs/latent-vllm/bin/python -m hidden.evaluation.replay_online_artifacts`；`first_rollout` 自动调用该接口。
- 每个 anchor 提交其精确完整 token prefix。adapter 按 rollout/session/branch 建立请求依赖，同一轨迹的前一个 anchor 完成并取回 capture 后，才提交下一个；不同轨迹可共享原生提交窗口。
- 同一 engine 的 APC 跨应用窗口保留，必须开启 `hidden.vllm.enable_prefix_caching=true`。只有 trajectory APC 执行策略，没有慢速模式选择或回退。
- 原在线对照的逐窗口 `reset_prefix_cache()` 以及 cold/warm/reordered 三轮执行已删除。历史报告和 run 快照保持原样；GPU smoke 的重复/乱序输入属于契约测试。

vLLM 继续负责 continuous batching、chunked prefill、attention KV 分配/淘汰和 GDN/SSM recurrent-state 恢复。项目只控制请求依赖，不自行拼接、裁剪或保存 GPU cache。后一个请求可能仍需要 prefill 调度，但只对原生 cache 未命中的部分计算。

## 调用示例

先核对源 run、事件清单和 GPU 资源，再使用新的输出目录：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/latent-vllm/bin/python -m hidden.evaluation.replay_online_artifacts \
  --source-run SOURCE_RUN \
  --online-dir /VM/wangtianyi/DyConGlobal/hidden/online/SOURCE_RUN \
  --event-index hidden/reports/SOURCE_RUN/first-rollout/paired-events.json \
  --extra-prefixes hidden/reports/SOURCE_RUN/first-rollout/missing-prefixes.json \
  --output hidden/evaluation/outputs/NEW_REPLAY_ID \
  --set 'hidden.vllm.gpus=[4,5,6,7]' \
  --set hidden.vllm.tensor_parallel_size=4
```

资源覆盖必须符合该主机当时的 GPU 分配；TP 必须与 source run 一致。示例不自动允许占用中的 GPU。共享 GPU 诊断须显式配置 `allow_shared_gpus=true`、正的 `min_free_gpu_bytes` 和有界 `kv_cache_memory_bytes`，沿用已有 GPU 锁和准入检查，不能终止 serving。`--source-run` 加载原模型、generation、steering、layers 身份；当前配置只提供离线资源设置。

## 产物契约

在线对照的 `report.json` 升为 schema 2：

| 字段/目录 | 内容 |
|---|---|
| `replay_policy` | 固定为 `trajectory_apc` |
| `comparison` | 在线匹配点的逐层汇总、阈值、最差误差与通过状态 |
| `cache` | 全部实际执行样本的 prompt/cached/uncached token 数、命中率、capture 耗时合计 |
| `windows` | 每个原生请求波次的计时、缓存统计、sample/native request/位置/token 对应关系 |
| `samples` / `replayed_samples` | 在线匹配点数 / 实际执行总点数（含在线缺失点） |
| `trajectory-apc/` | 在线匹配点的 NPZ、index 和完整 `comparison.json` |
| `offline-only/` | 在线缺失点，明确 `compared_to_online=false`，不进入在线精度验收 |

匹配点和缺失点先合并为同一条按轨迹、step、absolute position 排序的执行流，每个点仅重放一次。旧缺失点清单通过对应 request ledger 和 finalization 重新绑定 rollout/session/step，不能把 request ID 当作独立轨迹，否则会破坏跨 step 的执行依赖。

正式提取保持原 NPZ/index/partial/manifest 结构，receipt 的 `replay` 为 `native_trajectory_apc`。运行时/实现指纹和 `replay_policy` 进入 backend 身份，新旧实现不能共用 capture ID 或 partial。常驻 queue engine 的每个 receipt 只包含本次 extraction 的执行 metrics，避免混入上一任务。

## 复用边界和指标含义

每个 sample 的 metrics 增加 `prefix_tokens`、`uncached_prompt_tokens`、`previous_sample_id`、`common_prefix_tokens` 和 `append_only`。前驱/LCP 是有界 CPU 诊断，最多保留 `max_inflight_requests` 条轨迹最近的 prefix；其淘汰不影响 GPU APC。实际缓存命中始终以原生 `cached_tokens` 为准。

非追加历史、压缩、分支等输入仍按精确 token prefix 提交；vLLM 只复用哈希兼容的公共部分。GPU cache 淘汰、原生块对齐、GDN 状态快照边界或进程重启都可能导致重算。最后一个输入 token 必须实际执行以采集 hidden，按原生对齐也可能重算更多 token。

`uncached_prompt_tokens=prefix_tokens-cached_tokens` 是未命中输入量，不等于包含抢占/重算的精确 GPU 工作量。`capture_seconds` 是原生波次耗时合计，包含 capture RPC/分页和该波次诊断；不包含模型身份核验、模型加载、前后处理全部耗时。

## 验证

CPU 契约与回归：

```bash
.venvs/latent-vllm/bin/python -m pytest -q \
  tests/test_vllm_hidden.py tests/test_vllm_native_capture.py \
  tests/test_replay_trajectory_apc.py tests/test_first_rollout.py \
  tests/test_compare_vllm_replay.py
```

覆盖同轨迹依赖、跨窗口缓存连续性、独立轨迹并发、非追加 prefix、session 分离、失败后不提交后续 anchor、缺失点按时序插入、finalization 身份、重复样本拒绝、逐任务 metrics、resume 和原生末 token 采集契约。数值验收沿用 dense `cosine >= 0.999`、`relative L2 <= 0.05`，保留逐层误差；不要求逐 bit 相等。

### 实际 TP=4 close-only 验收

使用 `capture-closeonly325-gpu47-w18-s16-bt2048-20261019-184246` 首条轨迹的原 47 个 exact close，Qwen3.5-9B、FP16、全部 32 个公共 site、TP=4、GPU 4–7 共享准入、每 GPU 2 GiB KV cache、token budget=512。engine 为新建，只执行单次连续重放；第一个 close 命中 0，后续 46 个 close 全部为正命中。

| 指标 | 本次结果 |
|---|---:|
| prefix token 总数 | 1,009,439 |
| 原生缓存命中 token | 959,376（95.04%） |
| 未命中输入 token | 50,063 |
| 47 个原生波次 capture 耗时合计 | 77.05 秒 |
| 命令总墙钟时间（含模型核验/加载/预热/输出/退出） | 291.77 秒 |
| 对在线 hidden 的最差 cosine / 相对 L2 | 0.9999798709 / 0.0063449082 |
| 对历史逐窗口 cold hidden 的最差 cosine / 相对 L2 | 0.9999843327 / 0.0055977273 |

两组数值对照均通过。此前相同 47 个 close 的 cold 记录为零命中、capture 1,112.28 秒；本次 capture 耗时约为其 1/14.4。两次为不同时间的共享 GPU 运行，该比值是观测值，不是受控负载下的吞吐保证；本次也未用重复热跑同一批 prefix 来冒充跨 close 复用。

证据位于 `hidden/evaluation/outputs/kv-reuse-validation-20261019-222729/`：

- `launch.json`、`exit.json`：完整命令、资源覆盖、PID、退出码和总耗时。
- `close47/report.json`、`close47/trajectory-apc/comparison.json`：缓存指标、逐 close 前驱/LCP/命中和逐层在线误差。
- `close47-vs-prior-cold.json`：与已有冷缓存产物的逐层误差；历史产物未修改。
- `cpu-tests.txt`、`ruff.txt`：八个相关 CPU 测试文件共 **60 passed、2 skipped**，静态检查通过。隔离推理环境缺少 matplotlib 的绘图回归已在 analysis 环境补跑（`analysis-regression.txt`，**1 passed**）；另一项是原有 queue 测试的显式 skip。合计 61 个不同测试通过，1 个原有测试跳过。

### 缺失点与混合长度 GPU 验收

同一证据目录还包含两组通过的 GPU 结果：

| 验证 | 范围 | 实际结果 |
|---|---|---|
| `missing49/report.json` | `capture-verified200-20261018-143317` 的 34 个在线匹配点 + 15 个在线缺失点，TP=4、全层 | 49 个不同 sample 各执行一次；前驱链连续，缺失点在序列中间执行；命中 836,880 / 872,741 token（95.89%），capture 51.92 秒；匹配的 34 点最差 cosine=0.9999820849、相对 L2=0.0059977953，通过 |
| `mixed-smoke/report.json` | TP=1、3 个独立轨迹、长度 129/2049/4097、3 个 site；重复、乱序和 singleton | 原生并发与输出映射通过；热窗口命中 5,280 / 6,275 token；最差 cosine=0.9997505571、相对 L2=0.0224475937，通过 |

`missing49-launch.json`、`mixed-smoke-launch.json` 保存具体命令和 PID。两次真实轨迹重放报告的四个实现文件指纹均与最终 adapter/capture/engine/extract 源码一致。全部测试引擎已通过自己的 shutdown 退出；最终 GPU compute PID 清单与测试前的 8 个 serving worker 一致。
