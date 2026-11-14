# Rollout 在线 Raw T=1 Logprob 采集

在线采集在正在生成 token 的同一次模型前向中，记录每个实际输出 token 的原始 T=1 概率。正式入口仍是 `dycon run` / `dycon rollout`，配置开关为 `capture.logprobs.enabled`，默认关闭。数据集构建将已发布 sidecar 绑定为中性的 `raw_t1_output_tokens` stream；训练用途由训练配置选择。

本文说明当前实现、使用方式和验证边界。完整 rollout、capture writer、NAS 发布的三组吞吐 pilot 尚未验收，FP16 在线值与 BF16 训练值的真实模型误差也尚未量化。历史测试结果不能作为新 run 的覆盖率或性能证明。

## 概率语义与支持范围

```text
raw_t1[t] = log_softmax(pre_transform_logits[t].float())[output_token_id[t]]
```

`pre_transform_logits` 位于 grammar mask 和 sampler processor 之前。归一化使用完整词表，最终只导出每个 sampled output token 的一个 FP32 标量。现有生成 temperature、top-p/top-k、penalties、thinking budget 和 grammar 继续决定采样；例如生成 temperature=0.6 时，记录值仍然是 raw T=1。该概率不等于经过采样变换后的历史行为概率。

当前接入 pinned vLLM 0.29.0 的 **V2 model runner**。请求启用后，runner 在 grammar mask 原地修改前复制该请求当前步的 logits；采样后用 Triton 分块 FP32 max/sum reduction 和 target gather 计算标量，随后释放单步缓存。不会再运行一次 prompt/completion 前向，也不生成全词表概率 artifact。内部复用 sampled-token logprob 输出槽，插件只传输对齐的目标值，不要求客户端设置 OpenAI `logprobs=true`。

配置要求 `capture.enabled: true`、`capture.export.enabled: true`、managed serving、TP=1，并拒绝启用 steering。首版的已测试模型路径为 Qwen3.5-9B FP16、单 worker、FULL CUDA Graph；其他模型、PP/DP/CP 拓扑及 adapter serving 不属于该次验收。采集请求要求 native `n=1`；并行候选沿用现有多个独立 `n=1` 请求。speculative decoding 的多目标采样路径会在 observer 中报错，不属于支持范围。

## 配置与正式入口

在创建新 run 前，将以下片段合并到根 `config.yaml` 的对应节点。它不是完整配置；保留已有 model、capture layers、anchors、GPU、端口及 export/NAS 参数。

```yaml
capture:
  enabled: true
  export:
    enabled: true
  logprobs:
    enabled: true
    probability_semantics: raw_t1_sampled_token_v1
    temperature: 1.0
    queue_size: 64
    max_request_tokens: 131072
    shard_max_bytes: 4194304
    spool_max_bytes: 1073741824
    failure_policy: record_and_continue
```

| `capture.logprobs` 字段 | 默认值 | 含义及校验 |
|---|---|---|
| `enabled` | `false` | 默认不为普通 capture 请求收集概率 |
| `probability_semantics` | `raw_t1_sampled_token_v1` | 仅接受该语义；不提供 processed 模式 |
| `temperature` | `1.0` | 固定评分温度，独立于生成 temperature |
| `queue_size` | `64` | 独立 writer 的有界待写请求容量，范围 1–4096 |
| `max_request_tokens` | `131072` | 单请求概率向量的 token 上限，不截断实际输出 |
| `shard_max_bytes` | `4194304` | 单请求 FP32 sidecar 的最大字节数 |
| `spool_max_bytes` | `1073741824` | 概率独立逻辑预算；物理 spool 仍受 `capture.export` 总预算治理 |
| `failure_policy` | `record_and_continue` | 仅支持记录概率采集失败并保留原生输出 |

启用时需满足 `max_request_tokens * 4 <= shard_max_bytes` 和 `2 * shard_max_bytes < spool_max_bytes`，为原子写入保留空间；预算还会计入临时写入开销。未知字段、processed 语义或非 1 评分温度会被配置校验拒绝。managed serving 启动时还会深度读取 checkpoint inventory，并核对冻结的 model fingerprint。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon run create RUN_ID --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon capture status RUN_ID
```

`RUN_ID`、`INSTANCE_ID` 为占位符。创建时冻结开关和预算，resume 使用原快照；修改当前根配置不会给历史 run 补上概率。已有 run 的显式配置修订遵循[统一配置修订](CONFIG-REVISION-CN.md)，不能手改 `resolved-config.json`。

## Request sidecar、发布与恢复

```text
vLLM V2 pre-grammar logits -> 原生采样 token + raw FP32 target scalar
  -> plugin 按 native request 对齐输出，终态批量传输 float32 payload
  -> capture 独立有界 writer -> requests/<request-hash>.raw_t1.f32
  -> 原 request JSON 的 raw_t1_probability locator/hash/身份
  -> 既有 spool/upload/session publication ACK
  -> rl-dataset build -> dataset receipt 内的中性 raw stream
```

概率 bytes 采用 little-endian float32，持久化数值成本为每 token 4 bytes，另有 receipt/索引开销。插件终态控制请求使用 base64 编码的 float32 bytes，可传递 `-inf`，ledger 不保留概率 JSON 数组；概率不混入 hidden activation NPZ。最终写入和上传在 GPU 采样路径外完成。

`raw_t1_probability` descriptor 记录 `status: sealed`、语义、固定温度、transform 前/target-only 标记、locator、dtype、count、byte size、SHA256、prompt/output token hashes、tokenizer、native request/run/session/decision/branch/step/API attempt 身份、source model/checkpoint inventory、serving dtype、runtime hash、实际 generation/grammar hashes 和 collector 版本。`sample_key` 等可选身份字段按来源实际可用性保存，可能为 null。

`sealed` 只证明本地 sidecar 写入封存；NAS 可用性还依赖原 capture 的上传 ACK 和 session publication。spool 发布时将 request ledger 与关联的 sealed sidecar 纳入同一文件集合；恢复也检查 sidecar bytes。缺失或校验失败的文件不能作为完整概率发布。相同 request、相同值的重试幂等；冲突的值不能覆盖首份 ledger/sidecar。

队列、token 上限、shard/quota、缺值或非法数值导致概率写入失败时，原生 token ledger 仍保留，descriptor 记录 `status: failed` 和受限错误摘要；该请求的概率保持不可用，不将缺值补零。概率失败本身不改变 branch 选择或 rollout 任务成功状态，普通 hidden capture 状态也不能替代 raw 覆盖率。NaN、+inf 和正的 logprob 被拒绝；`-inf` 保留并统计，训练 shared-anchor reader 要求所选值全部有限。

`capture recover RUN_ID` 复用现有已封存 bytes、spool 和上传任务，不请求模型、不补算缺失向量。历史缺少 raw 的轨迹应使用正式[离线 raw replay](../RL/dataset/README.md#replay-semantics)；不能把 processed artifact 改标签当作 raw。

## 数据集与共享读取合同

`rl-dataset build` 沿原 native request/branch 绑定在线 sidecar。概率本身的语义、长度、checksum、prompt/output hashes、request/branch、tokenizer、model/dtype 校验始终执行，即使数据集的一般 native source audit 配为 `audit.enabled: false`。

| 层次 | 字段或文件 | 含义 |
|---|---|---|
| branch | `probability_streams.raw_t1_output_tokens` | 原 sidecar descriptor 和 source receipt proof |
| branch | `raw_probability_available` | 此 branch 是否有可验证的在线 raw 值 |
| group | `raw_probability_complete` | 此组所有分支是否有在线 raw 值；独立于 token `data_ready` |
| dataset | `raw-t1-probability.json` | schema 2、中性语义、raw contract、scoring identity |
| dataset | `raw-t1-probability-branches.jsonl` | 固定 group/branch/token/mask 绑定与 `values_ref` |
| dataset | `shards/*` | 按 dataset `storage.max_shard_bytes` 重分块的 `<f4` 数值 |
| dataset | `dataset.json.probability_streams.raw_t1_output_tokens` | `manifest`、`index` 与 `dataset_binding`，由 dataset receipt 覆盖 |

没有任何可用在线值时，dataset 不声明该 stream。只有部分分支有值时可封存部分索引，原 group/token inventory 仍保留；训练显式选择该 stream 后，要求所有训练 eligible branches 完整覆盖，缺值报错，不静默改资格或回退模型前向。当前 builder 拒绝同一 stream 中混合不同 scoring identities。

```bash
.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml
.venvs/runner/bin/dycon rl-dataset check DATASET_ID --dataset-root DATASET_ROOT --level full
.venvs/runner/bin/dycon artifact verify DATASET_ID --scope rl-dataset --storage-root DATASET_ROOT
```

`check --level full` 还会沿在线 source proof 校验原 ledger/sidecar；训练中性 reader 读取的是 dataset receipt 内的 manifest/index/值副本。`RL.dataset.online_probabilities.OnlineProbabilityReader` 用于追溯来源，`RL.dataset.probability.ProbabilityStream` / `open_probability_stream` 是训练与离线 raw artifact 共用的中性读取合同，详见[概率 stream 合同](../RL/PROBABILITY_STREAM.md)。

训练侧在自己的 YAML 中选择：

```yaml
grpo:
  reference_cache:
    source: dataset_probability
    stream: raw_t1_output_tokens
    dtype_compatibility: fp16_source_bf16_training
```

该片段合并到现有训练配置；完整参数和兼容性见 [GRPO 指南](../RL/GRPO.md)。采集端记录实际无 adapter 的基座身份，训练端判定 fresh LoRA 或 warm-start 是否兼容。一个导入缓存可承担 ratio 分母和 K3 reference，current 仍实时计算；这些角色仅属于训练 attempt。允许 FP16→BF16 不表示初始 ratio 必须为 1，也不代替误差量化。

## 与缺失离线补评统一对齐

部分在线值可以直接复用。`rl-dataset probability-plan --probability-config YAML`
对已封存 dataset 冻结 target/missing 清单和来源 receipts；在现有 replay YAML
绑定 `selection.coverage_plan` 后，只提交缺失 native branches。再通过
`probability-build --probability-config YAML` 将在线和离线值封存为
`ROOT/RL/probabilities/ID/` 的 schema-3 `rl-probability`，online-only 导出也走此入口。
无 missing 时不启动 GPU worker；不是重新采集、改写历史 ledger 或重算已有在线值。

原 branch `raw_probability_available` 和 group `raw_probability_complete` 继续描述
dataset 构建时的在线状态；补齐后的总体 coverage 由新 artifact 表达。在线
`raw_t1_sampled_token_v1` 经显式 raw contract adapter 转成统一消费语义
`raw_t1_output_tokens_v1`，原始语义和 receipt bytes 保留。每条分支的 origin、
来源 receipt/value hash 和 execution cohort 均可追溯；离线分支保留 replay estimate
和 `historical_behavior_verified=false`。同一 scoring function、相同 collector
compute dtype 才能合并；多个 runtime cohorts 需显式允许，不能以 FP32 存储格式
推定其计算口径相同。

训练可只绑定最终 `scope: rl-probability` 的 ID/root；本地验证和 checkpoint cache
恢复不需要原 NAS 在线。完整 CLI、配置、缺值/冲突和迁移说明见
[统一概率产物](../RL/dataset/PROBABILITY_ARTIFACTS.md)；本次 CPU 合同验收见
[对齐验收报告](../RL/PROBABILITY_ALIGNMENT_VALIDATION.md)。

## 验证证据与待验收事项

2026-11-11 的工程测试日志位于项目 `temp/grpo_online_raw_logprobs_20261111/`，属于当时工作树的证据，并非当前版本的不可变性能报告：

| 检查 | 已有结果 | 证据 |
|---|---|---|
| capture/native receipt、dataset、sidecar 回归 | 70 passed | `final-regression.log` |
| FP16 / FP32 / BF16 固定 logits GPU oracle | 3 passed | `gpu-oracle-2.log` |
| Qwen3.5-9B FP16、FULL graph、固定 seed 与 grammar | 直接 vLLM 生成路径通过 | `real-model-tests-2.log`、`real-model-tests-3.log` |

真实模型测试通过直接 `LLM.generate` 启动，禁用了 serving plugin；它验证采样和概率输出，不能单独证明 managed plugin→capture writer→NAS 的完整链路。synthetic 测试分别覆盖 sidecar 完整性、writer 上限、重复/冲突、abort/缺值、NAS 上传/恢复和共享 dataset reader。CPU/GPU 测试命令分别使用对应环境：

```bash
.venvs/runner/bin/python -m pytest -q tests/test_online_raw_logprobs.py tests/test_rl_dataset.py

# 需要预先选定空闲 GPU；GPU_INDEX 是占位符。
CUDA_VISIBLE_DEVICES=GPU_INDEX DYCON_ONLINE_RAW_GPU=1 \
  .venvs/easysteer/bin/python -m pytest -q tests/test_vllm_raw_t1_gpu.py -k observer

# 可选真实模型独立测试，不是业务 run 或 rollout 性能 pilot。
CUDA_VISIBLE_DEVICES=GPU_INDEX DYCON_ONLINE_RAW_GPU=1 DYCON_ONLINE_RAW_MODEL_TEST=1 \
  .venvs/easysteer/bin/python -m pytest -q tests/test_vllm_raw_t1_gpu.py -k real_qwen
```

真实模型测试包含三次重复的 generation-only 计时，`generation-pilot*.json` 不覆盖 writer、NAS 和业务 rollout 墙钟，不能据此宣称端到端性能通过。当前尚未创建用于本功能的业务 run，正式三组 pilot 已按用户要求暂缓；未设定已获确认的 10% 或其他吞吐预算。真实在线 FP16 与训练 BF16 的 delta/ratio/K3、长上下文、完整 managed 生命周期仍需后续验收。

vendor 源在 `vendor/easysteer-vllm/`，对应 hash pin 在 `vendor/easysteer-runtime.json`。更新 overlay 应走项目的 `runtime/scripts/install_easysteer_overlay.py`，用 `.venvs/easysteer/bin/python runtime/scripts/verify_easysteer_runtime.py` 核对 vendor 与 installed 文件；不要只修改 site-packages。源码位置为 `vllm/v1/worker/gpu/model_runner.py`、`sample/sampler.py` 和 `sample/raw_t1.py`；采集/存储位于 `runtime/serving/`，dataset 绑定位于 `RL/dataset/online_probabilities.py`。
