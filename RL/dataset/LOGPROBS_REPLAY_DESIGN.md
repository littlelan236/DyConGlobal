# 历史采样 token logprobs 重放提取设计

日期：2026-11-10。所属计划：[PLAN.md](PLAN.md)。状态：设计，尚未实现新 CLI、启动 GPU 或提取真实数据。

## 当前阅读入口（2026-11-11）

本文件保留 2026-11-10 的 processed 历史设计，不能据此配置当前 raw 提取或训练。
当前 raw replay schema 2、在线 dataset stream 及生产限制见 [README.md](README.md)；
中性概率合同见 [PROBABILITY_STREAM.md](../PROBABILITY_STREAM.md)。训练侧已通过
`grpo.reference_cache` 显式选择封存 raw 产物/stream，由一份冻结 anchor 同时承担
ratio 分母和 K3，current 实时计算；旧 processed 产物只能诊断，不能作为训练 anchor。
配置与恢复见 [GRPO.md](../GRPO.md)，工程验收及真实 FP16/BF16 待验收边界见
[GRPO_BINDING_VALIDATION.md](../GRPO_BINDING_VALIDATION.md)。

### 统一产物与 missing-only 补评已实现

当前最终消费格式为 schema 3 `rl-probability`，独立于来源 replay 的 schema 2。
`probability-plan` 冻结 coverage，既有 `replay-logprobs` 的 coverage-plan selector
只补缺失分支，`probability-build` 复用在线 bytes 并统一封存。
配置/迁移见 [PROBABILITY_ARTIFACTS.md](PROBABILITY_ARTIFACTS.md)，348 项 CPU 回归
与真实模型待验收边界见 [PROBABILITY_ALIGNMENT_VALIDATION.md](../PROBABILITY_ALIGNMENT_VALIDATION.md)。
以下 processed 设计仍作为历史记录保留。

## 1. 已确定的输出口径

用户确定：重放历史 token，提取**考虑 temperature、top-p、top-k 的采样分布 logprobs**。这些参数在项目根配置中设置且一直不变，可以直接使用。这里将用户表述的 logp/logk 对应到工程字段 `top_p/top_k`。

每个生成位置仅保存历史实际生成 token 的一个自然对数概率，float32；不保存全词表 logprobs，不重新随机生成轨迹，不调用工具或复跑 SWE-bench 环境。

设原始 prompt token 为 P，历史 completion 为 C，位置 t 的目标：

```text
ell[t] = log q_history(C[t] | P, C[:t])
q_history = 历史模型输出经过实际采样处理后重新归一化的分布
```

包括 reasoning、tool action 的原始序列化、实际生成的结构/终止 token，与基础 dataset 的 completion 和监督坐标一一对应。所有有效候选分支都处理，不只重放 winner 或 retained 分支。

这里的“还原”依赖历史模型权重、原始 token 和采样处理规则的重放。仅保存某 token 的 raw logprob，不能反推出经过温度和 nucleus 截断后的概率。产物标为 `replay_processed_logprobs`，与生成时直接采集的 `native_recorded_logprobs` 区分。

## 2. 已核实的可复用机制

当前固定 runtime 为 vLLM **0.29.0**；源码 pin 位于 `vendor/easysteer-runtime.json`。以下能力已在隔离 `.venvs/latent-vllm` 的实际源码中核对，尚未做本任务的 GPU 实测：

| 接口 | 实际能力 | 设计用途 |
|---|---|---|
| `ModelConfig.enable_trace_replay` | 开启 trace replay，要求 Model Runner V2 | 提取 engine 显式开启 |
| `SamplingParams.trace_decode_token_ids` | 每一步用指定历史 token 替换原采样 token | 令模型和请求历史沿原始 completion 推进 |
| V2 `Sampler.__call__` | 先产生 processed logits，再应用 trace token，再针对该 token 计算 logprobs | 不必先把非目标 token 屏蔽，保留真实采样归一化 |
| `logprobs_mode="processed_logprobs"` | 返回处理后的归一化 logprobs；默认值其实是 raw_logprobs | 必须显式指定并检查生效 |
| `SamplingParams(logprobs=0)` | 只取实际输出 token；0 不等于关闭，None 才关闭 | 不返回额外 top-N，不请求全词表 |
| V2 `compute_token_logprobs` | 扫描 logits 做 max/logsumexp，仅输出目标 token 值，不物化全词表 logprobs 矩阵 | 直接复用现有 kernel |
| `hidden/backends/vllm/engine.py` | 隔离解释器、GPU reservation、runtime pin 检查、受控 engine 参数 | 提炼/复用共享部分，不复制第二套资源管理和 runtime 校验 |

**关键选择：首版以 native V2 trace replay 为正式后端。** 不重写 top-p/k 算法，不使用重新随机采样后匹配原 token 的方法，也不以 Transformers 的全序列 raw CE 冒充采样分布概率。

### 已存在的 trace 限制

当前 `SamplingParams._validate_trace_replay` 明确拒绝同时使用：

- `prompt_logprobs`、`n != 1`、空 trace 或越界 token；
- speculative decoding；
- structured outputs/grammar、repetition_detection、thinking_token_budget、bad_words。

因此该机制不是对所有历史请求立即通用。首版只对没有上述约束的来源进行正式提取；遇到不支持的来源必须报告具体字段，不能删除约束后悄悄继续，也不能把关闭约束的结果标作历史采样概率。支持扩展见第 10 节。

另外，vLLM 明确规定：**prompt_logprobs 即使配置 processed 模式，仍然不会经过输出采样 processors。** 不能将 `P || C` 整体当 prompt，直接把 prompt_logprobs 当成本任务结果。

## 3. 参数与模型的来源

### 3.1 根配置提供 temperature/top-p/top-k

正式入口沿用项目 `--config` 的根配置解析，提取并冻结：

```yaml
generation:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
```

以上为本次检查的实际值。抽查历史 Qwen3.5-9B run 的这三个值相同。用户已确认其长期不变，因此不用另建 replay YAML 中的三份可覆盖值，也不要求用户重复填写。

仍执行轻量一致性检查：若来源 run 有这些字段，就逐字段比较并保存审计结果；若来源缺字段，记录 `root_config_user_asserted_constant`，不捏造旧快照。若发生明确数值冲突，报告差异并停止该 cohort 的历史资格提取；不擅自选择一份或覆盖历史证据。此检查不会修改根配置。

保存根配置文件 SHA、解析后这三个值的 sampling hash，以及用于概率计算的最小相关快照；不把连接凭据或 API key 带入 replay 产物。参数在任务启动时冻结，处理中根配置改变不影响本 attempt，也不影响其 resume。

### 3.2 历史模型与其余历史约束

根配置不是历史模型的身份来源。本次检查：当前根配置模型为 Qwen3.6-27B，已抽查的历史 SWE-bench 数据来自 Qwen3.5-9B。提取必须加载各来源真实模型，核对 model content fingerprint、配置/权重分片 SHA、tokenizer fingerprint、adapter/量化/steering 和 dtype 等。

其余会影响分布的设置，包括 min_p、presence/frequency/repetition penalties、logit bias、allowed tokens、min_tokens、grammar、thinking budget、bad words、EOS 和请求级 overrides，要从历史实际 request / 冻结配置及该 runtime 的默认值解析，构成 `effective_sampling_policy`。若请求 overrides 与 run 配置不同，以有证明的实际请求为准，并记录差异。

对历史证据没有保存的字段，仅在能证明当时 runtime 默认值时补齐；无法证明则记为 unknown，不能将今日配置默认值当作历史事实。

抽查历史 run 的 `min_p=0`、presence_penalty=0、repetition_penalty=1、thinking_token_budget=null；当前根配置有 thinking_token_budget 和 require_reasoning_close。后二者不能直接套到旧请求。本任务只采用用户确认固定的三个采样参数，其余维持历史语义。

按历史模型/adapter/tokenizer、effective sampling、约束和 runtime 身份分 cohort。可复用一个模型驻留处理若干兼容请求，但不能混用某个 run 的模型或默认值。

## 4. 正式执行链路

```text
immutable dataset + selection/receipt
  -> 校验 native prompt/output 与分支身份
  -> 冻结根配置采样参数，解析历史模型与完整请求约束
  -> 分 cohort 资格预检 / GPU 与容量预算
  -> native V2 trace replay
  -> 按历史 token ID 取 processed logprob
  -> token/长度/支持集/有限值检查
  -> 分片写入 + 独立 replay receipt
  -> dataset loader 显式绑定 replay ID
```

### 4.1 输入校验

每个 branch 读取 P 和 C，核对 dataset receipt、ledger 文件 SHA、prompt/output hash、native request ID、decision/branch/API-attempt，以及 model/tokenizer 身份。token 直接取 native IDs，不 decode 后重新 tokenize，不从终态 messages 拼接输入。

组内共享 P，但每条分支独立拥有输出 token 历史、penalty/grammar/recurrent 状态。保留 action 相同但 reasoning 不同的样本，保留同组重复输出的原始采样 multiplicity。已经在基础 dataset 判定为 alias 的来源复用同一逻辑结果，不再次增加权重。

### 4.2 Engine 构建

通过项目正式 CLI 的受控 dispatcher 进入 `.venvs/latent-vllm` 隔离 runtime，先设置 GPU/缓存环境再 import Torch/vLLM；复用 GPU 锁、可用性和 runtime pin 校验。不在 `.venvs/rl` 或 rollout serving 环境临时安装第二套推理依赖，不连接正在服务的 rollout endpoint。

显式 engine 能力：

```text
Model Runner V2
enable_trace_replay = true
logprobs_mode = processed_logprobs
historical model / adapter / dtype / tokenizer
generation_config = vllm，避免隐式模型 generation_config 覆盖
speculative decoding = disabled
```

默认先用 eager 和保守并发建立验收基准；性能选项、TP、attention/GDN kernel、缓存和调度设置全部记录。只读提取不使用 optimizer，不注入新 LoRA，不打开训练 dropout，不应用当前训练中的 adapter。

现有 hidden engine 的参数直接绑定 `hidden.vllm`，不能把它整份照搬成 RL replay 配置。应将通用隔离、身份和资源逻辑提炼为共享构件，由 replay 传入自己的显式 runtime 配置，保留原 hidden 入口行为。

### 4.3 每条 branch 的 replay 请求

以下为拟实现服务内部的概念调用，不是替代正式 CLI 的临时脚本：

```python
request_input = {"prompt_token_ids": P}
params = SamplingParams(
    n=1,
    temperature=frozen_root.temperature,
    top_p=frozen_root.top_p,
    top_k=frozen_root.top_k,
    seed=branch.historical_seed,
    # 其余已校验的历史分布参数：penalties、min_p 等
    trace_decode_token_ids=C,
    logprobs=0,
    max_tokens=len(C),
    detokenize=False,
    # 停止语义按历史请求恢复，见下文
)
```

在每个 t：

1. native engine 基于 `P || C[:t]` 得到 next-token logits，并应用该版本真实的 masking、penalties、temperature、min_p、top-k/p 等处理。
2. trace 机制将下一 token 设为 `C[t]`。它发生在采样分布确定之后，不能在 softmax 之前把其他 token mask 成 -inf，否则目标概率将错误地变为 1。
3. engine 在 processed logits 上归一化，仅返回 `C[t]` 的 logprob；随后模型状态、请求 token 历史、计数和约束状态必须按 `C[t]` 推进。
4. 对返回的完整 `output.token_ids` 与 C 逐 token 比较，逐位置按 token ID 读取 `output.logprobs[t][C[t]].logprob`。若条目缺失，不用 top-N 中其他 token 代替，不依赖字典遍历顺序。

底层现有 trace 实现可能仍执行一个随后被覆盖的随机抽取；这不改变记录的概率和推进的历史序列。首版复用其已存在路径，随机数本身不参与 logprob 定义。不要承诺该调用零 RNG 开销，也不需要复现原抽签随机数以确定条件概率。

### 4.4 停止、EOS 与末尾坐标

首个生成位置预测器为 `P[-1]`，监督原坐标为 `len(P)`；第 t 个 completion token 对应原坐标 `len(P)+t`。保存长度严格为 `len(C)`，包括 native output 中已有的终止 token；不额外生成或添加 EOS。

每个请求必须在 trace 长度处停止，避免 trace 用尽后退回自由采样。对首版支持的无动态预算/grammar 请求，`max_tokens=len(C)` 只用作输出上界；实现中验证它不改变该请求的 logits processor 语义。需依赖剩余额度的请求在支持扩展之前拒绝。

停止字符串、stop IDs、ignore_eos 等按历史语义恢复，不能一律清空 stop 或打开 ignore_eos。如历史数据的导出/停止处理导致 replay 提前截断、终止 token 不一致，标为 `trace_length_mismatch` 或 `stop_semantics_unavailable`，检查来源后再处理。任何为保留完整 trace 的“传输层停止绕过”都必须证明不改变 logits mask，并单独测试，不作为隐含默认值。

采样输出中只存 C 的 logprobs；prompt logprobs 不请求。模型仍要处理完整 P，不能因为不保存其概率就省掉其 attention/recurrent 计算。

## 5. 支持集与概率质量

top-p/k 让部分 token 的采样概率严格为 0。若历史 token 在重放时被排除，其正确计算结果是 -inf，不能用 raw logprob 替换，也不能夹到某个极小正数。

处理规则：

- trace、ID 或 hash 不符：硬错误，停止发布。
- 有限且≤0的 processed logprob：可进入对应已验收的 replay 视图；允许微小数值正误差的容忍阈值须预先定义，原始值保留，不用截断掩盖实现错误。
- -inf：写明 `zero_support_under_replay`、位置/token、cohort 与处理配置，保留诊断；该 branch 的历史概率资格不通过。
- NaN、+inf、输出缺条目、提前停止：显式失败状态，不能补 0。

抽样边界附近的 bf16/TP/kernel/batch 数值差异可能改变 top-k/p 的保留集合，即使模型权重相同也不能承诺逐位复原。对边界异常可用预先定义的单请求、固定后端基准复核，但不得更改 temperature/top-p/k、放大 p、关闭截断或试到某个结果“能过”为止。

默认完整组策略继续生效：任一成员的必需 logprobs 缺失或不合格，整个 group 不进入要求完整历史概率的训练视图；基础 dataset 的全部成员与失败记录保留，不静默缩小 group。

资格字段分别记录：`token_identity_verified`、`historical_model_verified`、`sampling_policy_verified`、`runtime_identity_status`、`numerical_validation_id` 和 `status`。历史 runtime 未存档或无法等价核验的结果可标为重放估计，默认不自动获得“历史行为概率已验证”资格。新生成对照的验收可验证实现，不能补造旧运行当时没有的概率证据。

## 6. 产物、存储开销与 dataset 绑定

```text
STORAGE_ROOT/RL/replays/REPLAY_ID/
  replay.json                 dataset、scope、分布类型、配置与身份
  root-sampling.json          三个根配置参数、根配置 SHA、比对结果
  effective-policies.json     每个 cohort 的完整生效规则与来源
  models.json                 历史权重/adapter/tokenizer/runtime 身份
  selection.json              group/branch 的稳定处理次序
  branches.jsonl              branch -> shard/offset/count/status
  shards/part-N.f32.npy       一维拼接的生成 token processed logprobs
  diagnostics.jsonl           不可用、零支持、长度和来源问题
  summary.json                完整率、失败原因、长度/概率/成本摘要
  validation.json             数值验收引用及通过范围
  receipt.json                最后提交的本地文件与来源哈希
```

每条 branch 索引绑定 `branch_sample_id/group_id`、native request/输出 SHA、prompt SHA、tokenizer/model/policy hash、`token_count`、array offset/count 和完整 logprobs 内容 hash。token IDs 不必重复存储，依赖不可变 dataset/materialization 的输出 hash。只要 C 长度为 M，结果就必须恰有 M 个值，训练时再按已有 mask 选择。

建议默认 float32，不量化成 bf16/fp16，避免小概率/比率精度进一步损失。仅值数组的成本为 `4 × 生成 token 数` 字节：100 万 token 约 4 MB，1 亿 token 约 400 MB（十进制；另加索引和 receipt）。全词表仅在当前 decode 批次作为临时 logits 使用，提取结束不留存。

基础 dataset 继续不可变。训练配置显式绑定 `{dataset_id, dataset_receipt, replay_id, replay_receipt, probability_semantics: processed_sampling_v1}`，loader 校验每个 member、坐标和 hash；不能自动寻找 latest replay，也不能把此产物冒充 reference logprobs。

与现有 label/mask 三项张量接口分开处理：dataset loader 提供 `behavior_logprobs` 与位置映射，后续 objective 接口显式接收；不要将 logprobs 塞进 labels 或假设现有 `RL.training.train` 已自动支持它。

## 7. 性能、资源与恢复

主要成本是历史模型对全部 prompt 和 completion 的前向，以及每个输出位置的词表投影与采样过滤；保存一个值不会省去归一化所需的词表运算。

- native decode 每个活跃请求每一步只产生 next-token logits；临时量主要随并发×词表变化，不物化整段长度×词表 logits。
- 同一 group 的 P 可利用该模型已有、已验收的 prefix caching/调度能力。不能手动复制 GDN 状态或 KV 指针给其他分支，也不承诺无条件复用全部 prompt 成本；Qwen 混合 recurrent 模型需专项验证缓存和预填充边界。
- 以小并发、受限 `max_num_batched_tokens/max_num_seqs`、显式 KV cache 预算起步；不强迫 32 个分支同时驻留。host trace buffer、输出累积、索引和缓存也计入预算。
- 使用流式/有界批次拉取输出，处理完成的 branch 后立即写 shard，禁止一次收集全 dataset 的 RequestOutput。
- 超过模型 context/KV/显存预算时报告不支持，不截断 P 或 C、不移动到未授权 GPU、不停止其他服务。
- 后续若引入批量 prefill/选位投影优化，必须保持与 native trace 的 processed 分布等价，并另作验收；不能用快的 raw prompt_logprobs 直接替换。

恢复按已封存 shard 的 branch 边界进行：journal、输入 selection hash、模型/policy/runtime 身份和结果分片校验全部一致才复用；中断分支从其完整 P/C 开始重放，首版不持久化跨进程 KV/GDN 状态。相同 replay ID 仅允许明确的 recover 操作，不覆盖失败目录。resume 使用已冻结根采样快照，不读取修改后的根参数。

同盘 staging、锁、原子 rename 与最终 receipt；默认训练 loader 只消费已提交完整视图。部分失败可发布明确的审计状态和有效组清单，但不能把缺失数组标为完整成功。

## 8. 正式配置和入口设计

以下为**待新增**入口，不是已经可运行的命令。代码归 `RL/dataset/`，建议 `replay/{config,service,artifacts,validation}.py` 加一个 native-vLLM adapter；按复杂度合并文件。共享 runtime 逻辑放原有公共层，避免复制 engine 或采样器。

```yaml
schema_version: 1
replay_id: swebench-step-logprobs-v1
dataset:
  id: DATASET_ID
  root: /data3/wangtianyi/DyConGlobal
sampling:
  source: root_config
  semantics: processed_sampling_v1
model:
  source: historical_dataset_bindings
runtime:
  backend: native_vllm_trace_v1
  python: .venvs/latent-vllm/bin/python
  gpus: [0]                 # 示例，执行前检查，不能隐式选卡
  tensor_parallel_size: 1
  max_num_seqs: 2
  max_num_batched_tokens: 4096
  kv_cache_memory_bytes: 8589934592
storage:
  output_root: /data3/wangtianyi/DyConGlobal
  max_shard_bytes: 268435456
```

参数示例不代表9B/27B的容量保证。standalone replay YAML 管理数据、输出与执行资源；三个采样值只在根配置存在一份，禁止 replay YAML/CLI 暗中覆盖。

```bash
# runner 轻量预检，不加载模型、不开 GPU、不提交成功 receipt
.venvs/runner/bin/dycon rl-dataset replay-logprobs \
  --replay-config REPLAY.yaml --dry-run

# 正式入口管理受控隔离子进程和资源生命周期
.venvs/runner/bin/dycon rl-dataset replay-logprobs \
  --replay-config REPLAY.yaml

.venvs/runner/bin/dycon rl-dataset check-logprobs REPLAY_ID \
  --storage-root STORAGE_ROOT --with-values

.venvs/runner/bin/dycon artifact verify REPLAY_ID \
  --scope rl-replay --storage-root STORAGE_ROOT
```

scope、resolver、recovery、check-data 等作为同一服务实现，不写临时采样/拼接脚本。`--dry-run` 输出覆盖和不支持约束，不应把“尚未实际前向”的 logprobs 状态报告为已通过。

## 9. 分阶段验收

### R0：只读资格预检

检查用户选定 dataset 的模型、native token、三参数根配置一致性、完整约束与当前 trace 支持范围。输出按 cohort 的可重放/被阻塞 group 数、具体阻塞字段与预算。不默认启动 GPU 或修改旧 run/capture。

### R1：同 logits 下的概率定义测试

使用已安装 native sampler 路径与 CPU float64 小词表 oracle 对照，覆盖温度、top-k、top-p 联合作用、边界/tie、penalties、零支持和 token gather。固定 logits 不包含模型前向差异；建议 float32 绝对误差≤1e-6，阈值先冻结。

验明只返回生成 token 值，`logprobs=0` 有效、None 无结果；默认 raw 与 processed 结果在非平凡采样参数下确实不同；prompt_logprobs 不会被当作 processed 输出。验证强制 trace 不会把目标概率错误归一化成1。

### R2：真实模型“原生记录 -> 重放”闭环

在指定空闲 GPU 上通过正式验收入口生成短对照序列，生成时同时记录 native processed token logprobs、原始 P/C、完整采样和runtime身份。使用同一模型和规则 trace replay C，比较每个位置：

- 输出 token 和数量必须完全一致，含 reasoning/action/终止 token；
- 原生和 replay processed logprobs 逐 token 对照，记录 max/mean/p99 绝对差与支持集变化；
- 对同 engine/相同调度基准预先设定容差；例如 bf16 起始验收门槛可设 max_abs≤0.005，支持集不一致为0。该值是拟定门槛，需在执行前冻结；失败后分析原因，不事后放宽；
- 覆盖单/多分支并发、重复 action、不同前缀长度、prefix cache 开关/可用模式、Qwen GDN/full attention；
- 校验第一 token、右侧长度边界、trace 用尽不续采样、停止/EOS，以及不支持字段在模型加载前被拒绝。

原生记录与重放必须来自同一个预先声明的 backend/kernel/policy cohort。跨 backend、旧 runtime 未保存或调度变化的结果单列；新闭环通过不代表所有旧样本都获得逐位一致证明。

### R3：少量真实历史轨迹与扩容

先选1–2条 native数据与模型身份完整的历史轨迹。必须对全部合格候选提取，检查完整 group 覆盖、模型取历史9B而非当前27B、根三参数快照、支持集异常、branch hash绑定和只读来源。

随后逐级扩大长度和并发，记录prefill/decode tokens/s、峰值GPU/host内存、trace/KV预算、每百万token耗时与输出体积。超长失败不以截断替代；验收只覆盖实际完成的最大长度和配置。

### R4：破坏/恢复与 loader 校验

篡改模型/输出/policy/数组任一hash、漏一个branch、错一位logprob或token、缺失EOS条目，loader必须拒绝。验证相同输入和配置恢复的结果索引确定；source/cache/output根可配置，未完成receipt不会被训练消费。

## 10. 受限历史请求的最小扩展路径

首版 trace 拒绝 grammar/thinking budget/bad words 等并不表示相关模型数学上不能重放，而是当前runtime没有声明该组合受支持。

只有选定来源确实包含这些约束、且用户需要纳入时，再做满足该范围的最小长期实现：

1. 在已固定的 native trace/sampler 状态推进点扩展，使历史 token 正确推进同一 grammar automaton、thinking counter、bad-words/penalty/request state；所有约束仍在强制token之前参与概率计算。
2. 保留现有能力探测与拒绝规则，对通过测试的具体组合逐项开放；不能直接删除 validator 的禁止条件。
3. 修改依项目vendor/runtime pin流程维护；如果隔离副本要求与serving pin一致，需要按既有流程准备一致的新runtime身份，不能只修改已安装site-packages绕过校验。生产serving升级和停止服务不属于本设计阶段自动执行范围。
4. 增加每种新组合的native生成/重放processed-logprobs闭环验收，再发布新的backend/policy版本和replay产物。

当前根配置有require_reasoning_close和thinking budget，不意味着旧9B来源一定有这些约束；资格预审按各来源实际请求判断。遇到不支持的组必须保留在报告中，不能为了提取成功改变历史分布。

## 11. 后续训练的使用边界

本方案确定的是历史行为采样分布 q 的重放估计，不确定reward、KL、reference、loss和训练归一化。

后续若以此作为重要性比率分母，分子也要明确是原始模型分布 p 还是处理后的采样分布 q；混用定义会改变目标，不能因两者都叫logprob而混接。top-p/k的硬支持集与策略更新后的支持变化，也要在目标设计中单独处理。本阶段不通过修改概率或平滑截断来预先决定算法。

重放不会把历史离线数据变成on-policy样本，也不产生未选分支的后续环境reward。独立reference概率或reward sidecar继续通过dataset/group/member身份绑定。

## 12. 本次核查依据

工程：`RL/memory.py`、`RL/modeling.py`、`rollout/model.py`、`hidden/backends/vllm/{engine,capture}.py`、`runtime/serving/vllm_plugin.py`、根`config.yaml`和上述历史run的`resolved-config.json`。

已安装vLLM：`config/{model,vllm}.py`、`sampling_params.py`、`v1/worker/gpu/sample/{sampler,trace_replay,logprob}.py`、`v1/sample/ops/topk_topp_sampler.py`。这些是已存在的底层能力，项目层`rl-dataset replay-logprobs`入口仍待实现。

只读核查经一次性非交互SSH执行，设置超时和输出限制，确认命令退出、连接关闭；未修改runtime、根配置、来源run或已有训练代码。
