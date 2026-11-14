# DyConGlobal 最小运行指南

命令中的 `RUN_ID`、`INSTANCE_ID`、`ROLLOUT_ID` 等均为占位符。先阅读[入口与工程契约](ENTRYPOINTS.md)，按 benchmark 查看[运行指南](BENCHMARKS-CN.md)。CLI 的当前参数以 `dycon COMMAND --help` 为准。

## 1. 检查配置和任务

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon images-check
```

`images-check` 校验已登记镜像，不等于任务容器和 harness 的完整启动验收。按需运行对应 benchmark 的 preflight 或 `harness ... --gold` 环境检查。

`execution.supervisor_max_open_files` 设置 supervisor 及其管理子进程的软文件描述符上限。新 run 会冻结该值，并在 `execution/attempts/<attempt>/supervisor-limits.json` 记录实际软/硬限制；它不能超过主机硬限制，也不能替代控制并发和检查 FD 生命周期。

## 2. 创建并运行 rollout

单次采样：

```bash
.venvs/runner/bin/dycon run create RUN_ID --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon run status RUN_ID
```

每个 task 计划多个独立采样时，在创建时指定 `--repeat-count N`。基础设施重试不增加独立采样数。也可用 `--tasks-file FILE` 或 `--partition train|val|test|excluded` 指定任务。run 创建后，任务选择与有效配置被冻结。

创建命令只建立计划；`run start` 才启动 workload。执行、resume 和绑定 run 的分析使用该 run 的冻结快照，不要用新的 `--set` 改写其行为。

## 3. 评分已生成的 patch

```bash
.venvs/runner/bin/dycon harness prepare RUN_ID --rollout-id ROLLOUT_ID
.venvs/runner/bin/dycon harness run RUN_ID EVAL_ID
.venvs/runner/bin/dycon harness status RUN_ID
.venvs/runner/bin/dycon report run RUN_ID
```

评分 selection 绑定 canonical rollout 和 patch SHA。`completed/resolved`、`completed/unresolved` 与 `infra_error` 是不同结果；历史 patch 重评应通过新的 harness run 明确指定来源 rollout。

## 4. 检查与恢复

```bash
.venvs/runner/bin/dycon run show RUN_ID
.venvs/runner/bin/dycon run status RUN_ID
.venvs/runner/bin/dycon artifact show RUN_ID
.venvs/runner/bin/dycon artifact verify RUN_ID
```

`run resume` 复用原 run 的计划和已结算样本，只继续未完成工作。它不会重跑已接受的样本，也不会把终态失败自动变成新样本。

### Recover or retry

优先用原阶段的恢复命令，例如 `capture recover RUN_ID` 恢复已存在的 capture 发布。只有确认 worker 已结束、清理状态完整，且明确需要重新执行选定失败样本时，才登记 `run retry-failed` 请求；之后通过 `run resume` 处理。每次新的主动重跑使用新的 request ID。使用可重复的 `--sample-key` 精确绑定 task/repetition，使用可重复的 `--exit-status` 指定要纳入的失败类型；不传 `LimitsExceeded` 即不会登记该类型。没有物理 attempt 的 `WorkerFailure` 也可以精确重跑：统一接口会从 `progress.json` 归档原始游标，`run resume` 再创建新的物理 attempt，并保留原证据。CaptureError、pending recovery 或未确认的 spool 不应直接换 seed 重跑；先执行 `capture recover RUN_ID`，确认已有证据已恢复或明确无法恢复后再登记重跑。

```bash
.venvs/runner/bin/dycon run retry-failed RUN_ID \
  --request-id REQUEST_ID \
  --exit-status OpenHandsError --exit-status WorkerFailure \
  --sample-key SAMPLE_KEY_1 --sample-key SAMPLE_KEY_2 \
  --reason "reason"
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

详细边界见[入口约定](ENTRYPOINTS.md#runs-samples-and-attempts)与[capture 指南](CAPTURE-CN.md)。

对于仅记录在 `progress.json`、没有 `result.json` 的 `WorkerFailure`，显式 `retry-failed` 请求会归档失败游标，并绑定已有未完成 attempt 的 context/start 哈希。后续 `resume` 从请求归档读取授权，不依赖重新生成的 progress 游标；跳过已授权的旧恢复路径，在 `rollout/operator-retry-consumptions/REQUEST_ID/SAMPLE_KEY/` 写消费回执，再以同一 seed round 创建新的物理 attempt。旧轨迹、capture、预算及身份文件原样保留。历史请求缺少目录快照时，仅允许绑定请求时间及之前执行范围内的旧 attempt，并记录解析结果；请求之后产生的新 attempt 仍走正常恢复，同一请求不会反复创建新 attempt。

### 配置修订

已有 run 的参数通过 `dycon run revise-config RUN_ID --revision-id ID --set sandbox.memory_mb=16384 --reason 'Increase memory'` 修订；先用 `--dry-run` 预览，应用前正式停止并确认清理完成。身份字段也可修改，需要显式 `--replan` 归档原状态。旧的专用字段修订命令已移除，完整参数、YAML patch 和恢复流程见[统一配置修订](CONFIG-REVISION-CN.md)。

## 5. 在线采集和 token 分析

需要 rollout 同步保存 raw T=1 logprobs 时，在创建新 run 前于根配置启用 `capture.logprobs.enabled: true`，同时保留 `capture.enabled: true`、`capture.export.enabled: true`、managed TP=1 和现有 layers/anchors 配置；当前不支持 steering。生成 temperature、top-p/top-k 无需改成评分参数。通过上文同一 `run create/start/wait` 生命周期运行，resume 使用冻结开关。完整配置、预算、sidecar receipt 和验证范围见[在线 raw logprob 指南](ONLINE-RAW-LOGPROBS-CN.md)。

`capture status` 或 hidden capture complete 不能单独证明 raw 概率覆盖。`rl-dataset build` 绑定可用 sidecar 后，检查 branch 的 `raw_probability_available`、group 的 `raw_probability_complete`，并运行 `rl-dataset check DATASET_ID --dataset-root DATASET_ROOT --level full`。概率失败保留原生 token；`capture recover` 只恢复已有 bytes 的发布，不补算缺值。

```bash
.venvs/runner/bin/dycon capture status RUN_ID
.venvs/runner/bin/dycon capture spool-usage
.venvs/runner/bin/dycon capture spool-usage RUN_ID --ledger-only
.venvs/analysis/bin/dycon online-index RUN_ID INDEX_ID
.venvs/analysis/bin/dycon enrich RUN_ID
.venvs/analysis/bin/dycon trajectory-token-audit RUN_ID
```

在线采集在生成阶段绑定 run、sample、attempt、rollout 和 session。大张量可存于配置的 NAS 根目录，run 中保存 publication、locator 和 receipt。先确认 capture 发布完整，再建立下游索引；不要用训练标签推断缺失的 native token 证据。

## 6. Action 与 difficulty

```bash
.venvs/analysis/bin/dycon action-classify ACTION_ID --source-run RUN_ID \
  --rollout-id ROLLOUT_ID
.venvs/analysis/bin/dycon action-check ACTION_ID
.venvs/analysis/bin/dycon artifact verify ACTION_ID --scope action-analysis

.venvs/analysis/bin/dycon difficulty-dataset resolved DATASET_ID \
  --source-run RUN_ID --index-id INDEX_ID
.venvs/analysis/bin/dycon difficulty-split DATASET_ID SPLIT_ID
.venvs/analysis/bin/dycon difficulty-train --training-config CONFIG.yaml
```

Action 与 difficulty 产物位于项目级目录，使用显式来源 run 和固定选择。difficulty 数据集、划分和训练配置不可互相替代；详细命令、数据语义及预测器发布见各自模块 README。

## 7. Run 比较和多卡

```bash
.venvs/runner/bin/dycon run compare RUN_A RUN_B --baseline RUN_A --format markdown
```

比较只读被冻结的 run 计划、最终选择及评测证据，不启动模型或 harness。同样的 repetition 数默认按 task×repetition 配对；数量不同需显式指定每个 run 的 repetition。指标解释见[trajectory comparison](TRAJECTORY-COMPARISON-CN.md)。

多 GPU 独立 TP=1 调度使用配置 `serving.trajectory_sharding: per_gpu_tp1`，由正式 run 入口执行，见[多 GPU 指南](MULTI-GPU-TP1-CN.md)。

## RL/dataset：离线 step 数据与 logprobs 重放

入口为 `.venvs/runner/bin/dycon rl-dataset`。先阅读 [RL/dataset 模块文档](../RL/dataset/README.md)，以 [dataset 示例](../RL/dataset/configs/swebench-consensus-example.yaml) 和 [replay 示例](../RL/dataset/configs/replay-logprobs-example.yaml) 编写独立 YAML；显式设置来源 run/rollout、全新产物 ID、物理根目录与重放 GPU。示例配置中的历史 ID 不表示已有成功数据集，以下 `DATASET_ID`、`REPLAY_ID` 和各路径均需与自己的配置一致。

```bash
# 只检查来源选择，不构建数据集
.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml --dry-run

# 明确需要实际构建时执行
.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml
.venvs/runner/bin/dycon rl-dataset check DATASET_ID --dataset-root DATASET_ROOT --level full
.venvs/runner/bin/dycon rl-dataset materialize DATASET_ID --dataset-root DATASET_ROOT --cache-root CACHE_ROOT

# 先预检；正式重放由 CLI 启动隔离 latent-vLLM worker
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml --dry-run
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml
.venvs/runner/bin/dycon rl-dataset check-logprobs REPLAY_ID --storage-root OUTPUT_ROOT --with-values --dataset-id DATASET_ID --dataset-root DATASET_ROOT
.venvs/runner/bin/dycon artifact verify REPLAY_ID --scope rl-replay --storage-root OUTPUT_ROOT
```

每组保留同一 decision step 的全部候选，输入使用 native prompt/output token，监督仅覆盖本分支生成的 reasoning+action，mask 使用原始 target 坐标。默认完整组至少有两个全部可用分支；未选中或未 retained 不会因此被丢弃。构建阶段把历史 score 保留为元数据，训练阶段由版本化 GRPO objective 冻结 Advantage。

离线训练沿用 `.venvs/runner/bin/dycon lora-train --training-config RL/experiments/grpo-offline-k3-smoke.yaml` 做 dry-run；实际训练使用 `.venvs/rl/bin/dycon lora-train --training-config TASK.yaml`。填写新 attempt ID、dataset ID 和空闲物理 GPU。训练扫描整个 dataset 的完整 eligible groups，只遍历一次 epoch，无 split manifest；raw T=1 ratio 与 token K3 共用冻结的 initial-policy logprob。配置、reference cache、checkpoint/resume、产物验证和验收边界见 [GRPO 指南](../RL/GRPO.md) 与 [测试报告](../RL/GRPO_VALIDATION.md)。

训练概率来源由 task YAML 的 `grpo.reference_cache` 选择：缺省 `model_forward`；
统一产物或旧离线导入设 `source: probability_artifact`，填写
`artifact.scope: rl-probability`（旧 raw replay 仍用 `rl-replay`）、
绝对 `root` 和 `id`；在线导入设 `source: dataset_probability` 和明确
`stream: raw_t1_output_tokens`。两种外部模式都必须显式填写
`dtype_compatibility: same_dtype` 或 `fp16_source_bf16_training`。先设置
`run.mode: dry-run` 验证封存来源，再以 `run.mode: smoke` 执行；只对冻结的 eligible
分支导入同一 shared anchor，current 实时计算。缺值/错绑/processed 不回退重算。
Prepared/legacy-cache 路径的 resume 使用父 checkpoint 内的 cache 和 provenance，
可不再访问原概率源。Native streaming 则继承原冻结 plan、advantage、reference
anchor 和已提交 cursor，继续读取剩余 native token ledger 与外部 raw 概率；
dataset 目录中的 `branches.jsonl` 仅保存 ledger 引用，不能替代 token 文件。
Streaming resume 要求 `runtime.data_preparation_async: true`、`prepared_data: null`，
保持原 world size、dataset/source 路径及完整 `training.max_updates`，通过
`validation.stop_after_update` 暂停。每次预检和实际执行分别使用新 attempt ID；
父 checkpoint 指向 `checkpoints/step-NNNNNN/`，不能只指向 adapter。父 contract
决定 prepared/streaming 路径，不能用 async 开关互转；streaming checkpoint
不支持 `warm_start` 或 `append_resume`。

Source/dtype、模型、依赖、schedule 和数值合同变化仍会拒绝；project source
fingerprint 只作审计记录。非 streaming 的 warm-start 必须匹配实际评分 adapter，
或显式选择 model-forward。完整配置、旧格式只读兼容和查询/验证命令见
[流式恢复](../RL/STREAMING_RESUME.md) 与 [训练指南](../RL/TRAINING.md)。
2026-11-14 单卡/双卡短 native 数据续训验收已通过；v10 本身仅完成只读预检、
anchor 和依赖核验，尚未执行 update 61，原长上下文 OOM 未解决。详见
[流式验收记录](../RL/STREAMING_RESUME_VALIDATION.md)。历史外部评分数值验收
边界继续见 [外部绑定报告](../RL/GRPO_BINDING_VALIDATION.md)。

`audit.enabled` 默认关闭，控制来源身份与哈希核验；`check --level full` 不会强行打开已关闭的来源审计。产物 receipt 始终核验。Dataset 保存在 `DATASET_ROOT/RL/datasets/DATASET_ID`，可选 mmap cache 保存在 `CACHE_ROOT/RL/cache/CACHE_ID`，重放产物保存在 `OUTPUT_ROOT/RL/replays/REPLAY_ID`。

新 schema 2 重放使用冻结 FP16 checkpoint 对原生 completion 做 raw T=1、transform 前全词表归一化评分，不读取根配置的采样温度/top-p/top-k。旧 processed 产物只供诊断，不能改名后导入训练。通用 loader 的 `allow_replay_estimates=True` 与训练侧严格导入合同独立；训练要求所有 eligible target 为有限 float32。raw 评分不是精确历史 processed 行为重要性校正。Qwen GDN、真实长上下文和多卡容量的当前生产限制见 [dataset 文档](../RL/dataset/README.md)。

中断后，分别使用 `build --dataset-config DATASET.yaml --recover` 或 `replay-logprobs --replay-config REPLAY.yaml --recover`。构建恢复采用冻结 selection，从完整 rollout 边界继续；raw 重放恢复采用冻结 dataset/cache、raw scoring、runtime/实现合同和已封存 branch。输入、实现或 runtime 不兼容时拒绝续跑；旧版无 checkpoint 的 staging 不能直接恢复。完整限制与合成测试命令见模块文档。

### 在线值与缺失补评统一封存

旧分支缺概率、新分支已有在线值时，先用
[概率配置示例](../RL/dataset/configs/probability-example.yaml)显式填写已封存
dataset、existing 来源、待生成 backfill replay ID/root、输出和 plan ID/root。
不同 runtime 的 cohort hashes 必须预先显式允许，来源 compute dtype 必须一致。

```bash
# 发布 coverage plan，不加载模型；返回 plan_sha256 与独立 receipt hash。
.venvs/runner/bin/dycon rl-dataset probability-plan --probability-config PROBABILITY.yaml
# 将 plan ID/root/plan_sha256 分别写入：
# REPLAY.yaml 的 selection.coverage_plan 和 PROBABILITY.yaml 的 coverage_plan。
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml --dry-run
# 确需实际 GPU 补评且已安排资源时才运行：
# .venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml
.venvs/runner/bin/dycon rl-dataset probability-build --probability-config PROBABILITY.yaml
.venvs/runner/bin/dycon rl-dataset probability-check PROBABILITY_ID --storage-root ROOT --with-values
.venvs/runner/bin/dycon artifact verify PROBABILITY_ID --scope rl-probability --storage-root ROOT
```

模型只接收 frozen missing branches；32 分支已有 31 条时只补 1 条，无缺失时不启动
GPU worker。合并不调用模型，保留已有 FP32 bytes 并发布自包含 schema 3 到
`ROOT/RL/probabilities/PROBABILITY_ID/`。plan 位于 `ROOT/RL/probability-plans/ID/`。
online-only/offline-only 导出使用同一 build；无 backfill 时可直接 build。
原 dataset/run/replay、在线 availability flags 和 receipts 不回写。

训练改用 `source: probability_artifact`、`artifact.scope: rl-probability`，仅指定
最终 ID/root，并显式保留 dtype 兼容规则。训练需要的 eligible branches 必须全部
有有限值；`coverage_status: complete` 本身不代表 finite 或训练就绪。来源 NAS
原上游来源离线时，普通 verify 和读取 canonical 概率产物仍使用其本地封存副本。
Prepared/legacy-cache resume 使用 checkpoint 内 cache；native streaming resume
仍需所选 canonical 产物及未消费 native ledgers 可达。`probability-check --sources`
才显式追溯来源。流程、失败语义和恢复见
[统一概率产物](../RL/dataset/PROBABILITY_ARTIFACTS.md)，CPU 证据及 GPU 待验收范围见
[对齐验收报告](../RL/PROBABILITY_ALIGNMENT_VALIDATION.md)。

### RL 训练进度、loss 曲线与 advantage

实际训练启动后，使用 runner 环境查看进度或查询冻结数据：

```bash
.venvs/runner/bin/dycon lora-status --active
.venvs/runner/bin/dycon lora-status ATTEMPT_ID --storage-root ROOT
.venvs/runner/bin/dycon lora-tensorboard ATTEMPT_ID --storage-root ROOT --action start --port 16006
.venvs/runner/bin/dycon lora-advantages ATTEMPT_ID --storage-root ROOT --group-ordinal 0
```

`ATTEMPT_ID` 和 `ROOT` 分别替换为训练 attempt ID 与实际存储根目录。
TensorBoard 仅监听远程 `127.0.0.1`；本地运行
`ssh -N -L 16006:127.0.0.1:16006 wangtianyi@10.249.187.9`，再打开
`http://127.0.0.1:16006`。用同一 `lora-tensorboard` 命令的 `--action status`
检查服务，`--action stop` 结束可视化；训练继续运行。

Status 分开显示已提交 update、inflight 工作和可恢复 checkpoint；曲线只记录完整
optimizer update。Advantage 来自冻结训练数据并标记来源；旧 JSON 需显式
`--build-index`，通过 `--index-status` 查看构建进度。旧训练无需重启，缺失的阶段
信息显示 `unknown`，索引和 events 不写入训练 artifact。更多用法见
[训练指南](../RL/TRAINING.md#训练进度实时-loss-与冻结-advantage-查询)和
[可观测性专题](../RL/OBSERVABILITY.md)。

## 8. 任务专用入口

SkillsBench、WebArena Verified、SWE-bench、Harbor 的 task 导入、容器准备和 preflight 各有约束。开始任务前阅读[benchmark 操作指南](BENCHMARKS-CN.md)，不要从实验报告复制旧 run 的命令或资源参数。


| Run | Accepted task change | Accepted sample change | Steps change | Prompt token change | Completion token change | Total token change | Reward sum change | Reward mean change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skillsbench-agent75-r3-qwen36-27b-positive-gpu23-tp1-20261031 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| skillsbench-test57-r3-consensus32-train30-gpu0123-tp1-w4-s64-20261102 | -41.54% | -51.66% | -20.18% | -20.54% | -36.69% | -21.05% | -5.09% | -5.09% |


## 追加 repetition 与重跑 accepted partial capture

以下命令只登记计划；随后用 run resume 正式执行。操作前要求原执行退出且
cleanup receipt 完整。Capture 异常应先检查并尝试恢复已有证据，保留旧 spool。

所有当前 TimeoutError（failed/blocked）与 accepted 且 capture partial 的并集：

    .venvs/runner/bin/dycon run retry-failed RUN_ID --request-id RETRY_ID --exit-status TimeoutError --accepted-capture-partial --reason 'Explicitly rerun timeout and partial capture samples'

仅重跑 accepted 且 capture partial：省略 --exit-status。

    .venvs/runner/bin/dycon run retry-failed RUN_ID --request-id PARTIAL_RETRY_ID --accepted-capture-partial --reason 'Explicitly rerun partial capture samples'

为所有冻结 task 各追加 1 次独立 repetition，先预览：

    .venvs/runner/bin/dycon run add-repetitions RUN_ID --request-id APPEND_ID --count 1 --reason 'Add one independent repetition' --dry-run

将预览中的 old_plan_sha256 填入 HASH 后应用：

    .venvs/runner/bin/dycon run add-repetitions RUN_ID --request-id APPEND_ID --count 1 --reason 'Add one independent repetition' --expect-plan-hash HASH
    .venvs/runner/bin/dycon run resume RUN_ID --detach

--count 是追加次数，针对所有冻结 task；例如原 5 次追加 1 次，新增 rep-005，
总数变成 6 次，不重建旧结果。新样本和主动重跑都使用统一调度入口。
重跑筛选默认覆盖整个 run；仅处理指定 sample 时重复传入 --sample-key，
可配合 --expect-attempt SAMPLE_KEY=ATTEMPT_ID 固定当前物理 attempt。

追加事务中断时，使用原 request ID 和原 reason 执行：

    .venvs/runner/bin/dycon run add-repetitions RUN_ID --request-id APPEND_ID --recover --reason REASON

恢复时不传 --count；存在未完成事务时执行入口会拒绝启动。seed 分配及审计规则见
[入口契约](ENTRYPOINTS.md#append-repetitions-and-explicitly-rerun-accepted-partial-capture)。


### 启动前核验实际队列

将重跑与追加 repetition 连用时，先登记 retry-failed，再执行 add-repetitions，
最后统一 resume。--count 10 表示每个冻结 task 追加 10 次；原 5 次变成 15 次，
新增 rep-005 至 rep-014。75 个 task 将新增 750 个独立 sample。

重跑队列应以 rollout/operator-retries/REQUEST_ID/request.json 中实际登记的 items、
它们的归档原始终态、统一 retry index 中的 superseded 身份及当前状态投影为依据。
不要只按登记前的筛选条件计数。启动前按 previous_exit_status 汇总实际 Pending 的
重跑 sample，并单独报告新增 repetition sample 的数量；新样本没有历史终态，不应
并入 TimeoutError 等重跑类别。仍处于 accepted 且 capture complete 的旧样本保留，
其他未明确授权的失败也不自动重跑。


## Dataset Ridge 两层划分评估

已有具备训练资格的 difficulty dataset 时，使用统一入口完成 task 留出、内层 event 选优、重拟合及全部 test 轨迹评估：

```bash
.venvs/analysis/bin/dycon difficulty-ridge-holdout NEW_EVALUATION_ID \
  --workflow-config difficulty/training/experiments/ridge-holdout.yaml
.venvs/analysis/bin/dycon artifact verify NEW_EVALUATION_ID --scope difficulty-evaluation
```

先编辑 workflow YAML 选择 dataset；`run.mode: dry-run` 只核验来源与划分，`check-data` 再检查 hidden/cache，`fit` 实际拟合。
各模式都使用新 ID，不覆盖已有输出。`visualization.enabled: false` 跳过绘图，保留预测、指标与 coverage audit。
这项工作只消费已发布数据，不执行 rollout 或 hidden 重放。

完整参数、显式 task manifest、恢复命令、产物位置及本次真实验收结果见
[Dataset Ridge 任务留出评估与全量轨迹可视化](DIFFICULTY-RIDGE-HOLDOUT-CN.md)。

## 8 卡 RL：distinct-parent 8×16（实验性）

在完整任务 YAML 中使用以下字段；其余模型、dataset、reference 和输出配置仍需明确填写：

```yaml
runtime:
  gpu: null
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
grpo:
  group_batch_size: 8
  parallel_branch_chunk_size: 16
  parallel_branch_schedule: distinct_parent_two_pass_v1
  branch_partition_seed: 42
  prefix_reuse:
    mode: boundary_grad
```

至少 8 个合格父组，每组完整 32 分支评分；每步从 8 个不同父组各执行 16 条，
两轮覆盖互补两半，尾窗分开。不得将同一父组两半安排在同一步。
先设置 `run.mode: dry-run`，通过 runner 的 `dycon lora-train --training-config TASK.yaml`
预检；8 张物理卡全部空闲后，设置 `run.mode: smoke`，再通过 rl 环境同一 CLI 训练。
两次调用各使用新 attempt ID，不额外套 `torchrun`。

G 个父组需要 `2*ceil(G/8)` 次更新；6773 父组为 1694 次更新、13546 个 half、
216736 条分支。暂停使用 `validation.stop_after_update`，恢复指向完整 step checkpoint，
保持冻结计划、world size 和完整 max_updates；旧 v10/1/2/4 卡 checkpoint 不能直接切换。
状态中的 consumed halves、seen parents、completed parents 含义不同。

2026-11-14 CPU 回归 426 passed、12 skipped，包含真实 8 进程 Gloo；8 卡真实模型、
长 prefix 容量、吞吐和质量仍待验收。完整说明见 [8×16 协议](../RL/DATA_PARALLEL_8X16.md)
及[验收记录](../RL/DATA_PARALLEL_8X16_VALIDATION.md)。
