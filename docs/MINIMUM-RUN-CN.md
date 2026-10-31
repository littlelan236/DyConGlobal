# DyConGlobal 最小运行清单

需要同一 run 在多张卡上各用 TP=1 运行不同轨迹时，配置 `serving.trajectory_sharding: per_gpu_tp1`、每卡端口与 `serving.max_cpu_workers_per_gpu`，见 [多 GPU TP1](MULTI-GPU-TP1-CN.md)。每卡空闲槽位立即领取下一项任务。

## 多个 run 横向比较

```bash
# repetition 数相同，直接比较全部 repetitions
.venvs/runner/bin/dycon run compare RUN_A RUN_B RUN_C --baseline RUN_A --format markdown
# repetition 数不同，分别指定某一个 repetition（索引从 0 开始）
.venvs/runner/bin/dycon run compare RUN_A RUN_B --baseline RUN_A \
  --repetition RUN_A=rep-000 --repetition RUN_B=rep-002 --format markdown
# 默认 JSON，可保存到一个尚不存在的文件（父目录须已存在）
.venvs/runner/bin/dycon run compare RUN_A RUN_B --baseline RUN_A --output temp/comparison.json
```

数量相同时默认按 **task×repetition 序号**配对，只保留各 run 均最终 submitted 的配对；resolved rate 分母为共同配对数，另列去重 task 数。显式选择 repetition 后按 task 配对，退出次数/概率也仅统计所选 repetition 的全部物理 attempt。每个多 repetition run 都要明确选择；只有一次的 run 可省略并自动选 `rep-000`。steps/token 只计共同配对的最终轨迹和实际选中分支。未知评测、缺失 token、零基准和活动/中断 attempt 单列。

多次 harness 评测会保留全部历史 evidence，但比较时只采用每个 rollout 当前 selected evaluation 的 `resolved`；不会累加、平均或按任意一次成功统计。没有当前评测的共同配对计为 unknown，不计为 unresolved；resolved rate 分母仍是全部共同配对数。命令只读证据，不启动 workload；完整定义与 JSON schema-2 字段见 [横向比较口径](ENTRYPOINTS.md#run-横向比较2026-10-29)。

Run 专属共识与 Action 熵：用 `dycon run create ... --parallel-config POLICY.yaml` 冻结参数，不修改根配置。代表策略支持 score/当前step最短；`parallel_enthophy` 支持三种原始 Shannon 熵映射。用法与无GPU测试见 [独立策略说明](ACTION-ENTROPY-RUN-POLICY-CN.md)。

核对日期：2026-10-21。工程：`/home/wangtianyi/DyConGlobal`。

入口语法、输入格式和分析口径见 [ENTRYPOINTS.md](ENTRYPOINTS.md)，容器故障见 [修复指南](container-environment-repair-cn.md)。

本清单按功能组织，命令里的大写 ID 需替换为实际值。runner 负责 rollout/Docker/harness；analysis 负责 tokenizer、action、训练与绘图；hidden 重放由统一入口派发到 latent-vllm。旧轨迹的 resolved 只表示旧 patch 的评分结果；新 rollout 必须重新评分。

<a id="skillsbench-think-close"></a>
## SkillsBench 每 step `think_close`：已验证配置与验收

在创建**新 run** 前，将 `generation.require_reasoning_close=true` 与 `generation.thinking_token_budget=512` 合并到根 `config.yaml`，同时保持 thinking、managed capture、`capture.anchors=[think_close]` 和 export 开启。此次 smoke 的 `generation.max_tokens=8192`；预算需为 action 与结束 token 留空间。`enable_thinking=true`、有 reasoning 字段或 capture complete 本身均不保证每步有 close hidden；仅开 grammar 的对照尝试出现过 `length`。

先单任务/单 repetition，通过正常 `run create/start/wait` 执行；用 `skillsbench audit RUN_ID --rollout-id ROLLOUT_ID` 检查已结束的轨迹，再逐一核对 canonical step、accepted native request、显式 close token 和 NAS persisted hidden 的集合，不能只比较已有 hidden 样本数。旧 run 快照不能修改，resume 不会应用新根配置。

2026-10-30 已验证 `skillsbench-thinkclose-mig6-smoke-v5-20261030`：42/42 个 step 有显式 close 及已校验 FULL hidden；agent 完成、capture complete，verifier 单独为 VerifierError。单任务结果不等于所有任务/seed 保证；thinking budget 会改变生成行为。

配置片段、完整入口、MIG preflight 兼容限制、native/hidden 验收口径及证据路径见 [SkillsBench 显式 close 生成与采集](SKILLSBENCH-OPENHANDS-CN.md#think-close-capture)。

## 1. 配置、Catalog、模型与环境

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/easysteer/bin/python runtime/scripts/verify_easysteer_runtime.py
# 检查任务镜像的 NAS 描述和内容；不等于执行 harness。
.venvs/runner/bin/dycon images-check
```

SkillsBench/OpenHands 首次使用流程与支持范围见 [SkillsBench + OpenHands 最小操作](SKILLSBENCH-OPENHANDS-CN.md)。

共用 generation、agent step/token limit、各种 timeout 配在根 `config.yaml`。创建 run 时冻结有效配置和 tasks。`runtime-check` 另要求 wheelhouse，不能用它代替 task 镜像/启动 gate。初次注册外部模型用 `model-register CHECKPOINT`。

## 2. 任务、轨迹与历史结果输入

```bash
# 输入结构见 ENTRYPOINTS.md 第 4 节；partition 是 task 属性。
.venvs/runner/bin/dycon tasks-import TASKS.json
# 拉取固定版本的 benchmark/NAS 镜像（已有数据时不重复执行）。
.venvs/runner/bin/dycon benchmark-pull COHORT_ID --count 600
```

历史轨迹、评测、hidden 元数据先显式筛选，再由隔离转换工具生成新版产物和 provenance；不得把旧 progress/PID 转成可 resume 状态。历史隔离、备份和转换边界见 [入口约定](ENTRYPOINTS.md#2-run-与产物)。跨运行比较须明确来源 run，并用冻结 selection 指定轨迹；全局 baseline profile 已删除。

## 3. 创建 run，启动 rollout

active run 统一使用 schema-2。升级已有 runs 前先执行一次显式迁移：

```bash
.venvs/runner/bin/dycon runs-migrate-to-repetition \
  --source-root runs --archive-root archive/runs-schema1 \
  --manifest migration/runs-to-repetition-20261024.json --dry-run
.venvs/runner/bin/dycon runs-migrate-to-repetition \
  --source-root runs --archive-root archive/runs-schema1 \
  --manifest migration/runs-to-repetition-20261024.json
.venvs/runner/bin/dycon migration-verify migration/runs-to-repetition-20261024.json
```

迁移后的历史 rollout 以 `repeat_count=1` active run 保存；无 plan 的历史目录只归档为 artifact-only。生产入口不自动读取或迁移 schema-1。

```bash
.venvs/runner/bin/dycon run create RUN_ID --operation rollout --repeat-count 5 \
  --task-id pydata__xarray-4629 --task-id pylint-dev__pylint-7277
.venvs/runner/bin/dycon run show RUN_ID
.venvs/runner/bin/dycon run start RUN_ID
```

baseline、steering、parallel 共用 rollout runner。创建 run 时配置 `parallel.enabled/branch_count/estimator_ref`、`steering.enabled/payload`、`capture.enabled/anchors/layers`，以及 GPU/TP/workers/显存比例。parallel 需要与模型、site 匹配的 predictor 和 close capture。steering 需要真实且 fingerprint 匹配的 payload；没有训练好的 payload 时不伪造 steering 验证。

parallel_consensus 仍使用同一入口；新 run 创建时将 parallel.mode=parallel_consensus、branch_count=32、max_inflight_branches>=32、匹配最新 estimator，并按需设置 parallel.consensus.retention_ratio。它只执行 exact action 共识的代表分支；平票由 estimator 分数裁决，证据写入 decision receipt。

`--repeat-count R` 表示每个 task 的独立 rollout 总数；每个 repetition 使用可复现且不同的 sample seed，基础设施重试不增加 repetition 数。一个 run 内所有 task×repetition 进入同一平坦队列，长尾 task 不阻塞后续 work item。

`run start` 前台等待 supervisor 结束并返回退出证据。后台模式使用：

```bash
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
```

wait 使用进程退出事件，不循环读取 progress。新建 run 时可在子命令前使用 `--set` 选择模式/资源，例如 `dycon --set parallel.enabled=false run create ...`；已有 run 不临时覆盖。parallel B>1 需同时启用 close capture 和匹配的 predictor。

### 3.1 多 GPU 各用 TP=1

在根 YAML 设置 `serving.trajectory_sharding: per_gpu_tp1`、`tensor_parallel_size: 1`、GPU 列表、对应的 `replica_base_urls` 和 `max_cpu_workers_per_gpu`，再按上面的统一入口创建新 run。总 CPU worker 数 = GPU 数 × 每卡上限；该模式不再使用 `execution.rollout_workers` 作为总上限。每卡空闲槽位立即领取下一项任务，一条轨迹及其候选分支始终留在同一卡。

完整 YAML、从已有冻结 run 创建新 clone 的命令及证据位置见[多 GPU TP1](MULTI-GPU-TP1-CN.md)。`run status` / `report run` 返回 `serving_workers`；`worker_id`、`gpu`、`active_cpu_workers` 与 `available_cpu_workers` 分别表示服务编号、设备及槽位占用。新建 clone 从零运行；resume 仍沿原 run 快照与完成证据恢复。

## 4. 状态、恢复、报告与产物校验

```bash
.venvs/runner/bin/dycon run status RUN_ID
.venvs/runner/bin/dycon run reconcile RUN_ID
.venvs/runner/bin/dycon run resume RUN_ID
.venvs/runner/bin/dycon report run RUN_ID
.venvs/runner/bin/dycon artifact show RUN_ID
.venvs/runner/bin/dycon artifact verify RUN_ID
```

配置/tasks/plan 在 `runs/RUN_ID/`，任务尝试在 `rollout/tasks/`，canonical trajectory/patch 在 `rollout/trajectories/`、`rollout/patches/`；执行证据在 `execution/attempts/`，日志在 `logs/`。resume 使用原快照，沿已有 task/attempt 结果恢复，不重新选题；不能保证从工具命令或模型请求中间精确继续。

### 4.1 指定失败任务重跑（带审计记录）

普通 resume 保留终态失败。先停止目标 run、核验实际退出和 cleanup，再登记重跑请求：

```bash
.venvs/runner/bin/dycon run cancel RUN_ID --reason 'explicitly authorized failed-task retry'
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon run status RUN_ID
# 确认 cancellation.termination.cleanup_complete=true 后再继续。
.venvs/runner/bin/dycon run retry-failed RUN_ID --request-id retry-valueerror-001 --exit-status ValueError --reason 'explicitly authorized ValueError retry'
# 可选：明确要求重新生成 CaptureError/blocked 时，另行登记并限定 task ID。
.venvs/runner/bin/dycon run retry-failed RUN_ID --request-id retry-capture-001 --exit-status CaptureError --task-id INSTANCE_ID --reason 'explicitly authorized pending-recovery task regeneration'
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

cancel 后 wait 的非零返回值可能是真实 TERM/KILL 退出码，不代表可忽略清理失败；核验实际 exit/cleanup 后才能继续。CaptureError 命令为可选操作。--exit-status 和 --task-id 都可重复；ValueError 是状态筛选，不会只筛 capture_control_unavailable 错误文字。

核对返回的 sample_count/items，零匹配不会新增重跑样本。resume 先调度显式请求，再继续原有未完成样本，完整 task/work-items 清单和已成功结果保留；待采集恢复的 blocked 样本若要重新生成，须明确登记。

请求和旧 cursor/attempt 副本位于 `runs/RUN_ID/rollout/operator-retries/REQUEST_ID/`。每一轮新的主动重跑使用新的 request ID，旧 ID 只幂等返回原请求。可选的120秒/16线程控制修订、seed 与审计细节见 [维护说明](CAPTURE-CONTROL-RETRY-CN.md)。

## 5. 官方 harness 评分

```bash
# 固定本 run 实际产生的 canonical rollout IDs 和 patch SHA。
.venvs/runner/bin/dycon harness prepare RUN_ID \
  --rollout-id ROLLOUT_ID_1 --rollout-id ROLLOUT_ID_2
.venvs/runner/bin/dycon harness run RUN_ID EVAL_ID
.venvs/runner/bin/dycon harness status RUN_ID
```

`runs/RUN_ID/harness/` 保存 selection、progress、summary、容器环境和官方日志。completed/unresolved/infra_error 分开记录；`--gold` 只验证环境。历史 patch 重新评分时，新建 `--operation harness` 的 run，并显式引用旧 rollout IDs。

## 6. 在线 hidden、NAS 与恢复发布

```bash
.venvs/runner/bin/dycon capture status RUN_ID
# 仅在 pending upload/发布失败时执行；不重新请求模型。
.venvs/runner/bin/dycon capture recover RUN_ID
.venvs/analysis/bin/dycon online-index RUN_ID ONLINE_INDEX_ID
```

NAS tensor 位于 `/VM/wangtianyi/DyConGlobal/hidden/online/RUN_ID/`，本地 run 保存 locator/receipt/publication。parallel 分支用 decision→branch→request→sample 绑定，保留 selected 标记；winner 的 GT 不赋给未选分支。close 未生成和 close 已生成但 hidden 缺失是不同状态。`capture-reclassify` 用于已有 finalized 证据的明确重分类。

## 7. Enrichment 与 token 证据

```bash
.venvs/analysis/bin/dycon enrich RUN_ID
.venvs/analysis/bin/dycon token-trace RUN_ID
.venvs/analysis/bin/dycon trajectory-token-audit RUN_ID
```

enrichment 生成 open/close prefix、remaining labels 和 coverage；token trace 保存 token/phase/step/position 的对齐证据。输入必须是冻结 selection 中的轨迹；精确 native IDs 与 tokenizer 重建证据分开。在线 finalize 在 `runs/RUN_ID/enrichment/online/`；本节离线 prefix/labels 和 token trace 在 `runs/RUN_ID/enrichment/offline/`。只有重新评分为 resolved 的轨迹才具有成功剩余量标签。

## 8. 离线 vLLM 重放与 hidden 提取

```bash
.venvs/analysis/bin/dycon hidden RUN_ID CAPTURE_ID
.venvs/analysis/bin/dycon dataset-index RUN_ID CAPTURE_ID INDEX_ID
# 有 exact token evidence 时再做 token-hidden join。
.venvs/analysis/bin/dycon token-join RUN_ID CAPTURE_ID JOIN_ID
.venvs/analysis/bin/dycon --run-id RUN_ID hidden-token-join-audit JOIN_ID
```

重放读取冻结 prefix，由 latent-vllm eager 提取 hidden，复用 trajectory APC；同轨迹 anchor 有顺序，不同轨迹可并发。大规模历史 extraction 的固定批次 launcher 只用于其原批次，不在此最小验证中启动。在线/重放数值对照使用 `hidden.evaluation.replay_online_artifacts --help` 所列输入；图执行、数值一致性和任务成功率是不同指标。

## 9. Action 分类与共享 hidden 视图

```bash
.venvs/analysis/bin/dycon action-classify ACTION_ID --source-run SOURCE_RUN \
  --rollout-id ROLLOUT_ID_1 --rollout-id ROLLOUT_ID_2
.venvs/analysis/bin/dycon action-check ACTION_ID
.venvs/analysis/bin/dycon artifact verify ACTION_ID --scope action-analysis
.venvs/analysis/bin/dycon action-plot ACTION_ID
.venvs/analysis/bin/dycon action-plot ACTION_ID --plot-id MACRO_VIEW --aggregation macro --format png --format svg
.venvs/analysis/bin/dycon action-index INDEX_ID ACTION_INDEX_ID
```

action 输出独立于 run，位于项目级 `action/outputs/ACTION_ID/`；`analysis.json` 冻结“哪些 run 的哪些轨迹”。分类直接读取 canonical trajectory，自动生成互斥标签、阶段占比图、转移概率图和 receipt，不依赖 enrichment。多 run 使用重复 `--source-run` 和带 `source_run_id` 的 JSON binding 清单（见 action/README.md）；不需要创建 analysis run。后加图使用独立 view receipt，不能覆盖父分析。action-index 在项目级引用已有 hidden，不重复提取。action taxonomy 学习、训练与独立评测尚无正式实现。

## 10. Difficulty 数据、划分、训练和评估

```bash
# 每个 SOURCE_RUN 单独生成索引/成功轨迹训练集；可组合多个训练集。
.venvs/analysis/bin/dycon online-index SOURCE_RUN INDEX_ID
.venvs/analysis/bin/dycon difficulty-dataset resolved DATASET_ID --source-run SOURCE_RUN --index-id INDEX_ID
.venvs/analysis/bin/dycon difficulty-dataset compose COMBINED_ID --source-dataset DATASET_A --source-dataset DATASET_B
.venvs/analysis/bin/dycon difficulty-split DATASET_ID SPLIT_ID --strategy event_random_v1 --seed 42
.venvs/analysis/bin/dycon difficulty-ridge PLAN_ID --dataset-id DATASET_ID --split-id SPLIT_ID --mode dry-run
.venvs/analysis/bin/dycon difficulty-ridge ATTEMPT_ID --dataset-id DATASET_ID --split-id SPLIT_ID --mode fit
.venvs/analysis/bin/dycon difficulty-mlp MLP_ATTEMPT_ID --dataset-id DATASET_ID --mode fit
.venvs/analysis/bin/dycon artifact verify ATTEMPT_ID --scope difficulty-attempt
.venvs/runner/bin/dycon predictor-publish PREDICTOR_ID --experiment difficulty/checkpoints/ATTEMPT_ID --target reasoning_length
.venvs/runner/bin/dycon difficulty-predict INDEX_ID runtime/parallel-artifacts/PREDICTOR_ID/manifest.json PREDICTION_ID
.venvs/analysis/bin/dycon difficulty-evaluate INDEX_ID difficulty/predictions/PREDICTION_ID/predictions.json SCORE_ID
```

训练参数仍在唯一 `config.yaml` 的 `difficulty.ridge_probe`；允许的 `--set` 只影响新 attempt，effective settings 与源码/环境 hash 写入 attempt.json。
dataset、split、attempt 均有独立 ID 和不可覆盖的 receipt；输出始终位于 `difficulty/`，无需 analysis run。
dataset split 在 `difficulty/datasets/DATASET_ID/splits/SPLIT_ID/`；模型/报告在 `difficulty/checkpoints/ATTEMPT_ID/`。
checkpoint 的 NPZ 与 JSON 标明架构、输入 site/维度、target/变换、数据/划分 receipt 和 fit scope。
独立预测与评估分别写入 `difficulty/predictions/ID/`、`difficulty/evaluations/ID/`。
schema 1 scorer/predictor 的现有能力边界保持；MLP 通过 `difficulty-mlp` 生成离线 attempt。MLP 参数和 scheduler 在 `difficulty/training/config.yaml`，Ridge+FFN 仍是 Python API。
旧 run 目录保持原样，可显式提供旧 index/experiment 目录读取，不能用相同 ID 隐式回退。

## 11. 并行分支可视化和评分稳定性

```bash
.venvs/analysis/bin/dycon difficulty-plot-parallel SOURCE_RUN_ID --list
.venvs/analysis/bin/dycon difficulty-plot-parallel SOURCE_RUN_ID PLOT_ID \
  --rollout-id ROLLOUT_ID
.venvs/analysis/bin/dycon difficulty-evaluate-parallel SOURCE_RUN_ID INSTABILITY_ID \
  --rollout-id ROLLOUT_ID
```

绘制分支得分、实际 winner 轨迹、缺失分数；稳定性分析按相同 action/prefix 比较 reasoning 扰动，产出指标、置信区间和 receipt。图中的 winner 必须来自记录，不能重新取最小值冒充实际选择。

## 12. 运维与诊断

环境安装脚本在 `runtime/scripts/setup_*.sh`；Docker/NAS smoke、capture matrix、eager replay smoke 和 owned-resource recovery 是维护工具。启动前核对 GPU/作用范围，不重复导入 benchmark、不无条件运行 gold repair、不删除其他任务的资源。`dycon recover` 只处理本工程遗留资源。额外验证命令及副作用见 [ENTRYPOINTS.md](ENTRYPOINTS.md)。

## 历史验收与适用边界

日常清单不承载单批次结果。2026-10-21 的两题回归、baseline/parallel、离线重放数值对照和迁移记录保存在各 run、工程外 `/home/wangtianyi/dycon-isolation-20261021/` 以及本次 docs 原文归档；位置见 [入口约定](ENTRYPOINTS.md#9-文档历史与追溯)。旧 snapshot/receipt 不因文档整理而变更。

没有真实匹配 payload 时不执行 steering 验收；已有数值 checkpoint 的发布/推理不等于训练完成。`run cancel` 写入请求后，由 runner 停止派发并通知在途请求，supervisor 按执行身份管理进程与容器收尾；`reconcile` 可接管已退出 supervisor 遗留的取消清理。是否结束以实际退出记录与 `cleanup_complete` 为准，详见入口表和后文取消说明。

## Run 级轨迹选择

Catalog 只维护 benchmark tasks；轨迹/评测写到 `runs/RUN_ID/evidence.json`。本次仅迁移 `migration-closeonly331-20261021`。新建数据集可用 `online-index SOURCE_RUN INDEX_ID --rollout-list FILE`，随后 `difficulty-dataset resolved DATASET_ID --source-run SOURCE_RUN --index-id INDEX_ID --rollout-list FILE`。列表中的轨迹必须属于给出的来源 run。

## GPU MIG serving

Managed rollout serving accepts physical GPU indices or one MIG instance UUID in serving.gpus. In the default single-service mode (trajectory_sharding absent or none), the number of entries must equal serving.tensor_parallel_size; one MIG instance therefore uses TP=1. The per_gpu_tp1 mode starts an independent TP=1 service per configured GPU and uses a separate replica_base_urls entry for each; its GPU count is not its TP size. This mode and the per-GPU CPU worker limit are documented in [Multi-GPU TP1](MULTI-GPU-TP1-CN.md). The current device validator still rejects mixing physical indices with MIG UUIDs or listing multiple MIG instances. The UUID must be discovered at launch time with nvidia-smi -L and frozen in the run resolved configuration. Example:

    serving:
      gpus: ["MIG-INSTANCE-UUID"]
      tensor_parallel_size: 1

Run creation and execution use the normal dycon commands. The serving manager preserves the UUID in CUDA_VISIBLE_DEVICES, checks visibility and records device memory in serving/launch.json (or serving/workers/worker-NN/launch.json in per_gpu_tp1 mode). MIG instances retain the existing device lock; physical GPUs permit sharing between managed services, so memory admission does not imply exclusive compute access. Physical GPU configurations such as gpus: [0, 1, 2, 3] with TP=4 remain supported in the default mode. Offline hidden replay keeps its separate integer-GPU contract.

### Harness ????

`harness run` records a stable outcome for every selected rollout:

- An empty or unappliable patch is `status=completed`, `resolved=false`, `failure_type=invalid_patch`.
- A transient network/connection failure is retried by the unified harness at most two times. Each attempt has its own evaluation task ID and is retained in the result; after exhaustion the outcome is `status=ConnectionError`, `resolved=null`, `failure_type=network_error`.
- Other execution failures remain `status=infra_error`, `resolved=null`, `failure_type=other_error` (or the more specific non-network classification).

The summary reports `network_retries`; retrying infrastructure does not create a new rollout, repetition, or canonical trajectory.


### difficultyEstimator 可配置训练

编辑独立配置 `difficulty/training/config.yaml`，执行 `source difficulty/scripts/train.sh`；可选传入另一 YAML 路径。入口通过 analysis venv 的正式 CLI 训练并生成新的 attempt。TensorBoard 默认绑定服务器 loopback；在本地建立 `ssh -N -L 16006:127.0.0.1:6006 wangtianyi@10.249.187.9`，打开 `http://localhost:16006`。使用 `.venvs/analysis/bin/dycon difficulty-tensorboard ATTEMPT_ID --action status|stop` 管理服务。


## Docker ??????

? `config.yaml` ? `environment.docker_data_root` ?? rootless Docker ?????????

????????? `paths.artifact_roots` ???`paths.artifact_scan_roots` ?? run/cache ?????????????????? `config-check`????? run ???????? run ????????

?????????????????? `.venvs/runner/bin/dycon docker-storage apply`??? `docker-storage status` ?????????????????????????????????????????????? NAS ???????????????????????

## Rollout 完整轨迹换 seed 重试（2026-10-26）

根 `config.yaml` 显式启用 `execution.limits_retry.enabled`，并冻结
`retry_statuses: [LimitsExceeded, reasoning_loop]`、
`seed_policy: repetition_round_offset_v1`。配置缺省关闭；旧 run 的快照、hash、
已完成 LimitsExceeded 样本和默认轨迹选择不变。仅新建 run 采用新策略。

整条 agent 轨迹以两种目标状态之一结束后，先保存原轨迹并完成 capture、容器清理，
然后把同一 task×repetition 放回平坦队列尾部，从干净环境以新 seed 重做。
这不是 step 续写，也不增加 repetition 或提高 token/step/cost limit。
其他已有正常终态（包括 ContextWindowExceededError）结束该 sample；Submitted 仍不等于 resolved。

初始 seed 与旧规则一致。设 base seed 为 S、repeat_count 为 R、repetition index 为 r、
seed round 为 k，冻结 branch_stride 为 B，则 `effective_seed = S + (r + k*R)*B`；
分支使用 `effective_seed + branch_id`。每次执行校验当前 vLLM OpenAI API 的 signed-int64
范围，禁止回绕。sample_seed 始终是计划初始值；effective_seed 表示真实执行值。

LimitsExceeded 和 reasoning_loop 无默认次数上限；infra retry 保持同一 seed，
按 sample 累计 `execution.infrastructure_retry_limit`，不因换 seed 刷新。
capture 的 block_rollout 故障优先阻断重试；未知故障不能冒充 accepted。
正式运行前应配置合理的 `execution.pipeline_wall_seconds`。`run cancel` 停止新派发与重试，并通知准入和在途模型请求；supervisor 在宽限期内管理本 execution 的资源收尾。请求被接受不等于任务已退出或资源已清理，须核对取消与清理证据。

每次真实尝试保存 `attempt-context.json`、`agent-outcome.json`、`attempt.json`；
`retry-state.json` 是可重建 cursor，`result.json` 只发布 accepted 或明确 failure。
恢复复用已经确认的 agent 结果和 canonical SHA；capture 已发布的任务沿既有
`dycon capture recover RUN_ID` 恢复，不重新调用模型。缺少可验证的 capture 收尾证据时，
保留 CaptureError 和原轨迹，先诊断恢复，不能跨过去换 seed。

运行统计固定在 `runs/RUN_ID/rollout/limits-exceeded-counts.json`，分别记录
limits_exceeded_count、reasoning_loop_count，以及两者合计 resample_trigger_count，
包含 run/task/sample 明细。计数源为原始 agent_exit_status，capture 阻断或取消仍保留
已发生的次数；实际启动新 seed 才增加 resample_retry_count。
`rollout/statistics-policy.json` 标识新 run 的统计协议；旧 run 不补写。
resume 和最终收尾核对/重建计数；artifact verify 对已结束 run 校验汇总与原始事实一致。

`run status`、`report run` 与 progress 使用同一统计来源；缺失的旧统计显示未记录。
progress 还显示 ready_samples、running_attempts、active_generation_samples、
finalizing_attempts、admission_waiting_samples；active_generation_samples 来自真实模型传输阶段。
attempt 的 timings 区分排队、admission、容器初始化、agent、capture drain/finalize 和统计写盘。
平坦队列没有 repetition 屏障，但工具执行、capture 收尾、背压和末尾独立 sample 不足仍可导致 GPU 空等。

启用新策略的 run 结束后，默认 harness/analysis/action/difficulty selection 只取每个
accepted sample 的 final rollout；run 尚未结束时拒绝冻结默认完整输入。
诊断中间尝试可显式给 rollout ID，记录 explicit_intermediate；原终态和 null 评测标签不变。
总费用、token 和时间应覆盖全部物理尝试（parallel 的 total_usage 包含所有分支），
`report run` 提供 all_attempt_usage。
这些结果是排除 LimitsExceeded / reasoning_loop 后接受的条件采样，不能当作固定 seed 单次采样的无条件结果。

Live phase counters are published in `rollout/phases.json` and merged into the
progress view by `run status` and `report run`. Model requests never rewrite the
full cohort or scan outcome history to publish these counters.


## Parallel 跨 step 主动循环脱离

根 `config.yaml` 中设置 `parallel.loop_escape.enabled: true` 后，在新建 run
的冻结配置中启用；默认 `false`。已有 run/resume 使用原快照，缺少该配置的
旧快照仍视为关闭，旧证据不回写。`parallel`、`parallel_baseline`、
`parallel_consensus` 共用此预筛选，然后各自沿用原有 score/fallback、均匀随机、
exact-action 共识选择。诊断 `first_valid_unscored` 不启用此机制。

每个 rollout attempt/model 独立维护 branch slot 历史（含未选中但解析成功的输出）。
槽位使用固定 branch ID 与 `generation.seed + branch_id`，与并发完成/admission
顺序无关；它不是连续保留分支自身世界状态的 tree search。从第二个实际 parallel
decision 开始，仅比较同槽位的全部先前有效输出；非 parallel step 不增加历史。
优先保留 action 新变化的分支并排除 action 重复者；只有所有有效 action 均确认
重复时才使用 reasoning 的变化。action 复用 canonical tool-action 身份，忽略 tool
call ID、JSON key 顺序和传输字段，保留命令空白、执行参数和 action 顺序。
缺失/不可比较的 action 或缺失/空 reasoning 是 unknown，不计为新变化，保守保留。

若全部可比较输出重复，或过滤后不满足原模式的可选条件（例如共识不足两条），
记录主动脱离失败并恢复本轮真实候选，交回原选择策略；不伪造 winner，不提前终止。
`agent.loop_repeat_limit: 8` 及其周期历史保持原样，最终由原 detector 报
`LimitsExceeded`。单次生成的 `parallel.reasoning_loop` token 检测不变；原有
baseline scoring error、API/format/limit 错误优先级和共识最少分支约束保持不变。

decision receipt 的 `loop_escape` 记录版本、parallel decision 序号、筛选阶段、
前/建议/实际候选和失败原因。各 candidate 的独立 `loop_escape` 记录历史可用性、
指纹、比较、匹配历史 decision、排除状态和原因；不会冒充 estimator_filtered 或
共识投票结果。写 receipt 时从同 session 先前 receipts 重放历史并校验；关闭时
保留旧 evidence 形态。配置检查使用 `.venvs/runner/bin/dycon config-check`。
# 高并发取消与验收（2026-10-26）

新版本取消不再仅等待 attempt 边界：runner 会停止派发、唤醒准入和在途 native 请求，supervisor 在独立宽限期内兜底终止本 execution 的 rollout、专属 serving 与容器。请求写盘、执行侧 ACK、实际退出和资源清理是不同状态。

```bash
.venvs/runner/bin/dycon run cancel RUN_ID --reason 'user requested stop'
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon run status RUN_ID
```

检查 `cancellation.termination.cleanup_complete`，不能仅凭 `cancel_requested=true` 判断停止。真实返回码可非零，已停止不等于任务成功。controller 异常退出后，已有取消请求可用 `run reconcile RUN_ID` 按原身份继续收尾；有资源残留时先处理清理，不能直接启动新 execution。

显式 `run resume RUN_ID` 保留旧取消历史，创建新 execution 代际；不会改旧快照或重新选择任务。取消的 sample 不接受为成功，也不自动换 seed/infra retry。采集数据继续按原 `capture recover` 恢复；不补造完整 native output 或 hidden。

真实并发验收先修改根配置、创建新 run，再执行 `run start NEW_RUN --detach --check-only`。`rollout_workers=4, branch_count=32, max_inflight_branches=64` 时完整提交四组请求；`parallel.validation_repetitions: 10` 覆盖十轮 overlap，报告含容量饱和等待峰值。验收会真实占用 GPU 和容器，应使用新 ID，并核对设备和服务端口。


## SkillsBench image preparation (2026-10-26)

The `skillsbench_images` block in root config.yaml owns the pinned source,
NAS publication, build workers, timeout, proxy and builder identity. This
stage prepares images only; it does not register SWE-shaped fake tasks or run
agents, oracle or verifier. Official `tasks/` only; `tasks-extra/` is excluded.

```bash
.venvs/runner/bin/dycon skillsbench-images plan
.venvs/runner/bin/dycon skillsbench-images prepare
.venvs/runner/bin/dycon skillsbench-images status
.venvs/runner/bin/dycon skillsbench-images verify
```

Prepare is resumable and locked; `--task-id` selects a temporary preparation
subset while the frozen publication still declares all official tasks. The
command exits 2 until all images are complete. A compose sidecar is a separate
image. Original task Dockerfiles remain unchanged. A separately staged Dockerfile adds a recorded apt mirror/retry profile; its SHA and profile are saved per image. Build logs
and progress live in `state/skillsbench-images/STORE/`; content-addressed OCI
blobs, per-service image records, frozen plan and final image manifest live in
`paths.nas_root/STORE/`. `verify` deeply hashes all published graphs and exits 2
if anything is missing. This verifies image integrity, not task correctness.

Builds use the native rootless Docker context, verified against environment.docker_host. Source images/frontends are fetched through the NAS registry client and preloaded locally. Daemon proxy settings are unchanged; no daemon restart or global prune. Apt mirrors retain signature checks; invocation-specific configuration receipts are saved under attempt-configs/. A 100 GiB free-space
floor protects /data1 before admission. Skill injection and verifier runtime
dependencies are not added by this stage. Existing NAS/OCI importer and loader
are reused. Changing source/environment requires a new publication store.

## Action 分支统计最小清单（2026-10-26）

先选来源 run，再按 rollout_id 选轨迹；同一 task 的多个 rollout 分别绘图，不合并。

```bash
# 该 run 的全部 eligible/final 轨迹，一条一图。
.venvs/analysis/bin/dycon action-branch-stats NEW_BATCH_ID \
  --source-run RUN_A --line-mode per-trajectory

# 仅指定的一条轨迹；图中只展示归一化熵。
.venvs/analysis/bin/dycon action-branch-stats NEW_SINGLE_ID \
  --source-run RUN_A --rollout-id ROLLOUT_ID \
  --line-mode per-trajectory --line-metric normalized_entropy

.venvs/analysis/bin/dycon action-branch-stats NEW_BATCH_ID --check
```

可选 `--termination-reason Submitted`，或用 JSON binding 清单选择轨迹子集；
进行中的重采样 run 仍需显式选择已发布的轨迹，不绕过原有 eligible/final 规则。
检查 `plots/trajectory-index.json`：每条所选 canonical 轨迹都有图或不可用说明图。
`curves/per-trajectory.csv` 保留 rollout_id、原始整数 step 和 value，不应出现 task 平均值。
普通 run 平均图用 per-run/all-runs/both，横轴继续为0–100%；`--task-id` 已移除。
完整说明见 [action 分支统计](../action/BRANCH-ACTION-STATS-CN.md)。

### 单 run 平均图最短操作（2026-10-27）

```bash
.venvs/analysis/bin/dycon action-branch-stats NEW_RUN_MEAN_ID \
  --source-run RUN_ID --branch-scope both --line-mode per-run --progress-points 101
.venvs/analysis/bin/dycon action-branch-stats NEW_RUN_MEAN_ID --check
```

不传rollout/终态筛选，使用该run全部默认eligible/final轨迹；平均图横轴0–100%，
先逐轨迹归一化再等权平均。需要逐轨迹单独成图则用per-trajectory，原始step不归一化。
检查selection、summary和curves/per-run.csv中的n_trajectories；缺失轨迹不当作零。
最新`consensus32-all-trajectory-average-20261027`包含86条选择，76条有parallel step，
不能与只选Submitted的子集混为同一统计范围。

## Repaired capture publication

Use `dycon capture-publish RUN_ID --repair-id REPAIR_ID --publication-id PUBLICATION_ID --resolved-only`
to publish complete native-exact repaired sessions as online finalized token labels.
The command verifies exact harness patch outcomes and preserves original evidence.
Partial or unbound sessions are explicitly excluded.
See [capture publication guide](CAPTURE-PUBLICATION-CN.md) for policy and downstream training commands.


### SkillsBench image integrity completion: 2026-10-27

The pinned 87-task / 88-service image publication is complete under
skillsbench-f860ecf02-20261026, with images-manifest.json and full
skillsbench-images verify SHA/OCI checks. This is image availability and
integrity, not oracle/harness success. Upstream task files remain unchanged.

Task-specific build compatibility is declared only in root
skillsbench_images.task_compatibility and recorded in image provenance:
SeisBench gets verified setuptools 80.9.0 for its missing pkg_resources provider;
the TicToc tar extraction skips unmappable archive ownership; Coursier uses
explicit, dynamically rendered proxy JVM flags with unchanged Scala 2.13.12.
verified_wheels mounts identical official-hash-checked artifacts read-only;
it does not bypass hashes or substitute package versions. AgentOps uses a
source-linked, metadata-only rootless ownership derivation in the separate
skillsbench-ownership-f860ecf02-20261026 NAS store. Oracle validation remains
explicitly not_run for these environment adaptations.

The build-only socat forwarder is managed by prepare and is closed afterwards.
It does not replace the existing 7897 service or change daemon proxy settings.
Final receipts and adaptation details are in
state/skillsbench-images/skillsbench-f860ecf02-20261026/final-summary.json.

## Capture spool archival

Use `dycon capture archive RUN_ID --archive-id ARCHIVE_ID` to move inactive run spool history to NAS.
All file hashes are checked before local duplicates and run-specific reservations are released.
Historical evidence paths remain readable. See [capture archive guide](CAPTURE-ARCHIVE-CN.md).


## SkillsBench startup repair validation (2026-10-30)

Maintain shared settings in root `config.yaml`; create a fresh run after changing the budget or profile. The current repair candidate keeps `generation.thinking_token_budget=81920` and reserves action space with `generation.max_tokens=98304`, `require_reasoning_close=true`. Persistent close validation requires managed FULL capture, the think_close anchor and export. These candidate values are not a declaration that long-reasoning GPU acceptance has passed.

Use a new dependency publication with `--python-runtime`, then a profile bound to it. A changed CLI/tools publication needs a real request-role qualification probe; old qualification is not silently transferred across changed tool wheels. Initial qualification can use an explicitly uncaptured baseline, followed by a freshly qualified profile for capture/parallel.

```bash
.venvs/runner/bin/dycon skillsbench preflight NEW_RUN_ID --workers 4
.venvs/runner/bin/dycon run start NEW_RUN_ID --detach
.venvs/runner/bin/dycon run wait NEW_RUN_ID
.venvs/runner/bin/dycon skillsbench audit NEW_RUN_ID --rollout-id ROLLOUT_ID
```

Preflight performs real task environment checks without LLM/GPU generation. Start/resume shares that check and then starts serving. Before a 75x3 launch, require every task environment receipt, representative real agent/tool runs, long-request cancellation, parallel isolation, and schema-3 per-step native/hidden coverage. Readiness, equal point counts and unit-test counts alone do not release the batch. Stop the authorized predecessor with `run cancel` and verify cleanup; preserve its frozen evidence.



## SkillsBench 修复后正常启动与 run 规模（2026-10-30）

修复范围已经接入统一生产入口：`dycon run create` 冻结配置，`run start`/`run resume` 先执行共享 SkillsBench preflight，再启动 managed serving 和 BenchFlow；`run retry-failed` 注册新的 physical attempt 后仍走同一 resume 链路。旧 run 的 resolved-config、selection、receipt 和历史轨迹不修改。普通 baseline、capture、parallel 和 cancellation 共用这些 owner；未启用的能力不会被隐式打开。

本轮修复覆盖：

- reasoning/action 预算和 tokenizer-bound close grammar；默认候选为 reasoning `81920`、总输出 `98304`，剩余空间留给 close、tool-call 和终止 token。
- native API attempt terminal receipt、实际 token usage、结构化 limit/protocol/transport/cancelled 错误和 request correlation。
- OpenHands 普通 baseline 的 reasoning token 周期检测，以及跨 assistant step 的 reasoning/action 周期检测；命中后在工具执行前早停，保留原始响应和检测证据。parallel candidate 仍保留原有 per-branch detector 和 loop escape。
- capture session draining、延迟 native output、幂等 stop/finalize、session 级取消和 condensation selector 隔离。
- hash-bound CPython 3.12 agent runtime、subprocess terminal backend、SDK bounded timeout、非 root skill discovery、compose image identity 和 75-task environment audit。
- workspace file/directory 探测、逐 step native/hidden coverage、全候选 parallel evidence、API terminal coverage 和 run-local startup failure circuit breaker。

新 run 的推荐流程：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon skillsbench dependencies-verify \
  runtime/dependency-snapshots/openhands-startup-safe-v2-20261030

# 使用已审核的 SkillsBench snapshot/profile；run 创建时冻结它们
.venvs/runner/bin/dycon run create RUN_ID --operation rollout \
  --repeat-count 3 --task-id TASK_ID_1 --task-id TASK_ID_2
.venvs/runner/bin/dycon skillsbench preflight RUN_ID --workers 4
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon run status RUN_ID
.venvs/runner/bin/dycon skillsbench audit RUN_ID
.venvs/runner/bin/dycon artifact verify RUN_ID
```

`preflight --workers 4` 会对所选 task 建立真实 BenchFlow sandbox，检查镜像、离线 agent runtime、非 root terminal/ACP、skill visibility、compose image identity 和 cleanup；它不发 LLM 请求。`run start/resume` 会再次通过同一内部 gate。只有在 preflight、真实 agent/tool smoke、逐 step audit 和 artifact verify 均通过后，才扩大到完整批量。

若改变 model/tokenizer、dependency publication、profile、generation budget、capture 或 serving placement，应创建新 run；resume 不重新读取根 YAML。`run cancel` 后必须检查 `cleanup_complete=true`，再进行 retry/resume。`run retry-failed` 使用新的 request/attempt ID，不覆盖旧证据。

本次验证 run 的真实规模如下：

| Run | 任务 × repetition | 用途 | 结果 |
|---|---:|---|---|
| `skillsbench-agent75-qualified-thinkclose-20261030` | 75×3 | 原问题 run | 已停止，保留历史证据 |
| `skillsbench-startup-qualify-v2-20261030` | 1×1 | GPU 2、3 baseline/长 reasoning | capture complete；reasoning loop/协议失败证据独立保留 |
| `skillsbench-capture-loops-v4-20261030` | 4×1 | GPU 2、3 capture、循环和逐 step | 4 个 attempt 完成；reasoning loop 早停；audit/artifact verify 通过 |
| `skillsbench-parallel-v5-20261030` | 1×1，2 branches | GPU 2、3 parallel candidate | 8 canonical steps、16 candidate native receipts；audit/artifact verify 通过 |
| `skillsbench-unified-repair-env75-v4-20261030` | 75×1 | 全任务环境预检 | 75/75 environment receipts 通过；未执行 LLM rollout |
| `skillsbench-offline-startup-v6-20261030` | 1×oracle | offline install/oracle | 1/1 valid oracle；不代表 OpenHands rollout |
| `skillsbench-noskill-startup-v6-20261030` | 1×preflight | no-skill 可见性 | environment preflight 通过 |

因此，修复后的新验证 run **不是 75×3**。`75×3` 只属于原问题 run 的冻结计划；本轮修复已经完成 75-task 环境覆盖，但尚未启动修复后的完整 75×3 agent rollout。完整 75×3 必须在新的 run 中创建，且先通过 G0–G4 gate。

`skillsbench audit` 当前生成 schema-4 receipt：逐 step 分母是全部 canonical assistant responses（含最终回答），并列出 missing、duplicate、unexpected、unfinished、API terminal coverage；parallel 还单独列出所有候选和未选工具执行情况。官方 verifier 的 resolved truth 与基础设施/capture truth 分开保存。
