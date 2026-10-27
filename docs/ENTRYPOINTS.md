# 任务入口与接口约定

## Harness requirements 元数据预取（2026-10-26）

运行前可用 runner 的 `dycon harness metadata-prefetch --instances INSTANCES.json --output NEW_RECEIPT.json [--proxy URL] [--refresh]`，按当前根 config 的 `environment.requirements_cache` 缓存官方 requirements。输入为完整 instance dict 列表，使用 `environment_setup_commit`（否则 base_commit），复用官方递归 requirements reader，不创建容器或修改评分定义。

`--refresh` 对精确 commit 的已有缓存重新取源并比对字节，内容不一致会拒绝替换；正常缓存命中校验 SHA，真实 404/410 保留官方候选路径 fallback。receipt 记录 URL、来源、状态码、内容 SHA 和各 task 结果。已有无来源缓存标为 `legacy-cache-unattributed`，可 refresh 后补齐来源。输出拒绝覆盖，incomplete 返回非零。proxy 只用于宿主预取，不改变 task 的离线依赖或容器网络策略。

共享 wheelhouse 的 `--download-build-tools` 现包含 `flit_core==3.9.0`，覆盖旧 Python 镜像中 Sphinx 的 `flit_core>=3.7` editable build 要求；依然通过 PyPI release SHA 校验并创建新的 immutable publication，不修改旧 wheelhouse。

## Pointwise capture recovery

`dycon capture repair RUN_ID --pointwise --apply --repair-id REPAIR_ID` recovers
every integrity-checked retained point, including points from incomplete sessions.
Use repeated `--session-id` arguments to freeze an explicit cohort. Original run
evidence and finalized sessions are never overwritten. Publication is resumable
under the same repair ID and selection. Local reports live in
`runs/RUN_ID/hidden/capture-repairs/REPAIR_ID/`; NAS metadata lives under the
run's configured capture export root in `capture-repairs/REPAIR_ID/`.

Native token/response evidence, identity-only bindings, and orphan tensors are
separate classifications. No missing token IDs or hidden tensors are invented.
The known expected denominator is explicitly incomplete when native receipts
are absent. All pointwise exports default to `eligible_for_training=false`;
partial-session points must be explicitly selected and qualified downstream.
`hidden.artifacts.read_repaired_session(NAS_RUN_ROOT, REPAIR_ID, SESSION_ID)`
verifies a repaired session receipt. Its point rows can be read with the existing
`read_arrays(NAS_RUN_ROOT, row)` reader. Tensor dependencies retain their original
NAS locations and checksums, avoiding needless tensor duplication.



核对日期：2026-10-21。工程根目录 `/home/wangtianyi/DyConGlobal`，NAS 根目录由 `config.yaml` 的 `paths.nas_root` 指定。



日常操作从 [最小运行清单](MINIMUM-RUN-CN.md) 开始；依赖、Docker、镜像、网络和 harness 环境故障查 [容器环境修复指南](container-environment-repair-cn.md)。本文保存完整入口表、输入格式、产物与分析口径。模块内部实现另见源码及各模块 README。



## 1. 入口、环境与配置



统一入口为 `core.cli:main`。已有入口时直接复用，不能复制 launcher、绕过锁/快照启动 worker，或另写模型请求和产物登记逻辑。active run 统一使用 schema-2；旧 runs 由一次性 `runs-migrate-to-repetition` 迁移，生产入口不保留 schema-1 fallback。



```bash

cd /home/wangtianyi/DyConGlobal

.venvs/runner/bin/dycon --help

.venvs/runner/bin/python -m core.cli --help

```



| 环境 | 主要用途 |

|---|---|

| `.venvs/runner` | run、rollout agent、Docker、官方 harness、Catalog、predictor 发布/推理 |

| `.venvs/analysis` | enrichment/tokenizer、action 分类、数据分析、Ridge、绘图 |

| `.venvs/easysteer` | managed serving、在线 hidden capture；由 serving manager 启动 |

| `.venvs/latent-vllm` | 独立 vLLM eager replay；由 `hidden` 入口派发 |



全局参数放在子命令前：`--config config.yaml`、`--set key=value`、`--run-id RUN_ID`。`--run-id` 可加载已有 run 的冻结配置；enrichment/hidden 输出归属 run，difficulty 与 action 始终写项目级目录。action 不需要 analysis run，来源由可重复的 `--source-run` 明确指定；一个分析可组合多个 run 的轨迹。`--run-id` 不改变 action 输出位置，也不隐式代替 source run。



根 `config.yaml` 保存共用配置，generation、agent step/token limit 和所有 timeout 按项目约定在根配置维护。run 私有模型/资源、capture/parallel/steering 等设置在 `run create` 时冻结；已有 run 的执行、resume 和分析拒绝临时 `--set`。当前 CLI 对 generation/agent/timeouts 的覆盖已有拒绝检查；嵌在其他配置块中的 timeout 也应遵守根配置约定。



`run create --set` 实际写法是 `dycon --set ... run create ...`。不带 run 的部分旧数据分析入口仍读取根配置并允许显式覆盖，不能将它与 run-scoped 输出混用。根 YAML 后续改变不应改变已创建 run 的有效配置。



## 2. Run 与产物



| 操作 | runner 命令 | 行为与边界 |

|---|---|---|

| 检查配置/索引 | `config-check`、`catalog-check` | 校验 strict YAML；校验 benchmark/task 清单；运行产物使用 `artifact verify RUN_ID` |

| 建立实验 | `run create RUN_ID --operation rollout --repeat-count R --task-id INSTANCE_ID` | `repeat-count` 表示每个 task 的独立 rollout 总数，默认为 1；也支持 `--tasks-file FILE`、`--partition train/val/test/excluded`；显式 ID/列表应非空 |

| 查看计划/列表 | `run show RUN_ID`、`run list` | show 含冻结 tasks；list 只列有新版 plan 的 run |

| 启动 | `run start RUN_ID [--detach]` | 当前 supervisor 执行 rollout；其他 operation 使用其专门命令 |

| 事件等待 | `run wait RUN_ID` | Linux pidfd 等待 supervisor 退出，再读 exit receipt；不轮询 progress |

| 直接执行 rollout | `rollout RUN_ID [--resume] [--check-only]` | 新版用已创建 run；`--check-only` 真实启动模型、容器和 capture gate，有资源副作用 |

| 恢复 | `run resume RUN_ID [--detach]` | 使用原 selection/config；完成的 terminal task 复用原结果；未完成 task 可能开启新 attempt |

| 状态 | `run status RUN_ID`、`harness status RUN_ID`、`capture status RUN_ID` | 只读投影和各阶段证据；rollout complete 不等于 harness resolved 或 capture complete |

| 协调检查 | `run reconcile RUN_ID` | 当前主要校验 plan/config/tasks 一致性，不能替代所有环境/依赖/GPU 验收 |

| 取消请求 | `run cancel RUN_ID --reason TEXT` | 当前写 cancel-request 记录；不能据此保证 worker 已退出或容器已清理 |

| 产物检查 | `artifact show RUN_ID`、`artifact verify RUN_ID` | 列文件，或校验已登记产物、receipt、canonical 输入、NAS 元数据；不等于重新检查 NAS 所有大张量 |

| 汇总 | `report run RUN_ID` | 输出 JSON 状态、结果与引用；当前没有 `--format csv/markdown` 参数 |



task ID 是题目，`task_key` 是 Catalog 任务身份；`attempt_id` 是尝试；`rollout_id` 是 canonical trajectory。评测选择轨迹而不是仅选 task，必须绑定实际 patch SHA。



```text

runs/RUN_ID/

  plan.json / config.yaml / resolved-config.json / tasks.json / selection.json

  code-identity.json / status.json / progress.json / events/

  execution/attempts/NNN/          # supervisor/child 身份、exit receipt

  logs/                          # supervisor.log、阶段日志

  rollout/tasks/                 # 各 task 的 attempt/result

  rollout/trajectories/ROLLOUT_ID/raw.traj.json

  rollout/patches/ROLLOUT_ID/patch.diff

  harness/                       # selection/selection-receipt/progress/summary/日志/环境

  analysis-selection.json         # 冻结的后处理轨迹选择

  enrichment/online/             # online finalize labels、coverage、request/index receipt

  enrichment/offline/             # 离线 prefix/labels、token-traces

  hidden/outputs/CAPTURE_ID/      # 离线 replay tensor、index、receipt

  hidden/publication.json         # NAS session metadata locator/hash

  # difficulty 新产物位于项目级 difficulty/，不在本 run 中；旧 run 内历史产物保持原位。

  # action 新产物在项目级 action/{outputs,datasets}/ID/；旧 run 内产物只读保留。

  reports/ / artifacts.json

```



`catalog/manifest.json` 是 benchmark/task 清单（schema 2），不再维护 rollout、评测或 baseline profile。运行事实保存在 `runs/RUN_ID/evidence.json`；只迁移 `migration-closeonly331-20261021` 的 331 条轨迹和相关评测，其余旧登记不参与运行时读取。历史隔离文件仍作为归档保留。



工程外原件/迁移报告：`/home/wangtianyi/dycon-isolation-20261021/baseline-cleanup/`。旧 receipt、权重和 NAS tensor 不改写；转换产生新 ID/receipt，禁止旧路径 alias/fallback。当前 Ridge 配置使用转换后的 `submission-shortest375-20261019-relocated-20261021`。训练输入是显式共享 dataset，训练输出在 `difficulty/checkpoints/ATTEMPT_ID/`；已发布 predictor 仍在 `runtime/parallel-artifacts/PREDICTOR_ID/`。



## 3. 全部业务命令



以下为子命令语法；省略解释器前缀。对 run 下的分析产物，使用 analysis/runner 对应环境的 `dycon --run-id RUN_ID ...`。`enrich/token-trace/hidden/dataset-index/token-join` 在首个 dataset 参数是新版 run ID 时也会自动绑定 run。



| 功能 | 环境 | 命令 | 实现/主要输出 |

|---|---|---|---|

| 注册 checkpoint | runner | `model-register CHECKPOINT` | `core.model_identity`，内容 fingerprint |

| 任务导入 | runner | `tasks-import TASKS.json` | `catalog.store.Catalog.register_tasks` |

| 拉取 benchmark | runner | `benchmark-pull COHORT_ID [--count 600] [--workers N]` | `catalog.benchmark`，固定 dataset/results revision、NAS OCI 与 task 登记 |

| 镜像核验 | runner | `images-check [--deep]` | 当前根 benchmark 范围的 NAS 描述/blob；不执行 load/harness |

| 轨迹导入 | runner | `rollouts-import ROLLOUTS.json` | `rollout.artifacts`，字节不变的轨迹与 run-local patch 登记 |

| 外部评分导入 | runner | `evaluations-import EVALUATIONS.json` | `benchmark_evaluation.import_evidence`，实际被评分 patch 的 SHA 必须一致 |


| Harness 选择 | runner | `harness prepare RUN_ID [--source-run SOURCE_RUN] [--rollout-id ID ...]` | 默认 producer run 轨迹；重评历史 patch 时显式传 ID，必须属于 run tasks |

| 官方评分 | runner | `harness run RUN_ID EVAL_ID [--gold]` | `benchmark_evaluation.harness`，SWE-bench 4.1.0；gold 只验环境 |

| Capture 发布恢复 | runner | `capture recover RUN_ID` | `hidden.recover_online`，不重新请求模型；旧命令 `capture-recover` 仍存在 |

| Capture 重分类 | runner | `capture-reclassify RUN_ID --session-id SID ...` | 已 finalized 证据的有界规则修正，生成新 revision，不补造 hidden |

| 在线索引 | analysis | `online-index SOURCE_RUN INDEX_ID [--rollout-id ID ... | --rollout-list FILE]` | `difficulty.datasets.online`，schema 3/NAS 引用 |

| 离线标签/prefix | analysis | `enrich RUN_ID` | `enrichment.trajectory`，冻结轨迹、token evidence、labels |

| Token trace/audit | analysis | `token-trace RUN_ID`、`trajectory-token-audit RUN_ID` | exact/reconstructed 证据、token/step/phase/position |

| 离线 hidden | analysis | `hidden RUN_ID CAPTURE_ID` | `hidden.scripts.extract_vllm` → latent-vllm eager |

| 离线索引 | analysis | `dataset-index RUN_ID CAPTURE_ID INDEX_ID` | schema 1，sample ID join，复用 labels/hidden |

| 精确 token join | analysis | `token-join RUN_ID CAPTURE_ID JOIN_ID` | schema 2 `rows.jsonl`；`hidden-token-join-audit JOIN_ID` 审计 |

| Action 分类与默认热力图 | analysis | `action-classify ACTION_ID --source-run RUN [--source-run OTHER] [--rollout-id ID ... 或 --rollout-list FILE]` | 项目级独立单元；多 run 清单必须逐条绑定 source_run_id |

| Action 校验 | analysis | `action-check ACTION_ID` | receipt/来源内容/分类输入核验 |

| Action 绘图视图 | analysis | `action-plot ACTION_ID [--plot-id NEW_ID] [--aggregation micro/macro] [--format png/svg]` | 默认复用分类时的图；新 ID 创建绑定 parent receipt 的不可变 view |

| Action 统一产物校验 | analysis | `artifact verify ACTION_ID --scope action-analysis` | 支持 show；冻结 source evidence、代码、标签、统计及图表校验 |

| 共享 hidden 视图 | analysis | `action-index INDEX_ID ACTION_INDEX_ID` | 引用共享 index，不重复提取 hidden |

| Ridge 尝试 | analysis | `difficulty-ridge ATTEMPT_ID [--dataset-id ID] [--split-id ID] --mode dry-run/check-data/fit` | 新目录拒绝覆盖；默认 fit，见下文 |
| 统一 difficulty 训练 | analysis | `difficulty-train [--training-config YAML]` / `source difficulty/scripts/train.sh [YAML]` | 独立严格训练配置统一选择 Ridge/MLP/Ridge+MLP，MLP 可配置隐藏层和优化器；fit 自动托管 TensorBoard |
| TensorBoard 管理 | analysis | `difficulty-tensorboard ATTEMPT_ID --action status/stop` | 只管理本项目登记的服务 |
| MLP 尝试 | analysis | `difficulty-mlp ATTEMPT_ID --dataset-id ID [--split-id SPLIT_ID] --mode dry-run/check-data/fit` | 参数来自 `difficulty/training/config.yaml`；默认 event split、scheduler 和三分区 epoch 指标 |
| Ridge+MLP 尝试 | analysis | `difficulty-ridge-mlp ATTEMPT_ID --dataset-id ID [--split-id SPLIT_ID] --mode dry-run/check-data/fit` | 在每个目标的已发布 train-only Ridge 上训练零初始化 FFN 残差；发布 manifest、checkpoint、split 和 SHA 快照到 attempt |
| 成功轨迹训练集 | analysis | `difficulty-dataset resolved DATASET_ID --source-run RUN_ID --index-id INDEX_REF` | 冻结精确评测和 canonical 样本关系 |
| 跨批次组合训练集 | analysis | `difficulty-dataset compose DATASET_ID --source-dataset A --source-dataset B` | 绑定父 receipt；拒绝重复样本/rollout、冲突 commit/hidden 模型 |
| 训练数据划分 | analysis | `difficulty-split DATASET_ID SPLIT_ID --strategy event_random_v1/task_random_v1` | seed/fractions 等参数见 --help；独立 immutable receipt |

| 发布 predictor | runner | `predictor-publish ID --experiment PATH --target reasoning_length` | 现有训练实验 validation-best；默认 refit；`--fit-scope train` 可选 |

| 显式 checkpoint | runner | `predictor-publish ID --checkpoint PATH [--loader module:callable] [--sites ...] [--hidden-size N]` | 可换 estimator provider；不在发布时训练 |

| Predictor 推理 | runner | `difficulty-predict INDEX_ID PREDICTOR_MANIFEST PREDICTION_ID` | 当前支持 schema 1、submission 目标、close hidden；保存真实 predictions/native scores |

| 预测评估 | analysis | `difficulty-evaluate INDEX_ID PREDICTIONS.json SCORE_ID` | schema 1 labels；metrics、轨迹图，保留缺失成功标签计数 |

| 分支可视化 | analysis | `difficulty-plot-parallel SOURCE_RUN_ID PLOT_ID --rollout-id ID` | `--list` 只需 SOURCE_RUN_ID；task 多次尝试须指定 attempt；支持 step 范围 |

| 分支一致性 | analysis | `difficulty-evaluate-parallel SOURCE_RUN_ID ID [--rollout-id ID ...]` | 或重复 `--task-id`，两类筛选不能混用；统计/CI/receipt |

| Owned 资源恢复 | runner | `recover` | `runtime.containers.recovery`，受 ownership/锁约束 |



源码中仍有 `prepare-evaluation` / `evaluate`、无 run 的 dataset 入口和 `runtime-check`。`runtime-check` 要求全局 wheelhouse，不能替代真实 rollout gate。新实验使用上表 run 路线；这些旧入口不是旧快照自动转换工具。



## 4. 显式输入格式



`run create --tasks-file` 接受 instance ID 的 JSON list，或含 `instance_id` 的对象列表/`{"tasks": [...]}`。使用已注册任务；不要把完整 task 定义列表与“本次选哪些 ID”混为一谈。



`tasks-import` 接受完整 task 的 JSON list。下面仅展示必要身份和 NAS 引用，`instance` 还须包含实际 SWE-bench 4.1.0 harness 所需字段：



```json

[

  {

    "benchmark": "SWE-bench_Verified", "revision": "PINNED_REVISION",

    "instance_id": "org__repo-123", "repo": "org/repo", "base_commit": "BASE_COMMIT",

    "partition": "train", "exclusion_reason": null,

    "instance": {

      "instance_id": "org__repo-123", "repo": "org/repo", "base_commit": "BASE_COMMIT",

      "problem_statement": "...",

      "nas_image": {"store": "images/COHORT", "manifest": "images/TASK.json",

                    "manifest_sha256": "SHA256", "image_id": "sha256:DIGEST"}

    }

  }

]

```



NAS `store` 相对 `paths.nas_root`。manifest 绑定 OCI config/layers descriptor、size/digest，blob 位于 `<store>/blobs/sha256/<digest>`。复用旧项目镜像须通过静态资产发布及 hash 核验。



轨迹导入列表每项包含 `source`、完整 `task`、`run_id`、`attempt_id`、`purposes`（training/baseline/action_analysis）、`model_fingerprint`，可加 `expected_sha256`。提交状态不推出 resolved；导入后未评分为 null。评分导入列表每项如下：



```json

{

  "evaluation_id": "IMPORTED_EVAL", "run_id": "SOURCE_RUN",

  "report": "/absolute/source/report.json", "report_sha256": "SHA256",

  "bindings": {

    "ROLLOUT_ID": {"evaluated_patch": "/absolute/actually-graded.diff",

                   "outcome_path": ["outcomes", "INSTANCE_ID"]}

  }

}

```



`outcome_path` 定位结果对象。completed 的结果必须有 boolean resolved，实际评分 patch 必须与指定 run 的 patch 内容一致。



预测文件为 JSON list：`[{"sample_id":"ROLLOUT:STEP:ANCHOR:OCCURRENCE","target":"remaining_steps","prediction":12.3}]`。target 可用 `remaining_steps` / `remaining_reasoning_tokens`；GT 从冻结 enrichment 读取，不接受文件自带 GT，不用 GT 冒充 predictor 结果。



## 5. Rollout、在线 capture 与 parallel 的关键契约



parallel 单次生成循环保护由 `parallel.reasoning_loop` 配置：当前 `include_post_close=true`，覆盖本次生成的 reasoning 与 `</think>` 后回答/工具参数，保持 native output token 绝对坐标；旧配置缺少该字段时仍仅检查未 close 的 reasoning。至少生成 2048 token，每 64 token 检查 1–8192 token 的精确连续周期，至少重复 8 次且覆盖至少 1024 token，不检查 prompt。服务端调用原生 abort 停止命中的单个 native 请求；结束已确认的请求不再 abort。命中保留真实 partial ledger、phase（reasoning/post_close）和 abort_completed 证据；post-close 分支可能已有 close hidden/score，但无论已有分数均从投票与 fallback 排除，部分工具参数绝不执行。其他分支继续；全部因循环中止仍沿用现有 LimitsExceeded/limit_reason。扩展规则为 exact_token_period_v2，旧 reasoning-only 规则保持 v1。参数随新 run 冻结，旧快照不回填。



开启上述保护且 `parallel.validation_repetitions>0` 时，正式 check-only 启动验证另含受控重复 token 检查，沿用已有 allowed-token 诊断机制和真实非流式请求/捕获链；要求所有受控分支在 token 上限前以原生 abort 结束且不执行工具。报告位于 `startup/parallel-validation/report.json` 的 `reasoning_loop_abort`。该诊断没有被接受的 assistant 或 close tensor，session 明确保留为 partial；正常候选验证仍要求完整捕获及真实 FULL graph 证据。



跨 step 循环保护由根 `config.yaml` 的 `agent.loop_repeat_limit`（当前 8，`null` 关闭）和 `agent.loop_max_period`（当前 7，范围 1–7）控制。仅当 reasoning、普通文本 content 和按顺序排列的 action 执行参数均精确相同时，才计为相同 step；忽略 tool-call/request ID、时间戳等传输元数据，不忽略文本空白差异。工具结果不参与比较，同一输出内多个工具调用仍只算一个 step。



每轮检查轨迹末尾连续的完整周期，包含首次出现共 8 次即终止：1-step 周期在第 8 条输出命中，7-step 周期在第 56 条输出命中。命中输出保留，其 action 不执行，并记录原生“action was not executed”工具结果；轨迹退出状态及 run evidence termination_reason 为 `LimitsExceeded`，`info.loop_detection` 和 exit 消息记录周期、次数、0-based 有效 assistant 输出范围及 `action_executed=false`。格式错误打断连续性；parallel 只计选中的 winner。既有 capture 收尾、故障优先级和 terminal resume 行为保持不变。



参数随新 run 冻结。旧快照缺少字段时关闭检测，原快照及 hash 不回填；要启用保护，应创建新 run。此规则只处理跨 step 循环，单次生成请求内部的重复仍受原 token/time limit 约束。



| 模式 | 开关 |

|---|---|

| 普通/baseline | `parallel.enabled=false`、`steering.enabled=false` |

| Steering | 启用 steering 并提供匹配 fingerprint 的 payload；禁用时 payload=null |

| Parallel | `parallel.enabled=true`、B>1、thinking、managed capture、think_close、匹配 sites 的 estimator_ref |

| 在线采集 | capture 独立开关，layers/anchors、storage budget 和 failure_policy 明确配置 |



parallel 使用并发完整 reasoning+action 请求，等待所有候选完成后比较 close hidden 的 predictor score；只执行 winner action。B=1 不代表四分支验收；fallback、终止和缺 close 的候选分别记录。predictor provider 返回有限 `[N]` 分数，输入 `[N, sites, hidden_size]`；manifest 绑定 checkpoint/model fingerprint、sites、target、endpoint、文件和实现 SHA。实现或权重变更须重新发布，不能改已有 predictor manifest。

parallel_consensus 在同一并发请求链上固定生成 32 个分支，使用 run snapshot 绑定的 estimator 保留分数最低的 50%（retention_ratio 可配置），再按 exact_action 计票；票数最多者胜出，票数并列时按支持分支的最低/平均 estimator 分数和 branch ID 稳定裁决。只有获胜代表分支的 action 会进入 agent 执行；decision evidence 同时记录 retained 分支、action hash、票组和 tie-break。启用 32 分支时 serving max_num_seqs 和 parallel.max_inflight_branches 必须至少为 32。



分支身份为 `decision_id → branch_id → native_request_id → sample_id`。branch index 的 `selected` 保存实际赢家，不能重选最小值冒充历史 winner，未选分支不能复制 winner 的 remaining GT。网络响应顺序不作为身份依据。



Qwen 在线支持 generated reasoning newline 和 close；prompt 中 `<think>` 的 FULL prefill 有额外限制。close token 需要作为后续 forward 的输入才能获得 hidden；未生成 close、长度截断前无 close、生成 close 但缺后续 forward/hidden 必须区分。`safe_boundary` 在 capture target 前限制缓存复用。



managed serving 实际请求 `FULL_AND_PIECEWISE`；`capture.graph_mode=FULL` 不保证每个 prefill/mixed batch 都走 FULL。`capture.prefill_token_chunk_size=0` 交给 vLLM 原生调度；不得用单 token prompt chunk 强行模拟 decode（Qwen GDN 已出现 NaN）。graph 执行以 run 证据为准。



native request 的完整 prompt/output token IDs 保存在 SHA 绑定 ledger 中，轨迹和事件引用 ledger。bounded queue/spool 异步编码、封装 tar shard、NAS `.part` 校验后原子发布，ACK 前保留本地数据。当前 spool 仍按冻结 `capture.export.spool_root` 配置，默认 `hidden/spool/RUN_ID/`，不能因最终产物 run-local 就提前删除 spool。NAS tensor 通常为 `<nas_root>/hidden/online/RUN_ID/`。pending_upload 用 capture recover 发布，不能为补发布重跑 rollout；partial token coverage 不会被上传重试自动补齐。



## 6. Enrichment、离线重放与标签



一次 assistant 回复及其动作阶段算一个 step，s 从 0 起，共 T 步时 open/close remaining steps 为 T−s，当前 action 尚未发生。open reasoning 标签包含当前轮及后续轮，close 只含后续轮，newline 标签从当前 token 之后计数。成功剩余量标签要求 resolved=true；未评测、unresolved、infra_error 保留 null。submission 端点 predictor 与 success 端点 GT 的解释需分别注明，不能把低分理解为保证解题成功。



offline replay 输入是 exact prefix 或明确声明 reconstructed 的 prefix；证据长度相同不代表 exact。`hidden` 用独立 eager runtime，generated trigger token 不作 anchor，不截断 prefix。vLLM 管 continuous batching、chunked prefill、APC、KV/GDN 状态恢复；项目只做同 rollout/session/branch anchor 的依赖顺序，跨窗口保持同一引擎/APC。同一轨迹每 wave 最多一个 anchor，独立轨迹可并发；APC 命中不保证全部前缀都避免重算。



N 层模型的公共 site `0..N−2` 是 decoder output，`N−1` 是 final norm；native final norm 是 N，不能直接混用。提取检查 request/position/token/site/shape/finite，读取 hash 校验的 partial sample 可用于相同身份的续提；修改 runtime/config/code 要新 capture ID。



| 数据视图 | 入口 | 能否直接给当前 scorer/predictor |

|---|---|---|

| schema 1 离线 index | dataset-index | difficulty-evaluate 与 difficulty-predict 支持；label_shard 与 hidden 引用 |

| schema 2 token join | token-join | exact 对齐审计 rows.jsonl，不能冒充普通 index |

| schema 3 在线 index | online-index | NAS shard/partition/labels；不能直接传给当前 schema 1 scorer/predictor |



评分后新建在线 index 才反映新标签，已有 index 不自动刷新。后处理冻结 `analysis-selection.json`；若先冻结未评分的标签，后续改变评测结果不会自动更新原产物，需要新的分析 run/明确新输入。



## 7. Ridge、可视化和一致性指标



Ridge settings 在 `difficulty.ridge_probe`，包含 dataset/receipt、split_id、target、layers/alphas、split_strategy、seed、selection_metric。`--dataset-id` 显式选择共享训练集并冻结实际 receipt 与计数；`--split-id` 指定已创建的 dataset-owned split。`dry-run` 核验来源并保存 attempt，`check-data` 另做 feature 检查，`fit` 才拟合；模式写入 receipt，输出 ID 拒绝覆盖。不要对当前全量数据随意执行 fit。

`difficulty.ridge_probe.targets` 支持 `submission_remaining_steps_log1p`、`submission_remaining_steps_linear`、`submission_remaining_reasoning_tokens_log1p`、`submission_remaining_reasoning_tokens_linear`。默认仍为 tokens（log1p）和 steps（linear）；显式指定可比较另外两种变换。四个目标复用既有原始 submission 标签，log1p 使用 `expm1` 还原，linear 保持原单位。图与选型使用拟合空间，跨变换比较查看 `test_raw_units`；本扩展仅覆盖 Ridge 训练/报告，现有 predictor 发布映射保持原有目标。



划分有 task quota/inherit、event_random_v1、task_random_v1。event 随机划分不是 task holdout；选择使用 validation R²，不看 test 选层；tie 规则和 refit scope 以 experiment/selected/metrics 为准。标准化只在训练子集拟合；reasoning 可用 log1p，原单位由 expm1 还原。负值与 undefined 指标明确保留。已有 Ridge/MLP/Ridge+FFN Python API；通用 MLP/残差无完整统一训练 CLI，不声称完成 GPU 训练工作流。



分支图读取 finalized request→canonical action step 绑定，显示实际 selected score、所有候选散点、缺失与 fallback，输出 PNG/SVG、steps/branches CSV、JSON 与 receipt。task 有多个 attempt 时须显式指定，支持 `--step-start/--step-end`（含两端）。FormatError query 计数不等于实际 action step。



一致性默认 text reasoning 身份，可选 `--reasoning-identity tokens`，缺 exact IDs 不悄悄重建。分组使用同 parent prefix/step/action、不同 reasoning，区分 reasoning 去重视图与采样视图。主要指标为平均绝对 pair gap D，辅助 std/range；log1p 分数下 `|Δs|≥log K` 对应模型隐含 `1+剩余量` 的 K 倍差，非真实剩余量比。默认 K=1.25/1.5/2/3，rate 是 [0,1]。



相对指标 Q 是同 action reasoning 差与不同 action reasoning 差的 paired、task-balanced 均值之比，不是各 step ratio 的平均；分母近零为 null。汇总按 action 组→step→rollout→task 等权；task/repo block bootstrap 保存簇内相关性，单簇 CI 为 null。`--bootstrap-samples 0` 关闭 CI。指标衡量条件敏感性，不能证明准确性、正确分支选择或成功率提升。



## 8. 环境、维护与当前边界



安装用 `runtime/scripts/setup_runner.sh`、`setup_analysis.sh`、`setup_easysteer.sh` 和 `locks/`。源码 package 迁移后需更新 editable 安装，否则已安装 dycon 可能找不到新包；重装项目用 `uv pip install --python .venvs/ENV/bin/python --no-deps --no-build-isolation -e .`，不无条件重装全部 runtime 依赖。



| 维护入口 | 用途与副作用 |

|---|---|

| `runtime/scripts/verify_easysteer_runtime.py` | easysteer 环境核验，读取 pinned runtime/source 身份 |

| `hidden.scripts.prepare_vllm_runtime` | runner 运行；复制独立 replay 环境，已有目标拒绝覆盖 |

| `hidden.evaluation.replay_online_artifacts` | latent-vllm 运行，读取 `--source-run`/`--online-dir`/`--event-index`，输出指定新目录；真实 GPU 工作 |

| `hidden.evaluation.compare_vllm_replay` | analysis 运行，已有 online/replay 目录数值比较 |

| `hidden.evaluation.capture_matrix`、`smoke_vllm`、`diagnose_native_dispatch` | 实际 GPU smoke/graph 诊断，先核对 --help 和作用范围 |

| `runtime.scripts.smoke_images` | 合成 OCI load/create/remove；不能代替 SWE-bench gold |

| `runtime.scripts.gold_image_audit` | runner，显式 instances/audit-id，走统一 ContainerFactory 和 harness |

| `runtime.scripts.repair_image_ownership` | rootless UID/GID 的派生 OCI 修复，保留原始 lineage |

| `runtime.scripts.wakeup_nonverified600_gold` | 历史 cohort pidfd 唤醒/有界重测；不是通用任意 cohort 配置器 |

| `enrichment.scripts.launch_extraction` | 仅原 anchors-20261017 批次/GPU 0–3 的历史入口，不直接起内部 worker |



`runtime.scripts.supervise_rollout`、`launch_selection` 是旧队列工具，新 run 用 `run start/wait/resume`。日期化 repair/finalizer/批量工具不能因名称看似通用就运行。容器环境修复操作及判定标准全部见 [修复指南](container-environment-repair-cn.md)。



已实现 action canonical 分类和共享 hidden 视图；action taxonomy 学习/训练/独立评测未实现。部分状态、cancel、reconcile、报告和全量历史转换仍有限制，以上逐项说明。命令存在不代表全部方案已完整实现；单测、两题 GPU smoke、gold 环境与全量性能实验是不同证据。



## 9. 文档历史与追溯



本次整理前的全部 `docs/` 原文及 SHA 清单保存到工程外 `/home/wangtianyi/dycon-isolation-20261021/docs-consolidation/`。其中 online capture、APC、parallel、Ridge、指标定义保留实现和科研追溯价值；旧 rollout 指南/重复入口页不再作为当前使用说明。验收、600-task repair、baseline 迁移等批次报告可供复现，在归档和各 run/report 中保留，不在日常三份指南中堆叠。



模块 README、测试和代码中的历史路径/源码 hash 可是实验身份的一部分，不因文档整理而改写冻结证据。完整历史归档不由 runtime 用作 fallback。





## 10. 已登记在线采集产物的一次性迁入



先用 `run create TARGET --operation harness --tasks-file TASKS.json` 创建独立目标 run。

`migration.import_capture` 只接收显式 `{"rollouts":[{"rollout_id":"...","trajectory_sha256":"...","patch_sha256":"..."}]}` 清单；

校验 canonical 与已发布 session，将轨迹/patch 放入目标 run，并登记到目标 run 的 evidence，并保持来源 run 证据不变。

rollout ID、attempt 和 payload SHA 不变，原 run、NAS 张量及源 finalization 不改写。

它会保存来源快照、identity mapping、migration receipt 和 NAS publication 引用。

这不重建历史执行状态，目标不得作为旧 rollout 的恢复入口。



```bash

.venvs/runner/bin/python -m migration.import_capture \

  --source-run SOURCE --target-run TARGET --selection RETAINED.json

.venvs/analysis/bin/python -m migration.finalize_capture --run TARGET --workers 4

.venvs/analysis/bin/python -m migration.verify_capture --run TARGET --workers 4

```



finalize 按选定 session 有界读取 canonical 绑定的原生 ledger，复用共享 token finalizer，

记录 exact/censored/coverage；verify 检查所选 shard/member SHA、native sites、shape、dtype 和有限值。

所有运行参数仍来自目标 run 冻结配置；历史生成配置单独保存在 migration/source-run。

评分继续使用正式 `harness prepare/run`，对实际已提交 patch 显式选择 rollout。

评分和必要的基础设施复评结束后才执行 `dycon online-index TARGET INDEX_ID`，

以当前精确 patch 评分生成成功剩余量标签；unresolved、未评测和 infra_error 的成功标签保持 null。

迁入 index 以 publication/receipt 中的原 NAS root 为唯一 tensor 来源，不使用旧路径 fallback。



## 11. 仅成功轨迹的 Ridge 输入

`difficulty.datasets.resolved` 从已评分的 schema 3 在线索引建立独立共享输入，
只接纳指定 producer run 中 completed/resolved=true 且绑定精确 patch SHA 的轨迹。
排除 censored、缺标签和非 action response；保留原生 sample ID、NAS locator 和公共 site 映射。
原索引和来源 run 证据不改写。评分发生变化时需新建 online index/dataset ID。

```bash
.venvs/analysis/bin/dycon difficulty-dataset resolved RESOLVED_DATASET \
  --source-run SOURCE_RUN --index-id GRADED_INDEX
```

输出 `difficulty/datasets/RESOLVED_DATASET/{rollouts.json,samples.jsonl,excluded.jsonl,source-evidence.json,summary.json,receipt.json}`。
receipt 标记 `eligibility=resolved_only`，Ridge loader 在训练前重新核验当前 resolved 和 selected evaluation 身份。
对成功轨迹，“到最终提交”与“到该次成功提交”的剩余量一致，因此沿用
`submission_remaining_reasoning_tokens_log1p`、`submission_remaining_steps_linear` 目标字段；
模型不预测成功概率，不能据此为失败轨迹补成功 GT。

通过根配置或允许的 `--set difficulty.ridge_probe...` 指定 targets、layers 和 alphas，
使用 `difficulty-split RESOLVED_DATASET SPLIT_ID` 冻结划分，随后
`dycon difficulty-ridge ATTEMPT_ID --dataset-id RESOLVED_DATASET --split-id SPLIT_ID --mode dry-run/fit`。
不需要创建 analysis run；每种 mode 使用不同 attempt ID。
建议 task_random_v1 将同题全部样本放入同一集合，验证集选层/alpha，
冠军在 train+validation 重拟合后只评估一次 test；测试指标仅代表该成功轨迹子集。


## 12. Difficulty 独立产物与身份

来源 run 不决定 difficulty 产物位置；`--run-id` 可继续提供旧配置快照，但新训练不必创建 analysis run。
数据集位于 `difficulty/datasets/DATASET_ID/`；每套划分在其 `splits/SPLIT_ID/{split.json,receipt.json}`。
dataset receipt 只包含数据自身文件，新增 split 不改写它。训练只消费冻结分配，不能用 Catalog task partition 冒充训练划分。
缺省未指定 split_id 时，为旧 CLI 调用按固定策略/seed/来源 receipt 推导 auto-ID，并通过相同 split service 创建或复用；
显式指定的 split_id 必须先创建。effective settings 记录实际加载的策略和 seed。

`difficulty/checkpoints/ATTEMPT_ID/` 保存 `attempt.json`、兼容 `experiment.json`、trials/selected/metrics/predictions/plots、模型及 receipt。
模型 NPZ 保存 scaler、weights、architecture、target transform、hidden fingerprint、site/shape、dataset/split receipt、fit scope 和训练样本 hash；同名 JSON 便于检查。
推理模块 `difficulty.models.ridge_grid` 保持原始实现，新增 metadata 写入由训练模块负责，不破坏既有 predictor implementation SHA。
新 cache 位于 `difficulty/cache/HASH/`；旧 `state/difficulty-cache/HASH/` 存在时只在通过原 cache manifest/SHA 校验后复用，不搬运或改写。
预测与独立评测分别在 `difficulty/predictions/ID/`、`difficulty/evaluations/ID/`。

统一校验命令：

```bash
dycon artifact verify DATASET_ID --scope difficulty-dataset
dycon artifact verify SPLIT_ID --scope difficulty-split --dataset-id DATASET_ID
dycon artifact verify ATTEMPT_ID --scope difficulty-attempt
dycon artifact verify PREDICTION_ID --scope difficulty-prediction
dycon artifact verify EVALUATION_ID --scope difficulty-evaluation
```

show 使用相同 scope。run 的 `artifact verify RUN_ID` 继续按原语义工作。
旧 index 按显式目录读取，例如 `difficulty-evaluate runs/OLD_RUN/difficulty/datasets/INDEX_ID ...`；
不通过当前 run 隐式搜索旧目录。resolved loader 同样接受该显式 `--index-id` 目录。
旧 checkpoint 使用原 experiment 路径显式发布，旧 snapshot/receipt/模型不自动迁移或重签。
MLP 正式训练使用 `difficulty-mlp`，Ridge+MLP 使用 `difficulty-ridge-mlp`，两者复用 dataset/split/checkpoint 结构与 verifier；Ridge+MLP 的 Ridge 来源通过 `difficulty/training/config.yaml` 的 `ridge_mlp_probe.ridge_publications` 绑定发布 manifest，入口不会重新拟合 Ridge。当前结果是离线 attempt 产物，不自动发布为在线 predictor。

## Catalog 与运行证据边界（2026-10-21）

`catalog-check` 只检查 benchmark/task 定义。`artifact verify RUN_ID` 检查运行内的轨迹、补丁、评测报告、receipt 和外部引用。新增 rollout 与评测不再改写 Catalog。`baseline-select`、`benchmark.purposes/profile`、跨 Catalog 的 `--unrolled` 查询已删除。

创建数据集先指定来源运行，再限定其轨迹；重复 ID、其他 run 的 ID、空 JSON 清单和给定 SHA 不一致均拒绝。多来源数据先分别生成固定数据集，再 `difficulty-dataset compose`，训练读取冻结数据集而不读取实时 Catalog。

```bash
.venvs/analysis/bin/dycon online-index migration-closeonly331-20261021 INDEX_ID \
  --rollout-list ROLLOUTS.json
.venvs/analysis/bin/dycon difficulty-dataset resolved DATASET_ID \
  --source-run migration-closeonly331-20261021 --index-id INDEX_ID \
  --rollout-list ROLLOUTS.json
.venvs/analysis/bin/dycon action-classify ANALYSIS_ID \
  --source-run migration-closeonly331-20261021 --rollout-list ROLLOUTS.json
```

`ROLLOUTS.json` 是该来源 run 内的 JSON ID 列表，例如 `["r-..."]`；省略过滤参数选取该 run 的全部合格轨迹。`evaluations-import` 每项必须含 `run_id`，报告复制到该 run 的 `harness/imports/EVAL_ID/`。独立重评通过目标 Harness run 的 `--source-run` 引用原轨迹，其结果只保存在目标 run，不回写来源 run。

一次性转换入口：`python -m migration.catalog_to_runs --keep-run migration-closeonly331-20261021 --apply`。本次保留 1058 个 benchmark tasks，转入该 run 的 331 条轨迹和 3 项相关评测记录，删除其余 1775 条全局 rollout 登记。原轨迹、NAS 大张量、历史冻结快照不改写。

离线 join 同样要求来源 run：`dataset-index DATASET_ID CAPTURE_ID INDEX_ID --source-run SOURCE_RUN [--rollout-list FILE]` 与 `token-join DATASET_ID CAPTURE_ID INDEX_ID --source-run SOURCE_RUN [--rollout-list FILE]`。使用 `--run-id` 绑定该来源 run 时可省略重复的 `--source-run`。

## GPU MIG serving

Managed rollout serving accepts either physical GPU indices or MIG instance UUIDs in serving.gpus. The number of entries must equal serving.tensor_parallel_size; one MIG instance therefore uses TP=1. The UUID must be discovered at launch time with nvidia-smi -L and frozen in the run resolved configuration. Example:

    serving:
      gpus: ["MIG-INSTANCE-UUID"]
      tensor_parallel_size: 1

Run creation and execution still use the normal dycon commands. The serving manager preserves the UUID in CUDA_VISIBLE_DEVICES, checks that the instance exists and is visible as one CUDA device, and records the device and measured memory in startup/launch.json. It takes a run-local device lock so two managed runs cannot claim the same configured device. Physical GPU configurations such as gpus: [0, 1, 2, 3] with TP=4 remain unchanged. Offline hidden replay keeps its separate integer-GPU contract in this release.

### Harness failure outcomes

`harness run` records a stable outcome for every selected rollout:

- An empty or unappliable patch is `status=completed`, `resolved=false`, `failure_type=invalid_patch`.
- A transient network/connection failure is retried by the unified harness at most two times. Each attempt has its own evaluation task ID and is retained in the result; after exhaustion the outcome is `status=ConnectionError`, `resolved=null`, `failure_type=network_error`.
- Other execution failures remain `status=infra_error`, `resolved=null`, `failure_type=other_error` (or the more specific non-network classification).

The summary reports `network_retries`; retrying infrastructure does not create a new rollout, repetition, or canonical trajectory.


### Trajectory normalization

Use `dycon difficulty-dataset resolved NEW_DATASET --source-run RUN --index-id INDEX --label-normalization trajectory_fraction`. Train fraction targets through the existing `difficulty-ridge`, `difficulty-mlp`, and `difficulty-ridge-mlp` commands using `submission_remaining_steps_fraction` and `submission_remaining_reasoning_tokens_fraction`. Raw labels and existing checkpoints remain unchanged.

物理 GPU 的 managed serving 允许多个 run 共用同一设备，不获取设备独占锁；每个 run 通过独立 serving 端口和各自显存配置启动，GPU 可见性及 vLLM 显存检查仍生效。MIG instance 保留独占锁及已有进程检查；同一 run 的运行锁和 supervisor 锁保持生效。


### Unified difficulty estimator training

Use `source difficulty/scripts/train.sh [YAML]` or `dycon difficulty-train --training-config YAML`. The standalone schema-2 config selects a dataset and frozen/generated split, regression targets, Ridge/MLP/Ridge+MLP, input site, MLP layer dimensions and optimizer. Ridge is closed-form and has no optimizer. Ridge+MLP consumes verified frozen publications. TensorBoard records MLP evaluation-mode train/validation loss by epoch and Ridge grid loss by candidate. Remote default is loopback; tunnel from the local PC with `ssh -N -L 16006:127.0.0.1:6006 wangtianyi@10.249.187.9`, then browse `http://localhost:16006`. Check/stop via `dycon difficulty-tensorboard ATTEMPT_ID --action status|stop`. Events close before receipt finalization.


## Docker ???????rootless?

??????? `config.yaml` ? `environment.docker_data_root`??? `/data1/wangtianyi/docker`??????NAS namespace?Docker socket ? Docker data-root ??????????????

```bash
.venvs/runner/bin/dycon docker-storage status
.venvs/runner/bin/dycon docker-storage apply
```

`status` ????????? daemon ?????`apply` ??????????? YAML?????????? `${XDG_CONFIG_HOME:-~/.config}/docker/daemon.json`????? daemon ???????????????? `docker.service`???????????????/???????????????????????????????????????????Docker ???? run ? image/container lease ????????????? daemon ????????????????? `state/docker-storage/ID/`?

?? YAML ?????? `apply`??? rollout/harness ?????? Docker?????????????? DockerRootDir???????????????????????????? YAML ???????????????????????????????????????????????????? run ???

NAS ?????? OCI ????? data-root ????? lease ??? cold load ???????????? `docker info` ? DockerRootDir??? containerd ??? root ????????????? run ?????????????????????????????????????????????? capture spool ???????


## Native receipt 契约与 bootstrap-only 环境预检（2026-10-26）

parallel 的 bind/output 使用 admission hook 观察到的完整 native request ID；外部 completion ID 仅作旁证。controller 按 `(session_id, decision_id, branch_id, api_attempt_id)` 建立索引，并核对 step、prompt hash 和 scorer 身份。相同 output 内容重试幂等确认；身份或内容冲突使该 decision 终止。插件在最终 stream yield 之前提交 receipt，避免消费方在最终项后关闭流时遗漏 ledger。control bind/output 仅对连接故障作一次幂等重试；失败 decision 的后续 payload 保存在 failures artifact，不继续提交该 decision 的 output。

`failed-DECISION.json` 保留原始 responses 及 `receipt_diagnostics`，分别报告 `no_receipt`、`receipt_delayed`、`duplicate_receipt`、身份冲突、deadline 或 control 故障。`decision_deadline_unix` 从 coordinator 传到 scoring、bind、output 和 decision freeze；原始 receipts 保留至 session finalization。capture HTTP 服务 backlog 为 128，适应 32 分支并发回调。

只验环境时扩展既有维护入口，不启动模型或 gold tests：

```bash
.venvs/runner/bin/python -m runtime.scripts.gold_image_audit \
  --root /home/wangtianyi/DyConGlobal \
  --instances FROZEN_INSTANCES.json --only-file SELECTED_IDS.json \
  --audit-id UNIQUE_ENVIRONMENT_AUDIT --workers 2 \
  --config-snapshot VERIFIED_AUDIT_CONFIG.json \
  --bootstrap-only --repeat-count 2
```

输入仍须为已冻结的真实 instance dict 列表。每次使用共享 ContainerFactory 新建/启动/清理容器；输出独立 attempt 的 environment-policy、bootstrap.log、environment.json、bootstrap-receipt.json。汇总比较两次 policy/packages hash，报告 environment drift；`mode=bootstrap_only` 且 `resolved=null`，不得将环境可用性当成 gold 或模型解题成功。保留 `--cold-start` 原有拒绝缓存语义，不清理其他 run 的镜像。

EnvironmentPolicy 会规范化 package 名及重复 requirement，并校验冲突 exact pins；repo pins 与全局 constraints 都要求非空离线 wheelhouse。requirements、constraints 和 wheel hashes 进入 policy/receipt。缺少 wheel 的失败在镜像 lease/load 之前写 `environment-preflight.json`，不能用删除 pins、升级 pytest 或安装网络 fallback 绕过。

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
正式运行前应配置合理的 `execution.pipeline_wall_seconds`。`run cancel` 在 attempt 边界
停止新派发，允许在途请求和收尾结束；不能立即中断 GPU 请求。

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


## ?? wheelhouse?2026-10-26?

?????`dycon wheelhouse build PUBLICATION_ID [--source DIR ...] [--download-build-tools]`????`dycon wheelhouse verify PATH`?
??? Catalog ???????????????runtime/cache ???? wheelhouse ???? wheel???? SHA ??????????????????? Python/???????task constraints/repo pins ?????? pins?
`--download-build-tools` ? PyPI ???????? Python ?? setuptools/wheel/packaging ??????? release SHA256??????????????? runtime/cache/PUBLICATION_ID?? manifest.json????Requires-Python?tags???? SHA?state/wheelhouse ????????? ID ????
? config.yaml ? environment.wheelhouse ????????????????????????? task ? wheelhouse ???task pins?????????????? wheelhouse=null ???????? task ???????????? hash ?????????????????? task policy?
network_fallback=false ??? wheelhouse ??PIP_NO_INDEX=1 ??? bootstrap??? harness ? pip ??????????? bootstrap ????? pip ???????????????????????
???????? run ???????????????????????????????????????????? gold_image_audit ????? Python ??????


## ?????????2026-10-26?

??????? preflight?bootstrap ??? probe ?????????? dependency-failure.json??????? state/environment-failures/RUN_ID.json???????? attempt?stage ???????????? sample ???????????????
???? failure_type=dependency_error?reason?stage?requirements/failed_requirements?wheelhouse?returncode?detail????? diagnostic_path?????????? version_checks????? spec ??????
reason ?? wheelhouse_unavailable?missing_distribution?version_conflict?python_or_platform_incompatible?dependency_download_network_error?installed_version_mismatch ? dependency_install_failed??? pip ?????????????? requirement/constraint???? ResolutionImpossible??????? wheelhouse ??/????????????????????
`dycon run status RUN_ID`?`dycon harness status RUN_ID` ? `dycon report run RUN_ID` ?? environment_failures ???rollout ? attempt/progress.sample ?? failure_type?dependency_failure?progress ?? environment_dependency_failure_count ????? attempt?stage ????????????????????????? infrastructure_exhausted?
????????? pip/python -m pip/uv pip/conda ???????????? ERR marker ????????????????????harness outcome ?? status=infra_error?resolved=null?failure_type=dependency_error?summary ?? dependency_failure_count?dependency_reason_counts?failure_counts ??? dependency_error????????????????????????????? wheel ?????????
???????????????? marker ???????????mock?????????/pip ?????????????????????????? run ?????????????


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


## Submitted patch 统计入口（2026-10-26）

`analysis` 环境：`dycon action-submission-audit [ANALYSIS_ID] [--list | --check] [--task-count 80] [--min-submitted 30] [--reference-run RUN_ID]`。
纳入相同冻结 task ID 集合、rollout operation、submitted 严格超过阈值的 run。
新 ID 生成项目级只读来源统计；--list 只列范围；--check 验证该审计的专属 receipt 与输入哈希。
最新评测兼容旧 infra_error 与新版 completed+invalid_patch，重评不重复计数轨迹。
详细口径、结果与复查长度提案见 [action 审计文档](../action/SUBMISSION-REVIEW-AUDIT-CN.md)。
# 高并发准入与有界取消（2026-10-26 修复）

以下约定更新前文“cancel 仅写文件”和“reconcile 仅检查快照”的历史描述。

- serving 将长期 reserve/acquire/results 等待、native bind/output 回执、tensor 发布、短控制分别放入有界执行池；准入等待不再消耗完成请求所需的默认 executor。native output 回执缺失仍记录 capture failure，不冒充完整采集或触发重复模型采样。
- 新 execution 的 rollout 子进程与 supervisor 隔离 session；managed serving 在 exec 前登记身份并等待 supervisor ACK。记录 PID、start_ticks、host、boot_id、PGID/SID 以及 run/execution 归属。取消只管理这些专属资源，不按 GPU 或进程名清理共享服务。
- `dycon run cancel RUN_ID --reason TEXT` 保持幂等的请求写盘语义，同时由 runner 文件事件监听唤醒调度、停止新派发/重试，并通过独立控制路径唤醒 broker、取消 native 请求。CLI 接受请求不等于实际退出。
- `run status`、`report run`、`run wait` 的 `cancellation` 分开展示 request、supervisor ACK、runner ACK、真实 exit 和 termination/cleanup。强杀保留实际 -15/-9；协作取消也保留实际返回码，是否取消及是否清理以独立事实判断。
- 根 `execution.cancellation` 默认 control/cooperative/TERM/kill-verify/cleanup 预算为 3/10/30/5/15 秒。supervisor 以独立截止时间推进，无法协作时 TERM，必要时 KILL，再核验本 execution 的容器。不运行全项目 recover 或镜像 prune。系统不可中断 I/O、Docker/NAS 故障会明确记录失败/未核验，不承诺无条件清理成功。
- `run resume` 在旧 execution 清理确认后归档取消请求并建立新代际；旧取消通知不能影响新 execution。旧快照缺少新字段时使用本 execution 记录的兼容默认值，不改历史 config/hash/selection。
- 若 supervisor 已退出且取消尚未完成，`dycon run reconcile RUN_ID` 可按原身份与原取消期限接管停止；非父进程无法取得的退出码写 null，并明确 `returncode_source=unavailable_non_child`，不猜测信号或成功。仍活跃的 supervisor 不会被 reconcile 替代；旧版非隔离进程组拒绝自动扩大停止范围。
- 可用 `dycon run start RUN_ID --detach --check-only` 执行正式启动验收并由同一个 supervisor 管理。该命令真实使用 GPU/容器。`parallel.validation_repetitions=N` 除既有自然请求外，会运行 N 轮、每轮完整 rollout_workers 的 overlap；不按 capacity/branch_count 截掉超额组。新报告记录真实容量饱和与等待峰值。

普通/parallel/baseline/consensus、capture/steering 继续使用既有唯一入口和冻结配置。原取消流程不做工具中途精确续写；未完成 native/trajectory/capture 保留 partial/interrupted 证据，已完成 sample 不倒改。


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


## 复查长度分析入口（2026-10-26）

analysis环境：action-review-length ANALYSIS_ID --audit-id AUDIT_ID --action-id ACTION_ID。
检查：action-review-length ANALYSIS_ID --check。
实现：action/analyzers/review_length.py、review_phases.py、review_effects.py；
统计：action/evaluation/review_statistics.py。
读取两个冻结父产物及SHA绑定canonical轨迹，只读来源，不执行轨迹命令。
新ID生成独立action/outputs产物及receipt；unknown不补0，不属于七分类action-check。
详见[action复查长度分析](../action/REVIEW-LENGTH-CN.md)。

## Action 分支统计与曲线（analysis 环境，2026-10-26）

入口 `dycon action-branch-stats ANALYSIS_ID --source-run RUN_ID`；
重复 source-run 可组合来源，使用 `--rollout-id`（单 run）或 `--rollout-list` 选择轨迹。
最小绘图身份是 `(source_run_id, rollout_id)`，不按 task 分组；不再提供 `--task-id`。

`--line-mode per-trajectory` 每轨迹一张原始 step 图，`--line-metric NAME` 可重复选择面板。
默认面板为 entropy、normalized_entropy、action_types、effective_action_count、winner_uncertainty。
图在 `plots/per-trajectory/RUN_ID/ROLLOUT_ID.*`，原值在 `curves/per-trajectory.csv`，
`plots/trajectory-index.json` 为逐张覆盖清单。无 parallel step 时生成该轨迹的不可用说明图。

`--line-mode per-run|all-runs|both|none` 保留；run 平均曲线按轨迹独立归一化到0–100%，
每个 run 内等权平均。默认 both 不含逐轨迹图，要批量逐条绘制须显式使用 per-trajectory。
`--branch-scope all|retained|both` 默认both；`--termination-reason Submitted` 可选；
`--format png|svg`、`--dpi 180`、`--progress-points 101` 沿用原约定。

```bash
.venvs/analysis/bin/dycon action-branch-stats NEW_ID \
  --source-run RUN_ID --rollout-id ROLLOUT_ID --line-mode per-trajectory --line-metric entropy
.venvs/analysis/bin/dycon action-branch-stats NEW_ID --check
```

产物在 `action/outputs/ANALYSIS_ID/`，使用独立 branch receipt 和同名 `--check`。
详见 [action 分支统计](../action/BRANCH-ACTION-STATS-CN.md)。

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
