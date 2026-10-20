# 任务启动入口与执行约定

并行完整响应 rollout 已增加可选 `parallel` 配置，继续使用下表的 `dycon rollout` / `--resume` / `--check-only` 入口。多个 reasoning+action 候选通过同一服务并发原生 `n=1` 请求产生，只执行 winner；全分支 close hidden 和选择证据沿原有 spool/NAS 发布。外部 predictor 尚需后续对接，默认关闭；不修改 difficulty 实现或生产配置。契约、NAS reader、失败规则及 FULL graph 验收边界见 [parallel-rollout.md](parallel-rollout.md)。CPU/协议测试不能替代真实 GPU 吞吐和选择效果验收。

2026-10-19新增单层Ridge pilot：analysis环境 `.venvs/analysis/bin/dycon difficulty-ridge EXPERIMENT_ID [--mode dry-run|check-data|fit]`。默认fit，消费唯一config.yaml的difficulty.ridge_probe及冻结submission清单。新实验ID拒绝覆盖；三种模式的receipt明确区分核验、数据检查和拟合。细节见 [ridge-submission-pilot.md](ridge-submission-pilot.md)。此入口补齐本专项训练/参数发布，后文原Python训练API边界仍适用于通用FFN/残差训练。

核对日期：2026-10-19。依据远程实际代码 `core/cli.py`、`pyproject.toml` 及各模块实现；历史报告中的旧命令和参数不覆盖本页。所有命令均从 `/home/wangtianyi/DyConGlobal` 执行，示例中的大写 ID/路径须替换为真实值。

## 先复用入口

已有统一入口时，必须直接复用，**不得自行创建临时脚本、复制 launcher 或绕过入口直接调内部函数**。入口负责任务选择、模型身份、配置快照、并发锁、启动验收、失败恢复及产物绑定；临时重写其中任一部分都会造成行为对齐问题。新能力确需新入口时，应扩展工程内可维护的 CLI/模块并更新本页。

`pyproject.toml` 将 `dycon` 注册到 `core.cli:main`。以下两种调用等价：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon --config config.yaml config-check
.venvs/runner/bin/python -m core.cli --config config.yaml config-check
```

全局参数 `--config`、`--set` 放在子命令前。`config.yaml` 是唯一可编辑配置；配置所在目录即 `project_root`，不能把 YAML 复制到临时目录再启动正式任务。离线任务允许显式 `--set`；正式 `rollout` 拒绝所有 `--set`，`evaluate` 也拒绝覆盖并加载保存的 run 配置。

## 正式 CLI 全表

Action trajectory 分类直接读取 manifest 绑定的 canonical rollout trajectory，不需要 enrichment 或 hidden。已有统一入口时不得另建临时脚本。

下表中的 `dycon` 表示使用指定环境的 `.venvs/<环境>/bin/dycon`，不是系统 Python。每个子命令均在 `core/cli.py` 分发。

| 任务 | 环境 | 命令 | 实现与主要产物/作用 |
|---|---|---|---|
| 解析并校验配置 | runner | `dycon config-check` | `core.config.resolve`；输出 resolved config，不启动服务 |
| 校验 catalog | runner | `dycon catalog-check` | `Catalog.verify`；manifest 引用、分区和内容核验 |
| 旧 workflow 前置检查 | runner | `dycon runtime-check` | `baseline.workflow.assert_runtime_ready`；检查模型、全局 wheelhouse 等，不等价于真实启动 gate |
| 导入任务 | runner | `dycon tasks-import TASKS.json` | `Catalog.register_tasks` → `baseline/manifest.json` |
| 导入历史轨迹 | runner | `dycon rollouts-import ROLLOUTS.json` | `baseline.importers.legacy` → canonical trajectories/patches、manifest |
| 导入历史评测 | runner | `dycon evaluations-import EVALUATIONS.json` | `baseline.importers.evaluation`；校验实际评测 patch 绑定 |
| 注册模型 | runner | `dycon model-register CHECKPOINT` | `core.model_identity.register` → `runtime/model-identities/`；返回 fingerprint，不自动改配置 |
| 核验任务镜像 | runner | `dycon images-check [--deep]` | `NasImageStore.verify`；不执行 docker load |
| 固定 baseline | runner | `dycon baseline-select PROFILE TASK_KEY ROLLOUT_ID` | `Catalog.select_baseline`；显式固定每 task 的比较轨迹 |
| 启动 rollout | runner | `dycon rollout RUN_ID [--task-id INSTANCE_ID ...] [--unrolled]` | `baseline.rollout.runner.run_manifest` → `runs/` 和 baseline canonical 产物 |
| 恢复 rollout | runner | `dycon rollout RUN_ID --resume` | 复用不可变 run 配置和 selection |
| 真实启动验收 | runner | `dycon rollout CHECK_RUN --check-only --task-id INSTANCE_ID` | 同一 runner；容器/工具/capture gate，验收后退出、不接纳正式 cohort |
| 冻结评测输入 | runner | `dycon prepare-evaluation EVAL_INPUT_RUN` | `baseline.workflow.prepare_run`；按 manifest rollouts 创建独立快照 |
| 官方评测 | runner | `dycon evaluate RUN_ID EVAL_ID [--gold]` | `baseline.evaluation.harness.evaluate` → `baseline/evaluations/`；gold 只验环境 |
| 轨迹 enrichment | analysis | `dycon enrich DATASET_ID` | `enrichment.trajectory.enrich` → `enrichment/outputs/DATASET_ID/` 和 `reports/` |
| 构建 token trace | analysis | `dycon token-trace DATASET_ID` | `enrichment.token_trace.build_token_traces` → dataset 下 `token-traces/` |
| token trace 审计 | analysis | `dycon trajectory-token-audit DATASET_ID` | `audit_token_traces`；核验 trace SHA、维度和计数 |
| 离线 hidden | analysis | `dycon hidden DATASET_ID CAPTURE_ID` | trajectory APC：同轨迹依赖顺序、跨窗口保留 cache → `hidden/outputs/CAPTURE_ID/` |
| 普通离线共享索引 | analysis | `dycon dataset-index DATASET_ID CAPTURE_ID INDEX_ID` | `difficulty.datasets.index.build_index` → `difficulty/datasets/INDEX_ID/index.json`，schema 1 |
| exact token-hidden join | analysis | `dycon token-join DATASET_ID CAPTURE_ID JOIN_ID` | `difficulty.datasets.token_join` → `rows.jsonl` + `receipt.json`，schema 2 |
| join 审计 | analysis | `dycon hidden-token-join-audit JOIN_ID` | 校验 join 行和 receipt；不重建数据 |
| 在线发布数据索引 | analysis | `dycon online-index RUN_ID ONLINE_INDEX_ID` | `difficulty.datasets.online.build_published_index` → schema 3 `index.json`，直接引用 NAS shard |
| action 共享视图 | analysis | `dycon action-index SHARED_INDEX_ID ACTION_INDEX_ID` | `action.datasets.index` → `action/datasets/`；action label 为空，不重复提 hidden |
| 显式预测评分 | analysis | `dycon difficulty-evaluate INDEX_ID PREDICTIONS.json SCORE_ID` | `difficulty.evaluation.run` → `difficulty/evaluation/outputs/` 的指标与轨迹图；当前读取 schema 1 label-shard 索引 |
| 并行轨迹评分可视化 | analysis | `dycon difficulty-plot-parallel RUN_ID PLOT_ID --task-id INSTANCE_ID [--attempt-id attempt-000]`；`RUN_ID --list` 列出轨迹 | `difficulty.evaluation.parallel`；核验最终轨迹与已发布分支记录，生成 PNG/SVG、评分表及 receipt，见 [使用说明](parallel-score-visualization.md) |
| 并行评分不稳定性评测 | analysis | `dycon difficulty-evaluate-parallel RUN_ID EVALUATION_ID [--rollout-id ID ... / --task-id ID ...]` | 复用绘图对齐；输出平均两两差、std/range、倍数分歧率、相对 Q 及 task/repo CI，见 [使用说明](parallel-score-instability.md) |
| capture 发布恢复 | runner | `dycon capture-recover RUN_ID` | `hidden.recover_online`；读取原 run 快照，重试 spool 发布，不发起新模型请求 |
| 容器恢复清理 | runner | `dycon recover` | `runtime.containers.recovery`；只清理无活跃 run 锁的本工程容器 |

输入清单格式见 [INPUTS.md](INPUTS.md)。`baseline/scripts/rollout.py`、`evaluate.py`、`prepare_run.py`、`import_rollouts.py`、`verify.py`，以及 `enrichment/scripts/enrich.py`、`hidden/scripts/extract.py`、`runtime/scripts/images_check.py`、`recover.py`、`difficulty/evaluation/scripts/evaluate.py` 是 `core.cli.module_main` 的兼容薄封装，不是第二套工作流。

## Action trajectory 分类

analysis 环境执行 action-classify ACTION_ANALYSIS_ID，可重复使用 --rollout-id 或提供 --rollout-list；action-check ACTION_ANALYSIS_ID 校验 selection、trajectory SHA、labels coverage、summary 和 receipt。该路径直接读取 manifest 绑定的 canonical trajectory，不需要 enrichment 或 hidden，也不执行 command。详细类别边界见 action/TAXONOMY-CN.md。

## rollout、capture 和 steering 共用执行链

```text
dycon / python -m core.cli
  → baseline.rollout.runner.run_manifest
  → Catalog.select_tasks → snapshot / load_snapshot
  → ContainerFactory + ServingManager
  → startup.check_tool_chain
  → agent / model → canonical trajectory、patch、manifest
                    + capture spool → NAS → token-event enrichment
```

新 run 选择 manifest tasks 并冻结配置/selection；`--task-id` 可重复，使用 instance ID；`--unrolled` 过滤已有正式轨迹的任务。恢复使用原 selection，不接受 `--task-id` 或 `--unrolled`。`--check-only` 仍会创建 run、启动真实模型和 Docker 工具验收，应使用独立 CHECK_RUN，不能当成无副作用的配置检查。

| 模式 | `capture.enabled` | `steering.enabled` | 启动入口 |
|---|---|---|---|
| 普通 | false | false | `dycon rollout RUN_ID` |
| capture | true | false | 同上 |
| steering | false | true | 同上 |
| steering + capture | true | true | 同上 |

开启 steering 还须设置匹配 checkpoint fingerprint 的 `steering.payload`；关闭时 payload 必须为 null。capture 要求 managed serving，Qwen GDN 当前正式 anchor 为 `reasoning_newline`、`think_close`；prefill chunk 由原生调度（`prefill_token_chunk_size=0`）。`think_open` FULL prefill 的诊断许可不能当成通用生产支持。`capture.graph_mode=FULL` 是采集契约，当前 managed serving 实际传给 vLLM 的 compilation mode 是 `FULL_AND_PIECEWISE`，应区分配置意图与执行证据。

默认 capture 失败策略 `record_and_continue` 允许 agent 继续，因此 run 完成不等于 hidden coverage 完整。分别查看 `capture_status`、coverage、发布状态和 receipt。run 收尾自动构建 token-event enrichment；恢复命令负责发布，不保证重做曾失败的所有后处理步骤，若缺 `enrichment/outputs/RUN_ID/receipt.json`，不可直接声称 `online-index` 就绪。

长任务如需 supervisor，复用已有工具：

```bash
.venvs/runner/bin/python -m runtime.scripts.supervise_rollout RUN_ID --task-id INSTANCE_ID
.venvs/runner/bin/python -m runtime.scripts.launch_selection RUN_ID --selection SELECTION.json
```

二者最终仍执行 `core.cli rollout`。supervisor 状态在 `state/supervisors/RUN_ID/`，目录已存在时拒绝启动，不能把它当作可重复调用的通用恢复命令。直接恢复 rollout 使用 `dycon rollout RUN_ID --resume`。

`baseline.scripts.capture_verified` 只发布静态 NAS/task 输入；名字不表示它会运行 capture。`runtime.scripts.validate_rollout_modes --run-prefix PREFIX --task-id INSTANCE_ID` 依次调用同一 CLI 的 `--check-only`，暂时切换并恢复 `config.yaml`；它会启动真实服务、构造测试 steering 向量，不作为生产 launcher。

## 两条数据工作流

离线重放：先按 model/partition/purposes 确认 catalog 选择，再执行：

```bash
.venvs/analysis/bin/dycon enrich DATASET_ID
.venvs/analysis/bin/dycon hidden DATASET_ID CAPTURE_ID
.venvs/analysis/bin/dycon dataset-index DATASET_ID CAPTURE_ID INDEX_ID
.venvs/analysis/bin/dycon action-index INDEX_ID ACTION_INDEX_ID
.venvs/analysis/bin/dycon difficulty-evaluate INDEX_ID PREDICTIONS.json SCORE_ID
```

需要 token 级审计时，对同一 dataset 执行 `token-trace`、`trajectory-token-audit`，再 `token-join`、`hidden-token-join-audit`。join 要求 exact prompt/output IDs，且每个 hidden 行唯一对应 absolute position/token ID；历史文本重建通过计数检查也不等于 exact。

在线采集：通过同一 `dycon rollout` 生成 NAS shard/ledger 和 `enrichment/outputs/RUN_ID/`，确认发布、token coverage 和评测标签后，再 `dycon online-index RUN_ID NEW_INDEX_ID`。如在评测前建了索引，resolved 标签仍为空；完成评测后需新建索引 ID，已有索引不自动更新。

三种索引不能互换：`dataset-index` 的 schema 1 供当前离线 scorer；`token-join` 的 schema 2 是精确对齐审计表，没有供 `action-index` 读取的 `index.json`；`online-index` 的 schema 3 带 task partition/labels 和 NAS shard locator，训练 loader 支持，但当前 `difficulty-evaluate` 仍依赖 schema 1 的 `label_shard`，不能直接接 schema 3。

## hidden 环境和已有批量入口

离线提取由统一 CLI 自动启动 `hidden.vllm.python`（当前 `.venvs/latent-vllm/bin/python`）的 `hidden.scripts.extract_vllm`。不要用 serving 环境替代，也不要重新实现 vLLM batching、capture 或 cache。

离线 adapter 只有 trajectory APC 执行策略：同 rollout/session/branch 的相邻 anchor 不会在同一个原生请求窗口中并发，前一个完成后才提交后一个；不同轨迹仍由 vLLM 并发执行。必须开启 `hidden.vllm.enable_prefix_caching=true`，关闭会明确拒绝。新实现/配置使用新 capture ID，不能复用旧 partial。恢复后 GPU cache 需要重新预热。细节见 [连续重放契约](vllm-trajectory-apc-20261019.md)。

独立 eager 环境尚未存在时，使用 `.venvs/runner/bin/python -m hidden.scripts.prepare_vllm_runtime`；它复制并核验 serving runtime，已存在的目标会拒绝覆盖。该主机已有四个环境：runner、analysis、easysteer、latent-vllm。

历史批量 extraction 已有 launcher：

```bash
.venvs/analysis/bin/python -m enrichment.scripts.launch_extraction \
  --root /home/wangtianyi/DyConGlobal
```

它针对已有 `state/anchors-20261017/queue.json`、`launch.json` 和 `state/extraction-models.json`，固定 GPU 0–3 的 vLLM TP 提取；校验 completed receipt、checkpoint/trajectory 绑定、旧 worker/GPU locks，再冻结配置并派发单个常驻 `run_extraction_queue.py`。该 worker 按模型复用 engine 和 APC。这是已有批次的维护入口，不是自动创建新批次的工具。系统错误保留 pending 并停止 worker，不能反复跳过错误耗尽队列，也不能直接手动起 worker 绕过 preflight。

多 checkpoint 历史 token trace 已有 `enrichment.scripts.build_all_token_traces --model-inventory INVENTORY.json --dataset-id DATASET_ID`；inventory 为含 `models` 映射的对象，各模型有 path/fingerprint。它固定选择 train/training，并记录弱证据和失败；不是当前单模型 `token-trace` 的可任意替代品。

## 训练及 action 的当前能力边界

已有 `difficulty.models.estimators.build_estimator` 和 `difficulty.training.train.train_estimators` Python API，支持 Ridge、FFN MLP、Ridge+FFN。submission-remainder 单层 Ridge 已有正式 `difficulty-ridge` CLI、固定 task 划分、参数/receipt/验证报告；通用 FFN、残差及其它标签尚无完整训练 workflow。`python -m difficulty.training.train` 不会启动通用训练；其它需求应扩展统一入口，不能自行写临时训练脚本。

schema 3 loader 按 manifest task partition 读 train/val；当前观察到的 458 个 task 全为 train，不能据此声称具备 task 级 validation。schema 1 loader 按 rollout ID 排序取约 80%/20%，只保证同 rollout 不跨组，不保证同 task 不跨组。FFN 当前使用 CPU tensor/模块；不能把它写成已支持 GPU 训练。详细 API、目标变换和限制见 [difficulty/README.md](../difficulty/README.md)。

action 现在包含直接 trajectory 分类、共享 hidden 索引和 receipt 审计；action 训练、taxonomy 学习和动作评测尚未实现，见 [action/README.md](../action/README.md)。

## 环境、验收和维护工具

| 用途 | 已有入口 | 边界 |
|---|---|---|
| runner / analysis 安装 | `bash runtime/scripts/setup_runner.sh` / `setup_analysis.sh` | 按 locks 安装，会修改环境 |
| EasySteer 安装 | `bash runtime/scripts/setup_easysteer.sh` | 会安装 vendor/overlay 并更新 lock，不是只读验证 |
| runtime 身份核验 | `.venvs/easysteer/bin/python runtime/scripts/verify_easysteer_runtime.py` | 校验已有 runtime |
| Docker/NAS smoke | `.venvs/runner/bin/python -m runtime.scripts.smoke_images` | 4 个合成 OCI 镜像的真实 load/create/remove，非 SWE-bench gold |
| 原生 capture 精度矩阵 | `.venvs/easysteer/bin/python -m hidden.evaluation.capture_matrix --output NEW_DIR` | 实际 GPU 诊断，读取 serving GPUs，不执行正式 cohort |
| eager replay smoke | `.venvs/latent-vllm/bin/python -m hidden.evaluation.smoke_vllm --output NEW_DIR` | offline runtime/capture 诊断 |
| 发布 artifact 重放 | `.venvs/latent-vllm/bin/python -m hidden.evaluation.replay_online_artifacts` | 唯一单次 trajectory APC 路径；`--source-run` 固定来源，`--event-index` 固定事件，`--extra-prefixes` 缺失点按同一顺序重放但不参与在线数值比较；用 `--help` 核对参数 |
| online/replay 结果比较 | `.venvs/analysis/bin/python -m hidden.evaluation.compare_vllm_replay --online-dir DIR --replay-dir DIR --output FILE` | 数值对比，不能证明所有 checkpoint 等价 |
| 原生 dispatch 诊断 | `.venvs/easysteer/bin/python -m hidden.evaluation.diagnose_native_dispatch --model MODEL --gpus GPU --output NEW_DIR` | 记录 graph/compile 实际执行路径 |
| gold 镜像审计 | `.venvs/runner/bin/python -m runtime.scripts.gold_image_audit --root ROOT --instances FILE --audit-id ID` | 真实 Docker/gold 维护，先核对输入和 `--help` |

2026-10-19 的在线轨迹对照使用 NAS 批次
`capture-verified200-20261018-143317`，报告保存在
`hidden/evaluation/outputs/replay-online-smoke-20261019-4/`。该验证只选取
存在匹配 request ledger 的事件；在线事件索引中的孤立 request 不得被当作
模型数值失败。共享 GPU smoke 必须显式设置 `allow_shared_gpus=true`、
`min_free_gpu_bytes` 和 `kv_cache_memory_bytes`，且不能终止已有 serving 进程。

`hidden/evaluation/validate_online.py` 等旧验收脚本带硬编码路径、GPU、配置和顶层执行，不能因文件名含 validate 就直接 import 或当作通用入口。迁移、清理和 NAS repair 工具还有专用 plan/apply、冻结 cohort 等约束，应先查对应源码/`--help` 和冻结输入，不能沿用历史任务范围直接执行。

`hidden.evaluation.first_rollout RUN_ID` 自动调用同一个快速重放接口。新报告使用 schema 2：在线匹配结果在 `trajectory-apc/`，缺在线点在 `offline-only/`，总报告使用 `comparison` / `cache` / `windows`；旧 `phases.cold/warm/reordered` 和逐窗口清缓存的执行逻辑已移除，旧报告不重写。`smoke_vllm` 内的重复/乱序输入仅用于 GPU 契约测试，不是可选择的生产重放模式。
