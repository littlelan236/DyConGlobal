# 任务入口与接口约定

核对日期：2026-10-21。工程根目录 `/home/wangtianyi/DyConGlobal`，NAS 根目录由 `config.yaml` 的 `paths.nas_root` 指定。

日常操作从 [最小运行清单](MINIMUM-RUN-CN.md) 开始；依赖、Docker、镜像、网络和 harness 环境故障查 [容器环境修复指南](container-environment-repair-cn.md)。本文保存完整入口表、输入格式、产物与分析口径。模块内部实现另见源码及各模块 README。

## 1. 入口、环境与配置

统一入口为 `core.cli:main`。已有入口时直接复用，不能复制 launcher、绕过锁/快照启动 worker，或另写模型请求和产物登记逻辑。

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

全局参数放在子命令前：`--config config.yaml`、`--set key=value`、`--run-id RUN_ID`。`--run-id` 将分析命令绑定到已有 run 的冻结配置和输出目录；与命令中的 `SOURCE_RUN_ID` 可以不同，例如在新 analysis run 下分析旧来源 run 的分支。

根 `config.yaml` 保存共用配置，generation、agent step/token limit 和所有 timeout 按项目约定在根配置维护。run 私有模型/资源、capture/parallel/steering 等设置在 `run create` 时冻结；已有 run 的执行、resume 和分析拒绝临时 `--set`。当前 CLI 对 generation/agent/timeouts 的覆盖已有拒绝检查；嵌在其他配置块中的 timeout 也应遵守根配置约定。

`run create --set` 实际写法是 `dycon --set ... run create ...`。不带 run 的部分旧数据分析入口仍读取根配置并允许显式覆盖，不能将它与 run-scoped 输出混用。根 YAML 后续改变不应改变已创建 run 的有效配置。

## 2. Run 与产物

| 操作 | runner 命令 | 行为与边界 |
|---|---|---|
| 检查配置/索引 | `config-check`、`catalog-check` | 校验 strict YAML；校验当前 Catalog 的关系和文件 SHA |
| 建立实验 | `run create RUN_ID --operation rollout --task-id INSTANCE_ID` | 也支持 `--tasks-file FILE`、`--partition train/val/test/excluded`；显式 ID/列表应非空 |
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
  difficulty/{datasets,predictions,checkpoints,evaluation/outputs}/ID/
  action/{outputs,datasets}/ID/
  reports/ / artifacts.json
```

`catalog/manifest.json` 是业务事实索引。`baseline/` 已为空且不参与源码安装；历史 trajectory/patch/evaluation 在 `isolated/history-baseline-20261021/`，由当前 Catalog 显式 locator 读取。5 个带旧路径的冻结评测/分析 run 原样保存在该隔离目录的 `frozen-runs/`，不可当作新版 run resume。

工程外原件/迁移报告：`/home/wangtianyi/dycon-isolation-20261021/baseline-cleanup/`。旧 receipt、权重和 NAS tensor 不改写；转换产生新 ID/receipt，禁止旧路径 alias/fallback。当前 Ridge 配置使用转换后的 `submission-shortest375-20261019-relocated-20261021`。训练输入是显式共享 dataset，训练输出在绑定 run 下；已发布 predictor 仍在 `runtime/parallel-artifacts/PREDICTOR_ID/`。

## 3. 全部业务命令

以下为子命令语法；省略解释器前缀。对 run 下的分析产物，使用 analysis/runner 对应环境的 `dycon --run-id RUN_ID ...`。`enrich/token-trace/hidden/dataset-index/token-join` 在首个 dataset 参数是新版 run ID 时也会自动绑定 run。

| 功能 | 环境 | 命令 | 实现/主要输出 |
|---|---|---|---|
| 注册 checkpoint | runner | `model-register CHECKPOINT` | `core.model_identity`，内容 fingerprint |
| 任务导入 | runner | `tasks-import TASKS.json` | `catalog.store.Catalog.register_tasks` |
| 拉取 benchmark | runner | `benchmark-pull COHORT_ID [--count 600] [--workers N]` | `catalog.benchmark`，固定 dataset/results revision、NAS OCI 与 task 登记 |
| 镜像核验 | runner | `images-check [--deep]` | 当前根 benchmark 范围的 NAS 描述/blob；不执行 load/harness |
| 轨迹导入 | runner | `rollouts-import ROLLOUTS.json` | `rollout.artifacts`，字节不变的轨迹与 run-local patch 登记 |
| 外部评分导入 | runner | `evaluations-import EVALUATIONS.json` | `catalog.importers.evaluation`，实际被评分 patch 的 SHA 必须一致 |
| 固定参考轨迹 | runner | `baseline-select PROFILE TASK_KEY ROLLOUT_ID` | baseline 是比较用途；不依赖旧 baseline 目录 |
| Harness 选择 | runner | `harness prepare RUN_ID [--rollout-id ID ...]` | 默认 producer run 轨迹；重评历史 patch 时显式传 ID，必须属于 run tasks |
| 官方评分 | runner | `harness run RUN_ID EVAL_ID [--gold]` | `benchmark_evaluation.harness`，SWE-bench 4.1.0；gold 只验环境 |
| Capture 发布恢复 | runner | `capture recover RUN_ID` | `hidden.recover_online`，不重新请求模型；旧命令 `capture-recover` 仍存在 |
| Capture 重分类 | runner | `capture-reclassify RUN_ID --session-id SID ...` | 已 finalized 证据的有界规则修正，生成新 revision，不补造 hidden |
| 在线索引 | analysis | `online-index RUN_ID INDEX_ID` | `difficulty.datasets.online`，schema 3/NAS 引用 |
| 离线标签/prefix | analysis | `enrich RUN_ID` | `enrichment.trajectory`，冻结轨迹、token evidence、labels |
| Token trace/audit | analysis | `token-trace RUN_ID`、`trajectory-token-audit RUN_ID` | exact/reconstructed 证据、token/step/phase/position |
| 离线 hidden | analysis | `hidden RUN_ID CAPTURE_ID` | `hidden.scripts.extract_vllm` → latent-vllm eager |
| 离线索引 | analysis | `dataset-index RUN_ID CAPTURE_ID INDEX_ID` | schema 1，sample ID join，复用 labels/hidden |
| 精确 token join | analysis | `token-join RUN_ID CAPTURE_ID JOIN_ID` | schema 2 `rows.jsonl`；`hidden-token-join-audit JOIN_ID` 审计 |
| Action 分类 | analysis | `action-classify ACTION_ID --rollout-id ID ...` | 直接 canonical trajectory；也支持 `--rollout-list FILE` |
| Action 校验 | analysis | `action-check ACTION_ID` | receipt/来源内容/分类输入核验 |
| 共享 hidden 视图 | analysis | `action-index INDEX_ID ACTION_INDEX_ID` | 引用共享 index，不重复提取 hidden |
| Ridge 实验 | analysis | `difficulty-ridge EXPERIMENT_ID --mode dry-run/check-data/fit` | 三种值选其一；默认 fit，见下文 |
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
  "evaluation_id": "IMPORTED_EVAL",
  "report": "/absolute/source/report.json", "report_sha256": "SHA256",
  "bindings": {
    "ROLLOUT_ID": {"evaluated_patch": "/absolute/actually-graded.diff",
                   "outcome_path": ["outcomes", "INSTANCE_ID"]}
  }
}
```

`outcome_path` 定位结果对象。completed 的结果必须有 boolean resolved，实际评分 patch 必须与 Catalog patch 内容一致。

预测文件为 JSON list：`[{"sample_id":"ROLLOUT:STEP:ANCHOR:OCCURRENCE","target":"remaining_steps","prediction":12.3}]`。target 可用 `remaining_steps` / `remaining_reasoning_tokens`；GT 从冻结 enrichment 读取，不接受文件自带 GT，不用 GT 冒充 predictor 结果。

## 5. Rollout、在线 capture 与 parallel 的关键契约

| 模式 | 开关 |
|---|---|
| 普通/baseline | `parallel.enabled=false`、`steering.enabled=false` |
| Steering | 启用 steering 并提供匹配 fingerprint 的 payload；禁用时 payload=null |
| Parallel | `parallel.enabled=true`、B>1、thinking、managed capture、think_close、匹配 sites 的 estimator_ref |
| 在线采集 | capture 独立开关，layers/anchors、storage budget 和 failure_policy 明确配置 |

parallel 使用并发完整 reasoning+action 请求，等待所有候选完成后比较 close hidden 的 predictor score；只执行 winner action。B=1 不代表四分支验收；fallback、终止和缺 close 的候选分别记录。predictor provider 返回有限 `[N]` 分数，输入 `[N, sites, hidden_size]`；manifest 绑定 checkpoint/model fingerprint、sites、target、endpoint、文件和实现 SHA。实现或权重变更须重新发布，不能改已有 predictor manifest。

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

Ridge settings 在 `difficulty.ridge_probe`，包含 dataset/receipt、target、layers/alphas、split_strategy、seed、selection_metric。`dry-run` 核验来源并写计划/split，`check-data` 另做 feature 检查，`fit` 才拟合；模式写入 receipt，输出 ID 拒绝覆盖。不要对当前全量数据随意执行 fit。

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
