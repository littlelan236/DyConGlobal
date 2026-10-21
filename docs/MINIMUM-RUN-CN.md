# DyConGlobal 最小运行清单

核对日期：2026-10-21。工程：`/home/wangtianyi/DyConGlobal`。

入口语法、输入格式和分析口径见 [ENTRYPOINTS.md](ENTRYPOINTS.md)，容器故障见 [修复指南](container-environment-repair-cn.md)。

本清单按功能组织，命令里的大写 ID 需替换为实际值。runner 负责 rollout/Docker/harness；analysis 负责 tokenizer、action、训练与绘图；hidden 重放由统一入口派发到 latent-vllm。旧轨迹的 resolved 只表示旧 patch 的评分结果；新 rollout 必须重新评分。

## 1. 配置、Catalog、模型与环境

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/easysteer/bin/python runtime/scripts/verify_easysteer_runtime.py
# 检查任务镜像的 NAS 描述和内容；不等于执行 harness。
.venvs/runner/bin/dycon images-check
```

共用 generation、agent step/token limit、各种 timeout 配在根 `config.yaml`。创建 run 时冻结有效配置和 tasks。`runtime-check` 另要求 wheelhouse，不能用它代替 task 镜像/启动 gate。初次注册外部模型用 `model-register CHECKPOINT`。

## 2. 任务、轨迹与历史结果输入

```bash
# 输入结构见 ENTRYPOINTS.md 第 4 节；partition 是 task 属性。
.venvs/runner/bin/dycon tasks-import TASKS.json
# 拉取固定版本的 benchmark/NAS 镜像（已有数据时不重复执行）。
.venvs/runner/bin/dycon benchmark-pull COHORT_ID --count 600
```

历史轨迹、评测、hidden 元数据先显式筛选，再由隔离转换工具生成新版产物和 provenance；不得把旧 progress/PID 转成可 resume 状态。历史隔离、备份和转换边界见 [入口约定](ENTRYPOINTS.md#2-run-与产物)。跨轨迹比较可用 `baseline-select PROFILE TASK_KEY ROLLOUT_ID` 冻结参考轨迹；baseline 是比较用途。

## 3. 创建 run，启动 rollout

```bash
.venvs/runner/bin/dycon run create RUN_ID --operation rollout \
  --task-id pydata__xarray-4629 --task-id pylint-dev__pylint-7277
.venvs/runner/bin/dycon run show RUN_ID
.venvs/runner/bin/dycon run start RUN_ID
```

baseline、steering、parallel 共用 rollout runner。创建 run 时配置 `parallel.enabled/branch_count/estimator_ref`、`steering.enabled/payload`、`capture.enabled/anchors/layers`，以及 GPU/TP/workers/显存比例。parallel 需要与模型、site 匹配的 predictor 和 close capture。steering 需要真实且 fingerprint 匹配的 payload；没有训练好的 payload 时不伪造 steering 验证。

`run start` 前台等待 supervisor 结束并返回退出证据。后台模式使用：

```bash
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
```

wait 使用进程退出事件，不循环读取 progress。新建 run 时可在子命令前使用 `--set` 选择模式/资源，例如 `dycon --set parallel.enabled=false run create ...`；已有 run 不临时覆盖。parallel B>1 需同时启用 close capture 和匹配的 predictor。

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
.venvs/analysis/bin/dycon --run-id RUN_ID action-classify ACTION_ID \
  --rollout-id ROLLOUT_ID_1 --rollout-id ROLLOUT_ID_2
.venvs/analysis/bin/dycon --run-id RUN_ID action-check ACTION_ID
.venvs/analysis/bin/dycon --run-id RUN_ID action-index INDEX_ID ACTION_INDEX_ID
```

上述 `--run-id RUN_ID` 指定输出和配置归属，rollout IDs 指定来源轨迹；也可用同样的方式绑定预测和绘图。分类直接读取 canonical trajectory，生成互斥 taxonomy 标签、统计与 receipt，不依赖 enrichment。action-index 引用已有 hidden，不重复提取。action taxonomy 学习、训练与独立评测尚无正式实现。

## 10. Difficulty 训练、发布和预测评估

```bash
.venvs/analysis/bin/dycon --run-id RUN_ID difficulty-ridge EXPERIMENT_ID --mode dry-run
# 数据、split、targets 确认后才运行 fit。
.venvs/analysis/bin/dycon --run-id RUN_ID difficulty-ridge EXPERIMENT_ID --mode fit
.venvs/runner/bin/dycon --run-id RUN_ID predictor-publish PREDICTOR_ID \
  --experiment runs/RUN_ID/difficulty/checkpoints/EXPERIMENT_ID --target reasoning_length
.venvs/runner/bin/dycon --run-id RUN_ID difficulty-predict INDEX_ID \
  runtime/parallel-artifacts/PREDICTOR_ID/manifest.json PREDICTION_ID
.venvs/analysis/bin/dycon --run-id RUN_ID difficulty-evaluate INDEX_ID \
  runs/RUN_ID/difficulty/predictions/PREDICTION_ID/predictions.json SCORE_ID
```

训练输入 dataset/receipt 在根配置指定；已存在的全局实验也可显式传其真实路径发布。Ridge CLI 负责冻结数据/split、层×alpha 选择及报告；predictor 发布模型 fingerprint、sites、target、checkpoint hash，供 parallel 在线评分。`difficulty-evaluate` 读取真实外部 predictions；不能用 GT 伪造预测验收。通用 MLP/Ridge+FFN 有 Python API，尚无完整通用训练 CLI。不同 index schema 的适用范围见入口表。

## 11. 并行分支可视化和评分稳定性

```bash
.venvs/analysis/bin/dycon --run-id RUN_ID difficulty-plot-parallel SOURCE_RUN_ID --list
.venvs/analysis/bin/dycon --run-id RUN_ID difficulty-plot-parallel SOURCE_RUN_ID PLOT_ID \
  --rollout-id ROLLOUT_ID
.venvs/analysis/bin/dycon --run-id RUN_ID difficulty-evaluate-parallel SOURCE_RUN_ID INSTABILITY_ID \
  --rollout-id ROLLOUT_ID
```

绘制分支得分、实际 winner 轨迹、缺失分数；稳定性分析按相同 action/prefix 比较 reasoning 扰动，产出指标、置信区间和 receipt。图中的 winner 必须来自记录，不能重新取最小值冒充实际选择。

## 12. 运维与诊断

环境安装脚本在 `runtime/scripts/setup_*.sh`；Docker/NAS smoke、capture matrix、eager replay smoke 和 owned-resource recovery 是维护工具。启动前核对 GPU/作用范围，不重复导入 benchmark、不无条件运行 gold repair、不删除其他任务的资源。`dycon recover` 只处理本工程遗留资源。额外验证命令及副作用见 [ENTRYPOINTS.md](ENTRYPOINTS.md)。

## 历史验收与适用边界

日常清单不承载单批次结果。2026-10-21 的两题回归、baseline/parallel、离线重放数值对照和迁移记录保存在各 run、工程外 `/home/wangtianyi/dycon-isolation-20261021/` 以及本次 docs 原文归档；位置见 [入口约定](ENTRYPOINTS.md#9-文档历史与追溯)。旧 snapshot/receipt 不因文档整理而变更。

没有真实匹配 payload 时不执行 steering 验收；已有数值 checkpoint 的发布/推理不等于训练完成。`run cancel` 当前只是写取消请求，`reconcile` 当前主要做快照一致性检查；执行和恢复边界见入口表。
