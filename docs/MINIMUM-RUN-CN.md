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

优先用原阶段的恢复命令，例如 `capture recover RUN_ID` 恢复已存在的 capture 发布。只有确认 worker 已结束、清理状态完整，且明确需要重新执行选定失败样本时，才登记 `run retry-failed` 请求；之后通过 `run resume` 处理。每次新的主动重跑使用新的 request ID。CaptureError、pending recovery 或未确认的 spool 不应直接换 seed 重跑。

```bash
.venvs/runner/bin/dycon run retry-failed RUN_ID \
  --request-id REQUEST_ID --exit-status EXIT_STATUS \
  --task-id INSTANCE_ID --reason "reason"
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

详细边界见[入口约定](ENTRYPOINTS.md#runs-samples-and-attempts)与[capture 指南](CAPTURE-CN.md)。

## 5. 在线采集和 token 分析

```bash
.venvs/runner/bin/dycon capture status RUN_ID
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
.venvs/analysis/bin/dycon difficulty-train ATTEMPT_ID --training-config CONFIG.yaml
```

Action 与 difficulty 产物位于项目级目录，使用显式来源 run 和固定选择。difficulty 数据集、划分和训练配置不可互相替代；详细命令、数据语义及预测器发布见各自模块 README。

## 7. Run 比较和多卡

```bash
.venvs/runner/bin/dycon run compare RUN_A RUN_B --baseline RUN_A --format markdown
```

比较只读被冻结的 run 计划、最终选择及评测证据，不启动模型或 harness。同样的 repetition 数默认按 task×repetition 配对；数量不同需显式指定每个 run 的 repetition。指标解释见[trajectory comparison](TRAJECTORY-COMPARISON-CN.md)。

多 GPU 独立 TP=1 调度使用配置 `serving.trajectory_sharding: per_gpu_tp1`，由正式 run 入口执行，见[多 GPU 指南](MULTI-GPU-TP1-CN.md)。

## 8. 任务专用入口

SkillsBench、WebArena Verified、SWE-bench、Harbor 的 task 导入、容器准备和 preflight 各有约束。开始任务前阅读[benchmark 操作指南](BENCHMARKS-CN.md)，不要从实验报告复制旧 run 的命令或资源参数。


| Run | Accepted task change | Accepted sample change | Steps change | Prompt token change | Completion token change | Total token change | Reward sum change | Reward mean change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skillsbench-agent75-r3-qwen36-27b-positive-gpu23-tp1-20261031 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| skillsbench-test57-r3-consensus32-train30-gpu0123-tp1-w4-s64-20261102 | -41.54% | -51.66% | -20.18% | -20.54% | -36.69% | -21.05% | -5.09% | -5.09% |