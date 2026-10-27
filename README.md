# DyConGlobal

当前 active run 统一使用 schema-2。历史 runs 已通过一次性迁移转为 `repeat_count=1` 的 active run；原始 schema-1 目录保存在 `archive/runs-schema1/`，不再由生产入口读取。迁移凭证位于 `migration/runs-to-repetition-20261024.json`，切换标记位于 `migration/CUTOVER.json`。

Coding-agent rollout、官方 SWE-bench harness、在线 hidden capture、离线 vLLM 重放及 action/difficulty 分析工程。远程工程位于 `/home/wangtianyi/DyConGlobal`，NAS 根目录由根 `config.yaml` 的 `paths.nas_root` 配置。

## 使用文档

| 文档 | 用途 |
|---|---|
| [最小运行清单](docs/MINIMUM-RUN-CN.md) | 每个功能模块的最少操作、统一 run 入口、产物与恢复 |
| [任务入口与接口约定](docs/ENTRYPOINTS.md) | 全部 CLI、输入格式、环境、token/site/label 契约及能力边界 |
| [容器环境修复](docs/container-environment-repair-cn.md) | Docker/NAS 镜像、依赖、线程、网络、harness 环境问题的定位与验收 |

从项目根目录用相应 venv 中的 `dycon`，等价于该解释器的 `python -m core.cli`。已有统一入口时不得另建 launcher、直接启动内部 worker，或重写任务选择、模型请求和产物登记逻辑。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon run create RUN_ID --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
```

同一 task 的多次独立 rollout 在一个 run 内声明：

```bash
.venvs/runner/bin/dycon run create RUN_ID --operation rollout --repeat-count 5 \
  --task-id INSTANCE_ID
```

`repeat-count` 是每个 task 的独立采样总数；同一 run 的 task×repetition 进入平坦队列，使用独立可复现 seed。基础设施重试不增加 repetition 数。

后三条命令会执行真实 rollout，需要可用的 GPU、模型、镜像和配置；status/show 不启动 workload。`wait` 等待进程退出事件，不轮询运行状态。

## 配置与证据

根 `config.yaml` 保存共用参数，generation、agent step/token limit 和 timeout 在根配置维护。每个 run 创建时冻结有效配置、tasks、selection 和 hash；执行、resume 及绑定 run 的分析使用原快照，不重新选择任务。`--set` 只能在相应入口明确允许的创建/规划阶段使用。

`catalog/manifest.json` 仅保存 benchmark/task 清单（schema 2）；rollout、patch 和 evaluation 的事实保存在 `runs/RUN_ID/evidence.json`。task、attempt、canonical rollout、patch SHA 和评测证据共同对齐。Submitted 不等于 resolved；未评测、unresolved 和 infra_error 不补造成功剩余量标签。token 重建证据不冒充 native exact IDs。

rollout/harness/hidden/enrichment 的运行产物在 `runs/RUN_ID/`；difficulty 的 datasets、splits、checkpoints、predictions 和 evaluations 独立保存在 `difficulty/`。action 的最小分析单元在 `action/outputs/ANALYSIS_ID/`，共享 hidden 视图在 `action/datasets/INDEX_ID/`；两者不依赖 analysis run，action 单元可组合多个来源 run，并冻结各自轨迹选择和证据。NAS 大张量通过 locator/receipt 引用；capture spool 在 ACK 前保留。`artifact verify`、Catalog verify、环境 gold 和模型性能评测各有范围，不能互相替代。

## 代码职责

| 目录 | 职责 |
|---|---|
| `core/`、`application/` | strict 配置/锁/I/O、CLI、RunPlan、supervisor、状态与阶段服务 |
| `catalog/` | benchmark/task 定义、版本与任务分区；运行证据由 `application/execution/evidence.py` 管理 |
| `rollout/`、`benchmark_evaluation/` | agent/model/parallel、canonical 产物、官方 harness |
| `enrichment/`、`hidden/` | prefix/token 标签、在线 capture 证据、独立 eager replay、NAS reader/recovery |
| `difficulty/`、`action/` | 数据索引、估计器与推理评估、分支绘图/一致性、canonical action 分类 |
| `runtime/`、`locks/`、`vendor/` | 容器/NAS/serving、依赖锁、固定推理实现 |
| `diagnostics/`、`migration/` | 诊断与显式一次性转换工具；不绕过正式入口 |

runner、analysis、easysteer、latent-vllm 是四个独立环境，按任务使用。模块细节见 [difficulty/README.md](difficulty/README.md)、[action/README.md](action/README.md) 和 [协作约定](AGENTS.md)。

## 历史与复现

`baseline/` 已清空，历史产物在 `isolated/history-baseline-20261021/`；仅作历史归档；本次只将 `migration-closeonly331-20261021` 的登记信息转入其 run 级证据，旧快照不可伪装成新版可恢复运行状态。原件和转换审计位于工程外 `/home/wangtianyi/dycon-isolation-20261021/baseline-cleanup/`。

本次精简前的全部 docs 原文、技术说明与实验/验收报告保存于 `/home/wangtianyi/dycon-isolation-20261021/docs-consolidation/`；日常只维护上面三份指南。历史指标仅说明当时的模型、精度、输入和运行范围，当前能力以源码和不可变 run 证据为准。

Harness ????? `invalid_patch`?`infra_error` ? `ConnectionError`???????????????????? attempt ???Managed rollout ?????? GPU TP=4 ?? MIG UUID ?????? TP=1??? [?????????](docs/ENTRYPOINTS.md)?

## Rollout seed retry

New runs freeze `execution.limits_retry` from the root config. Whole-trajectory
`LimitsExceeded` and `reasoning_loop` outcomes retain their evidence and requeue the
same sample with a deterministic new seed. Infrastructure retries keep their seed
and cumulative budget. Existing run snapshots and completed samples are unchanged.
See [the rollout retry guide](docs/MINIMUM-RUN-CN.md)
for capture recovery, counters, final selection, cancellation, and sampling caveats.
