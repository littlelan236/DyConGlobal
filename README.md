# DyConGlobal

DyConGlobal 管理 coding-agent benchmark 的任务选择、rollout、评分、证据采集和离线分析，并可将具备原生 token 证据的历史轨迹用于 RL 训练。

## 快速开始

~~~bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon --help
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
~~~

从项目根目录使用对应环境下的 `dycon`。新 run 在创建时冻结配置与任务，启动后才会占用模型、GPU 和容器资源。首次运行请先按运行手册检查配置和资源。

| 指南 | 范围 |
| --- | --- |
| [运行手册](docs/MINIMUM-RUN-CN.md) | 创建、启动、监控、恢复、比较和审计 run |
| [Benchmark 与容器](docs/BENCHMARKS-CN.md) | 各 benchmark 的环境、镜像、评分和 verifier |
| [采集与发布](docs/CAPTURE-CN.md) | 在线 hidden、raw logprobs、NAS 发布和证据恢复 |
| [离线分析](docs/ANALYSIS-CN.md) | Action、difficulty、steering 和轨迹对照 |
| [RL 训练](RL/README.md) | LoRA、GRPO、恢复和诊断 |
| [RL 执行与资源](RL/EXECUTION.md) | 全参 FSDP2、FLA、多卡执行和内存策略 |
| [RL 数据与概率](RL/dataset/README.md) | 原生 token 数据集、概率重放与封存、训练数据准备 |

## 工程结构

- `core/` 与 `application/` 提供统一 CLI、配置、执行和证据登记。
- `catalog/` 保存任务定义与 benchmark 分区。各 run 的轨迹、补丁和评分由该 run 的证据文件绑定。
- `rollout/`、`benchmark_evaluation/`、`enrichment/`、`hidden/` 分别处理执行、评测、轨迹增强和证据提取。
- `action/`、`difficulty/`、`steering/` 提供离线分析。大写 `RL/` 包含训练和 RL 数据处理。
- `runs/RUN_ID/` 保存冻结配置、计划、物理 attempts、当前采纳结果和回执。术语见 [GLOSSARY.md](GLOSSARY.md)，贡献约定见 [AGENTS.md](AGENTS.md)。
