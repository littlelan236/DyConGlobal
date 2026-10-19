# 文档索引

维护日期：2026-10-19。实际工作树和不可变 run 证据优先于说明文档；本目录只保留当前入口、实现约定和可复用的数据接口。

## 当前指南

| 文档 | 内容 |
|---|---|
| [工程 README](../README.md) | 目标、代码/产物职责、数据路径和验证边界 |
| [ENTRYPOINTS.md](ENTRYPOINTS.md) | 全部 CLI、环境、配置、rollout/capture/steering、恢复、队列与维护入口 |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | 当前实现、数据观察和未完成项 |
| [INPUTS.md](INPUTS.md) | 显式任务/轨迹/评测导入和预测格式 |
| [online-capture-fullgraph.md](online-capture-fullgraph.md) | 在线 anchor、cache、token ledger、spool 和 NAS 发布 |
| [vllm-eager-extraction.md](vllm-eager-extraction.md) | 独立 eager runtime、site 映射、离线 replay 与恢复 |
| [ridge-submission-pilot.md](ridge-submission-pilot.md) | 单层 Ridge 参数实验入口、数据核验、拟合产物和验收边界 |
| [difficulty/README.md](../difficulty/README.md) | estimator/loader API、目标变换、索引和划分限制 |
| [action/README.md](../action/README.md) | trajectory action 分类、共享索引、receipt 和未实现边界 |
| [AGENTS.md](../AGENTS.md) | 协作约定，已有统一入口时禁止临时脚本绕行 |
| [GLOSSARY.md](../GLOSSARY.md) | 工程术语 |

日期化实验汇报、临时故障记录、迁移记录和历史方案已从 `docs/` 清理。复跑任务前请以入口表、源码参数、唯一配置和对应 run 证据为准。
