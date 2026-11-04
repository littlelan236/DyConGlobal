# Benchmark 操作指南

所有 benchmark 都通过统一 run、harness 和 artifact 入口执行。benchmark 子命令负责任务导入、profile 检查、镜像准备和专用 preflight；实际 rollout 与结果登记仍属于正式 run。精确参数以 `dycon swebench|skillsbench|webarena|harbor --help` 为准。

## 通用流程

1. 使用 benchmark 导入入口准备任务定义，并通过 `catalog-check` 校验清单。
2. 使用固定 profile 与镜像入口准备运行环境；核对镜像身份和所选任务范围。
3. 运行对应 benchmark 的 preflight。任务容器、依赖和 verifier 检查不能由镜像存在性检查替代。
4. 创建 run 并冻结 tasks、profile、模型和资源配置，然后通过 `run start/wait/status` 管理 rollout。
5. 通过 `harness prepare/run/status` 对固定 rollout/patch 进行评分。

## SWE-bench 与 OpenHands

使用注册的 SWE-bench/OpenHands profile、任务定义和镜像。patch 与 canonical rollout 通过 run evidence 绑定；并行/capture 设置必须在创建 run 前按项目配置约定启用。运行前做 profile、容器和 harness preflight。不要直接启动 OpenHands 或内部 runner 来绕过 run 计划。

## SkillsBench 与 OpenHands

使用 `dycon skillsbench` 检查 benchmark 任务和 profile，使用 `skillsbench-images` 准备固定任务镜像，并在创建 run 前运行相应环境与任务 preflight。任务数据、verifier、网络依赖和 OpenHands ACP runtime 均由注册的 backend 适配；不应在任务容器内临时替换版本或凭一次 task smoke 推断全量数据可用。

若后续分析要求每步 `think_close` hidden，应在新 run 创建前设置并冻结显式 reasoning-close 生成配置和 capture anchor。语法字段、generation token budget、native close token 与已发布 hidden 必须分别核验；仅有 reasoning 文本或 capture 状态 complete 不代表每步 hidden 完整。具体接口使用 `dycon skillsbench audit --help` 和 [capture 指南](CAPTURE-CN.md)。旧 run 的配置不能通过改根 YAML 更新。

## WebArena Verified

使用 `dycon webarena` 导入/校验任务，并通过 `webarena-images` 为显式选择的任务准备共享站点镜像。Verified 任务与原版 WebArena 的数据和评分定义不能混用。冻结任务、站点服务与镜像版本后，使用统一 run preflight 和 harness 入口。浏览器、站点容器和离线评测服务需分别通过对应环境检查。

## Harbor backends

Harbor 集成通过 backend registry 绑定 benchmark、agent、runtime、subject 和 evaluator。使用注册的 Harbor CLI/profile 执行 SWE-bench Pro V2 或 Terminal-Bench 2.1，不要另建一套任务生命周期。对 OpenHands ACP transport、容器边界或 evaluator 做改动时，应通过对应 adapter 测试及目标 benchmark 的小规模 preflight 验证。

## Evidence and limits

任务 catalog、run rollout、patch 和 evaluation 是不同身份。Harness 报告绑定选定 patch；仅 `resolved=true` 表示该次选定评测成功。环境 gold、单任务 smoke、单元测试和完整 benchmark 运行分别回答不同问题，报告时不可互相替代。
