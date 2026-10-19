# 当前实现与验收边界

核对日期：2026-10-19。依据远程实际工作树，含未提交修改。任务启动见 [ENTRYPOINTS.md](ENTRYPOINTS.md)。单层 Ridge 实验已有专项 CLI 和参数发布入口，见 [ridge-submission-pilot.md](ridge-submission-pilot.md)；下文训练入口限制针对通用训练 API。

## 数据与配置观察

2026-10-19 11:22（北京时间）只读观察到 `baseline/manifest.json` 的 `catalog_revision=1788`：458 tasks、1463 rollouts、30 evaluations；task 全为 train，resolved=true/false/null 分别为 321/298/844。这是当时 catalog 原始条目数，不是去重后可训练、未隔离或验收通过的样本数；后续进度会改变它们。

HEAD 为 `ae3167daf2e819be4c93f43c048108102dd9803b`，工作树已有大量未提交修改和新增文件。本页依据实际代码，不能只按提交版本判断能力。

当前 YAML：Qwen3.5-9B、FP16、managed serving、GPU 0–3/TP=4、qwen3_xml 工具解析、capture=true、steering=false。capture anchors 为 reasoning_newline/think_close，layers=0..31，prefill chunk=0，NAS 发布启用。hidden adapter 为独立 vllm_eager。这描述配置，不代表本次已启动或验收任务。

## 已实现

| 范围 | 当前代码能力 |
|---|---|
| 配置/证据/资源 | strict YAML/schema、原子 I/O、进程锁、不可变 run snapshot、磁盘预算、模型身份 |
| 数据与 rollout | 显式导入、manifest 分区/用途、canonical 轨迹/patch、并发 rollout/resume、真实启动 gate |
| 评测 | SWE-bench 4.1.0、patch 绑定、success/infra error 区分、gold 只验环境 |
| 在线 capture | EasySteer control plane、native request ledger/token IDs、anchor/event 绑定、bounded spool、异步 NAS shard、上传恢复 |
| enrichment | open/close prefix、成功剩余量/终止诊断、证据分级、token trace、online token-event finalize |
| hidden | 独立 vLLM eager 重放、显式 site、常驻 TP queue worker、APC、receipt/partial 校验、同身份下续提 |
| 数据视图 | schema 1 离线索引、schema 2 exact join、schema 3 在线索引、无标签 action 引用 |
| difficulty | Ridge、FFN MLP、Ridge+FFN；训练/加载 Python API；显式预测评分和轨迹图 |
| runtime | NAS OCI streaming、load slots/lease/空间预约、owned cleanup/recovery、独立 EasySteer runtime |

## 限制与未完成项

- `difficulty.estimator`、`action.taxonomy` 仍为 null。submission-remainder 单层 Ridge 网格已有 `difficulty-ridge` CLI、冻结 task 划分、参数和 receipt/验证报告，见 [实验指南](ridge-submission-pilot.md)；通用 FFN/残差仍无完整训练 CLI、checkpoint 保存/恢复和发布契约。FFN 当前 CPU 执行。
- schema 3 loader 使用 task partition，当前 catalog 无 val/test；schema 1 loader 按 rollout ID 分组约 80%/20%，避免同 rollout 泄漏，但同 task 多轨迹仍可能跨组，不能称 task 级划分。
- `difficulty-evaluate` 依赖 schema 1 label_shard，不直接支持 schema 3；token-join 是 rows.jsonl 审计表，不等同于普通/在线 index.json。
- action 仅有共享索引，taxonomy、labels、analyzers、训练/评测未定义；动态 difficulty steering/control 未实现。
- `runtime-check` 要求全局 wheelhouse，当前为 null，不能代表真实任务环境是否就绪；以正式 rollout gate 和 task overrides 为准。
- capture 默认 record_and_continue，agent 可完成而 capture 仍 partial/pending_upload。恢复发布不重跑模型，也不补足所有 coverage 或失败后处理。
- serving manager 当前请求 `FULL_AND_PIECEWISE`；`capture.graph_mode=FULL` 不代表所有 prefill/mixed 均走 FULL graph，须查该 run 实际证据。

## 标签和验收

resolved=true 才提供“到最终成功提交”的剩余量标签；unresolved、未评测、infra error 保留 null。旧轨迹 token 重建不能提升为 native exact；exact join 必须唯一绑定 prompt/output IDs、token ID 和 absolute position。

测试涵盖 manifest 并发/分区、patch/evaluation 绑定、失败标签、prefix 无未来动作、hidden/resume、token alignment、trace/spool/capture、镜像并发/恢复、agent 协议及 estimator 数值契约。2026-10-19 的 vLLM GPU 证据还覆盖了四个在线轨迹事件的 NAS hidden 对照，以及 5,280/6,275 token 的 APC hot-cache smoke。Docker smoke 只验证四个合成 OCI 镜像，不替代 SWE-bench gold；GPU 验收证据仅证明其模型、dtype、输入和 runtime 范围，不能外推全量数值一致性或吞吐。

本页只描述当前实现与验收边界；冲突时以当前源码和 run 证据为准。
