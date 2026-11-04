# Capture 操作与证据

## 生命周期

在线 capture 在 rollout 的模型请求链中运行，并绑定冻结 run、sample、attempt、rollout 与 session 身份。Capture finalize 产生标签、coverage、索引和 publication receipt；大张量可以放在 `paths.nas_root` 指向的 NAS。run 目录保留 locator、hash 和 receipt，使下游流程可以验证远端数据。

普通 rollout、capture rollout、steering rollout，以及 steering+capture rollout 均使用统一 `dycon rollout`/run 生命周期。通过受支持的根配置选择开关，并在创建 run 时冻结；不要为 capture 单独启动 worker 或 launcher。

## 检查和恢复

```bash
.venvs/runner/bin/dycon capture status RUN_ID
.venvs/runner/bin/dycon capture recover RUN_ID
.venvs/runner/bin/dycon artifact verify RUN_ID
.venvs/analysis/bin/dycon online-index RUN_ID INDEX_ID
```

先检查 capture 状态和已登记 publication。`capture recover` 恢复已有 spool/finalization/upload，不会重新调用模型。只在确认原进程结束且 recovery 状态可解释后继续 run；保留未 ACK spool，直到 publication receipt 确认数据已安全发布。

capture 的控制预算或 receipt 并发若需修改已有 run，必须使用显式的 capture control revision 入口，填写 revision ID 和原因。它生成审计记录并保留旧快照；不能直接编辑 run snapshot 或重写根配置来伪装历史执行条件。

## 标签与下游分析

Capture publication 只证明指定 hidden 数据按其 receipt 发布；它不证明 rollout 成功或 patch 解决任务。在线索引和标签需绑定所选 trajectory、评测及 patch。未评测、unresolved 和基础设施失败的成功剩余量标签保持 null。

Parallel capture 使用 decision、branch、request 和 sample 身份关联候选。选中 winner 的标签不能传播给未选分支；未生成 close、close 已生成但 hidden 缺失、发布失败应分别统计。重分类只能基于已 finalized 的 native token evidence，通过专用入口写独立修订证据。

可复用的 sealed native session 可显式建立 trace index；选用的 session、run 和发布物由 receipt 冻结。不要把局部隐藏样本数相等当作 capture 完整性验收。

## 证据边界

- Exact native token IDs 与 tokenizer 重建结果必须分开标识。
- Receipt 可验证登记文件与 hash；外盘上的 tensor 内容需按相应 verify 命令显式检查。
- Artifact verify、capture coverage、环境 gold 和模型质量评测回答不同问题。
- capture 恢复、capture 重分类和主动重跑是不同操作；只有最后一项会重新请求模型。

完整子命令和 revision 参数见 `dycon capture --help`、`dycon capture-reclassify --help` 及 `dycon run revise-capture-control --help`。
