# Capture 操作与证据

## Parallel 可视化发布

启用持久化 `think_close` capture 且 `parallel.branch_count > 1` 时，run 在
producer、writer、serving 和 backend 停止后，按最终 accepted canonical
selection 执行独立 pointwise 审计和自动 publication。运行结束摘要中的
`purpose=visualization` 和 `eligible_for_training=false` 标识该制品只能用于
可视化；不创建训练 index、dataset、split、checkpoint 或 predictor。原始
session、decision、rollout 和历史 publication 均不回写。

`canonical_coverage` 描述实际 winner 路径，`all_candidate_coverage` 描述所有
候选。候选 inventory 没有完整 receipt 时 `expected_count_complete=false`，摘要
保守显示 partial。canonical hidden、native truth、候选 hidden 和 stored score
分别保存；native response 缺失只截断不可证明的前缀 token truth，已验证 hidden
仍可预测。无 close、缺 hidden、缺 native receipt 与候选未封存分别记录。
只有原 branch receipt 绑定的 stored score 才用于绘图；其余 decision 保留审计证据。

```bash
.venvs/runner/bin/dycon capture status RUN_ID
.venvs/runner/bin/dycon capture finalize RUN_ID
.venvs/analysis/bin/dycon difficulty-plot-parallel RUN_ID PLOT_ID --rollout-id ROLLOUT_ID --publication-id PUBLICATION_ID
.venvs/analysis/bin/dycon difficulty-plot-trajectories RUN_ID PLOT_ID --rollout-id ROLLOUT_ID --publication-id PUBLICATION_ID --attempt-id EXISTING_FIT_ATTEMPT --target submission_remaining_reasoning_tokens_log1p
```

轨迹图复用已存在且模型、layer、target 匹配的 checkpoint；无兼容 checkpoint 时
不能生成预测曲线，分支 stored-score 图仍可单独使用。轨迹图的 truth 是实际
agent 终点 remainder，缺失处保持 null，不能解释为成功标签。可选 `--repair-id`
会与 publication 所绑定的 repair 核对；不指定时由 receipt 自动定位。

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

完整子命令和 revision 参数见 `dycon capture --help`、`dycon capture-reclassify --help` 及 `dycon run revise-config --help`。例如同一次修订可传 `--set capture.control_seconds=120 --set capture.receipt_workers=16`，详见[配置修订](CONFIG-REVISION-CN.md)。
停止的单个 run 可使用 `dycon capture archive RUN_ID --archive-id ID --destination DIRECTORY` 归档到指定目录；省略 `--destination` 时使用 run 快照中的 NAS 根目录。归档会校验文件并释放该 run 的 spool reservation，原路径保留指向归档的软链接；如需再次运行该 run，须先恢复本地 spool。


## Capture 诊断摘要（只读）

启用 capture 的物理 attempt 会在 runs/<RUN_ID>/rollout/tasks/<INSTANCE_ID>/repetitions/<REPETITION_ID>/attempts/<PHYSICAL_ATTEMPT_ID>/capture-diagnostics.json 保存派生摘要。session 的 finalization-diagnostics.json 记录阶段汇总；控制 API 失败时，sessions/<SESSION_ID>/control-errors/<CORRELATION_ID>.json 保存异常类型、调用栈帧、阶段耗时及该 session 的队列前后快照。run 级 capture-finalization-error.json 记录 coverage 数量和受限的缺失点诊断；完整 missing 坐标仍以 coverage.json 为准。

stages 按 expected、selector_seen、capture_row、submit、persist、indexed 统计。missing 只包含受限的异常点及 request、position、stage、reason_code；truncated_count 表示超出上限的点数。正常 attempt 只保留聚合计数，队列快照只包含目标 session。HTTP 错误摘要最多 512 字节并过滤敏感字段。诊断不保存 prompt、模型输出、tensor、凭据或完整 token 数组。诊断文件可能包含 run/session/request 标识，应按普通运行证据控制访问。

只读查看示例：

    python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:d.get(k) for k in ("stages","by_anchor","missing_count","truncated_count","missing")}, ensure_ascii=False, indent=2))' runs/<RUN_ID>/rollout/tasks/<INSTANCE_ID>/repetitions/<REPETITION_ID>/attempts/<PHYSICAL_ATTEMPT_ID>/capture-diagnostics.json
    .venvs/runner/bin/dycon capture status <RUN_ID>

Session/物理 attempt 的 `indexed` 在 online finalizer 执行前为 null，`indexed_scope=pending_online_finalizer`；queued sample 不能当作 indexed。最终 accepted index 的阶段摘要位于 `enrichment/online/capture-diagnostics.json`。`selector_evidence_unavailable` 表示观察器失效或证据不足，不能解释为 selector 未命中。

截断保留完整聚合计数，逐点证据按 execution attempt 共用 256 个异常身份预算；session、物理 attempt 和 run 级重复视图复用同一点的名额，控制错误也占用预算。预算账本位于 `execution/attempts/<EXECUTION_ATTEMPT_ID>/capture-diagnostic-budget.json`，仅在错误或收尾时加文件锁并原子写入，不进入逐 token 热路径。若诊断存储故障，业务结果保持原规则，诊断证据可标记 unavailable。

Engine flush 的异常证据保存在 capture export/spool 根目录的 `control-errors/<CORRELATION_ID>.json`，响应头 `X-DyCon-Correlation-ID` 与客户端异常摘要关联；它只保存异常类型、受限堆栈位置与耗时，不保存异常 locals、源码行或请求正文。


## Run 自动正式发布（非并行 think_close）

Run 在所有 worker、serving、capture writer 与 backend 退出后，对正式最终选择的
accepted 且 submitted canonical result 批量做 pointwise integrity audit，再正式发布。
accepted/unresolved 也保留用于终点可视化；resolved-only 仍由下游资格策略控制。
SkillsBench 等 workspace/live-state benchmark 绑定正式 subject，`submission_status=not_applicable`
不会被 patch 的 submitted 条件排除；subject 仍须通过既有身份与内容校验。
重试替换的旧 attempt、quarantined 与未 submitted 结果不混入默认 publication。
该过程不调用模型，不重放、不重新 token 化，也不改变 disposition、exit_status、reward 或评测。

`partial` 是成功发布的稀疏覆盖。缺 hidden 但 native tokens 齐全时，已观察点可保留完整的
native 剩余量标签；缺 native response 时，只有最后一个 unknown canonical response
之后的安全 suffix 可发布标签。missing 坐标与 unknown cutoff 都保留，不能插值、补零或连线跨越缺口。
只有校验通过的 native_exact 点被 publication 显式授权为 exact hidden；诊断修复本身不授予训练资格。
没有安全点的 session 写审计排除原因，不生成空训练样本。其他 anchor 显式报告
unsupported；parallel 使用本文上方所述的独立 visualization-only publication。

产物关系（每次 selection 对应稳定的新 ID，不回写历史 online 目录）：

```text
runs/RUN_ID/
  capture-publication.json                   # 独立发布摘要和稳定 index_id/正式命令
  hidden/capture-repairs/REPAIR_ID/
    publication-selection.json               # 最终 canonical selection、config/source hash
    publication-selection-receipt.json
    selection.json, source-proofs.json, report.json, progress.json, session-reports/
  hidden/capture-publications/PUBLICATION_ID/ # 派生 finalization 审计
  enrichment/online-publications/PUBLICATION_ID/
    publication.json, coverage.json, index.json, requests.json, excluded.json
    point-exclusions.json, receipt.json
NAS/hidden/online/RUN_ID/capture-repairs/REPAIR_ID/
  sessions/SESSION_ID/{point-index,coverage,receipt}.json
  report.json, source-proofs.json, receipt.json
difficulty/datasets/INDEX_ID/{index,receipt}.json
```

`receipt.json` 是原子正式发布完成后才能读取的证据，单独存在的 stage 目录或旧
`enrichment/online/receipt.json` 不能证明完整 coverage。完整和 partial 自动 publication
均写新 ID 路径；同 selection 可幂等重试，不同 selection 派生新 ID。中断 stage 保留为
`.interrupted-*`；已有完成 publication/index 的 hash 冲突会拒绝覆盖。

```bash
.venvs/runner/bin/dycon capture status RUN_ID
.venvs/runner/bin/dycon capture finalize RUN_ID
# 从 capture-publication.json 的 index_command 取稳定 ID：
.venvs/analysis/bin/dycon online-index RUN_ID INDEX_ID --publication-id PUBLICATION_ID
.venvs/analysis/bin/dycon difficulty-dataset resolved DATASET_ID --source-run RUN_ID --index-id INDEX_ID
```

`capture finalize` 是正式 run-end 发布的无模型重试入口，持有唯一 run lock，先确认没有
producer；原 run 的 accepted/evaluation 不变。`capture recover` 处理未 ACK spool、上传和
原 session finalization；`capture repair --pointwise` 对留存证据独立审计。自动服务仅在
producer 退出后复用 spool ACK recovery，并保持原 session receipt 不变。

摘要状态为 pending、complete、partial、pending_recovery、unsupported、failed；与 run
stage 独立。摘要记录 repair/publication/index ID、selection hash、完整/部分/排除 session
数、expected/observed/missing、unbound/rejected、阶段耗时及失败阶段。
自动服务记录稳定 `index_id` 和 `index_command`，由正式 online-index 命令物化；不创建
用户命名的训练 dataset、split、checkpoint 或 predictor。

`capture.publication_seconds` 可为新 run 配置批量收尾预算，默认 3600 秒；旧快照缺此字段
也采用 3600 秒，不回写快照。在阶段和 session 边界检查预算；单次 NAS 上传仍由冻结的
`capture.export.upload_max_elapsed_seconds` 限制。中断或预算耗尽后使用同一 finalize
命令继续。该预算不是对不可中断内核/NFS I/O 的强制终止保证。


## accepted partial capture 的显式重跑

正式入口 run retry-failed 支持 --accepted-capture-partial，选择当前 disposition 为
accepted 且 capture_status 恰为 partial 的 sample。与 --exit-status TimeoutError
一起使用时取并集；默认覆盖整个 run，可用 --sample-key 精确限制，并用
--expect-attempt SAMPLE_KEY=ATTEMPT_ID 防止终态游标变化后选错。

    .venvs/runner/bin/dycon run retry-failed RUN_ID --request-id REQUEST_ID --accepted-capture-partial --exit-status TimeoutError --reason REASON

登记要求原 execution 已退出并有完整清理回执，保留旧 attempt、capture、发布物及
重跑授权审计，使样本进入 Pending。后续 run resume 为原 sample 创建一个新的物理
attempt，保留原 seed round；accepted 且 capture complete 不属于该新增选择条件。
已有 partial 发布物不会因登记重跑而被删除，也不被当作新 attempt 的完整 capture。
先检查已有 capture 恢复和 upload ACK；重跑会重新调用模型，与 capture recover 的
无模型调用恢复不同。

独立采样次数扩展使用 run add-repetitions --count N：向每个冻结 task 追加 N 个
repetition，保留旧样本与 capture 证据；与上述重跑可以在同一次 resume 中调度。
完整预览、哈希校验及恢复用法见 [最小运行指南](MINIMUM-RUN-CN.md#追加-repetition-与重跑-accepted-partial-capture)。
