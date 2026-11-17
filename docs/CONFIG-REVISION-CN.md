# 统一 run 配置修订

正式入口为 `.venvs/runner/bin/dycon run revise-config`。可以修订根配置 schema 和 run 并行策略 schema 中的所有可配置字段，不再按字段白名单增加子命令。原有八个 `revise-*` 专用修订命令及其 Python API 已移除。

配置值必须满足完整 schema、跨字段约束和 backend/资源引用校验。`run_id`、selection、execution_context、哈希等派生记录不是配置输入，接口会自动重建。`parallel.max_inflight_branches` 根据 `serving.max_num_seqs` 派生，修改后者即可。

## 预览与应用

先预览需要修改的任意字段组合：

```bash
.venvs/runner/bin/dycon run revise-config RUN_ID \
  --revision-id memory16g \
  --set sandbox.memory_mb=16384 \
  --set generation.temperature=0.4 \
  --reason 'Increase container memory and adjust sampling' --dry-run
```

移除 `--dry-run` 应用修订。`--set` 放在 `revise-config` 后，可重复传入；值按 YAML 解析，支持布尔、null、列表和字典。也可以使用 `--patch changes.yaml` 传入配置字段的嵌套映射；未指定的字段保留。重复或父子冲突的修改路径会被拒绝，拼错的字段由 schema 报错。

预览结果包含当前和候选 config hash 及字段差异。应用时可添加 `--expect-config-hash PREVIEW_OLD_HASH`，防止预览后其他修订使输入发生变化。接口只读目标 run 的快照，不把根配置当前值混入该 run，也不会修改根配置。

应用前必须通过正式 cancel/wait 停止执行并确认退出与 `cleanup_complete=true`。预览可在 run 活动时进行。修订不启动执行；随后通过正式 `run resume` 继续。普通修订保留已有结果，终态失败仍须通过 `run retry-failed` 明确选择重跑。

## 身份字段重新规划

seed、parallel enabled/branch_count、limits retry 种子策略、benchmark/任务快照/partition、agent backend/profile、模型路径或身份变化会影响样本身份或执行契约，也允许修改，但需要显式 `--replan`：

```bash
.venvs/runner/bin/dycon run revise-config RUN_ID \
  --revision-id newseed --set generation.seed=101 \
  --replan --reason 'Start new samples under a new seed' --dry-run
```

应用后，原运行状态目录及结果移至 `config-revisions/ID/archived-state/`，原配置、selection、任务和 work-items 保存在 `before/`；新状态按修订后的配置及正式任务选择/采样逻辑生成，progress 从待运行开始。执行历史、日志和事件保留。旧归档中的路径字段按原样保存，可通过归档目录对应原 run 相对路径查找旧文件。新物理 attempt 编号超过该 task/repetition 的旧编号下界，避免复用旧 ID。重复 replanning 也继续保留此编号下界。

改变 benchmark 或任务选择参数时会重新调用正式 catalog 选择逻辑；仅修改 seed 时保留原选中任务。已经归档的结果不参与新执行的 accepted 统计。修改资源、生成温度或时限则通常不需要重新规划，也可显式选择 `--replan` 开始新的采样。

## 审计与恢复

每次修订保存 `request.json`、`before/`、`after/`、`progress.json` 和 `receipt.json`，并写入 `events/config-revision-ID.json`。相同 revision ID 和相同输入在没有后续修订时幂等；不同输入不能复用该 ID。

可捕获的写入异常会恢复原快照和已归档状态。进程中断留下未完成事务时，使用同一入口恢复：

```bash
.venvs/runner/bin/dycon run revise-config RUN_ID \
  --revision-id REVISION_ID --recover --reason 'Recover interrupted revision'
```

恢复会检查 run 已停止，回滚未提交事务；已提交但最后状态写入中断时，核验提交文件后完成状态记录。存在未完成事务时，正常 load/start/resume 会被阻止，必须先恢复。应用前后均保留配置及 plan 哈希校验。后续 supervisor 的每个 execution attempt 会在 `configuration/` 保存该次使用的配置、计划、selection 和 work-items，避免用最新配置解释旧执行。

## 内存与存储

`sandbox.memory_mb` 的单位是 MiB，16 GiB 对应 16384；它是全局默认值，已有 profile 的任务专属内存覆盖仍优先。内存上限不是预分配。

修订 `paths.artifact_roots` 可改变后续产物位置；已有 run 的物理目录由 `run_storage_root` 继续绑定，不在配置事务中搬迁大量历史数据。文件搬迁应使用对应存储维护接口。


仅增加已停止 run 的 `capture.export.spool_max_bytes` 时，修订只改变该 run
后续 writer 的冻结配额，不搬迁共享 spool，也不修改其他 run 的限制；因此不要求
其他 run 的 capture writer 停止。目标 run 仍须退出且清理完成。减少配额、修改其他
export 字段或使用 `--replan` 仍保留原有项目 writer 停止检查。
