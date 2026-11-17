# SWE-bench 验证有效性审计

通过正式 CLI `.venvs/runner/bin/dycon harness audit RUN_ID --output ABSOLUTE_NEW_DIRECTORY [--labels JSON]` 对既有物理 attempt 进行只读审计。输出位于历史 runs 之外，拒绝覆盖既有目录。分类器与在线正式结算共用。原始官方 report 不变；新评测的有效 resolved/reward 在落盘与训练消费前按审计投影。

当前策略：`swebench_verifier_audit_v3`。输出保留官方结果，并区分：

- `ValidOutcome`：评分证据有效，`effective_resolved` 为官方布尔值。
- `VerifierError`：该次评测不能提供可信训练标签，`effective_resolved=null`。`cause_confirmed=true` 才表示具体根因已经证实。
- `Review`：输入身份、文件完整性、证据归属或普通模型提交责任仍未核实；不生成有效零分。

## Gold 控制策略

Gold 模式必须核对冻结任务的 reference patch、harness 输入 patch 摘要；有官方 `patch.diff` 时还须逐字相同。仅有 run 名称或 gold 标记不足以启用该策略。文件/身份损坏仍归 Review。

身份有效的 gold 控制返回 false 或未完成，归 VerifierError。若无法确认原因，使用 `gold_control_failed_unattributed` 或按实际观察细分为 `gold_dependency_api_failure`、`gold_runtime_recursion_failure`、`gold_runtime_warning_failure`、`gold_native_process_crash`、`gold_test_collection_failed`、`gold_evaluation_timeout`、`gold_required_test_evidence_missing` 等；此时 `attribution=unknown`、`evidence_strength=control_failure`、`diagnostic_status=needs_investigation`。它表达控制评测失效，不证明依赖/运行时已经被确认不兼容。具体异常候选与缺失证据继续保留。

只有人工排查提供与当前证据摘要绑定的 `gold_patch_defect` finding，明确 `attribution=gold_patch`、confirmed、未恢复、basis 和 supporting_sources，且实际为无其他验证故障的完整断言失败，才允许 gold 的有效 false。此类判断不能自动用于其他 attempt 或不同日志。普通模型提交的正常断言失败仍是有效 false；不会因为分数为零而自动成为 VerifierError。

## 解析与执行细节

pytest 仓库嵌套测试的输出可能使官方 parser 把 `[100%]`、`[`、`[100%]----` 等进度文本登记为测试 ID。这些已知形式保留在官方覆盖记录，另列 `coverage.parser_artifact_ids`，不作为真实测试冲突。其他 ID 和真实 node ID 冲突照常核查。

Pydicom 的官方 parser 会把绝对路径参数缩为最后一个路径分量。冻结 required ID 未做同样转换时，记录精确的 `parameter_identity_mismatches`，不通过模糊匹配或更改分数掩盖它。

Docker 的 stdout/stderr 分块合并可能把有效测试回执附到结束标记之后。只有原始 output SHA256 和 test command SHA256 均绑定时，才允许从完整输出读取唯一回执；重复、缺失、摘要错误仍拒绝。pytest、tox 和 shell 最终 returncode 保持各自语义。

官方 reward=1 仍需真实评分证据。确切必需测试声明缺失 mandatory tool 而 skipped 的 gold，归 `missing_verifier_tool`；不得将全量 skipped 的成功退出当成全部测试通过。历史 Django-15521 缺 xgettext 的 63 个 skipped 就属于这种情况。修复并通过新 run 验证后，才可能得到有效正例。

## 证据与版本迁移

原始 official report、selection、patch 和既有 run 产物不修改。审阅 finding 必须绑定当前 evidence_digest 及实际 source hashes；升级 collector 后旧 finding 不会自动继续生效。需要显式重核旧摘要与所有来源，再生成新标签和迁移回执。历史未知根因不会因改变策略而变成已确认根因。

本次策略由 classifier/collector/adapter 回归与历史 gold 回放验证。控制评测分类可剔除无效标签，但不代表环境修复或 gold patch 全量通过。

## 正式评分结算与训练消费

新 SWE-bench harness 默认使用 `swebench_validity_settlement_v1`；`verifier_audit_mode=settlement`，不再只是旁路。ValidOutcome 产生 completed、布尔 resolved、reward=0/1；VerifierError/Review 使用同名 status，resolved/effective_resolved/reward 均为 null，scoring_eligible=false。原执行字段保存在 raw_outcome，原官方结果保存在 official_result，官方 report.json/test_output.txt 不改写。审计文件引用包含 SHA256。

`RunEvidence.record_evaluation` 将有效投影登记到唯一 evidence，并验证 null 与状态约束。在线成功剩余量标签、difficulty resolved 选择、steering/RL source selection 通过同一 training_label 消费所选 evaluation；不会把旧 sidecar 的无效结果当作成功。旧 SWE-bench 无审计结果可读取，但不为新成功标签提供有效 bool。增量评测继承的未结算旧结果变为 legacy_unsettled_evaluation 的 Review，历史文件不重写。

新 RL 数据构建将明确 VerifierError/Review/基础设施错误轨迹的整个分组标为不可训练，保留分支、原 historical_score、native token 和 terminal_scoring_exclusion；general audit.enabled=false 也执行此评分限制。GroupBatch 还有消费端阻止绕过的校验。尚未评测的旧共识数据保持可表达；共识 historical_score 与终局 reward 是不同字段，不把其中任意一个代替另一个。已有 sealed 数据集、prepared/checkpoint、运行中训练和其冻结优势不追溯改写。

普通提交空 patch 可确认是有效零分；有精确 patch hash、官方 git syntax rejection 且尚未执行测试的损坏 diff 也可归有效零分。泛化 hunk mismatch 不能据此认定模型责任，保留 Review。重试决策读取 raw 执行事实，后续成功不会继承前次 VerifierError/Review。

已部署代码的 CPU 集成测试覆盖真实 collector/parser、harness、evidence、训练选择和 RL dataset/group 边界；模型/GPU训练质量与环境修复的验收另行记录。
