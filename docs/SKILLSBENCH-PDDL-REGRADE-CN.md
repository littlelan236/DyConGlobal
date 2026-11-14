# Airport PDDL 完整 verifier 重评与原回执替换

仅适用于 `pddl-airport-planning`，要求任务修复版为 `pddl-airport-plan-v1`。
通用 SkillsBench workspace replay 仍不受支持；该任务的 verifier 仅依赖冻结
PDDL domain/problem 和提交的两份纯文本计划，因此可完整恢复其评测输入。

```bash
.venvs/runner/bin/dycon skillsbench regrade-pddl RUN_ID REGRADE_ID \
  --snapshot catalog/snapshots/CORRECTED/manifest.json \
  --samples temp/TASK/samples.json --workers 2 \
  --replace-original-with-backup
```

缺少最后一项时仅执行重评，不改写原回执；`--apply-only` 只应用同 REGRADE_ID 下
已经全部成功的完整 verifier 结果，并核对 snapshot 和 samples 哈希。
REGRADE_ID 不可重用；失败后需要新的 ID。workers 范围 1–4。

正式入口复用 load_plan、当前 sample 投影、受限归档读取、任务包/修复收据校验、
SkillsBenchFactory、NAS 镜像租赁、OwnedPlanes、BenchFlow 生命周期、hardening、
原 test.sh 和既有 verifier 证据采集。不会启动模型或执行 oracle solve；
`agent=oracle` 仅作为 BenchFlow 无模型的环境准备模式，显式调用
setup/start/install_agent/verify/cleanup，不调用 run。

每条任务使用独立带 project/run/task/sample/attempt/epoch 标签的容器。
只恢复两份原提交计划；提交归档的 problem.json/domain/problem 必须和冻结任务一致，
容器内计划及输入也须逐项 SHA256 匹配。运行完整原 test.sh（含 apt/uv 安装和 pytest），
要求两个 CTRF 测试全部通过、两份计划均打印 VALID、reward=1、无 verifier/agent 异常，
且清理回执成功、无该 epoch 残留容器。任一失败，整批不替换。

结果保存于 `runs/RUN_ID/regrades/REGRADE_ID/`。在容器启动前保存 before-replay
原始回执；真正替换前再按文件保存逐字节 backup 和 replacement-manifest，
记录 before/after SHA256。替换异常时回滚已经改写的文件。
replacement-state.json 标记 prepared/applied/rolled_back。

原地替换需用户明确授权；它是普通历史 attempt 不可变规则的显式维护例外。
命令取得 run/supervisor/validation 锁并要求退出与 cleanup_complete 回执。
替换范围是指定物理 attempt 的 evaluation-raw、evaluation-report、attempt.json，
repetition/result.json、对应 artifacts_committed/settled 及证据索引/progress。
同步 reward、resolved、回执哈希、来源说明和 finalization receipt hash，保留原 evaluation ID。
新 verifier 的任务包身份单独记录，不改写 run 的冻结任务/配置或提交/轨迹/capture。
status-summary 通过原刷新接口更新。

旧 verifier 日志作为旧运行事实保留；原地更新后的评测回执通过 regrading 元数据
明确指向新完整容器执行日志和结果，不能再把旧日志当成新评测日志。
历史审计报告、已有导出/数据集仍描述各自原始快照，不随该维护操作自动更新。
需要从新结果创建数据集时应重新使用正式来源冻结入口。

`backup` 文件可按 replacement-manifest 定位，均保留原始字节。
硬中断后先核对 replacement-state、当前文件哈希和容器归属，再决定维护恢复，
不能无条件重新执行替换或覆盖已有 backup。
