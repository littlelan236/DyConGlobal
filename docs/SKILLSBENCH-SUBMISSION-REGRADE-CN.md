# 使用新 verifier 重评历史提交

`skillsbench regrade-submissions` 从指定只读审计报告中选择 VerifierError/Review，以独立冻结 run 的新任务变体、profile 和 dependency snapshot，在真实任务容器中运行原 verifier。它不请求模型、不运行 oracle，不修改源 run 的奖励、投影或结算。

```bash
dycon run create TARGET --operation rollout --from-run VERIFIED_DEFAULT_ENTRY
dycon skillsbench regrade-submissions TARGET --source-run SOURCE \
  --audit-report temp/skillsbench-verifier-repair-20261110/historical-audit-final/samples.jsonl \
  --output runs/TARGET/validation/regrade-plan --plan-only
dycon skillsbench job-start --seconds 21600 JOB -- \
  regrade-submissions TARGET --source-run SOURCE --audit-report AUDIT \
  --output runs/TARGET/validation/regrade-results --workers 3
dycon skillsbench job-status JOB
```

目标与源必须为不同冻结 run。目标任务变体的来源 package hash、原镜像、subject 及归档字节、当前历史 sample/attempt 投影都必须匹配。输出必须为目标 run 内新的 validation 子目录。并发上限为 4，沿用现有 sandbox 与 verifier 的预算和清理接口，不执行 `rollout.run()` 或 `solve()`。

历史 subject 的 `re_evaluation=false` 仍然保留：文件导出不是完整 live state。本接口只重放逐项认证的声明提交，并明确记录这个范围。目录恢复包含有界归档扫描、路径与链接检查、恢复前后文件哈希核对，不以初始镜像文件补齐未导出内容。归档内 `/workspace/skills -> /skills` 只引用原只读技能挂载。其余链接只能落在声明输出目录内。

syzkaller 历史导出的是空 `/app`，而 verifier 消费 `/opt/syzkaller`，因此返回 `RegradeUnavailable`。SimPO 历史未导出源码；重评容器移除初始 SimPO 源码，禁止使用初始代码冒充历史提交。若新 verifier 在缺失/错误输出的前置 gate 停止，可得到有完整证据的有效零分；若它进入必须重跑源码的阶段，该样例返回 `RegradeUnavailable`，新分数为空。

`ValidOutcome` 包括有效零分和部分分。`VerifierError` 表示新运行仍遭遇评测故障，`Review` 表示共享分类器尚无充分证据，`RegradeExecutionError` 表示重放/清理过程失败，不能当作 agent 零分。`complete=true` 只表示全部选中样例均已处理且历史源文件未变，不表示全部可重放或 verifier 全部通过。

目录内保存计划、逐样例恢复清单、原 subject 哈希、verifier stdout/CTRF/阶段日志、共享分类 Decision、结果 JSONL、历史保存证明及汇总。再次执行必须使用新目录，旧结果不覆盖。

## 显式覆盖历史评分

`regrade-submissions ... --replace-original-with-backup` 在全部样例处理完成、原始来源哈希未变且源 run 已退出并清理后，对 ValidOutcome（含有效零分和部分分）自动回写 canonical 评分。沿用既有事务备份/回滚接口，保存 replacement-manifest.json、replacement-state.json、backup/ 和逐样例 replacement-summary.json，同步 evaluation raw/report、attempt、repetition result、结算回执、evidence 和 progress，并刷新 status-summary。原 subject、轨迹、capture、原 verifier 运行日志及旧独立审计报告保留；新评分明确绑定新任务包、依赖、冻结 run、恢复范围及评测证据。

VerifierError、Review、RegradeUnavailable 和重放/清理失败只记录本次重评结果，不以无效数值覆盖旧奖励；replacement-summary.json 列明所有未覆盖样例。未传该选项的默认行为仍为只读历史来源。`--plan-only` 只检查覆盖前提，不写历史结果。
