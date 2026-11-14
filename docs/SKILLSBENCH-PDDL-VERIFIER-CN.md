# SkillsBench Airport PDDL verifier 修复与语义复评

`pddl-airport-plan-v1` 修正 `pddl-airport-planning` 的示例/解析器语法冲突。
它同时接受一行一个的标准 `(action arg1 arg2)` 与任务示例的
`action(arg1, arg2)`，仅做完整行语法转换，再交给原版本
`unified_planning==1.3.0` 的 PDDLReader 和 PlanValidator 检查动作、参数、
前置条件和最终目标。不求解、不执行提交代码、不降低语义通过条件。
支持 PDDL 注释和空白行；空计划也必须由语义校验确认目标已满足。

修复去除无效计划诊断对未定义 `plan` 的引用，并显式要求 `VALID` 状态。
1.3.0 的原 `if val` 已正确重载布尔转换，不能将其描述为发现的误通过漏洞。
原 verifier/test.sh、依赖 pin、二元 reward 和 task.md 均保留。

## 发布和选择

```bash
.venvs/runner/bin/dycon skillsbench verifier-variant NEW_SNAPSHOT \
  --source-snapshot catalog/snapshots/SOURCE/manifest.json
```

此入口复用任务快照、package manifest、verify_package 和 task_package_path。
只允许源 verifier SHA256
`1f8c6cd4be1ed7fdf0925494643dd2d78848c9ec4c855a7a7837612675ac87e4`，
修复内容来自 `catalog/verifier_patches/pddl_airport_test_outputs.py`。
新快照保留其他任务及原索引策略，增加修复来源/收据，只有 airport verifier 的
一个文件可变。原任务包、源快照、旧 run 和 profile 不变。
新发布物已建立，不自动切换根配置或活动 run；后续新 run 需显式选择新 snapshot_ref。
修复模板继续作为校验身份的一部分，修改模板需要新的显式修复版本。

## 历史提交的单独语义复评

完整 workspace replay 尚无通用保证；`harness run` 仍不支持将文件导出冒充完整状态。
因此新增的正式入口明确只验证 airport 计划文本的语义，不声称完整容器重评：

```bash
.venvs/runner/bin/dycon skillsbench revalidate-pddl RUN_ID \
  --snapshot catalog/snapshots/NEW_SNAPSHOT/manifest.json \
  --samples temp/TASK/selection.json \
  --output analysis/NEW_REVALIDATION \
  --validator-python .venvs/pddl-verifier-1.3.0/bin/python --timeout 60
```

selection.json 显式包含 run_id、task_id、samples；每条 sample 需 sample_key、
repetition_id、attempt_id、archive_sha256。入口使用冻结 run 的统一计划与当前 sample
投影，拒绝身份不符、重复样本、归档哈希不符或提交 PDDL 输入与冻结任务不一致。
复用受限只读归档接口，不提取/执行归档代码；仅将两份计划文本写到新分析目录。
worker 在显式项目隔离环境中加载可信 verifier，固定 unified_planning==1.3.0，
记录完整依赖版本，并受逐样本超时约束。

results.jsonl/CSV、summary.json、每条 request/result/record/log 以及前后来源哈希
单独保存；输出目录必须全新并位于项目 analysis/。失败和超时单列 RevalidationError，
原 reward 保留在 original_reward，revalidated_reward 仅指此次语义检查的二元结果。
不调用 planner/oracle、模型或容器，不更新历史 accepted/resolved/reward/capture。
旧审计 Review/Unresolved 与新语义结果属于不同评测契约，应分别保留。

2026-11-10 已发布 `skillsbench-agent75-pddl-plan-v1-20261110`（75 tasks，1 个修复 task）。
目标历史 run 的 14 份提交共 28 份计划均 VALID；输入哈希匹配，来源前后无变化。
结果位于 `analysis/pddl-airport-revalidation-20261110-01/`。
这证明所存计划在冻结任务输入下有效；不将该结果冒充原历史评测成功或全容器重放。
