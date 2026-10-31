# Action 熵动态筛选：实施与 CPU 调试报告

日期：2026-10-27（Asia/Shanghai）。远程工程：`/home/wangtianyi/DyConGlobal`。

## 交付状态

计划要求的功能已实施并同步到远程工作树，完成不占 GPU 的单元、mock/native-receipt 集成及全仓 CPU 回归。**未启动 GPU run、managed-serving check-only、模型服务或 GPU 效果对照。** 全仓仍有6项修改前即可复现的既有失败，不能称为全仓全绿；本次功能直接相关测试无失败。

使用说明见 [每个 Run 的独立策略](ACTION-ENTROPY-RUN-POLICY-CN.md)，设计与阶段状态见 [实施计划](ACTION-ENTROPY-DYNAMIC-ESTIMATOR-PLAN-CN.md)。

## 已实施内容

- Action identity 完整沿用 `core/action_identity.py::canonical_action`，reasoning 不参与分类；该文件字节 SHA 与实施前一致。
- `consensus.representative_policy` 支持 `estimator_score` 和 `min_step_tokens`。缺省保持 score/branch ID；最短策略按当前 step 的 native 输出长度、score、branch ID 决胜，不改变获胜 action 组的投票规则。
- 新 `parallel_enthophy` 基于 estimator 已评分、loop escape 已定案、top-k 截断之前的 canonical action 分布计算原始 Shannon 熵（nats）。A/B/C 三映射全部实现，并与两种代表策略独立组合。
- 新 `--parallel-config PATH` 接入同一正式 create/clone 入口。输入与有效策略按 run 独立冻结；新参数不进入根 YAML；旧 run 缺省行为与旧配置/hash/receipt 不回写。
- 最短策略在选择前调用共享 native receipt 准备阶段，提交时复核 request、ledger、output SHA 与长度，不使用字符计数或伪造缺失长度。score 策略不增加该前置等待。
- coordinator、loop escape 可行性、native 决策提交、启动选择验收、artifact verify 及 action-branch-stats 接入同一选择规则。compact receipt 保留重放输入，分析输出可按规则/代表策略分组。
- 离线 serving-memory/concurrency 资源修订兼容新 run 策略。选择策略的有效 SHA 与可独立修订的 admission/timeout 资源项分离；完整 config hash 仍按既有修订链更新。

主要代码：`core/parallel_policy.py`、`rollout/entropy.py`、`rollout/consensus.py`、`rollout/decision/coordinator.py`、`runtime/serving/capture.py`，以及现有 CLI/run planner、loop escape、验证/分析模块的定点修改。新增测试为 `tests/test_parallel_entropy.py`、`tests/test_run_parallel_policy.py`、`tests/test_parallel_entropy_integration.py`。

## 验证结果

全部 pytest 调试设置 `CUDA_VISIBLE_DEVICES=''`；全仓测试另设置 `MPLBACKEND=Agg`、`OMP_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、离线 Hugging Face 环境变量。测试通过 mock 生成与 CPU native-receipt 控制器验证行为，不创建生产 rollout run。

| 检查 | 结果 |
|---|---|
| 修改前相关基线 | 173 passed、1 skipped |
| 本次新增的三份测试 | **91 passed** |
| 最终全仓结果中直接相关的10份测试 | **264 passed、1 skipped、0 failed** |
| 最终完整 CPU pytest | **1077 passed、48 skipped、6 failed**，共1131个 testcase；68.39秒 |
| 对6个失败使用修改前源码隔离复测 | 相同6项、相同失败原因全部复现；0.62秒 |
| 新增模块/测试 Ruff | 通过 |
| 修改涉及的核心模块 compileall | 通过 |
| 正式 `dycon config-check` | 通过，未启动模型/GPU |
| 正式 `dycon run create --help` | 显示新增 `--parallel-config` |

48项 skipped 的具体原因保留在 JUnit 中，不当作通过。完整 CPU 测试命令未排除这6项失败，也未将其改成 xfail 或 skip。

覆盖内容包括8种 fixed/dynamic×代表策略组合，三公式数值示例与整数边界、canonical 身份一致性、同分排名、未知/冲突长度、full/compact receipt 重放与篡改、native 证据前移、Mini/OpenHands adapter、跨step loop escape及候选恢复、per-run隔离、输入文件后续变化、create/clone/切回固定比例、根参数误放拒绝、旧快照兼容、离线资源修订、artifact策略绑定及混合旧/新分析CSV。另补充6个动态模式格式纠错回归用例，确保无可投票 action 时仍保留 Mini 原有 FormatError 纠错路径，不将纠错分支误作 action winner。

## 既有失败清单

这6项均已使用实施前保存的源码，在工程 `temp/` 内隔离复现；未通过回滚生产工作树来验证，也未重启现有服务。

| 测试 | 修改前与修改后的相同失败原因 |
|---|---|
| `test_action.py::test_direct_cli_no_enrichment_immutable_and_tamper_detection` | action-classify 的现有 SkillsBench/OpenHands capability gate 抛 ValueError |
| `test_action_visualization.py::test_cross_run_same_rollout_id_cli_and_frozen_evidence` | 同上，CLI capability gate 拒绝 action analysis |
| `test_partial_publication.py::test_named_publication_flows_through_index_and_resolved_dataset` | resolved dataset 读取 task revision，现有测试输入缺少该键，KeyError |
| `test_predictor_publish.py::test_parallel_plugin_drains_before_score_results` | 对 vllm_plugin 源码特定字符串/注释的断言不匹配当前源码 |
| `test_ridge_expanded.py::test_inherited_split_rejects_cross_partition_hash_and_changed_task` | 预期异常文案含 base_commit，实际为 Task identity differs from parent split |
| `test_ridge_probe.py::test_tampered_selection_and_non_submission_fail` | 预期旧 submitted-canonical 文案，实际为 successful benchmark terminal trajectory 文案 |

以上保留为独立待处理问题，本次没有修改无关的 action 后端能力、difficulty 数据身份逻辑或旧测试断言。

## 源码及证据保护

实施前已有大量未提交变更。保存了工作树基线，每次发布以原始或上次发布 SHA 做并发修改检查，避免覆盖其他修改。未提交 git commit，未切换现有工作树，未改生产 run 或模型权重。

确认实施前后相同的文件包括：

- 根 `config.yaml`：`99ebe063bf199381645cb5ebe23c908f8d76e4a82fbd23f8bddde11f3f8e0d62`。
- `core/action_identity.py`：`50d5701f1a62011ef25a42cf6fa07449d56b9f6cee3d9fccc34db3c49b878d45`。
- `README.md`、原 `tests/conftest.py` 同样保持原字节内容。

远程 JUnit：`temp/action_entropy_dynamic_estimator_impl_20261027/cpu-suite.xml`。修改前隔离源码位于同任务 `temp/` 下的 `baseline-source/`。本地任务目录 `action_entropy_dynamic_estimator_impl_20261027/` 保存 baseline/work、发布 SHA、各次测试完整输出、`evidence/cpu-suite.xml` 与 `evidence/verification-summary.json`。临时文件均在任务目录或远程项目 `temp/` 内。

## 尚未验证的范围

真实 vLLM/GPU native 输出、FULL graph、TP 服务和高并发负载下的端到端行为尚未通过本次真实GPU实验验证；不以 CPU mock 成绩代替该验收。未测得真实 resolved rate、GPU token throughput、时延或总成本收益。用户后续需要且资源允许时，再按正式新 run / check-only / rollout 入口验证。
