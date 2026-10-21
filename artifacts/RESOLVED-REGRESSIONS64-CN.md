# Baseline resolved、Parallel 未 resolved 的 11 个任务分析

本报告分析 31-task 全部任务加 36-task run 已完成的 33 个任务，共 64 个互不重叠 task。筛选条件是 Baseline `resolved=true` 且 Parallel `resolved!=true`，共 11 个任务。

## 分类

| 类别 | 数量 | 任务 |
|---|---:|---|
| 无 patch / MissingPatch | 3 | `sphinx-doc__sphinx-10673`, `pytest-dev__pytest-7490`, `sympy__sympy-24443` |
| patch 无法应用 / Only garbage | 6 | `django__django-11099`, `django__django-10880`, `pydata__xarray-4075`, `scikit-learn__scikit-learn-26323`, `sympy__sympy-12481`, `scikit-learn__scikit-learn-13142` |
| harness 连接重置 | 1 | `matplotlib__matplotlib-13989` |
| patch 应用成功但测试失败 | 1 | `sympy__sympy-15875` |

## 结论

主要共性是提交和评测链路失败，而不是 Parallel 代码解题普遍失败。11 个回归中，3 个没有 patch，6 个 patch 文件不是 unified diff，1 个遇到 harness 容器连接重置；这 10 个任务都没有形成可用于判断代码正确性的 harness 结果。只有 `sympy__sympy-15875` 是可归因于最终代码内容的真实失败。

6 个 `Only garbage` 任务的 patch 文件已逐字核对：内容是 Python 源码或函数片段，分别以 `import re`、模块 docstring、缩进函数体等开头，不含 `diff --git`、`---`、`+++`、`@@` 等 unified-diff 结构。官方 harness 在应用阶段失败，尚未运行测试。这是 final patch 提取/提交格式问题，不能解释成测试不通过。

3 个无 patch 任务中，`sphinx-doc__sphinx-10673` 和 `pytest-dev__pytest-7490` 是 `LimitsExceeded`，轨迹结束前没有提交 patch；`sympy__sympy-24443` 是 `ParallelSelectionError`，没有形成有效最终提交。

`matplotlib__matplotlib-13989` 的 patch 是有效 unified diff，但 harness 报告 `ConnectionError(... ConnectionResetError(104, 'Connection reset by peer'))`，没有进入可比较的代码评测，应标记为 infra error。

`sympy__sympy-15875` 的 patch 成功应用，官方 harness 完成但 `resolved=false`；唯一失败项是 `FAIL_TO_PASS: test_Add_is_zero`，其它 PASS_TO_PASS 测试通过。这是 11 个回归中唯一能确认的真实代码质量回归样本。

## 是否是 Ridge 选错分支

当前证据不能支持“11 个回归都是 Ridge 选错分支”。6 个 malformed patch 和 3 个 missing patch 的失败发生在最终提交/评测链路；未选分支没有独立的最终 patch artifact，无法证明某个未选分支会生成可应用且正确的 patch。候选 decision receipt 只记录分支 score、资格、completion 和 selected 标志。

相关候选通常都为 `eligible`，赢家按当前估计器的 minimum score 规则选出；该规则估计剩余 reasoning 难度，不包含 harness patch 可应用性或测试正确性标签。因此当前数据只能判断：11 个回归中 **10/11（90.91%）没有有效 harness 结果**，只有 **1/11（9.09%）** 是真实测试失败。

## 后续处理建议

1. final submission 增加 unified-diff 格式校验；空 patch 和源码片段在 rollout 收尾时显式标记为提交失败。
2. `MissingPatch`、`Only garbage`、`ConnectionResetError` 与真实测试失败分开统计和重试。
3. 对 `sympy__sympy-15875` 单独诊断 patch 内容和测试回归；不要用 10 个 infra/提交失败任务推断模型质量。

逐任务 trajectory、patch、harness log、SHA 和 evaluation outcome 保存在 `resolved-regressions64-evidence.json`；本分析未启动 rollout、harness 或 GPU。
