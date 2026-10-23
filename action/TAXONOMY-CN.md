# Action 分类与轨迹分析

维护日期：2026-10-21。本模块从显式 source run 的 evidence.json 冻结 canonical trajectory bindings；产物独立保存在项目级 action/outputs/ANALYSIS_ID/，支持同一分析组合多个 run。不要求 enrichment、token trace、hidden tensor 或模型服务。

## 统一入口

从工程根目录执行：

    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN --rollout-id ROLLOUT_ID
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN --rollout-list /path/to/rollouts.json
    .venvs/analysis/bin/dycon action-check ACTION_ANALYSIS_ID

已有统一入口时不要创建临时脚本、复制 trajectory 读取逻辑或直接执行轨迹中的命令。action-classify 只解析原始 command 字符串，绝不会执行它们；它从 source run 的 evidence 选择 rollout，验证 trajectory SHA，再原子写入新的项目级 action/outputs/ACTION_ANALYSIS_ID。同一 ID 不覆盖已有产物。

轨迹选择先给出 `--source-run RUN_ID`，再用 `--rollout-id` 或 `--rollout-list` 选取该 run 内的成员；不指定轨迹时选择该 run 内全部可用轨迹。rollout-list 为 ID 或含 `rollout_id`、可选 `trajectory_sha256`、`cohort` 的 JSON 列表。不存在跨 run 的隐式查询或全局 baseline profile。

## 统计单位与唯一标签

一个 assistant message 是一个 action observation unit。历史数据中一个 assistant message 通常对应一次 bash tool call；多个 tool call、非 bash tool 或 malformed arguments 统一标为 OTHER，并写入 uncertainty。一个 bash 调用内部可以有多个 shell command，但只产生一个 label；同时保留 components、rules 和 compound。

主类别固定优先级：

    DELIVER > EDIT > ENV > RUN > CLEAN > INSPECT > OTHER

| 类别 | 目的 | 可观察判别 |
|---|---|---|
| INSPECT | 获取信息而不改变持久状态 | 读文件、搜索、查看 diff、git status/log/show、版本查询 |
| EDIT | 修改代码、测试、配置、补丁或工作树 | 写重定向/heredoc、Python 写文件、sed -i、patch、cp/mv/touch/mkdir、git restore/checkout |
| ENV | 准备依赖或运行环境 | pip/uv/apt/conda 安装或升级、权限和工具链准备 |
| RUN | 执行程序或验证 | Python、pytest、pylint、mypy、ruff check、诊断/构建执行 |
| CLEAN | 删除临时产物或清理状态 | rm/rmdir/unlink、git clean |
| DELIVER | 形成或提交最终结果 | submission marker、git diff 重定向到 patch、git add/commit/format-patch |
| OTHER | 无法安全静态判定 | shell AST 错误、工具形态未知、动态副作用未支持 |

RUN 有意合并复现和验证：历史 rollout 中二者会使用相同的 Python/pytest 命令，命令本身不足以互斥区分它们。网络请求、find -delete/-exec 等潜在副作用也进入 OTHER，而不是误归为只读或修改。

复合命令仍只有一个主类别。例如写入脚本后运行脚本归 EDIT；安装依赖后测试归 ENV；删除临时文件后测试归 RUN；marker 加 patch 输出归 DELIVER。具体子命令保留在 rules。

## 阶段定义

对每条 trajectory 的 step s、总 step 数 T 使用：

    phase_bin = min(4, floor(5 * s / max(1, T - 1)))

这是事后位置描述，不是在线分类输入，也不能表示模型当时知道未来。0 为前段，4 为末段。

## 产物

action/outputs/ACTION_ANALYSIS_ID/包含：

- analysis.json：source runs、轨迹选择、taxonomy、有效配置及实现身份。
- selection.json / source-evidence/：run evidence revision/SHA、冻结子集、trajectory bindings、duplicate aliases。
- resolved-config.json：本次分析使用的配置快照。
- labels.jsonl：每个 assistant step 一行，唯一主标签。
- summary.json：类别、阶段、cohort、compound 和 uncertainty 统计。
- examples.json：各类代表命令，供人工抽查。
- code/ / receipt.json：源码快照、配置、输入选择和产物 SHA；action-check 重新验证全部内容。
- transition.json / plots/：阶段占比与相邻 action 转移矩阵、默认图及独立 view receipt，详见 README。

同一 canonical trajectory 被多个 rollout 副本引用时，selection 会保留一个 source 并记录 aliases；不同 task 绑定相同 bytes 会直接拒绝。action 分析不改变 Catalog 或来源 run，也不修改已有 action-index 样本视图。

## 历史基线

action/provenance/taxonomy-20261019/保存早期审计的 snapshot summary、source bindings 和 rollout 清单。它是历史证据，不是新入口。当前正式分析以显式 source run 的冻结 evidence 和新的 analysis ID 为准。
