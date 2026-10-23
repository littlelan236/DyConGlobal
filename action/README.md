# Action 分析与共享索引

2026-10-21：action 新产物与 difficulty 一样，独立于任何 source run，统一写入项目级
action/outputs/ANALYSIS_ID/；共享 hidden 索引写入 action/datasets/INDEX_ID/。
无需建立 analysis run，来源 run 只提供轨迹与证据。即使配置来自全局 --run-id，产物位置也不改变。

## 最小操作

从项目根目录执行，analysis ID 必须全新。分类命令会原子生成 labels、统计和两张默认热力图：

    .venvs/analysis/bin/dycon action-classify ANALYSIS_ID --source-run RUN_A --rollout-id ROLLOUT_A
    .venvs/analysis/bin/dycon action-check ANALYSIS_ID
    .venvs/analysis/bin/dycon artifact verify ANALYSIS_ID --scope action-analysis
    .venvs/analysis/bin/dycon artifact show ANALYSIS_ID --scope action-analysis

同时分析多个 run：重复 --source-run，用 JSON binding 清单明确每条轨迹属于哪个 run。
下列字段名是严格校验的；trajectory_sha256 与 cohort 可省略，实际 SHA 总会冻结：

    [
      {"source_run_id":"RUN_A", "rollout_id":"R1", "trajectory_sha256":"ACTUAL_SHA", "cohort":"baseline"},
      {"source_run_id":"RUN_B", "rollout_id":"R1", "trajectory_sha256":"ACTUAL_SHA", "cohort":"parallel"}
    ]

    .venvs/analysis/bin/dycon action-classify ANALYSIS_ID --source-run RUN_A --source-run RUN_B --rollout-list bindings.json

省略筛选时冻结每个 source run 中全部 eligible 轨迹。单个 run 的 --rollout-list 兼容 JSON ID 列表；
多个 run 的清单必须含 source_run_id，--rollout-id 只接受单个 source run。
相同 run/rollout 重复请求拒绝；相同 bytes、同 task 的副本只计一次并记录 alias；
相同 bytes 绑定不同 task，或 model/cohort/classifier cwd 冲突时拒绝，避免不独立样本被重复统计。

## 最小分析单元

    action/outputs/ANALYSIS_ID/
      analysis.json / selection.json / resolved-config.json
      source-evidence/RUN_ID.json
      code/...                              # 分类/统计/绘图实现快照
      labels.jsonl / summary.json / transition.json / examples.json
      plots/progress-share.png
      plots/transition-probability.png
      plots/plots.json
      receipt.json
      plots/views/PLOT_ID/...                # 后增独立 view，不改 parent receipt

analysis.json 是本单元完整冻结配置，明确 source run、rollout ID、路径、SHA、分类 cwd、taxonomy、
配置/实现/package 身份和统计口径。labels 的唯一身份是 source_run_id + rollout_id + step_id。
source-evidence 保存选中的 run evidence 子集；后来 harness 更新原 evidence 不会改变旧分析，
但被绑定的 trajectory bytes 改变会使完整校验失败。读取不依赖实时 Catalog、enrichment 或 hidden。

## 热力图与不可变视图

progress-share 的列是相对进度五阶段，行是现有七类，默认 micro 为该阶段类别数/全部 action 数；
summary 同时保存按轨迹等权的 macro、计数与分母，以及各 source run/cohort 分组矩阵。
transition-probability 的行是当前 action、列是下一 action，按行归一化；不跨轨迹连接，末步没有下一步。
空阶段/空行用 null 和 N/A 表示，颜色范围固定 0–1。所有 assistant message 均是 observation unit，
无 bash、多个 tool call 等形态仍按既有 taxonomy 标 OTHER。

    # 默认图在分类时已产生；此命令只校验并返回它们，重复调用不改 receipt
    .venvs/analysis/bin/dycon action-plot ANALYSIS_ID
    # 新绘图配置使用独立 view ID，支持 macro 与 PNG/SVG
    .venvs/analysis/bin/dycon action-plot ANALYSIS_ID --plot-id macro-v1 --aggregation macro --format png --format svg

绘图只读冻结 labels/summary，不再打开来源 run 或 trajectory。view 自己保存 receipt，绑定 parent receipt SHA；
原分析 receipt 不包含后加 views（与 difficulty dataset/splits 的关系一致）。完整 action-check 会连同 views 校验。
修改参数必须使用新 analysis/view ID，不提供覆盖开关。实现快照被 hash 保护；当前安装代码与快照是否一致单独报告，
正常代码升级不会仅因版本不同就判定历史 receipt 损坏。

## 历史读取

旧 schema-1 产物保持原位；项目级旧产物按 ID 读取，run 内旧产物必须显式给路径，禁止隐式 fallback：

    .venvs/analysis/bin/dycon action-check runs/OLD_RUN/action/outputs/OLD_ID

不向旧目录补文件、不迁移、不重签 receipt；旧格式需重新分类为新 ID 才能生成新图。

Action 分为两个互补层：action-classify 直接读取 canonical rollout trajectory 并生成互斥 action 类别；action-index 用于复用 difficulty hidden 样本的共享索引。两者都不重复提取 hidden，也不执行 trajectory 中的 command。

## 轨迹 action 分类

完整规则和产物见 [TAXONOMY-CN.md](TAXONOMY-CN.md)。

    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN --rollout-id ROLLOUT_ID
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --source-run SOURCE_RUN --rollout-list ROLLOUTS.json
    .venvs/analysis/bin/dycon action-check ACTION_ANALYSIS_ID

该入口按显式 source run 和 rollout 清单选择 canonical 轨迹，验证 trajectory SHA，直接解析 assistant/tool-call 记录。它不依赖 enrichment、token trace、hidden 或 GPU；已有统一入口时不得另建临时读取/分类脚本。输出在项目级 action/outputs/ACTION_ANALYSIS_ID/，同一 ID 不覆盖。

## Hidden 共享索引

    .venvs/analysis/bin/dycon action-index SHARED_INDEX_ID ACTION_INDEX_ID

core.cli 调用 action.datasets.index.create_action_index，读取 difficulty/datasets/SHARED_INDEX_ID/index.json，记录源索引 SHA，写入 action/datasets/ACTION_INDEX_ID/index.json。它不是 trajectory action 分类的前置步骤。

action 复用已有 hidden 与 task 分区，不重复提取 latent，不向当前 prefix 泄漏未来动作。当前 taxonomy 只分析可观察 shell/tool 行为；action 训练、taxonomy 学习和动作评测尚未实现。完整任务链见 [ENTRYPOINTS.md](../docs/ENTRYPOINTS.md)。
