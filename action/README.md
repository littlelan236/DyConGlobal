# Action 分析与共享索引

Action 分为两个互补层：action-classify 直接读取 canonical rollout trajectory 并生成互斥 action 类别；action-index 用于复用 difficulty hidden 样本的共享索引。两者都不重复提取 hidden，也不执行 trajectory 中的 command。

## 轨迹 action 分类

完整规则和产物见 [TAXONOMY-CN.md](TAXONOMY-CN.md)。

    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --rollout-id ROLLOUT_ID
    .venvs/analysis/bin/dycon action-classify ACTION_ANALYSIS_ID --rollout-list ROLLOUTS.json
    .venvs/analysis/bin/dycon action-check ACTION_ANALYSIS_ID

该入口按 config 或显式 rollout 清单选择 manifest 轨迹，验证 trajectory SHA，直接解析 assistant/tool-call 记录。它不依赖 enrichment、token trace、hidden 或 GPU；已有统一入口时不得另建临时读取/分类脚本。输出在 action/outputs/ACTION_ANALYSIS_ID/，同一 ID 不覆盖。

## Hidden 共享索引

    .venvs/analysis/bin/dycon action-index SHARED_INDEX_ID ACTION_INDEX_ID

core.cli 调用 action.datasets.index.create_action_index，读取 difficulty/datasets/SHARED_INDEX_ID/index.json，记录源索引 SHA，写入 action/datasets/ACTION_INDEX_ID/index.json。它不是 trajectory action 分类的前置步骤。

action 复用已有 hidden 与 task 分区，不重复提取 latent，不向当前 prefix 泄漏未来动作。当前 taxonomy 只分析可观察 shell/tool 行为；action 训练、taxonomy 学习和动作评测尚未实现。完整任务链见 [ENTRYPOINTS.md](../docs/ENTRYPOINTS.md)。
