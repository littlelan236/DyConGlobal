# Action 共享索引与预留能力

2026-10-19 按实际代码更新。正式入口：

```bash
.venvs/analysis/bin/dycon action-index SHARED_INDEX_ID ACTION_INDEX_ID
```

`core.cli` 调用 `action.datasets.index.create_action_index`，读取 `difficulty/datasets/SHARED_INDEX_ID/index.json`，记录源索引 SHA，写入 `action/datasets/ACTION_INDEX_ID/index.json`。目标已存在时拒绝覆盖。

当前只建立 sample_id 共享视图：taxonomy_version、action_label、label_source 均为 null；taxonomy、analyzers、action 训练/评测未实现。普通离线和在线发布 index.json 可作为来源；token-join 的 rows.jsonl 不能直接传入。

action 复用已有 hidden 与 task 分区，不重复提取 latent，不向当前 prefix 泄漏未来动作。已有统一入口时禁止另建临时索引/提取脚本。后续定义预测方向与 taxonomy 时，应扩展正式入口和版本化标签契约。完整任务链见 [ENTRYPOINTS.md](../docs/ENTRYPOINTS.md)。
