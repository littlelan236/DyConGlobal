# enrichment 与 hidden 产物迁移记录（2026-10-17）

已将旧 tactRefractor 中已完成的 enrichment 与 hidden 提取产物迁移到 DyConGlobal。迁移只接收源轨迹 SHA-256 存在于当前 `baseline/manifest.json` 的记录；本次 191 个 task 全部匹配，未发现指向已清理轨迹的产物。

## 迁移结果

- enrichment：191 个 task，191 个 JSONL 产物，15,559 个旧工程状态行，约 8.4 MB。
- hidden：191 个 task，每个包含 `.npz`、`decisions.json`、`extraction.json`，共约 10.8 GB。
- source trajectory SHA-256：191/191 匹配当前 baseline manifest。
- 当前 Catalog：1,104 条轨迹、258 个 task，revision 1237，校验通过。

## 锚点与对应关系

`enrichment/manifest.json` 和 `hidden/manifest.json` 都明确声明：

```json
{"name": "think_close", "text": "</think>"}
```

这里的语义是当前 reasoning round 的 `</think>` close marker；hidden manifest 同时记录 `site=close_token_as_input`。每个 task 条目均包含 `rollout_id`、当前 `trajectory_path`、`trajectory_sha256`、`steps`、`annotation_steps` 以及每个产物文件的 SHA-256。enrichment JSONL 的每行还校验了 `instance_id` 和 `source_sha256`；hidden 的 decisions/extraction/npz 文件以同一 task 文件名绑定。

迁移产物位于：

- `enrichment/outputs/tact-hidden-close-20261017/`
- `hidden/outputs/tact-hidden-close-20261017/`
- `enrichment/manifest.json`
- `hidden/manifest.json`

迁移脚本为 `enrichment/scripts/migrate_products.py`。它拒绝重复 manifest、缺失产物、轨迹哈希不匹配和 enrichment 行的 task/hash 不一致。原始 tactRefractor 数据未修改。当前项目的 enrichment/hidden 运行逻辑仍只读取 DyConGlobal 自己的 manifest 和 baseline 轨迹。

旧工程 hidden 元数据显示 Qwen3.5 模型、`hf-hidden-v1` site schema 和 tokenizer token boundary `[248069]`；该模型信息已随 hidden manifest 保留。模型权重指纹仍应在后续正式分析前按当前 checkpoint 重新核验。
