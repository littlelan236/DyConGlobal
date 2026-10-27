# 修复 Capture 的正式在线标签发布

在 analysis 环境运行统一入口：

```bash
.venvs/analysis/bin/dycon capture-publish RUN_ID \
  --repair-id REPAIR_ID --publication-id PUBLICATION_ID --resolved-only
```

该命令使用 run 的冻结配置，持有 run 排他锁，并读取已经完成的 pointwise repair。
它不调用模型，不修改原始 tensor、repair receipt、run snapshot 或原有 session finalization。
新派生绑定存入 NAS 的 `capture-finalizations/PUBLICATION_ID/`。

`--resolved-only` 只接纳 source run 中 completed/resolved=true 且精确 patch SHA 与 harness report 一致的 Submitted 轨迹。
默认发布策略为 `complete_native_sessions_v1`：要求每个 canonical response 都具有唯一、完整、哈希匹配的 native token 凭证，
所有预期 close hidden 都存在。partial、identity-only、native 凭证缺失、没有完整 capture 的轨迹被明确排除；不补造 token 或标签。

标签计算复用 `enrichment.online_trace.finalize_token_events`，保持既有 reasoning-block suffix 语义。
输入的 repair/session/ledger/tensor/shard-index/trajectory/harness 凭证均核验；首次成功后原子发布至
`runs/RUN_ID/enrichment/online/`。`publication.json` 列出候选、接纳和排除轨迹及原因；
`receipt.json` 绑定派生标签、来源、发布身份和完整性。
相同 ID 的成功调用可校验复用；不同来源/策略不能覆盖已有 online 产物，失败 stage 保留待核对。

发布完成后继续使用原有接口：

```bash
.venvs/analysis/bin/dycon online-index RUN_ID INDEX_ID
.venvs/analysis/bin/dycon difficulty-dataset resolved DATASET_ID --source-run RUN_ID --index-id INDEX_ID
.venvs/analysis/bin/dycon difficulty-split DATASET_ID SPLIT_ID --strategy event_random_v1 --seed 42 --validation-fraction 0.15 --test-fraction 0.15
.venvs/analysis/bin/dycon difficulty-train --training-config TRAINING_YAML
```

event_random_v1 允许同 task 及同 rollout 的不同样本跨集合，不能解释为 task holdout。
被完整凭证策略排除的成功轨迹不属于最终训练/测试总体。

## 部分采集轨迹的按点发布

使用显式策略 `available_native_suffix_v1`，复用同一发布服务和精确 token 标签计算：

```bash
.venvs/analysis/bin/dycon capture-publish RUN_ID --repair-id REPAIR_ID --publication-id NEW_PUBLICATION_ID --resolved-only --coverage-policy available_native_suffix_v1
.venvs/analysis/bin/dycon online-index RUN_ID NEW_INDEX_ID --publication-id NEW_PUBLICATION_ID
.venvs/analysis/bin/dycon difficulty-dataset resolved NEW_DATASET_ID --source-run RUN_ID --index-id NEW_INDEX_ID
```

新版本发布保存在 `runs/RUN_ID/enrichment/online-publications/NEW_PUBLICATION_ID/`，旧 `enrichment/online/`、旧 dataset、split 和 checkpoint 不改写。命名 index 显式绑定 `publication_id` 与 `label_root`；dataset 验证两者指向同一 run 的对应发布。

- 仅缺 hidden：保留已有精确点，剩余 token 标签仍包括没有 hidden 的后续响应。
- 缺 native 响应凭证：只接纳最后一个无法唯一核验的 canonical response 之后、native 凭证完整的后缀；不把未知 token 数填成 0。
- FormatError 等非 action 模型响应也产生 reasoning，缺凭证时同样阻断较早点的标签。
- identity-only、orphan 和无法核验当前 token 的点不升级为 exact；哈希、canonical 或模型冲突仍失败。
- hidden 不补造、不插值。`coverage.json` 保留已绑定请求范围内的 missing，`expected_count_scope` 不声称覆盖未知请求。
- `publication.json` 记录 partial 轨迹、缺口边界和后缀缺 hidden 数；`point-exclusions.json` 记录排除原因；censored 标签继续由 dataset 排除。

扩充 dataset 后按相同 seed 重新划分，新旧测试集通常不同，不能仅根据 R² 数值变化断言模型改善。
