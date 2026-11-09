# Trajectory 成功次数与剩余工作量

Parallel run 的自动 visualization publication 可直接传给现有命令，无需创建
训练 dataset：

```bash
.venvs/analysis/bin/dycon difficulty-plot-trajectories RUN_ID PLOT_ID --rollout-id ROLLOUT_ID --publication-id PUBLICATION_ID --attempt-id EXISTING_FIT_ATTEMPT --target submission_remaining_reasoning_tokens_log1p
.venvs/analysis/bin/dycon difficulty-plot-parallel RUN_ID BRANCH_PLOT_ID --rollout-id ROLLOUT_ID --publication-id PUBLICATION_ID
```

轨迹预测仅使用经过 winner 和 native/tensor 校验的 actual-path hidden，truth
独立按 native suffix 计算。缺失值为 null 并断线，未评测或 unresolved 轨迹的
终点是实际 agent 终点。分支图展示 receipt 绑定的原始 scores，不重新评分，也不
将 score 当作 native truth；无兼容 checkpoint 时仍可单独生成分支图。

`difficulty-plot-trajectories` 将 run 中每个 task 的 resolved repetition 数量与指定 Ridge checkpoint 的逐步预测/实际剩余量并列展示。它不估计成功概率。

## 选择输入

默认数据集模式沿用冻结 difficulty dataset 的 resolved selection：

```bash
.venvs/analysis/bin/dycon difficulty-plot-trajectories SOURCE_RUN PLOT_ID \
  --attempt-id RIDGE_ATTEMPT --dataset-id DATASET_ID \
  --target submission_remaining_reasoning_tokens_log1p
```

也可显式传 `--rollout-id ID` 或 SHA-bound `--rollout-list FILE`，纳入指定的 canonical terminal rollout，包括未 resolved 的正常结束样本。显式选择不能与 `--dataset-id` 混用；它用于实际终点分析，不会声称失败轨迹具有成功标签。

指定 `--repair-id` 时，预测可读取已发布 pointwise repair 中通过 receipt、canonical trajectory、model、request/response 与 think_close 身份核验的 hidden。Native-exact 与 canonical-identity-only 预测绑定分别标记；后者不能冒充精确 token 证据。原始 NAS publication 与 repair 可按现有入口合并覆盖。`--reuse-predictions-id` 仅复用身份和 SHA 一致的已验证预测。

## 统计和曲线

- 最终成功次数按冻结的 task×repetition work item 和最终 canonical rollout 计算；缺少最终 result 的 repetition 单列 pending，不当作失败。
- 历史成功 attempt 可用于逐轨迹曲线，但与最终 repetition 成功数分开统计。completed/unresolved、未评测和 infrastructure error 也分别保留。
- 每个图以 canonical assistant step（0-based）为横轴。`think_close` 时当前 step action 尚未执行；剩余 steps 为 `T-s`。
- Reasoning-token 真值只来自已绑定 publication 的 native exact suffix。缺失/截尾未来 token 保持 unavailable，不从文本重建 exact token。
- 预测与真值独立保留：缺 hidden 时预测断开，缺完整 native 后缀时真值断开；不插值、补零、跨缺口连线或平滑。
- log1p checkpoint 预测先用 `expm1` 还原到目标原单位，再与原单位真值比较。图按轨迹展示，不跨轨迹平均或归一化长度。

曲线带有训练 split 与 `used_in_fit` 身份。如果同一 task 或轨迹样本参与 checkpoint 拟合，图只能解释为拟合诊断，不能当作 task holdout 泛化结果。

## 压缩事件标注

每条 canonical trajectory 的 ACP 事件流会检查明确的 condenser/compression 触发事件。图中以红色竖虚线标出对应 assistant step，HTML 同 task 的各 repetition 仍在同一行；`trajectories.csv` 保留事件索引、事件类型和 step。未记录触发事件时显示“未记录”，不会根据轨迹长度或配置阈值推测压缩发生。

## 产物和校验

PLOT_ID 在 `difficulty/evaluations/` 中保存 selection、summary、逐点 CSV、task 计数、独立 SVG 图和 HTML 报告；缺失覆盖以 unavailable/coverage 字段记录，不用零值替代。输出拒绝覆盖既有 ID。

```bash
.venvs/analysis/bin/dycon artifact verify PLOT_ID --scope difficulty-evaluation
```

并行分支在线评分图使用独立的 `difficulty-plot-parallel` 入口，语义与 trajectory remaining-work 图不同。
