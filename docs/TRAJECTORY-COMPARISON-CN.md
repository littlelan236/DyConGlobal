# 重复 rollout 成功次数与预测 / 真实剩余量对照

## 显式轨迹：包含未成功的正常结束样本（2026-10-31）

沿用同一入口，显式指定一条或多条 canonical rollout，不要求 resolved=true：

```bash
.venvs/analysis/bin/dycon difficulty-plot-trajectories SOURCE_RUN NEW_PLOT_ID \
  --attempt-id FITTED_RIDGE_ATTEMPT \
  --rollout-list path/to/sha-bound-rollouts.json \
  --target submission_remaining_reasoning_tokens_log1p
```

也可重复使用 `--rollout-id ID`。列表格式复用 difficulty 的 JSON ID 列表或
`[{"rollout_id":"ID","trajectory_sha256":"SHA256"}]`；未知、重复、隔离或 SHA 不匹配的轨迹拒绝。
显式模式读取 run 原始 NAS publication，复用 canonical/native 绑定、tensor 完整性和 Ridge 预测逻辑；
不需要创建训练标签数据集，也不与 `--dataset-id`、`--repair-id`、`--reuse-predictions-id` 混用。
默认不指定 rollout 的模式仍要求 dataset，并保留原有 resolved 筛选。

显式模式每条图及 CSV 单列 resolved、evaluation_status 和 termination_reason。
绿线来自 native exact 后缀，表示 **到本次 agent 结束的实际剩余量**，
`truth_endpoint=observed_agent_terminal`；未成功轨迹不冒充成功剩余量或成功概率。
缺 hidden 留预测断线，缺完整 native 后缀留实际量断线，不重建 token、不补零。
训练是否重叠未审计时 used_in_fit=null、split=not_audited。
SkillsBench 的 workspace subject 及导出文件通过既有 verify_subject 校验并绑定新 receipt。
生成后仍使用 `dycon artifact verify NEW_PLOT_ID --scope difficulty-evaluation`。

从项目根目录使用 analysis 环境的统一入口，不启动模型服务、不训练新模型：

    .venvs/analysis/bin/dycon difficulty-plot-trajectories SOURCE_RUN NEW_PLOT_ID \
      --attempt-id FITTED_RIDGE_ATTEMPT --dataset-id RESOLVED_DATASET \
      --target submission_remaining_reasoning_tokens_log1p

当前支持 Ridge 的 reasoning_tokens / steps 两种目标，各支持 log1p 和 linear。
模型来自指定训练 attempt 的已冻结 selected checkpoint；有 train+validation refit 时使用该模型。
复用 difficulty.datasets.submission.load_selection / prepare_cache 和 ridge_grid 的读取、验证与预测逻辑。
不会修改旧 attempt、训练数据集、run 快照或评分；旧 difficulty-plot-parallel 的在线分支评分图保持原入口。

## 统计口径

- 每个 task 的最终 resolved 次数以冻结 work items 的 task×repetition 为单位，由 repetition/result.json 指定最终 canonical rollout；复用 rollout.retry.verify_final 校验身份与 SHA。
- 缺少最终 result 的 repetition 单列 pending，不计入失败，也不凭历史 attempt 推断完成。计划次数与已登记最终结果数分别导出。
- completed 且 resolved=true 才计成功；completed/false、未评测或基础设施故障分别统计。Submitted 不等于 resolved。
- 图覆盖 run evidence 中全部未隔离的成功 canonical 轨迹，包括历史成功 attempt；每幅图标明 final / historical_attempt。all_resolved_attempts 与最终独立成功次数分列，不重复计次。
- 全部 task 都保留，包括零成功的 task。统计只涉及 SOURCE_RUN，不把原始 migration run 的成功加进重复次数。

## 绘图口径

每条成功轨迹一张双面板 SVG：上面为原始剩余量（reasoning tokens 或 steps），下面为 log1p 对照。
绿线是真实剩余量，蓝线是模型预测；原始 log1p checkpoint 分数先 expm1 后与原单位真值比较。
同 task 的所有图共用原始 step 横轴和两种纵轴范围，不归一化长度、不跨轨迹平均。

真实 reasoning tokens 只来自冻结数据集的 qualified native close 标签；缺失、不完整或不合格证据为 null。
缺 hidden 无预测；缺 native token 证据不从文本 tokenizer 重建成 exact 真值。
图像显式补出缺失整数 step 以断线，不跨缺口连线、不插值、不平滑、不截断负预测。
完全没有可用点的成功轨迹仍生成 unavailable 说明图，而不是零曲线。

横轴为 canonical assistant step（0-based），think_close 时当前动作未执行。
真实剩余 steps 为 T−s，末步为 1；close reasoning tokens 只计算当前 reasoning 之后的剩余 reasoning，末次通常为 0。

## 输出与校验

输出属于项目级 difficulty/evaluations/NEW_PLOT_ID/：

    report.html                 自包含页面，SVG 内嵌，同 task 同行横向滚动，支持筛选
    task-resolved-counts.csv     每个 task 的成功、未成功评测、未评测、pending 与计划次数
    trajectories.json / .csv    每条成功轨迹的身份、SHA、覆盖数和 final/historical 标记
    curves.csv                  每个 canonical step 的预测、真值、split、used_in_fit
    summary.json                总计、成功次数直方图、模型及缺失说明
    selection.json              冻结 run evidence、最终 selection 及 pending work items
    plots/ROLLOUT_ID.svg         独立图
    code/...                    CLI / 绘图 / 分析实现快照
    receipt.json                输入与输出 SHA

单独复制 report.html 即可离线浏览。缺图、缺线都保留明确说明。
同名输出拒绝覆盖；图在临时 staging 目录完成后原子发布，失败不会留下看似完整的输出。
统一校验命令：

    .venvs/analysis/bin/dycon artifact verify NEW_PLOT_ID --scope difficulty-evaluation

内部复用 difficulty.receipts.verify_receipt；数据集和 checkpoint 输入在生成前亦通过该既有校验器检查。

## 解释边界

这是剩余工作量对照，不是成功概率。若模型训练使用这些轨迹的样本，曲线仅可作为拟合诊断，
不能当作独立 task holdout 的泛化评估。curves.csv 中保留每个点的原始 train/validation/test
分区，以及它是否参与实际所用 checkpoint 拟合（refit 会使用 train+validation）。
即使一个点在 test，同一轨迹/同一 task 的其他点也可能参与训练。

新增回归测试位于 tests/test_trajectory_comparison.py。


## 扩大覆盖：预测与真值独立筛选（2026-10-27）

推荐显式选择已发布的 pointwise repair，以覆盖不满足训练标签条件的 hidden：

    .venvs/analysis/bin/dycon difficulty-plot-trajectories SOURCE_RUN NEW_PLOT_ID \
      --attempt-id FITTED_RIDGE_ATTEMPT --dataset-id RESOLVED_DATASET \
      --repair-id PUBLISHED_POINTWISE_REPAIR \
      --target submission_remaining_reasoning_tokens_log1p

不指定 --repair-id 时保留原数据集样本读取方式。指定后，预测点从完整 repair publication
读取，不再受 both_targets_available、censored、完整 native future suffix 等训练标签条件限制。
模型、成功轨迹集合、task/repetition 成功计数保持原有定义。

新增 difficulty/evaluation/trajectory_sources.py 复用现有 capture repair 的 _canonical、
_bindings、_classification 以及 read_repaired_session、read_request 和 iter_features：

- receipt、canonical trajectory SHA、run/task/model、原始 response 与 hidden 的绑定必须通过校验。
- native_exact 和 canonical_identity_only 都可用于预测；后者明确记录 prediction_binding，
  不冒充精确 native token 证据，也不会改变源 repair 的 eligible_for_training=false。
- 必须绑定 canonical assistant action 的 think_close；FormatError 等非 canonical action、
  orphan、冲突、未持久化点不用于预测。同一步内容相同的别名去重，内容不同的歧义拒绝。
- 所有实际读取的 shard / member SHA、feature shape 和有限数值通过既有张量读取器校验。
- 预测只依赖合法 hidden；没有真值也绘制蓝线，不从标签条件反向删预测点。

真值单独取训练数据集标签，以及其已绑定 publication 的 exact native request summaries。
request summaries 中由既有 finalizer 算出的 future_reasoning_tokens / remaining_censored
可在当前 step 缺 hidden 时继续提供真值。存在 censored、未知未来 token 或多 close 歧义时不
补造 reasoning token 真值。steps 真值仍直接由成功 canonical action 计数 T−s 得到。

因此可以有“只有预测”“只有真值”“两者都有”“两者都缺”四种点，分别保留断线，不插值补点。
curves.csv 新增 prediction_binding、hidden_available、truth_available；trajectories.csv
分别报告 predicted_points、true_points、prediction_only_points、truth_only_points。
coverage-audit.json 记录 repair 来源、证据类型、排除原因以及相对训练样本的新增点数。
同一 task 的全部图仍共用坐标范围，HTML 显示预测与真值各自的覆盖数。

新输出使用新的 PLOT_ID，保留旧图和旧 receipt。统一 artifact verify 会核对新报告绑定的
repair 元数据、native 证据与实际使用的 tensor shards；训练集、模型和原始 repair 不被改写。

### 原始 NAS 发布源也纳入并集

指定 --repair-id 后还会自动扫描该 run 原始 NAS index / publications，并与 repair 取并集。
这样不会遗漏早已原生发布、但不在某次 repair cohort 内的 hidden。原始点校验 publication
绑定的 index / shard SHA、成员 SHA、canonical request / response 身份、model 与 close anchor。
缺少 capture 身份或存在歧义的轨迹单列排除原因，不让它们阻断其他可验证轨迹的绘图。
新增点在 curves.csv 中标记 capture_source=original_publication。

原始 native request receipts 的完整后缀也独立提供真值。remaining workload 的实际计算提取为
enrichment.online_trace.annotate_remaining_workloads，原 finalizer 与可视化共用该函数，
不另写一套 suffix/token 标签定义。未知未来 response 或 censored block 之前仍不补造真值。

可选 --reuse-predictions-id PREVIOUS_PLOT_ID 复用同 run / repair / target / 模型 SHA 下已
通过 receipt 校验的预测值，避免反复读取同一批大张量。原始发布索引仍重新扫描；新增点仍读取
并校验实际 hidden。省略该参数可从全部 repair 张量重新计算。源模型或 receipt 不匹配会拒绝。
