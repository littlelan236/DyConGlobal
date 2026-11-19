# 冻结 estimator 的轨迹漂移评测

正式入口：

```bash
PYTHONPATH=. .venvs/analysis/bin/dycon difficulty-evaluate-drift EVALUATION_ID \
  --workflow-config WORKFLOW.yaml
```

该入口读取原 rollout 的冻结 `parallel.estimator_ref/estimator_sha256`，校验
导出 manifest、payload 和 predictor 实现哈希，并复用原 loader/preprocessing。
不选择 layer/alpha、不重拟合、不更新 scaler、不请求模型或补算 hidden。

workflow schema 1 必须包含 `estimator_run`、`allow_hidden_model_shift`、`groups`
和 `skipped_groups`。每个 group 包含 `name`、`dataset_id`、`sources`；每个
source 显式绑定 `run_id`、`publication_id`、`publication_receipt_sha256`、
`index_id` 和唯一 `dataset_id`。跨 checkpoint hidden 必须显式声明
`allow_hidden_model_shift: true`，报告同时保存 predictor 与数据真实模型指纹；
在线 serving 的同模型指纹限制不改变。各组内部不允许混合模型指纹。

来源必须是已封存的普通 run-end publication，按其 receipt 绑定的最终 accepted
canonical selection 取轨迹。复用 `online-index`、submission selection 与
`difficulty-dataset compose` 实现来源校验和合并。没有正式 publication 的活跃
run 不隐式暂停、发布或改选；调用方在 `skipped_groups` 中记录原因。

`difficulty-dataset resolved --eligibility submitted` 是显式的到提交终点工作量
视图，必须指定 rollout 选择，不要求 benchmark harness 成功。只允许已提交的
SWE 类 patch 轨迹，仍验证 canonical trajectory/subject、publication、native
exact action-close、标签和哈希；不产生成功标签。原 `resolved_only`、
`positive_reward` 和 `accepted_terminal` 规则保持各自定义。

每组所有有效 action-close 点合并后计算一次 pooled R²：主指标是
`log1p(remaining reasoning tokens to submission)` 与原 predictor score 的 R²；
附原 token 空间 `expm1(score)` 的 R²、MAE、RMSE、SSE、方差分母。
不裁负值，不平滑，不平均轨迹 R²，不只选择 resolved，不静默丢弃非有限配对。
读取复用 grouped tensor reader，校验 shard/member 哈希和所需 site/width。

产物位于 `difficulty/evaluations/EVALUATION_ID/`：workflow、estimator 合同、
progress、summary、各组 predictions/逐轨迹指标，以及绑定 composed datasets
与 predictor 文件的最终 receipt。未完成工作流没有成功 receipt；恢复必须保留
原失败目录并用新 evaluation ID，已有数据集只在 receipt/来源一致时复用。

这是观察到的轨迹分布上的 estimator 适用性检查，不是 task holdout；R²变化同时
受到 hidden 表示、任务/轨迹构成和真值方差变化的影响，不能单独证明因果漂移。
