# 外部磁盘上的 Ridge 训练与 accepted test 评估

`paths.artifact_roots.difficulty` 指定 difficulty dataset、split、checkpoint、
evaluation 的物理存储根；例如在 analysis CLI 前使用
`--set paths.artifact_roots.difficulty=/data2/wangtianyi/DyConGlobal`。
`paths.artifact_roots.difficulty_cache` 单独指定大体积特征缓存根。
正式 holdout 服务在调用期间使用该存储作用域，所有 source/run 路径仍按原项目
owned_path 规则核验；没有放宽全局路径边界，也没有改写原 run 的冻结配置。

默认行为继续使用项目原路径。相同 ID 同时出现在配置存储根和原根时拒绝歧义。
dataset-only 的构建器仍支持既有 `--storage-root`。

对已拟合模型进行固定 task 留出评估：

```bash
PYTHONPATH=. .venvs/analysis/bin/dycon \
  --set paths.artifact_roots.difficulty=/data2/wangtianyi/DyConGlobal \
  --set paths.artifact_roots.difficulty_cache=/data2/wangtianyi/DyConGlobal \
  difficulty-ridge-test NEW_EVALUATION \
  --attempt-id FIT_ATTEMPT --dataset-id ACCEPTED_DATASET \
  --task-partitions ALL_NOMINAL_TASKS.json
```

输入清单必须包含显式 train/test 身份；评估器校验原 checkpoint 的 task partition，
只对 test tasks 预测，任何 test task 与实际 fit task 重叠时立即报错。
模型不重新拟合，也不根据 test 结果重选 layer/alpha。
输出只计算 log1p 目标的拟合空间指标，分别报告 accepted 和其 resolved 子集。
每条轨迹先计算 R²，再取不加权平均、中位数和指定分位数；另保留 pooled R²。
不可定义的轨迹 R² 保留 null，并单列数量。

每次正式调用使用新的 evaluation ID；产物含逐点预测、逐轨迹指标、隔离审计
及绑定数据集和模型 SHA 的 receipt。可视化没有启用。
