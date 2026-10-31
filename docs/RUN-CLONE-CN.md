# 从现有 run 的冻结配置新建独立 rollout

`dycon --set serving.gpus=[0,1,2,3] --set serving.tensor_parallel_size=4 --set serving.base_url=http://127.0.0.1:8104/v1 --set parallel.estimator_ref=/absolute/new/manifest.json run create NEW_ID --operation rollout --from-run SOURCE_ID`

`--from-run` 通过正式 create_run 入口复用已有 snapshot、work-items 和 plan 机制。复制 source 的完整冻结 selection 和配置，不复制其结果、失败 cursor 或 operator retry 请求。来源 run 可继续运行，其配置/任务证据只读。

需要反转 cohort 执行顺序时，在创建克隆时加 `--reverse-task-order`。它反转冻结 selection 的 task_keys，重建新 run 的 tasks、plan 和 work-items 哈希，并在 `cloned-from.json` 记录 `task_order=reversed`；source 的字节和任务顺序不变。该选项仅适用于 `--from-run`。

当前允许的创建期 `--set` 覆盖与 `application/execution/run_plan.py` 一致：

- 服务设备与模式：`serving.gpus`、`serving.tensor_parallel_size`、`serving.base_url`、`serving.trajectory_sharding`、`serving.replica_base_urls`、`serving.max_cpu_workers_per_gpu`。
- 服务容量：`serving.gpu_memory_utilization`、`serving.max_model_len`、`serving.max_num_seqs`、`serving.max_num_batched_tokens`。
- 生成契约：`generation.max_tokens`、`generation.thinking_token_budget`、`generation.require_reasoning_close`。
- 估计器与存储：`parallel.estimator_ref`、`paths.artifact_roots`。

必须显式保持 source 的 `--repeat-count`；源 run 为 3 次重复时，创建 clone 也传 `--repeat-count 3`。不允许再次指定 task/partition/tasks-file。新 estimator 重新核验模型指纹、实现和 public site，并绑定 manifest SHA。

同一 run 在多张卡上各用 TP=1 时，创建期设置 `serving.trajectory_sharding=per_gpu_tp1`、TP=1、GPU/端口列表和每卡 CPU worker 上限。源配置必须已经是 managed serving。完整命令见[多 GPU TP1 指南](MULTI-GPU-TP1-CN.md)；新 clone 拥有独立进度，不会接着来源 run 的已完成数运行。

新 run 的人类可读配置从来源 run/config.yaml 生成，cloned-from.json 记录 source plan/config/selection/tasks 身份与覆盖项。它拥有独立 sample/attempt/trajectory/capture 身份，不能将 source 进度算作新 run 的结果。


Clone支持`--set paths.artifact_roots='{"run":"/data3/wangtianyi/DyConGlobal"}'`，必须在`run create`之前传入。该创建期覆盖仅改变新run的物理产物根；保留来源run的配置与完整task selection，覆盖写入cloned-from receipt。外置轨迹prepare/finalize复用core.io.owned_path对冻结run存储与工程alias的校验。
