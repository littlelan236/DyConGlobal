# 多 GPU、每卡 TP=1 的轨迹调度

日期：2026-10-31。

在同一个 run 内，每张指定 GPU 启动一个独立 TP=1 serving worker，完整 agent 轨迹固定使用领取它的服务。轨迹每一步的所有候选分支仍在该卡上采样；这不改变 `parallel.branch_count`、选枝策略或重复采样身份。

在根 `config.yaml` 的既有 `serving` 节中设置以下字段，其余配置继续使用原值：

```yaml
serving:
  trajectory_sharding: per_gpu_tp1
  gpus: [0, 1, 2, 3]
  tensor_parallel_size: 1
  base_url: http://127.0.0.1:18200/v1
  replica_base_urls:
    - http://127.0.0.1:18200/v1
    - http://127.0.0.1:18201/v1
    - http://127.0.0.1:18202/v1
    - http://127.0.0.1:18203/v1
  max_cpu_workers_per_gpu: 4
```

`replica_base_urls` 与 `gpus` 按列表顺序一一对应，须为不同端口的本机 HTTP `/v1` 地址；`base_url` 必须等于第一项。创建 run 时会冻结这些值。每卡完整装载一份模型，`gpu_memory_utilization`、KV 配置、`max_num_seqs` 和分支容量均是每个服务的值。

`max_cpu_workers_per_gpu` 是正整数，限制该 GPU 同时承载的 CPU 轨迹任务数，包含准入、容器初始化、agent 工具执行、模型请求和 capture 收尾。该模式下总 rollout CPU worker 数为 `len(gpus) * max_cpu_workers_per_gpu`，替代单服务模式的 `execution.rollout_workers`。上例为四张卡、每卡最多四条在途轨迹，总共最多十六条。

调度器从同一个全局任务队列领取样本，优先分配到当前在途轨迹最少且未达到上限的卡。任一在途任务结束并释放槽位后，立即补发下一项；不等待其他卡或同批轨迹结束。较快的卡会自然领取更多任务。完整轨迹中的请求不会中途换卡；整条轨迹换 seed 重试或基础设施重试是新的物理 attempt，可以重新领取空闲卡，原来的任务、repetition、seed、重试预算和证据约束保持不变。

这里的 CPU worker 是轨迹调度槽位，不是容器内部 CPU core 配额，也不是 vLLM 的单条 decode sequence。提高每卡上限可让一条轨迹执行 CPU 工具时，其他轨迹继续调用模型；仍需满足任务队列非空和 capture/容器资源可用，不能保证所有时刻 GPU 利用率为 100%。

使用既有统一入口：

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run start NEW_RUN --detach
.venvs/runner/bin/dycon run status NEW_RUN
.venvs/runner/bin/dycon report run NEW_RUN
.venvs/runner/bin/dycon run cancel NEW_RUN --reason 'user requested stop'
.venvs/runner/bin/dycon run wait NEW_RUN
```

也可通过 `run create NEW_RUN --operation rollout --from-run SOURCE_RUN` 的既有冻结克隆流程，显式 `--set` 上述 serving 字段。这里仅在创建新 clone 时使用 `--set`；正式执行或 resume 继续读取该 run 原始快照。必须保留源 run 的 repeat-count 和完整任务选择；已有 run 转换模式须走下文的审计修订入口。

例如，来源是 managed serving 且 `repeat_count=1`，创建两卡、每卡最多 4 个 CPU worker 的独立 clone（所有 `--set` 位于子命令前）：

```bash
.venvs/runner/bin/dycon \
  --set serving.trajectory_sharding=per_gpu_tp1 \
  --set 'serving.gpus=[2,3]' \
  --set serving.tensor_parallel_size=1 \
  --set serving.base_url=http://127.0.0.1:18200/v1 \
  --set 'serving.replica_base_urls=[http://127.0.0.1:18200/v1,http://127.0.0.1:18201/v1]' \
  --set serving.max_cpu_workers_per_gpu=4 \
  run create NEW_RUN --operation rollout --from-run SOURCE_RUN --repeat-count 1
```

`NEW_RUN` 须为新 ID，示例 GPU 与端口须替换为实际分配；来源有多个 repetitions 时，将 `--repeat-count` 改为来源的确切值。已有 run 可在停止并确认清理完成后，通过 `revise-serving-placement` 显式修订拓扑，再以原 ID resume；单独 `resume` 不改变模式。克隆的其他覆盖字段见[冻结克隆](RUN-CLONE-CN.md)。

`run status` 和 `report run` 的 `serving_workers` 展示每卡端口、CPU 槽位占用与上限；本地证据位于 `runs/RUN/serving/worker-pool.json`。各服务的 launch、process、service 和 server.log 位于 `serving/workers/worker-NN/`。每个真实物理 attempt 另存 `serving-assignment.json`，结果中也含 `serving_assignment`，可追溯 sample/attempt/execution 与 GPU/服务的绑定。

`serving_workers.workers[]` 中的 `active_cpu_workers` 是包含 CPU 工具和收尾阶段的在途轨迹数，`available_cpu_workers` 是当前可继续领取任务的槽位数；它们不等于 GPU 实时 decode sequence 数。`serving_workers.state` 在服务池运行时为 `running`，退出后为 `stopped`。判断取消是否完成仍检查 `cancellation.termination.cleanup_complete`，不能仅看 worker 槽位已归零。

所有服务共用一个 run 的 capture 控制器、发布队列与磁盘预算，各服务只负责自己的模型执行；采集证据仍落在既有 run/NAS 命名空间。取消会通知全部模型服务，supervisor 仍通过现有资源登记与进程身份管理全部 serving 进程组；任一卡服务启动失败会关闭已经启动的同 run 服务。恢复通过现有 capture recovery 和 attempt 恢复流程，已接纳样本不会重新生成。

省略 `trajectory_sharding` 或设为 `none` 时使用既有单服务 TP 模式，仍要求 GPU 数与 TP 一致。旧配置缺少新字段时不注入这些字段，因此不会改变旧配置或 run 的 hash。


## 保留原 run ID 和进度的离线拓扑修订

先通过 `run cancel` 停止并核验 `cancellation.termination.cleanup_complete=true`，再执行：

```bash
.venvs/runner/bin/dycon run revise-serving-placement RUN_ID \
  --revision-id tp1-gpu01 --gpus 0 1 \
  --base-url http://127.0.0.1:18200/v1 \
  --trajectory-sharding per_gpu_tp1 \
  --replica-base-urls http://127.0.0.1:18200/v1 http://127.0.0.1:18201/v1 \
  --max-cpu-workers-per-gpu 2 --gpu-memory-utilization 0.4 \
  --reason 'User authorized TP1 topology conversion preserving completed progress'
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

四个新增选项须一起显式传入，TP 自动设为 1。上例总 CPU 轨迹并发仍为 4；省略四项时保持原设备/端口迁移语义。复用原修订锁、退出/清理验证、before/after、配置与 plan 哈希链和 receipt；任务选择、work-items、evidence 及已接纳结果不变。显式 KV 字节预算会使显存比例无效，因此拒绝这类转换，须先另行审计资源政策。运行中或清理未完成时拒绝修订。共享 GPU 上的各服务仍需自行安排互异端口及可容纳的显存比例。
