# SkillsBench 启动预检与多卡服务启动

在根 `config.yaml` 中配置，并通过正式 `dycon run create` 创建新 run：

```yaml
execution:
  preflight:
    long_checks: true
serving:
  startup_mode: serial  # serial | parallel
```

两项均随 `resolved-config.json`、配置 hash 和 run 计划冻结。`run start`、`run resume` 及 `run retry-failed` 后的 resume 读取原 run 快照；修改根配置不会更新已有 run。缺省字段分别按 `true` 和 `serial` 执行，解析旧 YAML 时不额外序列化这些字段，因此历史配置 hash 不变。新建 frozen clone 默认继承源 run；可在创建 clone 时显式覆盖 `serving.startup_mode`，不修改源 run。

## 长耗时检查开关

`execution.preflight.long_checks: false` 只控制 SkillsBench/OpenHands 的共享启动预检，其他 backend 保持原有行为。关闭后跳过：

- 已登记模型文件的内容 SHA；仍核验登记身份、清单结构、文件路径、存在性和大小。`model.verify_identity` 继续保持独立含义。
- 任务包所有文件的内容 SHA；仍核验冻结清单身份、声明路径及存在性、文件模式、任务能力和输出声明。
- NAS OCI descriptor 的深度 SHA；仍遍历完整 descriptor 图并核验路径、存在性、大小、NAS manifest 引用和冻结 image identity。预检前的镜像租用/导入也使用同一策略；导入镜像本身仍需要读取和传输实际镜像内容。
- OpenHands 依赖快照逐文件 SHA；仍核验 manifest receipt、冻结 manifest SHA、文件清单、路径、存在性和大小。
- 全量任务容器环境审计；不为所有任务创建审计容器、安装依赖及运行 skills/agent 探针。

配置、backend/profile、generation/execution context、版本接口、GPU/MIG 探测、Docker daemon 和 CPU quota 检查仍执行。正式 physical attempt 的任务包和模型完整校验、compose image identity、依赖安装、skills visibility、agent environment probe、清理和取消逻辑保持执行；失败重试也不继承“免验证”状态。

`validation/startup-preflight.json` 中 `checks` 分别记录 `model_sha`、`task_package_sha`、`oci_deep_verify`、`dependency_snapshot_sha`、`environment_audit` 的状态和耗时。跳过项为 `status: skipped`，原因包含 `execution.preflight.long_checks=false`，不会标记为 verified。镜像提前准入记录在 `image_admission_checks`；runner 在 adapter 前的模型校验记录在 `validation/startup-model-identity.json` 并汇入共享预检 receipt。`ready` 仅表示当前策略的启动门禁满足，不代表跳过的完整性审计已经完成。独立 `skillsbench preflight RUN_ID` 同样服从冻结策略。

需要完整性审计时保留 `true`。设为 `false` 后，启动门禁无法发现文件内容被同大小替换的情况；正式 attempt 的验证仍可能发现相关变更。

## 多卡 TP=1 并发启动

在既有 `serving.trajectory_sharding: per_gpu_tp1` 配置中设置 `startup_mode: parallel`，先顺序准备所有 worker：配置与端口检查、共享 capture controller/token inventory/control server 初始化，以及各设备的资源准入。随后并发创建各自进程并等待 `/models` 就绪；所有 worker 成功后才向轨迹调度器开放 pool。

任一 worker 准备或启动失败时，取消其余 worker 的启动等待，等待启动线程退出，再按副本到 capture owner 的顺序终止进程组、关闭共享控制服务并释放设备资源。每个 worker 继续使用自己的 GPU、端口、进程组和原有设备锁策略。TP、轨迹调度、每卡 CPU 并发及模型请求语义不变。单服务模式仍只启动一个服务；现有 MIG 配置限制保持原规则。

`serving/worker-pool.json` 和每个 worker 的 `launch.json` 记录 startup mode；各 worker 的 `startup-timing.json` 及 pool receipt 记录准备、进程创建、API 就绪和清理时间。`model_load_and_api_seconds` 是进程创建到 `/models` 就绪的总间隔，包含模型加载、engine 初始化和 API 启动，不能当作纯权重读取时间。

可使用 `dycon run start RUN_ID --check-only` 做正式启动验收。用同一模型、同一组空闲 GPU、同一 NAS 存储分别创建 serial/parallel run，比较 worker timing 和整体墙钟时间。GPU 竞争、文件系统/page cache 和编译缓存状态会影响实测值；默认仍为 serial。
