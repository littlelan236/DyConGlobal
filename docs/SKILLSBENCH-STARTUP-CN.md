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

## SDK 格式恢复与提交导出验收

新的 SkillsBench/OpenHands run 应在创建前将根配置中的 `agent.profile_ref` 指向
`runtime/agent-profiles/openhands-sdk-v4-qualified-main-20261101/manifest.json`。
该 profile 绑定 `openhands-sdk-format-v4-20261101`，已包含 SDK transport 适配及
`agent_decision`、`condensation` 的真实请求角色资格证据。仅修改仓库中的
`runtime/agents/openhands_transport.py` 不会更新已发布 wheel 或旧 run 的依赖。
旧 profile/依赖/run 快照保持不可变，切换后通过 `dycon run create/start` 创建新 run。

可先运行以下命令逐文件校验已发布依赖，再检查新 run 的冻结 profile 和
`validation/startup-preflight.json`。实际 attempt 的 `dependency-install.json`、
`diagnostics/sdk-transport-*.json` 与 `diagnostics/sdk-local-transport.json` 用于确认
安装和执行了该 SDK 适配，不能以启动成功代替在线验收。

```bash
.venvs/runner/bin/dycon skillsbench dependencies-verify \
  runtime/dependency-snapshots/openhands-sdk-format-v4-20261101
.venvs/runner/bin/dycon skillsbench audit RUN_ID
```

bridge 返回经过归属校验的 `FormatError` 时，适配将其转换为 SDK 原生
`FunctionCallValidationError`，由 Agent 加入纠正消息后继续下一轮。非格式故障仍保留
原始错误；这不是隐藏的 SDK 自动重试，后续模型请求继续累计样本 token 预算。

`organize-messy-files` 的任务提示只指定五个目录名，官方 verifier 可在
`/root/papers`、`/root`、工作目录及 `/root` 下共同父目录发现结果。旧 profile
声明的 `/root/LLM` 等路径可能与实际位置不同。适配器仅对这套旧声明解析完整的五目录组，
按上述顺序发现已有输出，随后复用统一 `export_workspace` 归档；不移动或创建答案文件。
显式配置的其他输出路径仍按 profile 执行。检查
`diagnostics/submission-output-resolution.json` 的声明路径、发现目录和生效路径，
以及 `subject/subject.json` 的 `output_inventory`、`missing_outputs`、`rejected_outputs`。
解析回执哈希写入 subject binding。reward 成功与导出完整是两项独立条件。

Verifier 工具准备统一由 `runtime/agents/verifier_tools.py` 执行，online/offline 共用。
在 BenchFlow hardening 完成后，从只读发布目录复制 uv 到 sandbox 的可写
`/root/.local/bin`，生成可执行 uvx 和 shell env；发布的 curl bootstrap 适配复制到
sandbox 内原有可信 PATH 的 `/usr/local/bin/curl`。官方 uv 安装请求使用本地发布工具，
其他 URL 仍交给系统 curl。这样不会依赖硬化时被剔除的工作目录 PATH，也不会在
TLS 下载失败后缺少 uvx/env。保留上游硬化的 PATH 与 pytest 隔离规则，不修改官方测试脚本。
`diagnostics/verifier-tools.json` 记录复制后哈希、uv 版本和准备策略；源发布目录不被修改。

SkillsBench 的 capture stop 单独使用收尾等待上限：至少覆盖 native-output drain 与
export-queue drain 两段 `capture.finalize_seconds`，另加 30 秒响应余量，并取
`control_seconds`、`finalize_request_seconds` 的较大值。其他控制请求继续使用原超时。
实际 stop timeout 写入 `capture-drain.json` 的 `session_stop_timeout_seconds`；收尾失败
仍属于独立 capture 证据，不覆盖已经发生的 attempt deadline/agent/verifier 主错误。

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
