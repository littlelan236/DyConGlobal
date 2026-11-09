# 容器与环境操作

容器配置属于项目统一 runtime 和 benchmark backend。使用 `dycon` 提供的存储、镜像、环境和任务 preflight 命令；不要在运行期间手工改写 run 的容器快照、绕开 `ContainerFactory` 启动任务，或删除仍被 run 引用的镜像与卷。

## 检查顺序

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon runtime-check
.venvs/runner/bin/dycon images-check
.venvs/runner/bin/dycon docker-storage check
```

各命令检查范围不同：配置校验根 YAML；runtime 检查执行环境；images 检查已登记的任务镜像；docker-storage 检查配置的数据根与 daemon 状态。随后仍需运行对应 benchmark preflight 或 harness gold 环境检查。

## 数据根与镜像

需要变更 rootless Docker 数据根时，先按 `dycon docker-storage --help` 检查/应用 `config.yaml` 中声明的目标。变更会影响 daemon、镜像与容器可见性；确认服务状态和既有数据路径后再执行，不要通过手工改 daemon 文件创建第二套来源。

任务镜像由 benchmark/profile 固定。通过 `skillsbench-images`、`webarena-images` 或相应 benchmark 导入命令准备镜像，并核验冻结任务范围、镜像摘要与 NAS 路径。镜像可拉取不代表任务依赖、网站、浏览器和 evaluator 均可用。

## 故障定位

先从 `run status`、`harness status`、`report run` 和 attempt 日志定位失败阶段，再使用同阶段的 runtime、image、preflight 或 gold 检查。保留容器标签、run/attempt 身份与原始日志，修复前不要清理故障现场。

依赖错误应检查失败 distribution、Python 版本与平台标签、wheelhouse receipt 及冻结 requirements；网络或镜像失败应分别检查服务可达性、镜像身份和容器输出。不要用宿主机“已安装”推断任务容器依赖满足，也不要把基础设施错误记作 benchmark unresolved。

`runtime-check`、gold check、单任务 smoke 与全量 benchmark 运行是不同等级的验证。变更后使用目标 benchmark 的 preflight 和最小 run 验证，再按任务规模执行正式工作负载。确切参数见 `dycon runtime-check --help`、`dycon docker-storage --help` 与 benchmark CLI help。
# Docker 网络清理与地址池

任务退出按 `dycon.project/run/execution` 清理；单个 SkillsBench 环境另绑定 task/sample/attempt/epoch。
正常结束、失败、取消和 supervisor 异常退出均核验容器及网络。仅删除归属匹配且已无附着容器的网络，
不执行全局 prune，不断开其他容器连接，也不清理其他 run 或 execution 的网络。清理回执含
`removed_networks`、`remaining_networks`，存在残留时 `cleanup_complete` 不得标为成功。

用户 rootless Docker 使用 `~/.config/docker/daemon.json`，不要误改 `/etc/docker/daemon.json`。
例如 `default-address-pools: [{"base":"172.16.0.0/16","size":23}]` 可提供 128 个自动分配子网；
必须先确认网段与现有网络和访问目标无冲突。生效需要重启对应用户 Docker daemon，已有网络不会缩小。
重启前通过正式 cancel/wait 停止目标 run，并核对同一 daemon 上的其他工作。
