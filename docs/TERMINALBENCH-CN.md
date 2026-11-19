# Terminal-Bench 2.1 / OpenHands

正式入口仍为 `dycon harbor` 和 `dycon run`。固定 benchmark 为
`terminalbench_2_1`，任务 revision 为
`7131e4375048a0e408a8fb404b5f499d726b695b`；Harbor 为 0.25.0 /
`5eb713ff9500daced4e143b98765a7c732744a1b`，OpenHands CLI / SDK / tools
保持现有固定版本。

## 显式执行策略

在工程根 `config.yaml` 的 `harbor` 中设置：

```yaml
harbor:
  # 同时保留 runtime_ref、image_store、cpu_quota 等已有必填/运行设置。
  terminal_policy: terminalbench-live-v1
```

策略仅允许用于 `terminalbench_2_1`，新建 run 时连同有效参数冻结进
`execution_context.terminal_policy`，参与 context hash。旧 run 和没有此字段的
配置沿用原执行逻辑；不提升全局 backend registry revision，也不修改共享
`OpenHandsDriverProfile.sandbox_user=agent`。

- Terminal agent 在任务容器内使用 UID 0、`HOME=/root`；不需要宿主 sudo，
  不向容器暴露宿主 Docker socket。
- 保留官方工作目录和权限，不创建普通 agent 用户或递归 chown workspace。
  有效工作目录优先使用官方 task workdir，否则使用镜像 WorkingDir。
- 通过绝对路径启动私有 CPython/OpenHands；读取官方镜像 PATH 时不经过登录
  shell，实际工具使用任务 PATH。私有 Python 不替换任务 Python。
- agent 使用完整官方 timeout。environment build、setup（600 秒）、agent、
  verifier、cleanup（120 秒）分别记录在 worker request 的 phase_budgets。
  外层 worker deadline 覆盖其总和，但仍受冻结的 pipeline wall deadline 限制。
  oracle/nop 控制组同样使用该阶段预算，避免被短 container lifetime 截断。
- agent 结束时只停止已记录的 ACP supervisor、ACP 和 relay。使用 Linux pidfd
  和 `/proc` start ticks 核验身份，不按 UID 或整个进程组杀任务服务。
  同一容器中的服务存活到官方 verifier 完成，随后仍由 Harbor 和现有 ownership
  清理接口删除 attempt 容器/网络；取消与恢复继续使用正式 run 生命周期。
- 仅此新策略复用共享 condenser_context 冻结显式 condenser 配置；旧 context
  身份保持兼容。capture/parallel 仍要求独立 Harbor request-role qualification。

## 配置和资源准备

所有修改先落工程根配置，再创建新 run；不要编辑冻结 run 文件。典型设置为：

```yaml
benchmark:
  backend: terminalbench_2_1
  snapshot_ref: catalog/snapshots/tb21-7131e437-20261115/manifest.json
  partition: test
agent:
  backend: openhands
  profile_ref: runtime/agent-profiles/TARGET_PROFILE/manifest.json
harbor:
  runtime_ref: runtime/backend-snapshots/harbor-0250-pro6000-2-v1/manifest.json
  image_store: terminalbench21-7131e437-pro6000-2-v1
  cpu_quota: strict
  network_transport: unix_model_v1
  host_proxy: http://127.0.0.1:27890
  terminal_policy: terminalbench-live-v1
environment:
  docker_host: unix:///run/user/1023/docker.sock
  docker_data_root: /data1/wangtianyi/docker
capture:
  enabled: false
parallel:
  enabled: false
steering:
  enabled: false
```

以上是需要合并进现有根配置的片段。初次模型验收使用单 GPU、TP=1、单 rollout
worker；同时关闭 capture.logprobs，并清除仅支持 SkillsBench 的
execution.sample_completion_token_budget。pipeline wall 预算需容纳所选长任务。
host_proxy 只供宿主下载，不能代替容器 DNS/HTTPS 联网验证。

### 任务依赖镜像与 wheelhouse

Terminal-Bench 的 `test.sh`/`uvx` 在任务容器内安装依赖；宿主或 OpenHands 私有
依赖的下载设置不会自动传入该容器。需要为新 run 显式冻结国内 PyPI 索引和
`environment.wheelhouse` 与 `harbor.task_apt_mirror`：任务容器以只读方式挂载本地 wheels，
Python 包缺失时回退到 PyPI 镜像；Ubuntu 的 archive/security APT URI 也会改写到国内镜像。
`Signed-By`、suite、component 和签名校验保持不变；仅当官方 `solution/solve.sh` 在
apt-get update 前安装包时，DyCon 才在容器启动阶段先刷新索引。`UV_DEFAULT_INDEX`、
`UV_INDEX_URL` 和 `PIP_INDEX_URL`
指向同一 HTTPS 地址；不关闭 TLS 校验。无网络任务仅使用本地 find-links，不会
因该设置获得额外网络权限。

```bash
.venvs/runner/bin/dycon wheelhouse build terminalbench-task-wheels-v1 \
  --source runtime/dependency-snapshots/openhands-condenser-events-20261105-v2-rebuilt/wheels
.venvs/runner/bin/dycon wheelhouse verify runtime/wheelhouse/terminalbench-task-wheels-v1
.venvs/runner/bin/dycon --set harbor.task_python_index=https://mirrors.aliyun.com/pypi/simple/ \
  --set environment.wheelhouse=/home/wangtianyi/DyConGlobal/runtime/wheelhouse/terminalbench-task-wheels-v1 \
  run create TB21_RUN --operation rollout --task-id TASK_ID --repeat-count 1
```

`dycon wheelhouse build` 聚合已下载的 wheels，不会从任务 `test.sh` 推导 89 个任务
的完整依赖闭包；镜像是未命中本地 wheel 时的在线来源。OpenHands 自身仍通过其
profile 中固定哈希的离线快照安装。运行收据绑定 wheelhouse manifest SHA256；若要
确认具体包的实际来源，应检查官方 verifier 的 uv/pip 安装日志。APT 镜像只应用于
Ubuntu 容器，其他发行版保留镜像自带源。

先查看相应 `--help`，再使用已有 `harbor runtime-prepare/runtime-verify`、
`skillsbench dependencies-verify`、`harbor profile/schema-probe/import/images-prepare`
接口。不要把 Harbor/OpenHands 安装进 runner，不编辑不可变依赖发布。
完整官方任务集为 89 项；按排序 ID、逐项换行并以换行结尾的 SHA256 为
`3778b86071c74c8a342222cfff41089916adb5c7338d1afa69f42c9cae21fe3e`。
镜像准备和运行选择应复用同一份完整 snapshot。

## 验收和证据

在工程目录使用正式 CLI；从其他 cwd 调用时显式指定工程 config 的绝对路径。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon run create TB_RUN --operation rollout --task-id TASK --repeat-count 1
.venvs/runner/bin/dycon harbor preflight TB_RUN
.venvs/runner/bin/dycon harbor controls TB_RUN --control-id oracle-v1 --agent oracle
.venvs/runner/bin/dycon harbor controls TB_RUN --control-id nop-v1 --agent nop
.venvs/runner/bin/dycon run start TB_RUN --detach
.venvs/runner/bin/dycon run wait TB_RUN
.venvs/runner/bin/dycon report run TB_RUN
```

preflight 不发生成请求，检查实际身份/HOME、原生 terminal tool、任务 Python、
ACP 初始化和 Unix 模型通道认证；收集 terminal-identity.json、
agent-environment.json、model-channel-probe.json、acp-cleanup.json 及环境资源回执。
环境回执保留 Docker inspect 的实际 CPU/内存参数；storage_mb 是官方任务声明，
当前没有磁盘硬配额保证。

Rootless Docker 的 `info` 可能把 `Warnings` 返回为 null，也可能在用户 cgroup
没有委派 CPU controller 时仍报告支持 CPU quota。Harbor factory 同时检查本机
用户的 cgroup v2 controller；缺少 CPU 委派时 `strict` 会明确拒绝。
若接受 CPU 配额未强制执行，须在根配置显式设置
`harbor.cpu_quota: unenforced_with_receipt`，然后创建新 run；资源回执保留
requested/applied CPU、偏差和 Docker inspect 值，内存仍使用 limit。
不要编辑已经冻结的 run 来切换这一策略。


代表任务应包括文件/计算任务、nginx-request-logging 等系统服务任务，以及长
timeout 任务。模型 baseline 需验证真实模型请求、tool call/result、canonical
trajectory、live verifier reward、RunEvidence 和最终清理。再覆盖全部 89 项的
oracle/nop；控制组不能冒充模型解题结果。取消、超时及 resume 需使用新 physical
attempt，不能重放已执行工具动作。

Terminal subject 是原容器状态上的 live_state proof，容器销毁后不能仅从日志还原
进程/文件系统。禁止用 patch harness 事后 replay。正常零分保持 resolved=false；
无合法评分/基础设施故障保持 null；保留 Harbor reward_valid 允许的超时后有效奖励。
官方评分有效的 agent 超时以 `AgentTimeout` 结算；评分无效或待复核时仍按评测状态处理。
推理循环在 canonical 轨迹中记为 `reasoning_loop`，其他生成限制记为 `LimitsExceeded`。
启用 `execution.limits_retry` 时，两种退出均按新种子重试；原 attempt 的评测与
capture 结果继续单独保存。严格 capture 仍约束最终接收的 attempt。

无需任务镜像的代码回归：

```bash
DYCON_HARBOR_TEST_PYTHON="$PWD/runtime/backend-snapshots/harbor-0250-pro6000-2-v1/venv/bin/python" \
  .venvs/runner/bin/python -m pytest -q \
  tests/test_terminalbench_policy.py tests/test_terminalbench_worker_integration.py \
  tests/test_harbor_contract.py tests/test_backend_registry.py tests/test_openhands_bridge_lifecycle.py
```

进程集成测试真实启动 HTTP 服务，确认 ACP 正常退出、显式清理或超时后服务仍可供
评分访问，且 ACP/relay 被停止并回收。worker 接口测试使用真实隔离 Harbor 包和
模拟环境，不启动 Docker；这些检查不代表真实镜像、89 项官方控制组或模型
baseline 已通过。

## Canonical 工具反馈的超时边界

共享 OpenHands 适配器从每次 agent-decision 的有效请求前缀提取已选工具调用的反馈；
即使该请求随后超时、取消或失败而没有 selected response，已收到的反馈仍进入
canonical 消息。按工具调用 ID 去重，未选分支及 condenser 请求不提供工具反馈。
内容保留 SDK 请求中的原文，raw ACP 事件也继续保留；不会将 ACP 进度消息推断为完成结果。
若超时发生时尚无包含该反馈的后续请求，反馈仍仅保留在 raw ACP 事件中。
此修复对新导出生效，不改写历史 run 的 canonical 文件、哈希或评测结果。


## Verifier 健康判定与历史控制组复核

rollout 与 oracle/nop 共用 `harbor-verifier-health-v3` 判定。有限的 sub-one reward
只有在绑定 trial 的日志证明测试正常完成时，才认定为有效负例。verifier 工具缺失、
最终依赖下载失败、测试收集/fixture 失败或脚本异常归为 `VerifierError`：评测
`status=infra_error`、`resolved=null`，attempt 的既有状态名为 `VerifierError`。
保留 `raw_reward`、上游 result 哈希和 `verifier_audit` 原因/证据哈希，不将原始零分改成一分。
证据缺失或无法判定的 sub-one reward 为 `unknown/resolved=null`、审计分类 `Review`。
普通模型 rollout/nop 的断言失败仍是有效零分；已恢复的下载警告不会单独推翻正常测试结果。

oracle 官方参考脚本非零退出也使该控制组结果不可用，原因单独记录为
`oracle_reference_failed`。Oracle 是正控制组：所有有限的 sub-one reward 均为
`VerifierError`，不能作为有效负例；即使参考脚本退出 0 或 pytest 正常完成，
仍以 `oracle_reference_rejected` 拒绝结算，并保留原始 verifier 健康判定及证据。
该原因仅说明参考控制未通过，不宣称已证明 verifier 实现有 bug；可能涉及
参考解、资源/性能或环境兼容性，根因仍需排查。普通模型 rollout/nop 不使用这条规则。
新任务额外记录官方 verifier 的退出码；退出码 1 仍结合测试日志区分断言失败和环境故障。
成功奖励和上游允许的有有效奖励的 agent 超时语义保持原有规则。

历史全量控制组使用统一 CLI 创建单独的复核回执：

```bash
.venvs/runner/bin/dycon harbor controls-audit RUN_ID \
  --control-id CONTROL_ID --audit-id AUDIT_ID
```

输出位于 `runs/RUN_ID/validation/harbor-controls-audits/AUDIT_ID/summary.json`。
命令核对冻结任务、worker binding、原始 summary 哈希，绑定读取证据的哈希；
不运行模型/容器/verifier，不改写历史 `worker-response.json`、summary 或 run 配置。
审计 ID 不可覆盖。历史 report/run 和原控制组 summary 仍保留当时结算，查看修订
分类需读取新 audit receipt；真实重跑需新 control ID。


## 统一下载缓存

Terminal-Bench 的 verifier 会直接执行官方的 `curl`、`uvx`、Git 和 APT 命令，不能通过修改 verifier 来适配下载。`harbor resource-cache` 管理宿主机上的内容寻址 HTTP 缓存；公开网络任务容器通过只读 CA bundle 和代理环境变量接入，缓存私钥不会挂载进容器。缓存服务只缓存 GET/HEAD 和受限的 Git 下载请求，响应完整性由 SHA-256、大小和不可变索引校验。

典型流程是 `runtime-prepare`、`scan`、`prepare`、`start`、`prefetch`、`seal`、`publish`，然后以 `--mode replay` 启动封存副本。回放模式对未命中请求直接返回错误并记录 `denied`，不会偷偷连接上游。每个 run 的配置记录缓存 descriptor、CA 哈希和封存哈希；任务容器中的原始 verifier 命令保持不变。

现有封存目录可以直接用于 `resource-cache start --directory CACHE --mode record`。
首次启动验证封存内容，将已有响应引入 `CACHE/record-store`。在线下载、APT 索引刷新和请求日志都写入这个可写目录。
后续 record 启动会继续使用已有的录制结果。record 和 replay 共用目录指定的代理端口，切换时先停止旧服务。
record run 的 `harbor.task_resource_cache` 必须指向 `CACHE/record.json`，replay 则使用 `CACHE/replay.json`。
如需发布在线补录结果，可将 `CACHE/record-store` 传给 `publish --directory`，并按需通过 `--base-directory CACHE` 保留原有封存响应。

下载补录使用 `resource-cache warm`，它从私有请求日志读取未完成请求，并保留原始
User-Agent、Range、认证变体及 Git 下载 POST 内容。单纯用宿主机 curl 下载同一个
URL，不一定能命中容器里 uv/curl 的请求键。日志不会发布到 sealed 副本中。

```bash
.venvs/runner/bin/dycon harbor resource-cache stop --directory CACHE
.venvs/runner/bin/dycon harbor resource-cache configure-upstreams --directory CACHE \
  --upstream-map huggingface.co=hf-mirror.com
.venvs/runner/bin/dycon harbor resource-cache start --directory CACHE
# 完成一次资源补录后：
.venvs/runner/bin/dycon harbor resource-cache warm --directory CACHE \
  --workers 4 --max-time 900 --attempts 3
```

replay 服务也会在私有日志中保存未命中的原始请求，不改变封存索引或内容。
可以将其中被拒绝的请求导入可写缓存，再按原始请求变体补录；命中完整文件后派生的
Range 请求不会被当作缺失资源导入。`warm-last.json` 记录 curl 退出码和 HTTP 状态码。

```bash
.venvs/runner/bin/dycon harbor resource-cache warm --directory CACHE \
  --requests-from PREVIOUS_REPLAY_CACHE --workers 4 --max-time 900 --attempts 3
```

PyTorch 公共附加索引对不存在的包可能返回 403；限定在该站点的包索引路径时，缓存
保留原始 403，以允许 pip/uv 按原有规则继续查找主索引。授权请求、任意 403 文件下载、
其它站点和带 Set-Cookie 的响应不适用此例外；Cookie 请求变体仍独立保存。

镜像路由保留原 URL 的缓存键，并写入 descriptor；修改前必须停止可写缓存服务，
修改后创建或通过 `run revise-config` 冻结新 run。历史 run 保留旧 descriptor 哈希。
公开网络任务统一使用 uv/pip 600 秒下载超时与重试、HF 元数据/下载超时，并关闭
HF Xet 以让下载经过可缓存的 HTTP 路径；APT 统一配置重试和连接超时。

record 模式会刷新 APT `/dists/` 下可变索引并保留完整签名内容，避免永久复用旧索引
后下载已下架的 Debian 包。内容寻址的 `/by-hash/` 和包文件继续复用已有资源。
sealed replay 固定已记录索引和内容，未命中仍拒绝访问上游。404/410 资源响应也会
保留，以便缺少可选模型文件时继续遵循官方客户端的回退行为。

`warm` 的完成数仅说明请求已缓存；最终覆盖范围应由新容器中的真实 oracle/verify
以及 replay 的 `denied` 事件确认。缓存不能修复参考解本身或编译/权限等环境错误。

当官方大文件源极慢时，可从同一官方 URL 分段下载，校验完整文件后，通过统一接口
按已记录的原始请求键导入。`admit` 强制校验 SHA-256，仅接受完整 GET 和 identity
传输编码，禁止覆盖已有不同内容或修改 sealed 缓存；文件内容不会作为目录挂载给任务。

```bash
.venvs/runner/bin/dycon harbor resource-cache admit --directory CACHE \
  --request-key RECORDED_KEY --source-file VERIFIED_FILE --sha256 EXPECTED_SHA256 \
  --content-type application/x-gzip
```

完整文件缓存可响应同一请求变体的单段 HTTP Range 下载，包括后缀范围和 416；
If-Range 不匹配时返回完整 200 响应。认证、Cookie、User-Agent 等变体继续隔离。
`warm` 会移除条件验证头，以补录可独立回放的完整响应，避免仅拿到 304。

HF 主站及其 CDN 的 User-Agent 中只有随机 `session_id` 遥测字段被规范化；库版本、
其它请求头、完整签名 URL 继续隔离。CORS 的 Origin/预检 Vary 字段也加入请求变体，
确保 CDN 权重正文可缓存。`warm`/`publish` 自动为已有私有请求日志建立兼容键。
HTTPS 配置关闭上游证书预探测，缓存命中无需先解析目标主机或连接上游；上游 TLS
证书校验仍开启。回归测试使用无法解析的 `.invalid` 域名验证这一行为。

Debian 11 的官方 security 索引和带 deb11 版本号的 security 包已迁至 archive 时，
代理将原始请求转到 `archive.debian.org`，保留 URL 缓存键和 APT 的签名/哈希校验；
其它 Debian 版本不受这条路由影响。历史缺包 404 会在 record 中被已完成包替换。
任务本地服务地址 `0.0.0.0` 与 localhost 一样绕过代理。

这里的 strict replay 指代理不访问上游；不代表容器或嵌套 QEMU 来宾的所有网络流量
都被拦截。未使用 HTTP 代理的程序或虚拟机来宾下载需要单独核验，不能根据代理命中
数宣称整个任务完全离线。


## 统一任务环境修复与版本化参考解（2026-11-17）

`harbor.task_preparation` 是显式、随 run 配置和 execution context 冻结的任务准备策略。
键为任务短名，`*` 为默认配置；任务项覆盖默认项。同一实现用于 rollout、oracle 和
verifier 容器，不预先生成答案。旧 run 未设置该字段时维持原行为。

```yaml
harbor:
  task_preparation:
    '*':
      cpu_affinity: true
    git-multibranch:
      cpu_affinity: true
      sshd_runtime_directory: true
    mcmc-sampling-stan:
      cpu_affinity: true
      system_packages: [cmake]
    build-cython-ext:
      cpu_affinity: true
      python_constraints: [planarity==0.6]
```

`cpu_affinity` 使用真实 Linux taskset：限制容器 init 和统一 exec 路径及其子进程，
不替换 nproc。CPU 集合依据 ownership 在宿主允许集合中选择，并在环境收据记录。
这不会授予 rootless Docker 缺失的 CPU controller；`cpu_quota: unenforced_with_receipt`
的 quota 偏离仍如实保留，任务可用 CPU 数另由 affinity 限制。内存限制保持原 task 值。
`resource-final.json` 在评分结束、清理前保存可读取的 cgroup 内存/OOM/CPU 信息。
`sshd_runtime_directory` 仅检查并修正 `/run/sshd` 为 root:root/0755，拒绝符号链接，
不批量修改 /run，也不启动服务。`system_packages` 经相同 APT 缓存安装，收据记录版本。
`python_constraints` 只接受精确 package==version，经 PIP_CONSTRAINT/UV_CONSTRAINT 共享。
依赖准备改变了任务初始条件；报告必须标记环境适配。

可选 `post_verifier_diagnostics: {artifact-name: shell-command}` 在官方评分完成后、清理前
运行有界诊断，输出单独 `diagnostic-NAME.json`；明确标记不是评分证据。

共享代理支持按相同变体从 GET 派生 HEAD/单段 Range。认证、Cookie、User-Agent、编码
和请求体隔离保留。`resource-cache admit --request-key KEY --source-file FILE --sha256 SHA`
也接受已记录的 HEAD/Range，向同一变体完整 GET 映射已校验文件，再派生协议响应；
不会把完整文件错误地作为 Range 正文返回。`warm --request-key KEY` 可针对单个大 Git
POST 补录，Git 慢传输容忍窗口与显式 max-time 一致，最高3600秒。

仅 `https://data.rcsb.org/graphql` 的合法 GraphQL query 文档允许按完整请求体缓存；
通过 GraphQL parser 拒绝 mutation/subscription/batch/其他域名。公共 PubChem 的两个
只读 TXT 查询端点在无认证/Cookie请求时可移除响应 CDN Set-Cookie 后缓存，并记录原始
响应头哈希与去除动作；其它端点和私有变体仍保持原限制。

缓存服务启动时把 proxy/store 代码复制到内容哈希命名的 service-code 目录，实际运行
不可变副本并验证 addon-ready 回执。修改项目源码不再触发活动代理热重载。必须等活动
下载结束后，通过正式 stop/start 启用新版本；启动插件未就绪会停止服务，避免静默直通。

当官方参考解本身不兼容时，使用独立发布，不改原始 catalog/snapshots 或 verifier：

```bash
.venvs/runner/bin/dycon harbor reference-publish RUN \
  --task-id TASK --reference-id REFERENCE_ID --source temp/repaired-solve.sh \
  --reason '具体兼容性修复原因'
.venvs/runner/bin/dycon harbor controls RUN --control-id CONTROL --agent oracle \
  --task-id TASK --reference-ref catalog/reference-solutions/REFERENCE_ID/manifest.json
```

发布复制独立任务快照，校验差异严格只包含 solution/solve.sh；原任务定义、镜像、资源、
输入、instruction 和全部测试均不可改变。manifest 记录原/新脚本与包哈希、原因和来源。
父进程和 worker 均验证发布身份。control summary 的 scope 标记 derived reference，
每个 attempt 保存 reference-revision.json。该结果必须与原始官方 oracle 分开报告。

### 剩余 8 项最终封存验收（2026-11-17）

最终封存缓存为 `/data3/wangtianyi/terminalbench-remaining8-cache-sealed-20261117-v2/replay.json`，4465 条响应，字节合计 16285392172；seal SHA256 为 `fa63e986a4d1e2a7ca8bd11b99f82a368a83e093fcb6d3d0ed4015ab57f1fded`。评测后完整性校验通过。

`tb21-remaining8-sealed-replay-20261117-v2` 的 6 个官方参考解 control 均 reward=1：SAM、OCaml、RStan→PyStan、Windows 3.11、Git 多分支、Cython；SAM 两次新容器验证均通过。`tb21-remaining8-derived-replay-20261117-v2` 的 protein、MCMC 两个独立修订参考解 control 均 reward=1，测试文件不变。不能将修订结果宣称为原始官方 oracle 8/8。

MCMC 修订 publication：`catalog/reference-solutions/rstan-no-dependency-upgrade-20261117-v1/manifest.json`。仅对 remotes 安装调用添加 `upgrade='never'`，保持原始参考解声明的 StanHeaders/RStan 版本。

protein 修订 publication：`catalog/reference-solutions/protein-parent-components-20261117-v2/manifest.json`。仅修复原始 entity sequence 中修改残基的 CCD parent component 展开。两个发布包均仅 `solution/solve.sh` 与原始快照不同。

复制正式 run 可复用最终环境：

```bash
.venvs/runner/bin/dycon run create <new-run> --operation rollout \
  --from-run tb21-remaining8-sealed-replay-20261117-v2
```

普通模型 rollout 使用原始说明、输入和 verifier；`--reference-ref` 仅在显式检查修订 oracle 时使用。最终 task_preparation 包含全任务 CPU affinity、git 的 sshd 运行目录、MCMC 的 cmake、Cython 的 `planarity==0.6`。rootless CPU CFS quota 仍为 `unenforced_with_receipt`；CPU affinity 生效，内存限制未放宽。

最终 9 个 control（含 SAM 重复）均清理完毕、无残留。回归测试 72 passed。来源回执及逐项结果存于 `temp/remaining8-delivery-20261117/`，其中 `RESULTS.md`、`results.json`、`final-run-patch.yaml` 可供复用与审计。代理仍有非阻塞的额外索引/地址探测 denied/error，全部最终任务在禁止回源模式下完成；不宣称任意未来版本请求均已缓存。


## 全量 Oracle 回归修复（2026-11-17）

CPU affinity 的统一 exec 包装保持锁定 Harbor 的 `self._platform.exec_shell_args`
语义；Linux 主容器继续使用 Bash。无首行 shebang 的官方脚本依然按 Bash 回退解释，
显式 `#!/bin/sh` 则继续由其声明的解释器执行；不修改官方任务脚本。

官方 verifier 命令被异常或取消打断时，适配层在清理前最多用 15 秒回收 256 KiB
stdout 尾部和有界进程诊断，保存 `verifier-interrupted-stdout.txt`、
`verifier-interrupted-processes.json` 及 `verifier-execution.json`。
这是部分诊断材料，不是奖励文件，不合成 reward，也不延长官方 verifier 的测试预算。
原异常仍向上传播。

`torch-tensor-parallelism` 在硬单 CPU affinity 下可复现 Gloo
`destroy_process_group()` 挂起。经用户授权，对这个任务显式配置：

```yaml
harbor:
  cpu_quota: unenforced_with_receipt
  task_preparation:
    '*': {cpu_affinity: true}
    torch-tensor-parallelism: {cpu_affinity: false}
```

关闭 affinity 不等于实施官方 1 CPU 配额。当前 rootless Docker 未委派 CPU controller，
该任务可以在宿主允许 CPU 集合上调度；偏离在资源收据中保留。其余内存、测试内容和
900 秒 verifier 超时保持原任务值。此例外适用于相同配置下的 rollout/oracle/verifier。

公开 PyTorch wheel/PEP 658 metadata 的 Cloudflare 响应可能带附带会话 Cookie。
缓存仅在明确的 `download.pytorch.org` / `download-r2.pytorch.org` HTTPS wheel/metadata
路径、GET/HEAD 200、无 query/请求认证/Cookie 时去除响应 Cookie，并写审计收据。
不扩展到目录索引、403、其它域名或私有请求。资源正文、User-Agent、编码和请求体隔离保留。

`tune-mjcf` 的原始 PGS 参考解存在速度阈值波动；原始记录不覆盖。可选独立参考解
`catalog/reference-solutions/mjcf-implicitfast-20261117-v2/manifest.json`
只修改 `solution/solve.sh` 生成的模型，加入 `integrator="implicitfast"`；
原始物理属性、参考模型、task 配置和 verifier 均不修改。
必须通过 `harbor controls --reference-ref ... --task-id tune-mjcf` 显式启用，
默认 rollout 评分和默认官方 oracle 不替换为该参考解。


Oracle 健康策略 v3 也检查 reward=1 的参考脚本退出回执。明确非零退出使控制组归为
`oracle_reference_failed`，保留原始满分 reward；不可读退出回执归为 Review。
普通 rollout/nop 的上游成功语义不变。v1 MuJoCo 修订发布曾因 BOM/CRLF 导致脚本失败，
即使模型通过原始 verifier 也不应计为有效正控制；使用无 BOM/LF 的 v2 发布。
历史 summary 不覆盖，可通过 `harbor controls-audit` 生成新策略复核回执。

当前默认配置使用封存缓存
`/data3/wangtianyi/terminalbench-full89-cache-sealed-20261117-v6/replay.json`。
它完整保留 v4 的全部响应和元数据，追加 tmux 准备所需资源。旧 frozen run 不自动读取新的默认配置，
需克隆并通过 `run revise-config --patch temp/full-oracle-repair-20261117/final-run-patch-v6.yaml`
显式冻结后再创建新 control ID。


`git-multibranch` 的官方镜像未安装参考脚本末尾使用的 tmux。v3 退出码审计发现其
历史 raw_reward=1 但 reference exit=127；统一 `task_preparation.git-multibranch`
在既有 sshd 目录准备之外增加 `system_packages: [tmux]`，不改官方参考脚本。

增量补录可能刷新 APT 可变索引。对已经验证的缓存，应通过显式基线保留已有响应：

```bash
.venvs/runner/bin/dycon harbor resource-cache publish --directory WORKING_CACHE \
  --base-directory SEALED_BASELINE --destination NEW_SEALED_CACHE \
  --endpoint http://HOST:PORT
```

基线必须封存并通过完整性校验；其全部请求键/响应元数据优先保留，只从补录源添加
新请求键。冲突不会覆盖原已签名索引，真实字节和签名均保持原样；发布溯源写入
`publication.json` 并纳入 seal 验证。默认不传基线时保留原发布行为。


### TerminalBench capture 请求角色资格核验（2026-11-18）

`dycon harbor qualify RUN_ID --profile-id PROFILE_ID` 从冻结 TerminalBench/OpenHands run 的真实请求、ACP 事件和工具调用/返回证据发布不可变的 driver profile；`--profile-ref` 可指定兼容目标。该接口复用现有资格核验，保留 SkillsBench 的历史 profile 契约。Harbor 请求必须带有 `backend_family: harbor`，凭据绑定冻结 worker protocol/runtime；禁止直接继承 SkillsBench 凭据。

本次 `tb21-openhands-request-roles-20261118-v3` 来源于 `tb21-gpu-smoke-20261115-v2` 的30次请求、26对实际工具调用/返回。capture run 应通过 `run revise-config --replan` 冻结新 profile，而不是修改已发布 profile 或 run 证据。`images.load_workers` 现在允许1至12，使用已有 NAS 镜像加载槽位机制。


## 长耗时检查开关

Harbor/OpenHands 的 Terminal-Bench 2.1 和 SWE-bench Pro V2 均读取冻结的
`execution.preflight.long_checks`；缺省为 `true`。创建/克隆 run 的 context 校验、
`run start/resume` 及独立 `dycon harbor preflight RUN_ID` 使用相同策略。

| 检查 | false 时的行为 | 保留的校验 |
| --- | --- | --- |
| 模型内容 SHA | 跳过已登记模型的逐文件内容哈希 | 登记身份、清单、路径、存在性、大小；model.verify_identity 独立控制 |
| 任务包内容 SHA | 跳过逐文件内容哈希 | 冻结 manifest 哈希、文件集合、路径、存在性、模式；旧任务清单未声明文件大小 |
| OCI 深度 SHA | 镜像准入和导入均传递 deep=false | NAS manifest 哈希、冻结镜像身份、完整 descriptor 图、路径、存在性、大小及 Docker identity |
| OpenHands 依赖快照 SHA | 跳过逐文件内容哈希 | receipt、profile 绑定 manifest SHA、路径、存在性、大小、文件清单和运行契约 |
| Harbor worker 发布物 SHA | 跳过依赖文件及解释器内容哈希 | 发布版本/revision、receipt、路径、存在性、大小和可执行解释器 |
| 全量容器环境审计 | 不创建逐任务审计容器，不运行依赖安装、ACP、model-channel 或 network 探针 | 启动的 profile/context/CPU quota 契约；正式 attempt 仍执行实际环境安装与探针 |

`validation/harbor-preflight.json` 的 `checks` 逐项记录状态、耗时和跳过原因
`execution.preflight.long_checks=false`。模型启动回执同时来自
`validation/startup-model-identity.json`。跳过的检查不记为 verified，空的
`tasks` 审计列表不代表 89 项环境均已通过审计。

开关只降低启动前审计深度。正式 worker 的任务包完整校验、依赖安装验证、
Compose 镜像身份、环境探针、官方 verifier、capture、清理与取消保持执行；
oracle/nop 等显式控制组也不跳过实际执行。GPU/端口资源准入仍在 serving 启动路径执行。
运行中的 Python 进程不会热加载修改；当前 run 已执行的预检证据不改写。
