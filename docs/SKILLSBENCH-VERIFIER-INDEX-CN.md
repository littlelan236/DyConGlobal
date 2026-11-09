# SkillsBench verifier 包索引策略

SkillsBench 的 task image 构建索引与 verifier 运行索引分别冻结。根 `config.yaml` 的
`skillsbench_images.python_index` 只用于新镜像构建；运行时不把它作为 fallback。
`OpenHandsProfile.verifier_python_index` 决定新 run 的 verifier 索引。

```bash
.venvs/runner/bin/dycon skillsbench profile NEW_PROFILE \
  --from-profile runtime/agent-profiles/openhands-sdk-v4-qualified-main-20261101/manifest.json \
  --verifier-python-index https://mirrors.aliyun.com/pypi/simple/ \
  --verifier-index-compatibility generic-pypi-v1
.venvs/runner/bin/dycon skillsbench index-variant NEW_SNAPSHOT \
  --source-snapshot catalog/snapshots/skillsbench-agent75-oracle-passed-20261029/manifest.json
```

上述命令发布新的不可变 ID，不修改源 profile、任务包或旧 run。复制保留 output paths、
memory/timeout 等适配，清除旧 request-role qualification；新 profile 须用新的实际工具调用
run 和 `skillsbench qualify` 建立资格。`--verifier-timeout TASK_ID:SECONDS` 可显式发布新的
verifier 预算；创建 profile 不会启动 workload。根配置随后选择新的 `benchmark.snapshot_ref`
和 `agent.profile_ref`，通过 `run create` 冻结新 run。

public 有/无代理均设置 `PIP_INDEX_URL`、`UV_INDEX_URL`、`UV_DEFAULT_INDEX`；不改变 TLS 校验。
已发布 uv 是 0.13.5，其 `UV_DEFAULT_INDEX` 优先于 `UV_INDEX_URL` 及旧式 `--index-url`。
SimPO 是明确的专用索引例外：设置空 `UV_DEFAULT_INDEX` 保留脚本的 CPU PyTorch CLI 索引，
仅将通用 PyPI extra index 改成读取 `DYCON_VERIFIER_PYTHON_INDEX`。
不更改 `torch==2.1.2+cpu`、仓库顺序或 uv 的 first-index 解析规则。某仓库存在同名但不兼容
版本时仍可能报版本冲突，这不应通过关闭安全解析策略掩盖。

经用户确认的新策略 `generic-pypi-cpu-first-v2` 保留上述 v1 发布物，另行发布任务变体。
它只将 SimPO 的两个索引参数换位：通用镜像作为 `--index-url`，CPU 专用仓库作为
`--extra-index-url`；uv 0.13.5 优先 extra index，因此 CPU 仓库先参与同名包解析。
继续使用默认 first-index，不启用 unsafe-best-match。真实隔离索引测试覆盖 v1 冲突与
v2 解析成功，原文件 hash、完整替换片段及有效文件 hash 继续由白名单 receipt 绑定。
选择新策略时须同时发布并选择新的 profile 和 snapshot，并重新建立工具调用资格。

缺省或 null 索引保留旧策略；旧 profile 标准化不会新增空字段改变身份。
`network:none` 拒绝非空在线索引，并保留 `PIP_NO_INDEX=1`、`UV_NO_INDEX=1`、
`UV_OFFLINE=1`、`UV_PYTHON_DOWNLOADS=never` 和无网络容器拓扑。
public 显式镜像遇到任务声明离线或专用默认索引时拒绝冲突，不全局删除任务的 offline/index 标志。

依赖快照的 `PIP_FIND_LINKS/UV_FIND_LINKS` 是本地候选来源，和在线索引共同参与解析，
不保证严格的“先本地后联网”。agent 私有 CPython 3.12 安装继续使用
`--no-index --require-hashes`；verifier 的在线解析不等同于离线 hash 闭包。

`generic-pypi-v1` 只允许 AgentOps 两个文件、Google Auto 一个文件、SimPO 一个文件的
精确原文件 SHA256 和唯一位置替换。变体保留上游任务包，发布有效文件清单、有效 package hash、
补丁原因和 receipt；selection、加载校验与 verifier 均绑定有效包。源 hash 不符时停止发布。
测试断言、依赖 pin、构建流程、通过条件和 reward 不变。

```bash
.venvs/runner/bin/dycon skillsbench dependencies-verify runtime/dependency-snapshots/openhands-sdk-format-v4-20261101
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon run create VALIDATION_RUN --operation rollout --tasks-file TASKS_JSON
.venvs/runner/bin/dycon skillsbench environment-check VALIDATION_RUN --workers 2
.venvs/runner/bin/dycon skillsbench preflight VALIDATION_RUN --workers 2
.venvs/runner/bin/dycon skillsbench oracle VALIDATION_RUN --workers 1
.venvs/runner/bin/dycon skillsbench empty-control VALIDATION_RUN --task-id dialogue-parser
```

`environment-check` 复用既有环境审计与 BenchFlow 生命周期，不请求模型；显式镜像时在
hardening/工具准备之后执行受限 pip/uv/uvx 取包探针。探针单独清空 find-links、缓存，
使用临时目录并清理；正式 verifier 的本地候选不变。收据在
`validation/environments/**/diagnostics/verifier-index-probe.json`，包含实际 wheel 请求证据、
解释器版本、输出和清理。环境成功不能代替任务全部依赖或 oracle 成功。

attempt 的 `verifier-environment.json` 记录 hardening 后的有效环境、网络/代理模式、
索引来源、profile/dependency hash、timeout 和已知覆盖例外；它只表示已配置。
实际安装/请求证据在 verifier stdout/stderr、`diagnostics/verifier-execution.json`
或环境探针 receipt 中，不能仅凭变量字符串宣称使用了镜像。收据只导出允许字段并隐藏 URL 认证。
`empty-control` 对全新未解任务执行同一个官方 verifier，零分且无基础设施错误才视为对照成功。

镜像配置不控制 apt、GitHub、Python runtime、模型/数据下载。分别记录网络失败、索引未生效、
镜像缺版本、版本冲突、ABI、非 PyPI 下载和 verifier 超时。已有 task image 无需仅因运行索引
变化而重建；若将来修改构建输入，使用新 image store 和 `skillsbench-images plan/prepare/verify`
发布新身份，不能把复用旧缓存当成新镜像已构建。

## 追加修复与验证证据（2026-11-05）

容器启动时，发布的 uv 工具路径前置到实际租赁镜像的原始 PATH，保留镜像内的
`/opt/venv/bin` 等任务解释器路径。verifier 继续使用 BenchFlow hardening 管理的可信 PATH。
`diagnostics/task-interpreter.json` 在 agent 安装或 oracle 执行前记录镜像解释器版本、
可执行文件和少量依赖模块位置；私有 agent 解释器仍由 `dependency-install.json` 单独证明。

oracle 验证在创建容器时使用冻结 profile 的显式 verifier 索引，避免 oracle 获取验证
依赖时重新访问默认 PyPI。配置记录在 `oracle-environment.json`，实际下载来源仍以安装
日志为准。这一分支只用于 oracle 验证，不改变模型 agent 的离线 hash 安装。

不支持 rootless CFS 的显式 `unenforced_with_receipt` 适配同时设置
`LOKY_MAX_CPU_COUNT` 为任务请求 CPU 数的向上取整值，限制 joblib 的自动进程数量。
`resource-adaptation.json` 记录该提示；它不提供 CPU 硬配额保证，其他程序也不一定遵守。

验证命令意外退出或 Docker 服务被终止后，可通过统一入口复核已停止的验证容器：

```bash
.venvs/runner/bin/dycon skillsbench validation-cleanup RUN_ID --section oracle
.venvs/runner/bin/dycon skillsbench validation-cleanup RUN_ID --section empty-control --task-id TASK_ID
.venvs/runner/bin/dycon skillsbench empty-control RUN_ID --task-id TASK_ID --retry-failed
```

清理取得对应 sweep 锁，核对 project/run/task/sample/attempt/epoch 标签，仅删除已停止且
身份一致的容器以及空的所属网络。遇到运行中的资源即拒绝，不删除镜像、卷或其他 run。
`validation/<section>/cleanup-reconciliations/` 中的新收据记录事后清理结果；原始失败的
`result.json` 和当时的 `cleanup_confirmed` 不会被改写。

Docker 不可达必须与单容器内存不足分别判断。2026-11-05 的服务日志记录了
`systemd-oomd` 杀掉整个用户 rootless Docker 服务；Mars 的独立记录同时包含
`OOMKilled=true` 与 cgroup `oom_kill` 计数。仅有返回码 137 不足以得出相同结论。

AgentOps 的原始 tox 矩阵包括 Python 3.7。现代 virtualenv 可能为它选择要求 Python
≥3.10 的 pip 26.1.2，造成 pip 导入失败。显式在线镜像策略为该任务设置兼容全部矩阵
解释器的引导版本：`VIRTUALENV_PIP=23.3.2`、`VIRTUALENV_SETUPTOOLS=67.8.0`、
`VIRTUALENV_WHEEL=0.42.0`；这些版本只控制 virtualenv 引导工具，不改变项目依赖 pin。
修正引导后，原 verifier 的 `langchain==0.0.352`、`langchain-core==0.1.23` 均要求
Python ≥3.8.1，与 py37 矩阵矛盾。此问题属于任务本身的依赖/解释器契约不兼容，不能
通过镜像源、跳过测试或忽略 Requires-Python 宣称修复。保留失败及元数据证据单独报告。

oracle、offline-oracle、empty-control 共享 run 所属代理，统一取得
`validation/.oracle-lifecycle.lock` 与各阶段 `.sweep.lock`；同 run 的其他验证阶段及
validation-cleanup 在锁被占用时拒绝启动。oracle attempt 标识同时绑定验证阶段，避免
不同阶段生成相同 compose project/container 标识。环境审计请与这些命令分开执行。

环境审计被中断时，`skillsbench validation-cleanup RUN_ID --section environments`
可核对嵌套的 contract/attempt 所属资源；它拒绝仍在运行的审计进程或容器，保留未完成
attempt 的原始内容，另写 reconciliation 收据。清理只接受已停止、身份完整的所属资源。
`cleanup-network` 当前仅处理终态模型 run，不能用它冒充环境审计进程取消接口。
