# Benchmark 容器环境修复指南

本文说明：什么类型的 benchmark 任务容易遇到哪些环境问题，以及如何沿项目已有入口定位、修复和验证。适用于代码修复、测试执行和 gold patch 环境验收；不按具体任务或数据批次组织。

核对日期：2026-10-21。以当前源码和本次运行的证据为准，尤其注意旧文档可能仍使用重构前的目录与命令。工程根目录为 `/home/wangtianyi/DyConGlobal`，NAS namespace 为 `/VM/wangtianyi/DyConGlobal`。

## 1. 修复必须经过统一容器边界

正式任务继续使用现有入口，**不得另写临时容器启动脚本、复制 launcher、直接运行内部 worker，或为评测单独维护一份环境安装逻辑**。否则 agent 自测、gold 检验和最终 harness 可能运行在不同环境中。

| 用途 | 当前入口或实现 | 职责 |
|---|---|---|
| 新建、执行和恢复正式实验 | `dycon run create` / `dycon run resume` | 冻结配置、任务选择和执行计划 |
| 冻结轨迹并运行官方评测 | `dycon harness prepare` / `dycon harness run` | 绑定确切 rollout、patch、配置和评分证据 |
| 无需模型 rollout 的 gold 环境审计 | `python -m runtime.scripts.gold_image_audit` | 读取显式实例清单，用 gold patch 验证环境 |
| 容器创建、依赖准备、命令执行、清理 | `runtime/containers/factory.py::ContainerFactory` | rollout 与官方 harness 共用的容器生命周期 |
| 按任务适配依赖、线程和网络 | `runtime/containers/repairs.py::environment_overrides` | 统一修复规则，不直接创建容器 |
| 安装策略、环境变量和依赖探针 | `runtime/containers/policy.py::EnvironmentPolicy` | 离线 bootstrap、包版本约束、policy hash |
| NAS 镜像加载和资源协调 | `runtime/images/nas.py` | descriptor/SHA 校验、load slots、lease、磁盘预约、owned cleanup |
| 镜像文件所有者适配 | `runtime/images/ownership.py`、`runtime.scripts.repair_image_ownership` | 生成保留来源关系的派生 OCI 镜像 |

配置和数据的事实来源：

- 根 `config.yaml`：唯一可编辑的共用配置。
- `catalog/manifest.json`：当前 `catalog.store.Catalog` 使用的任务、分区、轨迹与评分索引。旧 `catalog/manifest.json` 不应作为新版运行时的写入目标。
- `runs/RUN_ID/`：不可变 plan、配置快照、任务选择与本次执行产物。
- `instance.environment_overrides`：随任务保存的已有环境适配。
- `repairs.environment_overrides`：在共享工厂中应用的修复补充。

实际优先级为：共用 environment → 任务已有 overrides → 修复函数返回的 overrides → `Environment` schema 校验 → `EnvironmentPolicy`。当前工厂对顶层 environment 做字典更新，**嵌套的 `variables`、`repo_pins` 不会自动递归合并**；修复函数必须显式保留应继承的项。

运行快照不等于冻结了以后所有 Python 源码。修复规则在创建容器时参与计算，应同时保存 repairs/factory/policy/harness 的代码身份，以及实际 `environment-policy.json`。修改实时配置或源码不能被描述为一次严格复用原环境的 resume。

## 2. 先判断失败发生在哪一层

| 观察到的现象 | 首先检查 | 常见误判 |
|---|---|---|
| NAS preflight 失败，还未创建容器 | NAS 挂载、namespace、manifest SHA、blob 大小与图结构 | 将目录错误当成依赖缺失 |
| load/create 失败 | 镜像兼容性、rootless UID 映射、磁盘、Docker API、并发 load | 反复安装 pip 包，或反复重试确定性的 layer 错误 |
| bootstrap/probe 失败 | wheelhouse、Python ABI、约束冲突、包版本和安装输出 | 在容器里手动装好就认为永久修复 |
| 测试收集或 import 失败 | 插件、历史测试框架版本、本地扩展、系统动态库 | 将缺 fixture 当成业务代码 bug |
| 测试运行后失败 | 原始断言、网络响应、资源内容、版本行为差异 | 所有失败都归因为网络，或都归因为模型 patch |
| 命令或整轮超时 | 具体命令、测试进度、线程栈、CPU/内存、网络等待 | 直接增加全局 timeout，掩盖确定性卡死 |
| agent 没有提交 | trajectory 和终止原因：上下文、步数、格式、命令错误 | 将 agent 终止与容器网络故障混为一谈 |

首先保存触发命令、完整输出、错误栈、实际包版本、镜像 digest 和阶段。不要只读最后一句 `failed`。

优先读取：

- `container-spec.json`：实际镜像、网络、挂载、环境变量。
- `environment-policy.json`：effective constraints、pins、wheel hash、bootstrap 和 policy hash。
- `bootstrap.log`、`environment.json`、`environment-after-eval.json`：安装与依赖探针证据。
- `container-events.jsonl`、image lifecycle 日志：load/create/start/remove 的阶段与归属。
- harness 的 `run_instance.log`、`test_output.txt`、`report.json`：执行错误与官方评分要求。
- canonical trajectory：模型实际发出的命令及工具输出。源码中的 `HTTPError`、故意坏链接、mock 异常不能直接当作真实网络故障。

新流程的产物位于对应 run 下，维护审计另有 `state/gold-image-audit/AUDIT_ID/`。以命令返回的路径及 receipt 定位，不按旧目录猜测。

## 3. 镜像、NAS 和容器启动问题

### 3.1 NAS 路径、校验和与镜像引用

所有任务必须有不可变 `nas_image` 引用。本地 tag 名相同不代表镜像相同。

排查顺序：NAS 是否挂载 → `paths.nas_root` 是否与相对 `store` 匹配 → manifest 路径/SHA → image ID → descriptor graph → blob 大小/SHA。特别注意维护审计的默认 NAS 根可能与工程配置不同：应显式传正确的 `--nas-root`；若使用 `--config-snapshot`，实际 NAS 根由快照决定。

修复目录或引用契约，不能忽略校验或改掉 checksum 让损坏文件“通过”。`images-check --deep` 可校验所选任务镜像，实际流式 Docker load 也会校验内容。数据损坏时复用既有导入、下载、续传或发布能力；不要改用未经来源绑定的 `docker pull latest`。

### 3.2 rootless Docker 无法解压高 UID/GID layer

典型任务是依赖原生构建工具或历史构建产物的项目。上游镜像里文件 owner 的 UID/GID 可能超出当前 rootless user namespace，出现 `Lchown ... invalid argument`、`subordinate IDs` 等错误。

这发生在容器创建前，pip 版本和 `--user root` 都不能解决 layer 解压映射限制。先核对实际 namespace/subuid/subgid，确认错误的文件及 ID。

项目已有处理方式：用 `runtime.scripts.repair_image_ownership` 从 NAS 原镜像生成**派生镜像**，只修无法映射的 ownership 元数据。保留原始 blobs、文件内容、路径、mode、链接和 whiteout；更新相关 layer digest、config diff IDs、manifest，并保存 `source_nas_image` 与 repair provenance。

示例（占位路径必须替换为冻结输入）：

```bash
.venvs/runner/bin/python -m runtime.scripts.repair_image_ownership \
  --root /home/wangtianyi/DyConGlobal \
  --instances INPUT_INSTANCES.json --only-file SELECTED_IDS.json \
  --store NEW_REPAIRED_STORE
```

生成后，使用派生 store 输出的实例清单做正式 cold gold。只有原始 patch 与派生 digest 的验收确实通过，才用已有 `--publish-summary GOLD_SUMMARY.json` 发布对应引用。原始 patch、test patch、任务身份与分区不能随镜像修复一起改变。

owner 修复会改变文件 owner 语义。若任务本身涉及权限、用户身份或所有者断言，必须额外验证语义，不能把它当成对所有任务都无影响的转换。现有工具也不提供任意 ABI、系统包或 Python 环境重建。

### 3.3 磁盘、load 并发与瞬时 Docker 错误

NAS 压缩字节小于 Docker 解压空间需求。镜像 load slots、lease 与磁盘预约由 `ImageLeaseManager` 协调，不应通过增加任务并发绕过 admission。

空间不足时先确认谁持有 lease、是否有活跃容器、预留空间是否合理；只回收已确认归属本工程且不再使用的产物。API reset/broken pipe 可使用工厂已有的有界 create retry；UID/GID、磁盘不足和镜像校验错误需要修根因，不能仅因为表现为 HTTP 500 就认为会自行恢复。

`dycon recover` 会实际删除判定为孤立的本工程容器并回收 owned image，**不是只读检查**。先核对活动 run 锁及需要保留的调试产物，尤其是专用维护进程。不要运行全局 prune，也不要手工删除其他实验使用的镜像。

## 4. Python 依赖、插件和历史版本兼容性

### 4.1 原生扩展与科学计算任务

常见于 NumPy/SciPy、VTK、图像或科学计算项目：Python 包存在，但 native ABI、运行库、Python 版本不匹配，表现为 import failure、二进制不兼容、属性变化或数值/图像断言变化。

处理方式：

1. 读取任务镜像内的 Python 和实际依赖版本，核对错误由缺包、ABI 还是 API 变化造成。
2. 按任务版本选择兼容的依赖范围或精确 pin，例如限制大版本升级、给旧项目固定兼容的测试框架或渲染库版本。不要对整个同仓库任务盲目升级/降级。
3. 取得匹配容器 Python、ABI 和 Linux 平台的完整 wheels，固定版本及 SHA；不能把 Windows wheel 或仅适配宿主 Python 的包拿给任务容器。
4. 将修复放入任务 overrides 或 `repairs.py` 的 task policy，使用已有 wheelhouse 挂载和离线 bootstrap。缺少系统运行库或没有可用 wheel 时，需要受维护的派生镜像构建能力；不能伪装成 pip pin 已修复。

项目已有的方式包括 NumPy/SciPy 配对约束、旧 pytest 兼容、VTK 范围约束、Matplotlib 版本固定，以及仅对缺包任务补充小型依赖。实际选用哪个版本，应按本次任务的源码与探针决定。

### 4.2 缺 pytest fixture 或插件

`fixture 'httpbin'` 递归依赖、`fixture 'mocker' not found` 可能是本应由插件提供的 fixture 缺失。检查测试定义、插件列表、历史 requirements 和 pytest 版本，而不是直接删除测试参数。

- 本地 HTTP 服务测试可由兼容的 `pytest-httpbin` 提供；还需固定它依赖的 httpbin/Web 框架、模板库和底层包。
- mocker fixture 通常需要核对 `pytest-mock` 是否存在及其兼容范围。
- 历史 `pytest.raises` 调用方式、warning API 或 fixture 行为可能与现代 pytest 不兼容。应恢复兼容的测试运行环境，不随意改动 benchmark 测试代码。

插件解决本地服务或 mock 能力，不保证整份 suite 完全离线：测试仍可能从本机服务重定向到公网。

### 4.3 按现有方式提交一个依赖修复

在 `repairs.environment_overrides` 中以 instance ID 或有证据支持的版本范围返回 overrides；不要把“一个任务需要此 pin”推成整个 repository 的默认行为。以下为结构示意，不是直接追加的生产配置：

```python
overrides = dict(instance.get("environment_overrides") or {})
pins = {**cfg["environment"]["repo_pins"], **overrides.get("repo_pins", {})}
pins[instance["repo"]] = compatible_specs
overrides["repo_pins"] = pins
overrides["wheelhouse"] = str(project_wheelhouse)
return overrides
```

保留同仓库其他必需 pin，并处理包名大小写/连字符归一化及重复约束；不要把嵌套字典覆盖造成的缺失误判为新的安装故障。约束与 wheel 的更改都会影响 policy 身份，应保存实际安装结果。

`EnvironmentPolicy.bootstrap` 默认使用 `pip --no-index --find-links /opt/dycon-wheels`，约束落到 `/tmp/dycon-constraints.txt`，实际创建时执行 probe。`network_fallback` 管的是**依赖安装 fallback**，不等于任务测试能联网。不能用开启 pip fallback 代替补全离线 wheelhouse。

## 5. CPU 并行、耗时与测试输出问题

### 5.1 旧 OpenMP/BLAS 任务超时

历史科学计算测试可能看到整台宿主的 CPU 数，在并发任务下产生严重线程过量订阅。先观察具体测试位置、CPU/线程数和执行阶段，区分本地运算慢、扩展重编译与网络等待。

对确有此问题的任务，通过共享 policy 合并设置 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`，需要时包含 `NUMEXPR_NUM_THREADS`。初次验收用小并发和保守线程数，检查速度与结果；不应更改训练/推理进程或无关任务的全局环境。

先修线程/资源问题，再依据实测耗时调整任务超时。`124/137` 或空输出只能说明命令被终止，不能证明是网络、OOM 或死锁中的哪一种。

### 5.2 输出、颜色与 tox 环境传播

某些测试执行完成，但官方 parser 无法从日志识别所需测试，可能与 `-rA`、verbose、ANSI 颜色、tox 子环境变量传播有关。项目已有 task-level `PYTEST_ADDOPTS`、`TOX_TESTENV_PASSENV` 的适配方式。

先比较 parser 输入和实际 test IDs，确认执行的是原定测试。不能为了填满 report 而虚构 PASSED 行、把 skip 改成 pass 或改评分集合。

管道也会掩盖失败：`pytest ... | head` 的 shell 返回码可能来自 head。诊断时保存完整 stdout/stderr 和 pytest 返回码，例如使用 `set -o pipefail` 后再按需展示节选。gold 验收以官方 report 和完整执行证据为准。

## 6. 网络问题：先区分资源下载与服务行为

### 6.1 缓存覆盖什么

| 缓存类型 | 能解决的主要问题 | 覆盖不了什么 |
|---|---|---|
| NAS OCI 镜像 | 重复获取镜像和任务预装环境 | 测试运行后新发出的 HTTP 请求 |
| 本地 wheelhouse | pip 依赖安装 | 数据集、网页、认证服务和 DNS |
| requirements 元数据缓存 | harness 根据 repo/commit 读取的真实 requirements 内容 | 容器内任意网络访问 |
| 数据文件缓存 | 固定 CSV、归档、图片、科学时间表 | 实时认证、重定向、timeout、页面状态以及未接入缓存的 URL 路径 |
| 本地 HTTP fixture | 测试原本支持的本地 HTTP 行为 | 跳往公网的下一跳、未模拟的服务语义和真实外部站点 |

当前 `ContainerFactory.spec` 只将 wheelhouse 挂到 `/opt/dycon-wheels`，**没有通用的任务资源缓存挂载接口**。下载进一次容器 writable layer 的文件，不会自动写回 NAS 镜像，也不会自动出现在下一次 cold start。

### 6.2 通常可以用资源缓存改善的任务

典型任务：读取公开科学数据集、DICOM 测试数据、PyVista 示例网格、固定网站图片，或依赖可固定的天文时间表。

修复步骤：确认实际 loader 与 cache path → 收集完整资源和相关索引 → 验证原始来源、内容 SHA、许可及有效期 → 通过共享 policy/factory 的可维护资源机制或受控派生镜像提供 → 用全新容器断网验证所需测试确实使用缓存。

缓存改造不是仅执行一次下载。还需考虑：用户 HOME、文件权限、目录层次、下载索引、压缩包解压、哈希校验、时间表过期策略，以及测试是否刻意清空缓存。现有接口不足时应扩展工程共享模块和配置 schema，不能在某个评测 launcher 里私自加 volume。

### 6.3 无法仅靠文件缓存解决的任务类别

**类别一：HTTP 客户端/协议行为测试。**

任务验证认证、Cookie、GET/POST/HEAD/PUT 回显、流式读取、重试、timeout 或跨主机 redirect。旧版 Requests 项目中较常见。静态响应文件无法自然保留有状态会话、请求参数、延迟、协议错误与 Authorization stripping 等语义。

如果原始测试支持本地 httpbin/socket fixture，应使用其原生能力；若仍请求真实外部服务，则需要稳定可用的出口或经过语义验证的服务本地化。**本地 httpbin 返回的 Location 仍可能指向公网**，必须核对完整请求链。

**类别二：真实网页/linkcheck/TLS 状态测试。**

部分历史文档构建任务检查真实站点是否 working、anchor 是否存在、404、重定向及错误信息。旧 Sphinx linkcheck 是典型例子。网站内容变化与网络故障均可能造成评测失败；存下 HTML 不会自动替代真实的 HEAD/GET/TLS 行为。

不能据此否定所有 Sphinx 任务：较新测试可能已使用 localhost，普通解析/构建测试也可能完全不需要公网。按测试版本、fixture 和官方必需范围判断，不按仓库名称统一处理。

**类别三：混合下载、目录枚举和连通性检测。**

例如数据工具同时验证 dataset names、下载/缓存、缺失数据集错误路径，并在模块加载时检测 internet。即使 CSV 已缓存，网络 gate 仍可能 skip 所有必需测试；目录枚举或人为清空缓存也会重新访问外网。

必须覆盖全部必需行为，才能称为离线可测。若只有静态文件可替代，而协议或检测仍依赖外网，应明确保留混合依赖标签。

这些类别中的“无法缓存解决”是指**现有测试保持不变时，单纯缓存文件、wheel 或镜像不够**。经过维护的本地服务、协议回放或资源适配可能消除公网依赖，但属于新的环境契约，需要来源记录与正反例验证，不能把伪造的成功响应混入原始 benchmark 结果。

### 6.4 判断是否需要代理

在实际任务客户端和容器内做 direct/proxy 对照；宿主能访问不代表 rootless 容器能访问，现代 Requests 能代理也不代表旧版 urllib3 能代理。

至少验证：目标 DNS/IPv4/IPv6 → TCP/TLS → HTTP 方法和状态 → 认证/回显 → 不跟随和跟随的 redirect chain → 正文是否符合预期 → 超时/重试。保留证书验证，并显式关闭测试探针的隐式代理继承，保证 direct/proxy 真正不同。

历史实测已出现：普通 httpbin 直连成功，包含跨站目标的 HTTP redirect 初始请求却被 reset；代理可以恢复重定向，但旧 Requests 经 HTTPS 代理报证书主机名不匹配。还出现过节点 timeout。因此要按协议、请求路径和客户端版本设计策略，不能全局强制直连或全局强制代理。

当前 `repairs.py` 对部分任务设置 bridge、清空继承代理并设置 `NO_PROXY=*`。**这是一项已知不完整策略，不应当作通用修复模板**：wildcard 会绕过测试临时设置的代理，破坏代理语义测试，也使需要代理的出口无法生效。处理时先限定 bypass 范围、核对测试自身的环境变量修改，再验证实际链路。

容器内 `127.0.0.1` 是容器自身，不是宿主代理。切换节点后复测完整请求矩阵，记录探测时间与出口；不为通过测试关闭 TLS 校验。生产任务不要在每次请求失败后无记录地随机切换出口，避免同次评测内部语义漂移。

### 6.5 公网依赖与“已通过”的关系

一次 gold 或模型自测通过不代表长期稳定。公网服务限流、代理波动、DNS、证书、页面 anchor 和重定向变化都会污染反馈。

报告应分别记录：原始测试集是否执行、官方必需项是否有 skip、网络异常、gold resolved、模型 rollout 自测和最终模型评分。agent 最后 Submitted，也可能经历过网络自测失败；`not_run` 不能写成模型未解决。

## 7. harness 元数据获取失败

有时还没创建容器，就在 `make_test_spec` 获取 requirements 时失败。这是**宿主 harness 元数据路径**的问题，与容器 bridge/代理设置不同。

现有 `benchmark_evaluation/metadata.py` 按 URL 缓存真实 requirements 内容，对 GitHub raw 请求使用有界重试和候选源，并保留真实 404/410 供官方路径 fallback 使用。

修复方式是补全真实缓存、检查源路径与 commit、修复下载链路和重试策略。不得在下载异常后返回空 requirements 来继续跑，也不得缓存错误页面为有效依赖文本。requirements-cache 不能当成任意测试资源缓存。

## 8. 标准修复与验收流程

### 8.1 固定可复现输入

保存任务选择、原始 patch/test patch SHA、FAIL_TO_PASS/PASS_TO_PASS、NAS 引用和来源 digest、配置快照、依赖与修复源码身份。使用新 run/audit ID，保留旧失败证据。任务问题不同，选择最小受影响集合。

需要变更任务 overrides 或 NAS 引用时，通过当前 `Catalog` 的受锁事务及既有发布入口维护事实索引，不手改 JSON 绕过验证。旧 run 的 task/config snapshot 不因修复而改写；修复后对照实验使用新的计划和身份。

### 8.2 在正确层实施

| 变化 | 应落的位置 |
|---|---|
| 指定任务的依赖、测试插件、线程变量、网络 | task overrides / `runtime/containers/repairs.py` |
| 所有调用者必须一致的 bootstrap、探针、环境传播 | `EnvironmentPolicy` / `ContainerFactory` |
| 重复的原生库或镜像 ownership 问题 | 受维护的派生镜像与来源验证 |
| harness 元数据、日志评分适配 | `benchmark_evaluation/` 中对应模块 |
| 资源缓存挂载等当前缺失能力 | 扩展共享 schema/policy/factory，记录资源 hash |
| 瞬时基础设施重试 | 已有统一基础设施重试，不另包一层随意重跑脚本 |

环境修复不修改被测业务代码、gold patch、测试断言、评分清单或成功标签。需要改变测试环境契约时，明确标注变体和验证范围。

### 8.3 执行已有入口

在远程工程根目录、runner 环境执行。以下是模板，不表示本次编写文档已启动这些任务。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check

# 新的正式 rollout；run create 冻结工程配置和选择。
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run resume NEW_RUN --detach
.venvs/runner/bin/dycon run status NEW_RUN

# 已有轨迹的官方评分；先冻结确切的 rollout。
.venvs/runner/bin/dycon harness prepare RUN_ID --rollout-id ROLLOUT_ID
.venvs/runner/bin/dycon harness run RUN_ID NEW_EVAL_ID
.venvs/runner/bin/dycon harness status RUN_ID
```

已有 run 的 harness 支持 `--gold` 环境验证，但 prepare 仍要求明确的已有 rollout 选择；不要假设任意 task 清单能直接当作 rollout selection。

仅验证环境而无需模型时，使用已有维护入口：

```bash
.venvs/runner/bin/python -m runtime.scripts.gold_image_audit \
  --root /home/wangtianyi/DyConGlobal \
  --instances FROZEN_INSTANCES.json --only-file SELECTED_IDS.json \
  --audit-id NEW_GOLD_AUDIT --workers 2 --cold-start \
  --nas-root /VM/wangtianyi/DyConGlobal
```

`FROZEN_INSTANCES.json` 是从当前 catalog 或受验证派生输出得到的**instance dict 列表**，每项含 patch、test patch、官方测试定义和 NAS 引用；不是字符串 ID 列表，也不是 run 的整个 tasks 包装对象。`SELECTED_IDS.json` 才是 instance ID 列表。不得为该输入自行拼造另一套任务定义。

`--config-snapshot AUDIT_CONFIG.json` 可复用已有 JSON 快照；不要复制一份根 YAML 到临时目录冒充工程。指定快照时要核对其中的 `project_root`、NAS 根和资源路径，不假设 `--nas-root` 会覆盖快照。

维护 audit 拒绝覆盖已有输出目录。`--cold-start` 要求首次 lease 时镜像不存在，会拒绝命中本地缓存；它不负责随意删除正在使用的镜像。先通过已有 preflight/owned cleanup 核对阻塞者和 lease，不全局 prune。

### 8.4 怎样才算修好

1. 最小复现的故障点消失，且没有通过跳过测试、伪造响应或减小评分范围实现。
2. 正式 gold `resolved=true`，所有官方必需 FAIL_TO_PASS/PASS_TO_PASS 符合要求；infra_error 和 skip 单独解释。
3. cold start 的 load/create/bootstrap/cleanup 证据完整，依赖和镜像身份可回溯。
4. 复核少量同类已通过任务，确认共享策略没有回归；网络修复同时检查代理语义和本地 fixture。
5. `catalog-check`、镜像来源/内容核验、run artifact 验证各自完成。catalog 通过不代表 NAS blob、环境和网络也通过。
6. 新结果带来源 summary、patch、镜像和 policy hash。原始与派生 image ID 不同，只能在验证了修复 lineage 后合并，不能放宽所有 image 身份检查。

gold 只是环境可用性证据，不能作为模型 rollout 的成功标签。完整 pytest 文件额外失败与官方必需项也要分开保存，不能只靠 pytest 汇总数字或进程返回码得出模型结论。

## 9. 自动等待、重测和发布工具的边界

工程已有 pidfd 唤醒工具，例如 `runtime.scripts.wakeup_nonverified600_gold`：等待 audit 进程退出后，根据结果筛选失败项、复用配置和统一容器入口重测。它应用已有修复规则，**不会自动理解错误并发明新的依赖或网络修复**。未改变的确定性失败不应连续原样重跑；瞬时连接故障才适合有界重试。

`runtime.scripts.finalize_rootless_gold_repair` 等收尾模块仍包含历史 cohort 路径假设，当前源码还有汇总输入与退出阶段不一致的待修点。`benchmark-pull` 的 cohort 复核也仍将冻结任务与当前 catalog 整行比较，合法的派生 image 替换可能触发不一致。这些工具**不能不看源码就当作任意新任务的通用修复流程**。

新需求应修正或参数化现有维护模块，校验全量 baseline、来源 lineage、冻结配置与明确退出状态后使用；不要复制一个新的临时 finalizer。等待用进程退出事件，不为状态查询增加后台轮询任务。

## 10. 建议记录的修复单

每次修复保留以下信息，便于换机器或重建容器时复现：

| 字段 | 内容 |
|---|---|
| 触发条件 | 项目/测试版本、所选范围、实际命令、错误阶段 |
| 原始证据 | trajectory/tool message 或 harness 日志路径、SHA、返回码和 traceback |
| 环境身份 | Python、包版本、wheel SHA、NAS manifest/image digest、网络与变量、policy hash |
| 根因等级 | 已证实 / 高度怀疑 / 待验证，避免将超时直接写成网络故障 |
| 实施位置 | task policy、共享模块、元数据缓存或派生镜像；保留修改前后差异 |
| 验收证据 | 新 run/audit ID、gold report、cold lifecycle、同类回归结果 |
| 剩余边界 | 是否仍依赖公网、是否需要代理、是否仅缓存静态文件、是否有环境变体 |

常用相关源码：`runtime/containers/{factory,policy,repairs,recovery}.py`、`runtime/images/{nas,oci,ownership}.py`、`benchmark_evaluation/{harness,metadata}.py`、`application/execution/harness.py`、`catalog/store.py`。

工程已有的针对性测试可按修改范围选择：`tests/test_images.py`、`tests/test_cold_gold.py`、`tests/test_network_repairs.py`、`tests/test_gold_repair_22.py`、`tests/test_gold600_repairs.py`。测试文件名中的历史范围不限制其共享契约价值；但单测通过只说明实现契约，不替代真实任务的 cold gold、代理访问和最终模型评测。
