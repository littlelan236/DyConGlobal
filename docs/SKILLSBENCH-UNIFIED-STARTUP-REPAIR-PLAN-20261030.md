# SkillsBench 统一启动入口修复与验收计划

日期：2026-10-30（Asia/Shanghai）
状态：待实施。本文件仅制定方案，不代表修复、验收或重启已经完成。
目标工程：远程 `/home/wangtianyi/DyConGlobal`。
问题 run：`skillsbench-agent75-qualified-thinkclose-20261030`。
远程计划副本：`docs/SKILLSBENCH-UNIFIED-STARTUP-REPAIR-PLAN-20261030.md`。

## 1. 目标与事实边界

把预算、超时、取消、capture、镜像、依赖、tmux、skill 可见性、输出导出和错误归因修复纳入 SkillsBench 正式启动与恢复链路。后续新建 run、clone run、resume、retry-failed 必须共用实现，不再依赖 baseline 专用命令、手工修改容器或 run 启动后的临时补丁。

依据本目录 `REPORT.md`、`failures.csv`、`diagnostic-evidence.txt` 和 `supplemental-evidence.txt`。2026-10-30 16:05:53 快照含 52 个失败 sample：

| 根因或待查分类 | 数量 | 证据含义 |
|---|---:|---|
| 请求最终达到输出上限 | 16 | completion=81920，典型 reasoning=81919；部分请求在 agent 退出后继续运行 |
| native output 在 capture session 停止后到达 | 10 | output_after_session_stop，随后 wait-output 500 |
| bridge 502 尚未确定底层原因 | 12 | 部分 request 缺少 terminal receipt；不能直接归为网关故障 |
| task image 无可用 Python 3.12 | 6 | dependency-install return_code=86 |
| tmux Permission denied | 5 | 说明此前仅注入变量的修复未覆盖全部真实执行路径 |
| skill 探测遇到 /root 权限错误 | 1 | 非 root 探测的目录/cwd 处理存在问题 |
| 输出探测格式错误 | 1 | directory 与 mode 被拼接成 directory755 |
| 未声明工具调用 | 1 | 相同 seed/输入发生重复协议错误 |

以上为历史快照，不是当前运行状态。39 个 bridge 502 是上述多种原因的外层表现，不能另算 39 个独立根因。已存在的代码或过去通过的单元测试，均不能替代本轮端到端验收。

## 2. 唯一入口与职责边界

正式执行链路：

```text
dycon run create（或受控 clone）
  -> create_run / 配置校验 / selection 与 execution context 冻结
dycon run start | run resume
  -> application.execution.supervisor
  -> rollout.runner.run_manifest
  -> SkillsBench preflight 与共享 startup gate
  -> SkillsBenchFactory / OwnedPlanes / execute_attempt
  -> OpenHandsBridge -> 项目 managed serving -> capture controller
  -> export_workspace -> 官方 verifier -> evidence / finalization / cleanup
dycon run retry-failed
  -> 登记新的物理 attempt，再通过上述 resume 链路执行
```

集成原则：

- 只扩展现有 `core.cli` 和其负责模块，不复制 launcher、runner、容器创建或模型请求逻辑。
- 共用参数在根 `config.yaml` 维护，创建时冻结；只使用项目明确支持的 clone/revision 参数例外。执行和 resume 不重新读取根配置以覆盖旧 run。
- 配置语义变化创建新的 run 或走已支持的正式修订入口，不直接修改旧 resolved-config、selection、plan、hash 或历史 receipt。
- 独立诊断命令与生产启动 gate 调用同一内部实现；不能只有手工执行诊断时才应用修复。
- baseline、parallel、capture 组合共用环境和请求策略；steering 仍受现有能力矩阵限制，不借修复开放未支持组合。
- 上游包如需适配，通过现有 hash-bound dependency publication 记录 source adaptation，发布新版本；不得临时修改正在使用的 site-packages 或旧快照。
- NAS image lease、资源所有权、任务身份、capture identity 与 artifact 登记继续由现有 owner 管理。

## 3. 修复与统一入口映射

| 编号 | 修改内容 | 共享模块/挂载点 | 必须自动覆盖的路径 |
|---|---|---|---|
| R1 | reasoning/action 预算与 close grammar 配合 | core/config.py、rollout/model.py、openhands_bridge.py、serving/manager.py | create/clone 校验、start/resume 的每个真实请求 |
| R2 | timeout、disconnect、取消与 terminal receipt | openhands_bridge.py、runtime/agents/lifecycle.py、core cancellation、supervisor、serving execution | 正常执行、超时、取消、SDK 重试、失败恢复 |
| R3 | native output drain 与 session finalize 顺序 | runtime/serving/capture.py、vllm_plugin.py、runtime/agents/skillsbench.py | 成功、agent 失败、capture 失败、取消的统一收尾 |
| R4 | Python 3.12 离线 agent runtime | runtime/agents/dependencies.py、offline.py、SkillsBenchFactory/OwnedPlanes、runtime/images/skillsbench.py | preflight、所有 task 安装 hook、offline-check |
| R5 | tmux 实际调用与权限契约 | SkillsBench agent env、BenchFlow/OpenHands 合格依赖发布、共享 sandbox probe | 所有 OpenHands attempt，包括 retry 和新镜像 |
| R6 | 非 root skill discovery 与可见性 | OwnedPlanes.deploy_skills 及共享探测逻辑 | 正式 agent 的 with-skill/no-skill 部署 |
| R7 | 输出文件/目录分类与 subject 导出 | application/execution/subject.py、OwnedPlanes harden/verifier 包装 | 正常提交、导出失败诊断、既有重评估能力 |
| R8 | 工具协议与结构化错误 | rollout/adapters/openhands.py、openhands_bridge.py | 每次 API attempt、解析、重试与 canonical 绑定 |
| R9 | 失败统计与物理 attempt 计数 | runtime/agents/skillsbench.py、application/execution evidence/status/report、rollout retry | status/report、retry-failed 与最终 evidence |
| R10 | 镜像映射与并发任务隔离回归 | SkillsBenchFactory、OwnedPlanes、NAS lease manager | 每个 task/service 的 compose 创建及 cleanup |

模块名以当前代码为锚；实施前核对实际函数及 owner，不依赖报告中的旧行号。新增必要功能只能挂入上述共享 owner，不另建第二套生产链路。

## 4. R1：预算与生成契约

- 保留单 step `generation.thinking_token_budget=81920`；不退回文档 smoke 的 512。
- 总输出必须为 close、action/tool-call 和结束 token 预留空间。先以 `generation.max_tokens=98304` 作为待验证方案，即额外预留 16384；它不是已验证或无条件足够的值。
- 在配置校验及 start/resume gate 中检查 reasoning budget 小于总输出上限、thinking 开启、qwen3 parser、require_reasoning_close、FULL capture、think_close anchor、export 与模型/tokenizer 身份的一致性。
- 验证 vLLM 当前版本如何执行 thinking_token_budget、close grammar、tool parser；不得只看 effective request 中存在字段。不得手工插入 close token 或伪造 hidden。
- 校验实际渲染 prompt 加最大输出不超过冻结 context；余量不足必须显式报错或进入已定义的 context recovery，不能静默降低 81920 reasoning budget。
- 根配置、clone override 必须调用同一 Generation 校验；同步冻结 execution_context.generation 等派生副本及 hash，验证没有新旧预算混用。
- 成功标准：长 reasoning 在预算边界正常生成显式 close，随后输出合法 action；若达到 length，保留 token 成本并归类为预算终止，不笼统伪装成 gateway 502。

## 5. R2/R3：请求、超时、取消与 capture 生命周期

先对真实 SDK/ACP/bridge/serving 各层超时逐项取证。项目 model_seconds=28800 不代表 SDK、agent idle、代理或连接超时同样允许长 reasoning。不得统一无上限延长超时。

实现要求：

1. 为 request、API attempt、parallel candidate、native output、capture session 建立可追踪绑定。所有 API attempt 在 finally/取消路径落 terminal receipt，记录 queued/running/completed/failed/cancelled 与原因、时间和实际 token 成本。
2. 长请求期间验证 agent idle 判断和 SDK read timeout；能证明请求仍在运行时保持合理活性，wall-clock deadline 与取消仍有效。
3. agent 断开或 attempt 终止，显式取消所属上游请求和子候选，等待取消确认并保留部分 native 证据；不能让生成在后台孤立运行，也不能影响其他 sample/run。
4. 关闭顺序固定为：停止接收新请求 -> 完成/取消在途生成 -> 排空 native output 与 hidden receipt -> 停止 session -> prepare/finalize -> cleanup。不新增独立收尾实现。
5. session stop/finalize 幂等；draining 状态仅接收已登记在途请求的延迟 receipt，禁止任意 reopen 已 finalized session。
6. 对合法延迟 output、真正超时、身份不匹配和写入失败分别记录；不将 output_after_session_stop 当成普通网络故障循环重试。
7. request response 已返回但 receipt 未落盘时，模型成功与 capture 不完整独立表达；任何 completeness 不能补造 token 或 hidden。

回归场景：慢生成、长 reasoning、SDK 断连、agent timeout、取消、并行候选取消、native receipt 延迟、重复 stop/finalize、服务退出和重启。每个场景都检查无孤立请求、无跨 run 取消、terminal receipt 完整。

## 6. R4/R5/R6/R10：环境与任务隔离

### 6.1 Python runtime

- 探测 task image 中真实可执行 Python ABI，不能把 `command -v python3` 或固定路径存在当成合格证明。
- 缺 Python 3.12 时，优先复用已有离线 runtime publication/镜像兼容机制；不足时通过现有发布入口扩展独立、hash-bound CPython 3.12 agent runtime，挂载或派生镜像方案由 glibc、架构和执行权限实测决定。
- 不覆盖任务 Python，不在线下载解释器，不修改旧 NAS image/snapshot。记录 runtime、完整 requirements closure、平台、版本和内容 hash。
- 统一 hook 执行 venv 创建、hash-bound 安装、关键包导入、openhands --help 及真实非 root ACP 工具初始化。
- 75 个 task 的环境必须全部预检；相同完整环境身份可复用凭证，身份/代码/快照变化使凭证失效。不能只做宿主 wheelhouse verify。

### 6.2 tmux

- 对五个失败任务捕获真实异常调用栈和 ACP 子进程有效环境，确认 BENCHFLOW_OPENHANDS_DISABLE_TMUX 是否被当前安装版本消费，是否覆盖到了 SDK terminal backend。
- 若上游支持无 tmux backend，统一选择并实测；若变量未受支持，使用已证明的兼容 backend 或经审计的依赖发布适配，而不是继续叠加无效环境变量。
- 若某路径必须使用 tmux，则在正式镜像/依赖发布层保证 binary、父目录和 socket 权限，按实际 agent UID/GID 验证。禁止全局 chmod 777、root agent 或修改宿主权限。

### 6.3 Skill visibility

- 以真实 agent 的 HOME、cwd、workspace 和官方 discovery path 构造探针，从可进入的 cwd 执行。
- 必需 skill 不可读必须失败；可选且不适用的 discovery 路径不可读单独登记，不能掩盖实际部署失败。with-skill/no-skill 保持一致的可审计契约。

### 6.4 NAS/compose 与并发隔离

- 继承现有 NAS digest/lease/tag 映射和 cleanup owner；逐 service 校验最终 compose image 与预期一致，禁止意外 bf__ pull/build 回退。
- 审计是否通过共享 cfg 写入 task-specific service_images 产生并发串扰；若存在，改为 attempt-local 数据绑定，由现有 owner 注入，不能引入全局可变 task 配置。
- 用至少两个不同 task image 并发测试 image ID、任务内容、标签、lease 和释放顺序。不得仅凭标签相同/不同判断镜像内容正确性。

## 7. R7/R8/R9：输出、协议与统计

- 修复 `export_workspace` 类型/mode 探测的分隔协议，覆盖 file、directory、missing、permission denied、含空格/特殊字符路径；修复可证明的 directory755 解析问题，不据此宣称所有 verifier 已兼容。
- 导出失败保留输入路径、原始探测输出、return code、traceback、classifier/probe 版本；保留官方 verifier 的独立结果。支持重评估时走既有入口，能力不支持时明确说明，不绕过 agent 隔离或评分契约。
- 未声明工具仍严格拒绝；核对请求 tools/schema 与 parser 映射。相同输入、相同 seed 的确定性失败不盲目重试；若允许修正后重试，要显式记录策略、上下文和 attempt 绑定。
- Bridge 返回结构化错误和 correlation ID；明确区分 limit、protocol、environment、capture、transport、cancelled。可重试性由已证实原因决定，不按所有 502 自动重试。
- 将 12 个未知 502 关联到 request/connection/SDK/cancel 时间线；无法确定者标 unknown，不臆造根因。
- 分开保存 agent_exit_status、environment failure、verifier truth 与 capture completeness；physical_attempts 按真实已启动 attempt 计数、去重，并显示历史重试成本。
- 本轮基础设施失败不增加逻辑 repetition；retry-failed 使用新 request ID，保留全部原始失败证据。

## 8. 自动启动 gate 与凭证

以下能力为待实施要求；若当前 CLI 未提供，扩展既有 skillsbench preflight/audit 或 run start 内部共享 gate，补文档后再使用，不把新命令写成已经可用。

| Gate | 位置 | 检查与阻断条件 |
|---|---|---|
| G0 | create/clone | 配置类型/跨字段预算、selection、执行上下文/hash 同步 |
| G1 | preflight + start/resume 共享调用 | 全 task 环境凭证、Python/tmux/skills、NAS/compose 身份完整 |
| G2 | managed startup | 请求能进入 grammar/budget 路径、显式 close native token 和 persisted hidden 实测 |
| G3 | attempt 运行与收尾 | terminal receipt、取消确认、capture drain、输出导出和清理 |
| G4 | 扩大批量前 | 代表失败任务、长 reasoning、真实并发和逐 step 覆盖全部验收 |

环境 probe 使用现有容器 owner，不能直接另起临时 Docker launcher。涉及 GPU 的 gate 不得伪称 CPU-only/check-only；对其他 run 无修改或终止行为。

每项证据至少绑定 run/execution/sample/attempt、policy 版本、code identity、配置 hash、model/tokenizer、dependency manifest、image digest 和环境身份。文件位置复用既有 attempt/startup/artifact receipt 布局，新字段走版本化 schema；不强制为每个 gate 创建一套新目录。

## 9. 验收顺序与放行条件

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| A | 读取最新证据、统一 cancel/reconcile、建立失败回归集 | 仅目标 run 停止且清理完整，旧证据保留 |
| B | 实现 R1–R10，共享策略与单元/竞态测试 | 每类错误有失败前复现和修复后测试；无绕过身份或权限的补丁 |
| C | 75-task 环境审计 | 每 task 具备有效环境凭证；Python/tmux/skills/compose 全部通过 |
| D | GPU 2–3 单任务及代表任务 smoke | 真正执行 agent/tool，不止 serving ready；close token/hidden 可读取并校验 |
| E | 长 reasoning、断连/取消、真实并发小规模 run | reasoning=81920 预算可用，总输出留余量；无孤立生成或 stop/output 竞态 |
| F | 新建修复后的 75×3 或正式重试 | G0–G4 通过，冻结配置与代码/依赖一致；继续监测新增错误 |

所有单元、集成、GPU 测试分别报告，测试数量不代替环境覆盖。旧 `78 passed`、clone `9 passed` 和 serving `validated` 不构成本次批量放行依据。

逐 step 验收以完整 canonical assistant action-response 集合为分母，包含最终回答；每个 step 唯一关联 accepted native request、真实 output token IDs、显式 close token、其后 forward 证据、persisted think_close event 与 NAS tensor/shard/member hash。parallel canonical winner 与全部候选分母分开报告。

必须列出 missing、duplicate、unexpected、unfinished 和失败原因；不能用 request metadata、capture session 创建、capture complete、coverage.missing=[]、总点数相等或 artifact verify 单项替代完整验收。测试允许暴露真实模型质量或 verifier 失败，但不得将其误报为基础设施成功/失败。

## 10. 配置、恢复与回滚

- 不直接编辑当前 thinkclose run 的冻结配置。预算或 runtime 身份变化时创建新 run；只改变实现且原冻结契约仍兼容时，可经正式 retry-failed/resume 恢复，并记录新 code identity。
- 普通新 run 从唯一根 config 创建。clone 仅采用已支持、带审计的 override；检查 effective request、execution context 与 serving launch 三者一致。
- 正式重启前归档最新错误矩阵和资源所有权。启动后首个真实 agent request 的 native token 与持久化 hidden 未证实时，不宣称成功。
- 回滚到上一个合格代码/runtime/image publication 通过同样入口重新验证；不改旧 publication 的内容，不将已失败轨迹拼接为同一新会话。
- 回归扩大时设置可配置的失败阻断阈值，阈值需通过 schema/快照冻结；检测到重复启动故障时仅中止目标 run 并保留证据。具体阈值在小规模验收后确定，不在本计划虚构已验证值。

## 11. 实施交付清单

- [ ] R1–R10 全部接入统一生产链路；逐项记录完成/待验证/阻塞，不能只更新文档。
- [ ] 根配置与 clone/revision 配置校验一致；有效预算和 close grammar 在实际请求中验证。
- [ ] 全任务环境审计、原五个 tmux 失败任务与六个缺 Python 任务回归。
- [ ] 长请求、取消、capture drain 和跨 run 隔离测试通过。
- [ ] 输出探测、未声明工具、12 个未知 502 与统计字段有对应证据。
- [ ] baseline/parallel/capture 组合回归，保留各能力边界和统一 owner。
- [ ] 更新 ENTRYPOINTS、MINIMUM-RUN、SKILLSBENCH-OPENHANDS 三份文档，标注现有与新增能力。
- [ ] 保存新 run ID、冻结配置差异、test/audit receipt、逐 step native/hidden 对齐结果。
- [ ] 全部门槛通过后再启动 75×3；不能以已生成少量 token 或 serving ready 提前结项。

本次交付范围为计划文档及本地/远程一致副本；未执行代码修复、修改运行参数、停止或启动 run。
