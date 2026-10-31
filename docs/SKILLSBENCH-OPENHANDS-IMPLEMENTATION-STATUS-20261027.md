## 2026-10-28 最新收尾：MIG consensus 与离线闭包已验收

本次用户限定的两项工作已经收尾，以下历史段落保留原文；与本节冲突的“consensus/离线尚未完成”不再代表最新状态。其他原始 plan 项不在本次完成声明内。

- 真实 consensus：skillsbench-consensus-mig7-offer-20261028 正常 complete、exit_code=0，全 run audit 与 artifact verify 通过；MIG UUID 接口、TP=1。
- 8 个 canonical/native 决策、8 个 exact close；2 次分支决策、4 个候选，4 个分数独立复算误差 0，losing tool 2 个均未执行，全候选成本 2350 completion tokens。
- 两次均为 1:1 平票后按 estimator_score 决胜，未证明多数票场景；该次 agent 官方 reward=0、resolved=false，不能称为解题成功。
- 离线闭包：skillsbench-offline-closure-two-tasks-v2-20261028 的两个官方 oracle/verifier 均 reward=1.0；实际 NetworkMode=none、仅 lo，禁用缓存并从锁定快照安装 207 包。
- 离线验收是安装/import/CLI 和 oracle/verifier 探针；不启动网络隔离下的 LLM rollout，不等于所有 SkillsBench 任务或冷启动离线 apt 构建均受支持。
- 预置 uv0.13.5/uvx 为明确登记的环境变体，官方评分脚本不变，不宣称原始 uv0.9.7 在线 bootstrap 逐字节复现。
- 清理复核通过：所属 supervisor/runner/模型/代理已退出，无遗留容器或网络；未停止其他实验。
- 当前工作区收尾回归：runner 1120 passed / 67 skipped，SkillsBench 专项 68 passed / 3 skipped。
- Audit receipt：e626335da75145825c65ce2e3d79555c4565c7194cd703afd22ba09aca732903。

完整报告：[SKILLSBENCH-CONSENSUS-OFFLINE-ACCEPTANCE-20261028.md](SKILLSBENCH-CONSENSUS-OFFLINE-ACCEPTANCE-20261028.md)。

---

# SkillsBench + OpenHands 实施状态核对

最后同步：2026-10-28（北京时间）；运行证据截至 2026-10-27 21:24:52（北京时间）<br>
目标工程：`/home/wangtianyi/DyConGlobal`<br>
对照计划：[PLAN.md](../../skillsbench_openhands_compat_plan_20261027/PLAN.md)<br>
状态口径：**已完成**表示计划交付有代码和适当验证证据；**部分完成**表示代码/局部测试存在，但计划门槛未全部满足；**未完成**表示尚无足够实现或验收证据；**阻塞/未验**表示因环境或前置能力未能执行，不能计作通过。

## 总体结论

SkillsBench 的基础注册、任务快照、OpenHands profile、BenchFlow sandbox/oracle 生命周期委托、subject/evidence、OpenHands bridge/canonical adapter 和共享分支核心已有实现。最近一次已记录组合回归为 **207 passed, 2 skipped**（早于共享 MIG 和离线安装最新改动）；此前专项测试为 **9 passed**。最新改动尚未完成整组回归。官方 oracle 已验证 `dialogue-parser` 与 `offer-letter-generator`，reward 均为 1.0。

这仍是**部分实现，未达到计划的全量验收**。OpenHands 工具执行、真实 agent 产物的官方 verifier、native exact capture 和真实 SkillsBench predictor 训练/发布均未通过。最新策略允许共享 MIG；指定 MIG 已通过共享预检，但独立 vLLM 在加载模型权重时 OOM，尚未进入任务执行。此前 GPU0–1 的独立 managed 服务曾启动成功；最新测试仅使用 MIG，禁止改用 GPU0–3 或绑定既有 vLLM。

本轮继续调试后，CPU 可验证范围已扩展：difficulty dataset compose/split 现在保存并校验 benchmark、revision、task package 和 agent 来源；SkillsBench 使用 workspace-success 身份，不要求伪造 SWE patch；训练 target 增加 success 语义别名；OpenHands bridge 的每个 attempt 使用 ephemeral 端口。新增 SkillsBench difficulty 合成测试和 bridge 端口测试，组合回归由此前 196 项提升为 **207 passed, 2 skipped**。这些改动不等于已有真实 SkillsBench capture 或 predictor 训练已完成。

GPU0–1 managed 验收已执行，使用独立 8108 vLLM、TP=2，不绑定既有 8104/8105/8106 服务。模型服务成功启动并完成 `/v1/models` 就绪检查，现有服务进程未停止。OpenHands baseline 的三次受控尝试分别发现：LiteLLM 裸模型名缺少 `openai/` provider 前缀；模型请求收到 HTTP 400；已修改 `prompt_cache_key`、`reasoning_effort` 转发与错误响应记录，但过滤改动是否解决 HTTP 400 尚未验证（不能将推测原因记为已修复根因）；修复后进入官方 agent 安装，但容器在线安装 `httpcore==1.0.9` 超时并以 `install_failure` 结束。当时 blocker 是 OpenHands/BenchFlow 依赖闭包未预装且网络下载不稳定；最新另有 MIG 权重加载 OOM，不能把该结果计作 baseline 成功。上述 managed runs 已清理其自有模型/容器资源，未影响原有 vLLM。

rootless Docker 不支持 CFS CPU quota。按用户选择允许去掉 quota 并在资源 receipt 中记录偏差；其他可用资源限制仍按 profile 执行。这是有记录的资源偏差，不表示 CPU quota 等价达成。


## 2026-10-28 同步：最新改动、实测与未完成项

本节覆盖此前报告的旧资源限制。这里只同步已有代码与运行证据；本次未启动新 GPU 作业，也未重新运行测试。

### 最新授权与实现

- 用户已明确允许在 MIG 上已有其他进程时启动测试；不再以 `process_count > 0` 自动阻断已显式开启共享的 run。
- 已部署 `serving.allow_mig_sharing`，默认 `false`；开启后贯通配置、preflight、ServingManager 与资源 admission。该开关不停止其他进程，也不保证显存足够；资源信息保留进程数/空闲量。共享专项回归尚未完成。
- 已部署离线 wheel profile `runtime/agent-profiles/openhands-acp-1221-offline-wheels-20261027/manifest.json`，SHA-256 为 `b320311404d420c8cb8c0aeba6ddfdd6ed7344368df9ff0ac8981b09c3519f8e`。配置了 `skip_agent_install=true`、只读 wheel 目录挂载和 pre-agent pip hook；这些仅为代码部署，**未通过实际安装验收**。
- wheel 候选目录为 `/data3/ouyangxiyi/agent_trs_native_v1/resources/clean-baseline-v1/python/wheels`，含 303 个 wheel；包括 OpenHands CLI 1.16.0、SDK/tools 1.22.1、httpcore 1.0.9、httpx 0.28.1。尚未复制/封存为项目自有不可变闭包、校验完整平台依赖或证明 wheel CLI 与锁定 Git revision 一致。
- 修复了执行路径中 `Cancelled` 与 `record_agent_outcome` 缺失导入；此前专项测试通过，但不能替代完整真实运行验收。

### 最新共享 MIG 运行证据

| Run | 实际结果 | 证据 |
|---|---|---|
| `skillsbench-openhands-mig-shared-20261027` | preflight 允许共享，但当时 manager/resources 尚未部署更新，仍按旧独占策略拒绝；不是 OOM | `runs/<run>/status.json`，2026-10-27 21:09:23 |
| `skillsbench-openhands-mig-shared-retry-20261027` | 共享策略部署后独立 vLLM 已进入模型加载；权重加载阶段 CUDA OOM，未进入 OpenHands 工具/verifier/capture | `runs/<run>/serving/server.log`、`status.json`，2026-10-27 21:24:48–52 |

目标为 `MIG-2f2fe213-508a-57bb-b2c7-52f4ae0c1021`，TP=1、`gpu_memory_utilization=0.3`、context=2048、单序列。预检时 NVML 显示总量 50,868,518,912 bytes、空闲 16,929,062,912 bytes（约 15.77 GiB）、2 个进程。这是历史观测，不代表 2026-10-28 当前资源。

OOM 日志显示新 engine 进程已占约 15.32 GiB，申请额外 1.89 GiB 时仅剩 449.50 MiB。此日志只能证明本次模型权重加载失败，不能据此宣称固定“20GB 以上必能启动”；还需保留 KV cache、activation、capture 与其他共享进程动态变化的空间。日志中的 `GPU 0` 是 `CUDA_VISIBLE_DEVICES` 映射后的逻辑设备，目标仍为上述 MIG UUID，不是物理 GPU0。

两个 run 的 `execution/attempts/000/exit.json` 均为退出码 1；`termination.json` 均记录 `processes_stopped=true`、`cleanup_complete=true`，任务容器清理 `not_required=true`。资源失败没有产生可用于训练的成功轨迹。

### 四项目标的当前验收结论

| 目标 | 当前状态 | 仍需完成 |
|---|---|---|
| OpenHands 工具执行 | 未通过 | 成功安装并验证锁定 agent，实际模型请求成功返回，ACP 工具调用及结果可配对 |
| 官方 verifier（真实 OpenHands 产物） | 未通过 | 真实工具生成产物、subject 导出、官方 verifier 正常评分、报告/cleanup 完整；已有两项 oracle 成功不能替代该门槛 |
| native exact capture | 未通过 | profile 仍 `request_roles=unqualified`；尚需角色验定、首中末请求 token/hidden/canonical 对齐、截断/重试等边界验收 |
| predictor | 未通过 | 缺少真实 SkillsBench resolved capture；尚无真实训练、来源完整 publication、加载/推理及匹配 feature/label 验收 |

P0–P7 与 V01–V20 的既有“部分/未完成”状态仍适用。不得把共享开关、离线安装代码、合成测试或模型服务曾就绪计作四项端到端验收完成。

## 按阶段核对

| 阶段 | 状态 | 已有证据 | 未完成/限制 |
|---|---|---|---|
| P0 基线、版本、模型请求接缝 | 部分完成 | SkillsBench/OpenHands/BenchFlow 锁定身份与版本记录；bridge、profile、preflight 实现 | OpenHands ACP 实际请求角色未可靠定类；完整版本/依赖闭包与无网探针未全部验收；此前物理 GPU managed 服务已启动；共享 MIG 启动仍失败 |
| P1 公共身份、schema、配置、分派 | 部分完成 | benchmark/agent context、快照、任务身份、配置检查、CLI 与 clone 支持；专项及 run/catalog 测试通过 | 外置根所有 SkillsBench 命令端到端矩阵、旧数据全量兼容门槛未逐项出具验收证据 |
| P2 任务、NAS 镜像、原生 baseline | 部分完成 | 固定 SkillsBench snapshot；NAS manifest/lease；两个官方 oracle 成功；GPU0–1 独立 managed vLLM 启动、模型就绪和任务 sandbox 生命周期通过 | 普通 OpenHands 单候选因官方依赖在线安装超时未完成；依赖闭包无网验收不完整；`no-skill` 隔离验收未完成 |
| P3 subject、评分、持久化 | 部分完成 | workspace subject、evidence/artifact、官方 verifier 委托及 oracle receipt 实现 | OpenHands 成功 rollout → subject → live verify → cleanup 的完整 attempt 尚未成功；中断/恢复边界未全覆盖 |
| P4 请求、轨迹、exact capture | 部分完成 | bridge 已实际接到独立 GPU0–1 vLLM；请求失败 receipt、ACP partial/error 轨迹和 canonical failure 产物已落盘；provider/metadata 转发调整已部署，HTTP 400 根因及修复效果未验证 | request-role 未定类；native token/hidden 与多请求首中末对齐、retry/condensation 映射尚未验收；capture 未完成 |
| P5 标签、混源训练、发布 | 部分完成 | 已增加 benchmark-aware dataset identity、SkillsBench workspace-success label/compose/split 代码和合成边界测试 | 没有真实 SkillsBench resolved capture；真实训练、混源 predictor publication/评估仍无完成证据 |
| P6 共享分支、共识、loop | 部分完成 | `rollout/decision/` 共享协调器和 OpenHands adapter 已实现；mini/SWE 相关并行回归包含在组合测试中 | OpenHands profile 未 qualified；未用匹配 predictor 做 OpenHands 小分支端到端；winner 工具副作用、候选成本及真实共识验收未完成 |
| P7 可靠性、分析、文档、回归 | 部分完成 | 中文操作文档、CLI/help、资源偏差 receipt；bridge ephemeral-port 修复；207 passed/2 skipped 组合回归 | OpenHands cancel/retry/high concurrency/cross-root 故障矩阵和全量 V01–V20 证据报告未完成 |

## V01–V20 验收矩阵

| 验收项 | 状态 | 核对结果 |
|---|---|---|
| V01 旧 SWE/mini 与旧 schema 回归 | 部分通过 | 相关 catalog/run-plan/parallel 测试在已记录的 207 项组合回归中通过；计划要求的旧 hash、harness、artifact/source 与现有 SWE action 分类/绘图完整矩阵未在本轮逐项核对 |
| V02 benchmark × agent 与同名 task | 部分通过 | registry/context 能力检查和 SkillsBench 测试通过；没有完整 CLI 组合与跨 revision 验收报告 |
| V03 oracle 与普通 OpenHands | 部分通过 | 两项官方 oracle reward=1.0；GPU0–1 managed 服务和 sandbox 生命周期通过；OpenHands 普通 baseline 被官方依赖安装超时阻断 |
| V04 with-skill/no-skill 隔离 | 未完成 | profile 支持 skill mode；镜像/home/session 可见性隔离的对照验收证据缺失 |
| V05 非 Git/二进制 subject 篡改校验 | 部分完成 | workspace subject/evidence 有实现；真实产物归档和篡改后拒绝的端到端验收未完成 |
| V06 live verify/重评边界 | 部分完成 | oracle 使用 BenchFlow 官方 verifier 生命周期；完整 OpenHands live attempt 与无快照重评拒绝验收未完成 |
| V07 多工具/finish/text/格式错误 | 部分完成 | adapter 代码、单测、真实 ACP 启动和失败轨迹已存在；尚无成功工具执行证据；最新 run 在 MIG 模型加载前置阶段失败 |
| V08 exact token/hidden | 未完成 | request bridge 不是 native exact 证据；request-role 未 qualified，未完成 native bind/close 与首中末对齐 |
| V09 close 缺失/截断/无 followup | 未完成 | 计划语义尚无完整 OpenHands capture 故障矩阵证据 |
| V10 label off-by-one | 未完成 | SkillsBench canonical trajectory label 未形成并通过人工小轨迹验收 |
| V11 双来源 compose/组件 predictor | 部分完成 | compose 已记录 source_benchmarks/source_agents，并支持 SkillsBench benchmark-aware task identity；无真实 SkillsBench 混源数据和 predictor 发布证据 |
| V12 split/feature/label 不兼容 | 部分完成 | SkillsBench task-level split、package/revision 冲突检查及 success target 兼容测试通过；真实 hidden feature contract 仍未验收 |
| V13 分支模式与共享边界 | 部分完成 | shared decision core 和既有 mini 回归通过；OpenHands 普通/随机/选优/共识端到端均未完成 |
| V14 score/transport/loop 故障 | 部分完成 | 有 adapter/协调器逻辑与测试；真实 OpenHands 故障注入覆盖不完整 |
| V15 超容量、多轮并发 | 未完成 | 未执行超过 admission 容量且跨多轮 OpenHands decision 的测试 |
| V16 排队/生成/工具/verifier/finalize 取消 | 部分完成 | 复用既有取消与清理实现；SkillsBench/OpenHands 每阶段取消矩阵没有完成 |
| V17 subject/评分/capture/cleanup 中断 | 未完成 | 没有完整 OpenHands attempt 的边界中断与恢复验收 |
| V18 项目根与外置根 | 未完成 | 有共享存储路径基础；SkillsBench 全部命令跨根与同名冲突矩阵未验收 |
| V19 NAS/镜像/依赖/网络失败 | 部分通过 | NAS 镜像 manifest/lease、两个在线 oracle 和独立 managed 服务通过；真实 OpenHands 安装明确暴露 `httpcore` 在线下载超时；离线依赖闭包、wheel/uv/apt snapshot 未完成 |
| V20 任务/agent/evaluator/训练来源报告 | 部分完成 | run/context、oracle receipt、资源偏差和 attempt 归属基础存在；完整训练来源与所有候选成本报告尚无数据/验收 |

## 当前可引用的测试与运行证据

- 组合回归：`.venvs/runner/bin/python -m pytest tests/test_skillsbench_difficulty.py tests/test_skillsbench_compat.py tests/test_skillsbench_runtime_integration.py tests/test_catalog.py tests/test_run_plan.py tests/test_difficulty_artifacts.py tests/test_resolved_selection.py tests/test_parallel_rollout.py tests/test_parallel_limit_classification.py tests/test_parallel_consensus.py tests/test_parallel_loop_escape.py -q` → **207 passed, 2 skipped**。
- SkillsBench 专项回归最近一次 → **9 passed**。
- `git diff --check` 通过；compileall 覆盖 `core application catalog rollout runtime/agents enrichment/reader.py`。
- 官方 oracle：`dialogue-parser`、`offer-letter-generator` 各有成功 verifier receipt，reward 1.0。
- profile `openhands-acp-1221-cpu-unenforced-20261027` 的 preflight 记录 BenchFlow 0.6.3、固定 benchmark revision、NAS image digest 与 CPU quota 不支持偏差。
- `skillsbench-openhands-offer-mig-small-20261027` 的 preflight 绑定 `MIG-2f2fe213-508a-57bb-b2c7-52f4ae0c1021`，因该 MIG 已占用而 `ready=false`；没有启动服务或 rollout attempt。之前的 MIG run 因 vLLM KV cache 不足失败，attempt 数为 0。
- `skillsbench-openhands-offer-gpu01-managed-20261027`、`...managed-retry-20261027`、`...managed-retry2-20261027`、`...managed-retry3-20261027` 使用独立 GPU0–1/TP=2/8108 managed 服务。前两次分别记录 provider/HTTP 400 协议问题；第三次记录 OpenHands 安装 `httpcore==1.0.9` 下载超时。每次 run 的 owned vLLM、容器和 proxy 均已由 supervisor 清理；既有 8104/8105/8106 未停止。

## 继续全量验收的必要顺序

1. 继续使用允许共享的 MIG，通过逐实例 NVML 查询选择可容纳实际权重和运行开销的实例；以独立 managed 服务启动，不使用 GPU0–3，不复用既有 vLLM。随后完成离线依赖方案的容器实测和不可变依赖 manifest；仅有 wheel 目录路径不算闭包验收。
2. 完成一个普通 OpenHands 单候选 rollout，验证 agent 工具执行、真实文件 subject、官方 verifier、receipt 与 cleanup。
3. 修复/判定 request-role；通过 native bind/output/close 将实际 OpenHands 请求与 exact token/hidden、canonical event 多点对齐，再完成标签边界测试。
4. 在真实 OpenHands capture 完成后生成 SkillsBench resolved 数据、split、训练/发布来源完整的 predictor，并做泄漏/feature contract 验收。
5. 使用该 predictor 完成 OpenHands 小分支、随机/共识对照、winner-only 工具副作用、候选成本及故障/取消验收。
6. 补齐依赖离线闭包、no-skill 隔离、外置根、并发/恢复及 V01–V20 剩余证据。

以上状态按当前可查看的工作区、测试输出、run receipt 与 preflight 记录填写。代码存在不自动代表验收通过；未运行、未留证或前置条件未满足的项目均明确保持部分/未完成。

## 2026-10-28 completion attempt

- Added immutable offline OpenHands dependency publication and verification (`runtime/dependency-snapshots/openhands-benchflow-only-20261028`), rebuilt the CLI from revision `3ca17446c5d9c1e35e054803478a3501ec251ecf`, recorded the BenchFlow-only skill discovery adaptation, and verified a 207-package install inside the task container with no network access.
- Added run-scoped external artifact references, binary workspace subject verification, durable cancellation during tool/verifier execution, request retry/attempt receipts, OpenHands qualification command, native capture prepare/flush ordering, and verifier/subject handling for workspace success.
- GPU run `skillsbench-final-baseline-gpu01-20261028` used GPU0-1 with a managed TP=2 service. It executed 7 OpenHands tool calls and produced the requested DOCX, but official verifier hardening reported `service "main" is not running`; this is recorded as `VerifierError`, not success. A second run with the service keep-alive override was started; it was cancelled after diagnostic completion and cleaned its owned resources.
- CPU validation after implementation changes: focused compatibility/difficulty/runtime tests passed `210 passed, 7 skipped`; full runner regression before final lifecycle changes was `1086 passed, 53 skipped, 6 failures`, with the six failures fixed and the focused regression subset passing `71 passed`. `compileall` passed.
- Native exact capture, resolved SkillsBench dataset, predictor training/publication, and OpenHands parallel end-to-end remain blocked until the BenchFlow service lifecycle is corrected and a run reaches the official verifier.


## 2026-10-28 持续验收记录（进行中）

本节覆盖历史 MIG/GPU 限制；用户现已允许 GPU0–3 的剩余显存用于测试。当前仅报告已发生的代码与运行事实，四项目标未全部验收前不宣称全量完成。

- 根因修复：`sandbox_user=root` 导致官方 hardening 的 `pkill -u root` 杀掉容器主进程；已改为固定 `agent` 用户，保留官方 hardening。真实 run 已观察到 agent 进程被清理、主容器存活并成功导出 subject。
- 生命周期修复：BenchFlow 的 `compose down --rmi all` 会绕过 NAS lease 删除共享 tag；现保留镜像回收给 ImageLeaseManager。公开网络 profile 可显式启用受管代理；verifier Python 包从锁定 wheels 安装。官方 dialogue-parser oracle 经此路径 reward=1.0，但不冒充真实 agent 成功。
- 协议修复：保留上游结构化 HTTP 400，允许 SDK 正确处理上下文超限；识别锁定 SDK 的 condenser 请求并排除 primary step。按真实 winner tool-call ID 恢复 SDK 遗漏的历史 reasoning，adapter 版本为 v2。
- 采集边界：无 thinking 预算时，模型多轮输出实际缺少 close token；这些点保留 unavailable/censored。新 run 显式冻结 `thinking_token_budget=8`，已连续收到真实 close 和 32 层 × 4096 维 hidden，无回退；仍待终态评分和首/中/末完整 audit，不将原始文本重新分词冒充 exact。
- 取消清理：代理新增 supervisor ACK/process-gate 归属；一个旧取消 run 的孤立代理已通过命令、PGID/SID 和进程出生时间检查后精确回收，有独立 reconciliation receipt。
- CPU 回归：本轮较早完整 runner 回归为 **1093 passed, 55 skipped**；随后代理/取消专项 **34 passed, 1 skipped**。新增协议、provenance、harness 边界测试另有通过记录；最终完整回归将在真实链路修复结束后重跑。

| 必须完成的目标 | 当前进度 |
|---|---|
| 真实 OpenHands + 官方 verifier 成功 | 非 root live verifier 已恢复；offer-letter 正常评分为 0，新的 dialogue-parser run 仍在执行，不能计作 resolved |
| native exact capture | 受控预算 run 已收到多轮真实点；最终 native/canonical/tensor audit 待完成 |
| resolved predictor 训练发布 | 已验证现有 SWE resolved 来源 106 条轨迹、6104 个样本；等待新的 SkillsBench resolved capture 后 compose、任务级 split、训练和发布 |
| OpenHands parallel 真实验收 | 已准备 2 分支、容量 1、2 个 repetition 的真实运行；等待匹配 predictor，随后核对 winner、所有候选成本和工具副作用 |

当前主要 run：`skillsbench-native-budget8-context64-gpu01-20261028`（GPU0–1，capture）与 `skillsbench-proxy-baseline-context64-gpu23-20261028`（GPU2–3，普通 baseline）。旧失败/取消 run 的配置、轨迹、评分和 capture 不改写、不补造成功。


## 2026-10-28 最新交接：成功采集与训练已完成，发布和 parallel 待完成

用户要求继续直到四项真实验收完成，遇到新问题继续修复。本次按用户要求只更新文档，不启动新测试。不要重跑已经完成的训练，也不要把 scoped session 成功等同于整个 run 全通过。

### 四项目标

1. 官方真实 OpenHands verifier：已通过一个 scoped rollout。run=skillsbench-native-close-grammar-offer-gpu01-20261028，rep-000，rollout=r-e1259d54dc88f838c9c680879e719c15。真实 SDK 执行工具并 invoke_skill，官方 reward=1.0、resolved=true、verifier_error=null，DOCX subject 已导出并验 hash。
2. Native exact capture：上述成功 rollout 的 scoped audit 已通过。10 个 canonical decisions 对应 10 个唯一 native requests，9 个实际 close hidden 点，每点 32 层×4096维；首/中/末 native 对齐及 tensor SHA 检查通过。audit receipt SHA=f2c7f164350ba4ce27627cd3ff3cca2bef6b2b47238f810463f21ef2f175d778，whole_run_acceptance=false。一个普通字符拼成的 closing 不作为 native close；3 个 censored 点被排除，最终 6 个可训练样本，没有补造 token、hidden 或标签。
3. Resolved predictor：真实训练已完成，但 publication 尚未执行。混合数据有107条轨迹、107个task、6110样本（6104 SWE +6 SkillsBench）；任务级 split 为75/16/16个task，样本4273/778/1059。Ridge site=3、alpha=10000，validation R²=0.7551244198131953；3个alpha trials，训练 fit 返回成功。拟发布 train+validation refit 的5051样本已验证同时包含 SkillsBench 与 SWE。旧SWE agent来源确实为unknown，不可编造。
4. OpenHands parallel：尚未启动/尚未验收。脚本已准备2分支、capacity=1、dynamic admission、workers=2、repeat_count=2、每4step分支一次。必须使用本次真实发布的 predictor，核对finite分数/argmin、不同候选、winner-only实际工具结果、全部候选native成本与cleanup。

### 已完成产物（远程工程根 /home/wangtianyi/DyConGlobal）

- 成功 session：cap-s-e976918014915fd5b33a3c0ac2a039f9-rep-000-attempt-000。
- scoped audit：该run的 validation/skillsbench-audit-*.json。
- 封存 trace：runs/skillsbench-native-close-grammar-offer-gpu01-20261028/enrichment/online-publications/sealed-resolved-rep0-20261028/。
- 新增 capture-trace 统一入口已实际执行：只读取封存成功session，复用既有 finalize_token_events；独立于其他仍在运行的sample，不改生产者证据。
- SkillsBench dataset：difficulty/datasets/skillsbench-native-resolved-20261028/，6样本；receipt=c6f2ba91f576822cac876c632af76683e4d43e6ad934a4ab5e1037c17e17679f。
- 混合dataset：difficulty/datasets/skillsbench-swe-resolved-20261028/；receipt=3bb3c06bc42d5846746db0e4bd57f1bf503ac1c61eb210f2f443e4e91fefd2a8；artifact verify已通过。
- split：task-grouped-20261028，task_random_v1，seed42，validation/test各0.15；整task分组，无event随机混切。
- 训练配置：difficulty/training/experiments/skillsbench-swe-resolved-ridge-20261028.yaml。
- 已完成训练：difficulty/checkpoints/skillsbench-swe-resolved-ridge-20261028/。不要同名重训或覆盖；test指标本次尚未核对，不编造。

### 当前唯一主阻断：辅助调用误触发全局capture（尚未修复）

成功run的另一repetition在SDK condenser调用后WorkerFailure，整个run后来为failed；不影响已独立封存并成功评测的rep-000，但不能宣称整个run通过。

错误：Native capture drain failed；辅助request的session_id=None，/dycon/capture/sample-file返回HTTP400，staging文件保留。这不是官方成功session失败，也不能把辅助调用硬绑定主session来掩盖问题。

已定位 runtime/serving/vllm_plugin.py：attach_router.provenance会移除body.capture_select；_attach_admission.process在metadata=None时没有明确禁止stream-wide全局selector。无dycon_capture元数据的condenser请求仍采到hidden，控制面无法接收无主sample。

中断前最后一次调用只读了原生 SelectSpec fields，没有写入禁采修复：generation_tokens=[]校验失败；generation_tokens=None是关闭filter，不能当作关闭capture。下一步需检查 request.capture_select 的覆盖/继承语义，验证明确禁采方式（例如显式空stream override或合法的全范围exclude窗口）；这些候选尚未验证，不可猜测采用。SelectSpec含prompt/generation、各自tokens/positions/window及exclude_* fields。修复后必须做真实auxiliary model调用，证明没有无主sample且不污染primary session。

### 重要已部署修复与边界

- sandbox_user固定agent，root hardening提前拒绝，避免官方pkill root杀PID1；完整官方hardening/verifier保留。
- BenchFlow stop(delete=False)清容器/网络；仅ImageLeaseManager回收镜像，修复down --rmi all绕过lease误删共享tag。
- profile显式公开网络代理，verifier Python包走锁定wheels。代理新增process_gate --role network-proxy、supervisor ACK/白名单归属。旧孤立代理已用命令、PGID/SID、出生时间校验后精确回收，保存reconciliation receipt。
- HTTP400结构化透传，不再包装502；锁定SDK condenser请求识别为辅助调用，不计primary step/训练标签。按真实winner tool-call ID恢复SDK遗漏的历史reasoning。
- context-safe依赖快照：runtime/dependency-snapshots/openhands-context-safe-20261028，receipt=464c2996b50f50460bf0da5f2e54619c3f7b7caf6cf3e8f80fddb45cc519b31f。CLI revision锁定，SDK/tools=1.22.1；显式CLI adaptations为BenchFlow-only skills和恢复最小agent settings缺失的原生condenser。安装环境探针已验证原生LLMSummarizingCondenser、max_tokens=24000、handles_condensation_requests=true，零模型调用。
- 可用qualified profile：runtime/agent-profiles/openhands-qualified-context-safe-20261028/manifest.json。transfer校验同一原始CLI archive及全部非CLI wheels一致，记录显式已知context/网络改动和源/目标receipt。
- 成功source run使用旧regex close策略；该策略允许普通字符拼closing，因此缺失点按censored处理。后续代码已改成Token级grammar：ID来自注册模型tokenizer_config.json，拒绝普通字符close，接受真正native ID，close后允许工具。已用实际xgrammar/tokenizer作CPU探针；新Token版本GPU E2E尚未验收，不能混记为成功source已验证。
- 当前adapter VERSION=openhands-openai-v3。共享Generation/model_config/ServingManager支持require_reasoning_close；根config.yaml只新增默认false，不改其他原有值。新服务使用xgrammar backend、enable_in_reasoning=true。
- workspace subject不落入SWE patch harness；没有完整状态快照时明确拒绝post-hoc重评。normalized partial reward合法但仅全部=1才resolved；agent与verifier错误分类已分开。
- publisher派生实际fit来源，保留源profile/config/generation与tokenizer fingerprint；composer校验model/tokenizer。审计使用受hash保护的canonical/ACP结果，检查native tensor、首中末、finite score/argmin、不同候选及losing tool ID未执行。

### 下一轮直接继续

1. 先核对最新run status、execution exit/termination、GPU0–3与managed资源归属；中断后不能假定进程仍在或已停。
2. 修复上述无metadata辅助请求的明确禁采，做真实aux probe；不要只靠单元测试。
3. 验证训练artifact后执行尚未执行的统一CLI：predictor-publish skillsbench-swe-resolved-ridge-20261028 --experiment difficulty/checkpoints/skillsbench-swe-resolved-ridge-20261028 --target reasoning_length --fit-scope refit --endpoint agent_terminal。核对manifest、actual fit sources、tokenizer/site/model、真实加载推理。
4. 本地 skillsbench_verifier_fix_20261028/start-parallel.sh 已准备：当前GPU0–1/8108、run ID skillsbench-parallel-live-gpu01-20261028；先检查端口/GPU。context65536、max_tokens8192、thinking_budget512、require_reasoning_close=true、capture=true、qualified context-safe profile，真实新predictor引用。不要用旧predictor或stub代替。
5. 完成真实parallel与skillsbench audit --require-parallel；遇新问题继续修复。未验证的P0–P7/V01–V20不能标通过。完成最新完整回归和本地/远程报告后才宣称四项完成。

### 执行材料与测试

本地命令/记录在skillsbench_verifier_fix_20261028/，包括WIP.md、observe-campaign.sh、build-resolved.sh、train-resolved.sh、start-parallel.sh。数据构建和训练已执行，勿整段重跑造成immutable ID冲突。

SSH helper为skillsbench_openhands_implementation_20261027/remote.py，密码只用进程环境DYCON_SSH_PASSWORD；Guarded deploy为skillsbench_completion_20261028/deploy.py，代码镜像同目录，比较baseline/deployed SHA并备份远程temp，保留其他未提交修改。不要在远程另写生产launcher；临时诊断仅temp任务子目录。

较早完整runner回归1093 passed/55 skipped；随后proxy/cancel专项34 passed/1 skipped、native/context相关11 passed等。所有最新变更后的最终完整回归尚未跑，不可用旧数字冒充最终验收。

模型fingerprint=f97643c15ea3cb04517379dd13482b48548458c3a3f6c3a7eb93dcaa6adf86a2。
tokenizer fingerprint=92c0805f26bd5df195cf7fda1df726420850b764155dea7b322052e7e228ea2c。
公共native向量：layer_0..30与layer_32，各4096维；storage float32、serving float16。所有旧snapshot、run、hash和失败证据保持不变。
