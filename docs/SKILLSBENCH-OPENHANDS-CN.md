# SkillsBench + OpenHands

该入口以官方 SkillsBench revision `f860ecf02ef53412a030f8661c83d644b739ff01` 和 BenchFlow `0.6.3` 为目标。执行由 `dycon run create/start/status/resume` 管理；BenchFlow owns task sandbox、ACP session、skills 与 verifier，DyCon 仅桥接 OpenAI-compatible model requests 和 run-local evidence。模型使用项目配置的既有 serving。

<a id="think-close-capture"></a>
## 每 step 显式 `think_close` 生成与采集（2026-10-30）

适用于本次已验证的 **Qwen3.5-9B + managed EasySteer/vLLM + SkillsBench/OpenHands**。采用“显式 close grammar + thinking budget + native hidden 采集 + 逐 step 验收”组合；仅设置 `enable_thinking=true`、看到 `reasoning`/`reasoning_content` 或 `capture_status=complete`，都不代表每个 step 已保存 close hidden。单任务 smoke 的 42/42 结果不是任意任务、seed、模型版本的无条件保证。

### 配置：创建新 run 前在根 YAML 中合并

以下为**配置片段**，合并到工程根 `config.yaml` 的既有同名块，保留其余必填字段，不要新增重复 YAML 块或覆盖完整配置。`512` 和 `8192` 是此次 smoke 的实际取值，扩大任务范围前需要重新验证预算。

```yaml
generation:
  enable_thinking: true
  preserve_thinking: true
  require_reasoning_close: true
  thinking_token_budget: 512
  max_tokens: 8192
serving:
  management: managed
  reasoning_parser: qwen3
  tool_parser: qwen3_xml
capture:
  enabled: true
  backend: easysteer
  mode: online
  graph_mode: FULL
  anchors: [think_close]
  prefix_cache_policy: safe_boundary
  layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  export:
    enabled: true
```

`rollout/model.py:model_config` 会把 `thinking_token_budget` 和 close grammar 放入请求；OpenHands bridge 复用同一映射。`runtime/serving/manager.py` 在启用 `require_reasoning_close` 时启动 xgrammar，并设置 `enable_in_reasoning=true`，使约束在 reasoning 阶段生效。对本次 tokenizer，请求实际 grammar 为：

```text
root ::= [^<]* Token(248069) ExcludeToken(248068)*
```

`248068/248069` 仅是此次 tokenizer 的 `<think>`/`</think>` ID；应由项目从已冻结 tokenizer 推导并与 capture inventory 核对，**不要自行硬编码、拼接 response 或手动补 token**。Grammar 阻止在显式 close 前直接进入 `<tool_call>` 路径；thinking budget 限制闭合前的生成长度。仍须给 action/tool-call 和终止 token 留出输出预算，并满足 context 与超时要求；`length`、取消、网络/ACP 中断或保存失败都可能让本次 attempt 不完整。

Qwen3 parser 支持以 `<tool_call>` 隐式结束 reasoning，因此有 `reasoning`/`reasoning_content` 不等于生成了 `</think>`。Capture 监听 native close token，不会因 parser 发出语义上的 `REASONING_END` 自动产生 hidden。不能靠修改 parser 返回字段或在事后文本中补 `</think>` 修复缺失的 hidden。

### 正式 smoke 入口

先确认选择的是普通 agent task snapshot（不是 `--oracle-only`），profile 已通过请求角色资格检查。MIG 使用启动时实际发现且空闲的 UUID，TP=1；选择空闲 serving 端口。本次 smoke 的资源值为 context=65536、KV cache=8 GiB、max_num_seqs=2、单任务 worker；普通 baseline 使用 `parallel.mode=parallel`、`parallel.enabled=false`，不要继承 `parallel_baseline` 或多分支策略。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon run create NEW_RUN_ID --operation rollout \
  --task-id offer-letter-generator --repeat-count 1
.venvs/runner/bin/dycon skillsbench preflight NEW_RUN_ID
.venvs/runner/bin/dycon run start NEW_RUN_ID --detach
.venvs/runner/bin/dycon run wait NEW_RUN_ID
.venvs/runner/bin/dycon run status NEW_RUN_ID
```

Generation/agent/timeout 参数按项目约定在根配置维护；创建后校对新 run 的 `resolved-config.json` 与 `serving/launch.json`。已有 run 的配置/receipt 不可修改，resume 使用原快照，修改根 YAML 不会让旧 run 获得新 grammar 或预算。不要通过把根配置复制到 `temp/` 后传 `--config` 来复现：配置所在目录会改变工程根路径解析。

**本次 MIG 兼容设置与风险：** 当前实现的 managed serving 先启动，再调用 SkillsBench preflight；后者可能把本 run 自己的 engine 误判为 MIG 已占用。本次 smoke 在创建时显式冻结了 `serving.allow_mig_sharing=true`。该设置会放宽进程占用检查，不应作为通用默认值；只在已核实设备原先空闲、run-local 锁与实际 PID/UUID 归属后采用，不得借此挤占其他任务。若启动顺序修复，应重新验证并移除此兼容设置。

### 每 step 验收：逐一匹配，不能只看总数

分母取完整 canonical trajectory 中**全部 assistant action-response（含最后一次响应）**，不是 hidden index 中已有的点数。成功、失败、重试及未完成请求分别统计，不能删掉未闭合响应后再宣称全覆盖。每个 canonical step 必须一一绑定最终选中的 native request，并同时满足：

1. Frozen model/tokenizer、session、request、query/attempt 与 canonical message/step 身份一致；native prompt/output token 哈希及 trajectory SHA 正确。
2. 原始 `output_token_ids` 中出现 inventory 指定的显式 close token，且之后确实执行了 forward。项目的 close hidden 取 close token 作为后续 forward 输入的位置；仅在最后输出一个 close 后立刻终止，不保证存在可导出的 hidden。
3. 该 request 对应 persisted `think_close` event，token ID/absolute position 与 native ledger 一致，FULL-graph 证据符合本次采集目标。
4. NAS tensor/shard/member SHA、层数、特征形状和有限数值通过共享 reader 验证；Public hidden 本次为 `[32,4096]`，native 层为 `0..30,32`，使用 `difficulty.datasets.submission.public_features` 的既有映射。
5. 每个 step 的有效 close hidden 数符合既定“first close”口径；列出缺 close、缺 hidden、歧义/重复、未发布及中断请求，要求缺失集合为空。若启用 parallel，canonical winner 与所有候选的分母必须分开。

现有正式审计入口：

```bash
.venvs/runner/bin/dycon skillsbench audit NEW_RUN_ID --rollout-id ROLLOUT_ID
.venvs/runner/bin/dycon artifact verify NEW_RUN_ID
```

Failed run 必须给出已有终态轨迹的 `--rollout-id`；仅验证 capture 时不要加 `--require-resolved`，普通 baseline 不加 `--require-parallel`。`skillsbench audit` 检查 native/hidden 身份、完整性与代表点；还要逐一核对 `outcomes[].points[].request_id` 与 finalization 接受的 action-request 集合，确保每 step 一一覆盖。**审计命令退出成功、总点数相等、`coverage.missing=[]` 或 `artifact verify` 成功，均不能单独替代这个集合核对。** 当前没有独立的通用“全 step 覆盖率”CLI；本次附加逐 step 检查使用既有 shared readers，结果来源见下节。

真实在线 hidden 在 `<nas_root>/<capture.export.root>/<RUN_ID>/`，spool 在 `<project_root>/<capture.export.spool_root>/<RUN_ID>/`（若有 relocation，以实际路径为准）。不要仅凭 `runs/RUN_ID/hidden/` 目录为空判定未捕获。Capture receipt 的 complete 表示声明的采集/发布流程完成，不等于所有请求均生成 close。

### 已验证结果与回归依据

- Run：`skillsbench-thinkclose-mig6-smoke-v5-20261030`；task：`offer-letter-generator`；rep-000，seed=42；MIG `MIG-2db3e350-5a6e-589c-8630-a7007170eea0`、TP=1。UUID 仅为历史证据，不可据此假定该设备当前空闲。
- Rollout：`r-140d624bbe6780d842e19d8bc6f5d84f`；42 个 canonical step，42 个 accepted native action-request，42 个显式 close，42 个已校验 FULL-graph close hidden；缺失集合为空。
- 正式 `skillsbench audit ... --rollout-id ...` 已再次通过：canonical_decisions=42、native_requests=42、exact_close_points=42，41 个工具结果已匹配；它是显式轨迹 scope，不是整个 run 的任务成功验收。
- Agent 为 `AgentCompleted`，capture 为 `complete`；run 为 `VerifierError`（verifier 无法分类 `/root/offer_letter_filled.docx`），`resolved=null`。该异常不应写成 capture 失败，也不能宣称解题成功。
- 对照尝试 `skillsbench-thinkclose-mig6-smoke-v4-20261030` 仅启用 close grammar、thinking budget 为 null；前三步有 close hidden，随后 step 3 请求达到 8192-token 上限并发生 `length` 重试，测试经正式取消入口停止。因此不能只开启 grammar 就承诺稳定。
- 逐 step 附加核验：`temp/skillsbench_estimator_eval_20261030/skillsbench-thinkclose-mig6-smoke-v5-20261030-audit/{summary.json,steps.csv}`；正式 run 位于配置的 run artifact root，原始 hidden/session/shard 在对应 NAS capture 根。临时审计结果删除前应归档；原始冻结 run/NAS 证据为复核依据。

扩大任务/seed/并发规模或更换模型、tokenizer、parser、capture/runtime 版本后，须重新 smoke 并按上述口径验收。512-token thinking budget 和强制显式 close 都会改变生成分布；不能把此配置的结果直接视作原始无约束 OpenHands 策略的可比性能。

## 初始化

snapshot 只接纳明确给出的 official `tasks/` task IDs；`tasks-extra/`、多服务、GPU 和 GUI 任务会拒绝。

```bash
.venvs/runner/bin/dycon skillsbench-images verify
.venvs/skillsbench/bin/dycon skillsbench profile PROFILE_ID \
  --bridge-advertise HOST_REACHABLE_FROM_DOCKER \
  --skill-mode with-skill --network public \
  --output offer-letter-generator:/root/offer_letter_filled.docx
.venvs/skillsbench/bin/dycon skillsbench import SNAPSHOT_ID \
  --task-id offer-letter-generator
.venvs/skillsbench/bin/dycon --set benchmark.backend=skillsbench \
  --set benchmark.snapshot_ref=catalog/snapshots/SNAPSHOT_ID/manifest.json \
  --set agent.backend=openhands --set agent.profile_ref=runtime/agent-profiles/PROFILE_ID/manifest.json \
  run create RUN_ID --operation rollout --task-id offer-letter-generator
.venvs/skillsbench/bin/dycon skillsbench preflight RUN_ID
.venvs/skillsbench/bin/dycon skillsbench oracle RUN_ID
.venvs/skillsbench/bin/dycon run start RUN_ID
```

Profile/snapshot/decoder identity is copied into the immutable execution context. Changing a profile or task creates a new run. Each task must declare expected output paths. `no-skill` is isolated by BenchFlow and cannot see the bundled skills directory. OpenHands + steering and other unsupported benchmark/agent combinations are rejected. New `public` profiles use the same host-proxy setting as the full official oracle environment by default; `--no-use-host-proxy` explicitly disables it. `network=none` defaults to no host proxy.

An `--oracle-only` snapshot is validation-only and cannot run an OpenHands rollout. Create a regular task snapshot for agent execution, even when it contains the same tasks. Oracle success confirms the official oracle/verifier lifecycle for that task package and profile; it does not by itself qualify an OpenHands rollout or guarantee the agent dependency installation path.

## Evidence And Limits

Model requests are authenticated to the owning attempt, buffered before return, recorded with their effective tool schema and unique request identity, and sent through the configured serving URL. OpenHands retains the agent loop and executes the returned response. Raw ACP events, selected model calls, task output manifests and live verifier rewards are preserved. Workspace subjects are not patches; verifier must run before sandbox cleanup. A file subject cannot be re-evaluated after sandbox cleanup.

Exact capture and branch selection remain blocked while the OpenHands request-role probe is unqualified. Preflight exposes this instead of substituting tokenizer-reconstructed or text-only evidence. Estimator-based selection also requires a compatible published predictor.

The current rootless Docker daemon reports that it cannot enforce CPU CFS quotas. The strict profile rejects tasks declaring a CPU quota. An operator can select `--cpu-quota unenforced_with_receipt` in a new immutable profile. The task's compose override then removes only the unavailable CPU quota, preserves the declared memory limit, and writes a resource-deviation receipt. Public task networking and online verifier/bootstrap dependencies are not offline-qualified. Receipts distinguish those limits from agent/task failures.

Action trajectory analysis supports only the existing SWE-bench/mini-swe-agent reader and rejects SkillsBench/OpenHands sources.


## 2026-10-28：非 root agent 与官方 hardening

OpenHands profile 固定 `sandbox_user=agent`。BenchFlow 负责创建用户、授予任务工作目录访问权、启动 ACP，并在 verifier 前停止该用户的进程；官方 hardening、测试与评分仍完整执行。

不得使用 `sandbox_user=root`：BenchFlow 0.6.3 的 hardening 会执行 `pkill -u <sandbox_user>`，UID 0 包括容器主进程，因此会造成 `service "main" is not running`。仅替换容器的保活 command 无法解决这个问题，也不应省略 hardening。

新 profile/run 冻结非 root 身份和代码修复；旧配置、旧失败轨迹与评测结果保持不变。`OwnedPlanes` 对 root hardening 提前拒绝，避免容器被误杀。验证使用新的 run ID，通过正式 `dycon run create/start` 执行。


## 2026-10-28：MIG consensus 与离线验收入口

本节更新前文的离线能力边界：仅对已验收的两任务/平台组合成立；历史公开网络 profile 不因此获得离线认证。

- MIG 使用真实 MIG UUID 写入 serving.gpus，TP=1，通过既有资源检测/锁接口启动；64K 上下文在本次单实例中使用显式 KV4GiB。
- 真实 consensus 已通过 skillsbench-consensus-mig7-offer-20261028 全 run audit；两次为平票后 estimator 决胜，agent reward0，不宣称解题成功。
- 正式离线探针为 dycon skillsbench offline-check RUN_ID；必须冻结 network=none 和依赖快照。入口不启动模型，复用 BenchFlow 官方 oracle/verifier 生命周期，另写 validation/offline-oracle/TASK/。
- OwnedPlanes 将 network=none 落实为 BenchFlow 原生无网络 overlay；探针检查实际 Docker NetworkMode=none、仅 lo、禁缓存安装、image/system/dependency 身份及 cleanup。
- network=none 可与明确的 cpu_quota=unenforced_with_receipt 组合；该资源偏差与联网许可相互独立。use_host_proxy 仍仅允许 public profile。
- 离线验证了 dialogue-parser 与 offer-letter-generator 的207包安装/import/CLI及官方 oracle/verifier（均 reward1），不代表无网络 LLM rollout或所有任务。
- 预置快照 uv0.13.5/uvx 为登记的环境变体，原始评分脚本未改写；不宣称其在线 uv0.9.7 bootstrap 完全复现。

复查：

    .venvs/skillsbench/bin/dycon skillsbench offline-check skillsbench-offline-closure-two-tasks-v2-20261028
    .venvs/skillsbench/bin/dycon skillsbench audit skillsbench-consensus-mig7-offer-20261028 --require-parallel
    .venvs/skillsbench/bin/dycon artifact verify skillsbench-consensus-mig7-offer-20261028

详细验收、回归与边界：[SKILLSBENCH-CONSENSUS-OFFLINE-ACCEPTANCE-20261028.md](SKILLSBENCH-CONSENSUS-OFFLINE-ACCEPTANCE-20261028.md)。


## 全量官方 Oracle 验证（2026-10-28）

`skillsbench import SNAPSHOT --oracle-only --task-id ID ...` 将明确指定的官方任务冻结为验证专用快照，可包含尚未完成 OpenHands 能力验收的浏览器/多服务任务。该标记绑定到 run 的 execution context，preflight 和 agent attempt 均拒绝将验证专用快照作为 agent rollout 执行；不会扩大现有 agent 支持声明。

本轮固定官方 `tasks/` 的 87 项任务，revision 为 `f860ecf02ef53412a030f8661c83d644b739ff01`，不包含 `tasks-extra/`。快照为 `catalog/snapshots/skillsbench-full-oracle-f860ecf02-20261028/manifest.json`，run 为 `skillsbench-full-oracle-20261028`。

统一执行入口：

```bash
.venvs/skillsbench/bin/dycon skillsbench oracle skillsbench-full-oracle-20261028 --workers 4
```

每个任务由官方 BenchFlow oracle/verifier 执行独立生命周期，保留官方 `task.md` 的 oracle 和 verifier 超时。`--workers` 限制同时活跃的任务数量（1–16）；NAS 镜像按任务租赁，单 run 共享其所属代理。原 compose 服务拓扑不改写，各服务镜像来自已登记的 NAS publication，并添加任务 ownership。CPU quota 不受 rootless daemon 支持时仍要求显式 profile 和偏差 receipt。

结果写入 `runs/RUN/validation/oracle/TASK/result.json`；进度为 `progress.json`，最终该次调用汇总为 `summary.json`。每项通过要求所有官方 reward 精确等于 1、无 agent/verifier 错误，并核实所属容器已清理。失败不会终止整个任务集。

重试入口：

```bash
.venvs/skillsbench/bin/dycon skillsbench oracle skillsbench-full-oracle-20261028 \
  --workers 4 --retry-failed --task-id TASK_ID
```

默认复用已有结果；`--retry-failed` 为失败项创建独立的 `attempts/attempt-NNN/`，保留原始失败记录，不覆写失败为成功。成功项不重复执行。仅选择部分任务时，汇总只代表该次选择；需要全量汇总时，重新运行不带 `--task-id` 的入口，读取各项最新结果。

该轮 profile 为公开网络、使用既有宿主代理的 oracle 环境；不附加只为两个任务验收过的离线 wheel closure，不启动 LLM/GPU，不宣称离线环境或 OpenHands 解题能力已全量通过。实际通过率以测评结果为准。

按用户本轮补充要求：确认全量运行中首个任务正常通过后即停止轮询，后台全量运行继续；仅有首项通过时不能出具 87/87 的完成结论。

## 2026-10-29?Oracle ?? task ??

?? SkillsBench ?? oracle/verifier ??? task ?????? [SkillsBench task availability](SKILLSBENCH-TASK-AVAILABILITY-20261028-CN.md)??????????? `catalog/qualifications/skillsbench-oracle-availability-f860ecf02-20261028.json`??? revision ? `f860ecf02ef53412a030f8661c83d644b739ff01`?

????? oracle ???agent ??????? OpenHands rollout ???79 ? oracle ????? 79 ? agent rollout ??????? `data-to-d3`?`fix-visual-stability`?`react-performance-debugging`?`shock-analysis-supply` ?????????????? snapshot ? `validation_only=true` ???? oracle??????? agent?


## Unified startup, environment and receipt lifecycle repair (2026-10-30)

This implementation supersedes the earlier manual full-step inspection limitation: the existing `skillsbench audit` now records schema-3 `step_coverage`, with all canonical assistant responses as the denominator. It checks explicit close IDs/positions against exact native token IDs, the forward after close, FULL event identity and NAS tensors, and reports missing/duplicate/unexpected/unfinished records. Whole-run and explicitly selected rollout acceptance remain distinct; official verifier truth remains independent.

The shared preflight (`--workers 1..8`) uses public BenchFlow setup/start/install/cleanup methods and the production environment owners. It checks every selected task before starting the model. Evidence binds configuration/context, task package, image publication, dependency publication and probe code. A wheelhouse-only check is insufficient. Optional discovery directories are recorded separately; required bundle hashes must be visible to the non-root agent from an accessible cwd.

New dependency publications can include an existing portable CPython 3.12 prefix. Installation probes actually execute the ABI, create a private environment and install the hash-locked closure offline. The task Python is preserved. Native terminal initialization and ACP initialize/session-new run as the actual agent user, without requesting the model. The published CLI consumes a bounded SDK timeout and native subprocess terminal selection; tools reset preserves that explicit backend.

The bridge records every model call and API terminal receipt. SDK disconnect/attempt termination cancels only the owning session through the authenticated serving control endpoint; run cancellation retains its execution scope. Baseline managed OpenHands uses the same plugin in request-control-only mode when capture is disabled. Capture stop enters draining, rejects new bindings, accepts delayed output only for already bound requests and waits for native receipts before closing; repeated stop and compatible finalization are idempotent. Native flush rejects an in-flight session rather than prematurely stopping it. Missing native evidence is retained as incomplete, never synthesized.

Run-specific acceptance results are recorded separately from these implementation contracts. The 81920/98304 candidate and all 75 task environments must pass the new checks before a new full batch is released.



## SkillsBench 修复后正常启动与 run 规模（2026-10-30）

修复范围已经接入统一生产入口：`dycon run create` 冻结配置，`run start`/`run resume` 先执行共享 SkillsBench preflight，再启动 managed serving 和 BenchFlow；`run retry-failed` 注册新的 physical attempt 后仍走同一 resume 链路。旧 run 的 resolved-config、selection、receipt 和历史轨迹不修改。普通 baseline、capture、parallel 和 cancellation 共用这些 owner；未启用的能力不会被隐式打开。

本轮修复覆盖：

- reasoning/action 预算和 tokenizer-bound close grammar；默认候选为 reasoning `81920`、总输出 `98304`，剩余空间留给 close、tool-call 和终止 token。
- native API attempt terminal receipt、实际 token usage、结构化 limit/protocol/transport/cancelled 错误和 request correlation。
- OpenHands 普通 baseline 的 reasoning token 周期检测，以及跨 assistant step 的 reasoning/action 周期检测；命中后在工具执行前早停，保留原始响应和检测证据。parallel candidate 仍保留原有 per-branch detector 和 loop escape。
- capture session draining、延迟 native output、幂等 stop/finalize、session 级取消和 condensation selector 隔离。
- hash-bound CPython 3.12 agent runtime、subprocess terminal backend、SDK bounded timeout、非 root skill discovery、compose image identity 和 75-task environment audit。
- workspace file/directory 探测、逐 step native/hidden coverage、全候选 parallel evidence、API terminal coverage 和 run-local startup failure circuit breaker。

新 run 的推荐流程：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon skillsbench dependencies-verify \
  runtime/dependency-snapshots/openhands-startup-safe-v2-20261030

# 使用已审核的 SkillsBench snapshot/profile；run 创建时冻结它们
.venvs/runner/bin/dycon run create RUN_ID --operation rollout \
  --repeat-count 3 --task-id TASK_ID_1 --task-id TASK_ID_2
.venvs/runner/bin/dycon skillsbench preflight RUN_ID --workers 4
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon run status RUN_ID
.venvs/runner/bin/dycon skillsbench audit RUN_ID
.venvs/runner/bin/dycon artifact verify RUN_ID
```

`preflight --workers 4` 会对所选 task 建立真实 BenchFlow sandbox，检查镜像、离线 agent runtime、非 root terminal/ACP、skill visibility、compose image identity 和 cleanup；它不发 LLM 请求。`run start/resume` 会再次通过同一内部 gate。只有在 preflight、真实 agent/tool smoke、逐 step audit 和 artifact verify 均通过后，才扩大到完整批量。

若改变 model/tokenizer、dependency publication、profile、generation budget、capture 或 serving placement，应创建新 run；resume 不重新读取根 YAML。`run cancel` 后必须检查 `cleanup_complete=true`，再进行 retry/resume。`run retry-failed` 使用新的 request/attempt ID，不覆盖旧证据。

本次验证 run 的真实规模如下：

| Run | 任务 × repetition | 用途 | 结果 |
|---|---:|---|---|
| `skillsbench-agent75-qualified-thinkclose-20261030` | 75×3 | 原问题 run | 已停止，保留历史证据 |
| `skillsbench-startup-qualify-v2-20261030` | 1×1 | GPU 2、3 baseline/长 reasoning | capture complete；reasoning loop/协议失败证据独立保留 |
| `skillsbench-capture-loops-v4-20261030` | 4×1 | GPU 2、3 capture、循环和逐 step | 4 个 attempt 完成；reasoning loop 早停；audit/artifact verify 通过 |
| `skillsbench-parallel-v5-20261030` | 1×1，2 branches | GPU 2、3 parallel candidate | 8 canonical steps、16 candidate native receipts；audit/artifact verify 通过 |
| `skillsbench-unified-repair-env75-v4-20261030` | 75×1 | 全任务环境预检 | 75/75 environment receipts 通过；未执行 LLM rollout |
| `skillsbench-offline-startup-v6-20261030` | 1×oracle | offline install/oracle | 1/1 valid oracle；不代表 OpenHands rollout |
| `skillsbench-noskill-startup-v6-20261030` | 1×preflight | no-skill 可见性 | environment preflight 通过 |

因此，修复后的新验证 run **不是 75×3**。`75×3` 只属于原问题 run 的冻结计划；本轮修复已经完成 75-task 环境覆盖，但尚未启动修复后的完整 75×3 agent rollout。完整 75×3 必须在新的 run 中创建，且先通过 G0–G4 gate。

`skillsbench audit` 当前生成 schema-4 receipt：逐 step 分母是全部 canonical assistant responses（含最终回答），并列出 missing、duplicate、unexpected、unfinished、API terminal coverage；parallel 还单独列出所有候选和未选工具执行情况。官方 verifier 的 resolved truth 与基础设施/capture truth 分开保存。
