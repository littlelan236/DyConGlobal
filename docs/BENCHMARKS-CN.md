# Benchmark 与容器手册

从镜像、环境和任务预检开始，再创建 rollout 与评分计划。各节记录对应 benchmark 的配置、验证和历史结果复评用法。Run 生命周期见[运行手册](MINIMUM-RUN-CN.md)。

## 详细参考

- [共享环境与镜像](#共享环境与镜像) · [通用准备与验收](#通用准备与验收)、[Docker 存储、网络与镜像](#docker-存储网络与镜像)
- [SWE-bench](#swe-bench) · [Catalog 分组和镜像同步](#catalog-分组和镜像同步)、[Gold 排查、验证回执和评分结算](#gold-排查验证回执和评分结算)
- [SkillsBench](#skillsbench) · [启动预检、SDK 与压缩器配置](#启动预检sdk-与压缩器配置)、[主轨迹生成 token 预算](#主轨迹生成-token-预算)、[Verifier 资源、发布与资格验证](#verifier-资源发布与资格验证)、[包索引策略、诊断与清理](#包索引策略诊断与清理)、[Zero reward 复查](#zero-reward-复查)、[离线评分审计的输入和输出](#离线评分审计的输入和输出)、[AgentOps 多解释器 verifier](#agentops-多解释器-verifier)、[Airport PDDL 变体与语义复评](#airport-pddl-变体与语义复评)、[Airport PDDL 完整评分重跑](#airport-pddl-完整评分重跑)、[历史提交复评](#历史提交复评)
- [Terminal-Bench 2.1](#terminal-bench-21) · [执行环境、资源准备和验证](#执行环境资源准备和验证)、[Terminal-Bench verifier 健康标记](#terminal-bench-verifier-健康标记)

## 共享环境与镜像

### 通用准备与验收

所有 benchmark 都通过统一 run、harness 和 artifact 入口执行。benchmark 子命令负责任务导入、profile 检查、镜像准备和专用 preflight；实际 rollout 与结果登记仍属于正式 run。精确参数以 `dycon swebench|skillsbench|webarena|harbor --help` 为准。

#### 通用流程

1. 使用 benchmark 导入入口准备任务定义，并通过 `catalog-check` 校验清单。
2. 使用固定 profile 与镜像入口准备运行环境；核对镜像身份和所选任务范围。
3. 运行对应 benchmark 的 preflight。任务容器、依赖和 verifier 检查不能由镜像存在性检查替代。
4. 创建 run 并冻结 tasks、profile、模型和资源配置，然后通过 `run start/wait/status` 管理 rollout。
5. 通过 `harness prepare/run/status` 对固定 rollout/patch 进行评分。

#### SWE-bench 与 OpenHands

使用注册的 SWE-bench/OpenHands profile、任务定义和镜像。patch 与 canonical rollout 通过 run evidence 绑定；并行/capture 设置必须在创建 run 前按项目配置约定启用。运行前做 profile、容器和 harness preflight。不要直接启动 OpenHands 或内部 runner 来绕过 run 计划。

#### SkillsBench 与 OpenHands

使用 `dycon skillsbench` 检查 benchmark 任务和 profile，使用 `skillsbench-images` 准备固定任务镜像，并在创建 run 前运行相应环境与任务 preflight。任务数据、verifier、网络依赖和 OpenHands ACP runtime 均由注册的 backend 适配；不应在任务容器内临时替换版本或凭一次 task smoke 推断全量数据可用。

若后续分析要求每步 `think_close` hidden，应在新 run 创建前设置并冻结显式 reasoning-close 生成配置和 capture anchor。语法字段、generation token budget、native close token 与已发布 hidden 必须分别核验；仅有 reasoning 文本或 capture 状态 complete 不代表每步 hidden 完整。具体接口使用 `dycon skillsbench audit --help` 和 [capture 指南](CAPTURE-CN.md)。旧 run 的配置不能通过改根 YAML 更新。

#### WebArena Verified

使用 `dycon webarena` 导入/校验任务，并通过 `webarena-images` 为显式选择的任务准备共享站点镜像。Verified 任务与原版 WebArena 的数据和评分定义不能混用。冻结任务、站点服务与镜像版本后，使用统一 run preflight 和 harness 入口。浏览器、站点容器和离线评测服务需分别通过对应环境检查。

#### Harbor backends

Harbor 集成通过 backend registry 绑定 benchmark、agent、runtime、subject 和 evaluator。使用注册的 Harbor CLI/profile 执行 SWE-bench Pro V2 或 Terminal-Bench 2.1，不要另建一套任务生命周期。对 OpenHands ACP transport、容器边界或 evaluator 做改动时，应通过对应 adapter 测试及目标 benchmark 的小规模 preflight 验证。

#### Evidence and limits

任务 catalog、run rollout、patch 和 evaluation 是不同身份。Harness 报告绑定选定 patch；仅 `resolved=true` 表示该次选定评测成功。环境 gold、单任务 smoke、单元测试和完整 benchmark 运行分别回答不同问题，报告时不可互相替代。


SkillsBench ??????????????? timing receipt ? [SkillsBench ????](#启动预检sdk-与压缩器配置)?


SkillsBench verifier resource releases and their exact profile/task snapshot references are documented in [verifier resources](#verifier-资源发布与资格验证). Local Python/Go/MathJax resources do not imply that every task or Maven business build is offline-qualified. New runs freeze the root configuration; old runs keep their original publication identities.

SWE-bench 官方分组和 NAS 同步见 [catalog 分类指南](#catalog-分组和镜像同步)。

### Docker 存储、网络与镜像

容器配置属于项目统一 runtime 和 benchmark backend。使用 `dycon` 提供的存储、镜像、环境和任务 preflight 命令；不要在运行期间手工改写 run 的容器快照、绕开 `ContainerFactory` 启动任务，或删除仍被 run 引用的镜像与卷。

#### 检查顺序

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon runtime-check
.venvs/runner/bin/dycon images-check
.venvs/runner/bin/dycon docker-storage check
```

各命令检查范围不同：配置校验根 YAML；runtime 检查执行环境；images 检查已登记的任务镜像；docker-storage 检查配置的数据根与 daemon 状态。随后仍需运行对应 benchmark preflight 或 harness gold 环境检查。

#### 数据根与镜像

需要变更 rootless Docker 数据根时，先按 `dycon docker-storage --help` 检查/应用 `config.yaml` 中声明的目标。变更会影响 daemon、镜像与容器可见性；确认服务状态和既有数据路径后再执行，不要通过手工改 daemon 文件创建第二套来源。

任务镜像由 benchmark/profile 固定。通过 `skillsbench-images`、`webarena-images` 或相应 benchmark 导入命令准备镜像，并核验冻结任务范围、镜像摘要与 NAS 路径。镜像可拉取不代表任务依赖、网站、浏览器和 evaluator 均可用。

#### 故障定位

先从 `run status`、`harness status`、`report run` 和 attempt 日志定位失败阶段，再使用同阶段的 runtime、image、preflight 或 gold 检查。保留容器标签、run/attempt 身份与原始日志，修复前不要清理故障现场。

依赖错误应检查失败 distribution、Python 版本与平台标签、wheelhouse receipt 及冻结 requirements；网络或镜像失败应分别检查服务可达性、镜像身份和容器输出。不要用宿主机“已安装”推断任务容器依赖满足，也不要把基础设施错误记作 benchmark unresolved。

`runtime-check`、gold check、单任务 smoke 与全量 benchmark 运行是不同等级的验证。变更后使用目标 benchmark 的 preflight 和最小 run 验证，再按任务规模执行正式工作负载。确切参数见 `dycon runtime-check --help`、`dycon docker-storage --help` 与 benchmark CLI help。
#### Docker 网络清理与地址池

任务退出按 `dycon.project/run/execution` 清理；单个 SkillsBench 环境另绑定 task/sample/attempt/epoch。
正常结束、失败、取消和 supervisor 异常退出均核验容器及网络。仅删除归属匹配且已无附着容器的网络，
不执行全局 prune，不断开其他容器连接，也不清理其他 run 或 execution 的网络。清理回执含
`removed_networks`、`remaining_networks`，存在残留时 `cleanup_complete` 不得标为成功。

用户 rootless Docker 使用 `~/.config/docker/daemon.json`，不要误改 `/etc/docker/daemon.json`。
例如 `default-address-pools: [{"base":"172.16.0.0/16","size":23}]` 可提供 128 个自动分配子网；
必须先确认网段与现有网络和访问目标无冲突。生效需要重启对应用户 Docker daemon，已有网络不会缩小。
重启前通过正式 cancel/wait 停止目标 run，并核对同一 daemon 上的其他工作。

#### 补齐官方 Verified / dev 镜像

`dycon swebench images-pull STORE_ID --plan PLAN.json --proxy HTTP_PROXY --workers 2 --attempts 3`
按固定官方版本的完整 ID 集校验补齐计划；`--check-only` 只校验计划、已有记录哈希和 NAS 图引用。
计划 schema 1 包含 `source`（dataset/revision/split）、`official_images`（ID 到官方镜像名）、
`existing`（ID 到 store/path/manifest_sha256）和 `missing_ids`，两组必须严格覆盖官方完整集合。
当前支持固定 Verified-500 与 dev-225 版本，来源见 `runtime/images/swebench_pull.py`。

命令复用 registry 下载、Range 续传、SHA256 校验和 OCI 发布格式；保留完整平台/attestation 图。
每项首次取得的 manifest digest 存入 `dataset/metadata/`，重试或重启均复用它，避免 latest 漂移。
仅在全部 blob 校验通过后写入公开 `images/ID.json`；相同 store 的排他锁防止重复进程写入。
任务失败后继续其他任务，最多执行指定轮数；再次运行相同命令即可恢复。
仅准备 NAS 镜像，不导入 catalog、不启动 Docker，不代表 harness/gold 通过。

NAS 产物：`dataset/plan.json`、`download-status.json`、`transfer-progress.json`、
`attempts/`、成功后的 `coverage.json` 和 `completion.json`。
首次网络正文到达时原子写入 `transfer-started.json`，可用 inotify 做一次性启动确认；
该历史事件需结合当前 systemd 进程状态判断存活，不要将它作为持续进度。
新镜像做完整 SHA256 校验；已有镜像复核记录哈希及图引用存在性/大小，不重新读取全部大层。

## SWE-bench

### Catalog 分组和镜像同步

正式清单为项目 `catalog/manifest.json`。SWE-bench 使用三个 benchmark 名称：

| 类别 | catalog benchmark | 定义 |
|---|---|---|
| verified | `swebench_verified` | 官方 Verified 的 test 子集 |
| dev | `swebench_dev` | 官方 SWE-bench dev |
| nonverified | `swebench_nonverified` | 官方 SWE-bench test 排除 Verified；不含 dev |

三类均由已有 SWE-bench backend 执行。历史名称仍保留 backend 兼容能力，以便加载冻结 run。
`partition` 继续表示项目内部 train/val/test/excluded 用途，不表示本表官方分类。

#### 查看与选择

```bash
.venvs/runner/bin/dycon swebench catalog-list
.venvs/runner/bin/dycon swebench catalog-list --group verified --output temp/my-selection/verified.json
.venvs/runner/bin/dycon run create MY_RUN --operation rollout --tasks-file temp/my-selection/verified.json
```

`catalog-list` 输出分组数量、选中数量和 ID 哈希；不会打印全部任务。`--output` 写入兼容正式
`run create --tasks-file` 的 `{tasks: [instance_id, ...]}`，拒绝覆盖已有文件。创建 run 时仍需按
项目配置约定配置 `benchmark.backend: swebench`、agent、模型、环境等。该任务文件记录导出时
catalog 身份，创建 run 将通过现有 run 接口绑定当时正式 catalog，不是历史快照恢复接口。

#### 同步现有 NAS 镜像

```bash
.venvs/runner/bin/dycon swebench catalog-sync \
  --source PINNED_OFFICIAL_SOURCE.json \
  --image-plan /VM/wangtianyi/DyConGlobal/VERIFIED_STORE/dataset/plan.json \
  --image-plan /VM/wangtianyi/DyConGlobal/DEV_STORE/dataset/plan.json \
  --output catalog/changes/UNIQUE_PREVIEW_ID
```

默认预览。确认指定来源和范围后，使用新的 output 目录并添加 `--apply` 进行原子同步。
来源文件 schema 1：`sources` 提供 verified/dev/test 的 dataset/revision/split/ids 等溯源；
`rows` 提供完整 verified/dev 官方任务正文。支持的固定版本与 ID 集哈希见
`catalog/swebench_sync.py`。两个 image-plan 为 `swebench images-pull` 使用的固定完整计划，
包含既有镜像引用和缺失 ID。同步只接受对应固定官方数据与镜像名。

同步会对旧 SWE 任务按官方 ID 重新分组、重建 catalog 键；补充 NAS 已有且未登记的 verified/dev。
旧任务正文、revision、partition、环境 overrides 和镜像 pin 保持原样；新任务使用官方完整正文、
已有可验证 NAS 引用和项目 train partition。未知官方归属或多个旧 revision 的重复 ID 会拒绝，
不会任选一个。新任务可选不代表其容器环境已经通过 gold 验收。

NAS `images/ID.json` 必须存在，镜像记录哈希与完整 OCI 图引用大小检查必须通过，才会登记。
下载失败、只有 metadata 或 partial blob 的任务留在报告 `missing_official_ids`，不会进入可选清单。
同步既不发起下载，也不自动改用其他 repair store；已有任务保留原来显式绑定的镜像引用。

catalog 通过原有事务锁更新，只替换 SWE 子集，保留其他 benchmark；提交前检查 catalog 未被
并发修改。`--output` 拒绝复用旧目录。应用产物包含原 catalog 精确备份、更新后 catalog、官方来源、
旧新任务键映射和带前后哈希的报告。重复同步相同输入不增加 catalog revision。

已有 `runs/*/plan.json`、`selection.json`、`config.yaml`、`resolved-config.json` 不被同步修改；
旧 run 继续使用自身冻结任务与旧键。无需移动或复制 NAS 大镜像层。

### Gold 排查、验证回执和评分结算

通过正式 CLI `.venvs/runner/bin/dycon harness audit RUN_ID --output ABSOLUTE_NEW_DIRECTORY [--labels JSON]` 对既有物理 attempt 进行只读审计。输出位于历史 runs 之外，拒绝覆盖既有目录。分类器与在线正式结算共用。原始官方 report 不变；新评测的有效 resolved/reward 在落盘与训练消费前按审计投影。

当前策略：`swebench_verifier_audit_v3`。输出保留官方结果，并区分：

- `ValidOutcome`：评分证据有效，`effective_resolved` 为官方布尔值。
- `VerifierError`：该次评测不能提供可信训练标签，`effective_resolved=null`。`cause_confirmed=true` 才表示具体根因已经证实。
- `Review`：输入身份、文件完整性、证据归属或普通模型提交责任仍未核实；不生成有效零分。

#### Gold 控制策略

Gold 模式必须核对冻结任务的 reference patch、harness 输入 patch 摘要；有官方 `patch.diff` 时还须逐字相同。仅有 run 名称或 gold 标记不足以启用该策略。文件/身份损坏仍归 Review。

身份有效的 gold 控制返回 false 或未完成，归 VerifierError。若无法确认原因，使用 `gold_control_failed_unattributed` 或按实际观察细分为 `gold_dependency_api_failure`、`gold_runtime_recursion_failure`、`gold_runtime_warning_failure`、`gold_native_process_crash`、`gold_test_collection_failed`、`gold_evaluation_timeout`、`gold_required_test_evidence_missing` 等；此时 `attribution=unknown`、`evidence_strength=control_failure`、`diagnostic_status=needs_investigation`。它表达控制评测失效，不证明依赖/运行时已经被确认不兼容。具体异常候选与缺失证据继续保留。

只有人工排查提供与当前证据摘要绑定的 `gold_patch_defect` finding，明确 `attribution=gold_patch`、confirmed、未恢复、basis 和 supporting_sources，且实际为无其他验证故障的完整断言失败，才允许 gold 的有效 false。此类判断不能自动用于其他 attempt 或不同日志。普通模型提交的正常断言失败仍是有效 false；不会因为分数为零而自动成为 VerifierError。

#### 解析与执行细节

pytest 仓库嵌套测试的输出可能使官方 parser 把 `[100%]`、`[`、`[100%]----` 等进度文本登记为测试 ID。这些已知形式保留在官方覆盖记录，另列 `coverage.parser_artifact_ids`，不作为真实测试冲突。其他 ID 和真实 node ID 冲突照常核查。

Pydicom 的官方 parser 会把绝对路径参数缩为最后一个路径分量。冻结 required ID 未做同样转换时，记录精确的 `parameter_identity_mismatches`，不通过模糊匹配或更改分数掩盖它。

Docker 的 stdout/stderr 分块合并可能把有效测试回执附到结束标记之后。只有原始 output SHA256 和 test command SHA256 均绑定时，才允许从完整输出读取唯一回执；重复、缺失、摘要错误仍拒绝。pytest、tox 和 shell 最终 returncode 保持各自语义。

官方 reward=1 仍需真实评分证据。确切必需测试声明缺失 mandatory tool 而 skipped 的 gold，归 `missing_verifier_tool`；不得将全量 skipped 的成功退出当成全部测试通过。历史 Django-15521 缺 xgettext 的 63 个 skipped 就属于这种情况。修复并通过新 run 验证后，才可能得到有效正例。

#### 证据与版本迁移

原始 official report、selection、patch 和既有 run 产物不修改。审阅 finding 必须绑定当前 evidence_digest 及实际 source hashes；升级 collector 后旧 finding 不会自动继续生效。需要显式重核旧摘要与所有来源，再生成新标签和迁移回执。历史未知根因不会因改变策略而变成已确认根因。

本次策略由 classifier/collector/adapter 回归与历史 gold 回放验证。控制评测分类可剔除无效标签，但不代表环境修复或 gold patch 全量通过。

#### 正式评分结算与训练消费

新 SWE-bench harness 默认使用 `swebench_validity_settlement_v1`；`verifier_audit_mode=settlement`，不再只是旁路。ValidOutcome 产生 completed、布尔 resolved、reward=0/1；VerifierError/Review 使用同名 status，resolved/effective_resolved/reward 均为 null，scoring_eligible=false。原执行字段保存在 raw_outcome，原官方结果保存在 official_result，官方 report.json/test_output.txt 不改写。审计文件引用包含 SHA256。

`RunEvidence.record_evaluation` 将有效投影登记到唯一 evidence，并验证 null 与状态约束。在线成功剩余量标签、difficulty resolved 选择、steering/RL source selection 通过同一 training_label 消费所选 evaluation；不会把旧 sidecar 的无效结果当作成功。旧 SWE-bench 无审计结果可读取，但不为新成功标签提供有效 bool。增量评测继承的未结算旧结果变为 legacy_unsettled_evaluation 的 Review，历史文件不重写。

新 RL 数据构建将明确 VerifierError/Review/基础设施错误轨迹的整个分组标为不可训练，保留分支、原 historical_score、native token 和 terminal_scoring_exclusion；general audit.enabled=false 也执行此评分限制。GroupBatch 还有消费端阻止绕过的校验。尚未评测的旧共识数据保持可表达；共识 historical_score 与终局 reward 是不同字段，不把其中任意一个代替另一个。已有 sealed 数据集、prepared/checkpoint、运行中训练和其冻结优势不追溯改写。

普通提交空 patch 可确认是有效零分；有精确 patch hash、官方 git syntax rejection 且尚未执行测试的损坏 diff 也可归有效零分。泛化 hunk mismatch 不能据此认定模型责任，保留 Review。重试决策读取 raw 执行事实，后续成功不会继承前次 VerifierError/Review。

已部署代码的 CPU 集成测试覆盖真实 collector/parser、harness、evidence、训练选择和 RL dataset/group 边界；模型/GPU训练质量与环境修复的验收另行记录。

## SkillsBench

### 启动预检、SDK 与压缩器配置

Verifier 国内包索引、新 profile/任务兼容变体与验证入口见
[SkillsBench verifier 包索引策略](#包索引策略诊断与清理)。

在根 `config.yaml` 中配置，并通过正式 `dycon run create` 创建新 run：

```yaml
execution:
  preflight:
    long_checks: true
serving:
  startup_mode: serial  # serial | parallel
```

两项均随 `resolved-config.json`、配置 hash 和 run 计划冻结。`run start`、`run resume` 及 `run retry-failed` 后的 resume 读取原 run 快照；修改根配置不会更新已有 run。缺省字段分别按 `true` 和 `serial` 执行，解析旧 YAML 时不额外序列化这些字段，因此历史配置 hash 不变。新建 frozen clone 默认继承源 run；可在创建 clone 时显式覆盖 `serving.startup_mode`，不修改源 run。

#### SDK 格式恢复与提交导出验收

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

#### 长耗时检查开关

`execution.preflight.long_checks: false` 只控制 SkillsBench/OpenHands 的共享启动预检，其他 backend 保持原有行为。关闭后跳过：

- 已登记模型文件的内容 SHA；仍核验登记身份、清单结构、文件路径、存在性和大小。`model.verify_identity` 继续保持独立含义。
- 任务包所有文件的内容 SHA；仍核验冻结清单身份、声明路径及存在性、文件模式、任务能力和输出声明。
- NAS OCI descriptor 的深度 SHA；仍遍历完整 descriptor 图并核验路径、存在性、大小、NAS manifest 引用和冻结 image identity。预检前的镜像租用/导入也使用同一策略；导入镜像本身仍需要读取和传输实际镜像内容。
- OpenHands 依赖快照逐文件 SHA；仍核验 manifest receipt、冻结 manifest SHA、文件清单、路径、存在性和大小。
- 全量任务容器环境审计；不为所有任务创建审计容器、安装依赖及运行 skills/agent 探针。

配置、backend/profile、generation/execution context、版本接口、GPU/MIG 探测、Docker daemon 和 CPU quota 检查仍执行。正式 physical attempt 的任务包和模型完整校验、compose image identity、依赖安装、skills visibility、agent environment probe、清理和取消逻辑保持执行；失败重试也不继承“免验证”状态。

`validation/startup-preflight.json` 中 `checks` 分别记录 `model_sha`、`task_package_sha`、`oci_deep_verify`、`dependency_snapshot_sha`、`environment_audit` 的状态和耗时。跳过项为 `status: skipped`，原因包含 `execution.preflight.long_checks=false`，不会标记为 verified。镜像提前准入记录在 `image_admission_checks`；runner 在 adapter 前的模型校验记录在 `validation/startup-model-identity.json` 并汇入共享预检 receipt。`ready` 仅表示当前策略的启动门禁满足，不代表跳过的完整性审计已经完成。独立 `skillsbench preflight RUN_ID` 同样服从冻结策略。

需要完整性审计时保留 `true`。设为 `false` 后，启动门禁无法发现文件内容被同大小替换的情况；正式 attempt 的验证仍可能发现相关变更。

#### 多卡 TP=1 并发启动

在既有 `serving.trajectory_sharding: per_gpu_tp1` 配置中设置 `startup_mode: parallel`，先顺序准备所有 worker：配置与端口检查、共享 capture controller/token inventory/control server 初始化，以及各设备的资源准入。随后并发创建各自进程并等待 `/models` 就绪；所有 worker 成功后才向轨迹调度器开放 pool。

任一 worker 准备或启动失败时，取消其余 worker 的启动等待，等待启动线程退出，再按副本到 capture owner 的顺序终止进程组、关闭共享控制服务并释放设备资源。每个 worker 继续使用自己的 GPU、端口、进程组和原有设备锁策略。TP、轨迹调度、每卡 CPU 并发及模型请求语义不变。单服务模式仍只启动一个服务；现有 MIG 配置限制保持原规则。

`serving/worker-pool.json` 和每个 worker 的 `launch.json` 记录 startup mode；各 worker 的 `startup-timing.json` 及 pool receipt 记录准备、进程创建、API 就绪和清理时间。`model_load_and_api_seconds` 是进程创建到 `/models` 就绪的总间隔，包含模型加载、engine 初始化和 API 启动，不能当作纯权重读取时间。

可使用 `dycon run start RUN_ID --check-only` 做正式启动验收。用同一模型、同一组空闲 GPU、同一 NAS 存储分别创建 serial/parallel run，比较 worker timing 和整体墙钟时间。GPU 竞争、文件系统/page cache 和编译缓存状态会影响实测值；默认仍为 serial。


#### OpenHands 压缩器的根配置

在根 `config.yaml` 的 `agent` 节设置 token 阈值、事件数量阈值和保留开头事件数：

```yaml
agent:
  condenser_max_size: 80
  condenser_keep_first: 4
  condenser_max_tokens: null
```

这两个值分别传给原生 `LLMSummarizingCondenser.max_size` 和 `keep_first`。
`max_size` 统计事件，不是 token 或 assistant step；事件数量超过阈值时触发压缩。
默认值是 `80/4`。`condenser_keep_first` 可以为 0，必须满足
`condenser_keep_first < condenser_max_size // 2 - 1`，给原生压缩器留出摘要和最近事件的位置。
根配置通过严格类型校验拒绝负值、浮点值和不合法组合。
`agent.condenser_max_tokens: null`（YAML 空值）表示自动使用
`serving.max_model_len - generation.max_tokens - 1` 作为 token 压缩阈值。
例如 `262144 - 98304 - 1 = 163839`。显式整数直接作为阈值，优先于 profile 中的值；
省略该字段则沿用不可变 profile 的 `condenser_max_tokens`，保持旧配置的行为。
显式或自动计算的阈值必须至少为 1024，自动值不满足时配置校验直接报错。
不要写字符串 `none`；应写 `null` 或 `~`。token 与事件数量阈值可以同时启用。
该设置适用于共用 OpenHands driver 的 SkillsBench 和 SWE-bench，要求现有已验证的依赖发布物；
不会修改 profile 内容、profile 哈希或资格证据，也不需要重新发布 profile。

新 run 会将事件设置和最终 token 阈值冻结到 `execution_context.condenser`；其中 `max_tokens`
记录最终值，`max_tokens_source` 记录自动计算或根配置来源。运行时通过已有
`OPENHANDS_CONDENSER_MAX_TOKENS` 环境变量传给原生压缩器，执行和恢复只读冻结设置。
创建 run 后再改根配置不会影响该 run；旧 run 不会被补写新字段或改变 context hash。
旧依赖快照可以继续使用原有 `80/4`；非默认设置必须绑定包含
`configurable_condenser_events_v1` 适配记录的新依赖发布物，否则创建 run 时直接报错，避免静默忽略。

现有统一发布入口支持从已验证的依赖快照离线派生新发布物，只更改 CLI 的压缩器设置适配及对应 wheel/lock 哈希：

```bash
.venvs/runner/bin/dycon skillsbench dependencies-build NEW_SNAPSHOT_ID \
  --from-snapshot runtime/dependency-snapshots/EXISTING_SNAPSHOT_ID
.venvs/runner/bin/dycon skillsbench profile NEW_TEMPLATE_ID \
  --from-profile runtime/agent-profiles/EXISTING_PROFILE_ID/manifest.json \
  --dependency-snapshot runtime/dependency-snapshots/NEW_SNAPSHOT_ID
.venvs/runner/bin/dycon skillsbench qualify EXISTING_TOOL_PROBE_RUN NEW_QUALIFIED_ID \
  --profile-ref runtime/agent-profiles/NEW_TEMPLATE_ID/manifest.json
```

资格迁移会完整校验父子发布物，只接受已知的压缩器源码差异、wheel RECORD 和锁定 hash 更新；
SDK、tools、解释器或其他 CLI 改动不符合迁移条件。随后将根配置的 `agent.profile_ref`
指向新 qualified profile，再通过标准 `run create/start` 创建和启动新 run。
不要修改旧 run 快照或旧依赖发布物，也不要修改已安装的 site-packages。
#### 任务内存覆盖

`dycon skillsbench profile NEW_PROFILE --from-profile PROFILE.json --memory TASK_ID:65536`
以 MiB 为单位为指定任务设置 64 GiB 容器上限；`--memory` 可重复。新 profile 保持不可变，
内存覆盖在支持或不支持 CPU quota 的 daemon 上均生效，并记录 resource-adaptation 回执。

已有 run 必须先正式 cancel/wait 并确认 cleanup_complete，然后使用：

```bash
dycon run revise-config RUN_ID --revision-id memory64g --set sandbox.memory_mb=65536 --reason 'User requested 64 GiB'
dycon run resume RUN_ID --detach
```

`sandbox.memory_mb` 是全局默认值，单位为 MiB，16 GiB 应填写 16384。既有 profile 的任务专属覆盖优先于全局默认值。
统一接口同时支持其他配置字段，复用离线修订锁、前后快照和哈希审计；普通内存修订保留 selection、样本 seed 和历史结果。
接口不宣称完成新 GPU 验证，详见[配置修订](MINIMUM-RUN-CN.md#配置修订和事务恢复)。
既有终态失败不会由 resume 自动重试，需先显式 `run retry-failed`。
内存上限不是预分配，所有容器实际用量仍受主机内存容量约束。

### 主轨迹生成 token 预算

在根 `config.yaml` 的 `execution` 中设置 `sample_completion_token_budget: 正整数`，通过 `dycon run create` 冻结。省略或为 `null` 时不启用；仅 SkillsBench/OpenHands 支持。字段名称、类型、配置规范化和 snapshot hash 算法不变，不改写历史冻结配置。2026-11-14 起，并行运行的该上限约束**主轨迹累计 completion tokens**，与单分支 baseline 对齐：包含已选 reasoning+action 和独立 condensation，排除 prompt 以及未选候选。它不是 context window 上限，也不是全部候选的算力预算。

预算绑定 `run_id/task_key/sample_key/repetition_id/attempt_id`。物理 attempt 如 `rep-000-attempt-001`，不同于 run execution attempt。初始执行、基础设施重试、seed resample、limits retry 和正式 `run retry-failed` 后的 resume，只要分配新物理 attempt ID，就获得完整额度，不继承旧 attempt。仅新建 execution attempt 不会重置同一个物理 attempt。

baseline 保持原有 schema 2 和逐物理调用结算，包括 condensation、单分支生成及其重试。并行配置使用 schema 3、`scope: physical_attempt`、`accounting: selected_trajectory_completion_v1`：每个物理调用的 usage 都记录在 `requests`，但 agent_decision 候选只有最终选中的一条才设置 `charged: true`；独立 condensation 仍直接扣费。`completion_tokens`、`by_role` 和剩余额度仅统计 charged 项。费用、attempt `usage` / `output_tokens`、coordinator `total_usage` 继续表示全部分支实际物理消耗，可能远大于主轨迹预算；这些字段不再用于解释主轨迹剩余额度。

账本位于 `runs/RUN/rollout/sample-token-budgets/SAMPLE/ATTEMPT.json`，通过冻结物理存储根解析；锁为同目录的 `ATTEMPT.lock`，沿用原子替换。旧 schema 1 sample 账本和 schema 2 历史并行 attempt 保留，不迁移、不重置、不将总消耗解释为已选长度。用新并行实现打开同一旧 schema 2 attempt 会拒绝；正式恢复应保留旧证据并分配新物理 attempt。禁止手改账本或历史 `attempt.json`。

并行候选都可以使用“主轨迹上限减已选累计值”作为各自的输出额度，实际 `max_tokens` 不超过该值及配置的单次上限，native request owner 同步该限制。互斥候选不会因为其他候选的预留而共享瓜分额度。生成完成先保留实际 usage；统一选择器选定后按 request/API attempt 精确绑定，仅结算选中的那条。重复结算幂等，同一逻辑请求的不同胜者、跨请求/attempt 结算、损坏计数或错误 schema/accounting 都拒绝。

未选候选缺失 usage 时保留其 unknown 证据与整体 `usage.complete=false`，但不据此阻止已知 usage 的胜者。选中响应缺失精确 usage，或独立 condensation 的 usage 未知，则阻止该 attempt 后续调用并以 `SampleTokenBudgetUsageUnknown` 停止；不填零。单分支 baseline 的 unknown 行为保持不变。

选中响应使主轨迹额度耗尽时，保留结算与停止 receipt，但不将该响应作为可执行 action 返回，终态 `SampleTokenBudgetExceeded`。若某个候选因主轨迹剩余额度限制而以 length 结束，它作为预算受限候选退出；较短的有效胜者仍可继续。所有候选均因此失败时保留预算停止原因及 receipt。单次 generation 自身的上限、reasoning loop、选择失败和实际 context window 拒绝继续分别处理。

`SampleTokenBudgetExceeded` 和 `SampleTokenBudgetUsageUnknown` 的 disposition 为 failed，不属于 `LimitsExceeded` / `reasoning_loop`，不自动换 seed 重采样。主动重跑使用正式 `run retry-failed --exit-status SampleTokenBudgetExceeded` 再 `run resume`；先停止并确认退出/清理，检查并恢复已有 capture，保留旧轨迹和账本。禁止覆盖冻结配置或手改 run 状态。

真实服务端 HTTP 400 上下文长度拒绝归为 `ContextWindowExceededError`，保留 `diagnostics/context-window-error.json` 和模型调用证据。仅失败决策的所有实际候选均明确被上下文拒绝时归此类；恢复成功或混合失败不强制分类。accepted 表示接纳轨迹结束，不表示解决任务。正常及 finalization 异常路径保留主因，capture、清理和取消约束独立生效。

同一物理 attempt 重建 bridge 复用其匹配 schema 的账本；不能以重建 bridge 清除 unknown 或重新发放额度。已经启动但中断的 attempt 由统一 runner 恢复证据或分配新 attempt，不重新执行旧调用；仅预留、从未启动的 attempt 可以沿用 ID 开始执行。

停止证据包括 `sample-token-budget.json`、attempt 账本、模型调用 `sample_budget` 和选择后的请求级 `sample_budget`、native terminal receipts、`attempt.json` 的 `sample_token_budget`。并行 ledger 的 schema/accounting 明确标识新口径；旧产物不回写。

本预算不替代 `serving.max_model_len`。实际 context window 约束单次 prompt+completion，压缩可缩短后续 prompt，但不减少已经累计的 completion。context preflight 和服务端输出余量处理属于独立约束。不同硬件与墙钟时间不能直接据此比较任务质量。

### Verifier 资源、发布与资格验证

<!-- agentops-public-release-20261111 -->
Current default AgentOps release (2026-11-11): the py38-py312 snapshot and public qualified profile supersede the older defaults described below. See [AgentOps release identities and usage](#agentops-多解释器-verifier). Older six-environment semantics and artifacts remain historical references.

`skillsbench-verifier-resources-v1` 在原 agent dependency snapshot 内增加独立的 `verifier/` 命名空间。agent 的 CPython 3.12、wheel、安装锁及来源元数据逐项继承；verifier 按 Python ABI 独立锁定。旧 snapshot、profile、任务包及 run 不作原地升级。

#### 发布身份与使用入口

本轮资源发布为 `skillsbench-verifier-unified-v2-20261110`，实际路径为 `/data3/wangtianyi/DyConGlobal/runtime/dependency-snapshots/skillsbench-verifier-unified-v2-20261110`。

| 对象 | 正式引用 | SHA256 |
|---|---|---|
| 依赖 manifest | 上述目录的 `manifest.json` | `c3082581c51bf76e2a170eb98125c9f90798b0477e031480c50887e5eb9b3f72` |
| Qualified profile | `runtime/agent-profiles/skillsbench-verifier-release-v2-20261110-qualified/manifest.json` | `d8df378fec78419fd6fb3fc51dec8c0caff18f12dd59759f0bbede3f42181932` |
| 75 任务快照 | `catalog/snapshots/skillsbench-verifier-repair-candidate-v2-20261110/manifest.json` | `98bec2a4c33885fa516f43c90a087d51c328154e043c697685b6a2b689c83d78` |

依赖发布 receipt 为 `8463076f1e10f0e67b972c99aee9047963bcacce3d26f86127d5a3d71fad2a73`。发布保留 65 个原任务，另外 10 个任务使用按内容 hash 注册契约的修复变体。镜像身份沿用原发布。准确验收、默认切换状态与限制见本次 `IMPLEMENTATION_REPORT.md` 和 `publication.json`，不得将候选目录存在等同于全量验收成功。

标准新 run 只读取根 `config.yaml` 的 `benchmark.snapshot_ref` 和 `agent.profile_ref`；已有 run 继续读取自身冻结配置。运行前检查当前根配置确实选择上表配套引用，然后通过正式入口创建、检查：

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout --repeat-count 1
.venvs/runner/bin/dycon skillsbench preflight NEW_RUN --workers 4
```

`run create` 只冻结计划，不启动模型或 GPU 工作负载。`execution.preflight.long_checks: false` 会跳过完整镜像/依赖字节校验和环境审计，不能将这种快速 preflight 宣称为完整验收；实际环境安装可独立运行 `skillsbench environment-check NEW_RUN --workers 4`。完整验收 run 应在创建前把该开关设为 true，并核对冻结结果。

#### 构建、资格与回执

```bash
.venvs/runner/bin/dycon skillsbench dependencies-build NEW_ID \
  --from-snapshot PARENT_PATH --verifier-resources INPUT.json
.venvs/runner/bin/dycon skillsbench dependencies-verify SNAPSHOT_PATH
.venvs/runner/bin/dycon skillsbench job-start --seconds 21600 JOB_ID -- \
  dependencies-build NEW_ID --from-snapshot PARENT_PATH --verifier-resources INPUT.json
.venvs/runner/bin/dycon skillsbench job-status JOB_ID
```

输入采用 `schema_version: 1`，包含 `runtimes`、`environments`、`tasks`、`assets` 和可选 Go 闭包。每个环境用独立 hash lock、目标 ABI wheels 和实际离线安装探针验证；复用环境绑定完整 snapshot manifest hash。构建输出遵循项目 artifact volume 配置；日志在 `temp/skillsbench-dependencies/ID`。失败 staging 保留，重试使用新 ID。Go 下载并发为 4，保留 checksum 校验和有限重试。wheel 来源、版本、Requires-Python 和许可证记录随发布保存。

```bash
.venvs/runner/bin/dycon skillsbench profile TEMPLATE_ID --from-profile OLD_PROFILE \
  --dependency-snapshot SNAPSHOT_PATH --verifier-resource-policy skillsbench-verifier-resources-v1
.venvs/runner/bin/dycon skillsbench qualify SOURCE_PROBE_RUN QUALIFIED_ID \
  --profile-ref runtime/agent-profiles/TEMPLATE_ID/manifest.json
```

资格转移由正式接口重新核验原始模型请求、实际工具调用与结果、agent 安装证据和依赖派生链，不直接复制旧资格字段。`profile --from-profile` 会清除资格；用于验证网络模式的新 profile 不因此自动获得 capture/parallel 资格。本次接续没有请求模型。

#### 运行时资源隔离

snapshot 只读挂载。verifier 使用固定本地解释器、uv/pip 离线模式，禁止隐式 Python 下载。OpenHands 私有安装仅使用原 agent hash lock；新资源策略下安装子 shell 会清除 verifier 的 `UV_CONSTRAINT`、`PIP_CONSTRAINT` 及 build constraints，避免任务 ABI 的版本锁污染 agent 闭包，verifier 随后仍使用自身约束。环境回执在新策略下包含 `verifier_dependencies.py` 源码 hash，资源安装逻辑变化后重新验收。

完整资源字节校验在私有安装/公共 oracle 的异步钩子中通过线程执行，同一事件循环中对同一冻结身份的并发请求只共用正在执行的那次完整校验，结束后立即移除等待记录，后续请求重新完整校验；单个等待者取消不会取消其他等待者的校验。完成后每个调用仍核对冻结 manifest hash；不在创建环境的同步回调中阻塞其他沙箱的子进程超时计时。环境审计的任一后置异常均令 verified=false，带 error 的旧“成功”回执不得复用。修改这类运行时代码后，最终验收使用新的源码身份重新运行，保留旧诊断回执。

Go 使用本地 1.24.4、完整模块闭包和 attempt 私有可写 module/build cache。MathJax 3.2.2 通过受控 URL 路由使用固定本地资源，Chromium 与 Playwright 版本绑定。public 有代理、public 无代理和 none 使用同一资源选择；none 还通过 Docker `NetworkMode=none`、仅 lo 网卡和离线私有安装回执证明无外网。

本地依赖可用不等于 75 个任务均可离线。Google Auto 原 BugSwarm 脚本还运行 apt 与 Maven；Maven 网络下载及 Java 7 兼容性必须独立验收。不能通过放宽 900 秒预算或删掉原构建步骤宣称通过。

#### 验证与错误边界

```bash
.venvs/runner/bin/dycon skillsbench environment-check ENV_RUN --workers 4
.venvs/runner/bin/dycon skillsbench verifier-control CONTROL_RUN --case all
.venvs/runner/bin/dycon skillsbench verifier-control CONTROL_RUN --case simpo-wrong-rerun --retry-failed
.venvs/runner/bin/dycon skillsbench audit-rewards HISTORICAL_RUN --output NEW_AUDIT_DIRECTORY
```

长任务通过 `skillsbench job-start` 托管并查询指定 job，禁止重复启动或遍历所有历史 run。同一 run 的 oracle/control 共享网络生命周期，必须顺序执行；环境审计应使用另一个 run。

SimPO 检查独立参考的 SHA256，关闭 NPZ 后复制数组；重跑前删除旧输出，重跑后重新读取新输出。正确、错误、缺输出、缺环境说明及旧输出正确而重跑错误均有对照。GT 缺失或损坏仍是 verifier 故障。受保护阶段产物绑定执行器 call、任务包、snapshot、verifier 来源及下载文件 hash；缺输出 gate 还核对运行前 subject inventory。

LaTeX 正确公式、错误公式及 MathJax 资源缺失分别验证；原 renderer 与本地 renderer 在相同 MathJax 字节下逐 PNG 比较。Fuzzing 保留 general 与五个库共六个套件，不能跳过 IPython 或改动 Black/minisgl driver 来提高分数。

AgentOps 保留 py37–py312 完整 tox matrix。最新 v2 实测 py37 已选用本地 coverage/langchain wheel，随后因 `langchain-core` 没有兼容 Python 3.7 的版本而失败；py38–py312 可通过。完整矩阵的 reward 仍为 0，分类为 VerifierError；`agentops-upstream-blocked` 控制通过只表示正确识别该阻塞，不能宣称完整构建通过。原 stderr、tox JSON、各环境返回码和原提交错误检查都保留。

历史审计只读取旧证据，不将新包语义套到旧包。旧 SimPO Review 保留；相邻 PDDL 和完整评分模式矩阵继续是独立遗留项。回滚仅恢复后续新 run 的原 profile/任务快照配套引用，保留所有已发布版本，不改变已有 run 的冻结身份或原奖励。

### 包索引策略、诊断与清理

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

#### 追加修复与验证证据

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

### Zero reward 复查

在线结算和离线复查共用 `skillsbench_verifier_audit_v2` 判定核心。在线评测归档原始回执、verifier 日志、CTRF 和 subject 输出后，在写入最终评测及 attempt 回执之前审计原本可接受状态的有限数值 reward < 1。证实必需评测阶段因 verifier 故障中止时，状态改为 `VerifierError`、resolved 置空，原始 reward 不变。`ValidOutcome` 和 `Review` 保留原结算。负数进入核心并返回 `Review/invalid_reward_range`，同时保留 worker 范围校验产生的 `UnsupportedRewardPolicy` 状态。布尔值、空值和非有限数不转换为零。

`VerifierError` 表示有证据证明必需阶段失败且没有有效恢复；`ValidOutcome` 表示绑定的评测契约和计分证据支持该结果，并不代表任务成功；`Review` 表示证据、归属、顺序或契约不足以作出上述判断。部分分数和其他测试通过不能覆盖已证实的必需阶段故障。不同日志流缺乏顺序时不能推定恢复；多个 verifier 调用缺乏最终调用绑定时返回 Review。

契约适配绑定冻结任务包清单及相关源码哈希。未知或变更的契约不会借用同名任务的适配器。SimPO 参考路径和提交路径重合及 PDDL 说明与语法冲突保持 Review。归档读取只读取限制大小的普通成员，不提取、执行或跟随链接；被请求的链接和路径穿越会被拒绝。保存的发布元数据用于 AgentOps 依赖兼容性判断，历史运行时参考输入本身没有逐字节证明。

新 attempt 的诊断回执保留 `verifier_zero_audit` 字段名，Decision 的 schema version 为 2。旧零分 tuple API 必须提供冻结任务身份，且只将零分的 `ValidOutcome` 映射为旧标签 `ValidZero`。已结算 attempt 读取既有回执，不重新分类；半写入的最终评测交由现有恢复流程处理。已运行 worker 是否加载新代码取决于进程加载时机，部署或重启需遵循项目正式流程。

#### 离线复查

```bash
.venvs/runner/bin/dycon skillsbench audit-rewards RUN_ID --output OUTPUT_DIR
```

命令使用冻结计划和 run status 的当前 sample/attempt 投影，复查当前 accepted 且有限数值 reward < 1 的样本。在线新判为 VerifierError 的样本不属于这个 accepted 范围；比较两个入口时必须使用相同的原始证据和候选范围。命令不运行模型、容器或 verifier，也不修改历史回执。输出目录必须是新的项目拥有目录，生成 `summary.json`、`samples.jsonl`、`samples.csv` 和 `report.md`。汇总报告 schema version 为 1，逐样本 Decision schema version 为 2。

退出码：0 表示报告完整生成，包括含 Review 的报告；1 表示运行、加载或输出错误；2 表示命令参数错误；3 表示读取不完整或扫描期间来源变化。活动 run 的结果仅代表扫描时点。日志和归档只证明已保存证据的内容，不证明未保存的参考输入或科学计算结果。

### 离线评分审计的输入和输出

#### SkillsBench accepted reward audit

Use the read-only audit entry point for a frozen SkillsBench run:

```bash
.venvs/runner/bin/dycon skillsbench audit-rewards RUN_ID --output OUTPUT_DIR
```

The command uses the same current task × repetition projection as `run status`. It scans all current samples whose disposition is exactly `accepted` and whose finite numeric `raw_reward.reward < 1`; it includes partial rewards and does not coerce missing, boolean, string, NaN, or infinite values to zero. The run must be explicitly named and bound to the frozen SkillsBench benchmark. `OUTPUT_DIR` must be new; the command never overwrites a report or writes into run attempts, evaluation receipts, status, or configuration.

A report directory contains:

- `summary.json`: policy version, scope, source hashes, candidate and processing counts, classification totals, task/rule summaries, and completeness.
- `samples.jsonl`: one detailed record per candidate, including the selected physical attempt, classification, primary rule, reason, evidence hashes, missing evidence, and source-change flags.
- `samples.csv`: tabular form of the candidate records.
- `report.md`: human-readable run and task summary plus VerifierError and Review records.

The policy is `skillsbench_verifier_audit_v2`, shared with online settlement. `VerifierError` means the evidence proves an unrecovered required verifier failure; `ValidOutcome` means the bound contract and scoring evidence explain the below-one reward; `Review` means evidence, contract ownership, or ordering is insufficient. Findings are normal audit results and do not make the command fail. A source change or unreadable candidate is recorded as Review and makes the audit incomplete. See [SkillsBench verifier audit](#zero-reward-复查) for online settlement, compatibility, and evidence limitations.

### AgentOps 多解释器 verifier

`agentops-py38-py312-v1` 明确将 AgentOps 必测范围调整为 `py38,py39,py310,py311,py312`。Python 3.7 不参与此变体评分。这改变原六环境评分契约，不等同于恢复旧任务；原始任务包、历史 subject 的 `tox.ini` 和 `requires-python` 不被改写。

通过既有发布入口生成新包与按 package hash 注册的新契约：

```bash
.venvs/runner/bin/dycon skillsbench verifier-variant NEW_SNAPSHOT \
  --source-snapshot catalog/snapshots/skillsbench-verifier-repair-candidate-v2-20261110/manifest.json \
  --policy agentops-py38-py312-v1
```

新策略只修订 AgentOps，其他任务保持输入快照身份。AgentOps 的修订直接绑定旧修复回执中的原始任务，允许正式历史重评入口验证来源和镜像身份。发布规则另存于 `catalog/verifier_patches/agentops-py38-py312-v1.json`；原 `repairs.json` 不变。规则按 SHA256 存档，已发布任务不依赖可变候选文件。

新任务包的 `verifier/agentops_matrix.py` 与运行时 `runtime/agents/agentops_matrix.py` 使用同一实现。校验五环境的解释器版本、依赖与本项目安装阶段、实际 `coverage run --source agentops -m pytest` 命令、测试结束摘要、退出码和成功时的 `coverage report -m`。正常断言失败允许 tox 按原规则停止该环境后续 coverage 命令，仍要求所有五环境实际执行。未固定测试数量；整个环境跳过或只有 skipped 结果不算完成。

受保护 CI 包装指定五环境并禁止跳过缺失解释器，PATH 使用快照内 CPython 3.8.20、3.9.25、3.10.22、3.11.13、3.12.15。wheel、requirements hash lock 与 constraints 复用已验证的五套闭包，无 py37 兼容移植。环境探针通过独立 run 在 Docker 无外网环境从空缓存重新安装。

共享分类器仅为注册了 `agentops_matrix_v1` 的包启用新校验。执行器下载的 tox JSON、CI 阶段回执和构建退出码必须与 call、ownership、任务 package、verifier 来源和 dependency snapshot 绑定。必需阶段故障、报告缺失或损坏不能自动成为有效零分。原包继续使用原契约与 py37 要求。

控制验证继续使用正式入口：

```bash
.venvs/runner/bin/dycon skillsbench verifier-control CONTROL_RUN --case agentops-correct
.venvs/runner/bin/dycon skillsbench verifier-control CONTROL_RUN --case agentops-wrong
.venvs/runner/bin/dycon skillsbench verifier-control CONTROL_RUN --case agentops-missing-dependency
.venvs/runner/bin/dycon skillsbench verifier-control CONTROL_RUN --case agentops-missing-interpreter
```

同一 run 的控制必须顺序执行。长操作使用 `skillsbench job-start`，失败使用原入口 `--retry-failed` 保存新 attempt。正确控制冻结 BugSwarm failed/passed SHA 与归档补丁文件 SHA256；归档中的 logo 二进制元数据没有可应用 diff，单独记录，不更改测试。错误控制保留原始未修复源码。缺依赖和缺解释器仅在独立控制容器的测试输入中注入；历史 subject 不做此类修改。

验证专用 `network=none` profile 无联网工具资格，关闭 capture 后用于 verifier 和重评，不运行 rollout。不得绕过资格校验或复制 qualification 字段。历史重评仅写新 run 的 `validation/regrade-results`。

#### 已发布的 rollout 配套引用

2026-11-11，用户在独立 verifier/重评验收后明确要求完成 rollout 发布。通过正式 `skillsbench profile` 和 `skillsbench qualify` 创建 public qualified profile，重新核验历史真实 probe 的 7 条 request、5 组工具调用/结果，以及完整 agent 依赖派生链；未再次请求模型。

| 对象 | 引用 | SHA256 |
|---|---|---|
| Task snapshot | `catalog/snapshots/skillsbench-agentops-py38-py312-candidate-v2-20261111/manifest.json` | `c1a2618a25955bb03b645a43fb4367d71efe504ca82cb897a89d65bc8ba36119` |
| Public qualified profile | `runtime/agent-profiles/skillsbench-agentops-matrix-public-qualified-20261111/manifest.json` | `0c173a259a6a75ee63f9490a93a8891a8bc984b254d0dd95c671e29fc1edc71b` |
| Dependency manifest | `/data3/wangtianyi/DyConGlobal/runtime/dependency-snapshots/skillsbench-agentops-py38-py312-r3-20261111/manifest.json` | `3096763812f86a3db31b64cd5b9cbf0ef7e9e4d1d0170d5436f1a882c509af77` |

根 `config.yaml` 的 `benchmark.snapshot_ref` / `agent.profile_ref` 发布为上表配套引用。普通新 run 直接冻结它们；已有 run 和从旧 run 克隆的 run 继续使用其原冻结身份，不会自动升级。`candidate-v2` 是不可变发布 ID，不能为改变名称而修改目录内容。

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout --repeat-count 1
.venvs/runner/bin/dycon skillsbench preflight NEW_RUN --workers 1
```

独立验收 run `skillsbench-agentops-public-release-preflight-20261111` 在 capture/export 开启、`long_checks=true` 下通过完整 task/image/dependency 和实际环境安装审计。public profile 的正确控制 run 为 `skillsbench-agentops-public-release-control-20261111`，五环境全部通过、reward 1、cleanup confirmed。另有 none profile 的正确/错误/缺依赖/缺解释器控制、222 项回归和四条历史重评证据。

日常 `long_checks=false` 的快速 preflight 不等同于完整环境验收；完整发布证据与回滚引用见 `temp/skillsbench-agentops-evaluator-revision-20261111/public-rollout-publication.json`。回滚只恢复后续新 run 的两处旧引用，保留已发布文件和历史 run。

### Airport PDDL 变体与语义复评

`pddl-airport-plan-v1` 修正 `pddl-airport-planning` 的示例/解析器语法冲突。
它同时接受一行一个的标准 `(action arg1 arg2)` 与任务示例的
`action(arg1, arg2)`，仅做完整行语法转换，再交给原版本
`unified_planning==1.3.0` 的 PDDLReader 和 PlanValidator 检查动作、参数、
前置条件和最终目标。不求解、不执行提交代码、不降低语义通过条件。
支持 PDDL 注释和空白行；空计划也必须由语义校验确认目标已满足。

修复去除无效计划诊断对未定义 `plan` 的引用，并显式要求 `VALID` 状态。
1.3.0 的原 `if val` 已正确重载布尔转换，不能将其描述为发现的误通过漏洞。
原 verifier/test.sh、依赖 pin、二元 reward 和 task.md 均保留。

#### 发布和选择

```bash
.venvs/runner/bin/dycon skillsbench verifier-variant NEW_SNAPSHOT \
  --source-snapshot catalog/snapshots/SOURCE/manifest.json
```

此入口复用任务快照、package manifest、verify_package 和 task_package_path。
只允许源 verifier SHA256
`1f8c6cd4be1ed7fdf0925494643dd2d78848c9ec4c855a7a7837612675ac87e4`，
修复内容来自 `catalog/verifier_patches/pddl_airport_test_outputs.py`。
新快照保留其他任务及原索引策略，增加修复来源/收据，只有 airport verifier 的
一个文件可变。原任务包、源快照、旧 run 和 profile 不变。
新发布物已建立，不自动切换根配置或活动 run；后续新 run 需显式选择新 snapshot_ref。
修复模板继续作为校验身份的一部分，修改模板需要新的显式修复版本。

#### 历史提交的单独语义复评

完整 workspace replay 尚无通用保证；`harness run` 仍不支持将文件导出冒充完整状态。
因此新增的正式入口明确只验证 airport 计划文本的语义，不声称完整容器重评：

```bash
.venvs/runner/bin/dycon skillsbench revalidate-pddl RUN_ID \
  --snapshot catalog/snapshots/NEW_SNAPSHOT/manifest.json \
  --samples temp/TASK/selection.json \
  --output analysis/NEW_REVALIDATION \
  --validator-python .venvs/pddl-verifier-1.3.0/bin/python --timeout 60
```

selection.json 显式包含 run_id、task_id、samples；每条 sample 需 sample_key、
repetition_id、attempt_id、archive_sha256。入口使用冻结 run 的统一计划与当前 sample
投影，拒绝身份不符、重复样本、归档哈希不符或提交 PDDL 输入与冻结任务不一致。
复用受限只读归档接口，不提取/执行归档代码；仅将两份计划文本写到新分析目录。
worker 在显式项目隔离环境中加载可信 verifier，固定 unified_planning==1.3.0，
记录完整依赖版本，并受逐样本超时约束。

results.jsonl/CSV、summary.json、每条 request/result/record/log 以及前后来源哈希
单独保存；输出目录必须全新并位于项目 analysis/。失败和超时单列 RevalidationError，
原 reward 保留在 original_reward，revalidated_reward 仅指此次语义检查的二元结果。
不调用 planner/oracle、模型或容器，不更新历史 accepted/resolved/reward/capture。
旧审计 Review/Unresolved 与新语义结果属于不同评测契约，应分别保留。

2026-11-10 已发布 `skillsbench-agent75-pddl-plan-v1-20261110`（75 tasks，1 个修复 task）。
目标历史 run 的 14 份提交共 28 份计划均 VALID；输入哈希匹配，来源前后无变化。
结果位于 `analysis/pddl-airport-revalidation-20261110-01/`。
这证明所存计划在冻结任务输入下有效；不将该结果冒充原历史评测成功或全容器重放。

### Airport PDDL 完整评分重跑

仅适用于 `pddl-airport-planning`，要求任务修复版为 `pddl-airport-plan-v1`。
通用 SkillsBench workspace replay 仍不受支持；该任务的 verifier 仅依赖冻结
PDDL domain/problem 和提交的两份纯文本计划，因此可完整恢复其评测输入。

```bash
.venvs/runner/bin/dycon skillsbench regrade-pddl RUN_ID REGRADE_ID \
  --snapshot catalog/snapshots/CORRECTED/manifest.json \
  --samples temp/TASK/samples.json --workers 2 \
  --replace-original-with-backup
```

缺少最后一项时仅执行重评，不改写原回执；`--apply-only` 只应用同 REGRADE_ID 下
已经全部成功的完整 verifier 结果，并核对 snapshot 和 samples 哈希。
REGRADE_ID 不可重用；失败后需要新的 ID。workers 范围 1–4。

正式入口复用 load_plan、当前 sample 投影、受限归档读取、任务包/修复收据校验、
SkillsBenchFactory、NAS 镜像租赁、OwnedPlanes、BenchFlow 生命周期、hardening、
原 test.sh 和既有 verifier 证据采集。不会启动模型或执行 oracle solve；
`agent=oracle` 仅作为 BenchFlow 无模型的环境准备模式，显式调用
setup/start/install_agent/verify/cleanup，不调用 run。

每条任务使用独立带 project/run/task/sample/attempt/epoch 标签的容器。
只恢复两份原提交计划；提交归档的 problem.json/domain/problem 必须和冻结任务一致，
容器内计划及输入也须逐项 SHA256 匹配。运行完整原 test.sh（含 apt/uv 安装和 pytest），
要求两个 CTRF 测试全部通过、两份计划均打印 VALID、reward=1、无 verifier/agent 异常，
且清理回执成功、无该 epoch 残留容器。任一失败，整批不替换。

结果保存于 `runs/RUN_ID/regrades/REGRADE_ID/`。在容器启动前保存 before-replay
原始回执；真正替换前再按文件保存逐字节 backup 和 replacement-manifest，
记录 before/after SHA256。替换异常时回滚已经改写的文件。
replacement-state.json 标记 prepared/applied/rolled_back。

原地替换需用户明确授权；它是普通历史 attempt 不可变规则的显式维护例外。
命令取得 run/supervisor/validation 锁并要求退出与 cleanup_complete 回执。
替换范围是指定物理 attempt 的 evaluation-raw、evaluation-report、attempt.json，
repetition/result.json、对应 artifacts_committed/settled 及证据索引/progress。
同步 reward、resolved、回执哈希、来源说明和 finalization receipt hash，保留原 evaluation ID。
新 verifier 的任务包身份单独记录，不改写 run 的冻结任务/配置或提交/轨迹/capture。
status-summary 通过原刷新接口更新。

旧 verifier 日志作为旧运行事实保留；原地更新后的评测回执通过 regrading 元数据
明确指向新完整容器执行日志和结果，不能再把旧日志当成新评测日志。
历史审计报告、已有导出/数据集仍描述各自原始快照，不随该维护操作自动更新。
需要从新结果创建数据集时应重新使用正式来源冻结入口。

`backup` 文件可按 replacement-manifest 定位，均保留原始字节。
硬中断后先核对 replacement-state、当前文件哈希和容器归属，再决定维护恢复，
不能无条件重新执行替换或覆盖已有 backup。

### 历史提交复评

`skillsbench regrade-submissions` 从指定只读审计报告中选择 VerifierError/Review，以独立冻结 run 的新任务变体、profile 和 dependency snapshot，在真实任务容器中运行原 verifier。它不请求模型、不运行 oracle，不修改源 run 的奖励、投影或结算。

```bash
dycon run create TARGET --operation rollout --from-run VERIFIED_DEFAULT_ENTRY
dycon skillsbench regrade-submissions TARGET --source-run SOURCE \
  --audit-report temp/skillsbench-verifier-repair-20261110/historical-audit-final/samples.jsonl \
  --output runs/TARGET/validation/regrade-plan --plan-only
dycon skillsbench job-start --seconds 21600 JOB -- \
  regrade-submissions TARGET --source-run SOURCE --audit-report AUDIT \
  --output runs/TARGET/validation/regrade-results --workers 3
dycon skillsbench job-status JOB
```

目标与源必须为不同冻结 run。目标任务变体的来源 package hash、原镜像、subject 及归档字节、当前历史 sample/attempt 投影都必须匹配。输出必须为目标 run 内新的 validation 子目录。并发上限为 4，沿用现有 sandbox 与 verifier 的预算和清理接口，不执行 `rollout.run()` 或 `solve()`。

历史 subject 的 `re_evaluation=false` 仍然保留：文件导出不是完整 live state。本接口只重放逐项认证的声明提交，并明确记录这个范围。目录恢复包含有界归档扫描、路径与链接检查、恢复前后文件哈希核对，不以初始镜像文件补齐未导出内容。归档内 `/workspace/skills -> /skills` 只引用原只读技能挂载。其余链接只能落在声明输出目录内。

syzkaller 历史导出的是空 `/app`，而 verifier 消费 `/opt/syzkaller`，因此返回 `RegradeUnavailable`。SimPO 历史未导出源码；重评容器移除初始 SimPO 源码，禁止使用初始代码冒充历史提交。若新 verifier 在缺失/错误输出的前置 gate 停止，可得到有完整证据的有效零分；若它进入必须重跑源码的阶段，该样例返回 `RegradeUnavailable`，新分数为空。

`ValidOutcome` 包括有效零分和部分分。`VerifierError` 表示新运行仍遭遇评测故障，`Review` 表示共享分类器尚无充分证据，`RegradeExecutionError` 表示重放/清理过程失败，不能当作 agent 零分。`complete=true` 只表示全部选中样例均已处理且历史源文件未变，不表示全部可重放或 verifier 全部通过。

目录内保存计划、逐样例恢复清单、原 subject 哈希、verifier stdout/CTRF/阶段日志、共享分类 Decision、结果 JSONL、历史保存证明及汇总。再次执行必须使用新目录，旧结果不覆盖。

#### 显式覆盖历史评分

`regrade-submissions ... --replace-original-with-backup` 在全部样例处理完成、原始来源哈希未变且源 run 已退出并清理后，对 ValidOutcome（含有效零分和部分分）自动回写 canonical 评分。沿用既有事务备份/回滚接口，保存 replacement-manifest.json、replacement-state.json、backup/ 和逐样例 replacement-summary.json，同步 evaluation raw/report、attempt、repetition result、结算回执、evidence 和 progress，并刷新 status-summary。原 subject、轨迹、capture、原 verifier 运行日志及旧独立审计报告保留；新评分明确绑定新任务包、依赖、冻结 run、恢复范围及评测证据。

VerifierError、Review、RegradeUnavailable 和重放/清理失败只记录本次重评结果，不以无效数值覆盖旧奖励；replacement-summary.json 列明所有未覆盖样例。未传该选项的默认行为仍为只读历史来源。`--plan-only` 只检查覆盖前提，不写历史结果。


#### Manufacturing 历史提交重评

支持 `manufacturing-equipment-maintenance` 的五个声明输出 `/app/output/q01.json` 至 `q05.json`。沿用原 subject 身份、大小与 SHA256 校验和统一容器恢复接口；在冻结任务镜像中保留原 `/app/data` 输入，恢复提交后执行官方 verifier，不请求模型或执行 oracle。

2026-11-15 的 `skillsbench-mfg-regrade-20261115/validation/manufacturing-rep008-results` 已对 baseline 的 rep-008 完成独立重评：7 项测试中 4 项通过、3 项失败，分类 ValidOutcome、reward=0；源记录保持不变，容器清理完成。

## Terminal-Bench 2.1

### 执行环境、资源准备和验证

正式入口仍为 `dycon harbor` 和 `dycon run`。固定 benchmark 为
`terminalbench_2_1`，任务 revision 为
`7131e4375048a0e408a8fb404b5f499d726b695b`；Harbor 为 0.25.0 /
`5eb713ff9500daced4e143b98765a7c732744a1b`，OpenHands CLI / SDK / tools
保持现有固定版本。

#### 显式执行策略

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

#### 配置和资源准备

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

##### 任务依赖镜像与 wheelhouse

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

#### 验收和证据

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

#### Canonical 工具反馈的超时边界

共享 OpenHands 适配器从每次 agent-decision 的有效请求前缀提取已选工具调用的反馈；
即使该请求随后超时、取消或失败而没有 selected response，已收到的反馈仍进入
canonical 消息。按工具调用 ID 去重，未选分支及 condenser 请求不提供工具反馈。
内容保留 SDK 请求中的原文，raw ACP 事件也继续保留；不会将 ACP 进度消息推断为完成结果。
若超时发生时尚无包含该反馈的后续请求，反馈仍仅保留在 raw ACP 事件中。
此修复对新导出生效，不改写历史 run 的 canonical 文件、哈希或评测结果。


#### Verifier 健康判定与历史控制组复核

rollout 与 oracle/nop 共用 `harbor-verifier-health-v2` 判定。有限的 sub-one reward
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


#### 统一下载缓存

Terminal-Bench 的 verifier 会直接执行官方的 `curl`、`uvx`、Git 和 APT 命令，不能通过修改 verifier 来适配下载。`harbor resource-cache` 管理宿主机上的内容寻址 HTTP 缓存；公开网络任务容器通过只读 CA bundle 和代理环境变量接入，缓存私钥不会挂载进容器。缓存服务只缓存 GET/HEAD 和受限的 Git 下载请求，响应完整性由 SHA-256、大小和不可变索引校验。

典型流程是 `runtime-prepare`、`scan`、`prepare`、`start`、`prefetch`、`seal`、`publish`，然后以 `--mode replay` 启动封存副本。回放模式对未命中请求直接返回错误并记录 `denied`，不会偷偷连接上游。每个 run 的配置记录缓存 descriptor、CA 哈希和封存哈希；任务容器中的原始 verifier 命令保持不变。

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


#### 统一任务环境修复与版本化参考解

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

可用 `dycon run create NEW_RUN --operation rollout --from-run SOURCE_RUN` 复用已核验环境的冻结配置。普通模型 rollout 使用原说明、输入和 verifier。`--reference-ref` 仅用于显式选择修订 oracle 的控制组。

### Terminal-Bench verifier 健康标记

`dycon harbor controls-audit RUN_ID --control-id CONTROL_ID --audit-id AUDIT_ID`
reclassifies saved Terminal-Bench oracle/nop evidence through the same verifier health
policy used by live rollout. It writes a new immutable audit receipt under the run;
original control results and frozen configuration remain intact. See
[Terminal-Bench](#执行环境资源准备和验证) for error, valid negative, and Review semantics.
