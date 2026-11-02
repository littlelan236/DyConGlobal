# DyConGlobal 启动预检流程、结果复用与冗余缩减审查报告

审查日期：2026-11-01（Asia/Shanghai）。  
远程项目：/home/wangtianyi/DyConGlobal。  
代码基线：HEAD f12b8e06bc482826cb7e20261f552c275c6077e8，结合审查时工作树。  
范围：正式 rollout 的 run create/start/resume/retry 链路；以用户描述直接对应的 SkillsBench/OpenHands 为重点，比较 Native SWE、WebArena、Harbor。harness 评分、离线 hidden 提取与训练不计入本报告的启动次数。

本次仅阅读文档、代码、注册清单及少量历史 receipt，生成本报告；没有启动 run、模型服务、任务容器、全量 hash benchmark，没有改动配置或执行代码。下文“次数”来自当前代码调用路径，“体积”来自已登记清单；不把理论扫描量当作实测物理磁盘流量或耗时。

## 1. 核心结论

1. **SkillsBench 不只是启动前两次模型核验，还会在每个正式 physical attempt 再全量核验一次。** 正常一次启动进入执行时，runner 和 SkillsBench preflight 各调用一次 verify_model；每个执行到该检查点的 attempt 再调用一次。一个新的 75 task × 3 repetition run，若全部正常进入检查、无重试且只启动一次，调用次数为 **2 + 225 = 227 次**。提前失败、重复启动和重试会改变实际次数。这是最高优先级的冗余。
2. **模型注册清单不是核验结果缓存。** 每次 verify_model 都重新读取清单中的权重、配置、tokenizer 等文件计算 SHA-256。页面缓存可能减少物理 I/O，但不免除完整数据读取和 hash 计算。模型服务自身加载权重是另一个用途，不能与校验读取混算或直接取消。
3. **SkillsBench 已有同 run 环境审计缓存，但没有跨 run 复用。** 合同和证据未变时，预检可跳过真实 sandbox 重建；新 run、整体配置 hash 或相关源码文件变化会产生新合同，重新审计。即使只改实验算法，仍可能使全部 task 环境证明失效。
4. **环境 receipt 命中不等于无重型扫描。** 外层 factory.lock_images 与 preflight 的 image_reference 都会进行 NAS OCI blob 深度校验；依赖快照也会逐文件重算 hash。它们位于环境 receipt 命中判断之外。当前还可能在每个新 sandbox 继续重复依赖快照扫描。
5. **不同 backend 不能混为一谈。** 当前 Native SWE 路径的 runner 有一次模型核验；WebArena 按站点组合进行两轮 reset 验收；Harbor 对每个 task 做真实无模型环境 probe。SkillsBench/Harbor 的 adapter.startup 均为 no-op，不能把它们描述成每次启动必有额外真实 LLM smoke。
6. **建议保留动态准入和首次内容核验，合并同一 execution 中重复的静态核验，再建设跨 run 的环境证明复用。** 不建议用“全跳过预检”开关代替这些改进。

## 2. 当前正式流程与各环节必要性

### 2.1 总体调用链

~~~text
run create / frozen clone
  -> strict config、backend/profile、任务选择与计划冻结
run start / run resume
  -> supervisor：execution 身份、互斥、进程生命周期与清理边界
  -> rollout.runner.run_manifest
       -> load_plan / assert_frozen
       -> 恢复时验证已结算样本；全部已结算则提前返回
       -> 必要的 capture publication recovery（resume）
       -> verify_model                         [SkillsBench 第 1 次]
       -> adapter.open_run / factory
       -> adapter.preflight(selection.tasks)
            -> SkillsBench factory.lock_images [镜像校验、lease，缺失时加载]
            -> SkillsBench preflight
                 -> verify_model               [第 2 次]
                 -> profile/版本/GPU/daemon/task/image/dependency 检查
                 -> audit_task_environments    [缓存命中或真实逐 task 审计]
       -> ServingManager / ServingPool
       -> adapter.startup                      [依 backend 不同]
       -> --check-only 在这里结束，否则调度正式 attempts
            -> SkillsBench _execute_attempt
                 -> verify_package
                 -> verify_model               [每个 attempt 再 1 次]
                 -> 正式 sandbox 安装与 agent 执行
~~~

run retry-failed 注册新物理 attempt 后仍经 resume 链路；不会因此绕开上述检查。这里的“同一次 execution”指一次 supervisor/runner 生命周期，不等同于 run 的整个可恢复历史。

| 环节 | 检查什么、保护什么 | 必要性与缩减判断 |
|---|---|---|
| create 时配置、task、profile/context 冻结 | strict 配置、backend 能力、任务/模型/算法/资源关联，写入可复现快照与 hash | 必要；创建与执行边界不同，不能只因两边都检查配置就删除执行端身份检查。 |
| supervisor、run lock、execution 身份 | 防止同 run 双启动，管理本 execution 的进程/取消/清理 | 必要的当前状态检查，不应复用旧“通过”记录。 |
| load_plan、assert_frozen、恢复结果核对 | config/plan/selection/tasks/work-items 一致；已结算样本绑定正确证据 | 必要。存在多层加载，但其内容通常比模型/OCI blob 小，应先度量再优化；跨进程边界保留独立校验。 |
| verify_model | 校验 identity JSON 的自洽性，核对登记文件存在、大小、SHA-256 | 首次启动前内容核验有价值；同输入、同 execution 内重复全量 hash 是主要可缩减项。 |
| backend/profile/版本/task package | 能力矩阵、固定依赖版本、任务包文件清单与输出路径，防止用错 profile/任务 | 保留身份绑定，可按唯一内容身份去重；不是全量 checkpoint 检查。 |
| NAS image 身份及 blob 校验 | 清单来源与 digest，OCI graph/层内容；lease 管理加载及资源所有权 | 保留内容完整性和 lease；重复读取相同 blob 可按内容身份合并。SkillsBench 的 lock_images 还会实际加载缺失镜像，不只是轻量“锁定”。 |
| SkillsBench 全 task 环境审计 | 真 sandbox 中安装 agent runtime、非 root Python/terminal/ACP、skill visibility、compose image 与清理 | 未证明过的环境必须验证；已有精确证明可复用。环境 readiness 不是模型推理/capture 正确性证明。 |
| serving runtime/GPU/endpoint 准入 | 固定 vLLM/EasySteer overlay、可用设备、端口、启动模型进程、GET /v1/models | 动态部分每次保留；相同解释器/overlay 的静态源码检查可在同 execution 去重。服务冷启动本身不是可删除的预检。 |
| Native adapter 的真实 model/tool/capture smoke | 模型生成工具调用，真实容器执行；启用 capture 时检查发布、token 坐标、hidden 等 | 与无模型环境审计职责不同；新服务实例的最小 smoke 应保留。不能推论所有 backend 都执行它。 |
| 每个正式 attempt 的环境安装/probe | 建立新的隔离任务 sandbox，安装任务所需 runtime 并验证实际环境 | 新容器确实需要运行环境；复用资格证明不会自动让新容器已经安装。可通过已审核派生镜像等长期方案减少安装，但须沿现有发布/owner 接口实施。 |

### 2.2 backend 差异

| backend | 正常启动路径中的环境 gate | 历史 receipt 是否使该 gate 自动跳过 | adapter.startup |
|---|---|---|---|
| Native SWE / mini-swe-agent | 锁定任务 NAS image 身份；随后用首个任务环境做真实模型工具链 smoke | 未见完整 preflight 成功记录缓存；镜像已在 daemon 中可避免再次导入 | 默认真实工具链 smoke；TP=1 每个服务 worker 各做。 |
| Native SWE / OpenHands | 上述 image 身份，加 profile/request-role 检查 | 写 receipt，但未见据此跳过后续 preflight 的读取分支 | 当前继承默认 startup；正式 task 执行另经 OpenHands bridge。 |
| SkillsBench / BenchFlow / OpenHands | 所选全部 task 的 package、image、dependency、环境审计；审计按 task 而非 repetition | **有：同 run、同合同、同证据时跳过 sandbox 审计。** 但上游 model/image/dependency 等检查仍执行 | no-op；无模型 ACP/tool 初始化已在环境审计覆盖，GET /models 不代表完成真实模型/capture验收。 |
| Native WebArena | 每个不同 sites 组合选择一个 task，连续 2 个 fresh reset cycle，检查真实 AX observation、scroll、HAR、资源隔离与清理 | 每次生成新 UUID 证据目录，未见成功 receipt 读取复用分支 | 另做真实模型驱动的 browser scroll smoke。两轮 reset 目的是验证隔离，不能简单认定第二轮完全重复。 |
| Harbor / OpenHands | 每个所选 task 执行无模型 probe，核对 ACP/tool 初始化、异常与相应网络策略 | 每个 task 使用新 UUID 目录，未见成功结果缓存分支 | no-op；模型通道由 shared bridge 管理。 |

WebArena 部分按当前工作树读取；该文件已有用户未提交修改，后续行号与细节可能随工作变化。本次没有修改这些文件。

## 3. 模型身份核验：重复次数与实际扫描规模

### 3.1 确定调用点

| 位置 | 调用条件与作用 |
|---|---|
| `rollout/runner.py:663` | 未在“所有样本已结算”等分支提前返回时，全量 verify_model。 |
| `runtime/agents/skillsbench.py:599` | SkillsBench preflight 内再次全量 verify_model；独立 CLI preflight 也走这里。 |
| `runtime/agents/skillsbench.py:1029` | SkillsBench 每个 physical attempt 到达此位置时再次全量 verify_model；不只首个 task/repetition。 |
| `core/model_identity.py:30-50` | 读取登记清单后逐文件 sha256；没有 execution 级、进程级或持久化 hash 结果缓存。 |

单次正常 SkillsBench execution 的调用量为 `H_model = 2 + A`，A 为实际到达逐 attempt 检查点的 attempt 数；独立手动 preflight 的次数还要另外加。每次重新进入完整启动路径，再付启动两次。retry 只要产生新 attempt 并到达检查点，也会增加次数。全部已结算的 resume 在模型检查之前返回，不适用 2+A。

model-register 在登记时通过 describe 对权重与元数据生成清单；它不是每次 run start 必经步骤。vLLM 加载 safetensors 的日志也不是这几个 verify_model 调用本身。当前文件核验范围是已登记文件，不能仅凭该核验宣称目录内新增的任意文件都已重新盘点。

### 3.2 现有冻结 run 的清单体积（元数据读取，未重新 hash 权重）

| 冻结 run | 登记文件 / 权重分片 | 登记文件总字节数 | 仅启动两次所需逻辑读取 |
|---|---:|---:|---:|
| skillsbench-agent75-r3-qwen36-27b-positive-gpu23-tp1-20261031 | 26 / 15 | 55,586,032,434（约 55.59 GB） | 约 111.17 GB |
| skillsbench-agent75-r3-qwen36-35b-a3b-positive-gpu45-tp1-20261031 | 37 / 26 | 71,926,788,362（约 71.93 GB） | 约 143.85 GB |

GB 采用十进制。若以上各自 75×3 无重试完成，227 次核验的理论读取量就是“登记总字节数 × 227”；这是当前调用路径的规模推导，不代表这些历史 run 已经按当前代码实际发生相同读取。多 worker 同时重复 hash 还可能争用 CPU、页面缓存与存储带宽。

## 4. SkillsBench 容器环境审计与复用机制

### 4.1 cache miss 时执行的功能

audit_task_environments 对冻结 selection 的每个 task 调用 _audit_task_environment。独立命令 workers 可取 1–8；正式 adapter 使用 min(8, execution.rollout_workers)，不是直接按总 task×repetition 数审计。

一次 cache miss：创建该 task 的 SkillsBenchFactory，锁定/lease 镜像，经 BenchFlow Rollout.create → setup → start → install_agent，检查 dependency-install.json、skills-visibility.json、agent-environment.json、compose-image-verification.json。probe_runtime / sandbox_probe 在真实非 root 用户下执行 terminal，并通过真实 ACP 子进程 initialize、session/new 验证初始化；LLM 指向不可用的本地探测地址，不发生成请求。完成后要求 sandbox-cleanup.json 存在且该 epoch 无残留容器；清理失败将 verified 置为 false。此过程不运行 oracle/verifier。

完整资格证明有价值，但独立审计建出的 sandbox 随即清理，正式 attempt 之后还会创建自己的 sandbox。前一次证明的是该环境构建路径可用，不会保留一份已安装的正式容器供任务直接继续使用。

### 4.2 缓存键、命中条件与失效

代码位置：`runtime/agents/skillsbench.py:458-566`。

环境合同包含：schema/policy、run_id、完整 config_hash、execution_context.context_sha256、model_fingerprint、task key/package hash、task images、dependency snapshot hash；另外 hash 四个完整源码文件：skillsbench.py、dependencies.py、sandbox_probe.py、openhands.py。

布局为：

~~~text
runs/<run>/validation/environments/<task>/<hash(contract)>/
  latest.json
  attempts/<environment-attempt>/result.json
  attempts/<environment-attempt>/<各项环境证据>.json
~~~

复用条件是 latest 指向路径位于合同目录内、目标文件 hash 与指针一致；读取结果后，contract 必须完全相同、verified 为真，并对 evidence_sha256 列出的每份证据确认文件存在和 hash 一致。命中则返回旧结果，**跳过 sandbox 重建**。

失败的分类要区分：合同改变、没有成功结果、证据缺失/不匹配等会导致重新审计；latest 指针越界或指针校验失败则直接抛错，不会默默重跑掩盖证据损坏。result 带 receipt_sha256，但这条复用路径显式检查的是 latest 的文件 hash 与各 evidence hash；不能称其为外部署名或独立可信签名。

| 情形 | 当前实际行为 |
|---|---|
| 同 run 先独立 preflight，再 start/resume | 输入与相关源码未变，可复用每个 task 的环境审计；上游完整预检仍执行。 |
| 同 task 多个 repetition | 环境审计按 task 一次覆盖；正式每个 attempt 仍有独立安装/执行。 |
| 新 run / clone 相同环境 | run_id 已改变且目录隔离，不复用旧 run 的环境审计。 |
| 同 run 修改相关文件中任意代码 | 完整源码 hash 改变产生新合同；即使修改的是文件里与环境无关逻辑，也可能使全部 task 重审。 |
| 新 run 只改模型/共识算法/生成预算/GPU placement 等 | 运行配置/上下文改变会影响合同，当前没有仅针对“环境实际输入”的细粒度共享键。部分参数例如 workspace、terminal timeout 确实影响 probe，不可一概移除。 |
| Docker daemon、host 重启或平台能力变化 | preflight 会现查 daemon/GPU；但当前环境合同没有显式独立 host/daemon epoch/TTL 字段，不能把旧环境 receipt 当成无限期跨主机通用证明。 |

历史抽样仅检查 summary/目录元数据：27B run 的环境 summary 记录 task_count=75、verified=true（2026-11-01 07:25:06 +08:00），但存在 225 个 task/contract latest 指针；35B-A3B run 对应 75 个。225 个目录是多个合同下的证据，不是 225 个不同 task，也不能仅据此断定由 repetition 或同一根因造成。本次没有重新执行这些审计或逐份完整验证历史 receipt。

## 5. 其他重复扫描与当前缓存的边界

### 5.1 镜像：环境缓存命中后仍有深度验证

`runtime/agents/skillsbench.py:28-43` 的 image_reference 每次调用 image.verify(deep=True)。NasImageStore.verify 进入 descriptor_graph；deep 会读取并 SHA-256 校验 OCI graph 的 layer/config/manifest blob，不是只比对一个短 digest。

| 调用位置 | 扫描与复用现状 |
|---|---|
| BenchFlowAdapter.preflight → super.preflight → SkillsBenchFactory.lock_images | 遍历所有选中 task 的 image units；每个都调用 image_reference、获取 lease，缺镜像还会 docker load。发生在环境 receipt 判断之前。 |
| skillsbench.preflight 的 task/image 循环（:630-640） | 再次 image_reference；即使随后环境审计全部 cache hit，也会重复深扫。 |
| 每个环境 audit miss 的临时 factory.lock_images | 再校验该 task 的镜像并 lease。不是因为 task 包改变才必然调用。 |
| 冷镜像导入 | NasImageStore.write_archive 还会 verify(deep=True)，随后流式读取 archive 给 docker load。后者是加载用途，不应简单当重复校验删除。 |

descriptor_graph 的 verified 集合可在一次调用中去重，且 API 可接收外部 verified 集合；当前 NasImageStore.verify 未传入跨调用集合，因此不能去掉多次调用间或不同 task 共享基础层的重复 hash。SkillsBenchFactory.started 能防同一 factory 再 lock，但不是全链路共同的 blob 验证缓存。

ImageLeaseManager 会复用 daemon 中已存在的 image，使用按 daemon ID/data root 分区的 journal/lease；这是“避免再次 docker load”和资源所有权保护，不是“NAS blob 已验证缓存”。cleanup=true 时，仅对 manager 登记 owned 且未被租用/使用的镜像尝试清理，预存在且未登记 owned 的镜像不自动纳入清理。因此独立 preflight 的 audit 如导入并释放了 owned 镜像，之后 start 可能重新冷导入；同 execution 中外层 lease 仍持有时，不能推论 audit 的清理一定让镜像被删除。

### 5.2 依赖快照、任务包、wheelhouse

verify_snapshot（`runtime/agents/dependencies.py:15-42`）核验 manifest 自身 hash，然后遍历全部登记文件、大小和 SHA-256，再 rglob 检查是否有未登记文件。SkillsBench preflight 调用一次；OwnedPlanes.create_environment 挂载依赖前再次调用；独立 offline acceptance 的 install_probe 也调用，但后者不是每次普通启动必经。

上述两个抽样 run 的同一依赖 publication 清单均为 **4,843 个文件、925,331,384 字节**。因此缓存命中前后的文件遍历、以及每个新 sandbox 中的相同 publication 重核，仍可能显著耗时。实际安装命令的 uv/pip require-hashes 又属于新环境安装的完整性边界，不能把宿主校验缓存直接等同为已安装合格。

verify_package 通过 package_manifest 重新读取 task 包清单；preflight 和正式 attempt 都使用。通用 SWE ContainerFactory 的 EnvironmentPolicy 初始化会 hash wheelhouse 中 *.whl；task policy 已可复用根 wheel_hashes 和 factory 内 task policy。上述均应按清单/内容身份去重，不能用全局“跳过依赖检查”替代。

### 5.3 serving 与 check-only

每个 ServingManager 在 managed 模式调用 verify_easysteer_runtime.py，检查 vLLM 版本/导入位置和相关 overlay 源码。per_gpu_tp1 的 pool 逐个创建 manager，因此相同解释器/overlay 会重复验证，服务启动也是依次发生。静态 runtime 核验可候选复用，GPU 准入、端口和每个服务 readiness 需要本次现查；进程启动、模型加载、编译预热应另计耗时。

runner 将 validation_mixed 设置为 check_only；默认 Native startup 中额外 mixed-batch/parallel validation 依该标志和具体配置触发，不能说每次普通 start 都执行完整并发验收。check-only 结束会退出 serving context；之后正常 start 重新建服务和走启动 gate，不继承一套仍存活的合格服务。SkillsBench 的 startup 是 no-op，也不能仅凭 check-only/GET /models 通过就宣称 grammar、真实生成和逐 step hidden capture 验收完成。

## 6. 冗余缩减建议（仅建议，尚未实施）

| 优先级 | 具体措施与边界 | 收益与约束 |
|---|---|---|
| P0 | 统一模型验证 owner：一次 execution 内首次做完整 verify_model，向 preflight/attempt 传递绑定 model 路径、fingerprint、文件身份、实现版本的验证上下文。同一输入重复调用复用验证结果，多 worker 用 single-flight 合并并发重复。 | 在受控不可变 checkpoint 前提下，正常 75×3 可由 227 次全量 hash 收敛为 1 次；仍有模型自身加载。**独立 skillsbench preflight 没有既有验证上下文时仍必须自行验证**，不能直接无条件删除其中的 verify_model。 |
| P0 | 以本次 execution 的“NAS store namespace + digest + 已验证文件身份”合并镜像深扫；让现有 NasImageStore/descriptor_graph 消费同一验证上下文，跨 task 对共享 blob 去重。 | 消除 factory 与 preflight 对相同 blob 的重复 hash；保留 graph/manifest 身份、lease、load、取消/清理 owner。不能仅凭 digest 字符串就信任可写 blob。 |
| P1 | 在同 execution 中集中验证依赖 publication 和 task packages，后续创建环境引用验证上下文并做必要的漂移检测；复用既有唯一接口。 | 减少 4,843 文件 publication 和相同 task 包的重复遍历。新容器安装与实际非 root/tool 初始化的必要部分仍保留。 |
| P1 | 将跨 run 环境资格证明与 run/attempt 证据分层：共享键绑定真正环境输入，run receipt 只引用证明并绑定本 run 身份。 | 相同 task 环境的新算法实验可复用资格证明，避免每个新 run 重建全部审计 sandbox；不能简单把 contract 中 run_id 删掉就算完成。 |
| P1 | 共享键纳入 task package/compose 每个 digest、dependency publication、skill mode、workspace/用户、安装/probe policy、实际影响环境的参数、相关实现及平台/daemon 能力。升级版本并明确 refresh/失效规则。 | 避免完整 config_hash 或整文件无关变更造成过度失效；又不漏掉 terminal timeout、network、ABI、CPU quota 等真实环境输入。新服务器/daemon/平台变化应重新资格验证。 |
| P2 | partial resume 针对本次仍需执行的 task 检查环境，而不是无条件遍历原 selection 全集。 | 当前 runner 虽跳过已结算样本，adapter.preflight 仍接收完整 selection；可减少无关 task 镜像与环境扫描。必须保留冻结 selection，另传本次必要子集，不改旧快照或原全量验收口径。 |
| P2 | 把“环境发布/首次资格验收”和“常规 run 的轻量接入检查”分开，同时为 WebArena/Harbor 设计适合其动态状态的证明复用。 | 避免每个 run 重做相同无模型全量环境实验；WebArena 两轮 reset 先保留为资格测试，本次网络/站点/浏览器最小 readiness 仍执行。Harbor 缓存也必须绑定 task/profile/platform，不能套用仅 SkillsBench 的合同。 |
| P2 | 对相同 serving interpreter/overlay 共享一次静态 runtime 核验，保留逐 GPU/endpoint 动态检查；避免把 check-only+start 作为每个已资格环境的固定操作。 | 减少每卡重复源码检查及不必要的双服务冷启动。不得把旧服务 receipt 作为当前存活/可用证明。 |
| 配套 | 输出每阶段 duration、cache hit/miss/invalidation reason、核验次数/文件数/逻辑字节数、唯一 blob 数、sandbox 审计与正式安装分别耗时。 | 先量化主要成本，再决定优化排序；本报告不编造“节省多少分钟”或加速百分比。 |

模型/镜像/依赖的 hash 复用需要明确不可变存储或受控修改锁。只看 size/mtime、只有“已通过”布尔值、或任意可传的 skip=True，都不足以维持原内容完整性保证。若无法保证运行期间不变，优先只合并紧邻的同输入重复校验，并保留受影响文件的重新内容校验；跨 execution 缓存应在不可变 publication/可信存储机制明确后实施。

无需改代码即可遵循的现有做法：同一冻结 run 的独立 preflight 通过后，正常 start 可复用环境 audit receipt；无需为了“再确认”反复手动预检。新环境/新依赖仍按现有全 task 资格要求执行，不绕开 gate。同一实验需要多次独立采样时使用一个 run 的 repeat-count 有利于共享 task 审计，但不能替代确实需要不同冻结配置的实验，也不会解决当前每 attempt 模型重 hash。

## 7. 建议实施顺序与验收标准

1. 先在现有入口记录分阶段耗时、hash 调用量与缓存命中原因；区分真实文件 hash、镜像导入、模型加载、sandbox 安装与模型生成。
2. 先做 P0 的 execution 内模型核验合并，再做相同镜像/依赖的内容身份去重。保持独立 preflight、正常 start、resume、retry 的统一入口行为。
3. 再建设版本化的跨 run 环境证明；用相同环境的新 run 验证 cache hit，用 task/image/dependency/policy/平台变化验证有针对性的失效。
4. 验收必须覆盖：模型/OCI blob/依赖内容变化仍被拒绝；旧 receipt/指针损坏明确报错；多个 worker 不重复执行同一全量校验；失败或取消不发布成功证明；实际 GPU/服务、sandbox 归属与 cleanup 每次仍可验证；不改写旧 run 配置和证据。

目标是“每个必要内容身份一次完整核验 + 每个新环境的必要动态检查”，不是降低实验复现和隔离要求。本次未执行性能对照测试，227→1 等是满足明确不可变前提后的设计目标，尚不是已实现结果。

## 8. 主要依据与审查边界

| 文件 / 位置 | 本报告采用的依据 |
|---|---|
| README.md、AGENTS.md、docs/ENTRYPOINTS.md、docs/MINIMUM-RUN-CN.md | 正式统一入口、冻结配置、四种 runtime 环境、启动/恢复/验收操作约定。 |
| docs/container-environment-repair-cn.md、docs/SKILLSBENCH-UNIFIED-STARTUP-REPAIR-PLAN-20261030.md | 容器环境职责及历史修复计划；计划要求与已落地代码分开判断。 |
| application/execution/supervisor.py:279-304；application/execution/run_plan.py:374-461；rollout/runner.py:589-727 | supervisor → runner，冻结验证、已结算提前返回、首次模型核验、backend preflight、serving 和 check-only。 |
| core/model_identity.py:8-56；runtime/agents/skillsbench.py:599、1029 | 注册/逐文件核验及 SkillsBench preflight、每 attempt 重复调用。 |
| runtime/agents/skillsbench.py:28-101、125-155、458-700 | 镜像深扫与 lease、依赖核验、环境合同/缓存及完整 preflight。 |
| runtime/agents/dependencies.py:15-42、253-322；runtime/agents/sandbox_probe.py | 依赖 publication 全量 hash、实际安装和真实非 root terminal/ACP 初始化。 |
| runtime/images/nas.py:58-70、200-319；runtime/images/oci.py:31-78 | OCI 深度核验、descriptor 内去重、导入/daemon image 复用与清理。 |
| runtime/backends/base.py:20-26；benchflow/adapter.py:11-22；native/swe_openhands.py:25-41 | 通用与 SkillsBench/SWE-OpenHands adapter 预检和 startup 继承行为。后三者均位于 runtime/backends/。 |
| runtime/backends/harbor/adapter.py:74-105；runtime/containers/webarena_adapter.py::preflight | Harbor 全 task 无模型 probe；当前 WebArena 工作树按 sites 组合双 reset 的实现。 |
| runtime/serving/manager.py:178-179、317-400；runtime/serving/pool.py:77-104；runtime/serving/startup.py:12-188 | 每 manager runtime 检查/服务启动、per-GPU 顺序启动、真实工具与 capture smoke。 |
| catalog/skillsbench.py:58-63；runtime/containers/policy.py:16-27；runtime/containers/factory.py:65-97、249-376 | task package 核验、wheelhouse hash、通用容器 bootstrap/probe。 |

历史抽样只证明当时保存的 metadata/目录现状；不证明当前运行状态或每份 receipt 的完整有效性。根 config 当前 backend 为 swebench，不能据其判断已冻结 SkillsBench run 的配置。源树审查前已有 runtime/containers/webarena_adapter.py、tests/test_webarena_browser.py 的未提交修改，本次保留；本次唯一远程写入为这份报告。未记录任何 SSH 密码或 API key。