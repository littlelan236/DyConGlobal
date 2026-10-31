# SkillsBench：MIG consensus 与离线依赖闭包验收

日期：2026-10-28（北京时间）。远程工程：/home/wangtianyi/DyConGlobal。
本次仅验收真实 parallel_consensus 和首批任务离线依赖闭包，不扩展为原始 plan 全量完成声明。

## 结论

| 项目 | 结果 | 范围 |
|---|---|---|
| 真实 MIG consensus | 完整结束，全 run audit 与 artifact verify 通过 | MIG UUID/TP=1、2 个分支决策/4 个候选；两次均为 exact-action 平票后 estimator 决胜 |
| agent 任务成绩 | 官方 reward=0，resolved=false | 不能声称解题成功或 consensus 提升任务效果 |
| 离线安装与官方评分 | 两个任务均通过，oracle/verifier reward=1.0 | dialogue-parser、offer-letter-generator；锁定依赖安装、无网络容器、官方评分脚本不变 |
| 清理 | 复核通过 | 相关模型/runner/supervisor/代理退出，无遗留所属容器与网络 |

## 1. 真实 consensus

- Run：skillsbench-consensus-mig7-offer-20261028。
- Rollout：r-875624e1301182430d24659ab08b278a；任务 offer-letter-generator，1 次 repetition。
- MIG：MIG-b13c6595-2246-526e-8a72-1c5f59a014e8；通过既有 serving.gpus MIG UUID 接口，TP=1，端口 8110，显式 KV cache=4 GiB。
- 使用新 run 冻结配置，不按 GPU 4–7 的物理编号进行多卡 TP，不修改旧 run。
- 模式 parallel_consensus，branch_count=2、dynamic admission、capacity=1、每 4 个 decision 分支一次。
- 使用真实发布 predictor：runtime/parallel-artifacts/skillsbench-swe-resolved-ridge-20261028/manifest.json。
- Run stage=complete，execution exit_code=0，结束时间 2026-10-28 13:07:12 +08:00。

| 审计指标 | 实测 |
|---|---:|
| Canonical decisions / unique native requests | 8 / 8 |
| Exact close points | 8，每点 32 层 × 4096 维 |
| 实际工具结果配对 | 7 |
| Parallel decisions / candidates | 2 / 4 |
| 产生不同候选的决策 | 2 |
| 全候选 completion token 成本 | 2350 |
| 独立校验的候选 native requests | 4 |
| published predictor 独立复算分数 | 4 |
| 最大在线/复算分数绝对误差 | 0 |
| 未执行的 losing tool calls | 2 |
| 未配对或 SDK 拒绝的所需工具调用 | 0 |

两次 exact-action 均为 1:1 平票，按冻结的 estimator_score tie policy 选择，分别选中 branch 0 和 branch 1。
本次证明真实 consensus 路径、候选区分、平票策略、native/cost/score 绑定和 winner-only 工具执行；**不证明多数票一致性已出现，也不证明解题成功或效果提升**。

复查命令：

    cd /home/wangtianyi/DyConGlobal
    .venvs/skillsbench/bin/dycon skillsbench audit skillsbench-consensus-mig7-offer-20261028 --require-parallel
    .venvs/skillsbench/bin/dycon artifact verify skillsbench-consensus-mig7-offer-20261028

未使用 --require-resolved，因为该次官方成绩真实为 0。
Audit receipt SHA-256：e626335da75145825c65ce2e3d79555c4565c7194cd703afd22ba09aca732903；whole_run_acceptance=true。
Artifact verify：16 artifacts、2 receipts、17 external metadata 校验通过。

## 2. 离线依赖闭包

Run：skillsbench-offline-closure-two-tasks-v2-20261028。
Profile：runtime/agent-profiles/openhands-offline-two-tasks-20261028/manifest.json。
正式复查入口：

    .venvs/skillsbench/bin/dycon skillsbench offline-check skillsbench-offline-closure-two-tasks-v2-20261028
    .venvs/skillsbench/bin/dycon skillsbench dependencies-verify runtime/dependency-snapshots/openhands-context-safe-20261028

复用 BenchFlow 官方 sandbox/oracle/hardening/verifier/cleanup 生命周期，不启动模型服务或新的 agent loop。
该 run 的 rollout stage 仍为 planned 是预期行为：这里只执行独立 validation，不将 oracle 冒充 OpenHands rollout。

| 任务 | 全新容器离线 OpenHands 安装 | 官方 oracle/verifier | Cleanup |
|---|---|---|---|
| dialogue-parser | 通过 | reward=1.0，verifier_error=null | 已确认 |
| offer-letter-generator | 通过 | reward=1.0，verifier_error=null | 已确认 |

- Docker 实际 HostConfig.NetworkMode=none，接口仅 lo，保存路由信息，不能访问宿主模型服务或公网。
- 207 个锁定包；安装启用 UV_OFFLINE=1、UV_PYTHON_DOWNLOADS=never、UV_NO_CACHE=1、PIP_NO_INDEX=1，使用只读 wheels 与 --require-hashes，无联网 fallback。
- 安装到独立 /opt/dycon-openhands，检查锁定 SDK/tools/ACP import 与 CLI help；不覆盖任务原有 Python 环境。
- 依赖快照包含 308 个受 hash 约束文件；Python 3.12 / linux amd64；每任务保存镜像身份、系统包清单、安装日志及二进制 subject hash。
- 快照 receipt：464c2996b50f50460bf0da5f2e54619c3f7b7caf6cf3e8f80fddb45cc519b31f。
- 两个任务的原始 task package 已重新校验，官方 verifier 测试脚本未改写。

### 环境变体与边界

DOCX verifier 原脚本会联网安装 uv 0.9.7。本次保留原脚本，预置快照中 hash 校验的 uv **0.13.5**、uv tool run 的 uvx 启动包装和标准 shell env。
变体名称 snapshot_uv_preinstalled_in_root_home_v1；uv SHA-256：8a70db72685b7138121bfcda71fa1a00cb709ed9f1fdc4ecf1dd871e8e167e52。
原脚本的在线 bootstrap 在无网络环境可能报 DNS 错误，但不产生联网回退，评分由已预置本地工具完成。
这是通过两个官方 oracle/verifier 的离线环境变体，**不是原始在线安装链逐字节复现**。
系统依赖来自已登记镜像层，不声称本次还验收了冷启动离线 apt 构建、其他平台、全部 SkillsBench 任务或无网络 LLM rollout。

## 3. 清理与回归

2026-10-28 14:26（北京时间）复核：

- Consensus supervisor PID 3210483、runner PID 3210489、模型 PID 3234984 的原进程身份均不再存活。
- 所属代理 PID 3233666：active=false，对应 /proc 已不存在。
- Consensus termination：processes_stopped=true、cleanup_complete=true。
- Consensus 和两个 offline oracle 的所属容器、网络无残留，未停止其他实验。
- Subject、offline-install、dependency-install、result、依赖快照和官方任务包 hash 再次核验通过。

本次收尾对当前工作区重跑回归：runner **1120 passed / 67 skipped**，SkillsBench 专项 **68 passed / 3 skipped**。跳过项不计作通过。
此前 13:30 的 runner 1107 passed / 67 skipped 保留为历史日志，不替代当前结果。
收尾日志：closeout-runner-tests.log / closeout-skillsbench-tests.log；交付汇总见 SKILLSBENCH-CONSENSUS-OFFLINE-CLOSEOUT-20261028.md。

## 4. 证据索引

本地 results/ 保存远程验收 receipt 副本及日志，collected-hashes.json 记录采集时的文件 SHA。
远程汇总：temp/skillsbench_consensus_offline_20261028/。

- consensus-audit.json、artifact-verify.json。
- offline-check.json；每任务详细证据在 offline run 的 validation/offline-oracle/TASK/。
- acceptance-summary.json：汇总审计、决策、两个任务离线结果。
- closeout-verification.json：进程/容器/网络/任务包和证据 hash 复核；receipt 58300229c09e967335297e98cb7715c4ed7c33c44dd36127b1dadb48dde861f9。

旧失败 run、KV2GiB 失败、GPU 编号误用的 NCCL 错误、dialogue-parser 未声明工具失败及离线 uvx 初次失败均保留，不覆盖为成功。
