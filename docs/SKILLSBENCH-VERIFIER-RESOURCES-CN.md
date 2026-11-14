# SkillsBench verifier 资源快照与发布验收

<!-- agentops-public-release-20261111 -->
Current default AgentOps release (2026-11-11): the py38-py312 snapshot and public qualified profile supersede the older defaults described below. See [AgentOps release identities and usage](SKILLSBENCH-AGENTOPS-MATRIX-CN.md#????-rollout-????). Older six-environment semantics and artifacts remain historical references.

`skillsbench-verifier-resources-v1` 在原 agent dependency snapshot 内增加独立的 `verifier/` 命名空间。agent 的 CPython 3.12、wheel、安装锁及来源元数据逐项继承；verifier 按 Python ABI 独立锁定。旧 snapshot、profile、任务包及 run 不作原地升级。

## 发布身份与使用入口

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

## 构建、资格与回执

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

## 运行时资源隔离

snapshot 只读挂载。verifier 使用固定本地解释器、uv/pip 离线模式，禁止隐式 Python 下载。OpenHands 私有安装仅使用原 agent hash lock；新资源策略下安装子 shell 会清除 verifier 的 `UV_CONSTRAINT`、`PIP_CONSTRAINT` 及 build constraints，避免任务 ABI 的版本锁污染 agent 闭包，verifier 随后仍使用自身约束。环境回执在新策略下包含 `verifier_dependencies.py` 源码 hash，资源安装逻辑变化后重新验收。

完整资源字节校验在私有安装/公共 oracle 的异步钩子中通过线程执行，同一事件循环中对同一冻结身份的并发请求只共用正在执行的那次完整校验，结束后立即移除等待记录，后续请求重新完整校验；单个等待者取消不会取消其他等待者的校验。完成后每个调用仍核对冻结 manifest hash；不在创建环境的同步回调中阻塞其他沙箱的子进程超时计时。环境审计的任一后置异常均令 verified=false，带 error 的旧“成功”回执不得复用。修改这类运行时代码后，最终验收使用新的源码身份重新运行，保留旧诊断回执。

Go 使用本地 1.24.4、完整模块闭包和 attempt 私有可写 module/build cache。MathJax 3.2.2 通过受控 URL 路由使用固定本地资源，Chromium 与 Playwright 版本绑定。public 有代理、public 无代理和 none 使用同一资源选择；none 还通过 Docker `NetworkMode=none`、仅 lo 网卡和离线私有安装回执证明无外网。

本地依赖可用不等于 75 个任务均可离线。Google Auto 原 BugSwarm 脚本还运行 apt 与 Maven；Maven 网络下载及 Java 7 兼容性必须独立验收。不能通过放宽 900 秒预算或删掉原构建步骤宣称通过。

## 验证与错误边界

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
