# AgentOps 五环境 verifier 修订

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

## 已发布的 rollout 配套引用

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
