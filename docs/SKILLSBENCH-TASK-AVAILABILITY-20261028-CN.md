# SkillsBench 任务可用性与 Oracle 验证记录

证据日期：2026-10-28（北京时间）；记录整理：2026-10-29。

固定官方 revision：`f860ecf02ef53412a030f8661c83d644b739ff01`。默认 `tasks/` 共 87 项，机器可读索引位于 `catalog/qualifications/skillsbench-oracle-availability-f860ecf02-20261028.json`。

**该索引标记的是历史官方 oracle/verifier 验证结果：79 项通过，8 项未通过。**每项通过要求该次运行所有 reward 为 1、agent/verifier 无错误且所属容器清理确认；不是模型或 OpenHands 已解题的声明，也不是当前环境或离线运行的持续健康保证。

79 项来自两个不同的冻结运行环境：原始公开网络 run 通过 66 项；修复 profile 的独立重跑通过 13 项。后者使用依赖快照、预置工具及部分任务内存调整；不能声称 79 项是在完全相同的配置下通过。索引逐项记录来源 run、配置和结果 hash。

任务目录是 benchmark 定义；oracle 是否通过属于有版本、配置、时间和来源的验证证据，因此本索引独立于 `catalog/manifest.json` 和不可变 snapshot，避免修改历史任务身份或 run 快照。

## 使用前的能力边界

- 该 87 任务快照的 `validation_only=true`，只能用于 oracle 验证；不能以该快照启动 OpenHands agent rollout。
- 下表“agent 能力门槛”反映原快照的 `inspect_capabilities` 检查，通过该门槛仍不等于已完成 OpenHands rollout 的全量验收。实际运行还须使用非验证专用快照、任务输出路径、冻结 profile 及 `dycon skillsbench preflight`。
- 4 个 oracle 通过任务仍未满足 agent 能力门槛：`data-to-d3`、`fix-visual-stability`、`react-performance-debugging` 和 `shock-analysis-supply`。
- 本记录为 public 网络环境；`dialogue-parser` 和 `offer-letter-generator` 的独立离线验证另见 [离线验收](SKILLSBENCH-CONSENSUS-OFFLINE-ACCEPTANCE-20261028.md)，不能推广为其余任务的离线资格。

## Oracle 已通过的任务（79 项）

| 任务 ID | 验证来源 | agent 能力门槛 |
|---|---|---|
| `3d-scan-calc` | 原始 run | 已通过静态检查 |
| `ada-bathroom-plan-repair` | 原始 run | 已通过静态检查 |
| `adaptive-cruise-control` | 原始 run | 已通过静态检查 |
| `azure-bgp-oscillation-route-leak` | 原始 run | 已通过静态检查 |
| `bike-rebalance` | 原始 run | 已通过静态检查 |
| `citation-check` | 原始 run | 已通过静态检查 |
| `civ6-adjacency-optimizer` | 原始 run | 已通过静态检查 |
| `court-form-filling` | 原始 run | 已通过静态检查 |
| `crystallographic-wyckoff-position-analysis` | 原始 run | 已通过静态检查 |
| `dapt-intrusion-detection` | 原始 run | 已通过静态检查 |
| `data-to-d3` | 原始 run | 未通过静态检查 |
| `debug-trl-grpo` | 原始 run | 已通过静态检查 |
| `dialogue-parser` | 原始 run | 已通过静态检查 |
| `drone-planning-control` | 原始 run | 已通过静态检查 |
| `dynamic-object-aware-egomotion` | 原始 run | 已通过静态检查 |
| `earthquake-phase-association` | 原始 run | 已通过静态检查 |
| `earthquake-plate-calculation` | 原始 run | 已通过静态检查 |
| `econ-detrending-correlation` | 原始 run | 已通过静态检查 |
| `edit-pdf` | 原始 run | 已通过静态检查 |
| `energy-ac-optimal-power-flow` | 原始 run | 已通过静态检查 |
| `energy-market-pricing` | 原始 run | 已通过静态检查 |
| `energy-unit-commitment` | 原始 run | 已通过静态检查 |
| `enterprise-information-search` | 原始 run | 已通过静态检查 |
| `exam-block-sequencing` | 原始 run | 已通过静态检查 |
| `exceltable-in-ppt` | 原始 run | 已通过静态检查 |
| `exoplanet-detection-period` | 原始 run | 已通过静态检查 |
| `financial-modeling-qa` | 原始 run | 已通过静态检查 |
| `fix-build-agentops` | 修复后重跑 | 已通过静态检查 |
| `fix-build-google-auto` | 原始 run | 已通过静态检查 |
| `fix-erlang-ssh-cve` | 原始 run | 已通过静态检查 |
| `fix-visual-stability` | 修复后重跑 | 未通过静态检查 |
| `flood-risk-analysis` | 原始 run | 已通过静态检查 |
| `glm-lake-mendota` | 原始 run | 已通过静态检查 |
| `gravitational-wave-detection` | 原始 run | 已通过静态检查 |
| `grid-dispatch-operator` | 原始 run | 已通过静态检查 |
| `hvac-control` | 原始 run | 已通过静态检查 |
| `invoice-fraud-detection` | 原始 run | 已通过静态检查 |
| `jax-computing-basics` | 原始 run | 已通过静态检查 |
| `jpg-ocr-stat` | 修复后重跑 | 已通过静态检查 |
| `lab-unit-harmonization` | 原始 run | 已通过静态检查 |
| `lake-warming-attribution` | 原始 run | 已通过静态检查 |
| `latex-formula-extraction` | 原始 run | 已通过静态检查 |
| `lean4-proof` | 原始 run | 已通过静态检查 |
| `llm-prefix-cache-replay` | 原始 run | 已通过静态检查 |
| `manufacturing-equipment-maintenance` | 原始 run | 已通过静态检查 |
| `mario-coin-counting` | 原始 run | 已通过静态检查 |
| `mars-clouds-clustering` | 修复后重跑 | 已通过静态检查 |
| `multilingual-video-dubbing` | 原始 run | 已通过静态检查 |
| `offer-letter-generator` | 原始 run | 已通过静态检查 |
| `organize-messy-files` | 原始 run | 已通过静态检查 |
| `paper-anonymizer` | 修复后重跑 | 已通过静态检查 |
| `parallel-tfidf-search` | 修复后重跑 | 已通过静态检查 |
| `paratransit-routing` | 原始 run | 已通过静态检查 |
| `pddl-airport-planning` | 原始 run | 已通过静态检查 |
| `pddl-tpp-planning` | 原始 run | 已通过静态检查 |
| `pdf-excel-diff` | 修复后重跑 | 已通过静态检查 |
| `powerlifting-coef-calc` | 修复后重跑 | 已通过静态检查 |
| `pptx-reference-formatting` | 修复后重跑 | 已通过静态检查 |
| `protein-expression-analysis` | 原始 run | 已通过静态检查 |
| `quantum-numerical-simulation` | 原始 run | 已通过静态检查 |
| `r2r-mpc-control` | 原始 run | 已通过静态检查 |
| `radar-vital-signs` | 原始 run | 已通过静态检查 |
| `react-performance-debugging` | 修复后重跑 | 未通过静态检查 |
| `sales-pivot-analysis` | 原始 run | 已通过静态检查 |
| `sec-financial-report` | 原始 run | 已通过静态检查 |
| `seismic-phase-picking` | 修复后重跑 | 已通过静态检查 |
| `setup-fuzzing-py` | 原始 run | 已通过静态检查 |
| `shock-analysis-supply` | 原始 run | 未通过静态检查 |
| `simpo-code-reproduction` | 原始 run | 已通过静态检查 |
| `software-dependency-audit` | 原始 run | 已通过静态检查 |
| `spring-boot-jakarta-migration` | 原始 run | 已通过静态检查 |
| `suricata-custom-exfil` | 原始 run | 已通过静态检查 |
| `syzkaller-ppdev-syzlang` | 原始 run | 已通过静态检查 |
| `threejs-structure-parser` | 修复后重跑 | 已通过静态检查 |
| `threejs-to-obj` | 修复后重跑 | 已通过静态检查 |
| `tictoc-unnecessary-abort-detection` | 原始 run | 已通过静态检查 |
| `video-silence-remover` | 原始 run | 已通过静态检查 |
| `weighted-gdp-calc` | 原始 run | 已通过静态检查 |
| `xlsx-recover-data` | 原始 run | 已通过静态检查 |

## 未通过的任务（8 项）

| 任务 ID | 本次结果 |
|---|---|
| `fix-druid-loophole-cve` | verifier crashed: Druid patched server did not become ready before verifier |
| `flink-query` | verifier timed out after 600.0s |
| `manufacturing-codebook-normalization` | verifier crashed: verifier exited with rc=1；无 reward 文件 |
| `manufacturing-fjsp-optimization` | verifier crashed: verifier exited with rc=1；无 reward 文件 |
| `python-scala-translation` | verifier timed out after 600.0s |
| `reserves-at-risk-calc` | reward={'reward': 0.0} |
| `shock-analysis-demand` | verifier timed out after 240.0s |
| `travel-planning` | verifier timed out after 300.0s |

## 核对与操作

读取索引可按 `oracle_status` 选择；每行的 `result_path` 和 `result_sha256` 指向原始结果，`receipt_sha256`、task package hash、冻结 profile 和 execution context 用于核对证据身份。`agent_rollout_qualification=not_established_by_this_oracle_sweep` 表示该次 oracle 测评没有为 agent rollout 授予资格。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/python -c 'import json; p="catalog/qualifications/skillsbench-oracle-availability-f860ecf02-20261028.json"; d=json.load(open(p)); print("\n".join(k for k,v in d["tasks"].items() if v["oracle_status"]=="passed"))'
# 历史 run 的官方验证结果及失败项：
cat temp/skillsbench_full_oracle_20261028/merged-summary.json
```

如需重新执行 oracle，先创建新的验证专用 snapshot/run 和明确的环境 profile，再使用 `dycon skillsbench oracle RUN_ID --workers N`；`--task-id ID --retry-failed` 创建独立失败重试并保留旧证据。重新运行后的结果需要发布新的索引，不能改写该历史索引。

OpenHands profile、任务导入、preflight 与边界说明见 [SkillsBench + OpenHands](SKILLSBENCH-OPENHANDS-CN.md)。
