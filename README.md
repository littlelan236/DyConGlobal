# DyConGlobal

coding-agent 轨迹、到最终成功提交的剩余工作量与 latent 分析工程。远程根目录 `/home/wangtianyi/DyConGlobal`，NAS namespace `/VM/wangtianyi/DyConGlobal`。本页按 2026-10-19 远程实际代码更新。

**启动任务先查 [任务入口表](docs/ENTRYPOINTS.md)。已有统一入口时必须复用，不得另建临时脚本、复制 launcher 或直接启动内部 worker，避免绕过快照、选择、身份核验、锁和产物绑定而造成行为不一致。** 普通、capture、steering、steering+capture 都使用 `dycon rollout`。

## 阅读顺序

- [任务入口、命令、环境与恢复](docs/ENTRYPOINTS.md)：全部 CLI、两条数据工作流、批量 extraction 和维护边界。
- [实现状态与限制](docs/IMPLEMENTATION.md)、[文档索引](docs/README.md)。
- [输入清单](docs/INPUTS.md)、[在线采集与存储](docs/online-capture-fullgraph.md)、[vLLM eager 重放](docs/vllm-eager-extraction.md)。
- [difficulty](difficulty/README.md)、[action](action/README.md)、[协作约定](AGENTS.md)。

## 配置、目录清单和快照

`config.yaml` 是唯一可编辑配置，拒绝重复 key、未知字段和非法组合。配置所在目录就是 project root，不能复制临时配置到其他目录启动正式任务。离线分析允许显式 `--set`；正式 rollout 拒绝 `--set`，resume/evaluate 使用原 run 快照。

`baseline/manifest.json` 是任务分区、轨迹用途、来源和评测关系的唯一事实来源，没有独立 `splits.json`。同 task 的全部轨迹共享分区；canonical trajectory、patch、评测按内容/来源绑定，submitted、resolved、基础设施故障分开记录。

`runs/RUN_ID/resolved-config.json`、`selection.json` 和 hash 是不可变证据。rollout 从 tasks 选择；`prepare-evaluation` 从 rollouts 选择并创建独立评测快照，不暗中继承之后新增的轨迹。修改输入/配置须按 artifact 契约使用新 ID。

## 启动与检查

在工程根目录使用指定环境：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/easysteer/bin/python runtime/scripts/verify_easysteer_runtime.py
```

`dycon` 注册到 `core.cli:main`，等价于同环境的 `python -m core.cli`。`runtime-check` 是旧 workflow 的模型/wheelhouse 前置检查；当前全局 `environment.wheelhouse=null` 会使它拒绝，不能代替真实启动验收。

以下命令会启动真实模型服务、Docker 和任务，应先核实范围与资源，按需执行：

```bash
.venvs/runner/bin/dycon rollout CHECK_RUN --check-only --task-id INSTANCE_ID
.venvs/runner/bin/dycon rollout BASELINE_RUN --task-id INSTANCE_ID --unrolled
.venvs/runner/bin/dycon rollout BASELINE_RUN --resume
.venvs/runner/bin/dycon prepare-evaluation EVAL_INPUT_RUN
.venvs/runner/bin/dycon evaluate EVAL_INPUT_RUN EVAL_ID
```

`--check-only` 用独立 run ID，真实验收后退出。`--resume` 不允许新选择/覆盖。`evaluate --gold` 仅验证环境，不注册模型轨迹成功标签；官方 harness 固定 SWE-bench 4.1.0。

## 两条 latent 数据路径

在线路径由 `dycon rollout` 在 managed serving 中采集当前生成 reasoning 换行和 `</think>`，保存 request token ledger、event coordinates、hidden shards，随后 finalize token-event labels。上传失败时用 `dycon capture-recover RUN_ID` 恢复发布，不重复 rollout。确认产物/标签完整后，以 `dycon online-index RUN_ID ONLINE_INDEX_ID` 引用 NAS shard。

离线路径按冻结 prefix 做 teacher-forcing 重放：

```bash
.venvs/analysis/bin/dycon enrich DATASET_ID
.venvs/analysis/bin/dycon hidden DATASET_ID CAPTURE_ID
.venvs/analysis/bin/dycon dataset-index DATASET_ID CAPTURE_ID INDEX_ID
.venvs/analysis/bin/dycon action-index INDEX_ID ACTION_INDEX_ID
.venvs/analysis/bin/dycon difficulty-evaluate INDEX_ID PREDICTIONS.json SCORE_ID
```

离线 hidden 统一使用 vLLM eager，由 CLI 派发独立 latent-vllm 解释器。显式 layers 使用 0-based 公共 site：前 N−1 个是 decoder output，最后一个是 final norm，不能直接混用 native site 索引。离线接口统一采用 trajectory APC：同一轨迹的前一个 anchor 完成后再提交下一个，不同轨迹可以并发，跨窗口保留缓存。vLLM 负责 continuous batching、chunked prefill、APC 和混合 SSM/attention 状态恢复；项目只提交精确 token prefix，不自行裁剪 cache。详见 [连续重放契约与验收](docs/vllm-trajectory-apc-20261019.md)。

token 级核验使用 `token-trace`、`trajectory-token-audit`、`token-join`、`hidden-token-join-audit`。普通离线索引、在线索引和 token join 格式不同；当前 scorer 只支持普通离线 label-shard 索引，兼容性详见入口表。历史四 GPU extraction 使用已有 `enrichment.scripts.launch_extraction`，不得直接启动 queue worker。

当前 vLLM 重放实现位于 `hidden/adapters/vllm.py`、`hidden/backends/vllm/` 和
`hidden/extract.py`；轨迹 prefix 重建位于 `enrichment/trajectory.py`，历史批量
入口位于 `enrichment/scripts/launch_extraction.py`。在线 NAS hidden 对照使用
`hidden/evaluation/replay_online_artifacts.py`，只运行一次按轨迹排序的连续重放，
输出 `trajectory-apc/` 及 `report.json` 中的 `comparison`、`cache`、`windows`；
不再执行逐窗口 reset 或 cold/warm/reordered 三轮重放。历史报告保留原格式。
诊断产物位于 `hidden/evaluation/outputs/`，不会进入 git。

## 代码与产物职责

| 目录 | 职责 |
|---|---|
| `core/` | strict config、CLI、原子 I/O、锁、磁盘预算、model identity |
| `baseline/` | manifest、importers、rollout、官方 harness、canonical trajectories/patches、evaluations |
| `enrichment/` | trajectory labels、prefix、token traces、online token-event finalize；outputs/reports/manifests |
| `hidden/` | vLLM adapter、提取、artifact reader、online recovery；outputs、spool、online capture 产物 |
| `difficulty/` | 共享索引、Ridge/FFN/残差估计器、训练 Python API、预测评分与轨迹图 |
| `action/` | canonical trajectory action 分类、receipt/provenance、shared hidden 索引；训练/评测预留 |
| `runtime/` | 容器、NAS OCI、serving/capture/cache/spool、EasySteer、安装/验收/恢复 |
| `runs/`、`state/` | run 快照/进度、启动证据、supervisor、批量任务和修复状态 |
| `docs/` | 当前入口、实现约定和数据接口 |

## 标签和能力边界

一次 assistant 回复及其动作阶段算一个 step；s 从 0 起，共 T 步时 open/close 的 remaining steps 均为 T−s，当前动作尚未发生。open 的 reasoning-token 标签含本轮及后续轮，close 仅含后续轮；在线 newline 使用当前 token 之后的剩余 reasoning tokens。最终 close 的 token 剩余量为 0 不表示动作已成功。

仅 `resolved=true` 轨迹产生成功剩余量标签；unresolved、未评测和基础设施故障保留 null。缺原始 prompt/output IDs 的历史轨迹明确标记重建证据；长度吻合不能冒充 exact。动作只存于分析证据，不能进入当前 anchor prefix。

Ridge、FFN MLP、Ridge+FFN 和训练 API 已实现。单层 Ridge × alpha 的 submission-remainder pilot 已有正式 `dycon difficulty-ridge` CLI、task 划分和参数/验证报告，见 [实验指南](docs/ridge-submission-pilot.md)；通用 FFN/残差训练 CLI、完整 checkpoint 管理和动态 difficulty 控制仍未实现。action trajectory 分类已有正式 `dycon action-classify`/`action-check` 入口，action taxonomy 学习、训练和评测仍未实现。当前 task 全属 train，不能声称已有 task 级 train/val/test 验证。

## 环境、镜像与验证

已有 runner、analysis、easysteer、latent-vllm 四个独立环境；锁文件在 `locks/`，EasySteer/vLLM 源码在 `vendor/`。长期镜像是 NAS 不可变 OCI blobs，流式 docker load 不生成本地 tar；Docker 解压仍占本地磁盘。load slots、镜像锁、lease 和空间预约协调并发；清理限本工程拥有的容器/镜像。

SSH 密码和 API key 不进入工程。managed serving 校验本地 checkpoint/runtime；external serving 的 checkpoint 身份来自显式配置声明。

已有测试命令（不表示本次文档工作全部重跑）：

```bash
.venvs/runner/bin/python -m pytest -q tests
.venvs/analysis/bin/python -m pytest -q tests/test_catalog.py tests/test_enrichment.py tests/test_hidden.py tests/test_difficulty_estimators.py
.venvs/runner/bin/ruff check core baseline enrichment hidden difficulty action runtime tests
```

合成测试、GPU 小样本、真实 Docker smoke 和 SWE-bench gold 是不同证据；日期化结果只对当时模型、runtime 和任务范围有效。
