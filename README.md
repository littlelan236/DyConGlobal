# DyConGlobal

独立的 coding-agent 轨迹、剩余工作量与 latent 分析工程。远程根目录为
`/home/wangtianyi/DyConGlobal`；NAS namespace 为 `/VM/wangtianyi/DyConGlobal`。

**当前刻意留空**：baseline 数据范围、正式模型 checkpoint、剩余难度估计器架构与训练策略、action 分类体系。
没有导入任何旧 benchmark 数据或运行正式 rollout/训练。已有基础设施代码可在这些输入到位后使用。

## 当前检查

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/python -m pytest -q tests
.venvs/analysis/bin/python -m pytest -q tests/test_catalog.py tests/test_enrichment.py tests/test_hidden.py
.venvs/runner/bin/ruff check core baseline enrichment hidden difficulty action runtime tests
.venvs/easysteer/bin/python runtime/scripts/verify_easysteer_runtime.py
```

`runtime-check` 在模型未配置时会明确拒绝执行。SSH 密码和 API key 不存入工程；API key 只从指定环境变量读取。

## 唯一配置与数据入口

- `config.yaml` 是唯一可编辑配置；拒绝重复 YAML key、未知字段和非法组合。CLI 全局 `--set key=value` 会进入运行快照。
- `baseline/manifest.json` 是唯一任务分区、rollout 用途、来源和评测关系的事实来源；没有 `splits.json`。
- `runs/<run>/resolved-config.json` 与 selection/hash 是只读运行证据。恢复运行使用原快照，不接受覆盖配置。
- 同一 task 的全部轨迹共享分区。复用副本按原始内容去重；submitted、resolved、评测故障分别记录。
- 正式对比可用 `baseline-select PROFILE TASK_KEY ROLLOUT_ID` 显式固定每个 task 的 baseline。已有 profile 不自动改选成功样本。

## 数据到位后的命令

先阅读 [导入格式](docs/INPUTS.md)。所有数据导入都需要显式清单。

```bash
.venvs/runner/bin/dycon tasks-import /path/to/tasks.json
.venvs/runner/bin/dycon rollouts-import /path/to/rollouts.json
.venvs/runner/bin/dycon evaluations-import /path/to/evaluations.json
.venvs/runner/bin/dycon catalog-check
```

checkpoint 可以是工程外的只读静态资产。注册会对完整权重及 tokenizer/模板元数据计算 SHA，随后将返回的 fingerprint、path、served_name 写入唯一配置。

```bash
.venvs/runner/bin/dycon model-register /path/to/checkpoint
.venvs/runner/bin/dycon images-check --deep
.venvs/runner/bin/dycon rollout BASELINE_RUN
.venvs/runner/bin/dycon rollout BASELINE_RUN --resume
.venvs/runner/bin/dycon prepare-evaluation EVAL_INPUT_RUN
.venvs/runner/bin/dycon evaluate EVAL_INPUT_RUN EVAL_ID
```

rollout 按 manifest tasks 查询；评测输入按 manifest rollouts 查询。两种运行快照分别冻结任务和轨迹，评测不偷偷继承 rollout 后新增加的数据。
`evaluate --gold` 仅用于环境验证，结果不注册为模型轨迹成功标签。official harness 固定为 SWE-bench 4.1.0。

```bash
.venvs/analysis/bin/dycon enrich DATASET_ID
.venvs/analysis/bin/dycon hidden DATASET_ID CAPTURE_ID
.venvs/analysis/bin/dycon dataset-index DATASET_ID CAPTURE_ID INDEX_ID
.venvs/analysis/bin/dycon action-index INDEX_ID ACTION_INDEX_ID
.venvs/analysis/bin/dycon difficulty-evaluate INDEX_ID /path/to/predictions.json SCORE_ID
```

hidden.layers 必须显式配置（0-based site index），没有假定的最优层。HF convention 中前 N-1 行对应 decoder output，最后一行是 final norm。
每个锚点独立 teacher-forcing 重放，分块预填充，仅取边界 token 的 hidden。不会跨请求裁剪混合 SSM/attention cache。
receipt 与 shard hash 支持核验、幂等重跑和中断后继续；换输入/配置必须使用新 artifact ID。

## 目录职责

| 目录 | 源码与产物 |
|---|---|
| `baseline/` | importers、rollout、官方 evaluation、唯一 manifest、canonical trajectories/patches、评测结果 |
| `enrichment/` | 步骤重建、token prefix、锚点、成功标签与缺失原因；产物在 outputs/、reports/ |
| `hidden/` | HF adapter、提取和核验；hidden tensor/index/receipt 在 outputs/ |
| `difficulty/` | models/training/checkpoints 暂留空；datasets 是共享索引；evaluation 含指标和逐轨迹图 |
| `action/` | datasets 保存无类别假设的样本索引；analyzers 预留，复用同一 hidden |
| `runtime/` | 共享容器工厂、NAS 镜像、serving、EasySteer、恢复与安装脚本 |
| `core/` | strict config、原子写、进程锁、model identity、CLI |
| `runs/` | 配置快照、尝试状态与环境日志；canonical rollout 归 baseline/ |

## 标签语义与证据等级

一次模型 assistant 回复及其动作阶段算一个 step。步骤序号从 0 起，成功轨迹共有 T 步：open/close 的 remaining_steps 都是 T-s，因为本轮动作尚未发生。
open 的 reasoning tokens 包含本轮及后续轮，close 只包含后续轮。最终 close 的剩余 reasoning tokens 可以为 0，但不表示当前动作已经成功。

仅 `resolved=true` 的轨迹产生成功剩余量标签；unresolved、未评测、基础设施失败保留 null。另存距终止的步骤数量用于诊断。
旧轨迹缺少实际 token IDs 时标记 template/count/parser 重建证据；长度吻合不冒充精确 token 校验。`token_evidence: exact` 可过滤低证据输入。
action 命令等只保存在分析证据字段，绝不进入当前锚点的 `prefix_token_ids`。

## NAS 与容器

长期镜像采用 NAS 上不可变 OCI blob store；stream 到 `docker load`，不生成本地 tar。运行时 Docker 仍需要本地解压空间。
四个跨进程 load slots、每镜像加载锁、生命周期 shared lease、短全局空间预约共同调度；相同镜像只加载一次，不同镜像允许并行。
任务后先保存轨迹/patch/环境证据，再删除容器和无引用的本工程镜像。不会登记或删除原先存在的镜像；不 broad prune，不重启 Docker daemon。

```bash
.venvs/runner/bin/dycon recover
.venvs/runner/bin/python -m runtime.scripts.smoke_images
```

recover 仅清理没有活跃 run 锁的本工程容器。smoke 使用 NAS 中 4 个极小合成 OCI 镜像验证真实 load/create/remove，不是 SWE-bench 或环境 gold 评测。

## 运行环境与复现

runner、analysis、easysteer 三个独立虚拟环境已在远程安装；固定包清单在 `locks/*.txt`。EasySteer/vLLM 源码物理保存在 vendor/，无旧工程可变代码依赖。
只保留 EasySteer steering 路径；baseline 明确发送 `steering:false`。启用干预需提供模型 fingerprint 绑定的原生 payload；动态难度控制和估计器训练未预先实现。
外部模型服务的 checkpoint 身份只能由用户配置声明，managed 服务才提供本地 checkpoint/runtime 校验。

实际 checkpoint 的层语义、online/offline 一致性、SWE-bench 环境修复及 gold 通过率仍需在模型/任务范围确定后验证。当前通过的合成测试不替代这些实验。
