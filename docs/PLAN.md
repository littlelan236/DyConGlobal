# DyConGlobal 实施方案草案

日期：2026-10-15。工程名已由用户确定为 DyConGlobal。

目标工程根目录建议为远程 `/home/wangtianyi/DyConGlobal`。本文件是规划，不代表已经建立远程工程、导入轨迹或启动任何实验。SSH 凭据不得写入工程、配置或文档。

## 1. 已核实的旧工程事实

远程实际路径是 `/home/wangtianyi/tactRefractor`。

| 已核实位置 | 可复用能力 / 已知问题 |
|---|---|
| `src/tact/environments/containers.py` | 唯一 ContainerFactory；rollout、harness 共享环境策略、镜像身份与容器创建逻辑 |
| `src/tact/environments/nas_images.py` | NAS OCI blob 校验、无本地 tar 的流式 docker load、镜像保护；全 daemon 锁覆盖 load 到 container create，当前会串行导入 |
| `src/tact/evaluation/runner.py` | 官方 SWE-bench harness adapter、环境一致性和评测记录 |
| `src/tact/rollout/` | agent 生命周期、任务协调、失败尝试保留、复用轨迹来源追踪 |
| `src/tact/research/trajectories.py` | 原始 `.traj.json` 读取、assistant steps、reasoning、tool calls、usage |
| `src/tact/research/decision_points.py` | 请求前缀重建、模板与 prompt token 数核对、think-close 位置定位 |
| `src/tact/research/extract.py` | 模型身份校验、分块 hidden 重放；当前层约定和模型适配必须核实，不能照搬层索引 |
| `src/tact/backends/easysteer.py`、`docs/EASYSTEER-BACKEND.md` | 固定 EasySteer runtime、显式 no-op、生成 close token 选择、TP/图执行证据 |
| `baseline/manifest.json`、`baseline/config/splits.json` | 两处均带成员分区信息，正是新工程需要消除的配置重复 |
| `docs/BASELINE-WORKFLOW.md` | enrichment 的 decisions 实际进入 `baseline/hidden/`，且旧工作流会替换后续产物，不适合直接沿用 |

旧 README 所述 canonical baseline 为 Qwen3.5-9B 的 258 条轨迹；旧通用配置另指向 Qwen3.6-35B-A3B。二者不能混用。用户稍后提供的清单才是新工程导入依据，不能据此默认导入 258 条或承袭其训练/评测分区。

抽查的旧轨迹包含原始 response、reasoning_content、tool_calls、reasoning_tokens usage；prompt_token_ids 和输出 token IDs 未保存。需要进行分级重建核验，不能声称所有旧 latent 都是在线原始状态。

远程 NAS 已挂载 `/VM`，可用容量约 42 TiB；本地根分区可用约 267 GiB（检查时值）。旧 NAS Verified200 已有去重 OCI blob store，环境修复也有记录。可复用经过 digest 验证的静态资产，但不能自动采用该 cohort、结果或 gold patch 作为新研究数据。

## 2. 第一阶段研究范围与标签定义

第一阶段完成可追溯的数据集、hidden 提取、剩余工作量估计器训练与离线评测。保留 EasySteer 接口，但不在预测器验证前默认实现或运行新的在线控制策略。

### 2.1 成功与终止必须分开

最终 harness resolved 证明最终 patch 解决了任务；它并不提供轨迹中最早修复正确的时刻。因此首版标签严格称为“到最终成功提交的剩余工作量”，作为剩余任务难度代理。

- `resolved=true`：可以构造该成功轨迹的剩余 steps / reasoning tokens。
- submitted 但 unresolved：可观测到终止的长度，不能赋予“距成功”的真实值。
- 预算耗尽、上下文溢出、超时或中断：不能把终止点标为成功的零剩余难度。
- 未评测或评测发生基础设施错误：成功状态未知，不能当作 unresolved 负例。

主回归实验先仅使用 resolved、记录完整、锚点可恢复的轨迹。失败轨迹完整保留，用于 OOD/误判分析；可单独训练距终止预测器作为对照。未来需要使用失败轨迹做成功时间学习时，再设计带有停止原因的删失/竞争事件模型，不能直接当普通回归标签或无条件套用独立删失假设。

成功样本筛选会带来选择偏差，必须报告成功轨迹数量、任务与仓库覆盖、长度分布，以及失败轨迹上的行为。这个首版模型不能宣称覆盖全部任务的真实难度。

### 2.2 Step 与边界的精确定义

采用从 0 开始的 assistant 调用序号 s；成功轨迹共有 T 个 agent steps，包含最终提交步骤。工具调用数另记，不能混为 step；基础设施 HTTP 重试单独记，不算新的 agent step。

两个锚点分别建立数据视图与模型：

| 锚点 | 可读取信息 | steps 标签 | reasoning-token 标签 |
|---|---|---|---|
| `think_open` | 当前调用 prompt 历史及实际 `<think>` token；尚无本轮思考与动作 | T-s，当前动作阶段仍未完成 | 从当前轮开始，所有剩余轮的 reasoning tokens 之和 |
| `think_close` | 当前轮 reasoning 和实际 `</think>` token；排除随后动作与观察 | T-s，当前动作阶段仍未完成 | 从下一轮开始的 reasoning tokens 之和 |

reasoning-token 标签不包含 prompt、重复历史、tool output、代码/工具参数和普通回复 token；定界符排除。优先由可核实的实际生成 token 区间计数；旧 response usage 的 reasoning_tokens 作为核对依据，文本重新分词结果须标记来源与误差。缺失时保存 null 与原因，禁止补成 0。

最终步骤的 close 锚点可能有 0 个剩余 reasoning tokens，但其动作或提交尚未执行。因此 0 reasoning tokens 不代表任务已经完成。若研究需要“当前动作执行后的剩余步骤”，应新增 after-action 锚点，不能偷偷把当前 close 的标签改为 T-s-1。

如果模板中没有实际 open token、生成被截断导致无 close、或回复出现多个 close，显式记录锚点策略和缺失原因。首版使用首个有效生成 close，后续出现单独保留 occurrence_index；不能与旧 EasySteer 的 all-generated-close 行为混为一谈。

变换默认 `log1p(y)`；steps 和 tokens 两个目标独立训练。对照 DyCon 可再使用仅由训练集拟合的 min–max。测试集不得参与 scaler、层位、超参数或阈值选择；预测非负化与超训练范围处理均写入模型元数据。

## 3. 目录结构：功能源码与产物归各自模块

采用仓库根目录功能包布局，由 `pyproject.toml` 显式声明包；不再另外建立一套 `src/` 镜像功能实现。

```text
DyConGlobal/
├── pyproject.toml
├── README.md
├── GLOSSARY.md
├── config.yaml                    # 唯一可编辑运行配置
├── core/                          # schema、路径校验、ID、原子写入、配置解析
├── baseline/
│   ├── scripts/                   # import_rollouts / rollout / evaluate / audit
│   ├── importers/                 # 旧格式导入适配，不依赖旧工程运行时
│   ├── rollout/                   # agent、协调、任务执行
│   ├── evaluation/                # 官方 harness 适配
│   ├── manifest.json              # 唯一权威 rollout 目录与 task split
│   ├── trajectories/<rid>/raw.traj.json
│   ├── patches/<rid>/patch.diff
│   ├── evaluations/<eval_id>/      # harness reports、日志、patch SHA
│   └── provenance/<import_id>/     # 来源证据、原配置的脱敏只读副本
├── enrichment/
│   ├── scripts/                   # normalize / labels / tokenize / audit
│   ├── normalize.py
│   ├── boundaries.py
│   ├── labels.py
│   ├── outputs/<dataset_id>/      # 标准轨迹、steps、anchors、标签、token 前缀
│   └── reports/<dataset_id>/       # 重建证据、缺失率、质量报告
├── hidden/
│   ├── scripts/                   # extract / verify
│   ├── extract.py
│   ├── adapters/                  # 模型层位、norm、cache 适配
│   ├── outputs/<capture_id>/      # 分片 hidden tensors 与样本索引
│   └── reports/<capture_id>/
├── difficulty/
│   ├── models/                    # ridge.py / mlp.py
│   ├── scripts/                   # train / predict
│   ├── training/                  # 采样、损失、scaler、按 task 聚合
│   ├── datasets/<dataset_id>/     # 标签与 hidden 的轻量 join 索引
│   ├── checkpoints/<model_id>/    # 权重、scaler、model card、训练来源
│   └── evaluation/
│       ├── scripts/              # metrics / trajectories / ablations
│       ├── metrics.py
│       ├── visualize.py
│       └── outputs/<eval_id>/     # JSON、表格、PNG/HTML、诊断
├── action/
│   ├── datasets/                 # 未来标签定义/版本、样本索引；引用相同 hidden
│   ├── analyzers/                # linear probe / MLP 等，细节由用户后续定义
│   └── evaluation/               # 后续动作分析指标和产物
├── runtime/
│   ├── containers/               # 唯一 ContainerFactory / policy / task lease
│   ├── images/                   # NAS catalog、流式 load、租约/资源配额
│   ├── serving/                  # 基础 serving 生命周期与模型客户端
│   ├── steering/easysteer/        # 唯一 steering backend，支持 disabled
│   └── scripts/                  # 镜像预检、prepare、环境核验、runtime setup
├── runs/<run_id>/                # 只读 resolved config、任务选择快照、阶段状态/日志
├── state/                        # 非权威运行锁、租约、恢复信息
├── locks/                        # runner / analysis / serving 依赖锁
├── vendor/                       # 固定版本 EasySteer runtime 与许可证
├── tests/                        # 仅契约、因果边界、并发与关键正确性检查
└── docs/                         # 实施规范、实验协议、复用来源
```

轨迹、模型权重和大张量的物理存储可映射到新工程自己的 NAS namespace；对应产物目录仍是其逻辑唯一入口，映射只在 config.yaml 声明。任何源码脚本不得分散硬编码到 `runs/`、`tmp/` 或其他功能目录。不要通过软链接把整个旧工程目录接进来。

`runs/` 不再另存一份 canonical 轨迹/hidden；它记录配置、任务尝试与阶段事件，通过 rollout_id 或 artifact ID 引用上述模块产物。中断尝试也保留各自原始轨迹并注册，选择哪条作为正式 baseline 必须显式。

## 4. 唯一 manifest 的数据契约

建议使用 `baseline/manifest.json`，小规模研究足够；不同时维护 SQLite/CSV/JSON 三个可写版本。更新通过统一 Catalog API、文件锁与原子替换；多 worker 只能提交更新事务，不能整表覆盖其他 worker 的状态。备份与只读运行快照不成为第二配置入口。

顶层包含：

- `schema_version`、`catalog_revision`。
- `tasks[task_key]`：benchmark 来源/revision、instance_id、repo、base_commit、唯一 partition（train/val/test/excluded）与分组/排除原因。每个 task 只存一次 partition。
- `rollouts[rollout_id]`：task_key、run_id、attempt_id、seed、原始终止原因、用途、轨迹路径与 SHA、patch SHA、模型/模板/运行配置指纹、来源和评测引用。
- `evaluations[eval_id]`：报告路径与 SHA、harness revision、环境策略与镜像 digest、绑定的 patch SHA 和逐轨迹结果。
- 正式 baseline 选择：每个 task 和 experiment profile 指向一个 selected_rollout_id；其他重复尝试不得被偷偷挑选为成功 baseline。

用途以 `purposes` 记录，例如 `training`、`baseline`、`action_analysis`；`partition` 只放在 tasks 表。training 在本方案中表示估计器数据池，train/val/test 的实际访问权限由 partition 强制约束。可以导出直观的训练/baseline 汇总表，但导出表不允许反向编辑成员关系。

不要把 resolved/submitted/unsubmitted 塞进同一个枚举：

```json
{
  "rollout_id": "r_<content-and-attempt-identity>",
  "task_key": "<benchmark_revision>:<instance_id>",
  "run_id": "<run>",
  "attempt_id": "<attempt>",
  "purposes": ["training", "baseline"],
  "submission_status": "submitted",
  "termination_reason": "Submitted",
  "evaluation_status": "completed",
  "resolved": false,
  "selected_evaluation_id": "<eval>",
  "trajectory_sha256": "<sha>",
  "patch_sha256": "<sha>",
  "reused_from": {"run_id": "<original>", "attempt_id": "<original>", "trajectory_sha256": "<sha>"}
}
```

`evaluation_status` 至少支持 not_run/running/completed/infra_error；resolved 只能在有效评测时取 true/false，否则为 null。Submitted 与 patch_present 分别记录，不能仅凭 patch 文件非空推断提交协议完成。

不建立 `splits.json`。difficulty/action 的 train/val/test 视图均查询同一 manifest；实验冻结 catalog revision 与只读 selection snapshot。之后修改 catalog 不得静默改变既有实验。

## 5. 配置、复用与独立性

### 5.1 唯一配置

`config.yaml` 分区：paths、benchmark、model、serving、generation、agent、environment、images、execution、enrichment、hidden、difficulty、action、steering。

- 不写第二套 task ID 分区；benchmark 选择由 manifest 查询条件决定，实际选中 ID 冻结为运行证据。
- 重复 YAML key、未知字段、无效跨字段组合均报错；不在 Python 默认值和 YAML 中同时维护两套业务默认。
- 运行前解析一次，输出只读 `runs/<run>/resolved-config.json`。配置包含模型、模板、预算与 stage 设置，产物写入各模块目录。
- CLI `--set` 只作为显式单次覆盖，全部写进 snapshot；修改已有 run 的有效配置需要新 run ID。
- 单次 pipeline 的预算按整体 deadline 管理；容器 lifetime、工具 timeout、模型 timeout、评测 timeout 分开，不相互隐式扩容。
- 模型路径在配置中显式确认，绝不继承旧工程当前默认模型。

### 5.2 复用策略

采用一次性迁移经核实的可用模块，在新 namespace 中维护；不能运行时 import tact、引用旧 .venvs、读取旧默认 config、依赖旧工程可变代码。保留迁移来源路径、代码 hash/commit（可得时）、许可证与适配说明。

可直接保留原理的部分：轨迹读取、agent/工具协议、harness 适配、环境 policy、ContainerFactory、协调与重试、NAS digest 校验和 OCI stream、EasySteer runtime。

需要有目的修改的部分：配置/schema 与 artifact path 注入；manifest-only 任务查询；enrichment 输出落点和幂等恢复；open/close 双锚点；NAS 并发锁与任务后回收。尽量参数化旧函数的输入输出，不新增另一套行为相同的 runner 或 tokenizer。

独立性验收：临时从新进程的可见路径中移除旧工程后，测试与小样本 pipeline 仍能运行；生产配置扫描不得含旧工程运行目录。静态共享模型/OCI blobs 只能引用外置、只读、digest 固定的资产；新项目有自己的资产索引，不依赖旧 catalog。

## 6. 数据链路与 hidden 提取

数据 DAG：

```text
用户清单 → baseline 导入/新 rollout → 唯一 manifest + harness 结果
                                      ↓
                       enrichment 标准化 / 标签 / token 锚点
                                      ↓
                           hidden 重放与身份核验
                                      ↓
                 difficulty datasets      action datasets
                        ↓                       ↓
                  ridge / MLP             后续分析器
                        ↓
                   evaluation
```

### 6.1 导入

导入器读取用户清单，列出待导入 task/run/attempt/评测证据；物理复制到新项目 namespace，校验 SHA，再注册。旧目录只读，不移动或覆盖。数据来源若包含多个 benchmark，逐条记录真实来源，不把非 SWE-bench Verified 的任务标成 Verified。

原始 raw trajectory 保持不变。标准化数据记录在 enrichment。通过原始 source hash 与 reused_from 识别协同 run 的同一执行副本，防止把转存轨迹当独立采样。每条评测必须绑定相同 patch SHA；缺评测时默认不自动启动大批 harness，而是生成 pending 项。

### 6.2 Enrichment

统一输出 step_id、message_index、调用/动作对应关系、reasoning 计数及来源、锚点位置、缺失原因、完整历史 token prefix 与 source SHA。无需继承 TACT 的 OT/OA/CAL 人工注释门槛；首版标签可从完整成功轨迹直接获得，不能只处理旧 annotation prefix。

质量等级区分：实际 prompt/generated token IDs；完整模板和实际请求字段可核对的重建；仅文本重建。长度相同不等于 token 序列相同。低证据等级单独报告，严格主实验优先采用可核验样本。

为新 rollout 增加最小记录：实际请求消息与模板 hash、工具 schema hash、prompt/generated token IDs（服务能力允许时）、token usage、截断原因、模型/runtime 身份。不能拿后续动作、观察、最终 patch、harness 结果拼入当前锚点输入。

### 6.3 Hidden

对新模型使用明确 adapter。保存 `sample_id=(rollout_id, step_id, anchor, occurrence)`、layer/site、hidden_dim、dtype、norm 位置、model/tokenizer/template/config 指纹与 prefix hash。

首版以 teacher-forcing 重放为主，在真实锚点 token 被消费后的 hidden 上读出。think-open 可能位于 prompt prefill 尾部，think-close 位于生成阶段；两者的在线 hook 路径不同。不要假设现有生成 close selector 同时覆盖 open。

HF 的最后一层 hidden 可能已经经过 final norm，EasySteer 的 decoder output/residual/final_norm 定义又不同；记录语义位置，不只记录整数层号。混合 attention/SSM 模型的 cache 不一定能任意裁剪，前缀不兼容时重置并完整重放，优化需有数值一致性证据。

先在少量长短轨迹和两类锚点上验证 chunk/full replay 与 online/offline 对齐，再批量抽取。按轨迹/分片存储，避免保存整序列全部 token 的所有层；候选层粗扫后再对验证集有希望的层细化。partial shard 原子发布、坏 hash 不跳过，支持按 content key 幂等恢复。

## 7. 估计器与评测协议

### 7.1 划分与训练

- 按 task 分组划分，同 task 的所有 run/attempt/锚点进入同一 partition；检查相同 issue/base_commit 衍生任务和代码修复重复，不只去重文件名。
- 训练与调参只读取 train/val；test 冻结。额外保留 repo-held-out 测试，以区分仓库风格记忆与泛化。
- 不混合不同 checkpoint 的 latent，默认各模型独立拟合；跨模型迁移是单独研究问题。
- 首版矩阵：两个 anchor × 两个目标（steps/tokens）× ridge 与小 MLP。优先完成 ridge 基线，再检验 MLP 增益。
- MLP 初始使用单层 128 或 256 hidden + GELU + dropout + 单输出；以 train/val 选择容量/正则，避免在成功任务少时上大网络。Loss 在 log1p 目标上比较 MSE/Huber，使用 early stopping。
- 按轨迹均衡采样或加权，避免长轨迹因 anchor 多支配训练；置信区间以 task 为聚类单元。

### 7.2 必要对照

1. train 常数/中位数基线。
2. 仅使用当前 step index、已用 tokens、prompt 长度、剩余预算、历史工具调用数量的便宜基线。
3. hidden-only、便宜特征-only、hidden+便宜特征。
4. open 与 close、层位、ridge 与 MLP、steps 与 tokens、跨 repo、成功与失败分层。
5. 排除/单独分析最终提交步骤与显式完成措辞，检查模型是否只识别结论模板；不能因知道完整轨迹长度而把 normalized progress 当输入。

核心证据是 hidden 在未见任务上的增量预测能力。成功轨迹的剩余标签必然随位置递减，曲线下降或较高 R² 本身不证明捕获了独立于进度的真实难度。

### 7.3 指标与图

同时报告 log 空间 R²/MAE 和反变换后的 steps/token MAE、RMSE，Spearman、分长度桶误差、过早预测低剩余量的比例与校准；输出逐轨迹和按 task 聚合结果。短轨迹上的百分比误差需谨慎，不用 MAPE 作唯一指标。

每条轨迹可视化真实/预测剩余 steps、reasoning tokens，横轴给实际 step；标出锚点、工具调用和可得的动作标签、提交与截断/失败类型。总体输出 scatter、残差分布、层位比较与分仓库统计。单例图必须显示预测原始波动，不默认平滑或强制单调；标签的机械单调与模型状态判断的波动分开解释。

补充预算轴：agent 的实际工作量还包括工具执行时间、测试耗时与代码生成，不把 steps/tokens 直接称作 wall-clock 难度。相关观测先记录，为后续扩展留接口。

## 8. Action 信息分析预留

建立 `action/datasets/` 与 `action/analyzers/`，复用 hidden 的 sample_id 和共享 split，不重新提取第二份 latent。先保存原始工具名、命令、动作发生时间与 parse 证据。

动作分类法、单标签/多标签、预测当前轮动作还是历史动作，由用户后续定义；不提前固定 read/edit/test/submit 之类标签。动作标签必须携带 taxonomy_version、来源与置信度。

未来至少区分两个问题：open 时点能否预测本轮将采取的动作；close 时点能否从已生成的 reasoning 中读出即将生成的动作。前者和后者的信息可见性不同，不把 close 对动作的高准确率直接解释为提前规划能力。保留便宜特征与 reasoning 文本等对照位置，具体实验后续确定。

## 9. NAS 镜像与 4 worker 设计

### 9.1 存储语义

镜像的长期权威存储在 `/VM/wangtianyi/DyConGlobal/images/` 的 OCI content-addressed store；优先复用旧数据的 digest 相同 blobs，可让独立 NAS 资产目录成为共享只读底座，不复制整份基础层。复用前验证可用性，旧路径迁移后新工程仍可独立引用共享资产索引。

`docker load` 必须导入 Docker 本地镜像存储；本方案的“不保留”指不长期保留任务容器、镜像和本地 tar，运行时仍需本地空间。若要求运行中也零本地写入，需要不同存储架构，不是 docker load 能实现的。

每个任务执行：

```text
claim task → reserve local bytes → acquire image lease
→ stream NAS OCI into docker load → verify image identity
→ create container → bootstrap → rollout 或 harness
→ export trajectory/patch/logs → remove container
→ release image lease → remove unreferenced owned image → release resources
```

容器删除前必须导出 patch、必要日志与环境证据，不保留 overlay/worktree。rollout 与 harness 使用不同的新容器，但同一个 factory 与相同镜像/环境 policy；顺序执行时可能重新 load，代价写进吞吐统计。

### 9.2 真正支持 4 路加载

- 首轮固定 `evaluation.workers=4`、`images.load_workers=4`，实际任务执行与加载均受总 slot 和空间准入约束，不预先加载全部镜像。
- 将旧的长全局锁改为短全局状态事务 + per-image lock + image lease/reference count。同 digest 只加载一次，不同镜像可进入四个 load 槽位。
- 在全局短事务内原子预约每个正在 load 的预计解压空间与安全余量；加载在锁外完成。不要让 4 worker 都看见相同 free bytes 后同时超额导入。
- 同镜像消费者在创建前已持有 lease，直到容器移除后才释放，消除 load/create/evict 竞态。记录 daemon identity 与 image identity，检查外部容器引用。
- Docker daemon 内部仍可能序列化部分导入；验收只承诺调度器允许四个重叠 load，不虚称吞吐提升 4 倍。
- 只回收本工程登记且无使用者的任务镜像，不执行 broad prune，不 force 删除被引用镜像。共享层由 Docker 引用保护；若有外部引用，记录 deferred cleanup 并稍后核查。
- 持久租约用于崩溃恢复，不能仅靠 PID 判断进程活跃；对照进程身份/锁与实际 Docker 容器重建状态。
- 每个任务结束和项目恢复时尝试清理本工程资源；不修改旧工程 daemon 配置或删除旧镜像。若需彻底空间隔离，可为新项目建立独立 rootless daemon，但先评估重复层、端口和维护成本，再决定。

空间阈值、load timeout、container lifetime、限流和重试全部来自唯一配置，去掉旧 NAS 文件里的硬编码 160/100 GiB 和 1800s。四路最坏解压空间不足时暂停准入，不用删旧镜像强行凑空间。

保留旧环境修复机制，将每个 override 纳入环境 hash。不能假设 image 可 load 就能通过 gold；先在代表性任务上做环境/gold smoke。gold patch 仅用于环境验证，不向 rollout 或估计器输入开放。

## 10. EasySteer 范围

只保留一个 steering backend：EasySteer；baseline 使用 `steering.enabled=false` 的同一执行链。基础无 steering serving 是否保留 native-vLLM 适配，只作为基础 serving 选择，不保留 registry steering，也不重复实现 agent/runner。

现有 EasySteer 的静态方向干预接口不等于支持任意 MLP 动态难度读出。第一阶段只验证指定 checkpoint 的 no-op、hook/site 身份和 runtime 独立安装；在线控制作为后续阶段。

若后续需要用预测难度动态分配预算或控制 reasoning，应先确定控制目标、阈值和代价，再评估 EasySteer 的 request-local state 与设备端读出能力。steps 难度、token 难度与可施加的方向并无天然一一对应关系，不能直接把回归权重当作已验证的 steering vector。

## 11. 实施顺序与验收

| 阶段 | 交付 | 必要验收 |
|---|---|---|
| P0 设计冻结 | 术语、schema、路径 owner、config 字段、迁移清单 | 无 splits 第二来源；成功/终止及 anchor 标签定义明确 |
| P1 独立骨架与复用 | 新包、环境锁、必要源码迁移、config checker、manifest API | 无旧运行时依赖；功能路径受约束；旧工程无改动 |
| P2 NAS 与 runtime | 流式 load、租约与 4 worker、共享 factory/harness、EasySteer no-op | 不同镜像并发与同镜像去重；崩溃恢复；任务后回收；环境一致性 |
| P3 小规模数据端到端 | 按用户清单导入代表性成功/失败/截断轨迹、enrichment、hidden | 来源/patch/eval 对齐；双锚点无未来信息；计数手工核对；模型层语义核实 |
| P4 首轮估计器 | ridge、便宜基线、小 MLP、逐轨迹图 | task 隔离划分；训练集 scaler；未见任务指标；失败轨迹诊断 |
| P5 扩展 | 全量数据、action 细节、在线难度控制实验 | 按后续研究定义单独启动；不提前假定控制收益 |

关键自动检查限于有意义的契约：同 task 跨 split 拒绝；复用副本去重；unresolved 不进入成功剩余量标签；两锚点 off-by-one；当前输入不含未来动作；缺失 tokens 不补零；final norm/site 对齐；并发 catalog 更新不丢数据；同镜像租约保护；NAS 损坏、低空间、任务中断清理。不会为简单目录或静态文档增加镜像实现式测试。

批量实验前的 smoke 建议覆盖一个短成功轨迹、一个长成功轨迹、一个 submitted-unresolved、一个预算耗尽/中断样本，并选择至少两个不同镜像验证并发；具体数量随用户清单和资源可用性确定。每阶段先达到验收，再扩大，不重复生成已可用 rollout。

## 12. 待用户后续提供或确定

1. 正式复用 rollout 清单：task、source run/attempt、所需用途，以及评测证据（可由导入器补查）。
2. 首个 checkpoint、推理模板与 SWE-bench 版本/子集；由新项目 config 显式确定，不采用旧默认配置猜测。
3. action 标签定义和分析目标。
4. 后续是否做在线 reasoning 控制，以及具体控制对象。

这些事项不阻碍先实现独立骨架、数据契约、导入预检和容器调度，但会决定正式数据导入、训练及在线实验的配置。
