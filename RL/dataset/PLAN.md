# DyConGlobal：GRPO 离线 step 分组数据集构建计划

日期：2026-11-10。项目：`/home/wangtianyi/DyConGlobal`。阶段：方案设计，尚未实现或生成训练数据。

## 1. 已确认的目标与本阶段边界

用户已确认：

1. 先使用已有 SWE-bench parallel-consensus run 构建**离线分组数据集**，在线重新采样留到后续。
2. 一个 group 对应同一 run、task、repetition、attempt、agent 决策 step 下共享同一实际模型输入的全部候选分支，包含未选中、未 retained 的分支。
3. 每个候选保留完整实际输入上下文，稀疏监督仅覆盖该候选本次生成的 reasoning + action。历史 assistant、历史 reasoning、tool observation 等仅作为上下文。
4. 实现源码放在原工程大写 `RL/dataset/`；大体积数据和缓存的物理根目录可配置。
5. 一个 dataset 能显式选择多个 run 中的特定轨迹，参考 `difficulty/` 的来源冻结、划分和 receipt 机制。
6. 历史 token logprobs 通过重放提取，采用经过 temperature、top-p、top-k 等实际采样处理后的概率；这三个参数按用户确认直接读取项目根配置，并冻结到重放产物。具体设计见 [重放提取方案](LOGPROBS_REPLAY_DESIGN.md)。

本阶段产物是可审计、可重放、保留 group 结构的数据基础。reward、advantage、KL、重要性比率、clip、loss 聚合和训练时在线 rollout 均待后续设计；不在构建过程中生成伪 reward 或固定 GRPO 公式。

**建议核心表示：一个实际 prompt + 同一 decision 的 N 个原始分支 completion + 分支监督坐标 + 来源证据。** 不将整条终态轨迹当作一条训练 completion，也不将不同历史前缀的 step 混成一个 group。

## 2. 已核实的工程事实

### 2.1 可复用能力与接口边界

| 已有位置 | 已核实能力 | 本方案的使用方式 |
|---|---|---|
| `RL/README.md`、`RL/service.py`、`RL/config.py` | 单 GPU Qwen3.5-9B LoRA、独立训练配置、attempt/checkpoint/receipt；正式 CLI 当前仍只接工程 fixture | 增加业务 input provider 时沿用现有 service、身份和恢复机制，不新增 launcher |
| `RL/SPARSE_SUPERVISION.md`、`RL/supervision.py` | `explicit_mask` 使用原始 target-token 坐标；保持完整历史输入；坐标 j 的目标由 hidden[j-1] 预测 | 数据 collator 输出原坐标 mask，不提前 shift，不删除前缀 |
| `RL/training.py` | 当前优化循环以监督 token 数归一化 CE 累积；loss callback 的模型 batch 只有三项张量 | 可做数据接入和 mask 验证；不能直接称为已实现 group GRPO，后续要专门设计 group 元信息和目标聚合接口 |
| `difficulty/datasets/selection.py::select_run`、`application/execution/evidence.py::RunEvidence` | 先 run 后 rollout 的显式选择与 canonical trajectory SHA 校验 | 复用选择和证据语义；不从 Catalog 反查训练轨迹 |
| `steering/datasets/selection.py::select_sources` | 多 run 显式 binding、来源快照、冲突检查 | 参考其实现；必要时将通用部分提炼到共享 evidence 层，避免复制第三套选择器 |
| `rollout/decision/coordinator.py`、`rollout/consensus.py` | decision、branch、父前缀、选择策略与原始候选结果 | 读取历史事实，不重新运行共识或重新筛 retained |
| `hidden/artifacts.py` | hash-checked `read_request`、branch index / proof reader | 复用 ledger 和 branch receipt 读取；RL 不要求 hidden tensor |
| `steering/datasets/sessions.py` | 发布证据和完整 decision 的严格读取 | 参考/提炼与 hidden 无关的证明逻辑；不照搬 think_close hidden 资格 |

运行统一使用 `dycon`。新增数据命令在 `core.cli` 注册并委托 `RL/dataset/`，不建立另一套脚本接口。独立 dataset YAML 是拟新增的显式配置例外，需要同步记录到入口文档；不改根 rollout 配置来控制本数据集。

### 2.2 真实数据抽查

只做定向读取，未启动 GPU、未重跑 agent、未恢复/发布 capture、未编辑现有 run。

抽查两个来源目录：

- `verified80-consensus32-fixed05-shortest-mig7a-tp1-20261030`
- `verified80-consensus32-resolved106plus331-gpu45-tp1-w4pg-s64-20261102`

两个冻结配置均为 `parallel.mode: parallel_consensus`、`policy: every_step`、`branch_count: 32`。第一个 run 抽查轨迹为 `r-400ffb2bc8a35ea04014364904e0d03d`。其中第一个 parallel decision 的 32 份 ledger 均可读，实际 prompt SHA 和 parent-prefix SHA 各只有一种，native finish 均为 `stop`，completion 长度为 98–217 tokens。模型配置指向 Qwen3.5-9B。这证明已有数据能表达所需 group，**不代表全 run 已通过训练资格审计**。

重要发现：

1. `raw.traj.json` 内 `extra.dycon_parallel.candidates` 是压缩摘要，抽查条目不含 `response`。完整候选 response 位于 NAS 上 session 的 `parallel/decisions/DECISION_ID.json`。抽查 response 有 `reasoning_content` 和 `tool_calls`，而 `content` 为空；仅提取 content 会丢失 action。
2. 原始 token 来自 request ledger 的 `prompt_token_ids`、`output_token_ids`；附带 request/output SHA、tokenizer fingerprint、decision/branch/request 绑定。抽查文件 SHA 与候选记录一致。
3. NAS 根不是直接拼 `paths.nas_root/sessions`。抽查的物理根为：
   `paths.nas_root / capture.export.root / SOURCE_RUN_ID`，即 `/VM/wangtianyi/DyConGlobal/hidden/online/SOURCE_RUN_ID`。实际实现必须复用冻结配置及正式 locator，禁止硬编码当前路径。
4. 原始 token ledger 和抽查 response 没有可用的逐 token 历史行为 logprobs；当前 native output writer 也未固定保存该字段。不能据 token IDs 推定已有 GRPO denominator。
5. 两个 run 抽查到了相同来源 token 内容，提示迁移/复用别名和重复权重问题。必须进一步按 provenance、request 和 decision 身份判别，不能只凭文本相同就认定同一次采样。
6. 抽查历史 run 没有 `hidden/publication.json` 或新的 `capture-publication.json`，但 NAS 上有完整 decision、native ledger、finalization、branch receipt 及旧 publication 文件。session 的 stopped receipt 还保留 queued/closed=false 等历史队列字段。因此“文件存在”不能直接等同于正式可训练资格，也不能直接假设 steering 的新 publication loader 能读这些历史 run。
7. 部分 run 的 task benchmark 标识同时出现 `SWE-bench_Verified` 和 `legacy-tactRefractor-swebench`。需要核对真实 task/revision 映射；不得只按 run 名或 benchmark 字符串猜身份。

## 3. 数据语义与 group 定义

### 3.1 身份层级

```text
dataset
  └─ source_run + canonical rollout binding
       └─ task/revision + repetition + physical attempt
            └─ session + decision_id + agent_step + actual prompt identity
                 ├─ branch 0: original generated reasoning + action
                 ├─ branch 1: original generated reasoning + action
                 └─ branch N-1: original generated reasoning + action
```

`agent_step` 使用既有 `step_id`，同时保留 `message_index`；两者不能互换。group 的稳定逻辑身份至少绑定 source run、rollout、session、decision、实际 prompt token SHA；task/repetition/attempt/step 是必须核对的关联字段。step 数字或文本 prompt hash 均不能单独充当 group ID。

`parent_prefix_sha256` 表示协调器的父 request/messages 身份，和 native `request_sha256 = hash(prompt_token_ids)` 是不同概念。组内两种证据分别校验，不能彼此替代。

组内应有完全相同的实际 `prompt_token_ids`、模型/adapter/tokenizer/steering 身份，以及除 seed/request ID 外一致的生成策略。同一 decision 若实际输入不同，视为身份错误，不悄悄拆成若干较小 group。

### 3.2 全候选与训练资格分开记录

构建清单必须保留原 decision 的全部 N 个 branch 槽位，包括未 retained、loop-excluded、评分失败、格式错误、生成失败的条目。`selected`、`retained`、在线 `score`、投票支持度和排除原因均为历史元信息。

- **算法层筛除**：仅因 estimator、retention、loop-escape 选择规则未进入 winner pool，不自动排除其完整、可验证的生成序列。
- **证据缺失/输出不完整**：缺原始 prompt/output、缺 action、finish=length/abort、API 失败等，单独标记训练不可用；不能构造空 completion 或以零 loss 假装纳入。
- **身份矛盾**：SHA 不符、decision/branch 错绑、winner 与 canonical response 冲突、组内 prompt 不同，属于硬错误，构建失败。

建议首版冻结 `complete_native_group_v1`：只有全部预期候选均有合格的完整 native completion，且 N≥2 的 group 才进入默认训练视图。其余 group 和候选仍进入清单与排除报告。缺少在线 score 本身不影响此资格。

这是一项**建议的质量策略**，不是用户额外指定的硬要求。它可能降低覆盖率，因此必须报告排除分布。后续如要使用有效候选子集，应显式发布新的 `available_native_group_v1` 视图，冻结原 N、实际 N、全部缺失/排除理由及选择偏差；禁止悄悄改变组内采样分布。单分支 step 不通过复制样本凑成 group。

## 4. 输入、completion 与稀疏监督坐标

令 native prompt 为 `P`，当前分支生成 token 为 `C`，输入为 `X = P || C`：

```text
input_ids:      [完整实际 prompt P] [本分支 reasoning / action C]
attention_mask: [全部有效位置 = 1；仅右侧 padding = 0]
labels:         [      -100       ] [             C           ]
loss_mask:      [      false      ] [            true         ]
```

对原坐标 j：`loss_mask[j]` 选择 `hidden[j-1] -> labels[j]`。第一枚 completion token 的目标坐标是 `len(P)`；不要额外丢掉它或再 shift 一遍。position 0 不可选，空 prompt/空有效 completion 要显式报错。

历史轨迹仅提供来源链；模型输入直接取当前 request 的 native prompt。不能简单用终态轨迹的 `messages[:message_index]` 重建：condenser、模板、tools、截断等可能使它与实际 prompt 不同。绝不包含该分支之后的 tool observation、下一步 response 或最终评测结果作为输入。

建议首版采用版本化的 `native_completion_reasoning_action_v1`：

- 监督当前模型实际生成的整段 response，包括 reasoning、action/tool-call 的原始序列化，以及其中实际生成的结构分隔 token、终止 token。
- prompt 中已有的 assistant/think 前缀保持 mask=false；未出现在 native output 中的 EOS 不补造。
- `reasoning_text`、`content`、`tool_calls`、canonical action 用于检查与展示，不能通过重新序列化 JSON/tool-call 替代原始 completion token。
- 允许合法的空 reasoning，但必须有有效完整 action/终态动作及非空生成 token。无 action 的异常 response 保留为不可训练条目。
- `reasoning_spans` / `action_spans` 在有确切模板证据时记录；无法证明细分边界时写 null，不通过字符串长度猜 token 坐标。整段生成区间仍可准确定位。

若后续要求排除 EOS/分隔符，或只监督某一种 action 格式，必须新建明确的 mask policy/version 和新视图，不能修改已有 dataset。

保留完整上下文的 attention 和梯度依赖。稀疏输出监督不等于截断历史、detach prefix 或可以无限降低主干显存。超长样本报告长度并拒绝或在新视图中显式排除；不默认截断 prompt，不切开 completion/group。复用现有 exact-length 和右 padding 约定，首版不做跨样本 packing。

## 5. 来源选择、重试、重复与划分

### 5.1 显式选择

支持以下互斥模式，选择器最终全部落成冻结 binding：

1. 单 run + 重复 rollout ID。
2. 多 run + binding JSON，每条含 `source_run_id`、`rollout_id`，可选 expected trajectory SHA。
3. 显式 `all_submitted` 便利模式，只冻结当时符合条件的 canonical binding；以后不自动随 run 增长。

不传 selector 时拒绝默认扫描全部 run。首版可直接采用多 run binding 作为主配置格式：

```json
[
  {"source_run_id": "RUN_A", "rollout_id": "ROLLOUT_A", "trajectory_sha256": "EXPECTED_SHA256"},
  {"source_run_id": "RUN_B", "rollout_id": "ROLLOUT_B"}
]
```

来源资格默认不要求 resolved=true：用户选择的成功、未成功、未评测轨迹均可按独立数据质量规则审计。`submitted`、`accepted`、`resolved`、基础设施故障要分开保存；不能继承 difficulty resolved-only 规则作为隐含筛选。

首版以 `RunEvidence` 已登记的 canonical rollout 为来源。显式选择保留其实际 repetition/attempt，不自动挑“最新成功 attempt”。旧的 superseded/quarantined attempt 不从磁盘任意发现；如未来需要支持，必须新增显式历史 attempt 来源契约。

### 5.2 重复策略

- 完全重复的 run+rollout binding：配置错误，拒绝。
- 同一原始 request/decision 被迁移或 reused 到多个 run：沿 provenance 验证后只保留一个逻辑 group，其他来源写入 `aliases.jsonl`，不增加训练权重。
- 两次独立采样碰巧生成相同 token/action：保留为两个样本，不能按 completion hash 合并。
- 同一 group 内两个分支生成相同 action 或完全相同 response：保留原采样 multiplicity，不把它们折成一票。
- 不能证明是别名或独立采样的身份碰撞：硬错误，要求明确来源，不能猜测。

### 5.3 划分

dataset 构建与 split 分离，split 使用新 ID 和独立 receipt。建议默认按已规范化的 benchmark/revision/task 身份做 train/validation/test 划分，同一任务跨 run、repetition、attempt、全部 step 和 branch 归同一分区；兼容别名任务也要同组。不能随机打散 branch 或 step 来宣称跨任务泛化。

可选按 repo 更严格留出，保留为后续扩展。最初 dataset 可不创建 split；一旦用于训练和评估，必须显式引用已冻结 split。

## 6. 原始 token 证据与历史 capture 兼容

建议首版只发布 native-exact 训练视图，不自动回退到重新 tokenize。文本重建可另作诊断输出，明确标记 reconstructed，默认训练 loader 拒绝混用。

证据路径按以下顺序解析：

```text
frozen run / canonical evidence
  -> trajectory SHA / embedded decision summary
  -> 已确认发布的完整 decision + branch receipt
  -> candidate native_request_id + ledger_ref / ledger_sha256
  -> native prompt/output IDs + inventory/tokenizer + request/output hash
```

仅使用已发布、已终止并能验证绑定关系的来源，不从正在写入的 spool 捞取训练数据。hidden tensor、think_close hidden、Ridge index、enrich 不是此数据集的必要前置。缺 hidden 不应导致 token 完整的分支被排除；相反，有 hidden 不证明 action token 完整。

历史 run 是首阶段必须解决的兼容点：

1. 优先复用现有 publication/receipt resolver，检查是否已有正式命令能为指定来源补齐发布证明；必要恢复应通过正式 `capture` 接口，单列执行任务，不让 dataset build 隐式修改来源。
2. 抽查 run 不具备 steering loader 期待的新总 publication。需检查旧 `publications/*.json`、session finalization、branch receipt、队列 ACK 与具体文件链，评估能否形成等价证明。
3. 如需历史 adapter，将**纯读取与校验**实现放到共享 capture/evidence 组件，由 RL dataset 调用；版本化其可接受 schema 和证明要求。不以目录存在、stopped 字样或文件 SHA 单点替代完整发布资格。
4. 无法证明的来源报告 `publication_proof_unavailable`；可以出审计清单，但不能标记训练 ready。不得降低全局 capture/difficulty 的资格规则。
5. parallel 的 visualization-only publication 没有 difficulty training index；本数据集需要自己的 token/action 证据契约，不把旧 visualization-only 标志修改成 difficulty 可训练，也不把 hidden 训练资格直接当作 RL token 资格。

模型、tokenizer、模板、adapter 和 intervention 身份全部冻结。首版训练视图建议要求一个兼容的模型/tokenizer/policy cohort；不兼容来源可在 audit 中列出，不能静默拼接。历史“无 steering/无 adapter”必须由来源证据表达，不能靠字段缺失擅自假定。跨模型训练或不同 tokenizer 的文本迁移另行设计。

## 7. schema 与物理存储

### 7.1 逻辑文件

```text
STORAGE_ROOT/RL/datasets/DATASET_ID/
  dataset.json                  schema、算法版本、source/mask/quality 策略
  source-config.yaml            用户配置原始快照
  resolved-config.json          解析后的全部构建参数
  selection.json                明确选择的 run/rollout/hash
  rollouts.jsonl                task/repetition/attempt/canonical 绑定
  groups.jsonl                  prompt 引用、原分支数、成员清单、组资格
  branches.jsonl                所有候选及 completion 引用、监督区间
  exclusions.jsonl              branch/group 排除理由和严重程度
  aliases.jsonl                 来源复用关系，避免重复计权
  source-evidence/               必需的精简冻结来源证明
  dependencies.json             外部 ledger/decision/inventory locator + SHA
  summary.json                  覆盖、长度、缺失、排除、证据级别统计
  receipt.json                  最后提交的本产物身份与来源哈希
```

每个 group 保存 `group_id`、来源关联字段、`parent_prefix_sha256`、native prompt hash、`prompt_ref`、预期/可用 branch 数、branch IDs、historical winner、质量状态。

每个 branch 保存 `branch_sample_id`、group/branch/request/API-attempt ID、seed、native finish、completion hash/ref/长度、原始坐标 mask spans、selected/retained/历史 score/投票与排除状态、action identity、native evidence locator/hash。text/tool-call 若保存，仅保存必要展示信息并 hash 绑定。

明确区分两种状态：`data_ready` 表示 token/group/证据合格；`grpo_ready` 取决于后续 reward、行为策略概率等具体目标契约，本阶段不宣称 true。

### 7.2 轻量清单与可选 materialization

- build 默认冻结元数据、原始区间和外部来源引用，不复制 hidden tensor 或所有历史文件。
- materialize 将训练所需 prompt/completion 以无 pickle、可 mmap 的分片数组保存到可配置 cache root；一个 group 的 prompt 只存一份，分支各存 completion。
- mask 在存储中用原坐标区间表示，collator 才展开 BoolTensor。token 存储 dtype 经 vocab 范围校验，loader 转为训练要求的 int64。
- 按 group/字节预算流式写 shard；不把整个 dataset 或整份 NAS 一次性读进 RAM。不得只按 N 个样本估预算而忽略长 prompt。
- 不同 step 的实际 prompt 仍可能带来近似二次增长；首版按真实 prompt 去重和引用，不凭前一步 completion 推演下一步 prompt，避免 condenser 不对齐。
- cache key 绑定 dataset receipt、tokenizer/model、mask policy、schema、序列顺序与 dtype；物理 cache root 不进入逻辑内容 hash。
- cache 缺失可重建；源 ledger 缺失与 cache 缺失不同。外部引用式 dataset 不是独立备份；需要可迁移训练包时，显式 materialize 并 seal 独立 payload receipt。

数据根、cache 根独立可配置。建议大产物默认示例为 `/data3/wangtianyi/DyConGlobal`，路径自动追加 `RL/datasets/ID` 或 `RL/cache/HASH`，使用前检查容量。只给本模块增加受限 resolver，不放宽全局 `owned_path`；读写拒绝越界、符号链接冲突和同 ID 异内容副本。

发布采用同盘 staging、排他锁、原子 rename 和最后 receipt。已存在的成功/失败/半成品 ID 不覆盖；可恢复阶段必须有明确 journal/相同身份校验，不能静默续写未知半成品。

## 8. 拟新增配置与 CLI

以下均为**待实现设计**，不是当前可执行命令或已存在配置 schema。

建议 `RL/dataset/configs/swebench-consensus-example.yaml`：

```yaml
schema_version: 1
dataset_id: swebench-consensus-step-v1
sources:
  runs: [RUN_A, RUN_B]
  binding_file: RL/dataset/configs/selected-rollouts.json
filters:
  benchmark_family: swebench
  rollout_modes: [parallel_consensus]
  outcome: any
quality:
  token_evidence: native_exact
  group_policy: complete_native_group_v1
  min_branches: 2
  alias_policy: canonical_origin_once_v1
supervision:
  policy: native_completion_reasoning_action_v1
storage:
  dataset_root: /data3/wangtianyi/DyConGlobal
  cache_root: /data3/wangtianyi/DyConGlobal
  max_shard_bytes: 268435456
  memory_budget_gib: 4
```

相对配置路径统一相对 project root；拒绝重复 YAML key、未知字段、冲突 selector 和隐含默认扩大来源。`parallel_enthophy` 是当前另一个 consensus 模式，但本阶段默认只纳入用户要求的 `parallel_consensus`；如要包含熵策略 run，必须显式扩展 allowlist 并冻结历史策略。

拟新增命令：

```bash
# runner：只检查配置、来源、预估与资格；不加载 Torch/tokenizer/GPU，不发布 dataset
.venvs/runner/bin/dycon rl-dataset build --dataset-config CONFIG.yaml --dry-run

# runner：真正冻结 dataset；材料化只读取现有 native token，不重新 tokenize
.venvs/runner/bin/dycon rl-dataset build --dataset-config CONFIG.yaml
.venvs/runner/bin/dycon rl-dataset check DATASET_ID --dataset-root STORAGE_ROOT
.venvs/runner/bin/dycon rl-dataset materialize DATASET_ID \
  --dataset-root STORAGE_ROOT --cache-root CACHE_ROOT

# 同现有 artifact 家族对齐，scope 名待注册
.venvs/runner/bin/dycon artifact verify DATASET_ID \
  --scope rl-dataset --storage-root STORAGE_ROOT
```

`check` 必须区分 metadata/来源验证与完整 token payload 校验，输出明确的检查等级；materialize 必须核验所读 token payload，不能仅相信 build 时的 SHA。dry-run 可以写用户显式指定的独立审计报告，但不能输出“已验证全数据可用”或生成成功 receipt。

模块建议拆为 `config.py`、`selection.py`、`sources.py`、`groups.py`、`schema.py`、`build.py`、`materialize.py`、`loader.py`、`artifacts.py`、`cli.py`；按实际复杂度合并，避免一文件一空壳。来源通用逻辑复用/提炼到既有共享组件；训练依赖不进入 runner 的轻量 import 路径。

## 9. 与现有 RL 训练接口的衔接

提供两层读取：

1. `iter_groups`：返回完整 group、成员、prompt/completion、mask 和 provenance，作为后续 group objective 的稳定输入。
2. CE/mask 验证适配器：逐候选输出现有 `input_ids/attention_mask/labels/loss_mask`，使用 `supervision.mode: explicit_mask`。

group sampler 先在逻辑上选中完整 group，再按显存预算顺序执行候选 microbatch；不要求 32 个长序列同时驻留 GPU，也不把普通 batch size 当作 group size。计算阶段可拆分，但 group advantage、归一化和样本权重不能因物理拆批改变。

现有 `RL.training.train` 的 callback 会丢弃 group 元信息且以监督 token 数归一化；后续 GRPO 需要受控扩展目标/归一化契约，不能把 group reward 乘进现有 CE 后声称等价。此阶段保留数据契约并验证稀疏 mask，不提前改写训练目标。

接入正式业务训练时，input identity 应绑定 dataset receipt、materialization、split、视图策略、group/member 顺序、sampler seed/epoch 和 cursor；checkpoint 保留 group/branch/token 消费位置。数据、mask 或成员顺序变化拒绝 resume；warm-start 走已有显式语义。

## 10. 为评分、KL、loss 保留的信息

| 信息 | 当前数据能否直接提供 | 处理方式 |
|---|---|---|
| 历史 estimator score、retention、投票、winner | 多数 decision 有；可能 null | 原样保存来源，不转成 reward/advantage |
| canonical 轨迹最终 resolved/evaluation | 由 run evidence 提供，可能未评测 | 作为 outcome 元数据；未选分支没有其自身后续 rollout，不自动继承 winner 的最终 reward |
| token-level behavior logprobs | 原生记录抽查没有；用户已确定通过历史 token 重放计算 processed logprobs | 使用独立 replay sidecar，绑定历史模型、根配置采样参数及原始 token；逐 token 只保存实际生成 token 的 float32 logprob，标记 replay 来源，禁止用当前模型或 raw logprobs 替代 |
| reference logprobs / KL | 本阶段没有 | 未来以明确 reference checkpoint、token/mask 和概率定义生成独立、可校验 sidecar |
| group reward/advantage | 待设计 | 后续以 dataset/group/member IDs 绑定独立产物，不修改基础 dataset |

离线历史分支来自历史行为策略，并非训练更新后当前策略的 on-policy 样本。已确定的教师强制重放必须区分行为模型与 reference 模型，并冻结 checkpoint、adapter、干预和采样分布定义。按用户确认，temperature/top-p/top-k 读取当前根配置；历史模型和其他历史约束仍按来源恢复。抽查根配置模型已变为 Qwen3.6-27B，而上述历史轨迹来自 Qwen3.5-9B，不能整体套用当前根配置。seed、token IDs 或相同基础模型名不能代替行为概率证据。重放结果不冒称生成时原生记录，验收和缺失处理见独立方案。

此边界不阻碍先构建 dataset，但决定其不能单凭本阶段产物被宣称为标准在线 GRPO 训练集。

## 11. 实施顺序与验收

### P0：指定来源并完成资格预审

输入：用户指定的 run/rollout bindings（尚未给出，计划中的 RUN_A 等仅为占位符）。

输出：每个来源的模式/模型/tokenizer/任务身份、canonical 与发布证明、decision/branch/token 覆盖、原始/可用 group 数、缺失 logprobs、历史 alias、长度和磁盘预算摘要。只读运行，不默认全量扫描或替用户选择某个实验来源。

先验证新旧 publication schema 的可读范围，再决定必要的正式 capture 兼容工作。若来源仅缺 hidden、但 token/action 证据已充分，应允许进入对应 token-only RL 资格；若缺发布证明，先报告并分离恢复工作。

### P1：冻结 schema、选择器与 metadata build

实现多 run binding、稳定 group/member ID、完整候选 inventory、alias、来源快照与 receipt。先用少量指定真实轨迹验收，再扩大来源。包括正式 CLI、配置示例和 `RL/dataset/README.md`，更新 `docs/ENTRYPOINTS.md`。

### P2：native token materialization 与稀疏 collator

实现有预算的 shard/mmap、prompt 去重引用、完整 hash 校验、原坐标 mask、group iterator。验证 exact tokens 与 collator 输出，不开展评分或 optimizer 训练。

### P3：真实小规模审计与只读训练接入验证

使用选定 1–2 条轨迹检查全候选覆盖、相同 prompt、分支 reasoning/action、首末监督坐标、历史/未来隔离、别名权重及超长拒绝。可以运行 CPU 小模型的 mask/输入契约验证；真实 GPU 训练和容量测试是后续独立任务。

### P4：历史采样 logprobs 重放与绑定

按 [LOGPROBS_REPLAY_DESIGN.md](LOGPROBS_REPLAY_DESIGN.md) 增加正式 replay/check 入口，复用已安装 vLLM 的 trace replay 与 processed-logprobs 能力。先完成参数、模型、trace 和采样分布验收，再重放指定数据范围并提交独立 receipt。此阶段只做推理，不做 optimizer 更新；其 GPU 容量验收单独记录。已有基础 dataset 不回写，训练显式绑定通过校验的 replay ID。

### 必须覆盖的有意义测试

- 同一步所有完整候选被保留；非 winner/非 retained 不因在线选择结果被丢弃。
- 两个 step 相同编号但来自不同 task/run/attempt 时不会误合并；组内实际 prompt 不同会硬失败。
- 压缩轨迹中没有 response 时，能从 receipt 绑定的 full decision 正确读取 reasoning 和 tool call。
- 原生 prompt/output 与构建结果逐 token 一致；首个 completion target、EOS、右 padding、单次 causal shift 正确。
- 缺失/abort/length/无 action、单分支等状态可解释，完整组默认规则不静默变成有效子组。
- ledger/decision/trajectory/inventory SHA 或身份被篡改时失败；缺 hidden 不等同缺 token。
- run 迁移副本不重复计权；独立同内容采样和组内相同 action 保持 multiplicity。
- 新旧 publication 分别按其明确证据链验证；未发布 spool 不可训练。
- 不同物理 storage/cache root 可读取同一逻辑产物；已有 ID、越界和半成品拒绝；缺 cache 可重建，源损坏不能补零。
- task split 不跨 task 别名泄漏；loader 顺序/cursor 可重现。

验收报告必须给出：选中轨迹数、原始/训练可用 group 数、原始/训练可用 branch 数、每组大小分布、selected/retained 覆盖、排除原因、alias 数、模型/tokenizer cohort、prompt/completion/有效监督长度分布及最大值、数据/缓存字节数与完整校验结果。不能用几条样本成功代替全量通过。

## 12. 本计划后的决策点

**已确认**：离线、step 分组、全部候选、仅当前分支 reasoning+action 有效监督、源码位置与可配置大产物路径；历史实际生成 token 的 logprobs 使用经过 temperature/top-p/top-k 等处理后的采样分布，通过历史 token 重放提取，这三个参数使用根配置。

**建议默认、实施前据预审调整**：native-exact、完整组训练视图、生成分隔符/实际终止 token 纳入监督、保留重复 action、来源别名只计一次、按 task 留出、历史 score 只作元数据。

**启动实际构建还需要的输入**：目标 run/rollout 清单；若希望包含 `parallel_enthophy` 或历史 superseded attempts，需要明确扩展来源范围。这不影响当前计划交付。

**后续算法另议**：分支 reward 及因果可用性、group 优势/零方差处理、reference/KL、token/branch/group 加权与 loss、离线策略偏差及在线采样。行为 logprobs 的来源与提取方式已按上述重放方案确定；基础数据不预先固化其余选择。

## 13. 阅读与核查依据

远程文档：`AGENTS.md`、`README.md`、`docs/ENTRYPOINTS.md`、`docs/MINIMUM-RUN-CN.md`、`RL/README.md`、`RL/SPARSE_SUPERVISION.md`、`difficulty/README.md`、`docs/STEERING-DATASET-LDA-CN.md`。

定向实现：`RL/training.py`、`RL/supervision.py`、`difficulty/datasets/selection.py`、`steering/datasets/{selection,build,sessions}.py`、`hidden/artifacts.py`、`rollout/decision/{coordinator,candidate}.py`、`rollout/consensus.py`、`runtime/serving/{capture,vllm_plugin}.py`。

CLI 核查：`dycon --help`、`difficulty-dataset compose --help`、`steering-dataset build --help`、`run active-status`、`capture --help`、`capture-publish --help`。检查时 active run 数为 0；此事实不作为历史来源资格依据。

本地 SSH 检查日志保存在 `C:/Users/wangt/Desktop/notes/tasks/grpo_dataset_20261110/ssh_logs/`；连接均为一次性非交互命令，设超时和输出限制，已确认退出并关闭。本文不保存任何连接凭据。
