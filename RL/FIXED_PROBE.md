# 固定 group 偏好 probe

`dycon lora-probe` 在显式 checkpoint 上，对固定、完整的 GRPO parent groups
进行 teacher-forced raw T=1 评分。每个分支一次完整 forward；不做 rollout、
backward、optimizer 恢复或更新，不写训练游标和 TensorBoard。

在项目根目录用 RL 环境执行正式 CLI（runner 环境可执行 `--help` / `--dry-run`）：

```bash
.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000100 \
  --group-id GROUP_A --group-id GROUP_B \
  --gpu 0 --output reports/rl-probe/step100.json

.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000200 \
  --baseline-report reports/rl-probe/step100.json \
  --gpu 0 --output reports/rl-probe/step200.json
```

第二条命令继承前一报告的有序 group ID 清单。也可再次传入完全相同的
`--group-id`。不自动选择最新 checkpoint，不随机抽样，不重算优势。
`--gpu` 是物理 GPU 编号或 MIG UUID：命令使用训练共用的 GPU 锁，并检查
设备空闲。应选择独立空闲设备；不会自动暂停、重启或抢占训练。

加 `--dry-run` 可检查 checkpoint receipt、所属 attempt 的
`resolved-config.json` 与冻结 schedule、group ID 和输出路径；无需 `--gpu`，
不加载模型，不创建报告。它不代表已验证选中 payload 或完成数值比较。

## 与训练实时指标的区别

训练期正负优势分支的 logprob 变化及偏好差使用当步 batch，随训练进度更换样本；
固定 probe 在不同 checkpoint 上重评同一份 group 清单，因此能排除样本更换造成的
曲线变化。两者都使用原始 anchor 与冻结优势，但固定 probe 需要额外 forward，
通过独立 CLI 随时调用，报告不进入 TensorBoard。

选取 ID 时可先通过 `lora-advantages ATTEMPT_ID --storage-root ROOT --group-ordinal 0`
查看实际 group 标识及冻结正负分支。不要将 group ordinal 或 8x16 半组 ID 当成
`--group-id`。选定后，在后续 checkpoint 中用 `--baseline-report` 继承清单。
应选择同时包含正负优势分支的 groups，且持续保留其源数据。

元数据预检示例（将路径和 ID 替换为实际值）：

```bash
.venvs/runner/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000100 \
  --group-id GROUP_A --group-id GROUP_B \
  --output reports/rl-probe/step100.json --dry-run
```

## 统计口径

对每个分支，先求真实 completion target 的平均 logprob。随后在每个
group 内按冻结优势符号划分正负分支，分别对分支等权平均；最后对 group
等权平均。不会让长 completion 或分支数较多的 group 支配整体结果。

| JSON 字段 | 含义 |
|---|---|
| `positive_logprob_mean` / `negative_logprob_mean` | 正/负优势分支的平均 logprob |
| `positive_logprob_change` / `negative_logprob_change` | 相对训练原始 anchor 的 logprob 变化 |
| `preference_gap` | 正优势 logprob 均值减负优势 logprob 均值 |
| `reference_preference_gap` | 原始 anchor 在相同分支上的偏好差 |
| `preference_gap_change` | 当前偏好差减 anchor 偏好差；越大表示相对 anchor 越偏向正优势分支 |
| `advantage_logprob_alignment` | 分支平均的 `advantage * logprob_change` |
| `pairwise_preference_rate` | group 内正负分支两两比较平均 logprob 的胜率，平局计 0.5 |
| `improved_group_fraction` | `preference_gap_change > 0` 的可比较 group 比例 |

报告包含 `aggregate`、`support`、逐 group / branch 明细、checkpoint step
与 receipt 哈希、probe 身份和评分配置。缺少正/负符号时相关比较为 `null`；
`support.comparable_group_count` 给出有效比较数量。优势为零的分支不进入
正负偏好差，但保留在总体分支统计及明细中。

`comparison.aggregate_delta` 和 `comparison.group_delta` 都是
**当前报告减历史报告**。必须保持有序 group ID、token、mask、冻结优势、
参考概率字节、base/anchor 身份以及评分协议、dtype、分块配置和依赖版本一致。
不兼容时明确报错，不生成可误读的改善幅度。

例如：anchor 偏好差为 `0.10`，step100 为 `0.25`，step200 为 `0.32`。
step200 的 `aggregate.preference_gap_change` 为 `0.22`；以 step100 为基线时，
`comparison.aggregate_delta.preference_gap` 为 `0.07`。前者始终相对训练原始
anchor，后者相对指定历史报告；传入基线报告不会更换训练 anchor。

命令 stdout 返回完成状态、报告路径、整体指标及比较摘要；`--output` 文件保存
完整 JSON，包括 `groups[].branches`、`probe_identity` 和 `probe_sha256`。

## Token 变化分布与集中度

每次正常 probe 默认在 JSON 顶层 `token_changes` 及
`groups[].token_changes` 中输出以下统计，不需要新增 CLI 参数。
它们复用现有评分得到的 target logprob，不增加模型 forward，不进入 TensorBoard，
也不输出逐 token 数组、位置、文本或逐 token 前后概率。

定义 `d_i = log(p_current_i) - log(p_anchor_i)`，使用自然对数；
概率指同一固定上下文下、raw T=1 的真实 completion target 概率。
所有变化始终相对训练原始 anchor；`--baseline-report` 不会改变这一参照。

每个统计区都有 `all`、`positive`、`negative`、`zero` 四个分区，分别包含
全部 token、正优势分支 token、负优势分支 token、零优势分支 token。
分区由冻结的分支优势决定。顶层合并所有选中 group 的实际 token 后计算，
group 内也直接合并 token；**每个 token 等权**，不平均分支或 group 的分位数。
因此长分支权重更高，与原有 `aggregate` 中分支等权再 group 等权的指标口径不同。

以下字段相对于 `token_changes.all`（或相应 group / 优势分区）：

| 字段 | 含义 |
|---|---|
| `token_count` | 实际参与统计的 completion target token 数 |
| `logprob_change_mean` | `mean(d_i)`，带方向，可能正负抵消 |
| `abs_logprob_change_mean` | `mean(abs(d_i))`，反映平均变化幅度 |
| `logprob_change_quantiles` | 带方向 `d_i` 的分位数 |
| `abs_logprob_change_quantiles` | `abs(d_i)` 的分位数 |
| `top_abs_change.top_1pct.abs_change_share` | 按 `abs(d_i)` 降序排列，最大 1% token 占全部 `sum(abs(d_i))` 的比例 |
| `top_abs_change.top_5pct` / `top_10pct` | 同上，分别取最大 5% / 10% token |
| `relative_probability_change.gt_10pct.either_fraction` | `abs(p_current_i / p_anchor_i - 1) > 0.10` 的 token 比例 |
| `relative_probability_change.gt_50pct.either_fraction` | 同上，阈值为 0.50 |
| `relative_probability_change.gt_10pct.increase_fraction` / `decrease_fraction` | 分别为相对概率上升 / 下降超过 10% 的比例；`gt_50pct` 同样提供方向拆分 |

分位点为 `p00/p01/p05/p10/p25/p50/p75/p90/p95/p99/p100`，使用线性插值。
Top k 为 `ceil(token_count * 百分比)`，非空时至少取 1 个；每项的
`token_count` 给出实际 k，小样本的实际比例可能高于名义百分比。
空分区的均值、分位数和比例为 `null`，计数为 0；全部变化为零时集中度
`abs_change_share` 为 `null`，非空分区的阈值占比为 0。

阈值是严格“大于”，相对概率变化不是百分点变化。例如从 0.20 变为 0.22
是相对上升 10%，不是上升 10 个百分点；恰好达到阈值不计入。
实现直接比较 `d_i > log(1+t)` 或 `d_i < log(1-t)`，避免指数溢出。
各概率只在固定的 teacher-forced 上下文下成立，不代表自由生成的轨迹概率。

例如所有 token 变化同样大，最大 1% token 的变化贡献率约为 1%；
若仅 1% token 发生变化，该贡献率可达 100%。结合带方向均值、绝对均值
和分位数，可以区分“普遍小幅变化”“少量 token 大幅变化”和“正负抵消”。

该统计区使用独立 `version: 1`，是报告 schema 1 的新增字段，不改变评分协议、
`probe_sha256` 或原有偏好指标。旧报告仍可作为 `--baseline-report` 继承分组并比较
原有指标；`comparison` 继续只比较原有标量指标，不对分位数或集中度做隐含差分。
历史报告仅保留分支均值，无法补算 token 分布；需重新运行 probe 并输出到新路径。
JSON 只保存汇总，评估期间在 CPU 保留选中 token 的 double 精度变化值，
空间随选中 target token 数线性增长，并进行精确排序和分位数计算。

## 实现验证范围

固定 probe 的 19 项专项测试已通过，覆盖真实 tiny-Qwen 前向与普通 logits 对照、
重复评估、跨 checkpoint JSON 比较、runner CLI dry-run、冻结 payload 与身份校验、
8x16 完整 parent 读取以及源文件不变检查。新增分布统计还覆盖均匀与稀疏变化、
正负抵消、token 池化与符号拆分、Top k 取整、空分区及全零变化、严格阈值与极端值、
JSON 有限值、无额外 forward、重复评估和旧报告兼容性。基础版此前通过相关训练、
目标函数、prepared / streaming 恢复回归共 87 项；本次未重跑该整套训练回归。
本次新增统计未重新执行完整 Qwen3.5-9B GPU probe；CPU 专项测试中的 GPU
准入使用替身，实际评分使用小型真实模型。入口与源码见 `RL/probe.py`、
`RL/probe_data.py`、`RL/probe_metrics.py`，测试见 `tests/test_rl_probe.py`。

操作流程见 [训练接口指南](TRAINING.md)，实时监控与独立报告的分工见
[可观测性文档](OBSERVABILITY.md)，目标函数及 anchor 语义见 [GRPO.md](GRPO.md)。

## 按 group 训练位置绘图

仅评测显式 `--group-id` 或 `--baseline-report` 继承的固定清单，不自动评测全部历史 groups。
新评测可用 `--plot-output` 同时生成 PNG / SVG / PDF；图表复用当前 JSON 指标，不增加 forward：

```bash
.venvs/rl/bin/dycon lora-probe \
  --checkpoint /absolute/attempt/checkpoints/step-000400 \
  --group-id GROUP_A --group-id GROUP_B --gpu 2 \
  --output reports/rl-probe/step400-new.json \
  --plot-output reports/rl-probe/step400-new.png
```

已有报告可通过正式绘图接口直接绘图，无需 GPU 或重新评分：

```bash
.venvs/runner/bin/dycon lora-probe-plot \
  --report reports/rl-probe/step400-new.json \
  --output reports/rl-probe/step400-chart.svg
```

输出图表及同名 `.data.json`，后者保留完整 group ID、零基 ordinal、坐标、
训练状态、参与更新的全部 step、逐组指标和源报告 SHA256，便于定位和复核。
输出拒绝覆盖且不能写入训练 attempt / 密封 artifact / `RL/` 树。
绘图需要当前命令所在 Python 环境安装 matplotlib，使用 headless Agg 后端。
若 JSON 已完成而绘图失败，CLI 明确报告已保存的 JSON，可用 `lora-probe-plot` 重试。

**横轴中的点是 group，但位置与训练 step 呈线性比例**：取冻结逻辑训练计划中
该完整 parent 首次参与 optimizer update 的 step（从 1 开始）。例如 step 20 到
40 的距离为 step 20 到 30 的两倍；同一 step 的多个 groups 重合，不做水平抖动。
稀疏选组不按样本序号等距排列，不连接或插值没有评测的 groups。图上标出
checkpoint step 的竖线，小样本标注 group ordinal；完整 ID 见配套数据文件。

所有点均由**当前所分析 checkpoint**评分，纵轴变化均相对原始训练 anchor；
横轴上的历史 step 仅表示 group 在训练中的位置，不是该历史模型的评分结果。
五个面板依次为：正负优势分支平均 Δlogprob、偏好差变化、token 绝对变化幅度
（均值/中位数/P95）、概率相对变化超过 10%/50% 的 token 占比、Top 1%/5%/10%
变化 token 的绝对变化贡献率。前两项按分支等权，后三项在每组内按 token 等权。

默认只绘制 checkpoint 时已经完整训练的选中 groups。需要对照未训练组时，
绘图命令加 `--include-untrained`；评分命令则加 `--plot-include-untrained`。
未训练组在其**计划首次训练 step** 处以叉号显示，部分训练 parent 用菱形显示，
不会冒充已完整训练。8x16 两次半组更新记录首次和末次 step；默认须两个半组
都已提交才算已训练。prepared append 的各 segment 使用实际冻结 window
（包括不足一个 batch 的末尾），跨 GPU 拓扑恢复仍按原始逻辑 window 定位。

新 JSON 新增 `training_schedule`，不改变评分身份或原有 baseline 兼容性，
且可离线复用此坐标绘图。旧 JSON 无此字段时，绘图读取其原 checkpoint，
验证 receipt、step 和冻结计划来补充坐标，不改写原报告；因此旧报告的源
checkpoint、所属 resolved config 和 schedule evidence 必须可读。
旧报告若无 token 分布，前两图正常绘制，其余面板明确标注缺失。

### 图内数值报告与解读

图表在原五个面板下方默认附带逐组结果表（正负优势变化、偏好差、优势对齐、token
数量、绝对变化、Top 1% 贡献及超过 10%/50% 的概率变化占比）、各训练状态的
分组均值/中位数与改善组数，以及正/负/all 分区的 token 加权汇总。
同时写明样本规模、原始 anchor、teacher-forced T=1、横轴语义、加权口径差异和
训练池诊断边界；有历史基线时展示源报告 aggregate 比较结果，注明其完整源报告范围。
这些数据从实际 JSON 动态汇总，不写死某次 step400 结果；`.data.json` 的 `summary`
保留相同数值。图中逐组表与汇总只包含当前实际显示的 groups。

全部源 groups 均显示且源报告提供 token 分布时，还附上精确池化的 median / P95 /
P99 绝对变化、Top 1%/5%/10% 贡献率及文字解读。过滤后的子集无法从旧报告的
分组汇总还原精确池化分位数/集中度，因此明确标注不可用；不会对 group 分位数
求平均或挪用未显示 groups 的总体分布。部分 group 缺少 token 统计时，对应 cohort
的 token 汇总标记 N/A，不静默只汇总部分 group。接口参数和评分行为保持一致。

## 数据与资源边界

- 使用 checkpoint 原始共享 anchor，支持 prepared、native streaming 和 legacy。
  8x16 训练也按完整 parent group 评分；不接受半组 ID。
- prepared 复用密封 token / reference artifacts；streaming 需要选中 groups
  的原始 native ledgers 和概率源仍可读，包括训练已消费的 group。
  legacy 需要所属 attempt 的 `frozen-groups.json`，逐组扫描并验证完整
  冻结内容哈希；旧文件很大时这一步可能耗时较长，不会整体读入内存。
  现有 checkpoint / artifact 完整性校验仍会读取关联文件计算校验和；
  额外模型 forward 仅覆盖指定 groups，但启动校验 IO 不保证仅限这些 groups。
- checkpoint 必须位于保留 `resolved-config.json` 的所属 attempt 下。
  模型路径使用该配置，base 身份遵循项目已有注册与文件元数据检查。
- 完整分支评分使用与训练共用的 native target scorer、MLP 分块和 attention
  设置；投影未配置分块时使用 256 token 的有界块，不启用训练专用 prefix
  backward 路径。长上下文仍需显存；OOM 直接报错，不截断或更改样本。
- JSON 写入显式新路径，原子发布且禁止覆盖；输出应放在
  `reports/rl-probe/` 等独立目录，不能放入已密封 attempt / artifact。
- probe 是**训练样本固定子集上的偏好诊断**。优势来自既有训练语义；
  它不能证明 held-out 泛化或下游成功率，绝对分支概率也受文本固有难度影响。
  应主要观察同一 probe 的跨 checkpoint 变化。
