# GRPO 8 卡 8×16 数据并行改造计划

日期：2026-11-14
目标工程：`/home/wangtianyi/DyConGlobal`
状态：工程实现计划；本文档不代表实现、训练质量或性能验收已完成。
本次修订：每步的 8 个 half-group 必须来自 8 个不同 parent group；32 条负责评分，16 条负责执行。

## 目标与边界

为 `offline_grpo_k3_v1` 增加显式启用的 8 卡执行模式。常规 optimizer update 消费 8 个不同原生 parent group 各自的 16 条轨迹，总计 128 条轨迹；8 个执行分片分别分配到 8 个 rank。不能把 4 个 parent group 的两半安排在同一 update。

原生 parent group 保留完整 32 条轨迹，是 eligibility、评分、retained、排名、冻结 advantage 和 reference 身份绑定的逻辑单位。half-group 只是执行分片，不是新的评分 group。所有统计和接口必须区分父组、执行分片和轨迹数。

同一父组的 32 条轨迹确定性随机拆成两个互补的 16 条分片，两半在不同 update 消费；一个 epoch 中每条轨迹恰好消费一次。提高每步 parent prompt 的覆盖是本方案的训练动机，不能保证下游效果必然优于 4×32。

本次范围包括 `boundary_grad`、v10 使用的 native step-streaming 和 indexed prepared 路径。现有 1/2/4 卡完整 group 模式及旧 attempt 的恢复协议保持兼容。暂不扩展 causal CE、在线 rollout、MIG、多节点或任意分片尺寸。

## 现状与证据边界

- `RL/config.py` 当前只允许 `runtime.gpus` 长度为 1、2、4。
- `RL/parallel.py` 按完整 group 分配 rank；update 末尾对 LoRA 梯度执行 FP32 SUM all-reduce，随后统一裁剪和更新。
- 现有 `grpo.group_batch_size` 表示每步完整父组数；loss 按 token mean → branch mean → group mean 聚合。
- 完整 32 条的 advantage 由现有唯一实现计算并冻结；不得为 half-group 另建评分实现。
- v10 的 follower `GroupObjective.prefix_metrics` 缺少逐 update reset/drain，历史记录会重复出现在后续 `prefix_reuse` 中。必须先核对生产代码与 rank 归属，再恢复每步实际记录；不能直接累加全部数组。
- v10 有 6,773 个合格父组，旧 4×32 计划为 1,694 次 update，最后提交到 step 63，随后发生 prefix backward 重计算 CUDA OOM。分片不改变 prompt 长度，且当前后端逐 branch 累加 boundary gradient，缓冲形状由边界状态决定，不能假定从 32 减到 16 会令显存减半或消除 OOM。

此前基于重复 prefix 行累计的比例，以及沿用该比例的提速预测，不能作为验收结论。文档和报告应明确区分历史实测、校正后的归因和未实测估算。

## 配置与协议

建议增加以下显式 opt-in 字段；实现时可按代码惯例调整名称，但含义必须一致：

```yaml
runtime:
  gpu: null
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
grpo:
  group_batch_size: 8                 # 新模式：每个常规 update 的不同父组数
  parallel_branch_chunk_size: 16
  parallel_branch_schedule: distinct_parent_two_pass_v1
  branch_partition_seed: 42
  prefix_reuse:
    mode: boundary_grad
```

新模式中 `group_batch_size: 8` 表示从 8 个不同父组各取一个分片，不能读为消费 8 个完整 32 条父组。旧模式缺省不启用分片，`group_batch_size` 继续表示完整父组数量。配置帮助、合同与状态必须明确这种区别，不能只改数字而沿用旧 batching。

校验要求：

1. GPU 数允许 1、2、4、8；本次 8 卡必须同时启用上述 16 分片及 distinct-parent 协议。
2. 新模式固定使用 8 个不同物理 GPU、`group_batch_size: 8`、`boundary_grad`，所有合格父组恰有 32 条 branch，且至少有 8 个合格父组。结构性数据检查复用 preparation，在模型加载前完成；不得用跳过 optional audit 绕过必需校验。
3. 拒绝 8 卡完整父组模式、分片配 1/2/4 卡、非 boundary-grad、其他分片尺寸和不满足父组数量/分支数的组合；错误需给出具体原因。
4. 冻结 split seed、划分算法、调度算法、GPU world size、loss 权重协议、window/tail 边界及其 hash。所有 ranks 从同一计划读取，不能按当前配置重新抽样。
5. 缺省字段保持旧行为；旧 checkpoint 的合同比较按旧协议处理，不因新默认字段改变历史身份。

## 分片与评分不变量

先对完整 32 条执行现有 eligibility、全局组内排名及 score-to-advantage，再建立执行分片：

- 基于全局 partition seed、稳定 parent ID 和版本化算法派生父组自己的随机排列，避免依赖 Python 的进程随机 `hash()` 或遍历顺序。
- 排列前 16 个索引为 half 0，其余 16 个为 half 1；每个 half 内按 native ordinal 排序执行，并保存原 ordinal。两半合并后按 ordinal 能恢复原始 32 条顺序。
- 分片身份为 `(parent_group_id, half_index)`，保留原 source group ID、branch IDs、native token spans、scores、retained、frozen advantages 和 reference；不修改 dataset schema 或源 artifacts。
- 两半 branch 集合不交叉，合并恰好为父组全部 32 条；每个 branch 的 score、advantage、reference 数值保持不变。
- 禁止按 score 排序后直接切成前后半、对 half 重新 top-k/top-half、重新计算标准差/cutoff、局部归一化 advantage 或强制 half 的 advantage 和为零。

随机 half 的 advantage 和可能非零，这是正常的执行采样现象。保留原 32 条 advantage 的零和合同即可，不能为抵消单步偏移修改训练信号。

## 调度：两轮消费，单步父组不重复

使用版本化的 `distinct_parent_two_pass_v1`，由现有统一 preparation 服务生成不可变执行计划：

1. 第一轮：每个合格父组仅消费 half 0，各出现一次。half 0 的成员本身已由确定性随机划分产生。
2. 第二轮：每个父组仅消费互补 half 1，各出现一次。
3. 每轮复用现有 prefix-length pool 的排序/成窗逻辑，以 8 个不同 parent 构造常规窗口；轮次顺序、轮内 seed 派生和候选池算法全部冻结。
4. 两轮各自的尾窗单独保留，不跨轮合并，避免同一 parent 的两个 half 因边界拼接进入同一步。尾窗禁止重复、补样或丢弃轨迹。
5. 每个窗口有 K 个执行分片，来自 K 个不同 parent，常规 K=8，尾窗 1≤K≤7。按冻结 window ordinal 分配 rank；无分片的 ranks 仍参与同步。
6. 不把两个 half 的间隔内参数变化称为 exact-equivalent execution。相比原来的完整父组 batching，这是明确的新训练顺序和新优化轨迹。

若合格父组数为 G，新模式总分片数为 `2G`，计划 update 数为 `2*ceil(G/8)`，不是直接沿用旧模式的 `ceil(G/4)`。二者对部分 G 会不同，必须从冻结窗口推导 `max_updates`。

对 v10 的 G=6773：每轮 846 个满窗加一个 5-parent 尾窗，共 847 个 update；两轮合计 1694 个 update，13546 个 half，216736 条 branch。两个尾窗各 80 条轨迹，其余满窗各 128 条。原方案的最后一个 32 条尾窗不能继续套用。

## 数学目标、权重与训练语义

令 `J[p,b]` 为使用原父组冻结 advantage/reference 的 branch token-mean loss。父组均值和 half 均值分别为：

```text
L_parent[p] = (1/32) Σ(b in parent p) J[p,b]
L_half[p,h] = (1/16) Σ(b in half h of p) J[p,b]
L_parent[p] = (L_half[p,0] + L_half[p,1]) / 2
```

新模式常规 update 的目标为：

```text
L_update = (1/8) Σ(p in 8 distinct parents) L_half[p,selected_half]
         = (1/128) Σ(128 selected branches) J[p,b]
```

尾窗按实际 K 个不同父组归一化：`(1/K) Σ L_half`，每条 branch 权重为 `1/(16*K)`。实现不能沿用旧的 `1/(32*8)`，否则梯度会缩小一半；也不能固定除以 8 而忽略尾窗。all-reduce 采用已归一化贡献的 FP32 SUM，不再除以 world size。

与旧方案相比，每个常规 step 都是 128 条、每条权重 1/128，但来自不同 prompt。保持原 branch 评分、advantage、reference 和 epoch 无遗漏覆盖，不意味着每步梯度、AdamW 状态或最终参数相同。在固定模型参数下，两半均值的平均等于完整父组均值；实际两轮之间会更新参数，不能据此宣称整个训练与 4×32 等价。尾窗权重遵循各自实际窗口，也不能声称跨方案的所有轨迹训练权重完全一致。

训练动机是每步覆盖 8 个 parent prompt，降低共享 prompt 分支间梯度相关性；效果需要同数据、同轨迹预算的训练对照验证。distinct parent ID 不保证 prompt 文本唯一，不额外改变原数据分布以强制 prompt 去重。

## 数据流与实现改造

1. **分片表示**：在现有 GRPO 数据层提供统一轻量 helper，只接受已冻结完整父组及计划中的 branch ordinal 子集。切 branch 对齐的 masks、scores、advantages、IDs 和 reference，保留父组元数据；prepared 和 streaming 共用它。
2. **Preparation/schedule**：保持源完整父组及其 receipt 不变，新增分片清单、两轮窗口、映射和调度 hash。新执行 schedule 与旧父组 schedule 是不同身份，不冒充旧 prepared 顺序/hash 未变化。沿用现有 prepared/streaming 计划发布、注册和验证机制，扩展 schema，避免第二套 launcher/数据准备接口。
3. **Streaming**：完整父组的 metadata/评分先冻结，native payload 可按需读取；每轮只为本 rank 的 16 条加载 completion/reference，共同 prompt 按现有接口读取。同一父组在两轮会再次读取，必须记录 I/O 与队列成本。仍绑定源 fingerprint，未消费输入在 exact resume 时必须可访问。
4. **Prepared**：保留原 indexed 完整父组 artifact，执行计划通过原 branch ID/ordinal 选择本步 half；不得将 shard 当成重新评分后的完整组或原地改写 artifact。
5. **训练循环**：以执行窗口迭代，使用 `1/(16*K)` 权重；每步仅一个 optimizer/scheduler update。半组执行独立 `begin_group → branch backward → finish_group → close_group`，本步结束后清理 prefix 图及临时梯度。
6. **同步校验**：`finish_window` 核对本步 K 个预期 half 身份、K 个不同 parent、每个 half 的 16 个 branch 及顺序，无漏算、重复、跨窗贡献。完整 epoch 再校验每个 parent 的两半覆盖，而不是要求本步两半齐全。保留 gradient presence、unused 参数、weight decay、裁剪与副本指纹校验。
7. **指标**：本步 loss 按 half mean 汇总，原 parent 的完整统计只能读取冻结 metadata 或在明确的跨步覆盖视图中展示；不能把不同时刻的两半模型 loss 拼成同一次完整父组计算。
8. **生命周期**：继续由 `RL.service.run` / `dycon lora-train --training-config YAML` 管理锁、8 workers、失败清理和 rank 0 发布；不增加独立内部 worker 入口。

建议核对：`RL/config.py`、`RL/grpo/groups.py`、`RL/grpo/data.py`、`RL/grpo/schedule.py`、`RL/grpo/streaming.py`、`RL/grpo/prepared.py`、`RL/grpo/backend.py`、`RL/grpo/metrics.py`、`RL/training.py`、`RL/parallel.py`、`RL/service.py`、`RL/checkpoints.py` 及 streaming resume 模块。实现前检查并保留已有未提交变更。

## Cursor、checkpoint 与恢复

同一父组可以只消费一半后形成合法 checkpoint，因此旧的“父组前缀已完整消费”cursor 不能复用于新模式：

- 使用版本化的执行 schedule cursor，绑定完整窗口边界、轮次、已消费 half 数、half ID 顺序 hash、branch/token 计数及计划 hash。
- 分别报告 `planned_parent_groups=G`、`planned_execution_chunks=2G`、`consumed_execution_chunks`、`distinct_parent_groups_seen` 和 `completed_parent_groups`。后者只在两半都提交后增加。
- 新 schema 不将 half 数塞进旧的 `group_cursor`/`consumed_groups`；对不适用字段提供明确兼容投影或缺省，并同步修改 status、TensorBoard 和 artifact reader。
- checkpoint 仅在完整 optimizer update、梯度清空后发布；允许部分 parent 尚有另一半未消费。依靠冻结计划与已提交执行 cursor 确定剩余分片，无需保存 prefix 图或 boundary gradient。
- 保存各 rank RNG、adapter、optimizer、scheduler 和新协议；恢复严格核对 parent/branch 映射、分片计划及消费前缀。async 预取不推进已提交 cursor，失败窗口的全部贡献从最近 checkpoint 重做。
- 8×16 exact resume 要求相同 world size、划分、调度及 reduction 协议，可换同数量物理卡；旧 v10/1/2/4 卡 checkpoint 不能直接切换到新协议。换协议只能另建明确的新阶段，例如 warm-start，并遵守真实起点的 reference/anchor 绑定，不能声称延续原 optimizer 或原始 anchor。

## 计时、显存与可观测性

- 修复 update-scoped prefix telemetry，使用明确 drain/reset，正常和异常路径都不保留旧 update 的待发布行。不能仅在序列化处截断数组掩盖累计。
- 每条分片遥测携带 parent ID、half index、native branch ordinals、window/round、rank、prefix length、prefix/suffix forward/backward、state-copy、boundary-accumulate、计时方法和显存观测。
- 顶层报告本步不同 parent 数 K、执行分片数 K、branch 数 `16K`、原 parent branch 数 32、各 rank compute/token/memory、gradient sync 和端到端 update seconds，不能把 half count 冒充完成父组数。
- prefix/suffix 为同步分段累计时间；跨 rank 相加不能当作 step wall time。性能用端到端时间和 rank critical path，历史去重必须保留证据和归属规则。
- 当前逐 branch 执行，32→16 减少串行 branch 工作，但 prefix graph、单 branch leaf、累加 boundary-gradient buffer 的形状通常不减半。分别验收 prefix backward 峰值、suffix 峰值及常驻 base/adapter。

## 验证矩阵

### CPU 正确性与调度

- 完整父组的 32 条 score、retained、advantage、reference 与分片映射逐项一致；两半无交集且按 native ordinal 恢复完整顺序。
- 同 seed/parent ID 的划分跨进程稳定，不受遍历顺序影响；不同 seed 可形成不同合法分片。覆盖错序、重复 branch、31/33 分支、非finite 和缺失 reference。
- 每个常规窗口恰有 8 个不同 parent，每轮各 parent 一次，每 epoch 各 half/branch 一次；两个尾窗不跨轮合并。覆盖 G=8、9、12、13、16、6773 的窗口数、计数及尾窗。
- 在固定参数下，验证两个 half 的均值等于完整父组 loss/gradient；另外对新模式一次窗口，用串行选定的 8×16 branch 集合作 oracle，对比 loss、全部 adapter 梯度、裁剪、AdamW 和 scheduler。不能用另一个 4×32 窗口要求逐步梯度相同。
- 验证常规 `1/128`、尾窗 `1/(16*K)`，并覆盖 half advantage 非零和、非零 K3、不等 completion、padding/mask。

### Gloo 多进程

- 扩展 `tests/test_grpo_parallel.py`，真实运行 8 ranks、8 个不同 parent 各一个 half，与同一选定分片的串行 oracle 对比全部梯度和更新后参数。
- 覆盖 1–7 个 half 的尾窗及 idle ranks；重复 parent、错误 half、缺失/重复 branch、unused 参数、weight decay、裁剪、非finite、指纹不一致及失败清理。
- scheduler 每窗只推进一次；单步父组数、已完成父组数和分片计数分别正确，不能以“本步同父两半齐全”为通过条件。

### Streaming、prepared、resume 与回归

- 两条数据路径使用相同映射与权重；source artifacts 不变，新执行计划有独立身份。reference 仅加载本 rank 所需 branch，第二轮不会重算 advantage。
- 在第一轮中途、第一轮尾窗、两轮交界、第二轮中途及最终尾窗 checkpoint/resume，与同协议连续训练比较 adapter/optimizer/scheduler/RNG/cursor 和覆盖 hash。
- 覆盖父组只提交一半时恢复、预取超前、源变化/缺失、计划篡改和 topology 不匹配，均不得跳样、重复提交或静默 fresh。
- 连续多个 update 的 prefix 遥测仅包含本步 half；合并 ranks 后身份恰好一致，无历史累计。保留 1/2/4 卡旧合同和旧 checkpoint 恢复回归。
- 扩展 config/service/checkpoint/status/observability/artifact 测试，确保 8-worker 锁与清理、字段含义及新 schema 都有覆盖。

### GPU 端到端

- 通过正式 CLI 在 8 张空闲物理卡运行 Qwen3.5-9B 短 GRPO 验收，覆盖跨轮、尾窗、连续与暂停恢复、8 rank 指纹、artifact verify 和故障回收。
- 单独验证 v10 代表性长 prompt 的显存容量，短 synthetic 成功不替代超长 prefix 验收。
- 输出放在项目 `temp/data-parallel-8x16-acceptance/`；全部使用新 attempt ID，不修改 v10 attempt，不另建 launcher。

## 性能与训练效果评测

需要区分两个对照，均固定代码版本、模型/dtype、source/reference、完整 32 条冻结评分、seed 和总轨迹预算：

1. **拆分性能**：4 卡 4×32 对比 8 卡“同 4 个 parent 两半同一步”的受控执行测试，用于解释 prefix 重复与 suffix 分摊；后者仅用于诊断，不作为本计划正式生产调度模式，不增加未经约定的配置组合。
2. **正式方案**：4 卡完整父组 batching 对比 8 卡 distinct-parent 8×16，两者遍历同一完整数据集，但单步 parent windows 必然不同。分别冻结计划，报告完整计划耗时或相同累计 branch 数下的吞吐，不声称每步输入完全相同。

第二项是交付必须提供的真实性能结果。报告 steady-state median/P90 update seconds、总轨迹数/尾窗、branches/s、总时长、rank critical path、sync、分段计时、I/O/reference 开销、峰值 allocated/reserved、失败率；启动和 checkpoint 成本单列。至少重复两次短性能验收，并覆盖长短 prefix，避免只选有利窗口。

估算上，单父组 prefix 成本为 P、32 条 suffix 成本为 S，均衡条件下 4 卡每卡约 `P+S`，8 卡每卡约 `P+S/2`；state-copy/accumulate 按实际 branch 数另计。正式方案每步处理 8 个不同 prefix；同一 parent 的 prefix 在两轮各重算一次，epoch 总 prefix 执行次数由 G 增为 2G。最长 rank、父组长度分布、half completion 总长度、collective 和数据读取决定实际提速，不能承诺线性扩展或直接沿用旧估算百分比。

性能建议门槛：相同硬件规格、同数据覆盖下 steady-state branches/s 至少提升 10%，且没有 correctness、OOM、恢复或 artifact regression；尾窗不同须单独说明，不能只比较 median step 时间。未达到门槛时保持显式实验开关，不设默认、不宣称提速。

训练效果另做同总轨迹预算的对照，比较共同评测集上的任务表现及多 seed 波动；loss 曲线下降或 prompt 覆盖增加本身不能证明质量更好。可记录梯度范数/波动、每步 parent 覆盖及 half advantage 和作为诊断，禁止据此改写冻结 advantage。

## 文档与实施顺序

实现后同步 `RL/DATA_PARALLEL.md`、`RL/SHARED_PREFIX.md`、`RL/GRPO.md`、`RL/README.md`、streaming resume/observability 文档，以及仍列出 1/2/4 卡限制的根说明。覆盖新配置、父组/half 计数、两轮尾窗、权重、恢复兼容性、真实性能及未验证项。

建议实施顺序：

1. 修正 update-scoped telemetry，核对 v10 历史记录的 rank 归属和去重证据。
2. 实现确定性分片、两轮 distinct-parent 计划及串行 loss/gradient oracle。
3. 冻结配置、执行身份和新 cursor schema，接入 prepared/streaming 的统一数据接口。
4. 接入 8-rank 分配、贡献校验、尾窗、checkpoint/RNG 恢复和 observability。
5. 完成 CPU/Gloo、跨轮恢复、config/lifecycle/旧模式回归。
6. 通过正式 CLI 完成 GPU correctness、长 prefix 显存和吞吐验收；记录训练效果验证边界后交付。
