# 关键 step 分支轨迹 GRPO

本功能在 RL 侧读取 `continuation-groups-v1` 采集封存，通过现有 `rl-train` / `lora-train` 服务训练；不运行容器、不重新采集轨迹，不要求同步其他服务器的源码。

部署位置：`10.249.189.84:/home/wangtianyi/DyConGlobal`。采集功能部署在 `10.249.187.9`，其本机说明为 `docs/unsorted/KEY-STEP-COLLECTION-CN.md`。采集使用 B=8、默认自适应/随机/生成熵三方法等数量对照、纯前缀重放以及 shuffle 调度；本训练入口消费其封存产物。

## 分组与奖励

一个母轨迹中的一个关键 step 对应一个 parent，原始 8 个候选各自单路续跑至结束，构成完整 GRPO 组。不会按 task 合并不同关键 step，不会丢弃重复动作分支，也不会在训练时重抽分支。三种选点方法分别准备数据，重叠 step 可出现在各方法独立的数据集中。

原始奖励如下，随后仅在此 8 成员组内计算 population std，并用 `(reward-mean)/(std+1e-8)` 得到 advantage：

| 轨迹结局 | 原始奖励 |
| --- | --- |
| 明确的推理/动作循环 | -3 |
| 有效 unresolved，或已接受的正常终止但未提交 | -1 |
| resolved | +1 + 长度附加分 |
| 基础设施错误、缺失/无效评测或缺失 native 证据 | null，整组不训练 |

循环优先于其他结局，但只有显式 `reasoning_loop`、`loop_detection` 或 `repeated_reasoning_action_loop` 证据才能判循环。普通 `LimitsExceeded` 不自动算循环，避免把步数/预算耗尽误标为 -3。collector 对 submitted 的基础奖励必须来自项目已有有效 harness settlement。

长度是**整条轨迹的生成 token 总数，包含母轨迹公共前缀中实际执行路径的生成 token、关键 step 原候选以及后续所有模型输出**。工具输出、用户输入和重复发送的 prompt 不计入，也不把母轨迹未执行的其他候选算进公共前缀。每个输出以原生 output_token_ids 计数一次，不使用累计 usage.total_tokens。

只在同一组的 resolved 轨迹间取最短 Lmin 和最长 Lmax：

`附加分 = 2 * (Lmax - L) / (Lmax - Lmin)`

因此最短 resolved 总奖励 +3，最长 resolved 总奖励 +1。多个 resolved 全部同长度时统一附加 +1，即总奖励 +2；只有一条 resolved 时附加 +2，即总奖励 +3。unresolved/循环轨迹长度不影响 resolved 的归一化范围。

零方差组保留审计并跳过。数据准备重新计算奖励与 advantage，不继承采集侧旧的二元奖励 advantage 或旧 zero_variance/train_eligible 标志。任一方法中的所选 step 缺分支/证据/有效结局时，同一母轨迹三种方法共同失去配对资格，保留排除原因。

## 公共前缀与梯度

公共 H_t 始终作为模型输入。它的 labels 为 -100、loss_mask 为 0，因此不产生直接监督 loss；前向没有 no_grad，也没有 detach 历史 KV 来切断参数梯度。关键 step 及后续生成 token 的 loss 可以通过 attention/GDN 传播到处理公共前缀的可训练 LoRA 参数。

逐轮使用真实的 native prompt/output IDs，工具观察和其他上下文都不作为目标。只有 native token 序列严格满足 append-only 关系时，才复用现有 prefix_trace 逻辑合并相邻轮次；若聊天模板改变或上下文不再是精确前缀，就拆成完整上下文的独立前向，不拼造 prompt。拆分保留完整历史和梯度，长度上限仅控制合并，不截断单个轮次。

归约为：每条 child 内生成 token 平均 → 8 条 child 等权 → update 中的关键 step parent 等权。长分支不会仅因 token 多而占更大的组权重。多卡仍先计算全局逻辑权重，再通过既有 SUM 梯度归约；分配到各卡的是物理前向单元，原始 8 分支分组不改变。

准备产物每个 parent 只保存一份公共 prompt，child 第一轮引用公共 prompt，后续轮次使用原有 token 增量编码。训练公共历史在内存中恢复，不重复落盘完整 H_t。

## 正式命令

先让本机可读取采集封存及其引用的母轨迹、child 轨迹、decision、native ledger 和 raw T=1 概率产物。这是数据依赖，不需要其他服务器的 Python 源码。当前 importer 按封存中的路径读取证据，移动数据时应保留可访问路径；不会篡改源 receipt 或重写冻结 hash。

采集端固定的 `grpo_ready=false` 指其旧训练入口未接入该 schema，不阻止本机新导入器处理有效封存。不要手动把它改成 true。新导入器以完整 8 分支、native 证据和有效结局重新判定资格，并重算本说明中的奖励、长度及 advantage；缺失证据依然拒绝，不能因已有二元奖励而补齐。须提供包含三种方法关系的完整封存，以便按同一母轨迹实施配对排除。

```bash
.venvs/runner/bin/dycon rl-prepare-key-steps \
  --groups COLLECTION_GROUPS.json --method adaptive \
  --output RL/datasets/key-step-adaptive

.venvs/runner/bin/dycon rl-prepare-key-steps \
  --groups COLLECTION_GROUPS.json --method random \
  --output RL/datasets/key-step-random

.venvs/runner/bin/dycon rl-prepare-key-steps \
  --groups COLLECTION_GROUPS.json --method entropy \
  --output RL/datasets/key-step-entropy

.venvs/runner/bin/dycon rl-train \
  --training-config RL/experiments/key-step-grpo.yaml
```

样例默认 dry-run。先准备数据，再设置 YAML 的 dataset、新 attempt_id、模型、GPU 和输出目录。正式运行把 mode 设为 smoke，仍使用相同 CLI。`training.max_updates: null` 按完整、非零方差 parent 数与 epochs 推导；groups_per_update 指关键 step parent 数。三个方法独立运行并比较，不混成一个采样池。产物目录不可覆盖，输入变化时使用新目录。

当前集成 LoRA，支持现有 1/2/4/8 卡 exact topology。full/FSDP2、append_resume 和 preserve_updates 不属于这个新 schema。fresh/resume/warm_start 走既有服务，resume 保留 parent shuffle 游标、优化器、scheduler、RNG 和初始 KL reference；不同分组、奖励、数据身份、模型数值设置或 GPU 数量会拒绝续训。资源和长上下文显存策略继续使用项目已有配置。

PPO 分母沿用现有离线轨迹训练器的 native raw T=1 selected-token logprob；它不是带 temperature/top-p 的实际历史行为策略概率。KL 可复用冻结训练初始策略并重新计算；本次不改变既有 PPO/KL 语义。

## 产物

`manifest.json` 冻结 selector、原始 parent/branch 身份、完整生成长度、循环/未解决/解决结局、原始奖励、长度附加分、组统计、advantage、零方差和配对资格。prefixes/ 保存每组公共输入，trajectories/ 保存各 child 的模型轮次与概率。训练 checkpoint 和 attempt receipt 复用项目现有格式，并增加 `key-step-contract.json` 与 `key_step_advantage` 指标。

工程测试覆盖真实 native 文件格式的导入、奖励、token mask、前缀梯度、逐轮数值 oracle、两进程 SUM、真实小型 Qwen+PEFT 优化器更新和 checkpoint 恢复。小模型/GPU 验收不等于正式 9B 长轨迹训练的吞吐和显存容量验收。

## 已验证范围

实现阶段相关回归 135 passed；GPU 上真实小型 Qwen+PEFT 更新和精确恢复 2 passed，覆盖 KL=0 与 KL=0.02。正式数据准备、dry-run、暂停/恢复和 artifact verify 接口已验证。

真实 Qwen3.5-9B 在 GPU4 通过正式 rl-train 完成一次更新、暂停并恢复至第二次更新，累计 2 组、16 分支、104 个监督 token，两个 attempt 的 artifact 校验通过。该验收使用合成短轨迹 fixture，仅验证训练和恢复链路。

尚未用 187.9 实际批量采集的长轨迹完成端到端训练，也未据此验收显存容量、吞吐、三方法奖励效果或任务成功率。样例仍默认 dry-run，正式训练需选择真实数据及匹配资源。
