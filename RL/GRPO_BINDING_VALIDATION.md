# GRPO 训练侧外部 raw logprobs 绑定实现

日期：2026-11-11（Asia/Shanghai）。远程项目：`/home/wangtianyi/DyConGlobal`。

训练侧实现已接入统一 `dycon lora-train --training-config YAML`。保留
`offline_grpo_k3_v1` 公式、一次 epoch、冻结 eligibility/优势/顺序和实时 current
前向。一份冻结 cache 同时绑定 ratio 分母和 K3 reference。

## 代码与接口

- `RL/config.py`：严格区分 `model_forward`、`probability_artifact`、
  `dataset_probability`；外部来源必须显式选择 dtype 兼容规则。旧配置保留原路径，
  默认字段不注入旧 schedule 合同。
- `RL/dataset/probability.py`：复用 receipt、路径和 float32 shard reader；读取在线
  dataset stream，适配离线 agent 的 native replay v2 元数据，不修改生产 writer。
- `RL/grpo/reference.py`：共用最终 writer；按训练顺序导入，不做 reference 模型
  前向。核对实际 fresh LoRA 零增量和 warm-start adapter 身份。
- `RL/grpo/contract.py`：外部 cache manifest v2；绑定 source receipt、原始 manifest
  与必要元数据、scoring identity、dtype 规则和共享 cache hash。旧 manifest v1
  保留独立验证路径。
- `RL/service.py`：fresh/warm_start 来源分派、无 Torch 外部 dry-run、首次 optimizer
  更新前合同校验、checkpoint 本地恢复，以及显式小样本 initial-current 诊断。
- `RL/GRPO.md`、`RL/PROBABILITY_STREAM.md`：配置、数据接口和生命周期说明。
- `tests/test_grpo_external_reference.py`、`tests/test_grpo_external_gpu.py`：合同、
  两端真实 producer schema、错误拒绝和正式 CLI GPU 生命周期验收。

不创建双份 anchor，不覆盖 current，不将训练角色写回源产物，不更改其他 agent 的
collector/replay 生产代码、根配置或运行任务。所有写回均限定为本任务文件；已有文件
采用预期内容哈希检查，保留其他未提交修改。

## 自动化结果

合并回归：**203 passed，1 skipped**。范围覆盖 GRPO、相关 LoRA config/checkpoint/
training/entrypoint、dataset、replay 与在线 raw stream。跳过项为需显式启用的 GPU
验收，另行执行。

已验证：

- 外部导入的 model/reference forward sentinel 不被调用；current 仍实时计算。
- 实际在线 sidecar writer → dataset builder，以及离线 ReplayWriter 产物，经训练
  导入得到相同向量；完成顺序不改变训练顺序。
- 源超集可用；缺分支、重复、无绑定额外记录、错 request/token/mask/hash、processed、
  非 float32、NaN/±inf、错误模型/tokenizer/adapter、模型覆盖配置均拒绝。
- FP16→BF16 必须显式配置；该选项不能绕过 adapter 身份检查。fresh 零 B 与非零 B、
  warm-start 匹配与不匹配都有测试。
- 中断写入不产生完整 cache commit；来源漂移失败，故障材料保留。
- 删除外部来源后恢复仍成功；anchor hash 和连续训练的最终 adapter 一致。
- 既有 objective 数值/梯度、聚合、尾窗口、单 epoch、旧缓存和旧配置回归通过。

## GPU 验收

**正式 CLI GPU 验收通过：1 passed，229.41 秒。** 本任务使用真实 Qwen3.5-9B、
5 个合成 group、15 个 branch、60 个 completion target；先比较 BF16 同精度导入，
不把该结果冒充真实 FP16 collector/BF16 历史数据验收。

物理 GPU 1，产物根 `/data3/wangtianyi/DyConGlobal/RL/checkpoints/`：

| Attempt | 结果 |
|---|---|
| `grpo-binding-2d5a653370-model-forward` | completed，2 updates |
| `grpo-binding-2d5a653370-external` | completed，2 updates |
| `grpo-binding-2d5a653370-paused` | paused，step=1 / cursor=4 |
| `grpo-binding-2d5a653370-resumed` | completed，step=2 / cursor=5 |

四个 attempt 均通过正式 `dycon artifact verify --scope lora-attempt`。基线、外部导入
和恢复的最终 adapter fingerprints 完全一致；optimizer/scheduler/RNG 一致。loss、
ratio、clip、K3、梯度 norm/覆盖和 cursor 在 `atol=rtol=1e-7` 下逐 update 一致。
恢复时外部 raw 产物已移离配置路径，anchor hash 仍为：

`24ebee1c745af18b840e64f322bf4a660f85bddb6758a503da241df8ce4f0832`。

| 单次合成测量 | model_forward | 外部导入 |
|---|---:|---:|
| reference 准备时间（含快照/IO） | 4.496 s | 0.695 s |
| 首次 optimizer update | 7.501 s | 6.256 s |
| 首次 update peak GPU allocated | 19,763,780,096 B | 19,763,780,096 B |
| 首次 update peak GPU reserved | 19,864,223,744 B | 19,843,252,224 B |
| 进程 CPU VmHWM（包含模型加载） | 17,378,455,552 B | 17,389,559,808 B |
| 完整训练 CLI 墙钟 | 53.587 s | 57.548 s |

external 显式开启 2 个 branch 的初始 current 诊断；同精度结果 delta=0、ratio=1、
K3=0，未校正 anchor。完整 CLI 墙钟还包含模型身份扫描、加载和诊断，不能拿它
代替 cache 物化耗时，也不能将这一微型数据的单次结果当成真实长序列吞吐结论。
上述结果没有混入 token/KV cache 的性能收益。

前两轮故障材料保留：第一轮在加载模型期间发生并发源码变化，exact resume 被正确
拒绝；第二轮 GPU 0 被其他 vLLM 任务占用，训练入口在更新前拒绝启动。未绕过校验，
未取消其他任务。最终使用空闲 GPU 1 和新 attempt ID，一次完成，无 source drift
重试。验收后 GPU 1 显存回到 0 MiB，所有本任务 SSH 命令已非交互退出。

## 后续 schema-3 概率对齐验收

后续实现将在线/离线并集封存为 `rl-probability`，由同一 importer 冻结最终
receipt 和全部 execution cohorts。保留本报告的 schema-2 来源和历史 GPU 验收记录；
本次新 schema 的 CPU 联合回归为 **348 passed，6 skipped**，涵盖 missing-only
replay、原在线 bytes 不变、partial eligibility、来源离线的本地验证和 cache 恢复。
没有为本次对齐启动 GPU pilot、业务 run 或 paired 真实模型数值验收。
独立证据、复现命令和范围见
[PROBABILITY_ALIGNMENT_VALIDATION.md](PROBABILITY_ALIGNMENT_VALIDATION.md)。

## 正式数据边界

真实 FP16 raw artifact/在线 stream 的具体 ID、完整 receipt、eligible selection 及
长序列容量仍需在来源封存后绑定。当前没有启动真实历史数据的正式第一轮训练。
同精度合成 GPU 验收与 CPU dtype 规则测试，不替代真实 FP16/BF16 数值差异测量。

远程证据目录：`temp/grpo_training_logprob_binding_20261111/`。本地同名任务目录保存
补丁、源文件快照、测试日志和 GPU 摘要；正式使用方式见远程 `RL/GRPO.md`。
