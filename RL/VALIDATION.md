# LoRA 基础能力实施与验收报告

日期：2026-11-10。状态：计划要求的实现与测试已完成。

## 交付位置与 Git 状态

- 服务器：`wangtianyi@10.249.187.9`；原工程：`/home/wangtianyi/DyConGlobal`。
- 已整理 `.gitignore` 并提交现有 SkillsBench verifier 修改：`d4e74e6ab4027727ba9dc74ffb3417d96412e436`，提交说明严格为 `debug：skillsbench verifier`。提交前 89 项相关测试通过。
- 开发分支：`feature/lora-foundation`。LoRA 实现保留在该分支工作区，未额外提交、合并或推送；未建立 worktree 或第二份工程。
- 正式源码：大写 `RL/`；独立环境：`.venvs/rl`，链接到 `/data3/wangtianyi/DyConGlobal/RL/environment/rl`；依赖和缓存位于 `/data3`。
- 8 个受保护基线文件逐字节 SHA 校验保持一致，包括根 `config.yaml`、SkillsBench 实现及其原有测试；共享入口与文档在原内容上追加/最小修改。
- 验收运行时源码指纹与最终源码一致：`8fad0eca7b66375dc4947b5d85266fc445e19c606ce57b72bd82c95698303df5`。HEAD、dirty 文件摘要、实际源码哈希、依赖及输入身份均已写入训练产物。

## 已实现

- 正式 `dycon lora-train --training-config YAML`，提供 `dry-run` 与固定工程夹具 `smoke`；CLI 与 Python API 共用 `RL.service.run`。
- Transformers conditional-generation loader、单卡 bf16、SDPA、gradient checkpointing、PEFT LoRA；不使用自动 device map、CPU offload 或跨卡加载。
- 文本 backbone 全部 `nn.Linear`：248 个投影，包括 96 个 MLP、32 个 full-attention、120 个 linear-attention。rank=16 时可训练参数为 43,278,336；所有目标均实际执行、获得有效梯度并发生更新。
- vision、embedding、lm_head 和全部基座参数冻结。实际实例化基座参数为 9,409,813,744；checkpoint 中未实例化的 15 个 `mtp.*` 权重完整列入产物清单，未以名称过滤实际 backbone。
- 有效监督 token 加权梯度累积、最后不足窗口提交、有限性校验与梯度裁剪；测试 CE 与通用训练循环分离，causal shift 只执行一次。
- `fresh`、完整 checkpoint `resume`、adapter `warm_start`；保存/恢复 optimizer、完整 scheduler、Python/NumPy/CPU/CUDA RNG、global step、microbatch/token 游标。恢复严格绑定 base/tokenizer、targets/shape、schedule、输入及源码/依赖身份，无静默回退。
- 标准 PEFT adapter 导出、完整 checkpoint 同盘原子发布、最后提交 receipt、不可覆盖 attempt、并发锁、失败状态/traceback保留，以及 `artifact show/verify --scope lora-attempt`。

## 测试结果

40 项 CPU 测试、1 项真实 9B 完整 GPU 集成验收、95 项既有功能回归均通过，共 136 项。新增代码 Ruff 与编译检查通过，`git diff --check` 通过。

CPU 测试使用真实微型 Qwen3.5 混合 attention 模型与 PEFT，覆盖全目标注入、所有基座参数逐项不变、mask/shift、变长监督与不足累积窗口、非有限数值拒绝、损坏/不兼容 checkpoint 拒绝、导出重载、禁用 adapter、含 dropout 的 RNG 恢复及 warm-start 重置。另有超过 PEFT 自动压缩阈值的目标列表导出/重载回归测试。

95 项回归覆盖 `test_active_run_status.py`、`test_run_entrypoints.py`、`test_skillsbench_verifier_audit.py` 和 `test_difficulty_artifacts.py`。runner 的帮助、config-check、无 Torch/PEFT 的 LoRA 产物校验及非法配置退出码 2 已验证。共享 `core/cli.py` 原有 6 个 lint 问题与提交基线相同，未借此修改其他任务逻辑。

所有 GPU 验收经正式 CLI 在物理 GPU 0 上顺序启动新进程；单卡同时只常驻一份 9B。

| 验收项 | 序列长度 | 最终累计监督 tokens | allocated 峰值 GiB | reserved 峰值 GiB | 每 update 秒数 |
|---|---:|---:|---:|---:|---|
| 连续 4 步 | 512 | 3996 | 19.797 | 20.303 | 1.844, 1.265, 1.405, 1.264 |
| 同一 schedule 暂停于第 2 步 | 512 | 2001 | 19.796 | 20.303 | 1.831, 1.251 |
| 新进程恢复到第 4 步 | 512 | 3996 | 19.797 | 20.793 | 1.744, 1.290 |
| 新进程重载并 warm-start 1 步 | 512 | 1004 | 19.474 | 20.303 | 1.725 |
| 1024 token 容量验证 1 步 | 1024 | 2028 | 21.031 | 22.234 | 2.145 |

连续 4 步与 2+2 恢复的最终 adapter 最大绝对差异为 **0.0**，loss、学习率、global step 及输入/token 游标一致。warm-start 更新前全部 adapter 参数指纹与源导出相同，新进程输出重载通过预设 logits 容差；optimizer 与计数从零开始，后续更新有效，来源未改变。GPU 进程退出后 GPU 0 占用为 **0 MiB**；未停止其他服务，SSH 命令均已退出并关闭连接。

验收前固定容差：adapter atol=2e-5/rtol=2e-3，logits atol=0.125/rtol=0.01，loss atol=0.05/rtol=0.01。没有在失败后放宽容差。运行启用确定性 Torch/cuDNN、关闭 cuDNN benchmark/TF32，并固定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；相关设置绑定在恢复契约中。

环境：Python 3.12.15、CUDA 13.0、PyTorch 2.13.0、Transformers 5.17.0、PEFT 0.21.2、Accelerate 1.15.0、safetensors 0.8.0。linear attention 与 causal-conv1d 使用已验证 backward 的参考 PyTorch 路径，未安装可选优化内核。

## 产物与入口

最终成功验收 ID 前缀：`lora-accept-20261110-122152-`，后缀为 `baseline`、`paused`、`resumed`、`warm`、`capacity-1k`。产物根为 `/data3/wangtianyi/DyConGlobal/RL/checkpoints/`。

远程完整验收报告：`/home/wangtianyi/DyConGlobal/temp/lora-foundation-20261110/acceptance-20261110-122152/report.json`。对应目录保存验收前容差、冻结 YAML、各进程日志与验证结果。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/rl/bin/dycon lora-train --training-config RL/experiments/lora-smoke.yaml
.venvs/runner/bin/dycon artifact verify lora-accept-20261110-122152-baseline \
  --scope lora-attempt --storage-root /data3/wangtianyi/DyConGlobal
```

完整说明见 `RL/README.md`；依赖输入/哈希锁为 `locks/rl.in`、`locks/rl.txt`。首次新训练应使用尚不存在的 attempt ID；继续已有权重应显式选择 `resume` 或 `warm_start`。

## 本次范围与限制

交付的是可验证的 LoRA 训练基础能力。固定 synthetic CE 仅用于工程验收；未实现 RL/GRPO/PPO、reward、业务 dataset、SWE-bench 训练/评测、多卡、量化或 vLLM adapter 服务。实际验证长度为 512/1024 token，不能据此宣称 32K/262K 训练容量或模型解题能力提升。文件哈希验证与质量评测严格区分。
