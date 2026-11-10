# GRPO 工程实现与测试报告

日期：2026-11-10，Asia/Shanghai。远程工程：`wangtianyi@10.249.187.9:/home/wangtianyi/DyConGlobal`。

代码阶段 Phase 0–3 已完成。已额外完成真实 Qwen3.5-9B 单卡 synthetic smoke 和独立进程暂停/恢复验收。真实历史数据的 Phase 4 尚未执行。

## 已实现

- 使用现有 `dycon lora-train`、`RL.service.run`、单 GPU bf16 Qwen3.5-9B 和全部 text-backbone linear LoRA；保留 CE 与稀疏监督路径。
- 严格配置 `offline_grpo_k3_v1`，固定 raw T=1、clip=0.1、beta=0.02、initial-policy reference、一次 epoch。
- 按用户修订取消 split：扫描指定 dataset 的全部 group，记录 eligibility 排除原因，冻结符合条件的完整 group、branch 顺序、优势与 shuffle 顺序。
- 历史 score 越小越优；连续 top-K 正优势和负优势余项保持零和。缺分数、全同分及边界近似并列明确排除。score 不解释为成功概率，也不做指数变换。
- PPO clipped ratio 与 token K3 使用同一份 selected target logprob；reference/优势无梯度，当前策略保留梯度，数值与聚合使用 fp32。
- selected logprob 复用现有 text backbone 与 CE 因果 target 坐标；分块完整词表归一化，在 backward 重算投影块，保持第一枚 completion target 和完整 native prefix。
- 物理 microbatch 为单个完整 branch；按 branch token mean → group branch mean → update group mean 归一化，最后不足窗口按实际 group 数计算；dataset 不回绕、不补样本。
- 保存 initial adapter snapshot、reference manifest 和 detached fp32 target cache；绑定模型/tokenizer、dataset receipt、group/branch、prompt/completion、mask、objective/probability mode 与 adapter tensor fingerprints。
- checkpoint 携带原始 reference snapshot/cache 与指标前缀；resume 严格复用它们，恢复 optimizer/scheduler/RNG 与 group/branch/token cursor。完成 epoch 后拒绝 resume，warm-start 冻结新的 reference 并从零计数。
- 扩展 receipt 和 artifact verify：核验 objective/reference/cursor，并从已封存 contract、metrics 和 final state 重建 `grpo-report.json`。
- 保留梯度覆盖、梯度裁剪、非有限检查、CUDA/host memory、time 和投影/MLP/offload 指标；增加覆盖率、排除原因、优势分布、ratio 分位数、clip fraction、K3 和正负分支 logprob 变化。
- 更新 `RL/README.md`、`RL/TRAINING.md`、`docs/ENTRYPOINTS.md`、smoke/task YAML，新增 `RL/GRPO.md`。

## 自动化测试

| 测试范围 | 结果 |
|---|---:|
| 全部 `tests/test_grpo_*.py`，最终 CPU 环境 | 70 passed |
| 既有 LoRA config/training/checkpoint/memory/entrypoint | 68 passed |
| 既有 sparse supervision，CPU 环境 | 79 passed |
| dataset/native-runtime/replay，runner 环境 | 44 passed, 8 skipped |
| 正式 CLI synthetic dry-run | validated |

这四组测试合计 261 passed、8 skipped。跳过项是 native-runtime 集成项；没有以跳过作为真实 GPU 验收依据。

GRPO 测试覆盖纯函数和严格 schema、N=2/3/9/17/32、分数异常与 ties、K3 数值/导数、clip 方向、dense/chunked logprob 与真实 tiny Qwen+LoRA 梯度、完整 prefix/首 target 坐标、可变 branch 长度、尾窗口归一化、单 epoch 不回绕、非有限更新拒绝、完整服务生命周期、精确恢复、warm-start、身份/缓存篡改拒绝、无 Torch dry-run 和 artifact/report 重建。

合并回归曾超过一次远程命令时限，随后分组完成。稀疏监督独立进程测试要求 CPU RNG 拓扑一致：父进程可见 CUDA、子进程禁用 CUDA 时失败；将该组放入 `CUDA_VISIBLE_DEVICES=` 的 CPU 环境后全部 79 项通过。未更改既有测试断言来回避失败。

## 正式 Qwen3.5-9B GPU 验收

使用正式 `.venvs/rl/bin/dycon lora-train --training-config YAML`，物理 GPU 0，明确的 synthetic fixture，所有 attempt 产物位于：

`/data3/wangtianyi/DyConGlobal/RL/checkpoints/`

| Attempt | 结果 |
|---|---|
| `grpo-offline-k3-smoke-20261110-gpu0` | completed，连续 2 updates |
| `grpo-offline-k3-pause2-20261110-gpu0` | paused，update=1，group cursor=4 |
| `grpo-offline-k3-resume2-20261110-gpu0` | completed，恢复后 cursor=5 |

三项均通过正式 `dycon artifact verify ID --scope lora-attempt --storage-root /data3/wangtianyi/DyConGlobal`。

连续训练与暂停/恢复对比：

- 最终 adapter tensor fingerprints 完全相同。
- optimizer、scheduler、RNG state 完全相同。
- 关键 loss、K3、clip、ratio/分位数、正负 logprob 变化、梯度 norm 与 cursor 完全相同。
- 恢复复用原始 reference cache 字节，未使用更新后的 adapter 重算 denominator。
- 最终 `global_step=2`、`group_cursor=5`、`consumed_groups=5`、`consumed_branches=15`、`consumed_tokens=60`。
- 实际 update 的 group 数为 4、1；248 个 LoRA 投影均有非零 B 梯度。
- 峰值 allocated 19,763,780,096 bytes，峰值 reserved 19,866,320,896 bytes；此容量仅适用于长度 6–8 的 synthetic 序列。
- update 2：`k3_penalty_mean=0.02374385545651118`，`clip_fraction=0.7166666686534882`。

第一次暂停 attempt `grpo-offline-k3-pause-20261110-gpu0` 也成功封存，但其 resume 因另一个任务修改 `core/cli.py` 导致 source fingerprint 改变，在预检阶段被正确拒绝；未创建该 resume 的训练 attempt，也未覆盖旧 checkpoint。随后使用上述 pause2/resume2 完成验收。

本地证据位于 `execution/evidence/`，含三项 receipt、metrics、attempt、validation、report 和 `gpu-equivalence.json`。远程比较结果另存于 `temp/grpo-objective-20261110/gpu-equivalence.json`。执行配置在 `RL/experiments/grpo-objective-20261110/`；模板默认仍为 dry-run。

## 边界与后续

Phase 4 尚未完成：未构建或启动真实历史 dataset 训练，未验收真实长序列显存容量、业务 resolved rate 或 estimator 校准。synthetic 工程验收不能代表这些指标。

checkpoint 绑定实现源码和依赖身份；并发编辑 `core/cli.py` 等绑定源码后，旧 checkpoint 的精确 resume 会拒绝，这是现有工程契约的延续。

验收阶段保留远程仓库原有未提交修改，未执行 reset/revert 或变更已有 rollout run 的冻结证据。代码、文档与必要 RL 依赖按用户要求整理为 `feat：RL - GRPO` 提交；其他任务改动保留在工作区。所有 SSH 为带超时、有限输出的非交互命令；训练命令已结束。验收后 GPU 0 显存使用为 0 MiB。

机器专用配置 `RL/experiments/grpo-objective-20261110/` 与生成的 dataset/cache/replay/extraction、checkpoint、日志和 profiler 产物由 `.gitignore` 排除。可复用模板、synthetic fixture、源码和精选 Markdown 验收报告随代码维护。
