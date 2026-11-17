# GRPO 数据并行：完整父组 1 / 2 / 4 GPU，实验性 8×16

新增显式 `runtime.topology_policy: preserve_updates`：完整 parent 与原 two-pass 计划均可在 2/4/8 张物理 GPU 间双向恢复，保持原 update windows、权重、anchor 和 optimizer 状态。默认 exact 模式仍遵守本文的原卡数限制。详见 [跨卡数恢复](TOPOLOGY_SWITCH.md)。

默认 `exact` 策略下，`dycon lora-train --training-config YAML` 支持单机 1、2、4 张物理 GPU 的同步
LoRA 数据并行。每张 GPU 保留完整 base 与 adapter，计算单位是完整 group。
本功能适用于现有 `offline_grpo_k3_v1`，包括 `prefix_reuse.mode: boundary_grad`；
单条轨迹的容量仍由单卡显存决定。

## 配置与启动

旧单卡配置 `runtime.gpu: 0` 保持兼容。多卡时，用 `runtime.gpus` 替换
`runtime.gpu`，或将 `gpu` 显式设为 `null`：

```yaml
runtime:
  gpu: null
  gpus: [1, 2]             # 也支持 [1]、[0, 1, 2, 3]
  distributed_timeout_seconds: 3600
  output_root: /data2/wangtianyi/DyConGlobal
grpo:
  group_batch_size: 4     # 全局每个 optimizer update 的 group 数
  prefix_reuse:
    mode: boundary_grad
training:
  gradient_accumulation_steps: 1
```

示例只展示相关字段，其他训练、数据、reference 配置保留原完整 YAML。
默认 exact 完整父组模式的 GPU 列表长度为 1、2、4，不能重复；`gpu` 与 `gpus` 必须二选一。
多卡暂不接受 MIG UUID 或 causal CE 配置。通信超时涵盖初始化、rank 0
准备 reference 期间的等待、训练同步和最终收尾；长时间 reference 构建需要
相应增大该配置，超时范围为 30–86400 秒。

1. 在 `RL/experiments/` 中建立任务配置，选择新 attempt ID 与空闲 GPU。
2. 设置 `run.mode: dry-run`，通过 `.venvs/runner/bin/dycon lora-train
   --training-config PATH.yaml` 检查配置。dry-run 不导入 Torch、不加载模型；
   `memory_plan.local_world_size` 来自显式 GPU 列表。
3. 设置 `run.mode: smoke`，通过 `.venvs/rl/bin/dycon lora-train
   --training-config PATH.yaml` 执行实际训练。沿用项目现有 mode 命名。

service 获取全部 GPU 锁并内部启动 workers。每个 worker 只看到自己的一张 GPU。
不要额外套用 `torchrun`，也不要启动多份独立训练命令来模拟数据并行。

## 一次更新的含义

每步仍为全局 4 个 group、每组 32 条分支，共 128 条。GPU 数不放大全局 batch：

| GPU 数 | 各 rank 在常规窗口中计算的 group 序号 | 全局每步分支数 |
|---|---|---|
| 1 | rank 0: 0,1,2,3 | 128 |
| 2 | rank 0: 0,2；rank 1: 1,3 | 128 |
| 4 | rank 0/1/2/3: 各自对应一个 group | 128 |

窗口、group 顺序、组内分支、冻结 advantage 和 reference 都遵循冻结的 prepared
artifact 或 native streaming plan/source binding。每卡执行原来的 `begin_group → branch backward → finish_group →
close_group`，保留组内共享前缀前向和合并前缀反向。

损失仍按 token mean → branch mean → **全局实际 group mean** 聚合：

```text
L = (1/G) Σ_g (1/N_g) Σ_b mean_t(loss_gbt)
```

各卡只计算自己 group 的上述贡献，全部完成前缀反向后，以固定参数顺序将
LoRA 梯度转换为 FP32、分桶 SUM all-reduce。此处不再除以 GPU 数。
随后各副本对相同梯度裁剪，分别执行相同的 AdamW/scheduler step，形成一次逻辑更新。
每步校验所有副本 adapter 指纹相同，再推进全局 cursor 和发布指标。

尾窗口按实际 group 数 G 归一化，全部 group 必须消费；无 group 的 rank
仍参与同步。全局均未使用的参数保持 `grad=None`，避免意外触发 AdamW
weight decay。共享前缀边界与计算图完全保留在所属 GPU，不跨卡传送。

## 产物、恢复和异常

- rank 0 唯一发布全局 metrics、checkpoint 和 receipt。Prepared 路径由统一
  preparation 服务创建 indexed prepared/reference artifact，各 rank 读取同一
  不可变产物。Native streaming fresh/resume 无需全量 prepared/reference cache；
  各 rank 使用同一冻结 plan、advantage、原 reference anchor 和 source binding，
  从已提交 cursor 准备剩余窗口。async=true 不会把 prepared resume 转为 streaming。
- checkpoint 保存一份相同的 adapter/optimizer/scheduler、全局 cursor，以及
  **每个 rank** 的 Python/NumPy/CPU/CUDA RNG。只在完整更新且梯度清空后发布。
- 默认 `exact` 的 `resume` 要求相同 world size 和执行协议；可换同数量的物理 GPU，但必须
  保持每个 rank 的 CUDA 可见设备数与保存时一致，不做 RNG 设备列表重映射。
  各 rank 在启动 producer/forward 前恢复 optimizer/scheduler 和自己的 RNG。
  旧单卡 checkpoint 与多卡 checkpoint 不能直接互换。
- `append_resume` 仅支持原 prepared 路径，同样要求原 world size。Streaming
  checkpoint 只支持 exact resume，不支持 warm_start/append_resume。其他支持
  的非 streaming adapter warm-start 仍重置 optimizer/cursor 并重新绑定 anchor；
  不能用它冒充继续旧步数或保留旧 initial-policy anchor。
- 单卡与多卡数学目标一致；浮点累加顺序不同，不承诺跨 GPU 数逐位相同。
  BF16 adapter 对照验收固定使用 `atol=2e-4, rtol=2e-3`；这只描述跨设备
  归约顺序造成的数值差异，不放宽 loss、梯度有限性或全局 group 计数检查。
- 非有限梯度、漏算/重复 group、副本不一致或 worker 异常会终止当前任务。
  失败窗口不发布可恢复 checkpoint，从最近完整 checkpoint 重新执行。
  supervisor 负责停止、回收其他 workers 并释放全部锁；通信也有显式超时。
  SIGTERM/SIGINT 触发清理，SIGKILL 由操作系统强制终止，无法执行 Python 清理。

`metrics.parallel` 包含 world size、梯度同步时间和每卡 group/branch/token 数、
计算时间、显存峰值。顶层 loss/ratio/K3 和 `prefix_reuse` 汇总全局贡献；
分位数从实际 token 数据重建。现有 profiler、forward/backward 分段和顶层
显存指标对应 rank 0，在 `parallel.local_telemetry_rank` 明确标注。

查询及核验继续使用 `lora-status`、`lora-tensorboard` 和
`artifact verify ATTEMPT_ID --scope lora-attempt --storage-root ROOT`。

## 验证

CPU 多进程验证通过 Gloo 运行真实 1/2/4 rank，将全部梯度及更新后参数与独立
串行 oracle 比较，覆盖 4×32、尾窗口、空闲 rank、全局/局部 unused 参数、
weight decay、裁剪、暂停恢复、非有限梯度和缺失贡献。

```bash
CUDA_VISIBLE_DEVICES='' .venvs/rl/bin/python -m pytest -q tests/test_grpo_parallel.py
```

真实 Qwen3.5-9B / NCCL / CLI 验证需要显式选择空闲 GPU：

```bash
DYCON_GRPO_PARALLEL_GPU=1 DYCON_GRPO_GPUS=1,2 \
DYCON_GRPO_EVIDENCE=/home/wangtianyi/DyConGlobal/temp/data-parallel-validation/gpu \
  .venvs/rl/bin/python -m pytest -q -s tests/test_grpo_parallel_gpu.py
```

四卡可将 `DYCON_GRPO_GPUS` 改为四个空闲物理 GPU。验收包含串行比较、连续多卡、
暂停/完整恢复和统一 artifact verify。测试使用短 synthetic token fixture，
不能替代真实长上下文、4×32 生产 workload 的显存容量和吞吐验收。

Native streaming 的单卡/双卡正式 CLI 连续、暂停、独立进程两次恢复验收于
2026-11-14 通过，含尾窗口空闲 rank；未新增四卡 streaming GPU 通过结论。
历史 v10 仅完成只读兼容、anchor/RNG/依赖核验，未执行 update 61。见
[流式恢复指南](STREAMING_RESUME.md) 与 [验收记录](STREAMING_RESUME_VALIDATION.md)。

## Whole-step memory policy

Deployment status (2026-11-14): CPU state-machine and real 1/2/4-process Gloo
checks pass; the v11 fresh configuration passes CLI dry-run. Production GPU
acceptance is deferred while existing GPU workloads are preserved. Earlier short
streaming-resume GPU tests do not establish long-context OOM recovery. See
[OOM validation](OOM_RECOVERY_VALIDATION.md) for evidence and pending checks.

Before each window, each rank evaluates prompt and sequence length thresholds;
the control group ORs the result. A high-risk window uses checkpoint-boundary
offload on every rank. A local CUDA OOM is exchanged as a CPU control status
before gradient collectives; when the first attempt used device-only activations,
all ranks clear the window and retry once with boundary offload. FATAL errors and
OOM after gradient synchronization are not retried.

The OOM boundary ends before the first gradient collective. Workers exchange
checked `(inflight_step, step_attempt, phase)` Gloo envelopes containing
`OK / CUDA_OOM / FATAL`. Any FATAL takes precedence over retry. A local attempt is
a function scope whose result contains scalars and metrics, not exception objects.
Before retry, every rank clears all gradients, closes generators/prefix state,
rolls back attempt coverage, restores registered model buffers, collects garbage,
synchronizes the CUDA context and empties the allocator cache. A second Gloo
acknowledgement confirms cleanup on every rank. Python/NumPy/Torch CPU/CUDA RNG is
restored before recomputing the same yielded window and reference values. The
single streaming producer is retained and its queue remains bounded to two windows.

Already-offloaded attempts do not retry an identical policy. CPU MemoryError,
host pinned-budget exhaustion, nonfinite values, data errors, device errors and
collective failures are terminal. Optimizer/scheduler errors, including OOM after
partial optimizer mutation, are terminal and never enter the retry loop. There is
no checkpoint, cursor or scheduler advance for failed attempts. Recoverable local
OOMs do not create `worker-error.json`; terminal failures retain the existing worker
supervision and timeout path. Production `distributed_timeout_seconds` stays 3600
in the new configuration, allowing skew from long groups.

This feature is same-process retry. Existing independently implemented streaming
checkpoint-resume support and its identity checks remain intact. Registered model
buffers (including rotary buffers) are snapshotted for a retry-capable step; native
shared-prefix cache objects remain forward-local. No parameter/Adam snapshot is
created because retry is forbidden after the update boundary.


## Experimental 8-GPU distinct-parent 8x16

The existing whole-parent mode described above remains the default. In `exact` mode, eight GPUs
require the explicit 16-branch, two-pass protocol; see [the configuration,
cursor, normalization and acceptance boundaries](DATA_PARALLEL_8X16.md).
Every regular update selects one half from each of eight different parents;
parent scoring still uses all 32 branches. The two round tails stay separate.

## 2/4/8 卡双向恢复

`preserve_updates` 将来源逻辑计划与物理 GPU 数分开：完整 parent 计划保持
原窗口和 `1/(K*N)` 分支权重，每个 parent 拆成最多 16 条分支的物理任务；
原 two-pass 计划保持 halves、rounds、窗口与 `1/(16*K)` 权重。
每个窗口的全部任务结束后，仍只执行一次 SUM 归约、裁剪、AdamW 和 scheduler
更新，不按新 world size 再平均。恢复后 checkpoint 继续记录来源逻辑合同，
另存实际物理拓扑，允许再次切回 2/4/8 卡。

完整配置、RNG 限制和操作见 [跨卡数恢复](TOPOLOGY_SWITCH.md)，
13 通过、4 GPU 项跳过及真实 v10 dry-run 证据见
[验证记录](TOPOLOGY_SWITCH_VALIDATION.md)。这些结果尚未验证真实 GPU 提速或长上下文容量。

## 8×16 validation status (2026-11-14)

The explicit experimental [8-GPU distinct-parent protocol](DATA_PARALLEL_8X16.md) has passed CPU/Gloo regression: **426 passed, 12 skipped**, including real eight-process synchronization and tiny-Qwen prepared/streaming resume. See [validation details and pending acceptance](DATA_PARALLEL_8X16_VALIDATION.md). The skips do not count as GPU acceptance; real Qwen3.5-9B eight-GPU correctness, long-prefix capacity, throughput and quality remain pending.
