# RL OOM 恢复与自适应激活卸载验证记录

日期：2026-11-14（Asia/Shanghai）。状态：代码及 CPU 验证完成，正式 GPU 验收延期。
工程：`/home/wangtianyi/DyConGlobal`，服务器 `10.249.187.9`。

## 问题与实现

历史 v10 完成 63 次更新，在第 64 步共享前缀 backward 重计算中发生 CUDA OOM。
该窗口最长 prompt 为 77,304 tokens；日志记录申请 1.18 GiB 时仅剩约 1.09 GiB。
具体失败张量的身份属于推断；reserved 与 allocated 的差值也不能单独证明碎片或泄漏。
历史证据、模型、数据和 replay 均保留。

本次通过原有 `dycon lora-train --training-config YAML` 实现：

- 在父进程及 worker 初始化 CUDA 前合并 allocator 环境，启用 expandable_segments，
  保留其他选项并拒绝冲突，分别记录请求设置、实际 backend 与支持警告。
- 当前 window 按 prompt/最大总序列长度判断风险，经 Gloo OR 决定全 step 的卸载模式；
  共享前缀和普通 checkpoint 路径共用 BoundaryOffload，保留局部重计算 hooks 隔离。
- 本地计算全部结束后先交换 OK/CUDA_OOM/FATAL，再进入梯度 collective。首次未卸载的
  可恢复 CUDA OOM 触发所有 rank 清理、回滚并恢复 RNG，卸载重算同一 window，最多一次。
- 成功更新才提交 streaming 游标；保留同一 window/reference、现有 producer 与有界队列。
- 每 rank 追加显存事件，统一阶段峰值采样，区分失败与提交，通过正式只读 CLI 汇总。

恢复范围不覆盖 collective 或 optimizer 内 OOM。首次已卸载仍 OOM、第二次 OOM、
FATAL、host pinned 预算耗尽或清理失败均终止。不跳过 group、不截断轨迹，不改变
loss、reference、shuffle、group/branch 数或学习率语义。既有跨进程 streaming resume
及其 checkpoint 校验保留；本次同进程重试没有取代该能力。

协议和配置详情分别见 [数据并行](DATA_PARALLEL.md#whole-step-memory-policy)、
[长上下文](LONG_CONTEXT.md#oom-recovery-configuration-2026-11-14)及
[显存事件](OBSERVABILITY.md#memory-and-oom-recovery-events)。

## 正式配置及当前执行状态

新配置：[v11-oom YAML](experiments/grpo-data-parallel-gpu01-pool128-kl-off-step-stream-fresh-20261113-v11-oom.yaml)。
attempt ID 为 `grpo-swe169-consensus-half-prefix-pool128-kl-off-step-stream-fresh-20261113-v11-oom`。

| 参数 | 值 |
| --- | --- |
| 启动、设备、更新数 | fresh；物理 GPU `[0,1]`；1694 updates |
| allocator | expandable_segments=true |
| activation_offload | auto |
| prompt / sequence 阈值 | 49152 / 49152 tokens |
| device_headroom_gib | 8.0，仅记录；没有容量预测公式 |
| host_budget_gib | 64.0 / rank |
| oom_retry | enabled=true，max_retries=1 |
| memory_telemetry | enabled=true，granularity=phase |

保留 KL=0、原 reference、seed=42、pool128 shuffle、4 groups×32 分支、
有界异步预取，以及 BF16 禁止低精度归约、允许 Split-K。49152 仅为候选阈值，
不是已验证的安全容量或最优值。`run.mode: smoke` 为正式实际执行枚举。

用户明确选择“保留现有任务，GPU 空闲后再验证”。本次未启动 v11 训练、独立 GPU
smoke、故意耗尽显存的程序或自动等待器，也未停止其他任务。最近检查的 GPU0/1
各使用 88212/97887 MiB；这是历史资源快照，启动前应重新查询。

## 已执行验证与证据

远程证据根目录：`temp/rl_oom_recovery_20261114/`。CPU 测试使用 `.venvs/rl`，
设置 `CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。

| 检查 | 结果 | 证据文件（相对证据根目录） |
| --- | --- | --- |
| 核心回归 | 69 passed，10 warnings，95.05 秒 | `focused-final.log` |
| 完整相关回归 | 299 passed，24 warnings，338.18 秒 | `cpu-regression.log` |
| 最终新增与受影响回归 | 68 passed，2 warnings，45.10 秒 | `final-additions.log` |
| 正式 CLI 配置 dry-run | validated | `v11-dry-run.yaml`、`v11-dry-run.json` |
| 历史 v10 只读显存汇总 | 通过；0 event files / 0 buckets | `legacy-memory-summary.json` |
| 原代码交付同步 | 23 文件哈希一致 | `change-manifest.json`、`delivery-integrity.json` |

测试集合重叠，不累加数量。warnings 来自测试临时 attempt 无法通过全局 observability
定位；pytest 通过。`compileall -q RL core/cli.py` 和 `lora-memory --help` 亦通过。
原 23 文件哈希是代码交付时快照，后续文档补充单独记录，不能用它证明后续版本未变。

测试覆盖真实 1/2/4 进程 Gloo 风险 OR 与重试、成功 rank 也重算、尾部空 rank、
FATAL 优先、清理失败和进程退出有界处理；覆盖 RNG/梯度/Adam/scheduler/buffer
等价、同一 window/reference、单次提交、checkpoint hooks 和阶段峰值聚合；另以真实
StreamingGroups 与 tiny Qwen 共享前缀验证 CPU 重试路径。

测试源码主要为 `tests/test_lora_oom_recovery.py` 与 `tests/test_grpo_prefix.py`。
CPU 故障注入不能证明真实 CUDA OOM 恢复、pinned 传输或显存节省。历史 v10 没有新格式
显存事件，查询为空不代表其未发生 OOM。完整任务报告另存于证据根目录 `REPORT.md`。

## GPU 空闲后的验收

通过正式训练 YAML 执行，不另开 GPU 测试程序；先重新检查 GPU0/1、主机 RAM 与磁盘。
本阶段尚未验证以下各项：

1. 两个 rank 的实际 allocator/backend、BF16 设置与物理/逻辑设备映射。
2. 普通 step 卸载字节为零；高风险 step 有真实 pinned D2H/H2D、host live/peak、
   预算与触发原因，并测量显存节省、传输成本和可比长度的性能。
3. 原第 64 步相同 group 顺序成功提交，并继续观察后续更长数据。
4. 自然发生可恢复 CUDA OOM 时，所有 rank 重试同一 window，最终只提交一次。
   若未发生，该项仍记为 GPU 恢复未实证，不故意制造 GPU OOM。
5. loss/梯度有限、参数副本一致、group/branch 与消费计数正确，正常/重试路径无死锁。

只读汇总使用 runner 环境：

```bash
timeout 30s .venvs/runner/bin/dycon lora-memory ATTEMPT_ID --storage-root /data3/wangtianyi/DyConGlobal --limit 30
timeout 30s .venvs/runner/bin/dycon lora-memory ATTEMPT_ID --storage-root /data3/wangtianyi/DyConGlobal --rank 1 --phase prefix_backward --limit 30
```

事件分别位于 attempt 根目录及 `distributed/rank-N/` 的 `memory-events.jsonl`。
`metrics.jsonl` 仍只包含已提交 update；CPU 预取数量不能作为 GPU 完成步数。
