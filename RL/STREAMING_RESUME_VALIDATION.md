# Native streaming resume：2026-11-14 验收

项目：`/home/wangtianyi/DyConGlobal`，服务器：`10.249.187.9`。
使用说明见 [STREAMING_RESUME.md](STREAMING_RESUME.md)，多卡执行合同见
[DATA_PARALLEL.md](DATA_PARALLEL.md)。本页记录 streaming resume 的验收范围，
不替换早期 prepared/cache、概率评分或业务效果验收记录。

## 已验证的恢复行为

- fresh/resume 使用正式 `dycon lora-train --training-config YAML` 与同一服务。
  父 contract 决定数据路径；prepared 不因 async 开关变成 streaming。
- 新 streaming checkpoint 携带原 frozen plan、provenance、anchor tensor
  指纹、原 reference adapter 和 schema-1 recovery bundle，全部纳入 receipt。
  新 checkpoint 可脱离原父 attempt；损坏/不完整的新 bundle 不回退旧格式。
- 旧 streaming checkpoint 只从自身固定父 attempt 读取原辅助证据，原文件只读；
  failed 父 attempt 没有顶层 receipt 时，独立有效 checkpoint 仍可恢复。
- 恢复当前 adapter、optimizer、scheduler、各 rank RNG 与已提交 cursor；
  在 producer/forward 之前完成原 anchor tensor 校验和状态恢复。
  不重新计算原 advantage、顺序或 reference policy。
- 已消费 native/probability payload 不重读；预取窗口不推进已提交进度。
  剩余 payload 的 token/hash/mask/概率身份及 finite 检查不受历史可选 audit
  开关关闭的影响。异常和结束时检查源元数据并有界关闭 producer。
- 保留原数学 `contract.streaming.version: 1`；project source fingerprint
  只作审计信息，其余模型、依赖、数据、reference、schedule、数值与并行合同
  仍受约束。不支持 world-size 改变、CUDA RNG 列表重映射、路径自动迁移或
  streaming warm_start/append_resume。

## 测试结果

| 检查 | 结果 | 日志 |
|---|---|---|
| 计划指定 CPU 回归与新增 streaming 测试 | 160 passed，5 skipped，14 warnings；232.33s | `cpu-regression.log` |
| 补齐 TensorBoard 依赖后的 observability 回归 | 34 passed，0 skipped；28.60s | `observability-final.log` |
| probability artifacts / dataset replay 额外回归 | 72 passed；14.49s | `probability-regression.log` |
| 真实 Qwen3.5-9B，正式 CLI，单卡/双卡 native streaming | 2 passed；398.72s | `gpu-acceptance-01.log` |

测试集合有重叠，不相加为独立测试总数。CPU 的 5 个 skip 是当时缺少
TensorBoard；补齐后相关模块 34 项全通过。14 个 warning 来自隔离测试产物
未被 observability 唯一定位。测试依赖仅补入 `.venvs/rl`，未升级 runner/serving
环境或修改 lockfiles。

CPU 测试覆盖独立进程两次恢复、真正消费 Python/NumPy/Torch CPU RNG 的梯度、
producer RNG 隔离、脱离原父 attempt、legacy failed parent、新格式与源 payload
损坏拒绝、懒读取、完成或不前进的恢复拒绝、缺失 rank RNG/错误 CUDA 拓扑、
单进程/多进程窗口中途注入 OOM 后从最近 checkpoint 重试，以及队列满时关闭。

复核 CPU 命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout 720s .venvs/rl/bin/python -m pytest -q \
  tests/test_grpo_streaming_resume.py tests/test_lora_checkpoint.py \
  tests/test_grpo_checkpoint.py tests/test_grpo_lifecycle.py tests/test_grpo_service.py \
  tests/test_grpo_external_reference.py tests/test_grpo_prepared.py \
  tests/test_grpo_parallel.py tests/test_grpo_schedule.py \
  tests/test_grpo_source_fingerprint.py tests/test_rl_observability.py \
  tests/test_rl_observability_training.py > temp/streaming-cpu-review.log 2>&1
```

## 真实 GPU 短数据验收

物理 GPU 4、5；输入是工程化短 sealed native-token dataset 和 sealed external
raw probability fixture。单卡与双卡分别比较连续完成 4 个 update，与暂停于 1、
独立进程恢复至 2、再次独立进程恢复至 4 的结果。

比较包含中间及最终 adapter、optimizer、scheduler、各 rank RNG、已提交 cursor
和数学指标，沿用既有容差。双卡覆盖尾窗口某 rank 没有 group 的情况。
所有验收 attempt 的正式 `artifact verify --scope lora-attempt` 通过。
本次没有新增四卡 streaming GPU 通过结论。

需要复核 GPU 测试时，先选择实际空闲 GPU，并给证据目录一个新的路径：

```bash
DYCON_STREAMING_GPU=1 DYCON_GRPO_GPUS=4,5 OMP_NUM_THREADS=1 \
DYCON_GRPO_EVIDENCE=/home/wangtianyi/DyConGlobal/temp/streaming-resume-gpu-NEW \
  timeout 1500s .venvs/rl/bin/python -m pytest -q -s \
  tests/test_grpo_streaming_resume_gpu.py > temp/streaming-resume-gpu-NEW.log 2>&1
```

这验证状态继承与训练连续性，不建立真实历史长序列的显存容量、吞吐或模型质量结论。
短 fixture 每个 checkpoint 的恢复 bundle 增量为 21893351 字节；当前 adapter
另为 21712976 字节。历史 v10 的必要恢复辅助证据约 268080540 字节，不能将小
fixture 的开销直接当作历史任务每次 checkpoint 的开销。

## 历史 v10：只读验证

父 attempt 与 checkpoint：

```text
/data3/wangtianyi/DyConGlobal/RL/checkpoints/grpo-swe169-consensus-half-prefix-pool128-kl-off-step-stream-fresh-20261113-v10
checkpoints/step-000060
```

| 字段 | 验证结果 |
|---|---|
| `global_step` | 60 |
| `group_cursor` / `consumed_groups` | 240 |
| `consumed_branches` | 7680 |
| `consumed_tokens` | 1492544 |
| 原 reference anchor | 496 个实际 tensor 校验通过 |
| `reference_identity.adapter_hash` | `25f96abd3763f1117961f15788aaa8b0b8ffa31acace2712ae4986479092d8b6` |
| RNG 拓扑 | 2 个 rank，各含 1 个可见 CUDA 状态，无需重映射 |
| Runner dry-run | 通过；阻止 Torch 导入的 hook 未触发；保留完整 `max_updates: 1694` 及原 preflight |
| Status | failed；最后完成 update 63，可恢复 checkpoint 为 step 60 |
| 历史目录不变性 | 原 74 个文件全部哈希不变，无新增/删除 |

只读预检另核验 checkpoint receipt/state/payload metadata、冻结 plan 与原 source
binding。基础模型注册文件路径/大小匹配；本次没有重新逐字节哈希全部约 19 GB
权重，不将此表述为完整权重重哈希。

**没有执行 v10 的 update 61 或其后更新。** 原任务长上下文 CUDA OOM 仍可能
发生；短 fixture GPU 验收、元数据 dry-run 和独立 anchor tensor 检查不能替代
v10 实际 forward。保留原完整 schedule；需要执行时使用新 attempt ID，可设置
`stop_after_update: 61`，不能把 `max_updates` 改为 61 或剩余次数。

## 剩余外部输入

冻结计划共 6773 个 eligible group，checkpoint 已提交 240 个，剩余 6533 组，
每组 32 个 branch，共引用 209056 个 native request ledger。它们是含原始
prompt/completion token ID 的 JSON 文件，与 reference logprob 数组、hidden
activation tensors 分开存储。

共同根前缀为 `/VM/wangtianyi/DyConGlobal/hidden/online/`，文件在各根目录的
`requests/<hash>.json` 下：

| 根目录后缀 | 剩余引用文件数 | 字节数 |
|---|---:|---:|
| `verified80-consensus32-loopescape-gpu23-tp2-w4-s64-fresh-20261026` | 57472 | 11636424302 |
| `verified80-consensus32-resolved106plus331-gpu03-tp4-w4-s64-20261027` | 59360 | 9738639882 |
| `verified80-consensus32-resolved106plus331-gpu45-tp1-w4pg-s64-20261102` | 61952 | 10460708020 |
| `verified80-consensus32-resolved106plus331-gpu45-tp2-w4-s64-20261103` | 30272 | 4722050254 |
| 合计 | 209056 | 36557822458 |

全部文件存在、已有证据 SHA256 均存在且匹配，缺失和 hash mismatch 均为 0。
约 36.56 GB / 34.05 GiB，只统计剩余计划引用的文件，不是各根目录全量大小。

Dataset：`/data3/wangtianyi/DyConGlobal/RL/datasets/swebench-consensus32-resolved169-four-runs-20261110-v1`。
其中 `groups.jsonl`/`branches.jsonl` 是分组和 ledger 引用，不包含完整 token payload。

Reference raw 概率源：`/data3/wangtianyi/DyConGlobal/RL/replays/swebench-resolved169-raw-t1-fp16-gpu012-tp1-20261111-v2`。
剩余引用的 209056 个概率数组文件共 157925920 字节，逐文件存在性及哈希检查通过。
基础模型为 `/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B`。
完整迁移还需 dataset/replay 元数据、checkpoint 与原恢复证据；不能将 ledger
和概率数组之和当成完整迁移预算。本次没有自动搬迁数据或重写路径。

## 证据索引

本次实施证据在服务器以下任务目录，不写入历史 attempt：

```text
/home/wangtianyi/DyConGlobal/temp/rl_streaming_resume_impl_20261114/
```

- `gpu-acceptance-summary.json`、`gpu-acceptance-01/world-{1,2}/acceptance.json`：
  正式 CLI attempt 路径、GPU 拓扑及连续/两次恢复验收。
- `v10-inspection.json`、`v10-dryrun.json`、`v10-status.json`、
  `v10-anchor-tensors-validation.json`、`v10-immutability.json`：历史只读检查。
- `native-dependencies-complete.jsonl`、`native-dependency-paths.json`、
  `probability-dependencies.jsonl`、`dependency-inventory-validation.json`、
  `migration-dependencies-summary.json`、`external-inputs-inventory.json`：逐文件
  依赖、大小、哈希与统计口径。
- `v10-resume-dryrun.yaml`：已执行的示例，不可复用其 attempt ID。
- `v10-resume-update61.yaml`：准备的实际运行配置，尚未执行；保留原 schedule，
  在 61 暂停。GPU 列表为 0、1，使用前须核对实际空闲资源和新 attempt ID。

`temp/` 是本次现场证据位置，不是可复用的训练 launcher。长期行为、配置与
恢复逻辑仍由项目正式 `RL/` 模块和统一 CLI 实现。
