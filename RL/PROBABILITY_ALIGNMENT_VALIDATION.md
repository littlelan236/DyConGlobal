# 概率产物对齐：实现与 CPU 验收

日期：2026-11-11（Asia/Shanghai）。项目：`/home/wangtianyi/DyConGlobal`。

统一 raw T=1 概率产物、coverage plan、missing-only replay、合并封存、共享 reader
和训练 cache 导入已完成实现与 CPU 验收。**最终联合回归：348 passed，6 skipped，
156.46 秒。** 本次未启动 GPU pilot、真实模型补评或业务 run；本报告不新增真实
模型数值一致性、吞吐或生产可用性的通过结论。

使用入口：[统一概率产物与迁移](dataset/PROBABILITY_ARTIFACTS.md)、
[严格配置示例](dataset/configs/probability-example.yaml)、
[训练配置](TRAINING.md)、[共享 reader 合同](PROBABILITY_STREAM.md)。
此前训练绑定的 203 项回归及真实 Qwen 同精度 synthetic GPU 结果保留在
[外部绑定验收](GRPO_BINDING_VALIDATION.md)，属于该次实现，不能代替本次 paired 验收。

## 实现范围

| 部分 | 已实现行为 |
|---|---|
| 正式 CLI | `rl-dataset probability-plan/build/check`，既有 replay lifecycle 和全局 `artifact show/verify` |
| 最终产物 | `rl-probability`，`ROOT/RL/probabilities/ID/`，schema 3，自包含 receipt-covered manifest/index/values/provenance |
| 冻结计划 | `rl-probability-plan`，`ROOT/RL/probability-plans/ID/`，冻结 dataset/source receipts、target/missing、策略与实现 |
| 来源迁移 | 显式 dataset stream、旧 raw replay、canonical 来源；online-only/offline-only/mixed 共用 writer |
| 身份 | 一个 scoring function；采集 compute dtype 必须相同，不同执行 cohorts 需精确显式允许 |
| 补评 | `selection.coverage_plan` 与 group selector 互斥，只向现有引擎提交 frozen missing branches |
| 发布与恢复 | 复用 lock/stage/receipt/原子提交；同输入显式 recover，committed ID 不覆盖 |
| 训练 | schema 3 接入既有 importer，冻结 final receipt/cohorts 与一份 shared anchor，resume 使用本地 cache |

`plan_sha256` 是 `plan.json` 的文件 SHA256；plan receipt hash 单独记录。
原 dataset branch/group probability flags、原始 run/request ledger/replay/receipts
不由合并回写。在线值保留 recorded origin；离线值保留 replay estimate 及
`historical_behavior_verified=false`，不冒充历史实时记录。

## 最终联合回归

在项目根目录使用专用 RL 环境，父子进程均隐藏 CUDA 以保持 CPU RNG 拓扑一致：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venvs/rl/bin/python -m pytest -q \
  tests/test_rl_probability_artifacts.py \
  tests/test_rl_dataset.py tests/test_rl_dataset_replay.py \
  tests/test_rl_dataset_raw_queue.py tests/test_rl_dataset_raw_native.py \
  tests/test_online_raw_logprobs.py tests/test_grpo_*.py tests/test_lora_*.py --tb=short
```

上述范围覆盖新增概率对齐合同、dataset/replay、在线 sidecar、GRPO objective/训练/
恢复和 LoRA 配置、监督、内存、训练及 checkpoint。6 个 skipped 均保留为未执行；
不计作 GPU/native 实机验收通过。正式 CLI help、新 scopes、配置示例解析和相关
`git diff --check` 也已检查。

新增 `tests/test_rl_probability_artifacts.py` 通过实际在线 sidecar writer 和 dataset
builder 构造固定 native tokens，使用正式 replay dispatcher、生产 ReplayWriter，
以 CPU trace engine 替换实际模型执行。缺失请求测试还覆盖原 continuous scheduler、
Prefetch 和 AsyncWriter；训练导入使用既有 cache writer，并设置 forward sentinel。
这些测试验证数据合同与请求选择，不声称实际 GPU logprobs 与 fixture 等价。

| 场景 | CPU 验证结果 |
|---|---|
| online-only / offline-only / mixed / canonical 再封装 | 同一 schema/scope/writer/reader；保留原始 float32 bytes |
| 全缺失、部分缺失、零缺失 | missing 集合正确；零缺失在 GPU reservation/worker 启动前返回 |
| 32 分支已有 31 条 | 只提交剩余 1 条 native trace；原 31 条值 hash 不变，最终 32 条完整 |
| 多组交错缺失、显式 group 子集 | 保留 target/mask/group/branch 顺序和整组 token inventory |
| continuous 调度与乱序完成 | 稳定 request/branch 身份对齐，复用实际异步 writer |
| 模型/tokenizer/adapter/overrides/dtype/cohort 差异 | 不兼容或未声明者拒绝；训练 dtype 选项不能绕过来源合同 |
| 错 plan、重复/额外/缺失输出、错 token/request/mask、坏 hash/locator | 拒绝发布；不变成普通 missing 静默补算 |
| recorded_failed、available 但漏 index | 失败原因需显式允许；提供过值却缺索引按腐败拒绝 |
| `-inf`、NaN、+inf、正 logprob | 合法 `-inf` 保留并报告，所需训练值拒绝；其余非法值拒绝 |
| partial 与训练资格 | 缺 eligible branch 导入失败；仅缺已按 token/reward 排除的 group 可导入，不改变 eligibility |
| 中断、恢复、同 ID 锁 | partial replay 只复用已封存 branch；canonical recover 绑定同输入，stage 不冒充 committed |
| 源 NAS 离线、artifact 搬迁 | canonical 本地读取/verify 和训练 cache 合同恢复通过；显式来源审计单独报告 incomplete |
| 旧来源和训练行为 | schema 2 仍可读，processed 拒绝作 raw；原 objective/shared-anchor/旧 cache 回归通过 |

`coverage_status: complete` 仅表示选定 branch 的向量齐全，
`finite_coverage_complete` 另外表达数值有限性。训练所需全部 eligible targets 仍必须有限。
普通本地 `probability-check`/`artifact verify` 不追溯原 NAS；显式 `--sources` 的审计
结果与本地验证结果分开。

## 失败材料与证据

证据位于 `temp/grpo_probability_artifact_alignment_20261111/`：

| 文件 | 内容 |
|---|---|
| `final-cpu-regression.log` | 最终 348 passed、6 skipped 的完整输出 |
| `regression-01.log` | 中间相关回归 126 passed、1 skipped |
| `grpo-regression-02.log` | 中间 GRPO 联合回归 136 passed、1 skipped |
| `final-regression.log` | 首次扩大回归超时及末尾失败标记，未计作通过 |
| `lora-diagnostic.log` | 既有独立进程 CPU 恢复测试的 CUDA RNG 拓扑差异诊断 |
| `IMPLEMENTATION_REPORT.md`、`changes.patch`、`change-inventory.json` | 初始实现说明、部署变更和哈希清单 |

首次扩大回归超过 210 秒时限；单独诊断确认父进程可见 CUDA、子进程显式隐藏 CUDA，
触发现有 `Checkpoint CUDA RNG topology differs` 校验。未修改或放宽该校验；统一
CPU 环境后上述最终联合回归全部通过。初始实现采用逐文件预期哈希和原子替换保护
其他未提交修改；本次文档同步也使用新下载的基线校验。

## 仍待真实模型验收

同一 checkpoint/tokenizer、相同 FP16 计算口径和固定 native tokens 下，仍需另行
授权 GPU 小样本，预先冻结代表样本与阈值，再报告在线/离线 max/mean/p95/p99
绝对 logprob 差异、delta/ratio/K3，覆盖 grammar、内部 EOS、cache 和代表性长度。
未设阈值只能报告测量值，不能宣称通过；不得事后放宽阈值或改写原始概率。

本次核对了 Qwen GDN graph 源码及 worker 的 requested/effective graph 模式和
actual FULL-dispatch 拒绝检查，保留现有执行要求；没有切换模式绕过限制。
GPU runtime 生产可用性、长上下文容量、端到端采集/补评性能及业务运行均未由
本次 CPU 结果证明。
