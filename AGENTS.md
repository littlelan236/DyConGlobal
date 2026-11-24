# DyConGlobal 工作约定

先读 `README.md` 和 `docs/MINIMUM-RUN-CN.md`。容器问题查阅 `docs/BENCHMARKS-CN.md`。历史 run 证据只覆盖当时的配置、实现和数据，当前行为以对应命令和代码为准。

## 统一入口与行为对齐

- 正式任务使用 `core.cli`（安装后的 `dycon`）。各任务的环境和手册索引见 `README.md`。
- 已有统一入口时，不得自行创建临时脚本、复制 launcher、直接调用内部 runner/worker，或另起一套配置、任务选择、模型请求和产物写入逻辑。这会绕过身份校验、锁、不可变快照、恢复和 receipt，造成行为不一致。
- 普通、capture、steering、steering+capture rollout 都通过 `dycon rollout`；只在唯一 `config.yaml` 中切换开关。正式 rollout 拒绝 `--set`，resume 使用原始快照。
- Run 独立共识/熵策略是明确授权的例外：`dycon run create ... --parallel-config POLICY.yaml` 冻结本 run 的代表策略和熵映射，不向根 `config.yaml` 添加新策略参数；执行/resume 只读本 run 快照。见 `docs/MINIMUM-RUN-CN.md`。
- 多卡独立 TP=1 轨迹调度使用 `serving.trajectory_sharding: per_gpu_tp1`，复用正式 `run create/start/status/cancel/resume`；GPU/端口列表与 `max_cpu_workers_per_gpu` 在创建时冻结。完整轨迹和其候选分支固定使用同一卡，空闲 CPU 轨迹槽位立即补发；不得另建分片 launcher。见 `docs/MINIMUM-RUN-CN.md`。
- 诊断先查找已有 `hidden/evaluation/`、`runtime/scripts/` 工具，核对参数、硬编码范围和副作用。历史 speed-test/验收脚本不能替代生产入口。
- `difficulty/scripts/train.sh` 是 `difficulty-train` 的正式 Bash `source` 包装，只委托统一 CLI。`difficulty/training/config.yaml` 是明确授权的 difficulty 专用配置，不改变 rollout 的根配置约定。`dycon difficulty-train` 完成配置校验、dataset/split 身份绑定、服务管理和 attempt 产物登记。
- 只有 Python API 的能力应明确说明边界。需要新增正式任务时，应在既有 CLI/模块中实现可维护入口并补充文档；不要用临时训练/提取脚本填补入口空缺。
- action trajectory 分类使用 dycon action-classify，直接读取 manifest 绑定的 canonical trajectory；不得要求 enrichment 作为隐含前置，也不得另写 trajectory 读取脚本。

## 配置与数据

- rollout 等共享运行参数仍使用根 `config.yaml`；difficulty estimator 训练专用配置位于 `difficulty/training/config.yaml`，由正式 `difficulty-train` CLI 读取。`catalog/manifest.json` 仅维护 benchmark 任务定义与分区；轨迹、补丁和评测属于指定 run 的 `evidence.json`。
- `runs/<run>/resolved-config.json`、selection、hash 和 artifact receipt 是受审计证据，禁止直接编辑。明确授权的配置修订统一使用 `dycon run revise-config`，保留 before/after 与哈希链；身份变化必须显式 `--replan` 并归档原状态。不得恢复已移除的专用字段修订接口。
- unresolved、未评测和基础设施故障的成功剩余量标签保留 null；重建 token 证据不能冒充 exact IDs；action 分析复用现有 hidden。
- 使用任务对应的 `.venvs/runner`、`.venvs/analysis`、`.venvs/easysteer`、`.venvs/latent-vllm`，不混用推理和提取 runtime。
- 保留已有未提交修改。SSH 密码和 API key 不得写入工程文档或配置。


## LoRA foundation

- `.venvs/rl` is the dedicated Qwen3.5-9B LoRA/GRPO training environment (single GPU, 1/2/4-GPU whole-parent GRPO, and experimental 8-GPU distinct-parent 8x16). Its dependencies and cache may live on `/data3`; source stays in the original uppercase `RL/` package.
- `dycon lora-train --training-config YAML` is the formal CLI. `RL/experiments/lora-smoke.yaml` and explicit task training YAMLs are authorized standalone LoRA configuration exceptions, read by `RL/config.py`, with paths relative to the project root. Do not import rollout model/serving settings, change root `config.yaml`, or pass `--set`/`--run-id` to train.
- `fresh`, complete-checkpoint `resume`, and adapter `warm_start` share `RL.service.run`; no alternate launcher, third-party adapter fallback, implicit latest checkpoint or silent fresh restart. A new attempt ID is required for every invocation.
- `dycon artifact show/verify ID --scope lora-attempt --storage-root ROOT` uses the existing artifact interface. Outputs belong to `ROOT/RL/checkpoints/ID`; LoRA resolution does not relax global `owned_path`. See `RL/README.md` for identities, recovery and engineering-only validation boundaries.

- Default exact-mode eight-GPU LoRA GRPO requires the explicit 16-branch `distinct_parent_two_pass_v1` protocol and `boundary_grad`. Full 32-branch parents retain scoring identity. Each regular update uses eight different parents. Use the same formal CLI and new attempt IDs. Exact resume preserves the frozen execution plan/world size and cannot directly convert v10/1/2/4-GPU checkpoints. Use the CLI without a torchrun wrapper. See `RL/EXECUTION.md` for the protocol and validation scope.

- Explicit `runtime.topology_policy: preserve_updates` permits 2/4/8-GPU bidirectional LoRA GRPO resume via the same CLI. Preserve the original logical windows, branch weights, advantages, anchor and optimizer state; physical task assignment alone changes. Whole-parent plans may split branches across GPUs; two-pass checkpoints retain their original halves/rounds. Portable training rejects RNG-consuming computations. Historical checkpoints remain immutable; new physical metadata is separate from the source logical contract. See `RL/EXECUTION.md` and `tests/test_grpo_topology.py`; real Qwen/NCCL cross-topology GPU acceptance remains separate.
