# DyConGlobal 运行手册

## 运行环境

从项目根目录执行。下文的 `dycon` 指 `.venvs/runner/bin/dycon`，大写参数为占位符。命令的实际选项以 `dycon COMMAND --help` 为准。

| 环境 | 用途 |
| --- | --- |
| `.venvs/runner` | 配置与任务校验、run、harness、数据管理、只读状态查询 |
| `.venvs/analysis` | Token 处理、离线分析、可视化和索引 worker |
| `.venvs/easysteer` | Managed serving、在线 steering 和 capture |
| `.venvs/latent-vllm` | 离线 hidden replay 和隔离的 RL 概率重放 |
| `.venvs/rl` | LoRA 和全参训练、模型诊断 |

所有正式任务走统一 CLI。它负责资源锁、身份校验、计划、恢复和回执。根 `config.yaml` 控制 rollout，包含普通、steering、capture 与组合模式。独立训练配置见 [离线分析](ANALYSIS-CN.md) 和 [RL 训练](../RL/README.md)。

## 创建和启动

~~~bash
dycon config-check
dycon catalog-check
dycon runtime-check
dycon images-check
dycon run create RUN_ID --operation rollout --task-id INSTANCE_ID
dycon run start RUN_ID --detach
dycon run wait RUN_ID
~~~

启动前先通过对应 benchmark 的环境与任务检查。多任务可通过 `--tasks-file FILE` 或 `--partition train|val|test|excluded` 选择。每个 task 的独立重复数由 `--repeat-count N` 指定。镜像存在性检查和任务启动验收见 [Benchmark 与容器](BENCHMARKS-CN.md)。

Schema 2 run 在创建时冻结 `resolved-config.json`、`selection.json`、`plan.json` 和 work-item 身份。执行、恢复和 run-bound 分析使用该快照。正式 rollout 拒绝 `--set`。目录由 run storage root 定位，`catalog/manifest.json` 仅定义任务与分区，run 的轨迹、补丁和评分由 `evidence.json` 绑定。

一个 task 可有多个独立 repetition。一个 repetition 可有多个物理 attempt，例如基础设施重试或按策略换种子。只有当前采纳的 attempt 代表该 repetition。报告时区分已提交、已评测、已解决和基础设施故障。

### 同一 run 的多卡 serving

在根配置将 `serving.trajectory_sharding` 设为 `per_gpu_tp1`，同时配置 `tensor_parallel_size: 1`、`gpus`、`replica_base_urls`、`base_url` 和 `max_cpu_workers_per_gpu`。GPU 和本机 HTTP `/v1` 地址按顺序一一对应，端口互异，首地址等于 `base_url`。

每卡完整装载模型，显存比例、KV 预算和 sequence 容量按服务实例计算。总 CPU 轨迹并发为 GPU 数乘每卡槽位数。完整轨迹及其候选分支固定在领取该轨迹的 GPU 上，空闲槽位立即领取下一项。新的物理 attempt 可重新分配。普通单服务用 `none`，GPU 数与 TP 一致。

通过 `run status` 或 `report run` 的 `serving_workers` 检查 GPU、端口和轨迹槽位。每个物理 attempt 的 `serving-assignment.json` 记录绑定。已有 run 更换拓扑通过停止后的配置修订；新 clone 使用 `run create --from-run`，保留完整选择和明确的重复数。

### 并行共识与 Action 熵

在创建新 run 时用 `run create ... --parallel-config POLICY.yaml` 冻结该 run 的代表策略和熵映射。共识组的选择和获胜组内代表选择分开计算。保留原始 Shannon 熵，再按显式策略得到下游映射值。检查冻结配置和 step 证据，按 run 对比策略。既有 run 通过配置修订，运行和恢复统一读取 run 快照。

## 监控、取消与恢复

~~~bash
dycon run active-status
dycon run show RUN_ID
dycon run status RUN_ID
dycon report run RUN_ID
dycon artifact verify RUN_ID
dycon run cancel RUN_ID --reason 'Stop for maintenance'
dycon run wait RUN_ID
~~~

优先用 `run active-status` 查活动 run。单 run 的 `status-summary.json` 是可刷新的进度摘要，诊断时检查新鲜度并结合 attempt 日志。换 GPU 或端口前先准备并验证新计划，等待旧 run 停止且 `cancellation.termination.cleanup_complete=true` 后启动。

对中断 run 先检查进程身份、回执与队列，必要时使用 `run reconcile`。用 `run resume RUN_ID --detach` 继续未完成任务。终态失败通过 `run retry-failed` 按明确范围登记重跑，再 resume。用 `--accepted-capture-partial` 显式选择 capture partial 的已采纳样本，细节见 [采集与发布](CAPTURE-CN.md)。

### 追加重复与合并结果

额外独立采样用 `run add-repetitions`。先预览队列变化，再用同一 request ID 和预览得到的 plan hash 安全应用。

~~~bash
dycon run add-repetitions RUN_ID --request-id REQUEST_ID --count 1 --reason 'Add independent sample' --dry-run
dycon run add-repetitions RUN_ID --request-id REQUEST_ID --count 1 --reason 'Add independent sample' --expect-plan-hash HASH
~~~

该操作要求 run 已停止并清理，保留现有采纳结果。中断事务用相同 request ID 和 `--recover`。启动前检查实际队列。对同一逻辑任务范围下的已有执行结果，先用 `run merge SOURCE_RUN TARGET_RUN --mode append-executed --dry-run` 核对目标方向、兼容性和计划变化，确认后应用并恢复目标 run。

### 配置修订

~~~bash
dycon run revise-config RUN_ID --revision-id REVISION_ID \
  --set sandbox.memory_mb=16384 --reason 'Increase task memory' --dry-run
~~~

用子命令内的 `--set` 或 `--patch FILE` 更新冻结配置。预览可在运行中执行；应用前停止目标 run 并确认清理。应用时可用 `--expect-config-hash` 锁定预览的旧值。一般修订保留现有结果，seed、branch count、任务/benchmark 或模型身份变化要求显式 `--replan`，原状态归档后重新规划。中断事务通过同一 revision ID 和 `--recover` 恢复。不要直接编辑快照、selection、哈希或 receipt。

`sandbox.memory_mb` 以 MiB 计，profile 的任务覆盖优先。`execution.supervisor_max_open_files` 冻结 supervisor 与其子进程的软上限，受主机硬限制约束，实际值见 attempt 的 `supervisor-limits.json`。配置修订不搬迁旧产物，既有 run 目录继续由 `run_storage_root` 绑定。

## 评分与比较

~~~bash
dycon harness prepare RUN_ID --rollout-id ROLLOUT_ID
dycon harness run RUN_ID EVAL_ID
dycon harness status RUN_ID
dycon run compare RUN_A RUN_B --baseline RUN_A --format markdown
dycon run diff-config RUN_A RUN_B --only-changed
~~~

使用 `dycon harness run RUN_ID EVAL_ID --gold` 执行 gold 环境检查。使用 `harness prepare/run --incremental` 追加评分新提交，保留已有结算。各 benchmark 的 preflight 见 [Benchmark 与容器](BENCHMARKS-CN.md)。

按共同已提交 task/repetition 比较，分别查看缺失和基础设施故障。用 `--resolved-source solution` 时按原始解题结果统计分子。配置差异来自冻结 run。轨迹级预测与真值对照见 [离线分析](ANALYSIS-CN.md)。

## 详细参考

- [运行状态与比较](#运行状态与比较) · [进度、状态字段与重试投影](#进度状态字段与重试投影)、[在线生成吞吐](#在线生成吞吐)、[冻结配置差异](#冻结配置差异)、[SkillsBench 采纳样本对比](#skillsbench-采纳样本对比)、[结果合并合同](#结果合并合同)、[追加和重跑的审计证据](#追加和重跑的审计证据)
- [冻结配置与资源](#冻结配置与资源) · [配置修订和事务恢复](#配置修订和事务恢复)、[多卡 TP1 调度与拓扑修订](#多卡-tp1-调度与拓扑修订)、[并行代表策略与熵映射](#并行代表策略与熵映射)

## 运行状态与比较

### 进度、状态字段与重试投影

#### Runs, samples, and attempts

An active run uses schema 2. `repeat_count` is the number of independent samples planned per task. Each sample has a stable repetition identity and seed; infrastructure retry belongs to the same sample and does not increment the repetition count. Startup probes are not task samples.

`run wait` waits for the supervisor exit event. It does not poll progress. `run status` and `report run` are read-only views. `run cancel` records a cancellation request; inspect termination and cleanup receipts before restarting or repairing a run.

`run status RUN_ID` uses the same current-sample projection and `sample_summary` as `run active-status`, including `exit_status_counts` and the latest-execution cohort. Its detailed `progress.samples` contains those projected rows; `planned_samples`, `accepted_samples`, `failed_samples` (failed/blocked dispositions), and `pending_samples` (queued/running pending disposition) are recalculated from them. Historical retry/workload counters and source timestamps remain recorded diagnostics. The read-only query does not rewrite `progress.json` or refresh persisted snapshots.

`run active-status` is a compact, read-only view of rollout and RL progress. Its schema 2 `count`, `runs`, `queued_count` and `queued_runs` retain their rollout lifecycle meaning. Each row in `runs` keeps `run_id`, `operation`, `task_count`, `status`, `progress_updated_at`, and the planned `sample_count`, followed only by `sample_summary`:

- `sample_summary` exposes sample/accepted/pending/retry counts, `failure_counts`, and `exit_status_counts` across all planned samples. `pending_count` includes queued and running samples (the `pending` disposition); `exit_status_counts.Pending` counts only those waiting to run, while `exit_status_counts.Running` counts those already running. For example, 127 queued plus 63 running gives `pending_count: 190`.
- Each sample counts exactly once. An explicit `run retry-failed` authorization supersedes the matching historical failure immediately: the sample is `Pending` until a new physical attempt starts, including a progress-only WorkerFailure. A resumed progress cursor marked `Pending` also takes precedence over an older execution's physical attempt. The latest started physical attempt (by numeric suffix) otherwise determines its state; a result from the current execution takes precedence over a stale queued progress cursor. Merely reserving an attempt does not replace the preceding state. Attempts archived by a configuration replan are excluded.
- `sample_summary.latest_attempt` retains `execution_attempt_id`, sample/accepted/pending/retry counts, `failure_counts`, and `exit_status_counts`. Its cohort includes the current execution's running/finished samples **and its queued samples awaiting a physical attempt**. Previously settled samples from older executions remain only in the overall summary. Historical attempts for the same sample are never added to either count.

Both exit-status maps are mutually exclusive counts: their sums equal the enclosing sample count. If an unfinished physical attempt has not been requeued, an exited execution is shown as `Cancelled` when its termination receipt records a cancellation request, otherwise `Interrupted`, matching rollout recovery. An unfinished attempt without a verifiable live/exited execution is `unknown`. The view never performs recovery or changes run evidence. `Pending` is not a failure. The former top-level completed/pending counts, duplicate status/disposition counts and anomalies block are omitted; detailed diagnostics remain available from `run status`.

The same response adds `rl_count` and `rl_runs` for locally live LoRA training attempts, and `rl_stale_runs` for discovered nonterminal attempts without a matched local trainer. Each attempt includes the physical storage root, the committed update and planned target, its current stage and inflight update when recorded, telemetry freshness, recent ETA, available scalar metrics and warnings. Offline GRPO usually records loss and preference diagnostics. Reward fields appear only when present in the latest committed metrics. Discovery shares the implementation of `lora-status --active`. The progress view skips checkpoint payload verification. Use `lora-status ATTEMPT_ID` for recovery evidence. `rl_errors` lists partial discovery failures, and `rl_count: null` indicates the RL query failed as a whole.

`run steps RUN_ID` is a read-only view of running rollout samples. It returns one row per task, selecting the most recently updated running attempt and its latest request `step_id`; the latest zero-based `step_id` is accompanied by `step_count` (`step_id + 1`), and tasks without a recorded agent decision have null step fields.

Each run also maintains `runs/RUN_ID/status-summary.json`, a schema-versioned snapshot of the same single-run fields and sample counts shown by `run active-status`, plus `summary_updated_at`. It is atomically refreshed at run creation, during rollout progress (coalesced to at most one full refresh per second), on lifecycle status changes, and when a failed-retry authorization changes the projected sample state. `progress_updated_at` remains the source progress timestamp. `run active-status` remains a read-only live calculation; the snapshot is the latest successfully written view, so `summary_updated_at` indicates freshness after a hard interruption. Existing runs are not backfilled automatically. A failed refresh is recorded in `status-summary-error.json` and does not change rollout outcomes.

`run create --from-run SOURCE` creates a new run from a source snapshot and accepts only the explicitly supported clone overrides; it does not modify the source run. Existing runs use the single `run revise-config` interface for any configurable field, including run parallel policy fields. See [configuration revisions](#配置修订和事务恢复) for preview, application, identity replanning and recovery. The former resource-specific revision commands have been removed.

Use `run retry-failed` only to register an explicitly selected failure for a later resume. Repeat `--sample-key` to bind exact task/repetition samples, and repeat `--exit-status` for the failure classes to include. The request archives the prior result and physical attempt; a `WorkerFailure` without a physical attempt is archived from its `progress.json` cursor and can still receive one fresh physical attempt. `run resume` consumes the request and preserves the same sample seed round while creating the new attempt. Ordinary resume preserves terminal failures. To leave `LimitsExceeded` untouched, do not include it in `--exit-status`. Capture failures must first pass the capture recovery contract; only after existing evidence is recovered or confirmed unavailable should a new attempt be registered. See the [minimum run guide](#监控取消与恢复).

### 在线生成吞吐

Run `.venvs/runner/bin/dycon serving-token-speed` to inspect active runs and their GPUs. The read-only command outputs JSON grouped by run and GPU. `--window SECONDS` sets the sampling delay (default 2 seconds); `--timeout SECONDS` sets the timeout for each metrics request (default 3 seconds). Both must be positive. An empty `runs` list means no active runs were found.

For a running single-GPU or per-GPU TP=1 serving worker, `generated_tokens` is the difference between two readings of the worker's vLLM `vllm:generation_tokens_total` counter, and `tokens_per_second` divides that difference by the reported `elapsed_seconds`. This is a recent window estimate for all generation served by the worker, not a run-lifetime average or a per-request rate. Endpoints are sampled sequentially, so their measurement times are not identical; request latency also affects the actual interval between readings. Input/prefill tokens are excluded.

Each GPU row has a `status`: `ok` for a successful measurement, `metrics_unavailable` when a metrics request or counter is unavailable, or `serving_pool_not_running` when the run has no current running pool. For a shared multi-GPU tensor-parallel service, each GPU is marked `shared_tensor_parallel` with `tokens_per_second: null`: the shared service counter cannot be attributed to individual GPUs, and this command does not report the shared service total. A zero rate for `ok` means the counter did not increase during the sample; a null rate means it could not be measured.

### 冻结配置差异

Use `run diff-config LEFT_RUN RIGHT_RUN` to compare configurable values from the two validated immutable `resolved-config.json` snapshots. The default Markdown report contains a compact task/sample summary followed by tables grouped by configuration section, including unchanged fields. `--only-changed` filters configuration rows to differences; the task/sample summary stays visible. The comparison direction is always left → right.

Fields are discovered recursively from the current `RunConfig` schema (including run-local consensus/entropy options), rather than a maintained field allowlist. At least one frozen snapshot must contain a field for it to appear; missing fields are never filled from current defaults. Schema constants and `readOnly` fields are omitted. The derived branch capacity is marked `readOnly` in the schema; estimator/payload receipts, run hashes, task bodies, execution contexts and other generated metadata are not configurable rows. Nested legacy/extension fields inside configuration sections remain visible and are marked as outside the current schema. Mapping/list settings occupy one row per setting, rather than expanding arbitrary task-specific keys. Newly declared fields and sections are discovered automatically; new derived schema fields should declare `json_schema_extra={"readOnly": True}`.

Markdown cells escape HTML, pipes, backticks and line breaks. Long values, including prompts, retain their beginning and end, original character count and SHA256 prefix; equality is checked on the full typed values before shortening. `--format json` returns schema 2, with exact unshortened values in `rows` and changed fields in `differences`. Missing, explicit `null`, and different JSON value types remain distinct. This replaces schema 1's full-snapshot leaf diff; `difference_count` now counts configurable settings only. The generic leaf-diff utility used by audited configuration revisions is retained.

The command does not re-resolve either run against the current root configuration, mutate snapshots or query live status. The summary uses the frozen plan's repetition count, including audited additions, rather than parsing the run name. `--output PATH` writes a new file and refuses overwrite.

    .venvs/runner/bin/dycon run diff-config RUN_A RUN_B
    .venvs/runner/bin/dycon run diff-config RUN_A RUN_B --only-changed
    .venvs/runner/bin/dycon run diff-config RUN_A RUN_B --format json --output config-diff.json

### SkillsBench 采纳样本对比

Use the read-only `run compare-skillsbench RUN_A RUN_B --baseline RUN_A` entry point to count final accepted repetitions by shared task and compare the accepted trajectories using each task's minimum accepted count. Repetition IDs do not need to match across runs. The output keeps zero-minimum tasks, sorts by per-task minimum accepted count descending, and includes each run's accepted count and the minimum. `--baseline` adds relative changes for accepted counts and trajectory metrics.

For the minimum-count accepted trajectory cohort, JSON exposes `common_accepted_samples_by_run` and per-task `common_accepted_samples_by_run`. Each run contributes its first `minimum_accepted_count` accepted repetitions ordered by repetition index; the selected repetition IDs may differ between runs. These records include canonical selected-branch assistant steps, prompt/completion/total tokens, and `raw_reward.reward`; unselected parallel branches are excluded. Missing reward remains null. Aggregate metrics are in `common_accepted_metrics_by_run`, with baseline deltas under `changes_from_baseline`.

Both `run compare` and `run compare-skillsbench` expose `token_savings_concentration` when `--baseline` is provided. It reports positive-saving trajectory count, total positive token savings, the share captured by the top 10% of saving trajectories, the largest-trajectory share, and HHI. A high top-10% share or HHI means token savings are concentrated in fewer trajectories; zero or missing savings produce null shares/HHI.

Subcommands and spelling can evolve; CLI help is authoritative. The action and difficulty guides define their module-specific data contracts.

### 结果合并合同

`run merge SOURCE_RUN DESTINATION_RUN` defaults to `--mode by-repetition`,
which matches task/repetition identities and prefers destination accepted results.

Use `--mode append-executed` to append physically executed source repetitions,
including failures, into unexecuted destination slots for the same task. Keep
destination executed repetitions first; cap each task at its frozen destination
`repeat_count`. A repetition's retry history is retained but consumes no extra
slot. A queue/admission cursor without an attempt-started receipt is not an
executed repetition. Source results over the cap remain in the original run.

    .venvs/runner/bin/dycon run merge REVERSE_RUN FORWARD_RUN --mode append-executed --dry-run
    .venvs/runner/bin/dycon run merge REVERSE_RUN FORWARD_RUN --mode append-executed
    .venvs/runner/bin/dycon run resume FORWARD_RUN --detach

Apply only after both runs stop and supervisor exit/cleanup receipts are complete.
`--dry-run` previews the mapping without writing result/evidence/progress/receipts;
previewing active runs can be refused by their exclusive run locks. The original
source run and frozen destination plan/selection/configuration are preserved.
Imported results retain source sample/repetition/seed and attempt provenance;
canonical trajectory/subject/report bytes are copied without alteration. Capture
session IDs and producer identity remain those of the source run, whose capture
receipts and published data must be retained. Import does not regenerate hidden
states or logprobs, and repeated imports skip already imported source repetitions.
Patch and workspace subjects are supported; live-state/web-trace subjects require
their own immutable identity import contract and are explicitly rejected.

Ordinary resume preserves settled failures; interrupted work follows the existing
recovery policy. Importing a failure does not mark it accepted or authorize a new
retry. Use the established recovery/retry commands when a new attempt is wanted.

### 追加和重跑的审计证据

The run retry-failed flag --accepted-capture-partial authorizes one fresh physical
attempt for samples whose current disposition is accepted and capture_status is
exactly partial. Combine --exit-status TimeoutError to select the union of these
samples and failed/blocked TimeoutError samples. Existing --task-id, --sample-key
and --expect-attempt filters apply to both groups. Complete accepted samples
remain ineligible. Selection is run-wide unless exact task/sample filters are
provided; it is not implicitly restricted to the latest execution. Sources are
archived and superseded through the same operator retry index, so resume and
status agree; accepted historical attempts cannot accidentally satisfy an
authorized rerun. Prior outcomes, trajectories, capture and publication evidence
stay intact. Inspect/recover existing capture evidence first as described in the
minimum guide. Registration requires an exited execution with complete cleanup
and never calls the model. Use run resume to execute registered retries.

The run add-repetitions command with --count N appends N independent repetitions
to **every frozen task** in an offline rollout run. N is an increment, not a target
total. For repeat_count=5, count=1 appends rep-005 and changes the total to 6; all
existing sample identities, seeds, attempts, results and frozen configuration
remain unchanged. New samples appear as Pending in status summaries and are
executed by the ordinary run resume. An already terminal status.json describes
the preceding execution; the pending sample summary describes the expanded plan
until resume starts a new execution.

Use --dry-run to preview counts, repetition IDs and old/new plan hashes without
changing the run. Apply with --expect-plan-hash HASH to reject a stale preview.
Requests are idempotent by request ID, count and reason. Before/after plan and
work-item documents, a journal, receipt and event are retained in
repetition-extensions/ID/. An interrupted commit blocks plan loading and execution;
recover with the same ID and original reason plus --recover (omit --count).
Recovery verifies the before/after bytes and frozen configuration/selection/tasks,
then completes the prepared transaction. Unrelated edits are rejected.

The first extension freezes sampling_extension.original_repeat_count. Existing
samples continue using that count for the limits-retry seed stride, including
historical attempt validation. Each appended repetition receives a separate block
of 2^32 seed rounds below the original base seed: block starts are
base_seed - (appended_index + 1) * 2^32 * branch_stride. Each round advances
branch_stride. Blocks are disjoint from each other and from original samples'
nonnegative retry rounds, including parallel branch seeds. Exhausted blocks or
signed-int64 overflow are rejected. Later append operations retain all prior
seed assignments. Configuration replanning continues using this explicit policy;
a new frozen clone uses its own newly created sampling plan.

## 冻结配置与资源

### 配置修订和事务恢复

正式入口为 `.venvs/runner/bin/dycon run revise-config`。可以修订根配置 schema 和 run 并行策略 schema 中的所有可配置字段，不再按字段白名单增加子命令。原有八个 `revise-*` 专用修订命令及其 Python API 已移除。

配置值必须满足完整 schema、跨字段约束和 backend/资源引用校验。`run_id`、selection、execution_context、哈希等派生记录不是配置输入，接口会自动重建。`parallel.max_inflight_branches` 根据 `serving.max_num_seqs` 派生，修改后者即可。

#### 预览与应用

先预览需要修改的任意字段组合：

```bash
.venvs/runner/bin/dycon run revise-config RUN_ID \
  --revision-id memory16g \
  --set sandbox.memory_mb=16384 \
  --set generation.temperature=0.4 \
  --reason 'Increase container memory and adjust sampling' --dry-run
```

移除 `--dry-run` 应用修订。`--set` 放在 `revise-config` 后，可重复传入；值按 YAML 解析，支持布尔、null、列表和字典。也可以使用 `--patch changes.yaml` 传入配置字段的嵌套映射；未指定的字段保留。重复或父子冲突的修改路径会被拒绝，拼错的字段由 schema 报错。

预览结果包含当前和候选 config hash 及字段差异。应用时可添加 `--expect-config-hash PREVIEW_OLD_HASH`，防止预览后其他修订使输入发生变化。接口只读目标 run 的快照，不把根配置当前值混入该 run，也不会修改根配置。

应用前必须通过正式 cancel/wait 停止执行并确认退出与 `cleanup_complete=true`。预览可在 run 活动时进行。修订不启动执行；随后通过正式 `run resume` 继续。普通修订保留已有结果，终态失败仍须通过 `run retry-failed` 明确选择重跑。

#### 身份字段重新规划

seed、parallel enabled/branch_count、limits retry 种子策略、benchmark/任务快照/partition、agent backend/profile、模型路径或身份变化会影响样本身份或执行契约，也允许修改，但需要显式 `--replan`：

```bash
.venvs/runner/bin/dycon run revise-config RUN_ID \
  --revision-id newseed --set generation.seed=101 \
  --replan --reason 'Start new samples under a new seed' --dry-run
```

应用后，原运行状态目录及结果移至 `config-revisions/ID/archived-state/`，原配置、selection、任务和 work-items 保存在 `before/`；新状态按修订后的配置及正式任务选择/采样逻辑生成，progress 从待运行开始。执行历史、日志和事件保留。旧归档中的路径字段按原样保存，可通过归档目录对应原 run 相对路径查找旧文件。新物理 attempt 编号超过该 task/repetition 的旧编号下界，避免复用旧 ID。重复 replanning 也继续保留此编号下界。

改变 benchmark 或任务选择参数时会重新调用正式 catalog 选择逻辑；仅修改 seed 时保留原选中任务。已经归档的结果不参与新执行的 accepted 统计。修改资源、生成温度或时限则通常不需要重新规划，也可显式选择 `--replan` 开始新的采样。

#### 审计与恢复

每次修订保存 `request.json`、`before/`、`after/`、`progress.json` 和 `receipt.json`，并写入 `events/config-revision-ID.json`。相同 revision ID 和相同输入在没有后续修订时幂等；不同输入不能复用该 ID。

可捕获的写入异常会恢复原快照和已归档状态。进程中断留下未完成事务时，使用同一入口恢复：

```bash
.venvs/runner/bin/dycon run revise-config RUN_ID \
  --revision-id REVISION_ID --recover --reason 'Recover interrupted revision'
```

恢复会检查 run 已停止，回滚未提交事务；已提交但最后状态写入中断时，核验提交文件后完成状态记录。存在未完成事务时，正常 load/start/resume 会被阻止，必须先恢复。应用前后均保留配置及 plan 哈希校验。后续 supervisor 的每个 execution attempt 会在 `configuration/` 保存该次使用的配置、计划、selection 和 work-items，避免用最新配置解释旧执行。

#### 内存与存储

`sandbox.memory_mb` 的单位是 MiB，16 GiB 对应 16384；它是全局默认值，已有 profile 的任务专属内存覆盖仍优先。内存上限不是预分配。

修订 `paths.artifact_roots` 可改变后续产物位置；已有 run 的物理目录由 `run_storage_root` 继续绑定，不在配置事务中搬迁大量历史数据。文件搬迁应使用对应存储维护接口。


仅增加已停止 run 的 `capture.export.spool_max_bytes` 时，修订只改变该 run
后续 writer 的冻结配额，不搬迁共享 spool，也不修改其他 run 的限制；因此不要求
其他 run 的 capture writer 停止。目标 run 仍须退出且清理完成。减少配额、修改其他
export 字段或使用 `--replan` 仍保留原有项目 writer 停止检查。

### 多卡 TP1 调度与拓扑修订

在同一个 run 内，每张指定 GPU 启动一个独立 TP=1 serving worker，完整 agent 轨迹固定使用领取它的服务。轨迹每一步的所有候选分支仍在该卡上采样；这不改变 `parallel.branch_count`、选枝策略或重复采样身份。

在根 `config.yaml` 的既有 `serving` 节中设置以下字段，其余配置继续使用原值：

```yaml
serving:
  trajectory_sharding: per_gpu_tp1
  gpus: [0, 1, 2, 3]
  tensor_parallel_size: 1
  base_url: http://127.0.0.1:18200/v1
  replica_base_urls:
    - http://127.0.0.1:18200/v1
    - http://127.0.0.1:18201/v1
    - http://127.0.0.1:18202/v1
    - http://127.0.0.1:18203/v1
  max_cpu_workers_per_gpu: 4
```

`replica_base_urls` 与 `gpus` 按列表顺序一一对应，须为不同端口的本机 HTTP `/v1` 地址；`base_url` 必须等于第一项。创建 run 时会冻结这些值。每卡完整装载一份模型，`gpu_memory_utilization`、KV 配置、`max_num_seqs` 和分支容量均是每个服务的值。

`max_cpu_workers_per_gpu` 是正整数，限制该 GPU 同时承载的 CPU 轨迹任务数，包含准入、容器初始化、agent 工具执行、模型请求和 capture 收尾。该模式下总 rollout CPU worker 数为 `len(gpus) * max_cpu_workers_per_gpu`，替代单服务模式的 `execution.rollout_workers`。上例为四张卡、每卡最多四条在途轨迹，总共最多十六条。

调度器从同一个全局任务队列领取样本，优先分配到当前在途轨迹最少且未达到上限的卡。任一在途任务结束并释放槽位后，立即补发下一项；不等待其他卡或同批轨迹结束。较快的卡会自然领取更多任务。完整轨迹中的请求不会中途换卡；整条轨迹换 seed 重试或基础设施重试是新的物理 attempt，可以重新领取空闲卡，原来的任务、repetition、seed、重试预算和证据约束保持不变。

这里的 CPU worker 是轨迹调度槽位，不是容器内部 CPU core 配额，也不是 vLLM 的单条 decode sequence。提高每卡上限可让一条轨迹执行 CPU 工具时，其他轨迹继续调用模型；仍需满足任务队列非空和 capture/容器资源可用，不能保证所有时刻 GPU 利用率为 100%。

使用既有统一入口：

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run start NEW_RUN --detach
.venvs/runner/bin/dycon run status NEW_RUN
.venvs/runner/bin/dycon report run NEW_RUN
.venvs/runner/bin/dycon run cancel NEW_RUN --reason 'user requested stop'
.venvs/runner/bin/dycon run wait NEW_RUN
```

也可通过 `run create NEW_RUN --operation rollout --from-run SOURCE_RUN` 的既有冻结克隆流程，显式 `--set` 上述 serving 字段。这里仅在创建新 clone 时使用 `--set`；正式执行或 resume 继续读取该 run 原始快照。必须保留源 run 的 repeat-count 和完整任务选择；已有 run 转换模式须走下文的审计修订入口。

例如，来源是 managed serving 且 `repeat_count=1`，创建两卡、每卡最多 4 个 CPU worker 的独立 clone（所有 `--set` 位于子命令前）：

```bash
.venvs/runner/bin/dycon \
  --set serving.trajectory_sharding=per_gpu_tp1 \
  --set 'serving.gpus=[2,3]' \
  --set serving.tensor_parallel_size=1 \
  --set serving.base_url=http://127.0.0.1:18200/v1 \
  --set 'serving.replica_base_urls=[http://127.0.0.1:18200/v1,http://127.0.0.1:18201/v1]' \
  --set serving.max_cpu_workers_per_gpu=4 \
  run create NEW_RUN --operation rollout --from-run SOURCE_RUN --repeat-count 1
```

`NEW_RUN` 须为新 ID，示例 GPU 与端口须替换为实际分配；来源有多个 repetitions 时，将 `--repeat-count` 改为来源的确切值。已有 run 可在停止并确认清理完成后，通过 `run revise-config` 显式修订拓扑，再以原 ID resume；单独 `resume` 不改变模式。克隆与离线修订的边界见[入口约定](#runs-samples-and-attempts)。

`run status` 和 `report run` 的 `serving_workers` 展示每卡端口、CPU 槽位占用与上限；本地证据位于 `runs/RUN/serving/worker-pool.json`。各服务的 launch、process、service 和 server.log 位于 `serving/workers/worker-NN/`。每个真实物理 attempt 另存 `serving-assignment.json`，结果中也含 `serving_assignment`，可追溯 sample/attempt/execution 与 GPU/服务的绑定。

`serving_workers.workers[]` 中的 `active_cpu_workers` 是包含 CPU 工具和收尾阶段的在途轨迹数，`available_cpu_workers` 是当前可继续领取任务的槽位数；它们不等于 GPU 实时 decode sequence 数。`serving_workers.state` 在服务池运行时为 `running`，退出后为 `stopped`。判断取消是否完成仍检查 `cancellation.termination.cleanup_complete`，不能仅看 worker 槽位已归零。

所有服务共用一个 run 的 capture 控制器、发布队列与磁盘预算，各服务只负责自己的模型执行；采集证据仍落在既有 run/NAS 命名空间。取消会通知全部模型服务，supervisor 仍通过现有资源登记与进程身份管理全部 serving 进程组；任一卡服务启动失败会关闭已经启动的同 run 服务。恢复通过现有 capture recovery 和 attempt 恢复流程，已接纳样本不会重新生成。

省略 `trajectory_sharding` 或设为 `none` 时使用既有单服务 TP 模式，仍要求 GPU 数与 TP 一致。旧配置缺少新字段时不注入这些字段，因此不会改变旧配置或 run 的 hash。


#### 保留原 run ID 和进度的离线拓扑修订

先通过 `run cancel` 停止并核验 `cancellation.termination.cleanup_complete=true`，再执行：

```bash
.venvs/runner/bin/dycon run revise-config RUN_ID \
  --revision-id tp1-gpu01 --set 'serving.gpus=[0, 1]' \
  --set serving.base_url=http://127.0.0.1:18200/v1 \
  --set serving.trajectory_sharding=per_gpu_tp1 --set serving.tensor_parallel_size=1 \
  --set 'serving.replica_base_urls=[http://127.0.0.1:18200/v1, http://127.0.0.1:18201/v1]' \
  --set serving.max_cpu_workers_per_gpu=2 --set serving.gpu_memory_utilization=0.4 \
  --reason 'User authorized TP1 topology conversion preserving completed progress'
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

所有字段在同一候选配置中一起校验，TP=1 需显式设置。上例总 CPU 轨迹并发为 4。统一接口保留退出/清理检查、before/after、配置与 plan 哈希链和 receipt；任务选择、work-items、evidence 及已接纳结果不变。显式 KV 字节预算仍优先于显存比例；若要改用比例，可在同一修订中设置 `--set serving.kv_cache_memory_bytes=null`。运行中或清理未完成时拒绝应用修订，但可以预览。共享 GPU 上的各服务仍需安排互异端口及可容纳的显存比例。

### 并行代表策略与熵映射

`parallel_consensus` 继续按固定比例筛选；新增 `parallel_enthophy` 使用当前 step 的**原始 Shannon 熵**动态确定 estimator top-k。两个模式均可选择获胜 action 组内的实际分支。沿用 `core/action_identity.py::canonical_action`，不改变 action 身份口径，不考虑 reasoning 来划分类别。

#### 创建与冻结

为目标 run 准备独立输入，如 `experiment-configs/entropy-b-shortest.yaml`：

```yaml
schema_version: 1
enabled: true
mode: parallel_enthophy
fallback_policy: stop
consensus:
  representative_policy: min_step_tokens
  min_retained_branches: 2
  retention_ratio: 0.5
entropy:
  retention_rule: shannon_exp_keep_v1
```

通过正式入口创建；创建只冻结配置及任务，不启动 GPU、模型或容器：

```bash
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout \
  --task-id INSTANCE_ID --parallel-config experiment-configs/entropy-b-shortest.yaml
```

复制来源 run 的完整配置与任务，再为新 run 指定策略：

```bash
.venvs/runner/bin/dycon run create NEW_RUN --operation rollout --from-run SOURCE_RUN \
  --repeat-count SOURCE_REPEAT_COUNT \
  --parallel-config experiment-configs/entropy-b-shortest.yaml
```

clone 保留 source 的 repeat_count，不同时指定 task/partition/tasks-file。不传新文件时可继承动态 source 的冻结策略；首次选择动态模式必须显式提供映射规则。切回固定比例模式时，输入中删掉 `entropy`，创建器会移除新 run 继承的动态专用块，source 不变。

输入原样冻结到 `runs/NEW_RUN/parallel-config.yaml`；有效参数进入本 run 的 `config.yaml`、`resolved-config.json` 和 plan/config hash；`parallel_policy` 绑定输入 SHA 与生效选择参数 SHA。**不向项目根 `config.yaml` 写入新模式、representative_policy 或 entropy**；根 schema 拒绝这些新增配置。

start/resume/analysis 只读本 run 快照，不重新读取外部策略文件或临时更改参数。更换策略创建新 run。旧 run 缺少新字段时仍使用原固定比例及 estimator-score 代表，不补写旧配置、hash 或 receipt。

共享模型、capture、estimator artifact、容量等仍须满足现有前提。新输入仅接受上例字段：schema_version、enabled、mode、fallback_policy、consensus 的三个参数和 entropy.retention_rule，以及下述新映射专用的 entropy.full_retention_threshold。拒绝未知字段、重复 YAML 键、null、冲突的 --set 来源及无效组合，不开放任意模型/任务/seed/资源覆盖。

#### 获胜组内代表策略

| consensus.representative_policy | 获胜 action 组内升序比较键 |
|---|---|
| `estimator_score`（缺省） | `(estimator_score, branch_id)` |
| `min_step_tokens` | `(当前step生成token总数, estimator_score, branch_id)` |

先选最多票 action 组；组间并列仍按组内最低 score、平均 score、最小 branch ID 决胜。代表策略只改变获胜组中的实际分支，不改变组间规则，不捞回已过滤分支。

最短策略使用 `len(native_receipt.output_token_ids)`，包括本次 reasoning+action 的输出边界/格式 token，排除 prompt、历史、工具返回和后续 step。统一 receipt 验证在选代表前取得长度，提交时再次核验；缺失或矛盾不填0、不静默退回 score。score 策略不增加前置长度等待。费用和 total_usage 始终覆盖全部生成分支。

#### 原始熵与映射

统计集合为通过原合法性检查、有有限 score、经 loop escape 最终保留并具备 canonical action 的当前候选池。设实际数量为 N，每类数量 n_i、p_i=n_i/N：`H=-sum(p_i*ln(p_i))`，单位 nats。**不使用 H/ln(K) 或 H/ln(N)**；在 estimator 排名截断前计算，不用筛选后集合反推自身阈值。N=0 的熵为 unavailable；N<m 为候选不足。

令 m 为本 run 的 min_retained_branches，r_0 为 retention_ratio：

| entropy.retention_rule | raw_keep | 特点 |
|---|---|---|
| `effective_count_v1`（A） | `exp(H)` | 有效 action 类数，筛选较强 |
| `shannon_exp_keep_v1`（B） | `N-(N-m)*exp(-H)` | 筛除量指数衰减 |
| `baseline_floor_v1`（C） | `N*(1-(1-r_0)*exp(-H))` | 原比例为低熵下限 |

最终 `k=min(N,max(m,ceil(raw_keep)))`。在整数边界吸附至多16个 machine ULP，避免 exp/log 舍入使整数保留量无故多一条。A/B 不使用 r_0，C 使用；动态模式不额外乘固定比例。固定 N、m 时三种映射均对 H 单调不减。

按 `(score, branch_id)` 升序保留前 k 条，score 越低越优；边界同分按 branch ID，不能仅用 score<=cutoff 而超额保留。两种代表与三映射可独立组合；固定比例共识另有两种代表组合。

#### 证据与检查

##### Full retention threshold

新增映射先完整计算 `effective_count_v1` 的最终整数分支预算
`base_keep_count=min(N,max(m,stable_ceil(exp(H))))`。仅当该预算**严格大于**
`entropy.full_retention_threshold`（默认 10、可配置正整数）时，改为保留候选池全部 N 条。
无效或 loop escape 排除的分支不会被恢复；候选不足仍按原规则报告不可用。
例如原 k=10 保留 10，原 k=11 改为 32。比较发生在取整、最小值和候选数量裁剪之后。

```yaml
entropy:
  retention_rule: effective_count_full_above_v1
  full_retention_threshold: 10
```

参数仅用于此映射，通过 `--parallel-config` 写入新 run 快照及策略 SHA，旧映射不接受它。
decision entropy 元数据新增 `base_keep_count`、`full_retention_threshold` 和
`full_retention_triggered`；`raw_keep` 仍为 `exp(H)`，`keep_count` 为最终预算。
协调器、loop escape 可行性检查和 full/compact evidence 重放均使用冻结阈值。

decision 记录候选池、canonical 类别、原始 H、规则、raw_keep、k、cutoff、保留 IDs、投票、代表比较键与 run 策略绑定，compact receipt 保留重放输入。执行工具前完成 native 证据与选择校验；启动验收及 artifact verify 按本 run 冻结规则重放。

action-branch-stats 增加 decision_pool_entropy、decision_keep_count、representative_policy、retention_rule，可按策略分组。旧分析的指标定义不变。策略 SHA 不包含可独立审计修订的 admission/timeout 资源项；正式离线 serving-memory/concurrency 修订不改变选择策略，完整 config hash 仍按原修订链更新。

##### Additive effective-count mapping

Run 独立策略新增 `effective_count_plus_n_v1`，必须显式配置非负整数 `entropy.n`：

```yaml
entropy:
  retention_rule: effective_count_plus_n_v1
  n: 6
```

先按 `effective_count_v1` 完整计算整数预算 `k_base`（沿用原始 Shannon 熵、
16 ULP 整数边界处理和 `min_retained_branches` 下限），再取
`k = min(N, k_base + n)`。`N` 为 loop escape 后、estimator top-k 前的有效分支数。
有效分支不足原下限时仍保留原有 insufficient/unavailable 行为，不用偏移量补足。
例如 N=32、全部 action 相同、下限为 2 时，n=6 保留 8；k_base=30 时保留 32。
此规则不使用 `full_retention_threshold`，切换映射的 clone 会移除不适用的旧字段。

输入和 n 随新 run 冻结；选择、loop escape 可用性检查与证据回放都使用同一个偏移量。
entropy 证据额外记录 `base_keep_count`、`n`，`raw_keep` 保留原 exp(H)。
旧 run 的快照、缺少此映射的选择行为均不补写。


`parallel.max_inflight_branches` 不接受根 YAML、`--set`、clone 或并发修订中的手工值。
创建或修订新 run 时，项目自动将它设为该 run 的 `serving.max_num_seqs`；当候选
分支总数大于该值时，仍须启用 `dynamic_branch_admission`，通过已有动态准入机制
分批运行，不改变候选池大小。历史冻结 run 保留原快照值，不能被重新解释。
