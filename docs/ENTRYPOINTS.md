# Entrypoints and Project Contracts

## One supported interface

Use the installed `dycon` command from the project root, or the equivalent `python -m core.cli`. Check the live command and option set with `dycon --help` and `dycon COMMAND --help`; this guide records stable contracts rather than duplicating every parser option.

Formal work goes through `core.cli`. Do not launch internal workers, copy a task launcher, or reimplement selection, model requests, locks, recovery, or artifact registration. These paths share run identity, snapshots, receipts, and recovery behavior.

## Command families

| Work | Entry points | Environment |
|---|---|---|
| Configuration and catalog checks | `config-check`, `catalog-check`, `runtime-check`, `images-check` | runner |
| Run planning and lifecycle | `run create [--from-run SOURCE]/show/list/start/wait/status/steps/resume/cancel/reconcile/retry-failed/compare/diff-config` | runner |
| Rollout | `run start` or `rollout RUN_ID` | runner; managed serving uses easysteer |
| Live generation throughput | `serving-token-speed` | runner |
| Benchmark evaluation | `harness prepare/run/status` and benchmark-specific preflight commands | runner |
| Capture and publication | `capture`, `capture-trace`, `capture-recover`, `capture-publish`, `capture-reclassify`, `online-index` | runner or analysis as shown by command help |
| Enrichment and offline hidden | `enrich`, `token-trace`, `hidden`, `dataset-index`, `token-join` | analysis; hidden replay dispatches to latent-vllm |
| Action analysis | `action-classify/check/plot/index`, `action-branch-stats`, `action-review-length` | analysis |
| Difficulty estimation | `difficulty-dataset`, `difficulty-split`, `difficulty-train/ridge/mlp/predict/evaluate`, `difficulty-ridge-holdout` | analysis |
| Steering analysis | `steering-dataset`, `steering-visualize` | analysis |
| Artifact inspection | `artifact show/verify`, `report run` | runner or analysis per command |

### Live generation throughput

Run `.venvs/runner/bin/dycon serving-token-speed` to inspect active runs and their GPUs. The read-only command outputs JSON grouped by run and GPU. `--window SECONDS` sets the sampling delay (default 2 seconds); `--timeout SECONDS` sets the timeout for each metrics request (default 3 seconds). Both must be positive. An empty `runs` list means no active runs were found.

For a running single-GPU or per-GPU TP=1 serving worker, `generated_tokens` is the difference between two readings of the worker's vLLM `vllm:generation_tokens_total` counter, and `tokens_per_second` divides that difference by the reported `elapsed_seconds`. This is a recent window estimate for all generation served by the worker, not a run-lifetime average or a per-request rate. Endpoints are sampled sequentially, so their measurement times are not identical; request latency also affects the actual interval between readings. Input/prefill tokens are excluded.

Each GPU row has a `status`: `ok` for a successful measurement, `metrics_unavailable` when a metrics request or counter is unavailable, or `serving_pool_not_running` when the run has no current running pool. For a shared multi-GPU tensor-parallel service, each GPU is marked `shared_tensor_parallel` with `tokens_per_second: null`: the shared service counter cannot be attributed to individual GPUs, and this command does not report the shared service total. A zero rate for `ok` means the counter did not increase during the sample; a null rate means it could not be measured.

### Frozen configuration comparison

Use `run diff-config LEFT_RUN RIGHT_RUN` to compare the complete immutable `resolved-config.json` snapshots. The command validates and reads both run plans, reports leaf-level differences by JSON Pointer, and distinguishes an absent field from an explicit `null`. It never re-resolves either run against the current root configuration. Output defaults to Markdown; `--format json` returns the complete structured report. `--output PATH` writes a new file and refuses to overwrite an existing one.

    .venvs/runner/bin/dycon run diff-config RUN_A RUN_B
    .venvs/runner/bin/dycon run diff-config RUN_A RUN_B --format json --output config-diff.json

### Accepted repetition and trajectory comparison

Use the read-only `run compare-skillsbench RUN_A RUN_B --baseline RUN_A` entry point to count final accepted repetitions by shared task and compare the accepted trajectories using each task's minimum accepted count. Repetition IDs do not need to match across runs. The output keeps zero-minimum tasks, sorts by per-task minimum accepted count descending, and includes each run's accepted count and the minimum. `--baseline` adds relative changes for accepted counts and trajectory metrics.

For the minimum-count accepted trajectory cohort, JSON exposes `common_accepted_samples_by_run` and per-task `common_accepted_samples_by_run`. Each run contributes its first `minimum_accepted_count` accepted repetitions ordered by repetition index; the selected repetition IDs may differ between runs. These records include canonical selected-branch assistant steps, prompt/completion/total tokens, and `raw_reward.reward`; unselected parallel branches are excluded. Missing reward remains null. Aggregate metrics are in `common_accepted_metrics_by_run`, with baseline deltas under `changes_from_baseline`.

Both `run compare` and `run compare-skillsbench` expose `token_savings_concentration` when `--baseline` is provided. It reports positive-saving trajectory count, total positive token savings, the share captured by the top 10% of saving trajectories, the largest-trajectory share, and HHI. A high top-10% share or HHI means token savings are concentrated in fewer trajectories; zero or missing savings produce null shares/HHI.

Subcommands and spelling can evolve; CLI help is authoritative. The action and difficulty guides define their module-specific data contracts.

## Configuration and run identity

- Root `config.yaml` owns shared rollout, generation, agent, timeout, model, and runtime settings.
- `execution.supervisor_max_open_files` controls the supervisor soft `RLIMIT_NOFILE` and is frozen into each new run. The value cannot exceed the host hard limit; older snapshots without this optional field retain their original startup limit.
- `run create` freezes effective configuration, selected tasks, code identity, and their hashes. A later change to the root file does not change an existing run.
- Formal rollout and resume use the frozen snapshot. Resume does not reselect tasks or apply new root configuration.
- Creation-time overrides are limited to options explicitly supported by the relevant command. Formal rollout does not accept ad hoc `--set` overrides.
- Difficulty training has an explicitly separate `difficulty/training/config.yaml`. A SkillsBench shell wrapper delegates to the project `difficulty-train` command; it is not a second training implementation.
- A run may freeze parallel/entropy policy through `run create --parallel-config POLICY.yaml`. Do not add run-specific strategy values to the shared root configuration. See [Action entropy policy](ACTION-ENTROPY-RUN-POLICY-CN.md).

## Runs, samples, and attempts

An active run uses schema 2. `repeat_count` is the number of independent samples planned per task. Each sample has a stable repetition identity and seed; infrastructure retry belongs to the same sample and does not increment the repetition count. Startup probes are not task samples.

`run wait` waits for the supervisor exit event. It does not poll progress. `run status` and `report run` are read-only views. `run cancel` records a cancellation request; inspect termination and cleanup receipts before restarting or repairing a run.

`run status RUN_ID` uses the same current-sample projection and `sample_summary` as `run active-status`, including `exit_status_counts` and the latest-execution cohort. Its detailed `progress.samples` contains those projected rows; `planned_samples`, `accepted_samples`, `failed_samples` (failed/blocked dispositions), and `pending_samples` (queued/running pending disposition) are recalculated from them. Historical retry/workload counters and source timestamps remain recorded diagnostics. The read-only query does not rewrite `progress.json` or refresh persisted snapshots.

`run active-status` is a compact, read-only view of live runs. Each run keeps `run_id`, `operation`, `task_count`, `status`, `progress_updated_at`, and the planned `sample_count`, followed only by `sample_summary`:

- `sample_summary` exposes sample/accepted/pending/retry counts, `failure_counts`, and `exit_status_counts` across all planned samples. `pending_count` includes queued and running samples (the `pending` disposition); `exit_status_counts.Pending` counts only those waiting to run, while `exit_status_counts.Running` counts those already running. For example, 127 queued plus 63 running gives `pending_count: 190`.
- Each sample counts exactly once. An explicit `run retry-failed` authorization supersedes the matching historical failure immediately: the sample is `Pending` until a new physical attempt starts, including a progress-only WorkerFailure. A resumed progress cursor marked `Pending` also takes precedence over an older execution's physical attempt. The latest started physical attempt (by numeric suffix) otherwise determines its state; a result from the current execution takes precedence over a stale queued progress cursor. Merely reserving an attempt does not replace the preceding state. Attempts archived by a configuration replan are excluded.
- `sample_summary.latest_attempt` retains `execution_attempt_id`, sample/accepted/pending/retry counts, `failure_counts`, and `exit_status_counts`. Its cohort includes the current execution's running/finished samples **and its queued samples awaiting a physical attempt**. Previously settled samples from older executions remain only in the overall summary. Historical attempts for the same sample are never added to either count.

Both exit-status maps are mutually exclusive counts: their sums equal the enclosing sample count. If an unfinished physical attempt has not been requeued, an exited execution is shown as `Cancelled` when its termination receipt records a cancellation request, otherwise `Interrupted`, matching rollout recovery. An unfinished attempt without a verifiable live/exited execution is `unknown`. The view never performs recovery or changes run evidence. `Pending` is not a failure. The former top-level completed/pending counts, duplicate status/disposition counts and anomalies block are omitted; detailed diagnostics remain available from `run status`.

`run steps RUN_ID` is a read-only view of running rollout samples. It returns one row per task, selecting the most recently updated running attempt and its latest request `step_id`; the latest zero-based `step_id` is accompanied by `step_count` (`step_id + 1`), and tasks without a recorded agent decision have null step fields.

Each run also maintains `runs/RUN_ID/status-summary.json`, a schema-versioned snapshot of the same single-run fields and sample counts shown by `run active-status`, plus `summary_updated_at`. It is atomically refreshed at run creation, during rollout progress (coalesced to at most one full refresh per second), on lifecycle status changes, and when a failed-retry authorization changes the projected sample state. `progress_updated_at` remains the source progress timestamp. `run active-status` remains a read-only live calculation; the snapshot is the latest successfully written view, so `summary_updated_at` indicates freshness after a hard interruption. Existing runs are not backfilled automatically. A failed refresh is recorded in `status-summary-error.json` and does not change rollout outcomes.

`run create --from-run SOURCE` creates a new run from a source snapshot and accepts only the explicitly supported clone overrides; it does not modify the source run. Existing runs use the single `run revise-config` interface for any configurable field, including run parallel policy fields. See [configuration revisions](CONFIG-REVISION-CN.md) for preview, application, identity replanning and recovery. The former resource-specific revision commands have been removed.

Use `run retry-failed` only to register an explicitly selected failure for a later resume. Repeat `--sample-key` to bind exact task/repetition samples, and repeat `--exit-status` for the failure classes to include. The request archives the prior result and physical attempt; a `WorkerFailure` without a physical attempt is archived from its `progress.json` cursor and can still receive one fresh physical attempt. `run resume` consumes the request and preserves the same sample seed round while creating the new attempt. Ordinary resume preserves terminal failures. To leave `LimitsExceeded` untouched, do not include it in `--exit-status`. Capture failures must first pass the capture recovery contract; only after existing evidence is recovered or confirmed unavailable should a new attempt be registered. See the [minimum run guide](MINIMUM-RUN-CN.md#recover-or-retry).

## Configuration, catalog, and evidence boundaries

- `catalog/manifest.json` describes benchmark/task identity and task partitions; it is not run evidence or a training split.
- `runs/RUN_ID/plan.json`, frozen configuration, selection, and `evidence.json` bind tasks, samples, attempts, canonical trajectories, patches, and evaluations.
- A submitted result is distinct from an evaluation. `resolved=true` requires an evaluation bound to the selected patch; unresolved, ungraded, and infrastructure-error outcomes remain distinct.
- Exact native token IDs and reconstructed token evidence are different evidence classes. Reconstructed IDs must not be presented as exact IDs.
- Difficulty datasets and action analyses are project-level immutable artifacts with their own source selection and receipts. They do not require creating a fake analysis run.
- Large tensors can be referenced through NAS locators and receipts. Capture spool data is retained until publication is acknowledged.

`artifact verify` checks registered artifacts within its selected scope. It is not a full re-evaluation of the benchmark, model quality, or every remote tensor. Use the relevant harness, environment, capture, or model verification command for those questions.

## Serving and capture

Ordinary, capture-enabled, steering, and steering-plus-capture rollout all use `dycon rollout` through the run lifecycle. Change supported switches in the shared config before creating a run; existing runs retain their snapshot.

For independent TP=1 serving on multiple GPUs, use `serving.trajectory_sharding: per_gpu_tp1` with `run create/start/status/cancel/resume`. GPU and port assignments are frozen in the plan. Do not create a separate GPU sharding launcher. See [Multi-GPU TP=1](MULTI-GPU-TP1-CN.md).

Capture is finalized and published under the run's source identity. Recovery resumes publication of existing evidence and does not request a new model response. See [Capture operations](CAPTURE-CN.md).

## Benchmark backends

Backend registry entries bind benchmark, agent, runtime, subject, and evaluator. Use the benchmark-specific CLI for task import, image preparation, profile validation, and run preflight. Current integration boundaries are summarized in [Benchmark operations](BENCHMARKS-CN.md).


### Automatic capture publication

After worker, serving and backend teardown, persisted think_close runs
audit the final accepted/submitted canonical cohort and publish complete or sparse
native-exact data under a new immutable publication ID. `capture status RUN_ID`
shows the compact `capture-publication.json` summary when available. `capture
finalize RUN_ID` retries this receipt-bound service under the run lock without
model replay; its failures never rewrite rollout outcomes. Capture with other
anchors reports unsupported for this automatic path. Multi-branch
parallel runs use a separate visualization-only publication: selected hidden,
native truth and candidate evidence have independent coverage, and no training
index is produced. Both `difficulty-plot-trajectories` and `difficulty-plot-parallel`
accept `--publication-id`; the former uses an existing compatible fit attempt.

For nonparallel training-compatible publications, the summary freezes an index ID and an `index_command`. Run that official
`online-index RUN_ID INDEX_ID --publication-id PUBLICATION_ID` command to
materialize the schema-3 view. Index creation verifies the publication source
chain, preserves interrupted stages and supports identical retries. Complete and
partial automatic publications both use `enrichment/online-publications/ID`;
the legacy `enrichment/online` directory is preserved. See CAPTURE-CN.md for
coverage semantics, recovery boundaries and the configurable batch timeout.


## Append repetitions and explicitly rerun accepted partial capture

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


## Dataset Ridge 外层任务留出评估

完整配置、两层协议、指标口径、失败恢复与 2026-11-09 真实验收记录见[任务留出评估专题](DIFFICULTY-RIDGE-HOLDOUT-CN.md)。

正式入口（analysis 环境）一次完成来源校验、外层 task train/test、内层 event train/validation、
每目标独立 layer × alpha 搜索、outer_train 重拟合、全量 outer_test 预测及 NMSE/R²：

```bash
.venvs/analysis/bin/dycon difficulty-ridge-holdout EVALUATION_ID \
  --workflow-config difficulty/training/experiments/ridge-holdout.yaml
.venvs/analysis/bin/dycon artifact verify EVALUATION_ID --scope difficulty-evaluation
```

编辑该独立 workflow YAML 指定已发布 dataset。`run.mode: dry-run|check-data|fit` 分别对应只检查来源与划分、
再检查 hidden/cache、实际拟合。`model.ridge.layers: auto` 展开 dataset 契约的全部合法单层；
默认 13 个严格正 alpha（10^-4 至 10^8），不拼接层。配置、完整网格和特征契约冻结在 evaluation 中。
外层默认 seed=42、test_fraction=0.2；内层默认 seed=42、validation_fraction=0.2，其分母仅为 outer_train。
内层同轨迹相关性允许存在；跨任务泛化只看 outer_test。外层 WebArena 当前因 template 泄漏约束明确拒绝。

`outer_split.strategy: task_manifest_v1` 使用 `outer_split.task_partitions: FILE.json`；
此时不要填写随机 seed/test_fraction。JSON 为 `{"task_assignments":[{"task_key":"DATASET_TASK_KEY",
"instance_id":"TASK_ID","partition":"train"}]}`，须恰好覆盖所有 dataset task，partition 只允许 train/test。
task_key 从冻结 rollouts.json 读取，SkillsBench 包含 revision；未知、遗漏、重复和冲突身份均拒绝。

`visualization.enabled: false` 跳过 Ridge 搜索图、轨迹图和 HTML，且不加载 matplotlib；
预测、指标及 coverage audit 仍完整生成。默认 true，每个 test rollout/repetition、每个 target 均导出 PNG/SVG，
HTML 自包含并支持筛选。图只读取权威预测，无二次推理；positive_reward 中 resolved=false 的部分成功轨迹也保留。
fraction 仅使用 dataset 已发布标签，不临时用完整轨迹真值归一化 raw 预测。

产物：`difficulty/evaluations/EVALUATION_ID/` 的 workflow、resolved-config、schema-2 training-config、
cohort、coverage、metrics、by-task/by-rollout CSV、predictions、report，以及可选 plots/index.html。
split 属于原 dataset，attempt 仍由统一 training service 管理并独立封存。
主指标为原单位 pooled NMSE_energy=SSE/sum(y²)、NMSE_variance=SSE/sum((y-mean(y))²)、R²；
另报拟合空间、分组与等权 macro、outer_train 均值基线。有限配对计数及未定义原因显式记录，
零能量、零方差、单点、非有限预测和逆变换溢出不冒充零误差。预测不裁负、不平滑、不强制单调。

每次使用新 evaluation ID，含未完成目录也拒绝覆盖。若训练完成后绘图失败，保留诊断且不写 evaluation 成功 receipt；
使用新 ID 和 `--reuse-attempt-id ATTEMPT_ID` 正式恢复，dataset/split、特征契约、完整有效训练配置与 receipt 必须相同。
关闭可视化不能用于改变已有 attempt 的有效训练配置。任一 test event 失败则标记 partial、无成功 receipt。
Python 调用方使用同一 `difficulty.evaluation.holdout.run` service。

Standalone 训练仍为 `dycon difficulty-train --training-config CONFIG.yaml`，attempt ID 写在 YAML 的
`run.attempt_id`，不是 CLI 位置参数。Standalone `model.ridge.plots_enabled` 默认 true，以保持旧入口行为。

## SkillsBench accepted reward audit

Use the read-only audit entry point for a frozen SkillsBench run:

```bash
.venvs/runner/bin/dycon skillsbench audit-rewards RUN_ID --output OUTPUT_DIR
```

The command uses the same current task × repetition projection as `run status`. It scans all current samples whose disposition is exactly `accepted` and whose finite numeric `raw_reward.reward < 1`; it includes partial rewards and does not coerce missing, boolean, string, NaN, or infinite values to zero. The run must be explicitly named and bound to the frozen SkillsBench benchmark. `OUTPUT_DIR` must be new; the command never overwrites a report or writes into run attempts, evaluation receipts, status, or configuration.

A report directory contains:

- `summary.json`: policy version, scope, source hashes, candidate and processing counts, classification totals, task/rule summaries, and completeness.
- `samples.jsonl`: one detailed record per candidate, including the selected physical attempt, classification, primary rule, reason, evidence hashes, missing evidence, and source-change flags.
- `samples.csv`: tabular form of the candidate records.
- `report.md`: human-readable run and task summary plus VerifierError and Review records.

The policy is `skillsbench_accepted_reward_audit_v1`. `VerifierError` means the evidence proves an unrecovered verifier failure; `ValidOutcome` means completed evaluation evidence explains the below-one reward; `Review` means evidence, contract ownership, or ordering is insufficient. Findings are normal audit results and do not make the command fail. A source change or unreadable candidate is recorded as Review and makes the audit incomplete.


## LoRA engineering foundation

Use `.venvs/rl/bin/dycon lora-train --training-config RL/experiments/lora-smoke.yaml` for single-GPU bf16 Qwen3.5-9B LoRA engineering validation. The `RL/` source remains in the original project; `.venvs/rl` is an independent environment with `locks/rl.in` and `locks/rl.txt`, stored on `/data3` on this server. The standalone YAML is an explicit training configuration exception, resolved against the root project selected by the ordinary root CLI configuration. Rollout `config.yaml` stays unchanged. Neither `--set` nor `--run-id` is accepted.

Only `dry-run` and synthetic-CE `smoke` are supported. Select `run.start.mode: fresh`, `resume` (a committed complete step checkpoint), or `warm_start` (this module's adapter export). All paths delegate to `RL.service.run`; resume failure never becomes a new training phase. Every invocation requires a new output attempt ID; an existing failed directory is not overwritten. The default GPU is physical 0, with selection applied before importing CUDA/PyTorch; an occupied GPU is rejected without stopping services or changing cards.

Inspect these outputs through `artifact show/verify ID --scope lora-attempt --storage-root /data3/wangtianyi/DyConGlobal`. The optional artifact-root registry key is `paths.artifact_roots.lora-attempt`; normal `paths.artifact_scan_roots` discovery also applies. The physical tree is `ROOT/RL/checkpoints/ID`. Artifact verification checks committed content and state; model equivalence is a separate integration test, and neither constitutes benchmark quality evaluation.

See [RL/README.md](../RL/README.md) for all-linear text-backbone LoRA coverage, frozen base/vision/output-head rules, token-weighted accumulation, immutable identities, resume/warm-start contracts, failure artifacts, and acceptance commands. This foundation does not yet implement an RL objective, reward, business dataset or vLLM adapter serving.
