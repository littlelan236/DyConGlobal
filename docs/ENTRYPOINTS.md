# Entrypoints and Project Contracts

## One supported interface

Use the installed `dycon` command from the project root, or the equivalent `python -m core.cli`. Check the live command and option set with `dycon --help` and `dycon COMMAND --help`; this guide records stable contracts rather than duplicating every parser option.

Formal work goes through `core.cli`. Do not launch internal workers, copy a task launcher, or reimplement selection, model requests, locks, recovery, or artifact registration. These paths share run identity, snapshots, receipts, and recovery behavior.

## Command families

| Work | Entry points | Environment |
|---|---|---|
| Configuration and catalog checks | `config-check`, `catalog-check`, `runtime-check`, `images-check` | runner |
| Run planning and lifecycle | `run create [--from-run SOURCE]/show/list/start/wait/status/resume/cancel/reconcile/compare` | runner |
| Rollout | `run start` or `rollout RUN_ID` | runner; managed serving uses easysteer |
| Benchmark evaluation | `harness prepare/run/status` and benchmark-specific preflight commands | runner |
| Capture and publication | `capture`, `capture-trace`, `capture-recover`, `capture-publish`, `capture-reclassify`, `online-index` | runner or analysis as shown by command help |
| Enrichment and offline hidden | `enrich`, `token-trace`, `hidden`, `dataset-index`, `token-join` | analysis; hidden replay dispatches to latent-vllm |
| Action analysis | `action-classify/check/plot/index`, `action-branch-stats`, `action-review-length` | analysis |
| Difficulty estimation | `difficulty-dataset`, `difficulty-split`, `difficulty-train/ridge/mlp/predict/evaluate` | analysis |
| Steering analysis | `steering-dataset`, `steering-visualize` | analysis |
| Artifact inspection | `artifact show/verify`, `report run` | runner or analysis per command |

### Accepted repetition and trajectory comparison

Use the read-only `run compare-skillsbench RUN_A RUN_B --baseline RUN_A` entry point to count final accepted repetitions by shared task and compare the accepted trajectories using each task's minimum accepted count. Repetition IDs do not need to match across runs. The output keeps zero-minimum tasks, sorts by per-task minimum accepted count descending, and includes each run's accepted count and the minimum. `--baseline` adds relative changes for accepted counts and trajectory metrics.

For the minimum-count accepted trajectory cohort, JSON exposes `common_accepted_samples_by_run` and per-task `common_accepted_samples_by_run`. Each run contributes its first `minimum_accepted_count` accepted repetitions ordered by repetition index; the selected repetition IDs may differ between runs. These records include canonical selected-branch assistant steps, prompt/completion/total tokens, and `raw_reward.reward`; unselected parallel branches are excluded. Missing reward remains null. Aggregate metrics are in `common_accepted_metrics_by_run`, with baseline deltas under `changes_from_baseline`.

Both `run compare` and `run compare-skillsbench` expose `token_savings_concentration` when `--baseline` is provided. It reports positive-saving trajectory count, total positive token savings, the share captured by the top 10% of saving trajectories, the largest-trajectory share, and HHI. A high top-10% share or HHI means token savings are concentrated in fewer trajectories; zero or missing savings produce null shares/HHI.

Subcommands and spelling can evolve; CLI help is authoritative. The action and difficulty guides define their module-specific data contracts.

## Configuration and run identity

- Root `config.yaml` owns shared rollout, generation, agent, timeout, model, and runtime settings.
- `run create` freezes effective configuration, selected tasks, code identity, and their hashes. A later change to the root file does not change an existing run.
- Formal rollout and resume use the frozen snapshot. Resume does not reselect tasks or apply new root configuration.
- Creation-time overrides are limited to options explicitly supported by the relevant command. Formal rollout does not accept ad hoc `--set` overrides.
- Difficulty training has an explicitly separate `difficulty/training/config.yaml`. A SkillsBench shell wrapper delegates to the project `difficulty-train` command; it is not a second training implementation.
- A run may freeze parallel/entropy policy through `run create --parallel-config POLICY.yaml`. Do not add run-specific strategy values to the shared root configuration. See [Action entropy policy](ACTION-ENTROPY-RUN-POLICY-CN.md).

## Runs, samples, and attempts

An active run uses schema 2. `repeat_count` is the number of independent samples planned per task. Each sample has a stable repetition identity and seed; infrastructure retry belongs to the same sample and does not increment the repetition count. Startup probes are not task samples.

`run wait` waits for the supervisor exit event. It does not poll progress. `run status` and `report run` are read-only views. `run cancel` records a cancellation request; inspect termination and cleanup receipts before restarting or repairing a run.

`run create --from-run SOURCE` creates a new run from a source snapshot and accepts only the explicitly supported clone overrides; it does not modify the source run. Offline changes to an existing run use their dedicated revision commands, such as `run revise-serving-placement` or `run revise-capture-control`, and preserve before/after snapshots and receipts.

Use `run retry-failed` only to register an explicitly selected failure for a later resume. Ordinary resume preserves terminal failures. Capture failures must follow the capture recovery contract; do not regenerate a sample while its capture evidence is unresolved. See the [minimum run guide](MINIMUM-RUN-CN.md#recover-or-retry).

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
