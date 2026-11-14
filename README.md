# DyConGlobal

DyConGlobal runs coding-agent benchmarks and manages their rollout, evaluation, capture, and offline analysis. The project root on the Linux server is `/home/wangtianyi/DyConGlobal`.

## Documentation

| Document | Purpose |
|---|---|
| [Minimum run guide](docs/MINIMUM-RUN-CN.md) | Common setup, run, evaluation, recovery, and analysis workflows |
| [Entrypoints and contracts](docs/ENTRYPOINTS.md) | CLI families, configuration ownership, evidence, and identity rules |
| [Configuration revisions](docs/CONFIG-REVISION-CN.md) | Unified frozen configuration revision, preview, replanning and recovery |
| [Benchmark operations](docs/BENCHMARKS-CN.md) | SWE-bench, SkillsBench, WebArena Verified, and Harbor integration |
| [SkillsBench verifier audit](docs/SKILLSBENCH-VERIFIER-AUDIT-CN.md) | Evidence-based zero-reward classification |
| [SkillsBench verifier resources](docs/SKILLSBENCH-VERIFIER-RESOURCES-CN.md) | Immutable verifier dependencies, validation and release boundaries |
| [Capture operations](docs/CAPTURE-CN.md) | Online capture, publication, recovery, and downstream use |
| [Online raw T=1 logprobs](docs/ONLINE-RAW-LOGPROBS-CN.md) | Opt-in rollout capture, FP32 sidecars, dataset stream and validation scope |
| [Container operations](docs/CONTAINER-CN.md) | Docker storage, task images, and environment checks |
| [Action entropy policy](docs/ACTION-ENTROPY-RUN-POLICY-CN.md) | Run-frozen consensus representative and entropy rules |
| [Multi-GPU TP=1](docs/MULTI-GPU-TP1-CN.md) | Per-GPU serving and trajectory scheduling |
| [Steering datasets](docs/STEERING-DATASET-LDA-CN.md) | Immutable branch-hidden datasets and visualizations |
| [RL training foundation](RL/README.md) | LoRA training, sparse supervision and checkpoint contracts |
| [RL OOM recovery validation](RL/OOM_RECOVERY_VALIDATION.md) | Adaptive activation offload, whole-step retry, CPU evidence and pending GPU acceptance |
| [RL training observability](RL/OBSERVABILITY.md) | Live progress, TensorBoard over SSH, and frozen group/branch advantage queries |
| [RL observability acceptance](RL/OBSERVABILITY_VALIDATION.md) | Query, replay/recovery, immutable-artifact and GPU fresh/resume validation |
| [Native streaming resume](RL/STREAMING_RESUME.md) | Checkpoint recovery, original anchor/rank RNG, external inputs and historical compatibility |
| [Streaming resume acceptance](RL/STREAMING_RESUME_VALIDATION.md) | Single/two-GPU short native fixtures, v10 read-only checks and remaining long-context limits |
| [Offline GRPO training](RL/GRPO.md) | Frozen groups, one epoch, raw T=1 shared anchor from model-forward or external sources, and exact resume |
| [GRPO engineering acceptance](RL/GRPO_VALIDATION.md) | Automated tests and Qwen3.5-9B synthetic GPU/resume evidence |
| [GRPO external-anchor acceptance](RL/GRPO_BINDING_VALIDATION.md) | Training source binding, 203 CPU/integration checks and real Qwen synthetic GPU/resume evidence |
| [Raw probability stream contract](RL/PROBABILITY_STREAM.md) | Neutral online/offline source identities, receipts and float32 targets |
| [Canonical probability artifacts](RL/dataset/PROBABILITY_ARTIFACTS.md) | Schema 3, frozen coverage plans, missing-only replay, portable merge and training import |
| [Probability alignment acceptance](RL/PROBABILITY_ALIGNMENT_VALIDATION.md) | 348 CPU regression checks, coverage/identity/portability evidence and GPU acceptance boundary |
| [Offline GRPO datasets](RL/dataset/README.md) | Native-token step groups, target masks and raw T=1 probability streams |
| [Difficulty feature contract](docs/DIFFICULTY-FEATURE-CONTRACT-CN.md) | Hidden identity, eligibility, task splits, and Ridge input |
| [Dataset Ridge task holdout](docs/DIFFICULTY-RIDGE-HOLDOUT-CN.md) | Nested task/event split, Ridge search and refit, NMSE/R², full test trajectories, and verified results |
| [Trajectory comparison](docs/TRAJECTORY-COMPARISON-CN.md) | Resolved repetitions and terminal workload curves |
| [Action analysis](action/README.md) | Canonical action analysis and shared hidden indexes |
| [Difficulty estimation](difficulty/README.md) | Data, training, prediction, and evaluation |
| [Glossary](GLOSSARY.md) | Project terminology |

## Quick start

Use the project CLI from the corresponding environment. These checks do not start a workload:

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon run create RUN_ID --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
```

`run create` freezes a plan; `run start` launches the workload and requires the configured model, GPU, container image, and runtime. See the minimum run guide before starting a new benchmark or resuming an existing run.
Every new run includes `status-summary.json`, a refreshable per-run JSON snapshot of its `active-status` progress summary; see [Entrypoints and contracts](docs/ENTRYPOINTS.md) for its freshness and update behavior.

## RL training observability

Use `.venvs/runner/bin/dycon lora-status --active` to discover training attempts,
or `lora-status ATTEMPT_ID --storage-root ROOT` for committed progress, recoverable
checkpoints and available stage telemetry. `lora-tensorboard ATTEMPT_ID --action
start` manages a TensorBoard service for completed-update loss curves;
`lora-advantages ATTEMPT_ID` queries the frozen group/branch advantages actually
stored for training. The latter two commands also accept `--storage-root ROOT`.

Queries use the runner environment; visualization/index workers use analysis.
Mutable state stays outside training artifacts, and existing training can be
observed without restarting it. For SSH forwarding, stop commands, legacy index
preparation and provenance limits, see [RL observability](RL/OBSERVABILITY.md).

## Project model

Offline GRPO supports step-wide adaptive checkpoint-boundary activation offload,
allocator configuration before CUDA initialization, and at most one coordinated
same-window retry for local CUDA OOM before gradient collectives. Inspect per-rank
memory events with `dycon lora-memory`. As of 2026-11-14, implementation and CPU
regressions pass; GPU acceptance is deferred until GPU0/1 are free, preserving
existing workloads. See [OOM recovery validation](RL/OOM_RECOVERY_VALIDATION.md)
for configuration, evidence and the remaining acceptance criteria.

- `core/` and `application/` provide strict configuration, CLI, run planning, execution, and evidence services.
- `catalog/` defines benchmark tasks and partitions. Run-specific trajectories, patches, and evaluations live in run evidence.
- `rollout/`, `benchmark_evaluation/`, `enrichment/`, and `hidden/` implement execution and post-processing.
- `action/`, `difficulty/`, and `steering/` hold immutable analysis datasets and outputs.
- `RL/` provides single-GPU LoRA causal CE and single/data-parallel `offline_grpo_k3_v1` through `dycon lora-train`. GRPO consumes every eligible complete group once and computes current logprobs with the live BF16 LoRA model. One frozen raw T=1 anchor serves both ratio and K3, selected by `grpo.reference_cache` from model-forward, a committed `rl-probability` artifact (schema 3), a legacy raw `rl-replay` artifact, or an explicitly named dataset probability stream. `rl-dataset probability-plan/build/check` freezes coverage, reuses existing values and exports a self-contained artifact; the existing replay lifecycle scores only missing branches. `RL/dataset/` records neutral data/probability identities; training assigns roles and verifies compatibility. No training split is required.
- `runs/RUN_ID/` contains frozen run configuration, task selection, attempts, canonical outputs, and receipts.

Active runs use schema 2. A run freezes its effective configuration and selected tasks at creation. Resume and run-bound analysis use that snapshot. Catalog task partitions are not training splits, and a submitted patch is not evidence that a task was resolved.

## Runtime environments

Use `.venvs/runner` for run execution, Docker, harness and `rl-dataset` commands; `.venvs/analysis` for token processing and offline analysis; `.venvs/easysteer` for managed serving and online capture; `.venvs/latent-vllm` for offline hidden replay and isolated dataset logprob replay; and `.venvs/rl` for LoRA training. Do not mix environments to bypass the project entrypoints.


Offline GRPO also has an explicit experimental 8-GPU distinct-parent 8x16
execution mode through the same `dycon lora-train` service. Scoring uses complete
32-branch parents; each regular update consumes 16 branches from eight different
parents. See [configuration and acceptance boundaries](RL/DATA_PARALLEL_8X16.md).
The default 1/2/4-GPU whole-parent mode and historical resume contracts remain
supported. GPU throughput and long-prefix capacity require separate acceptance.

The 2026-11-14 [8x16 CPU/Gloo validation](RL/DATA_PARALLEL_8X16_VALIDATION.md) records 426 passed and 12 opt-in skips, including real eight-process Gloo and prepared/streaming checkpoint resume. Real-model eight-GPU correctness, long-prefix capacity, throughput and training quality remain pending.
