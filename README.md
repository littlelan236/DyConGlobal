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
| [Capture operations](docs/CAPTURE-CN.md) | Online capture, publication, recovery, and downstream use |
| [Container operations](docs/CONTAINER-CN.md) | Docker storage, task images, and environment checks |
| [Action entropy policy](docs/ACTION-ENTROPY-RUN-POLICY-CN.md) | Run-frozen consensus representative and entropy rules |
| [Multi-GPU TP=1](docs/MULTI-GPU-TP1-CN.md) | Per-GPU serving and trajectory scheduling |
| [Steering datasets](docs/STEERING-DATASET-LDA-CN.md) | Immutable branch-hidden datasets and visualizations |
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

## Project model

- `core/` and `application/` provide strict configuration, CLI, run planning, execution, and evidence services.
- `catalog/` defines benchmark tasks and partitions. Run-specific trajectories, patches, and evaluations live in run evidence.
- `rollout/`, `benchmark_evaluation/`, `enrichment/`, and `hidden/` implement execution and post-processing.
- `action/`, `difficulty/`, and `steering/` hold immutable analysis datasets and outputs.
- `runs/RUN_ID/` contains frozen run configuration, task selection, attempts, canonical outputs, and receipts.

Active runs use schema 2. A run freezes its effective configuration and selected tasks at creation. Resume and run-bound analysis use that snapshot. Catalog task partitions are not training splits, and a submitted patch is not evidence that a task was resolved.

## Runtime environments

Use `.venvs/runner` for run execution, Docker, and harness; `.venvs/analysis` for token processing and offline analysis; `.venvs/easysteer` for managed serving and online capture; and `.venvs/latent-vllm` for offline hidden replay. Do not mix environments to bypass the project entrypoints.
