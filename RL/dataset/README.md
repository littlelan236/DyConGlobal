# Offline GRPO datasets

## Independent multi-GPU TP=1 replay

The existing `dycon rl-dataset replay-logprobs` lifecycle also supports independent
engines, with one GPU and one TP rank per worker. Set these fields in the standalone
replay YAML (the default remains one worker):

```yaml
runtime:
  gpus: [0, 1, 2]
  tensor_parallel_size: 1
  parallel_workers: 3
```

All other runtime, scheduling, prefetch and KV-memory limits apply **per worker**.
Host prefetch budgets and KV allocations therefore multiply by worker count.
Complete groups are deterministically assigned by remaining token cost, preserving
group prefix locality and dataset order within each worker. Each engine has its
own prefix cache; caches are not shared between GPUs. The same raw T=1 trace
producer, token cache reader, bounded scheduler and result writer serve every
worker. Combined TP>1 plus independent workers is rejected.

The coordinator reserves all requested GPUs through the existing reservation
interface, imports only hash-verified committed child results into one journal,
and publishes one `rl-replay` artifact only after full coverage validation.
Worker journals, execution metadata and logs live under `workers/gpu-N/` in the
ordinary staging directory. `--status` reports total and per-worker committed
branches and effective prefix-cache/graph settings. `--cancel` tears down the
whole process group; `--recover --detach` retains the exact frozen assignment and
verified branch files. Completed groups start no additional model request.

To change topology while retaining an interrupted raw prefix, use a **new replay
ID**, keep the dataset/cache/selection and per-engine scoring settings unchanged,
and explicitly bind the old replay:

```yaml
resume_from:
  id: STOPPED_RAW_REPLAY_ID
  root: /data3/wangtianyi/DyConGlobal
```

First dry-run the new YAML, then cancel the source and confirm dispatcher exit
and GPU cleanup before starting the new ID with `--detach`. Migration holds the
source publication lock and requires stopped source state, verified raw model/
policy/data identities, and a hash-checked committed journal prefix. It checks
every imported branch binding and float32 array before hardlinking (or copying)
bytes. Source staging stays unchanged. `resume-source.json` freezes the source
signature, journal hash, runtime and execution evidence, while branch
`execution_source` preserves imported-versus-new provenance. Uncommitted tails
are excluded. Exact `--recover` remains strict about the new config and code;
migration never silently relaxes recovery identity checks.

For mixed historical/new probability coverage, use
[canonical probability artifacts](PROBABILITY_ARTIFACTS.md) and the strict
[probability configuration](configs/probability-example.yaml).
`probability-plan` freezes missing branches, the existing `replay-logprobs`
lifecycle submits only those traces, and `probability-build` publishes one
portable `rl-probability` artifact. Online-only and offline-only migration use the
same writer and reader. These commands preserve dataset flags and receipts.

`rl-dataset build` freezes explicit SWE-bench parallel-consensus sources into step groups. Every group keeps all native candidates for one actual prompt. The prompt is context only; `mask_spans` covers the original completion token coordinates, including reasoning and action tokens.

The builder never invents reward, advantage, KL, or loss values. Groups with missing native evidence remain in `exclusions.jsonl` and do not become empty completions. `data_ready` is the only readiness state produced here; dataset-level `grpo_ready` stays false because the versioned training service performs its own objective eligibility scan. No business dataset is created by installing or testing this code.

Training uses the existing `dycon lora-train --training-config YAML` entrypoint;
see [offline GRPO](../GRPO.md) and [engineering acceptance](../GRPO_VALIDATION.md).
It consumes all eligible complete groups from one sealed dataset, without a split
manifest, freezes advantages and group order, and uses raw T=1 initial-policy target
logprobs for PPO ratios and token K3. Processed replay estimates are audit inputs,
not the denominator. Source tokens and actual group prompts are checked again at
training preflight even when the dataset was built with `audit.enabled=false`.

`audit.enabled` defaults to `false`. In that mode source selection/content hashes, publication/session/decision/ledger identity comparisons, and native token hash comparisons are skipped for the general native source audit, as requested for sources already audited during rollout. The chosen values are frozen in `dataset.json` and `summary.json`. Set it to `true` to run those checks during build and subsequent source checks. Available online raw probability sidecars have a separate mandatory integrity path: their semantics, byte/token hashes and branch/source identities are validated even when this general audit is off.

## Training probability binding

Training selects probability roles in its own YAML. `grpo.reference_cache.source`
is `model_forward` by default, or explicitly `probability_artifact` (committed
`rl-probability` schema-3 ID/root/scope, or legacy raw `rl-replay`) / `dataset_probability` (named dataset stream). Both
external modes require `dtype_compatibility: same_dtype` or
`fp16_source_bf16_training`. Source artifacts remain neutral: no reference/current
roles, objective version, clip or K3 settings are added to them.

The shared reader in [PROBABILITY_STREAM.md](../PROBABILITY_STREAM.md) validates
receipts and raw T=1 pre-transform completion-target semantics, scoring
checkpoint/tokenizer/adapter/runtime identities, native request/branch mappings,
token hashes, masks and finite float32 targets. It accepts canonical schema 3, offline native
replay v2 metadata and the online builder's
`dataset.json.probability_streams.raw_t1_output_tokens` binding. A source may cover
extra valid dataset groups, but every training-eligible branch must be present;
missing values never change training eligibility or trigger a silent forward.
Training preserves its frozen group/branch order when writing its single cache.

One imported cache serves ratio denominator and K3; current remains live model
output. External imports use cache manifest v2 and local source provenance,
carried into every prepared/legacy-cache checkpoint. That resume path does not
reopen the external probability source. Native streaming resume instead binds
the original source metadata and reads remaining native token ledgers and raw
probability payloads. Dataset `branches.jsonl` stores `native_evidence.root` and
`ledger_ref`; copying the dataset directory alone does not copy those tokens,
even when `audit_native_tokens: false`. Keep the frozen dataset/source paths
and selected payload files accessible; see [streaming recovery](../STREAMING_RESUME.md).
Fresh LoRA base equivalence and supported non-streaming warm-start
scoring-adapter compatibility are checked by training. See [GRPO configuration](../GRPO.md), [training guide](../TRAINING.md)
and [binding acceptance](../GRPO_BINDING_VALIDATION.md). The 203 CPU/integration
checks and real Qwen same-dtype synthetic GPU test do not certify real historical
FP16/BF16 numerical differences or long-context capacity.

## Canonical export, missing-only backfill and verification

Use the [probability YAML](configs/probability-example.yaml) and
[canonical workflow](PROBABILITY_ARTIFACTS.md) for all-source export and merge.
`probability-plan --probability-config YAML` freezes whole target groups and
per-branch missing coordinates. Bind its ID/root/`plan_sha256` in replay
`selection.coverage_plan` and build `coverage_plan`; the replay group selector is
mutually exclusive. Only frozen missing branches enter the existing bounded
producer/continuous scheduler, and an empty set starts no worker. Partial replay
receipts describe only that exact selection, without declaring the dataset ready.

`probability-build --probability-config YAML` copies existing and backfilled values
into `ROOT/RL/probabilities/ID/` (`rl-probability`, schema 3). It also handles
online-only/offline-only migration. Plans use `ROOT/RL/probability-plans/ID/`
(`rl-probability-plan`). Both reuse publication/receipt/locks and explicit compatible
`--recover`; committed IDs remain immutable. Vector bytes and original
dataset probability flags/receipts are preserved. Per-branch provenance separates
online recording from offline estimates; scoring function and execution cohorts
are checked independently. Missing coverage and nonfinite values are reported
separately, and conflicting supplied data fails rather than becoming missing.

```bash
.venvs/runner/bin/dycon rl-dataset probability-check PROBABILITY_ID --storage-root ROOT --with-values
.venvs/runner/bin/dycon artifact verify PROBABILITY_ID --scope rl-probability --storage-root ROOT
# Optional external-source audit, separate from local verification:
.venvs/runner/bin/dycon rl-dataset probability-check PROBABILITY_ID --storage-root ROOT --sources
```

The self-contained final artifact and checkpoint-local cache work while source
NAS is offline. See [alignment acceptance](../PROBABILITY_ALIGNMENT_VALIDATION.md)
for the 348-pass CPU regression, 32/31+1 branch test and real-model boundary.

## Online raw capture to dataset stream

Enable `capture.logprobs.enabled` in the rollout root configuration before `run create`, alongside persisted native capture and managed, unsteered TP=1 serving. The existing rollout collects `raw_t1_sampled_token_v1` from logits before grammar and sampler transforms, with one FP32 scalar per actual output token; generation temperature/top-p/top-k stay unchanged. See [online capture configuration and evidence](../../docs/ONLINE-RAW-LOGPROBS-CN.md). This path performs no separate model replay.

During `rl-dataset build`, `RL.dataset.online_probabilities.bind_online_probability` binds each available request sidecar to its exact native branch. Its probability integrity checks remain active with `audit.enabled=false`. Branch rows store `probability_streams.raw_t1_output_tokens` and `raw_probability_available`; group rows store `raw_probability_complete`. Missing or failed sidecars leave probability coverage unavailable without changing the original token inventory or `data_ready` rule. Invalid supplied sidecars fail validation rather than falling back to other values.

`freeze_dataset_stream` copies verified values into bounded `<f4` shards, writes `raw-t1-probability.json` (schema 2) and `raw-t1-probability-branches.jsonl`, then adds `dataset.json.probability_streams.raw_t1_output_tokens` with `manifest`, `index` and `dataset_binding`. Dataset receipt covers these files. No stream is declared when no online values are available; partial indexes can exist, but an explicit training source must cover every selected eligible branch. One stream requires a consistent scoring identity.

The shared `RL.dataset.probability.ProbabilityStream` / `open_probability_stream` reads the dataset-local manifest/index/arrays under the same neutral protocol as raw replay. `RL.dataset.online_probabilities.OnlineProbabilityReader` instead follows the original ledger/sidecar/source receipt and is used by `check --level full`. The dataset stream has no reference/current role. Training binds `source: dataset_probability`, `stream: raw_t1_output_tokens`, and an explicit dtype compatibility policy as described above. Source `-inf` remains recorded, while the training reader rejects nonfinite selected targets.

## Workflow

Create standalone dataset and replay YAMLs using [dataset configuration](configs/swebench-consensus-example.yaml) and [replay configuration](configs/replay-logprobs-example.yaml) as references. Set fresh IDs and your explicit source bindings and physical roots. Example files contain historical experiment names, not committed artifacts. All uppercase IDs/paths below are placeholders and must match your YAML. Commands run from the project root; neither `--set` nor `--run-id` is accepted.

```bash
# Selection preflight only: no native-token coverage or model check yet.
.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml --dry-run
.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml
.venvs/runner/bin/dycon rl-dataset check DATASET_ID --dataset-root DATASET_ROOT --level full
.venvs/runner/bin/dycon rl-dataset materialize DATASET_ID --dataset-root DATASET_ROOT --cache-root CACHE_ROOT
.venvs/runner/bin/dycon artifact verify DATASET_ID --scope rl-dataset --storage-root DATASET_ROOT
```

`check --level metadata` verifies local receipt and group/member structure. `sources` also verifies frozen dependencies when source audit is enabled; `full` additionally reads native evidence under that same audit setting. These levels do not enable the general native audit when `audit.enabled=false`. Online raw probability checks are separate: build always validates available sidecar semantics, bytes, tokens and identities; `full` also follows their frozen source proofs regardless of that audit setting. Materialization is optional: replay and the loader can read native ledgers directly.

Replay schema 2 creates only `raw_t1_output_tokens_v1` artifacts from exact historical completion IDs. It does not read root sampling values or reapply historical temperature, top-p/top-k, grammar, penalties, stop strings, or token masks. Each value is the full-vocabulary float32 log-softmax at the native target token under the frozen FP16 checkpoint. Historical processed artifacts remain read-only diagnostic data; an old replay ID cannot be recovered as raw.

Bind `token_cache.id` and `token_cache.root` explicitly to a verified cache receipt. Replay streams frozen groups through one ordered cache reader, with bounded metadata workers and an expanded-host-memory budget. `runtime.enable_prefix_caching` uses the pinned vLLM GDN `mamba_cache_mode: align` path; the runtime records the effective cache mode. `runtime.cudagraph_mode` explicitly requests `FULL` (or a named comparison mode) and the worker fails if the effective mode differs. `scheduling` sets a continuous request window and token/group budgets. `selection.group_ids` is an explicit complete-group selector for small pilots; the receipt records exact coverage.

```bash
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml --dry-run
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml --recover
.venvs/runner/bin/dycon rl-dataset check-logprobs REPLAY_ID --storage-root STORAGE_ROOT --with-values --dataset-id DATASET_ID --dataset-root DATASET_ROOT
.venvs/runner/bin/dycon artifact verify REPLAY_ID --scope rl-replay --storage-root STORAGE_ROOT
```

## Configuration and artifacts

The YAML loader rejects unknown fields and duplicate keys. Use `schema_version: 1` and one source selector:

| Configuration | Meaning |
|---|---|
| `sources.runs` + `rollout_ids` | Explicit rollout IDs from exactly one run |
| `sources.runs` + `binding_file` | JSON bindings for one or multiple declared runs; each entry has `source_run_id`, `rollout_id`, and optional `trajectory_sha256` |
| `sources.runs` + `all_submitted: true` | Freeze submitted canonical rollouts at construction; later run growth is not added |
| `audit.enabled` | Source hash/identity verification; default false, frozen into dataset metadata |
| `storage.dataset_root`, `storage.cache_root` | Dataset configuration roots; materialize takes its actual cache root explicitly through `--cache-root` |
| `storage.ledger_read_workers` | Bounded ledger readers: default 8, range 1–32 |
| `storage.max_shard_bytes`, `memory_budget_gib` | Token-cache shard and working-array budgets; oversized sequences are rejected |
| Replay `dataset.id`, `dataset.root` | Explicit committed dataset binding |
| Replay `runtime.gpus`, `tensor_parallel_size` | Unique physical GPUs, one per TP rank; occupied GPUs are rejected |
| Replay `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `kv_cache_memory_bytes` | Fields under `runtime`; bound inference scheduling/context/cache without truncating tokens |
| Replay `runtime.subprocess_seconds` | Child timeout, default 86400 seconds |
| Replay `storage.output_root`, `storage.max_shard_bytes` | Replay output root and maximum per-branch float32 shard size |

The complete-native-group policy requires every expected candidate to be usable and at least two branches (or the configured larger `quality.min_branches`). Missing, truncated or action-less branches stay in the inventory; winner/retained status alone does not decide eligibility. Repeated actions retain their multiplicity. Source aliases share one logical sampling weight. `mask_spans` selects the whole native completion, including generated structural/termination tokens; history remains context with full attention, and no EOS is invented.

| Artifact | Location | Main contents |
|---|---|---|
| Dataset | `DATASET_ROOT/RL/datasets/DATASET_ID/` | Frozen selection/config/source evidence; rollout/group/branch inventories; aliases, exclusions, summary and receipt |
| Token cache | `CACHE_ROOT/RL/cache/CACHE_ID/` | Deduplicated prompt/completion references and typed mmap shards; cache ID derives from dataset receipt and cache policy |
| Replay | `OUTPUT_ROOT/RL/replays/REPLAY_ID/` | Sampling/policy/model snapshots, branch index, float32 result shards, diagnostics, validation, summary and receipt |

`artifact show/verify ID --scope rl-dataset|rl-cache|rl-replay --storage-root ROOT` locates these committed artifacts. `check-logprobs --with-values` checks branch arrays and zero-support counts; providing both dataset arguments also verifies member bindings against that dataset. Incomplete staging is never a committed training artifact.

## Replay semantics

Replay loads the historical model and dtype from frozen source snapshots, verifies registered weight contents, reserves the explicitly configured idle GPUs, and dispatches into the isolated pinned latent-vLLM interpreter. Native V2 trace replay uses `logprobs_mode=raw_logprobs`, T=1/no-transform parameters and the exact completion IDs; values are read by target token ID. Trace length, not EOS, controls completion. Continuous request IDs bind outputs independently of return order. A bounded writer commits float32 branch shards to its journal in configured count/time intervals. `runtime.subprocess_seconds` bounds the worker; engine shutdown and child-process cleanup are confirmed before GPU reservation release.

The raw scoring contract excludes sampling transforms and uses trace length to retain every native target; historical generation controls are provenance rather than scoring parameters. Unsupported scoring model-function changes such as active steering, adapters, quantization and speculative execution are rejected by the producer. Dry-run checks frozen dataset/model metadata and sequence budgets without loading models. Full native-ledger scans before model load are optional through `preflight.enabled`; branch-specific token checks still run during consumption. See the current replay schema and runtime acceptance section below.

Outputs have status `replay_estimate` and `historical_behavior_verified=false`; they are never presented as original recorded probabilities. `-inf` is preserved and reported with completion positions, and makes its complete group unavailable to the probability loader. NaN, positive infinity, missing entries or changed output IDs fail publication. `iter_groups(..., replay=PATH)` rejects estimates by default; explicit `allow_replay_estimates=True` opts in. This flag does not make zero-support groups usable. The generic loader returns its legacy-named `behavior_logprobs` field separately from labels. Training uses the neutral strict probability reader above and assigns the shared-anchor role itself; raw values are not exact historical processed behavior probabilities.

## Progress and recovery

For an interrupted build, run `.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml --recover`. For replay, use the replay recovery command above. Do not combine `--dry-run` with `--recover`.

Build concurrency is bounded by `storage.ledger_read_workers` (default 8, maximum 32); workers return results and only the main thread writes inventories. `progress.json` updates after each group. Build `--recover` verifies its config, implementation, frozen selection, committed file prefixes and source hashes when audit is enabled. It resumes at a completed rollout boundary, discarding an interrupted rollout tail. Legacy staging from earlier code has no compatible checkpoint and cannot be recovered by this interface. Raw replay recovery checks frozen dataset/cache receipts, raw scoring policy, runtime, implementation and completed float32 shards; root sampling changes do not redefine the raw contract. Same IDs are immutable; neither build nor replay silently resumes or overwrites.

## Python loading and training boundary

```python
from RL.dataset.loader import iter_groups, collate_group

for group in iter_groups(dataset_path):
    batch = collate_group(group)  # input_ids, attention_mask, labels, loss_mask

# Explicit replay binding and opt-in to estimated probabilities.
for group in iter_groups(dataset_path, replay=replay_path, allow_replay_estimates=True):
    for member in group["members"]:
        completion_logprobs = member["behavior_logprobs"]
```

`iter_groups` defaults to complete data-ready groups in frozen order. Optional `cache`, `start_group` and `max_sequence_length` support mmap tokens, a deterministic group cursor and explicit length rejection. The first completion target is at `len(prompt)`; the collator does not shift targets, and causal prediction is shifted exactly once by the consuming loss. Replay values align one-to-one with completion tokens. `collate_group` leaves group metadata and optional probabilities on the loader records. The dataset module supplies neutral evidence; `RL.service` / `RL.grpo` own training eligibility, advantages, shared-anchor role assignment and optimizer updates through the existing `lora-train` entrypoint. Dataset build does not assign those roles or create a split.

## Tests and validation scope

Tests use temporary synthetic captures only:

```bash
.venvs/runner/bin/python -m pytest tests/test_rl_dataset.py tests/test_rl_dataset_replay.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DYCON_REPLAY_TEST_GPU=0 .venvs/latent-vllm/bin/python -m pytest tests/test_rl_dataset_native_runtime.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DYCON_REPLAY_TEST_GPU=0 .venvs/latent-vllm/bin/python -m pytest tests/test_rl_dataset_raw_native.py -q
```

The opt-in GPU test verifies the pinned API, sampler/trace kernels and a generated tiny Llama checkpoint. It does not validate Qwen GDN, long context, TP or historical runtime equivalence. The installed sampler has kernel-dependent tie behavior: a batched Triton top-k=1 keeps one tied maximum while the small-batch PyTorch mask keeps both. A regression test records this limitation; numerical validation on an unrelated tiny model does not promote historical replay estimates.

The raw native GPU test compares FP16 trace target values against the model's full-vocabulary float32 oracle, includes EOS inside the trace, and measures raw cache on/off plus actual FULL graph dispatch on an attention backend that supports it. The production Qwen3.5 GDN backend currently advertises only uniform-batch graph support; pinned vLLM downgrades its requested FULL mode to `FULL_AND_PIECEWISE`. The replay worker rejects that downgrade, so production FULL remains blocked until a pinned-runtime change is validated. `dycon rl-dataset replay-logprobs --detach|--status|--cancel` manages bounded asynchronous replay jobs; use a new replay ID for a new identity.

## Optional full replay preflight

Replay YAML `preflight.enabled` defaults to `false`. The dispatcher inspects frozen branch metadata and source policies without rereading every native ledger. The worker validates native token vocabulary, sequence budgets, request overrides and trace parameters as each branch is consumed, including recovery bindings. Set `preflight.enabled: true` to additionally scan all native ledgers in the parent and validate every branch before model construction. Dataset receipt checks, GPU reservation, pinned runtime, deep checkpoint verification and output token/value checks remain mandatory. With full preflight disabled, a later invalid request can stop an already-running replay; completed branch shards remain recoverable.
