# RL 数据与概率手册

构建原生 token step groups，检查完整性，按需物化 token 缓存。已采集的 raw T=1 值与离线补评结果可封存到同一概率 artifact。训练使用显式绑定，其中 prepared 输入与 streaming 输入有不同的恢复依赖。

构建、重放和概率封存用 `.venvs/runner/bin/dycon rl-dataset`。实际重放由正式 CLI 管理隔离的 latent-vLLM worker。训练与 checkpoint 用法见[训练手册](../README.md)。

## 详细参考

- [构建与重放](#构建与重放) · [数据集创建、物化和 replay](#数据集创建物化和-replay)、[多卡 TP1 概率重放](#多卡-tp1-概率重放)
- [概率封存](#概率封存) · [Raw 概率流的身份与读取接口](#raw-概率流的身份与读取接口)、[Coverage plan、缺失补评与可携带封存](#coverage-plan缺失补评与可携带封存)、[训练侧概率绑定](#训练侧概率绑定)
- [训练输入](#训练输入) · [准备不可变输入与追加续训](#准备不可变输入与追加续训)

## 构建与重放

### 数据集创建、物化和 replay

#### Workflow

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

#### Configuration and artifacts

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

#### Replay semantics

Replay loads the historical model and dtype from frozen source snapshots, verifies registered weight contents, reserves the explicitly configured idle GPUs, and dispatches into the isolated pinned latent-vLLM interpreter. Native V2 trace replay uses `logprobs_mode=raw_logprobs`, T=1/no-transform parameters and the exact completion IDs; values are read by target token ID. Trace length, not EOS, controls completion. Continuous request IDs bind outputs independently of return order. A bounded writer commits float32 branch shards to its journal in configured count/time intervals. `runtime.subprocess_seconds` bounds the worker; engine shutdown and child-process cleanup are confirmed before GPU reservation release.

The raw scoring contract excludes sampling transforms and uses trace length to retain every native target; historical generation controls are provenance rather than scoring parameters. Unsupported scoring model-function changes such as active steering, adapters, quantization and speculative execution are rejected by the producer. Dry-run checks frozen dataset/model metadata and sequence budgets without loading models. Full native-ledger scans before model load are optional through `preflight.enabled`; branch-specific token checks still run during consumption. See the current replay schema and runtime acceptance section below.

Outputs have status `replay_estimate` and `historical_behavior_verified=false`; they are never presented as original recorded probabilities. `-inf` is preserved and reported with completion positions, and makes its complete group unavailable to the probability loader. NaN, positive infinity, missing entries or changed output IDs fail publication. `iter_groups(..., replay=PATH)` rejects estimates by default; explicit `allow_replay_estimates=True` opts in. This flag does not make zero-support groups usable. The generic loader returns its legacy-named `behavior_logprobs` field separately from labels. Training uses the neutral strict probability reader above and assigns the shared-anchor role itself; raw values are not exact historical processed behavior probabilities.

#### Progress and recovery

For an interrupted build, run `.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml --recover`. For replay, use the replay recovery command above. Do not combine `--dry-run` with `--recover`.

Build concurrency is bounded by `storage.ledger_read_workers` (default 8, maximum 32); workers return results and only the main thread writes inventories. `progress.json` updates after each group. Build `--recover` verifies its config, implementation, frozen selection, committed file prefixes and source hashes when audit is enabled. It resumes at a completed rollout boundary, discarding an interrupted rollout tail. Legacy staging from earlier code has no compatible checkpoint and cannot be recovered by this interface. Raw replay recovery checks frozen dataset/cache receipts, raw scoring policy, runtime, implementation and completed float32 shards; root sampling changes do not redefine the raw contract. Same IDs are immutable; neither build nor replay silently resumes or overwrites.

#### Python loading and training boundary

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


#### Optional full replay preflight

Replay YAML `preflight.enabled` defaults to `false`. The dispatcher inspects frozen branch metadata and source policies without rereading every native ledger. The worker validates native token vocabulary, sequence budgets, request overrides and trace parameters as each branch is consumed, including recovery bindings. Set `preflight.enabled: true` to additionally scan all native ledgers in the parent and validate every branch before model construction. Dataset receipt checks, GPU reservation, pinned runtime, deep checkpoint verification and output token/value checks remain mandatory. With full preflight disabled, a later invalid request can stop an already-running replay; completed branch shards remain recoverable.

### 多卡 TP1 概率重放

#### Independent multi-GPU TP=1 replay

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
[canonical probability artifacts](#coverage-plan缺失补评与可携带封存) and the strict
[probability configuration](configs/probability-example.yaml).
`probability-plan` freezes missing branches, the existing `replay-logprobs`
lifecycle submits only those traces, and `probability-build` publishes one
portable `rl-probability` artifact. Online-only and offline-only migration use the
same writer and reader. These commands preserve dataset flags and receipts.

`rl-dataset build` freezes explicit SWE-bench parallel-consensus sources into step groups. Every group keeps all native candidates for one actual prompt. The prompt is context only; `mask_spans` covers the original completion token coordinates, including reasoning and action tokens.

The builder never invents reward, advantage, KL, or loss values. Groups with missing native evidence remain in `exclusions.jsonl` and do not become empty completions. `data_ready` is the only readiness state produced here; dataset-level `grpo_ready` stays false because the versioned training service performs its own objective eligibility scan. No business dataset is created by installing or testing this code.

Training uses the existing `dycon lora-train --training-config YAML` entrypoint;
see [offline GRPO](../README.md#离线-grpo-分组共享锚点与目标) and [engineering acceptance](../README.md#验证范围).
It consumes all eligible complete groups from one sealed dataset, without a split
manifest, freezes advantages and group order, and uses raw T=1 initial-policy target
logprobs for PPO ratios and token K3. Processed replay estimates are audit inputs,
not the denominator. Source tokens and actual group prompts are checked again at
training preflight even when the dataset was built with `audit.enabled=false`.

`audit.enabled` defaults to `false`. In that mode source selection/content hashes, publication/session/decision/ledger identity comparisons, and native token hash comparisons are skipped for the general native source audit, as requested for sources already audited during rollout. The chosen values are frozen in `dataset.json` and `summary.json`. Set it to `true` to run those checks during build and subsequent source checks. Available online raw probability sidecars have a separate mandatory integrity path: their semantics, byte/token hashes and branch/source identities are validated even when this general audit is off.

## 概率封存

### Raw 概率流的身份与读取接口

Canonical final artifacts use scope `rl-probability`, schema 3 at
`ROOT/RL/probabilities/ID/`. `ProbabilityStream` reads that self-contained format
as well as the legacy source adapters described below. Coverage planning,
missing-only replay, merge/verification and migration examples are documented in
[Canonical raw probability artifacts](#coverage-plan缺失补评与可携带封存).

`RL.dataset.probability` is a Torch-free adapter over existing immutable dataset
and replay receipts and `RL.dataset.materialize.read_array`. It supplies the
training importer with provenance, scoring identity and float32 completion vectors.
It does not score a model or assign a training role.

Canonical schema 3: `probability.json` binds the actual enclosing dataset receipt
and branch/group inventory hashes, frozen selection, one
`scoring_function_identity` and `execution_cohorts`. Its `branches.jsonl` retains
the native token/request/mask BINDINGS, with `scoring_function_sha256`,
`execution_cohort_id`, receipt-local `values_ref`, per-branch `origin`, complete
`source_ref`, `finite_count` and `negative_infinity_count`. It does not use the
legacy per-record `scoring_identity_sha256` field. The shared reader adapts these
cohorts to the existing training compatibility interface and preserves all factual
cohort evidence in local provenance.

`coverage_status: complete` describes vector presence relative to frozen targets;
`finite_coverage_complete` additionally requires finite values. Canonical storage
preserves legal `-inf`; training still rejects nonfinite required targets. A partial
artifact can be imported only when all training-eligible branches are present and
finite. Source coverage never changes token/reward eligibility or training order.
Only explicit `probability-check --sources` follows a canonical artifact's
original upstream source roots; ordinary canonical reading/verification uses
receipt-local copies. Prepared/legacy-cache resume uses checkpoint-local
probabilities. Native streaming resume still opens its selected committed
probability artifact and unconsumed native token ledgers. Canonical values are
portable from their original upstream roots; this does not make the whole
training input self-contained in a checkpoint. See
[streaming recovery](../README.md#原生流式恢复).

The schema-2 source adapters below remain readable for migration and direct legacy
training. Their identity/index fields should not be copied verbatim as schema 3.
See [CPU acceptance and limits](#验证范围).

Offline source: committed `rl-replay` at the explicit root and ID. `replay.json`
is schema 2 and `branches.jsonl` indexes the values. The manifest contains:

```json
{
  "schema_version": 2,
  "replay_id": "EXPLICIT_ID",
  "dataset_receipt_sha256": "FROZEN_DATASET_RECEIPT_SHA256",
  "probability_semantics": "raw_t1_output_tokens_v1",
  "raw_contract": {
    "temperature": 1.0,
    "normalization": "full_vocabulary",
    "logits_stage": "pre_transform",
    "targets": "completion_tokens"
  },
  "scoring_identity": {
    "checkpoint_fingerprint": "DEEP_CHECKPOINT_FINGERPRINT",
    "tokenizer_fingerprint": "NATIVE_TOKENIZER_FINGERPRINT",
    "adapter_fingerprint": null,
    "collector_dtype": "float16",
    "effective_model_overrides": {},
    "runtime": {"producer_runtime": "FROZEN_RUNTIME_IDENTITY"},
    "implementation": {"producer": "FROZEN_IMPLEMENTATION_IDENTITY"}
  }
}
```

The existing replay producer's native schema 2 is also accepted through an
explicit adapter: its manifest binds `raw_contract_sha256`, while the full fixed
raw policy is stored in `root-sampling.json`/`effective-policies.json` and model,
dtype and runtime in `models.json`. The adapter checks the producer's exact raw
contract, raw-logprob trace mode, checkpoint inventory, implementation identity,
validation identity, and per-branch raw/model/policy hashes. Target spans derive
from the producer's native completion coordinates and are compared to the frozen
dataset mapping. Original metadata bytes remain in local provenance, and their
hashes are checked against the original source receipt during checkpoint verify.

Online source: `dataset.json.probability_streams[STREAM_KEY]` freezes `manifest`,
`index` (safe relative paths within the dataset receipt), and `dataset_binding`.
The manifest uses the same schema, `stream: STREAM_KEY` and semantics
`raw_t1_sampled_token_v1`. Its `dataset_receipt_sha256` equals the frozen
`dataset_binding` content identity; the enclosing dataset receipt is independently
verified against the training input. This avoids a circular receipt hash. Request
evidence can predate the dataset; its final branch mapping is frozen by the dataset
builder, without regenerating source evidence.

The online producer is `capture.logprobs.enabled` in the existing rollout
lifecycle; see [configuration, receipts and verification scope](../../docs/CAPTURE-CN.md#在线-raw-t1-采集与-sidecar).
`RL.dataset.online_probabilities.freeze_dataset_stream` writes
`raw-t1-probability.json`, `raw-t1-probability-branches.jsonl` and bounded `<f4`
dataset shards. The stream key is `raw_t1_output_tokens`; its `dataset_binding`
hashes the frozen branch/group inventory files, and is not the enclosing receipt
hash. Branch rows retain original sidecar/source proof alongside
`raw_probability_available`; groups retain `raw_probability_complete` independently
of token `data_ready`. No values means no declared stream; a partial index does
not satisfy an import requiring complete eligible-branch coverage. Build validates
available online probability bytes, semantics and identities even if the general
native source audit is disabled. Mixed scoring identities are rejected.

The dataset-local `ProbabilityStream` reads receipt-covered copies. The separate
`OnlineProbabilityReader` in `online_probabilities` follows original sidecars and
source receipts for source auditing (`rl-dataset check --level full`). A local
request descriptor's `sealed` status alone does not prove NAS upload/publication.
The collector preserves and reports `-inf`; the training importer still requires
finite selected targets.

Both legacy schema-2 indexes contain `group_id`, `branch_sample_id`, `native_request_id`,
`prompt_sha256`, `completion_sha256`, `prompt_count`, `completion_count`,
`tokenizer_fingerprint`, `mask_spans` (the existing native target span policy),
`scoring_identity_sha256` (existing `hash_json(scoring_identity)`), and `values_ref`
(the existing `<f4` shard locator/count/hash). The receipt covers manifest, index
and shards. Source order may differ; import writes the frozen training order.
All source records must bind to real dataset members. A source superset is allowed;
missing eligible branches fail. Selected values are checked while loaded, without
an additional model preflight. The reader rejects duplicate keys, processed
semantics, uncommitted sources, wrong masks/tokens/requests, dtype/count/hash errors
and nonfinite selected values. A source receipt change during import fails.

Production writers must provide factual runtime/implementation/scoring metadata.
Renaming an old processed artifact is not a conversion: it lacks the schema-2 raw
contract and scoring/index bindings. Compatibility with a producer's separately
versioned schema belongs in this neutral adapter, with explicit validation.

The training-local `source-provenance.json` embeds the original source receipt
bytes and the manifest/scoring identity so checkpoint verification works while
the source is offline. Only the training contract assigns the shared anchor roles.

### Coverage plan、缺失补评与可携带封存

Use `dycon rl-dataset probability-plan/build/check` to align online and offline
completion logprobs. These commands do not score a model. Only the existing
`replay-logprobs` lifecycle can start the isolated native trace worker.

#### Storage and immutable identities

`ROOT/RL/probabilities/ID/` is scope `rl-probability`. It contains schema-3
`probability.json`, `branches.jsonl`, `selection.json`, `targets.jsonl`,
`missing.jsonl`, `coverage.json`, `sources.json`, `provenance/`, bounded
`shards/part-*.f4`, config snapshots, `summary.json` and `receipt.json`.
`ROOT/RL/probability-plans/ID/` is scope `rl-probability-plan`; `plan.json` binds
the dataset receipt/inventory, selection, source receipts, policy, implementation
and per-branch coverage. Large branch lists are JSONL; CLI output is a summary.

Both scopes reuse `RL.dataset.artifacts.directory/publication/seal`: lock,
stage, compatible explicit recovery, receipt and atomic rename. A committed ID
cannot be overwritten. Changed source receipts, selection, policy, configuration
snapshot or implementation require new IDs. Recovery rebuilds canonical shards
from the same verified inputs; replay recovery reuses its sealed branch journal.
Stages are never consumer artifacts. No source artifact, ledger, dataset flag,
replay or checkpoint is rewritten by alignment.

The canonical manifest binds `dataset.id`, the actual enclosing
`dataset.receipt_sha256`, and branch/group inventory hashes. The older online
stream's `dataset_binding` remains its content hash; the explicit adapter checks
that binding and does not reinterpret it as the enclosing receipt hash.

The stream is `raw_t1_output_tokens`, semantics `raw_t1_output_tokens_v1`, raw
contract `{temperature: 1.0, normalization: full_vocabulary,
logits_stage: pre_transform, targets: completion_tokens}`, dtype `<f4`. Each
completion token retains one scalar in its original coordinate system, including
tokens outside a downstream loss mask. Repackaging preserves original FP32 bytes.
It performs no averaging, normalization, truncation, interpolation or model call.

#### Source configuration and single-source migration

Start with [the strict configuration example](configs/probability-example.yaml).
Paths resolve relative to the project root; physical roots must resolve to safe
absolute non-root directories. `sources` explicitly select:

| kind | Required binding | Adapter |
|---|---|---|
| `dataset_stream` | dataset ID/root and explicit `stream` | Receipt-covered online schema 2; old sampled-token semantics validated as equivalent raw targets |
| `replay` | replay ID/root | Raw schema 2, including native producer model/runtime/policy metadata |
| `probability` | canonical ID/root | Schema 3, preserving branch origin and parent provenance |

The default role is `existing`. For online-only or offline-only migration, set
the corresponding source and run `probability-build`; a plan is optional when
there is no `backfill` source. A missing target fails unless `allow_partial: true`
was explicitly configured. A historical dataset with no online stream can be
declared as `dataset_stream`; it contributes no values. A dataset that declares
available values but loses their index is corrupt and fails.

One scoring function is allowed per artifact. `scoring_function_identity` binds
the checkpoint fingerprint (whose producer must deeply inventory model/config/
tokenizer/weights), tokenizer, adapter, effective model overrides and raw contract.
`execution_cohorts` separately bind collector compute dtype, FP32 reduction,
runtime/backend/version/engine/graph/cache options and implementation evidence.
FP32 storage does not imply FP32 model compute. Source compute dtypes must agree.
Unknown/incomplete identities fail. Multiple cohorts require their exact hashes
in `cohort_policy.allowed_execution_cohorts`; this cannot authorize a different
checkpoint, adapter, tokenizer or dtype. Plan/check summaries expose known cohort
hashes; undeclared cohorts are rejected, rather than inferred from a directory.

For a planned but not yet produced replay, declare the approved stable cohort
hashes from factual runtime evidence before freezing the plan. Timing,
queue counters and dispatch measurements do not define the cohort hash; original
runtime bytes, including those measurements, remain in source provenance. Graph
and cache configuration and effective execution modes remain cohort factors.
If no existing source supplies a scoring identity, provide
`expected_scoring_identity` with all legacy scoring fields (checkpoint/tokenizer/
adapter/collector_dtype/effective_model_overrides/runtime/implementation). This
is an expected contract, not evidence that an execution already occurred.

#### Missing-only replay and merge

Add at most one source with `kind: replay`, `role: backfill`, and the fresh replay
ID/root it will use. Planning does not require that artifact to exist. Defaults
select every branch in the dataset loader's complete native groups;
`group_ids` freezes a subset of whole groups. Probability availability never
changes token-group eligibility, reward, mask or ordering.

```bash
.venvs/runner/bin/dycon rl-dataset probability-plan --probability-config PROBABILITY.yaml
```

The returned `plan_sha256` is SHA256 of `plan.json`; the separately returned
receipt hash identifies its publication. Add this binding to the probability
configuration and to the replay configuration:

```yaml
# In PROBABILITY.yaml, leave the other resolved fields unchanged:
coverage_plan:
  id: COVERAGE_PLAN_ID
  root: /ABSOLUTE/PLAN/ROOT
  sha256: REPLACE_WITH_RETURNED_64_HEX_PLAN_SHA256

# In REPLAY.yaml, alongside the existing schema-2 dataset/runtime/storage:
selection:
  coverage_plan:
    id: COVERAGE_PLAN_ID
    root: /ABSOLUTE/PLAN/ROOT
    sha256: REPLACE_WITH_RETURNED_64_HEX_PLAN_SHA256
```

`selection.coverage_plan` and `selection.group_ids` are mutually exclusive. The
replay ID/output root must equal the plan's declared backfill. Preparation verifies
the frozen dataset, source receipts, implementation and model/dtype before scoring.
The existing bounded producer/prefetch/continuous scheduler/async writer retains
group locality but submits only missing branches. With 31 of 32 branches present,
only the remaining native P/C trace enters the engine. Replay publication requires
exactly the frozen missing set and binds the plan ID/hash; partial group output
does not claim the whole dataset is probability-ready. Cancellation/detach/status
and recovery use the existing replay commands, without an alternate launcher.

```bash
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml --dry-run
.venvs/runner/bin/dycon rl-dataset replay-logprobs --replay-config REPLAY.yaml
.venvs/runner/bin/dycon rl-dataset probability-build --probability-config PROBABILITY.yaml
.venvs/runner/bin/dycon rl-dataset probability-check PROBABILITY_ID --storage-root ROOT --with-values
.venvs/runner/bin/dycon artifact verify PROBABILITY_ID --scope rl-probability --storage-root ROOT
.venvs/runner/bin/dycon artifact verify COVERAGE_PLAN_ID --scope rl-probability-plan --storage-root PLAN_ROOT
```

With no missing branches, replay returns `no_missing` before GPU reservation or
worker startup; build directly exports the existing values without requiring an
empty replay artifact. Build preserves frozen existing values and accepts the
backfill only for missing coordinates. Wrong plan, missing/extra/duplicate output,
changed receipts/tokens and conflicting values fail. Duplicate existing sources
require `deduplicate_identical: true`, matching raw/model contracts and identical
value hashes; all source proofs remain in `source_ref`.

Coverage states are `valid`, `absent`, `recorded_failed` and `nonfinite`.
`allowed_missing_reasons` defaults to `[absent]`; recorded collection failures
require an explicit `recorded_failed` entry. Corruption/conflicts fail, rather than
becoming missing work. Legal `-inf` is copied and reported; it is not backfilled.
NaN, +inf, positive logprobs, wrong length/hash and unsafe locators fail.
`coverage_status: complete` means all selected branches have vectors;
`finite_coverage_complete` separately states whether every value is finite.

#### Portable reading and training

Every value locator belongs to the new receipt. Shards are copied, not symlinked.
Provenance includes original receipt bytes, original manifests/adapter identity
metadata, parent IDs and original value hashes. Ordinary `probability-check`,
`artifact verify` and `ProbabilityStream` reading do not reopen original
upstream source NAS. Prepared/legacy-cache training resume uses the checkpoint
cache. Native streaming resume still requires the selected canonical artifact
and unconsumed native token ledgers, but canonical probability reads need not
reopen the artifact's original upstream sources. Explicit
`probability-check --sources` audits original source receipts
and reports `source_audit_status: incomplete` for unavailable/changed sources,
separately from local verification.

```yaml
grpo:
  reference_cache:
    source: probability_artifact
    artifact:
      scope: rl-probability
      root: /ABSOLUTE/STORAGE/ROOT
      id: PROBABILITY_ID
    dtype_compatibility: fp16_source_bf16_training
```

The shared reader and existing cache importer consume schema 3 without per-origin
paths. All training-eligible branches must have finite values. A partial artifact
is accepted only when its missing branches fall outside the training eligibility
already frozen from tokens/rewards. Import never removes a group to hide missing
probabilities and never falls back to scoring. Local checkpoint provenance binds
the canonical receipt and cohorts. Prepared/legacy-cache resume uses the
frozen cache; native streaming inherits its source binding and reads remaining
vectors from the selected artifact without importing a full cache. The same
anchor serves ratio and K3, current remains live, and the objective is unchanged.
Checkpoint recovery metadata does not embed all external training inputs; see
[streaming resume](../README.md#原生流式恢复).
Old `rl-replay`, `dataset_probability`, cache schemas and checkpoint resume remain
supported; processed probabilities cannot be relabeled as raw.

### 训练侧概率绑定

#### Training probability binding

Training selects probability roles in its own YAML. `grpo.reference_cache.source`
is `model_forward` by default, or explicitly `probability_artifact` (committed
`rl-probability` schema-3 ID/root/scope, or legacy raw `rl-replay`) / `dataset_probability` (named dataset stream). Both
external modes require `dtype_compatibility: same_dtype` or
`fp16_source_bf16_training`. Source artifacts remain neutral: no reference/current
roles, objective version, clip or K3 settings are added to them.

The shared reader in [概率流合同](#raw-概率流的身份与读取接口) validates
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
and selected payload files accessible; see [streaming recovery](../README.md#原生流式恢复).
Fresh LoRA base equivalence and supported non-streaming warm-start
scoring-adapter compatibility are checked by training. See [GRPO configuration](../README.md#离线-grpo-分组共享锚点与目标), [training guide](../README.md#配置预检和首次训练)
and [binding acceptance](../README.md#验证范围). The 203 CPU/integration
checks and real Qwen same-dtype synthetic GPU test do not certify real historical
FP16/BF16 numerical differences or long-context capacity.


#### Online raw capture to dataset stream

Enable `capture.logprobs.enabled` in the rollout root configuration before `run create`, alongside persisted native capture and managed, unsteered TP=1 serving. The existing rollout collects `raw_t1_sampled_token_v1` from logits before grammar and sampler transforms, with one FP32 scalar per actual output token; generation temperature/top-p/top-k stay unchanged. See [online capture configuration and evidence](../../docs/CAPTURE-CN.md#在线-raw-t1-采集与-sidecar). This path performs no separate model replay.

During `rl-dataset build`, `RL.dataset.online_probabilities.bind_online_probability` binds each available request sidecar to its exact native branch. Its probability integrity checks remain active with `audit.enabled=false`. Branch rows store `probability_streams.raw_t1_output_tokens` and `raw_probability_available`; group rows store `raw_probability_complete`. Missing or failed sidecars leave probability coverage unavailable without changing the original token inventory or `data_ready` rule. Invalid supplied sidecars fail validation rather than falling back to other values.

`freeze_dataset_stream` copies verified values into bounded `<f4` shards, writes `raw-t1-probability.json` (schema 2) and `raw-t1-probability-branches.jsonl`, then adds `dataset.json.probability_streams.raw_t1_output_tokens` with `manifest`, `index` and `dataset_binding`. Dataset receipt covers these files. No stream is declared when no online values are available; partial indexes can exist, but an explicit training source must cover every selected eligible branch. One stream requires a consistent scoring identity.

The shared `RL.dataset.probability.ProbabilityStream` / `open_probability_stream` reads the dataset-local manifest/index/arrays under the same neutral protocol as raw replay. `RL.dataset.online_probabilities.OnlineProbabilityReader` instead follows the original ledger/sidecar/source receipt and is used by `check --level full`. The dataset stream has no reference/current role. Training binds `source: dataset_probability`, `stream: raw_t1_output_tokens`, and an explicit dtype compatibility policy as described above. Source `-inf` remains recorded, while the training reader rejects nonfinite selected targets.

## 训练输入

### 准备不可变输入与追加续训

Prepared 与 native streaming 是两条由冻结 contract 区分的数据路径。async=true 不会把 prepared resume 转为 streaming。Streaming fresh/resume 不要求预先发布全量 prepared/reference 产物，但持续依赖未消费的外部 native ledgers 和 raw probabilities；不支持 append_resume。详见 [流式恢复](../README.md#原生流式恢复)。本页“不访问外部来源”的 exact resume 描述仅适用于 prepared 路径。

All training uses `dycon lora-train --training-config YAML` and `RL.service.run`.
New GRPO attempts using the prepared path publish prepared and reference artifacts before updates. They
do not write `frozen-groups.json`. CE and legacy checkpoint schemas keep their
existing paths. No existing attempt, dataset, checkpoint or receipt is rewritten.

#### Prepare and inspect

```bash
.venvs/runner/bin/dycon lora-prepare --training-config TRAINING.yaml \
  --prepared-id PREPARED_ID --shard-groups 16
.venvs/runner/bin/dycon artifact show PREPARED_ID --scope rl-prepared --storage-root ROOT
.venvs/runner/bin/dycon artifact verify PREPARED_ID --scope rl-prepared --storage-root ROOT
```

Preparation is CPU-only and does not import Torch or load model weights. It scans
native groups once and uses the same eligibility/advantage implementation as
training preflight. New GRPO configurations default to
`grpo.shuffle_algorithm: prefix_length_pool128_v1`: seeded compact group IDs are
partitioned into candidate pools of up to 128, sorted by prompt token length,
formed into complete `group_batch_size` windows, then the windows are seeded and
shuffled. Equal-length groups retain their seeded order. A final partial window
is retained; no group is dropped or duplicated. Group contents, branch order,
advantages and reference values remain intact. The exact output order is frozen
in `schedule.json`; `max_updates` must equal the planned update count.

For four-GPU one-group-per-card execution, this groups four similar prompt
lengths into each optimizer update to reduce prefix-backward stragglers. Pool
size 128 is a fixed schedule contract value, not an automatic memory/batch size.

Set `grpo.shuffle_algorithm: stable_random_v1` to explicitly create a new
random-order schedule. Artifacts created before this field existed retain their
recorded `stable_random_v1` order and continue to resume without rebuilding. An
append segment freezes its own named shuffle algorithm and remains a strict child
of the parent's schedule.
The output includes a `prepared_data` binding; copy that mapping to the training
YAML to explicitly reuse this release. Without a binding, a fresh invocation
publishes an artifact named from its unique attempt ID. There is no implicit
search for a compatible or latest artifact.

Artifacts use the existing root/publication/locking/receipt machinery:

| Scope | Directory | Payload |
|---|---|---|
| `rl-prepared` | `ROOT/RL/prepared/ID` | manifest, schedule, dataset binding, SQLite inventory and sealed binary shards |
| `rl-reference` | `ROOT/RL/references/ID` | manifest, initial adapter snapshot, per-group logprob chunks and source provenance |
| `rl-extension` | `ROOT/RL/extensions/ID` | sealed parent checkpoint, prepared release and constant scheduler extension request |

Each prepared group stores one prompt in little-endian int64, branch completions
in int64, and scores/advantages in float64. SQLite indexes store stable group and
branch IDs, evidence metadata, offsets, counts and content hashes. Native target
masks have the existing `all native completion targets` semantics and reconstruct
at window load time. Padding and tensors are not persisted. This replaces the
illustrative JSONL indexes in the design with a standard indexed store; binary
arrays, stable identities, shard receipts and parent references retain the same
responsibilities. New reference chunks contain one group each; adapter weights
are stored once per reference artifact, not repeated in every chunk.

`show` returns counts, hashes, segment count and parent bindings. `verify` also
walks every prepared group to validate bounds, branch order and native token
hashes. Resume verifies sealed inventories and checks token/offset/content hashes
when loading the needed windows; it does not collate consumed groups.

#### Exact resume

```yaml
run:
  attempt_id: NEW_UNIQUE_ATTEMPT
  mode: smoke
  seed: 42
  start:
    mode: resume
    checkpoint: /ABSOLUTE/PARENT/checkpoints/step-000001
```

Keep the same complete training configuration, dataset receipt, seed, advantage
policy, batch size, model/LoRA/dependencies and reference
selection. The checkpoint pins prepared/reference manifest and receipt hashes,
schedule ID/hash, segment/ordinal, last consumed ID, cumulative consumed-ID hash,
and group/branch/token counters. Mandatory checks run even with optional preflight
checks off. Optimizer, scheduler, adapter and Python/NumPy/CPU/CUDA RNG restore
from the complete checkpoint. Legacy source-code fingerprints remain in old
receipts for audit but are not a code-version compatibility gate. Dataset,
prepared schedule, checkpoint, adapter, model/tokenizer and reference content
hashes remain enforced. No native ledger or external probability source is
opened and no shuffle is repeated. An exact resume after epoch completion fails.
An exact resume inside a previously appended segment keeps the same extension
lineage and does not create another segment.

#### Append complete new groups

Publish a new immutable dataset with the existing dataset build interface. It may
contain just new complete groups or unchanged parent groups plus new groups.
Parent metadata, branch IDs, token hashes, scores and retained membership must be
identical. Mutating a parent group or reusing one of its branch IDs fails. An
append is not a way to change advantage policy or add branches to an old group.

Create an extension training YAML with the new dataset ID, a fresh attempt ID,
the same mathematical training settings, and the new total `max_updates`.
The total is the **sum** of each segment's ceiling division by group batch size.
For five old groups plus three new groups and batch size two it is `3 + 2 = 5`.
Set the start fields and choose a new extension ID:

```yaml
run:
  start:
    mode: append_resume
    checkpoint: /ABSOLUTE/PARENT/checkpoints/step-000001
    extension_manifest: /ABSOLUTE/ROOT/RL/extensions/EXTENSION_ID/manifest.json
training:
  scheduler: constant
  max_updates: 5
```

Retain all other required fields from the parent configuration. Then:

```bash
.venvs/runner/bin/dycon lora-prepare --training-config EXTENSION.yaml \
  --prepared-id CHILD_PREPARED_ID --parent-checkpoint /ABSOLUTE/PARENT/checkpoints/step-000001 \
  --extension-id EXTENSION_ID --append-seed 19
.venvs/rl/bin/dycon lora-train --training-config EXTENSION.yaml
.venvs/runner/bin/dycon artifact verify EXTENSION_ID --scope rl-extension --storage-root ROOT
```

The request validates the parent's complete checkpoint and consumed prefix first.
Parent segments are an exact prefix of the child schedule; only new IDs are
shuffled using `append_seed`. Parent and child partial windows never merge.
Training finishes the old remainder before the added segment. Completed parent
epochs may be extended. Constant schedules inherit optimizer/scheduler state and
global update counts; linear extensions fail explicitly because no linear
extension rule is implemented.

Reference values for old groups remain in their sealed parent artifact. New
groups use the **original** initial-policy adapter snapshot, temporarily loaded
and then restored to the checkpoint adapter before optimizer/RNG restoration.
External sources must provide complete new-branch coverage with the same raw T=1
anchor, dtype rule, tokenizer, adapter and scoring identity. Probability metadata
may describe the whole new dataset, but only new branch vectors are imported.
Run `rl-dataset probability-plan`, the existing missing-only replay, and
`probability-build` first when values are missing; training never launches a
second scoring worker or treats partial coverage as a complete anchor.

The report records `extension_of_attempt`, parent checkpoint, extension request,
new group/branch/token counts, old/new update totals, prepared cache hits, source/native
scan counts, materialized groups, reference reused/new counts, segment count,
verification seconds and bytes read. Model loading, reference preparation and
per-update training times are reported separately. Byte counts represent local
verification reads and omit operating-system page-cache effects.

#### Migration, failures and rollback

A checkpoint with a streaming contract uses the
[native streaming recovery path](../README.md#原生流式恢复); absence of prepared
metadata alone does not identify a legacy non-streaming checkpoint. Only
checkpoints with neither prepared nor streaming contracts use the old
non-streaming path. No path silently converts a checkpoint into another format. Historical
source-code identity metadata is retained but does not block exact resume. For explicit
read-only import of a committed legacy attempt:

```bash
.venvs/runner/bin/dycon lora-prepare --training-config MATCHING.yaml \
  --prepared-id MIGRATED_ID --source-attempt /ABSOLUTE/LEGACY_ATTEMPT
```

This reads the old JSON once and produces a new release with `source_attempt`.
The original group order/content hash must match. It does not rewrite the old
checkpoint contract or add append support to it; use explicit adapter warm-start
for a new stage. Warm-start freezes a new initial policy and cannot reuse
base-policy probabilities unless they match that policy's adapter.

An interrupted writer leaves a hidden `.ID.staging` directory, never a consumable
published ID. Retry with new prepared/reference/extension and attempt IDs; these
scopes intentionally do not offer in-place recovery. Preserve stages for
diagnosis. An orphan sealed prepared artifact from a failed extension request
remains valid but cannot authorize append without a sealed request. A failed
reference import prevents updates and leaves the parent unchanged. Never edit a
receipt or reuse an incomplete/failed attempt ID. Back up and restore only this
change's code diff for rollback, preserve all artifacts, and restore the source
implementation matching the checkpoint you intend to resume. Restoring an older
reader does not make it capable of consuming the new schema.

CPU/tiny-Qwen and short native-token 9B smoke tests establish format, cursor and
state inheritance behavior. They do not establish long-context GPU capacity or
business-model quality. Test evidence and timing limits are recorded separately.


#### 8x16 indexed execution

The [8x16 protocol](../EXECUTION.md#两轮-distinct-parent-816) retains complete indexed 32-branch
parent artifacts and their original IDs, scores, advantages and token storage.
The preparation service adds a separate frozen execution plan and binds its
hash into the published schedule. Readers validate this identity and native
parent metadata before model loading. During execution, a rank reads only its
selected 16 completions; remote-rank groups use metadata placeholders. Reference
parent chunks are memory-mapped, and only selected target arrays are validated
and copied for the active half. Receipt verification can still scan all sealed
bytes for integrity. No source dataset or old prepared artifact is rewritten.
The new schedule supports exact resume; append execution is explicitly rejected.

## 验证范围

CPU 验证覆盖 sidecar、coverage plan、缺失补评调度、schema 3 合并、可携带校验、输入身份拒绝和中断恢复。真实模型的 raw 概率差异、GPU 数值和吞吐需要固定 token、固定身份和分别声明的验收阈值。
