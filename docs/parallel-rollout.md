# Parallel complete-response rollout

Implementation date: 2026-10-19. Uses the existing `dycon rollout` entry point.
The default configuration and old snapshots retain the original single-branch
path. No `difficulty/` code or production config is changed by this implementation.

## Execution

Each selected logical query submits B independent native `n=1` completions to
the same managed engine. These reuse `model_config`, LiteLLM transport, native
chat template/tool/reasoning parsers, capture stream and vLLM batching/APC/TP.
Each completion generates the full reasoning and action text. The original
agent executes only the selected message's tools. Rejected messages are audit
data and never enter the next model context.

Native `n=B` is deliberately not used: the existing capture wrapper records
only `outputs[0]`, while vLLM expands parent requests into children internally.
Concurrent `n=1` requests preserve exact per-request token ledger identity.
The transport is fixed in decision evidence; no silent runtime switching.

Close hidden is collected from its subsequent forward, without probes/replay.
Candidates with no follow-up forward, absent/invalid hidden, truncated output,
invalid score or malformed tool calls cannot win by score. Valid candidates
rank by `(score, branch_id)`. A score failure can choose the first already
generated valid response, or stop according to policy. No fallback resampling.
All malformed outputs preserve the existing agent FormatError correction path.
All branches contribute to recorded usage/cost, not just the winner.

## External predictor integration boundary

The scorer implementation is owned by the user's separate difficulty work.
There is no Ridge/FFN formula, training or parameter loader here. The serving
adapter accepts an external JSON manifest with this provisional contract:

```json
{
  "schema_version": 1,
  "loader": "USER_PROVIDER_MODULE:load",
  "model_fingerprint": "CHECKPOINT_FINGERPRINT",
  "sites": [15],
  "hidden_size": 4096,
  "target": "USER_TARGET",
  "endpoint": "submission",
  "files": {"provider-and-model-file": "SHA256"}
}
```

The explicit provider must be importable in the serving environment. Its
`load(artifact_directory: Path, contract: dict)` returns an object with
`predict(batch)` accepting `[N, selected_public_sites, hidden_size]` and returning
finite `[N]` scalar scores. Feature preprocessing, trained parameters, dtype
and target transformation are the provider's responsibility. The currently
bounded CPU worker can submit an N=1 batch as each close arrives. It runs
outside the asyncio loop and CUDA Graph. A stuck call cannot create unlimited
threads/queued features, and worker shutdown does not wait for it indefinitely.

The manifest and listed payload hashes are checked during config resolution
and service startup. Resume uses the manifest hash from the saved snapshot.
Include all provider code/model payload needed to identify the scorer in the
hash list, and validate tokenizer/runtime/feature evidence in the external
loader. This interface can be adapted to the actual user implementation later;
it does not constitute completed difficulty integration.

An optional `runtime.serving.ridge_provider:load` bridge now reuses the existing
`difficulty.models.ridge_grid.load_model/predict_model` API without modifying
that module or implementing another prediction formula. Its contract adds
`model_file`, which must be present in `files`. It accepts one public site and
checks the feature width. Actual GPU debugging uses an existing completed Ridge
debug experiment; this is not a claim that the full training grid was completed.

## Configuration (do not copy old run snapshots)

Add to the sole project `config.yaml` only when the real provider is ready:

```yaml
parallel:
  enabled: true
  branch_count: 2
  policy: first_step
  every_k_steps: null
  estimator_ref: path/to/external-manifest.json
  max_inflight_branches: 16
  admission_timeout_seconds: 1800
  score_timeout_ms: 200
  decision_ttl_seconds: 28860
  persist_all_branch_close: true
  save_sites: [15]
  fallback_policy: first_valid
  validation_repetitions: 0
```

Policies are `first_step`, `every_k_steps` and `every_step`. The first is the
initial acceptance recommendation, not a restriction on the algorithm. B=1
bypasses parallel orchestration and predictor setup. `save_sites` must include
the provider sites and be a subset of `capture.layers`. The capture layer list
is the authoritative actual export selection: the implementation does not
silently suppress other saved layers/newlines. Capture must be enabled,
managed, exported and include `think_close`, with thinking enabled. Capacity
must accommodate one whole group and not exceed configured max_num_seqs;
application reservations bound active parallel groups with a finite wait.

Score timeout bounds the score-result wait after response generation; model
requests retain the original model/deadline timeout. Choose the reservation
TTL longer than the maximum in-flight request duration. A disconnected runner
can leave reservations until TTL/next admission. CPU queue capacity remains
bounded even when a scorer hangs. Degraded scoring is explicit in evidence.

Formal commands remain `dycon rollout RUN_ID`, `--resume`, and independent
`--check-only` acceptance runs. Existing restrictions on `--set` remain.
Config fields are drafts to be reconciled with the actual external provider.

For an independent real `--check-only` run, setting `validation_repetitions`
to 1–10 additionally performs rotated single/unscored B=2/scored B=2 requests,
and scored B=4 when configured branch_count is at least four. It uses the same
service and capture, does not execute generated benchmark tools, finalizes all
sessions to NAS, verifies actual FULL close evidence, and independently recomputes
every scored branch from published hidden using the external provider. Results
are stored under `startup/parallel-validation/`. This diagnostic is disabled
by default and does not change formal rollout policy. Natural query lengths and
shared GPU load must be reported alongside small-sample latency percentiles.
It also submits three B=2 groups concurrently to check whole-group admission,
cross-decision isolation and concurrent publication/finalization; active groups
remain limited by the configured branch capacity.

## Evidence and NAS

CaptureRequest/ledger/events carry `decision_id`, `branch_id`, `branch_count`,
`parent_prefix_sha256`, query and API attempt identity. Every generated branch
has an independent native request. A decision validates its scored close token,
position, sample ID and exact output receipt before any tool executes.

Data follows the existing asynchronous queue -> bounded spool -> tar -> NAS
copy/hash/atomic-rename chain. Scoring uses the same in-memory captured arrays
before staging export, without reading NAS or waiting for upload ACK. All
branch hidden uses the configured capture layers and anchors. NAS failures
remain capture failures/partial publication according to existing policy.

Additional files under `sessions/SESSION/parallel/`:

- `decisions/DECISION.json`: immutable all-candidate responses, validity,
  native IDs/ledger hashes, scores, external estimator identity, selected branch,
  sampling seeds, costs, decode and selection timing, fallback reason.
- `actions/DECISION-{started,completed,failed}.json`: separate execution evidence.
- `branch-index.json`: all candidates, selected flag and hash-bound close tensor
  locators, with unknown future workload labels explicitly null.
- `branch-coverage.json`: missing close rows per candidate.
- `branch-receipt.json`: hashes binding these records and final canonical SHA.

The existing spool session metadata publisher/recovery covers these nested
files. Tensors and ledgers are not rewritten after selection. The canonical
trajectory contains only the selected response, with a decision summary in
`extra.dycon_parallel`; accepted_native_request_ids keeps its original meaning.
Canonical enrichment excludes rejected requests. The all-candidate index shares
underlying shards and can be read with
`hidden.artifacts.read_branch_index(nas_root, session_id)` followed by
`read_arrays(nas_root, row['close_events'][i])`.

Selection, tool start/completion, trajectory termination, success evaluation
and all-branch publication are distinct statuses. No rejected branch receives
the winner's future labels. Branch-index labels remain null even for winners;
their observed labels remain available in the canonical pipeline. Counterfactual
continuations and bootstrap training are outside this implementation.

## FULL graph and validation limits

Managed serving retains `FULL_AND_PIECEWISE`, `steer_graph_mode=in_graph`,
native prefill scheduling and the existing capture implementation. No model
forward hook or offline eager adapter is introduced. Batch composition can
still change actual FULL/mixed/eager execution proportions. Only real per-event
graph evidence proves FULL replay; CPU mocks do not.

The official startup gate exercises parallel selection when enabled, executes
only the winner's real tool action, checks all-branch NAS close readback, and
retains existing FULL evidence checks. GPU performance comparisons require a
free independent resource allocation and a real external predictor. Do not
stop current rollouts or claim GPU/throughput acceptance from CPU tests.

The Python tests cover native identity, nonzero winner, FormatError isolation,
all-branch NAS hash/readback, canonical-only labels, partial coverage, transport
fallback/strict stop, bounded hung scorer, B=1/old config and actual plugin HTTP
lifecycle. Cross-environment tests complement runner's mini-swe-agent and
serving's FastAPI/vLLM dependencies.
