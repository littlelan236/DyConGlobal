# Neutral raw probability stream reader

Canonical final artifacts use scope `rl-probability`, schema 3 at
`ROOT/RL/probabilities/ID/`. `ProbabilityStream` reads that self-contained format
as well as the legacy source adapters described below. Coverage planning,
missing-only replay, merge/verification and migration examples are documented in
[Canonical raw probability artifacts](dataset/PROBABILITY_ARTIFACTS.md).

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
[streaming recovery](STREAMING_RESUME.md).

The schema-2 source adapters below remain readable for migration and direct legacy
training. Their identity/index fields should not be copied verbatim as schema 3.
See [CPU acceptance and limits](PROBABILITY_ALIGNMENT_VALIDATION.md).

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
lifecycle; see [configuration, receipts and verification scope](../docs/ONLINE-RAW-LOGPROBS-CN.md).
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
