# DyConGlobal token-aligned online hidden capture plan

## 1. Goal and verified input facts

The capture product must support a deterministic join between one hidden-state row, the exact token that produced that row, the original rollout trajectory, the assistant reasoning step, and downstream labels. The join must remain valid after batching, prefix-cache reuse, CUDA Graph replay, asynchronous export, and artifact repacking.

The current migrated scope contains 1,104 trajectory files and 113,217 assistant reasoning messages. All 1,104 trajectories contain reasoning newlines. Raw reasoning text contains 293,406 single-newline runs, 228,453 double-newline runs, and 6,534 triple-newline runs. Therefore a selector that captures only the `\n\n` token is incomplete.

For the current Qwen3.5/3.6 tokenizers, the observed encodings are:

```text
`\n`       -> [198]
`\n\n`     -> [271]
`\n\n\n`   -> [1358]
`<think>\n` -> [248068, 198]
`</think>` -> [248069]
```

The implementation must still derive newline candidates from the tokenizer fingerprint at runtime. BPE tokenization is context dependent, and future checkpoints may use different IDs or merge a newline with adjacent text.

## 2. Rollout and steering precision as a unique configuration concern

Add one serving-level precision option to the unique `config.yaml`:

```yaml
serving:
  dtype: float16  # `float16` or `bfloat16`
```

`serving.dtype` controls the model computation dtype used by the managed vLLM rollout server. It applies identically to baseline rollout and EasySteer steering because both use the same serving process and model instance. It must be included in the resolved-config hash and run snapshot, and the server launch receipt must record the effective dtype. The EasySteer vector payload dtype remains a separate implementation detail; a `float32` steering vector does not change the model computation dtype.

The existing `hidden.dtype` remains the storage/replay dtype for offline hidden extraction and must not be confused with `serving.dtype`. Online capture metadata must record both when both are present. When capture is enabled, the model computation dtype still comes from `serving.dtype`; capture serialization may use a separately validated storage dtype.

The precision acceptance matrix is:

| Rollout mode | BF16 | FP16 |
|---|---:|---:|
| baseline, steering disabled | required | required |
| baseline, EasySteer steering enabled | required | required |

For each cell, use the same checkpoint, prompt set, seed, sampling parameters, tool parser, and container images. Verify server startup, model dtype attestation, logits/token generation, tool-call format, rollout completion, patch submission, steering activation, and absence of NaN/Inf. Exact token equality across BF16 and FP16 is not required, but a capture-disabled baseline and its corresponding capture-enabled run must preserve the expected protocol and task behavior.

After the matrix is accepted, all subsequent rollout, steering, online FULL-graph capture, and numerical replay tasks in this project use `serving.dtype: float16`. BF16 remains available only for an explicitly named precision experiment with its own config hash and receipt. This policy prevents BF16 products from being mixed into the FP16 training/evaluation cohort.

The canonical data flow will be:

```text
trajectory bytes
  -> exact token trace
  -> online capture row labels
  -> hidden event artifacts
  -> token/hidden join index
  -> difficulty/action training examples
```

The trajectory hash, model fingerprint, tokenizer fingerprint, capture configuration hash, and runtime hash are part of every stage's identity.

## 3. Coordinate system and alignment contract

Every token event receives coordinates from the original model request. Coordinates must never be inferred from HTTP response order or batch order.

| Field | Meaning |
|---|---|
| `trajectory_id` | Stable rollout ID, e.g. `r-...` |
| `trajectory_sha256` | SHA-256 of the canonical trajectory artifact |
| `task_id` | SWE-bench instance ID |
| `run_id` | Source rollout run ID |
| `step_id` | Zero-based assistant/tool interaction step |
| `message_index` | Original message index in the trajectory |
| `request_id` | Internal vLLM request ID when captured online |
| `request_token_index` | Token index inside the exact prompt/input sequence processed by this forward |
| `absolute_position` | Position in that request's causal sequence, including prompt tokens |
| `phase` | `prompt` or `generation_input` |
| `generation_index` | Zero-based generated-token index when the row is a decode input |
| `token_id` | Exact integer ID used by the model |
| `token_text` | Token text reconstructed from the tokenizer and stored for selected events |
| `token_ref` | Immutable tuple `(trajectory_id, step_id, phase, absolute_position, token_id)` |
| `sample_id` | Unique hidden event ID; never derived from array order |

The key semantic rule is that a hidden row represents the model state after processing an input token at `absolute_position`. For the prompt `<think>` row this is a prefill position. For a generated reasoning/newline/close token this is a later decode input position. The token that is merely predicted at the end of a forward has no hidden row until a follow-up forward processes it.

For each capture row, the exporter must preserve both the vLLM row label and the reconstructed trajectory token reference. A join is valid only when all of the following match:

```text
request_id
absolute_position
 token_id
trajectory_sha256
step_id
prefix_sha256 or exact request token sequence hash
```

## 4. Canonical token trace artifact

The current enrichment samples store prefixes and anchor labels, but they do not provide a complete per-token index for all reasoning tokens. Add a separate immutable token trace product under enrichment:

```text
enrichment/outputs/<dataset_id>/
  <rollout_id>.samples.json          # existing anchor samples and labels
  rollouts/<rollout_id>/
    token_trace.npz                  # numeric token and coordinate arrays
    token_trace.json                  # schema, hashes, tokenizer evidence, block table
    token_texts.jsonl                # optional debug text for every token or selected tokens
```

`token_trace.npz` contains aligned arrays of equal length `N`:

```text
token_ids                  int32[N]
step_ids                   int32[N]
message_indices             int32[N]
absolute_positions          int64[N]
request_token_indices       int32[N]
generation_indices          int32[N]    # -1 for prompt tokens
phases                      uint8[N]    # prompt=0, generation_input=1
block_ids                   int32[N]    # -1 outside reasoning blocks
reasoning_offsets           int32[N]    # -1 outside reasoning content
reasoning_lengths           int32[N]    # -1 when block is incomplete
newline_counts              int16[N]    # 0 for non-newline events
special_kinds               int8[N]    # ordinary/open/newline/close/other
```

`token_trace.json` contains:

```json
{
  "schema_version": 2,
  "trajectory_id": "r-...",
  "trajectory_sha256": "...",
  "model_fingerprint": "...",
  "tokenizer_fingerprint": "...",
  "tokenizer_path": "...",
  "evidence": "exact_prompt_and_output_ids",
  "token_count": 12345,
  "steps": [],
  "reasoning_blocks": [],
  "newline_token_inventory": [],
  "files": {}
}
```

The token trace is generated from exact `prompt_token_ids` and `output_token_ids` whenever those fields are present in the trajectory. If they are absent, the existing chat-template reconstruction is allowed only with an explicit `alignment_evidence` value such as `template_and_prompt_count` or `text_reconstructed`. Weak-evidence traces are retained for audit but excluded from supervised difficulty training by default.

The trace builder must validate that the reconstructed prompt matches the recorded prompt token count and that `<think>` and `</think>` markers decode exactly. It must not silently re-tokenize a reasoning string and call that exact evidence.

## 5. Reasoning block model

A reasoning block is the causal token interval belonging to one assistant reasoning segment. Normally one assistant step contains one block, but the schema supports multiple marker occurrences.

Each block record contains:

```json
{
  "block_id": 12,
  "step_id": 7,
  "open_token_span": {"start": 1840, "end": 1842},
  "content_span": {"start": 1842, "end": 2391},
  "close_token_span": {"start": 2391, "end": 2392},
  "content_token_count": 549,
  "newline_event_count": 23,
  "complete": true,
  "evidence": "exact_output_ids"
}
```

The content span excludes the `<think>` and `</think>` control markers. If the close marker is generated but no follow-up forward processes it, the block is marked `complete=false` for hidden capture even if the text parser can see the marker. No preceding token is substituted for the unavailable close hidden state.

For every token inside a complete content span:

```text
reasoning_offset = absolute_position - content_start
reasoning_length = content_token_count
reasoning_fraction_start = reasoning_offset / max(reasoning_length, 1)
reasoning_fraction_center = (reasoning_offset + 0.5) / max(reasoning_length, 1)
block_remaining_reasoning_tokens_after = content_end - (absolute_position + 1)
block_remaining_reasoning_tokens_including_current = content_end - absolute_position
```

Both fraction definitions are stored. Downstream experiments can choose start-based or center-based percentages without rebuilding alignment.

For a newline token containing multiple line breaks, `newline_count` is derived from its tokenizer text. A token such as `1358` representing `\n\n\n` is one model token and one hidden row, but three newline characters. The artifact must not expand it into three hidden rows.

## 6. Step and remaining-difficulty labels

The existing enrichment code has step-level labels, but the new token-aligned dataset must make their convention explicit.

For a trajectory with `T` assistant reasoning/tool steps and current step `s`, store:

```text
remaining_steps_including_current = T - s
remaining_steps_after_current = max(T - s - 1, 0)
remaining_steps_log1p = log1p(remaining_steps_after_current)
```

For a token at position `p` in a reasoning block, store two different quantities:

```text
block_remaining_reasoning_tokens_after
= current_block.content_end - (p + 1)

future_reasoning_tokens_after_current_block
= sum(content_token_count for every later reasoning block in trajectory order)

remaining_reasoning_tokens_after
= block_remaining_reasoning_tokens_after
  + future_reasoning_tokens_after_current_block

remaining_reasoning_tokens_log1p
= log1p(remaining_reasoning_tokens_after)
```

`remaining_reasoning_tokens_after` is a trajectory-level label: it includes the ungenerated reasoning tokens after the current token in the current block plus all reasoning tokens in every later reasoning block, including later blocks in the same assistant step if multiple blocks exist. `block_remaining_reasoning_tokens_after` is retained as a separate local diagnostic label and must not be used as the global remaining-reasoning target.

Outcome-dependent labels are available only when the catalog binds a completed evaluation and a boolean `resolved` value. For unresolved, truncated, infrastructure-failed, or unsubmitted trajectories, the outcome-conditioned label is `null`; the raw terminal position and token counts remain available as descriptive metadata.

The joined training row should include both raw and transformed labels:

```json
{
  "event_id": "...",
  "resolved_label_available": true,
  "resolved": true,
  "remaining_steps_after": 3,
  "remaining_steps_log1p": 1.386294,
  "block_remaining_reasoning_tokens_after": 41,
  "remaining_reasoning_tokens_after": 184,
  "remaining_reasoning_tokens_log1p": 5.220356,
  "reasoning_fraction_start": 0.664,
  "reasoning_fraction_center": 0.665,
  "label_sources": ["catalog_evaluation", "exact_output_ids"]
}
```

This keeps step difficulty, token difficulty, total remaining reasoning length, and normalized block position as separate targets rather than conflating them.

## 7. Three-token online capture design

The online capture stream will use one native EasySteer `tact_states` stream with a union selector:

```python
SelectSpec(
    prompt_positions=[-2],
    generation_tokens=[
        *newline_token_ids,
        *close_token_ids,
    ],
)
```

The native selector is reused for batching, row labels, TP assembly, paging, and FULL graph replay. No new scheduler, KV manager, GDN restoration, or CUDA Graph implementation is introduced.

The selector has three logical event types:

- `think_open`: prompt position `-2`, validated as `<think>` followed by the template newline;
- `reasoning_newline`: any generated input token in the tokenizer-derived newline inventory, with `newline_count` and `token_text` recorded;
- `think_close`: exact close-marker sequence, normally token `248069` for the current Qwen models.

The native selector is intentionally a superset. A request-level state machine filters newline events to the interval after the current `<think>` and before the first exact close marker. Newline candidates after close are discarded from the training index. This avoids changing the selector or graph at every decode step while preserving reasoning-only semantics.

If close markers can be multi-token for a future tokenizer, candidate close IDs are captured and matched with a rolling exact sequence matcher. Partial matches are marked as candidates until the full marker is confirmed.

The request metadata must include:

```json
{
  "dycon_capture": {
    "session_id": "cap-...",
    "task_id": "...",
    "run_id": "...",
    "rollout_id": "...",
    "step_id": 7,
    "anchors": ["think_open", "reasoning_newline", "think_close"],
    "tokenizer_fingerprint": "...",
    "newline_policy": "all"
  }
}
```

The hard-coded `capture_select` containing only `prompt_positions=[-2]` must be replaced by a selector built from this request's tokenizer inventory. The server records the effective selector and its hash in the capture receipt.

## 8. Hidden event artifact organization

The raw online capture remains append-only and asynchronously exported. Each selected row gets a stable event record, independent of array or queue order:

```text
hidden/online/<capture_id>/
  session.json
  receipt.json
  queue.jsonl
  raw/
    <event_id>.npz
    <event_id>.json
  rollouts/<rollout_id>/
    events.npz
    events.jsonl
  joins/
    token_hidden_index.jsonl
    token_hidden_index.npz
```

`raw/<event_id>.json` contains:

```json
{
  "event_id": "...",
  "sample_id": "...",
  "trajectory_id": "r-...",
  "trajectory_sha256": "...",
  "task_id": "...",
  "run_id": "...",
  "rollout_id": "...",
  "step_id": 7,
  "message_index": 14,
  "request_id": "...",
  "phase": "generation_input",
  "absolute_position": 1903,
  "generation_index": 63,
  "token_id": 271,
  "token_text": "\\n\\n",
  "anchor": "reasoning_newline",
  "token_ref": ["r-...", 7, "generation_input", 1903, 271],
  "block_id": 12,
  "reasoning_offset": 61,
  "reasoning_length": 549,
  "graph_execution": "full",
  "cache_hit_tokens": 1840,
  "capture_stage": "post_steering",
  "status": "persisted"
}
```

The NPZ contains hidden arrays indexed by explicit site/layer IDs, for example `hidden[layer_0]`, `hidden[layer_1]`, or a packed `[events, layers, hidden_size]` array in the rollout pack. The sidecar records layer ordering, site semantics, storage dtype, and checksum.

The raw per-event representation is the recovery boundary. A packer may later concatenate events for training, but it must preserve the original event IDs and sidecar checksums. Repacking must never change token order or hidden row identity.

## 9. Join index for downstream training

Create a materialized join product under `difficulty/datasets/<index_id>/`:

```text
difficulty/datasets/<index_id>/
  rows.jsonl
  rows.npz
  receipt.json
  splits.json
```

One row corresponds to one hidden event and contains:

```text
trajectory/task/run identifiers
trajectory_sha256
step_id/message_index
block_id
absolute_position/generation_index/token_id/token_text
anchor and newline_count
hidden_event_id and hidden artifact path
layer/site mapping
alignment_evidence and alignment_status
remaining step labels
remaining reasoning-token labels
reasoning fraction labels
resolved/evaluation labels
```

The index builder performs a strict join:

1. Verify the catalog trajectory checksum.
2. Verify the enrichment token-trace checksum.
3. Verify the hidden sidecar checksum.
4. Match `trajectory_id`, `trajectory_sha256`, `step_id`, `request_id`, `absolute_position`, and `token_id`.
5. Recompute the prefix/request hash and compare it with the capture row.
6. Reject duplicate event IDs or multiple hidden rows claiming the same `(request_id, absolute_position, token_id, site_set)` unless explicitly marked as a rerun.
7. Write a receipt containing all source hashes and the exact selector/tokenizer configuration.

Rows with weak token evidence, missing close follow-up, or unresolved outcome labels are retained with status fields but excluded from a supervised target by default. A later experiment may opt them in explicitly.

## 10. Manifest and receipt changes

Existing historical migration and offline products remain unchanged. Add an `online_captures` section to both `enrichment/manifest.json` and `hidden/manifest.json`.

Each capture entry records:

```text
capture_id/session_id
model and tokenizer fingerprints
trajectory/run/task bindings
anchor inventory and effective selector hash
newline token inventory
capture graph mode and fallback counters
raw and packed artifact receipts
queue/export status
alignment counts: exact, weak, missing, duplicate, filtered-after-close
```

`baseline/manifest.json` remains the source of task, rollout, and evaluation identity. It must not be duplicated in the token trace or hidden index; those products only reference its rollout ID and checksum.

## 11. Validation and failure policy

The implementation must provide deterministic audit commands before model training:

```text
trajectory-token-audit DATASET_ID
hidden-token-join-audit CAPTURE_ID INDEX_ID
```

The audit reports:

- trajectory files and reasoning messages scanned;
- exact versus reconstructed token evidence;
- expected versus captured open/newline/close events;
- token-ID and absolute-position mismatches;
- missing close follow-up cases;
- duplicate or out-of-order rows;
- hidden checksum failures;
- post-close newline rows filtered;
- graph replays, eager fallbacks, cache-hit tokens, and export backlog.

Failure handling:

- token mismatch: mark the event `alignment_failed` and exclude it from training;
- missing exact output IDs: mark `weak_alignment`, preserve for audit;
- close without follow-up: mark `no_followup_forward`, do not fabricate hidden state;
- export queue overflow: record `export_backpressure`, continue rollout under the default policy;
- artifact checksum mismatch: fail the join/index build rather than silently repairing it.

## 12. Implementation phases

### Phase A: token trace and schema

1. Add token-trace schema and tokenizer inventory builder.
2. Build traces for the existing 1,104 trajectories.
3. Produce an audit showing exact/weak evidence and newline token inventory.
4. Add tests for prompt/generation coordinates, merged newline tokens, and reasoning block boundaries.

### Phase B: online selector and labels

1. Replace the hard-coded open-only `capture_select` with the union selector.
2. Add request metadata for tokenizer fingerprint and selector hash.
3. Extend native row classification and the per-request reasoning state machine.
4. Add close follow-up and post-close filtering tests.

### Phase C: asynchronous artifacts and manifests

1. Extend the export queue sidecar with token references and block labels.
2. Add raw event receipts and recovery scanning.
3. Add online capture entries to both manifests.
4. Add a deterministic raw-to-packed event builder.

### Phase D: difficulty/action join index

1. Build `token_hidden_index` from exact source checksums.
2. Materialize step, token, reasoning-length, and normalized-position labels.
3. Add split logic by trajectory/task, preventing tokens from the same rollout crossing train/validation splits.
4. Add visualization utilities that plot hidden-derived predictions against reasoning offsets and newline fractions.

### Phase E: precision and GPU acceptance

1. Execute the BF16/FP16 precision matrix for baseline and steering-enabled serving described in Section 2.
2. Set the canonical follow-up config to `serving.dtype: float16`; record the precision policy in every later run snapshot.
3. Run one Qwen3.5-9B request with only `<think>` and a small layer set.
4. Run the same request with all three event types and mixed batch lengths.
5. Verify actual FULL graph replay and no unexpected eager fallback.
6. Test cold/warm safe-boundary cache, chunked prefill, TP=1/2, dense/MoE, and steering in-graph.
7. Compare capture-enabled and capture-disabled rollout text, tool calls, finish reasons, and patch outputs.
8. Scale to all layers only after event counts, positions, checksums, and asynchronous export are stable.

### Phase F: FP16 vLLM replay numerical acceptance

This is a separate mandatory gate. It compares online FULL-graph capture against a vLLM-native replay capture, never against an HF/Transformers forward for this acceptance.

1. Use the same checkpoint, tokenizer, exact prompt/output token IDs, layer/site list, TP layout, sampling seed, and `serving.dtype: float16` for both paths.
2. Replay the exact captured prefixes through the existing vLLM native capture path, using the same `tact_states` stream and row labels. Do not compare rows selected by array order.
3. Join online and replay rows by `token_ref`, layer/site ID, token ID, absolute position, trajectory hash, and request/prefix hash.
4. Compute per-layer and per-event cosine similarity, relative L2 error, absolute max error, and finite-value checks. Report cold-cache, warm-cache, chunked-prefill, mixed-batch, TP=1/2, dense, MoE, and steering cases separately.
5. Use the following initial gates, calibrated against the existing FP16 vLLM validation reports: dense models require cosine >= 0.999 and relative L2 <= 0.05; MoE/TP cases require cosine >= 0.997 and relative L2 <= 0.08. Any model-specific baseline may tighten these limits, but no result may be accepted solely because it agrees with HF.
6. Verify that enabling capture does not change generated token IDs, finish reasons, tool calls, or submitted patches for the same FP16 rollout seed. A numerical pass with changed rollout behavior is a failure.
7. Store the comparison report, source receipts, dtype, runtime hash, selector hash, and thresholds under `hidden/evaluation/outputs/<evaluation_id>/` and bind it from the online capture receipt.

A capture implementation is eligible for the full cohort only when the FP16 vLLM replay gate, token alignment audit, graph replay audit, and rollout behavior comparison all pass.

## 13. Rollback and compatibility

`capture.enabled: false` remains the default rollback switch. Disabling it stops new sessions, restores the ordinary request body and cache lookup, and leaves historical baseline, enrichment, hidden, and difficulty artifacts intact. Online products are versioned separately, so a failed capture experiment cannot invalidate existing offline products.

No downstream estimator may consume a hidden vector without an `alignment_status=exact` row unless the experiment explicitly opts into weak evidence. This rule prevents tokenizer reconstruction errors from silently becoming remaining-difficulty labels.
