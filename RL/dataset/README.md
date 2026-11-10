# Offline GRPO datasets

`rl-dataset build` freezes explicit SWE-bench parallel-consensus sources into step groups. Every group keeps all native candidates for one actual prompt. The prompt is context only; `mask_spans` covers the original completion token coordinates, including reasoning and action tokens.

The builder never invents reward, advantage, KL, or loss values. Groups with missing native evidence remain in `exclusions.jsonl` and do not become empty completions. `data_ready` is the only readiness state produced here; dataset-level `grpo_ready` stays false because the versioned training service performs its own objective eligibility scan. No business dataset is created by installing or testing this code.

Training uses the existing `dycon lora-train --training-config YAML` entrypoint;
see [offline GRPO](../GRPO.md) and [engineering acceptance](../GRPO_VALIDATION.md).
It consumes all eligible complete groups from one sealed dataset, without a split
manifest, freezes advantages and group order, and uses raw T=1 initial-policy target
logprobs for PPO ratios and token K3. Processed replay estimates are audit inputs,
not the denominator. Source tokens and actual group prompts are checked again at
training preflight even when the dataset was built with `audit.enabled=false`.

`audit.enabled` defaults to `false`. In that mode source selection/content hashes, publication/session/decision/ledger identity comparisons, and native token hash comparisons are skipped, as requested for sources already audited during rollout. The chosen values are frozen in `dataset.json` and `summary.json`. Set it to `true` to run those checks during build and subsequent source checks.

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

`check --level metadata` verifies local receipt and group/member structure. `sources` also verifies frozen dependencies when source audit is enabled; `full` additionally reads native evidence under that same audit setting. None of these options overrides frozen `audit.enabled=false`. Materialization is optional: replay and the loader can read native ledgers directly.

Historical behavior probabilities are a separate replay artifact. The replay config freezes the native vLLM trace backend and binds one dataset receipt. Root `temperature`, `top_p`, and `top_k` are read from the frozen project config; replay YAML cannot override them. Unsupported historical grammar, thinking-budget, bad-words, or speculative requests must be rejected before model load.

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

Replay loads the historical model and dtype from frozen source snapshots, verifies registered weight contents, reserves the explicitly configured idle GPUs, and dispatches into the isolated pinned latent-vLLM interpreter. Native V2 trace replay uses `logprobs_mode=processed_logprobs`, `logprobs=0`, and the exact completion IDs; values are read by target token ID. Generation stops at the trace length. Batches retain at most `runtime.max_num_seqs` outputs. Completed branches are written as little-endian float32, checked and journaled before continuation; no full-vocabulary probabilities are stored. `runtime.subprocess_seconds` bounds the worker, whose process group is cleaned up before GPU reservation release.

The backend rejects grammar/reasoning-close, thinking budgets, bad words, string stops, speculative decoding, active steering, adapters, quantization and greedy temperature zero. Historical stop token IDs, min_tokens, penalties, min_p, ignore_eos, allowed tokens and logit bias are retained when present. Request overrides take precedence over source snapshots. Missing historical runtime defaults are recorded as unknown, with the explicit defaults used for estimation. Root sampling conflicts block replay. Dry-run checks dataset, policies and sequence budgets without loading models; native runtime validation runs in the isolated child before model load.

Outputs have status `replay_estimate` and `historical_behavior_verified=false`; they are never presented as original recorded probabilities. `-inf` is preserved and reported with completion positions, and makes its complete group unavailable to the probability loader. NaN, positive infinity, missing entries or changed output IDs fail publication. `iter_groups(..., replay=PATH)` rejects estimates by default; explicit `allow_replay_estimates=True` opts in. This flag does not make zero-support groups usable. The loader returns `behavior_logprobs` separately from labels; the existing CE trainer does not implement GRPO.

## Progress and recovery

For an interrupted build, run `.venvs/runner/bin/dycon rl-dataset build --dataset-config DATASET.yaml --recover`. For replay, use the replay recovery command above. Do not combine `--dry-run` with `--recover`.

Build concurrency is bounded by `storage.ledger_read_workers` (default 8, maximum 32); workers return results and only the main thread writes inventories. `progress.json` updates after each group. Build `--recover` verifies its config, implementation, frozen selection, committed file prefixes and source hashes when audit is enabled. It resumes at a completed rollout boundary, discarding an interrupted rollout tail. Legacy staging from earlier code has no compatible checkpoint and cannot be recovered by this interface. Replay recovery checks dataset, policy, runtime, implementation and completed float32 shards and uses the original root-sampling snapshot. Same IDs are immutable; neither build nor replay silently resumes or overwrites.

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

`iter_groups` defaults to complete data-ready groups in frozen order. Optional `cache`, `start_group` and `max_sequence_length` support mmap tokens, a deterministic group cursor and explicit length rejection. The first completion target is at `len(prompt)`; the collator does not shift targets, and causal prediction is shifted exactly once by the consuming loss. Replay values align one-to-one with completion tokens. `collate_group` leaves group metadata and behavior probabilities on the loader records; a future objective must receive them explicitly. There is no split command, reward, reference/KL, advantage or GRPO optimizer integration in this module.

## Tests and validation scope

Tests use temporary synthetic captures only:

```bash
.venvs/runner/bin/python -m pytest tests/test_rl_dataset.py tests/test_rl_dataset_replay.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DYCON_REPLAY_TEST_GPU=0 .venvs/latent-vllm/bin/python -m pytest tests/test_rl_dataset_native_runtime.py -q
```

The opt-in GPU test verifies the pinned API, sampler/trace kernels and a generated tiny Llama checkpoint. It does not validate Qwen GDN, long context, TP or historical runtime equivalence. The installed sampler has kernel-dependent tie behavior: a batched Triton top-k=1 keeps one tied maximum while the small-batch PyTorch mask keeps both. A regression test records this limitation; numerical validation on an unrelated tiny model does not promote historical replay estimates.
