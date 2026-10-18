# Online FULL-graph hidden capture

The optional online path is controlled by the `capture` block in `config.yaml`.
Set `capture.enabled: false` for an ordinary rollout; the checked-in remote configuration
currently enables capture. The four modes share the same official CLI and common
model, parser, generation, agent and container configuration. Enabling capture
it requires managed serving and starts a localhost control plane alongside the
managed vLLM process.

## Control plane

The control plane exposes:

- `POST /dycon/capture/session/start` with `{metadata, graph_mode, prefix_cache_policy}`
- `POST /dycon/capture/request/bind` to bind vLLM's internal request ID
- `GET /dycon/capture/status` and `/dycon/capture/session/{id}/{status,results}`
- `POST /dycon/capture/session/{id}/stop`

Request metadata is explicit (`task_id`, `run_id`, `rollout_id`, `step_id`, and
anchors). It is never inferred from HTTP response order or batch order. The
rollout adapter adds the metadata under `extra_body.dycon_capture` only when the
feature is enabled.

## Anchor and cache semantics

The supported Qwen3.5 online configuration selects generated reasoning newline
tokens and `</think>`. `<think>` is excluded because it is part of the prompt
and the current GDN backend supports FULL capture graphs only for decode.
For experiments requesting an opening anchor, `<think>\n` is validated as token
IDs `[248068, 198]`, with the opening marker at prompt position `-2`.
`</think>` is accepted only when it is an input suffix of a later forward; an
immediate response termination is recorded as `no_followup_forward`.

`safe_cache_hit_length()` caps prefix-cache reuse at the boundary before the
capture target. The controller does not implement a scheduler, KV/GDN manager,
page allocator, TP assembly or graph compiler; those remain EasySteer/vLLM
responsibilities. `start_online_capture()` forwards the FULL graph request to
the native capture RPC.

## Asynchronous export

The native plugin hands off uncompressed temporary NPZ files on local storage.
The bounded asynchronous writer compresses each tensor exactly once under
`hidden/spool/<run>/`; a project-wide byte ledger includes handoff, queue,
temporary archive and metadata reservations and shares disk admission with
Docker image loading. The default budget is 50 GiB, with 80%/60% admission watermarks.

Complete prompt/output IDs are stored once per native request in an immutable
`requests/<request_key>.json` ledger. Trajectories and events reference that ledger
and its SHA, together with query/API attempt/native IDs and exact token coordinates.
Both successful responses and FormatError responses participate in final alignment.

Existing compressed NPZ bytes are sealed into standard uncompressed tar shards
(about 128 MiB; session end seals a short final shard). Two bounded, independent
uploader processes copy to NAS `.part` files, verify SHA/size, rename atomically,
and acknowledge publication before local tensor bytes are released. Sealed event
indices are uploaded per shard, rather than as per-token NAS sidecars. Final data is
under `<nas_root>/hidden/online/<run>/`; no NAS path is required to complete a model query.
`hidden.artifacts` reads both legacy loose NPZ and new shard/member locators.

Upload failures retain local data as `pending_upload`; `capture-recover RUN` uses
the immutable run snapshot to retry publication without new model requests.
Hard-mounted NFS stalls occupy their existing upload slots; parent shutdown is bounded
and recovery refuses to replace a still-running uploader. A session is `complete`
only after exact token coverage and publication of its ledgers, indices and finalization.
`export.enabled=false` drains native buffers but creates no hidden dataset.
Queue/storage failures use the configured `record_and_continue` or `block_rollout`
policy; agent outcome and capture outcome remain separate.
`enrichment/manifest.json` and `hidden/manifest.json` receive an
`online_captures` entry containing the task/run/rollout binding, anchor, graph
execution, cache boundary evidence and export state.

Run all four modes through `python -m core.cli --config config.yaml rollout RUN`.
Change the two independent feature switches in the same config; do not launch
production via historical speed-test scripts. Startup verifies a real bash call
in a NAS-loaded container and, for capture, both configured anchors, finite layers,
FULL decode evidence and NAS artifact readback before admitting the task cohort.
## Qwen3.5 decode capture and prefill scheduling

The earlier attribution of token-0 output to GDN capture in general was not
supported by an isolated control. FP16 native tests on 2026-10-18 demonstrated
working Qwen3.5 decode capture, including 92 FULL capture graph replays, finite
raw logits, and identical 48-token natural continuations with capture on/off.

The reproduced failure requires forcing single-token prompt chunks via
`--long-prefill-token-threshold=1`. This is a request-wide scheduling option,
not an instruction to process only `<think>` separately. The GDN metadata
builder classifies one-token chunks as decode; the decode kernel consumes
existing recurrent/conv state while normal fresh-prefill initialization is in
the prefill branch. Failing tests showed all-NaN hidden states and raw logits.
The allowed-token mask was present, but the remaining allowed logit was NaN;
the sampler returned token 0. No APC reads or asynchronous exporter were
needed to reproduce this failure. The first individual kernel producing NaN
was not instrumented.

`capture.prefill_token_chunk_size: 0` now delegates prefill scheduling to vLLM
and omits the threshold override. The project no longer changes CUDA graph
dispatch to treat prompt chunks as uniform decode. Decode-only anchors use
`FULL_DECODE_ONLY`: ordinary model-supported prefill followed by FULL decode
capture. Decode capture preserves native APC reads and explicit caller skips.

Managed serving rejects the known-invalid single-token Qwen configuration.
The `allow_unsupported_gdn` diagnostic flag applies to prompt-anchor FULL
experiments, not to normal decode capture. See
[the diagnostic report](capture-zero-token-diagnosis-20261018.md) for controls
and the limits of the current acceptance results.
