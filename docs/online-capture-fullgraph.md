# Online FULL-graph hidden capture

The optional online path is controlled by the `capture` block in `config.yaml`.
`capture.enabled: false` is the default and leaves the OpenAI data plane, rollout
request body, EasySteer steering and ordinary vLLM startup unchanged. Enabling
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

For the current Qwen templates, `<think>\n` is validated as token IDs
`[248068, 198]`, so the opening marker is selected at prompt position `-2`.
`</think>` is accepted only when it is an input suffix of a later forward; an
immediate response termination is recorded as `no_followup_forward`.

`safe_cache_hit_length()` caps prefix-cache reuse at the boundary before the
capture target. The controller does not implement a scheduler, KV/GDN manager,
page allocator, TP assembly or graph compiler; those remain EasySteer/vLLM
responsibilities. `start_online_capture()` forwards the FULL graph request to
the native capture RPC.

## Asynchronous export

The capture hook stages CPU arrays into a bounded queue. A background writer
atomically writes compressed NPZ tensors, JSON sidecars and a queue journal
under `hidden/online/<session>/` (the configured root is respected). Queue
backpressure records `export_backpressure` and continues rollout by default.
`enrichment/manifest.json` and `hidden/manifest.json` receive an
`online_captures` entry containing the task/run/rollout binding, anchor, graph
execution, cache boundary evidence and export state.

This implementation provides the tested control and export contract and the
native FULL capture request adapter. The final EasySteer worker hook must call
`OnlineCaptureController.bind()` and `.record()` with its captured tensor pages
after the pinned runtime's graph replay/fetch step.
## Qwen3.5 GDN acceptance result

On 2026-10-17 the managed FP16 Qwen3.5-9B service was started on GPUs 4--7 with TP=4 and an explicitly reserved KV cache. The service loaded the endpoint plugin, hooked three `tact_states` layers, and recorded native graph replays on all four TP workers.

The rollout behavior gate failed: enabling the native `tact_states` capture stream changed generated token IDs. Fixed-token requests returned token 0 or 1023 instead of the requested token. The failure remained for decode-only newline/close selectors, after disabling per-request selectors, disabling APC reads for capture requests, and setting the prefill chunk threshold to one. vLLM reports that Qwen GDN supports FULL graphs only for decode and downgrades prefill to FULL_AND_PIECEWISE; the capture hook path also changes logits in this pinned runtime. The online capture path must therefore block supervised rollout data for this runtime until the upstream EasySteer/vLLM GDN capture behavior is fixed and the FP16 behavior gate passes.
`capture.allow_unsupported_gdn` defaults to `false`. Managed serving rejects
Qwen3.5 GDN capture before starting a server, so an accidental experiment
cannot append corrupted trajectories. Set it to `true` only when collecting a
diagnostic failure report.
