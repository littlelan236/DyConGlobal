# Verified200 capture rollout

Run: `capture-verified200-20261018-120720`.

The source cohort is the 200 instances already validated by the project's NAS
cold-load gold audit. Before registering them, the launcher checked that none
of their instance IDs occurred in any existing rollout in `baseline/manifest.json`.
Gold results remain environment-audit evidence; they are not outcomes for these
new agent patches. The new tasks use the `train` partition and the runner's
existing `baseline` rollout purpose.

The sole configuration source remains `config.yaml`. The immutable resolved
configuration and explicit task selection are in `runs/<run_id>/`. Necessary run
overrides select the registered Qwen3.5-9B checkpoint, managed EasySteer serving
with the Hermes tool parser and Qwen3 reasoning parser, GPUs 4–7 with TP=4,
12 rollout workers, `max_num_seqs=12`, and 48 GiB of KV cache per GPU. Capture
is enabled for all 32 logical hidden sites, `reasoning_newline` and `think_close`.
The run keeps FP16, native prefill, the configured generation/agent limits,
`max_num_batched_tokens=1024`, and four concurrent NAS image loaders.

Artifacts:

- `baseline/trajectories/<rollout_id>/raw.traj.json`: canonical trajectory.
- `baseline/patches/<rollout_id>/patch.diff`: agent patch.
- `hidden/online/<run_id>/`: asynchronous raw hidden, token-index sidecars,
  exact native request receipts, and session-to-canonical-trajectory bindings.
  This run directory links to the same path under the project's NAS namespace.
- `enrichment/outputs/<run_id>/`: exact token alignment, remaining-trajectory
  labels, coverage, and excluded events; generated after the rollout exits.
- `runs/<run_id>/supervisor.log`, `serving/server.log`: process logs.
- `runs/<run_id>/startup-verified.json`: first real tool observation, exact
  response-token alignment, all-site finite hidden, and FULL execution evidence.

The global enrichment/hidden manifests contain bounded session entries that
reference the authoritative per-token index. They no longer embed every token
of the growing cohort. Session finalization binds the canonical rollout ID and
trajectory hash without rewriting immutable raw sidecars. Consumers apply
`sessions/<session_id>/finalization.json` when reading those sidecars.

`baseline.scripts.capture_verified` reuses the native runner, image manager,
configuration snapshot, and token-event finalizer. Static NAS image assets are
published under the new project's namespace, using hard links when possible;
runtime operation does not depend on the old project directory. Actual image
load still verifies blob hashes. A final engine drain precedes each session stop;
serving shutdown drains the asynchronous disk writer before final alignment.

Startup verification uses filesystem notifications and a process-exit descriptor,
and exits as soon as the evidence is sufficient. It installs no progress poller.
