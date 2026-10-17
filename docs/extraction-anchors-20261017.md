# Anchor extraction batch anchors-20261017

Scope: all 1,104 current training rollouts. The 191 rollouts with migrated historical close-marker products receive think_open only; remaining 913 receive think_open and think_close. Existing historical products are preserved.

Physical GPUs: 0, 1, 2, 3, one worker per GPU (CUDA_VISIBLE_DEVICES isolates each worker; adapter uses cuda:0 within the worker). Shared locked queue, checkpoint affinity, one model load per worker/model, 8 CPU threads per worker, bfloat16 inference, 2,048-token prefill chunks, all actual text decoder layers. Optimized FLA 0.5.2 and einops 0.8.2 installed; causal convolution uses the supported PyTorch fallback.

Queue and per-GPU logs: state/anchors-20261017/. Workers claim jobs atomically, write running/completed/failed status, and continue after individual failures. No automatic failure retry loop. Hidden extraction resumes verified partial samples; completed receipts are immutable. A worker killed mid-job leaves running status for explicit recovery. Checkpoints remain read-only references to the available model assets in the previous project, avoiding duplicated weights.

All checkpoint files are SHA-256 registered before launch. Rollout checkpoint binding is based on the recorded legacy model name; historical serving checkpoint bytes cannot be independently attested. Original legacy identities and fingerprints remain in provenance. Registration fingerprints are recorded in state/extraction-models.json. Workers verify bytes once per model and retain the verified model in process memory.

Per-rollout products: enrichment/outputs/anchors-20261017-<rollout_id>/ and hidden/outputs/anchors-20261017-<rollout_id>/. Each has a receipt; hidden index binds sample IDs, rollout IDs, step IDs, anchor names, prefix checksums and tensor paths. Top-level enrichment/manifest.json and hidden/manifest.json retain historical records and append extraction_jobs with task ID, run ID, source trajectory checksum, model identity, explicit marker text and product receipts.

Reconstruction evidence remains explicit. Missing/truncated anchors are reported in enrichment receipts rather than fabricated. Successful remaining-task labels are only provided for resolved rollouts; unsuccessful rollouts retain terminal-distance metadata without treating that distance as time to task success.

After initial launch verification, no ongoing status polling is performed.

Startup verification: all four workers loaded Qwen3.5-9B; hidden NPZ files were observed and two rollouts were completed. Runtime polling stopped after this check.
