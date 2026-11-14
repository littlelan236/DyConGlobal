# RL observability validation — 2026-11-12

Remote dirty worktree baseline: HEAD `64baac2`; pre-existing changes preserved.

- 29 analysis observability tests, 6 training/callback/append tests, and 78 existing regressions passed (113 total).
- Runner environment: 24 passed and 4 visualization tests skipped; CPU queries require no Torch.
- Qwen3.5-9B on GPU 1: prepared fresh paused at step 1, exact resume completed step 2. The inherited metric prefix was byte-identical; measured event delay 1.683 seconds.
- Sealed attempt artifact verification passed before/after visualization lifecycle; the resumed receipt hash was unchanged.
- Legacy current source: 62,292,066,422 bytes, 6,773 groups, 38,544 KiB worker high-water RSS. Frozen order/content and source hashes validated; unsealed integrity explicitly labeled. Group 0 returned all 32 saved advantages.
- Current training PID 2523786 was never restarted/stopped. Acceptance snapshot: update 71, recoverable checkpoint 70. Old stage remains unknown.
- Current TensorBoard runs on 127.0.0.1:16006 (sync PID 2163855, TB PID 2164430); SSH-forwarded scalar API returned HTTP 200. Repeated status added no points.
- Legacy index worker exited successfully; acceptance services on 16007/16008 stopped with receipts, and acceptance training exited.

Detailed local evidence: `rl_observability_plan_20261112/REPORT.md` and its `implementation/evidence/` on the initiating PC. Server task evidence: `temp/rl-observability-20261112/`.
See [usage and limits](OBSERVABILITY.md), including historical wall-time estimates, unavailable unproven advantages and prepared-only bounded token preview.
