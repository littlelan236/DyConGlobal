# 2/4/8 GPU checkpoint resume: validation record

Date: 2026-11-15. Implementation deployed to `/home/wangtianyi/DyConGlobal`.
No formal training was started/stopped and no GPU acceptance was executed.

## Implemented contract

Enable `runtime.topology_policy: preserve_updates` in a new attempt's complete
resume YAML and select two, four or eight devices with `runtime.gpus`.
The default remains `exact`.

Both whole-parent and existing distinct-parent two-pass schedules preserve
logical optimizer windows, branch weights, frozen advantages, original anchor,
optimizer/scheduler state and committed cursor. Whole-parent work splits into
at most 16 branches per physical task; two-pass work retains its original halves.
Physical topology is recorded separately from the source logical contract.
Historical checkpoints are not edited.

This version rejects RNG-consuming training computations and nonzero dropout.
It preserves the objective and update boundaries; floating-point reduction order
can differ across GPU counts. See `TOPOLOGY_SWITCH.md` for scope and commands.

## Completed checks

| Check | Result | Evidence under `temp/rl-topology-switch-20261115/` |
|---|---|---|
| Dedicated CPU/Gloo topology suite plus GPU acceptance collection | 13 passed, 4 skipped, 173.26 s; exit 0 | `topology-final.log`, `topology-final.exit` |
| Broader regression | 664 passed, 26 skipped, 3 failed, 582.50 s; exit 1 | `regression-final.log`, `regression-final.exit` |
| Same failures on original RL source | 8 passed, same 3 failed, 4.30 s; exit 1 | `baseline-failures.log`, `baseline-failures.exit` |
| Real v10 checkpoint -> eight-device CLI dry-run | status `validated`; exit 0 | `v10-eight-dry-run.json`, `.stderr`, `.exit` |
| Changed tracked files whitespace check | `git diff --check` passed; CRLF normalization warning only | Final bounded SSH inspection |

The dedicated suite covers real CPU/Gloo cycles `2 -> 8 -> 4 -> 2` and
`8 -> 2 -> 4 -> 8`, independent unsplit objective/gradient oracles, Adam moments,
scheduler, branch/token metrics, non-32 branch groups and tails. Tiny Qwen+PEFT
service checkpoint cycles cover both schedules with prepared and streaming
inputs. Checks include logical contracts/cursors, unchanged source bytes,
new checkpoint receipts and rejection of RNG consumption before optimizer mutation.
The four skipped cases are opt-in real Qwen/NCCL GPU tests.

## Existing regression failures

All three failures reproduce using an in-memory import hook that loads original
RL sources from the pre-edit `baseline.tar.gz`; current project files were not
replaced. The baseline reproduction script is `run_baseline_checks.py` in the
same remote task evidence directory.

- `test_all_preflight_switches_are_strict_and_default_off`: an old test expects
  eight switches, while the existing config also has `verify_trajectory_integrity`.
- `test_training_prepares_groups_once_and_keeps_required_artifacts`: an old
  expectation for the legacy `prepare()` call/artifacts disagrees with the
  existing prepared-data path.
- `test_append_observation_uses_segment_schedule_and_keeps_inherited_prefix`:
  preceding resource-selection tests change `CUDA_VISIBLE_DEVICES`, leaving
  inconsistent cached CUDA availability/device-count state in the CPU suite.

These were not changed by weakening checkpoint or RNG checks. The broader suite
is therefore not reported as clean.

## Real checkpoint dry-run

Config:
`RL/experiments/topology-switch-20261115/v10-resume-eight-dry-run.yaml`.
Source checkpoint:
`/data3/wangtianyi/DyConGlobal/RL/checkpoints/grpo-swe169-consensus-half-prefix-pool128-kl-off-step-stream-fresh-20261113-v10/checkpoints/step-000060`.

The source attempt was failed, not active. Its last reported update was 63; the
recoverable committed checkpoint used here is 60. The new YAML retains the source
frozen training fields and selects eight devices with `preserve_updates`.

The formal `.venvs/runner/bin/dycon lora-train --training-config ...` dry-run
returned `validated`, physical world size 8 and restored global step 60.
Metadata validation was `verified`; model tensor validation remains
`pending_training_model`, and RNG validation remains `pending_model_and_update`.
No training model was loaded and no optimizer update was executed by this dry-run.

## Pending acceptance

Real Qwen3.5-9B/NCCL/BF16 cross-topology correctness, long-context GPU capacity,
throughput and downstream quality remain unverified. The opt-in GPU suite is
provided, but its execution would start actual training workloads. CPU results
and metadata dry-run do not establish GPU capacity, speedup or quality.
