# Runs Migration Manifest (2026-10-24)

Source: `/VM/wangtianyi/DyConGlobal/runs`
Destination: `/home/wangtianyi/DyConGlobal/runs`

Selection rule: `hidden.dtype` and `capture.dtype` must be `float16`/`fp16`, and `capture.anchors` must equal exactly `["think_close"]` (close-only). Runs without an unambiguous matching frozen config are excluded.

- Inspected: 51 runs
- Migrated: 40 runs
- Excluded: 11 runs
- Estimated migrated bytes from inventory: 3,760,394,498

## Migrated runs

- `baseline-evacuation-check-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `1,706,974`
- `capture-closeonly255-gpu03-w18-s16-bt2048-20261019-184246` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `28,572,421`
- `capture-closeonly325-gpu47-w18-s16-bt2048-20261019-184246` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `41,820,601`
- `closeonly331-harness-retry-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `1,463,458`
- `closeonly331-harness-retry1-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `1,465,852`
- `migration-closeonly331-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `1,601,693,585`
- `migration-thinkclose53-01-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `66,364,422`
- `migration-thinkclose53-02-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `63,604,482`
- `migration-thinkclose53-03-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `62,562,279`
- `migration-thinkclose53-04-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `138,720,268`
- `migration-thinkclose53-05-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `32,031,834`
- `migration-thinkclose53-06-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `112,257,466`
- `migration-thinkclose53-07-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `51,993,904`
- `minimum-baseline-one-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `1,115,116`
- `minimum-parallel-two-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `11,125,181`
- `nonverified600-baseline-close-gpu47-w24-s16-20261022` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `93,371,862`
- `parallel-baseline-check-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `605,554`
- `parallel-ridge-event-b4-w12-check-20261019` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `131,595`
- `parallel-ridge-event-b4-w12-check2-20261019` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `231,733`
- `parallel-ridge-event-b4-w12-check3-20261019` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `493,257`
- `parallel-ridge-event-b4-w12-disjoint31-20261019` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `6,043,687`
- `parallel-ridge-event-b4-w12-overlap36-20261019` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `8,830,986`
- `reasoning-loop-gpu-check-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `207,573`
- `reasoning-loop-gpu-check-v2-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `281,082`
- `reasoning-loop-gpu-final-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `280,689`
- `reasoning-loop-gpu-off-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `267,985`
- `resolved106-baseline-r1-20261022` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `70,961,969`
- `resolved106-baseline-r2-20261022` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `6,045,742`
- `resolved106-baseline-r3-20261022` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `6,045,742`
- `resolved106-baseline-r4-20261022` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `6,045,742`
- `ridge-resolved-closeonly106-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `99,664,610`
- `ridge-resolved-closeonly106-events-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `99,486,937`
- `ridge-resolved-closeonly106-events-alt-targets-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `99,482,662`
- `ridge-resolved-closeonly106-events-finealpha-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `238,445,632`
- `ridge-resolved-closeonly106-events-finealpha2-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `357,627,399`
- `verified80-common-baseline-harness-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `21,849,380`
- `verified80-common-step-linear-harness-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `21,531,787`
- `verified80-parallel-baseline-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `168,186,550`
- `verified80-parallel-reasoning-log1p-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `142,241,073`
- `verified80-parallel-step-linear-20261021` — hidden `float16`, capture `float16`, anchors `['think_close']`, inventory bytes `95,535,427`

## Excluded runs

| Run | Reason / observed config |
|---|---|
| `capture-continue278-gpu03-w24-s16-bt2048-20261019-155639` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-continue332-gpu03-w24-s16-bt2048-20261019-112749` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-continue341-gpu03-bt1024-20261019-104711` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-continue356-gpu47-w24-s16-bt2048-20261019-112749` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-continue382-gpu47-bt8192-20261019-104711` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-noneval391-gpu03-20261018-204351` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-repaired200-20261018-162700` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-restart12-gpu03-bt1024-20261018-230130` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-restart12-gpu47-bt8192-20261018-230130` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-verified200-20261018-143317` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |
| `capture-verified421-20261018-195500` | hidden dtype='float16'; capture dtype='float16'; anchors=['reasoning_newline', 'think_close'] |

## Verification

The selected directories were copied with `rsync -a` from the NAS source. The destination contains only the 40 selected run directories plus `.gitkeep`; source data was not modified. SHA256 spot/full verification is recorded by the migration command logs where applicable.

This manifest records selection and transfer scope; it does not change immutable run snapshots.

## Environment and validation on the new server

- Git source updated via SSH to GitHub `origin/main` commit `a009944cb7ff00b33f5d43e4145c67b44bf5eeb4`.
- Python 3.12.15 environments installed: `runner`, `analysis`, `easysteer`, and independent `latent-vllm`; EasySteer vLLM overlay and eager runtime verification passed.
- Rootless Docker 29.6.2 started at `unix:///run/user/1018/docker.sock`; the task image check passed for the smoke task.
- `dycon config-check`, `catalog-check`, `images-check`, model registration, runtime verification, and the project test suite were run. Pytest result: 557 passed, 29 skipped, 5 failed. The five failures are existing test/config mismatches involving unsupported `parallel.mode` fields (four tests) and a missing external Ridge predictor manifest (one test); no source files were changed to mask them.
- Short rollout `migration-smoke-20261024` used one task, one agent step, 128 max generation tokens, GPU 2 only, FP16, and 0.4 GPU memory utilization. It reached `complete`; the task ended with expected `LimitsExceeded`, with no infrastructure error. GPU 2 returned to 0 MiB after cleanup.

The smoke run and temporary test logs/scripts were removed after recording this result. The migrated run directories and this manifest were retained.

## Local-model parallel + capture smoke validation

- NAS model `/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B` was copied to `/home/wangtianyi/models/Qwen3.5-9B` (about 19 GB). `config.json` SHA256 matched: `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`.
- A real temporary predictor was published through `dycon predictor-publish` from the historical Ridge checkpoint bound to the same Qwen fingerprint, layer/site 14, hidden width 4096.
- `migration-parallel-capture-smoke-20261024` completed with no infrastructure errors. It used local model files, FP16, GPU 2, 0.4 memory utilization, `capture.enabled=true` with `think_close`, one captured layer (14), and two parallel branches.
- Evidence showed parallel `/reserve`, `/results`, and `/release` calls; one decision recorded two branches and selected branch 0 by the configured lowest-id fallback because both outputs hit the one-step format limit. Branch 1 nevertheless produced predictor score `7.750390936075956` and a persisted `think_close` tensor: position 2348, layer 14, shape `[4096]`, serving dtype `float16`, storage dtype `float32`, sealed in a NAS shard. Branch 0 correctly recorded `truncated_before_close` and no required close tensor.
- Run status was `complete`; task exit was expected `LimitsExceeded` from the smoke step/token cap, and `capture_status` was `complete` with no capture errors.

The temporary parallel run, predictor publication, NAS smoke capture, temporary scripts/logs, and smoke compilation cache were removed after this validation. The local model copy remains for future runs.
