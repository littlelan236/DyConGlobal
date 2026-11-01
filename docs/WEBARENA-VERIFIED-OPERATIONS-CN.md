# WebArena Verified 接入与首轮验收

当前只支持固定 revision `6473f72db5dcefc97b5725b59e734504edc28a21` 与 mini-swe-agent。任务选择、回答、HAR 和官方离线 evaluator 均绑定这个版本；原版 WebArena 的 evaluator 不会参与评分。Agent 只收到 intent 和浏览器 accessibility tree，标准答案保存在权限受限的 evaluator dataset 中。

首次准备时先完成依赖与共享镜像，再建立一次性 smoke 选题：

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon webarena evaluator-prepare --source .venvs/webarena-evaluator-src
.venvs/runner/bin/dycon webarena browser-prepare
.venvs/runner/bin/dycon webarena-images status
.venvs/runner/bin/dycon webarena-images prepare
.venvs/runner/bin/dycon webarena environments-prepare --site reddit
.venvs/runner/bin/dycon webarena import --source benchmarks/webarena-verified \
  --subset full --task-id 108
```

Image preparation can be repeated with the same configuration to resume. `environments-prepare` only publishes site images already verified in NAS; Wikipedia and Map also need their immutable data baselines. A pool's site list is not a statement that login/reset/end-to-end task execution passed. `run create` freezes the benchmark snapshot, profile, evaluator/runtime checksums, site pool and browser image. Set `benchmark.backend=webarena_verified`, `benchmark.snapshot_ref` and `agent.profile_ref` to the returned pinned refs, configure `webarena.environment_pool_ref` and `webarena.browser_image_ref`, then create a new smoke run:

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon run create WAV_SMOKE --operation rollout --task-id wav-108
.venvs/runner/bin/dycon webarena preflight WAV_SMOKE
.venvs/runner/bin/dycon run start WAV_SMOKE --detach
.venvs/runner/bin/dycon run wait WAV_SMOKE
.venvs/runner/bin/dycon harness prepare WAV_SMOKE
.venvs/runner/bin/dycon harness run WAV_SMOKE WAV_EVAL_001
.venvs/runner/bin/dycon harness status WAV_SMOKE
.venvs/runner/bin/dycon report run WAV_SMOKE
.venvs/runner/bin/dycon artifact verify WAV_SMOKE
```

`preflight` uses the same image admission, per-attempt private site network, login verification, persistent Chromium worker, initial page, actual browser action, HAR flush, teardown and clean re-creation as rollout. It sends no model request. Offline evaluator fixture tests are separate; they cannot pass environment preflight for current site images. If setup or real preflight fails, do not start rollout. The first rollout acceptance requires a real mini-swe-agent `stop [JSON]`, a persisted and flushed HAR, exact browser-action log, official `evaluate_task()` result and successful cleanup. A failed score is a completed benchmark result; parser, HAR-integrity and evaluator infrastructure errors remain unknown/infra errors.

Every assistant response may execute at most one browser action; only constrained `browser ...` commands are accepted. The final response must follow the pinned public `FinalAgentResponse` schema. An execution receipt for a click is not a task score. Agent limits, browser-action limits and sampling policy are frozen in the run; browser state is rebuilt for each physical attempt. An incomplete action/answer/HAR subject is retained for diagnostics but cannot enter WebArena harness selection.

Reports count success against all planned task×repetition samples, with pending, unsubmitted, ungraded and infrastructure-error outcomes reported separately. `Submitted` alone is not resolved. Historical 812/258 Full/Hard task availability and image bytes do not establish site usability or task completion. Do not launch Full or Hard until canaries cover each required site, RETRIEVE/NAVIGATE/MUTATE and the multi-site lease path; Full/Hard imports are available through `webarena import --subset full|hard`.


### 2026-10-31 v6-v8 canary ??????

- v6?`webarena-canary-reddit-gpu23-v6-20261031`?????????HAR????????????? 10 ?????? `ContextWindowExceededError`??????????
- ? `rollout/model.py` ?? WebArena ???????????? system/task ???????? assistant/tool ?????? benchmark ??????????? 80,000?
- v8?`webarena-canary-reddit-gpu23-v8-20261031`??????34 ?????? prompt ??? 34k token????????????? `wav-027` ??? 35 ?????????? `LimitsExceeded`/`unsubmitted`?
- v8 preflight ?? fresh reset?? GPU startup gate?HAR/action/cleanup ????`artifact verify` ?? 7 ???? 4 ? receipt ??????? WebArena harness ??????? subject?????? 1?`unsubmitted=1`???? 0?
- ?????`tests/test_model_protocol.py`?`tests/test_webarena_agent.py`?`tests/test_webarena_contract.py` ? 25 passed?1 ? LiteLLM ??????
- ????????????????? smoke???????? RETRIEVE/NAVIGATE/MUTATE????? preflight ?????????????????? Full/Hard?


### ??? smoke ??? clone ??

- ???????? `runtime/webarena-environments/371ba950084165c43e0285258ca1591a7a4fb4c0b4800af0198c5b2f9924bca8/manifest.json`??? Reddit?Shopping?Shopping Admin?????? fresh preflight ?????`webarena-smoke-multisite-20261031` receipt ?? 6 ? reset cycle?`model_requests=0`?
- ?????? v8 ????????????? `run create --from-run ... --task-id ...`?clone ???? benchmark snapshot ???????????? WebArena environment/browser refs??????? source ??? subset selection?
- `webarena-smoke-multisite-v9-20261031` ??? `wav-000`?Shopping Admin RETRIEVE??`wav-118`?Shopping NAVIGATE??`wav-399`?Reddit MUTATE???????????? preflight ?????? Docker daemon ???????????? agent rollout?
- GitLab/Map ???????????????????Wikipedia ? linux/amd64 ?????????
