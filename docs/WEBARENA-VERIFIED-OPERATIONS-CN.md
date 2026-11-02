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
.venvs/runner/bin/dycon webarena environments-prepare --site reddit --site shopping --site shopping_admin --site gitlab
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
### 2026-11-01：真实闭环与运行注意事项

本次固定 Verified commit 为 6473f72db5dcefc97b5725b59e734504edc28a21，仍使用 mini-swe-agent 和官方离线 evaluator。

- WebArena 上下文保留 system/task 和最近完整 assistant/tool 对，默认 80,000 字符。v8 验证未再上下文越界；其他 benchmark 不变。
- 同一 parse_command 支持原括号语法与 type/tab_focus/goto/press/scroll 简写；stop 接受直接 JSON 或 [JSON]，仍校验公开 schema、限制单动作，不执行 shell。type 裸尾部 0/1 按提交标记解释，字面尾数用规范括号形式。
- Shopping Admin 的基址必须是 http://shopping-admin/admin，health_path=/admin，login.path=/。v12/v13 真实 preflight 已确认任务在管理后台；旧根路径 pool 只能作历史证据。
- 健康检查使用 spec.port 和完整 Host，覆盖 GitLab 8023 等非默认端口。startup gate 通过同一 parser 比较规定 down-scroll 动作，保留真实模型/工具验收。
- _git_identity 使用实际可用 Git，不再硬编码已失效 executable；旧 unavailable 身份记录不回写。
- 默认 harness prepare 从混合 run 只选择实际 submitted 的轨迹；全未提交仍拒绝，显式指定未提交 ID 也拒绝。完整性/来源校验仍保留，所有计划样本留在 report 分母。
- 官方结果路径用 core.io.artifact_reference，支持经 frozen snapshot 绑定的外置 run alias；不要用字符串替换或绕过 owned_path 校验。旧评估不覆盖，修复后使用新 eval_id 重评相同 subject。

最新回归：tests/test_webarena*.py、tests/test_model_protocol.py、tests/test_run_plan_code_identity.py，共 122 passed, 1 warning；修改的 Python 文件 Ruff / git diff --check 通过。

当前四站点 pool：runtime/webarena-environments/ceda4142c902fe34554df0eb832fb88db85631c6d4bb587629a69be5971cd51f/manifest.json，SHA256 f842736dbe184112541a63d141f6e4834e2169d85433f6fc2413a8f11763bebd。包含 GitLab、Reddit、Shopping、Shopping Admin；发布本身不代表任务成功。

#### 已完成的真实 smoke

webarena-smoke-multisite-v13-20261101：6 个 fresh preflight cycle、双 GPU startup、三任务 rollout 和官方评分已完成。最终使用 webarena-v13-official-v2-20261101：

| 任务 | 官方结果 |
|---|---|
| wav-118 / Shopping NAVIGATE | success，resolved=true |
| wav-399 / Reddit MUTATE | failure，resolved=false，非基础设施错误 |
| wav-000 / Shopping Admin RETRIEVE | LimitsExceeded，未提交，不补造答案 |

run report 分母为全部 3 个计划样本，resolved=1、unresolved=1、unsubmitted=1、infra_error=0，即 33.3%。不能将已评分两题的 50% 当作 run 成功率。最终 artifact verify 通过 15 artifacts / 4 receipts，cleanup_complete=true，环境依赖失败为 0。首次路径故障的 infra_error 评估历史保留。

官方 evaluator runtime manifest 的本轮冻结 SHA256 为 330109e04dfd9e4ab294464857400d5067c2100e6e6508a00f56164e3ed1bffc，evaluator checksum 仍为 35c3385b1db4b3378657589f95f50defd4234bd36e5b93d44733fd561b01db4e。以 run 的冻结引用为准，不沿用过时文档值。

#### GitLab 与双站点接续

webarena-preflight-gitlab-crosssite-20261101 的 wav-044 / wav-684 已于 10:10:37 通过 4 个 fresh reset cycle，覆盖 GitLab 与 GitLab+Reddit，HAR/cleanup 通过，model_requests=0。

新 canary webarena-smoke-gitlab-crosssite-20261101 使用 GPU2–3、独立端口 18228/18229；通过现有 run queue-resume 在 v13 的 GPU 进程退出后自动启动。request-id=webarena-gitlab-after-v13-20261101。队列 resumed 不代表 startup 或任务成功，仍须正式 harness/artifact/report 验收。

#### 剩余范围

Wikipedia 已有 linux/amd64 派生镜像发布记录（webarena-verified150-h15-n35-20261031-wikipedia-amd64/images/wikipedia-amd64.json），仍缺全量 ZIM baseline 和真实环境验收。Map 的数据 baseline 未发布；已核实的 Nominatim 与 OSRM 官方归档合计约 146.1 GB，另需 tile 数据。不可用 CI 小数据冒充 benchmark 全量数据，不因四站点 smoke 通过而启动 Full/Hard。
### 2026-11-01：GitLab 浏览器接续与镜像派生

GitLab 登录仍走真实 UI，账户链接 a[href='/byteblaze'] 显式配置 verify_state=attached；其他未指定状态的 UI 登录默认 visible，header 登录默认 attached。仅允许 attached/visible。worker 保存不含密码、Cookie、HTML、query 的 login-verification.jsonl，记录匹配数和首个元素可见性。

browser-prepare 在上游源码、Python base、requirements 和 Dockerfile 身份全部一致、父镜像 selftest.passed=true 时，可在 NAS 父镜像上仅替换 worker/selftest。派生保存父 reference、父 image ID 和派生 Dockerfile SHA，仍执行真实 AX/ID/POST/HAR contract。发布不修改旧 run 的 browser reference。

已发布 browser：runtime/webarena-browser/webarena-browser-ee05eb8d765a8f5c0ae8/reference.json，NAS manifest SHA=840e7bb3b1596b142aed3a3737f9f6e65c3ea9b471341f174ab5957e0b7725c7。

新四站点 pool：runtime/webarena-environments/1b8e2be7093529a649b7faf8da7582e84ece4117821dad90822213321ba6cac3/manifest.json，SHA=80cec84e9e0ba7b2e135e7f270f08a4631e21db9c883f48deec4e52abd0b5edb。

webarena-smoke-gitlab-crosssite-v5-20261101 于 12:23:56（北京时间）完成独立 preflight：GitLab、GitLab+Reddit 共 4 个 fresh reset cycle，登录/scroll/HAR/cleanup 通过、model_requests=0。诊断中 GitLab 首个账号链接实际可见，因此不能据新结果倒推旧失败现场一定隐藏。12:24:41 经 run start --detach 启动正式两题，成绩以完成后的 report/harness 为准。

外层 timeout 直接终止独立 preflight 可能留下容器（v4 已实际发生并清理）。需先确认原进程退出，再使用现有 WebArenaFactory.cleanup_owned 精确按 project/run/backend 清理；不要删除别的 run 的资源或把清理成功当作 preflight 通过。临时诊断命令必须位于 temp，不能变成另一套执行入口。

接续定向测试为 140 passed、1 warning；旧 warning 为 LiteLLM event-loop 弃用提示。Map/Wikipedia baseline 和 Full/Hard 不在本轮范围。
