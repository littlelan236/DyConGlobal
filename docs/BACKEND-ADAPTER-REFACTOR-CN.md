# DyCon 后端与 adapter 重构说明

核对日期：2026-10-31。主工程：`/home/wangtianyi/DyConGlobal`。

本次重构已从隔离工程合并到主分支，合并提交为 `7716afb`；隔离工程的关键实现提交为 `1b1c914`、`9cc1872` 和 `3f8497c`。合并前主工程已有修改保存在检查点提交 `9810087`，可通过分支 `pre-backend-refactor-merge-20261031` 回退检查。

## 运行后端与组合

统一 registry 位于 `runtime/backends/`，不在静态配置读取时导入第三方 SDK。当前支持三类 backend family：

| benchmark / agent | runtime binding | subject / evaluation |
|---|---|---|
| `skillsbench` / `openhands` | `benchflow` / `BenchFlowAdapter` | workspace / live verifier |
| `swebench` / `minisweagent` | `native_swe` / `NativeSWEAdapter` | patch / SWE harness replay |
| `webarena_verified` / `minisweagent` | `native_webarena` / `NativeWebArenaAdapter` | web trace / offline evaluator |
| `swebench_pro_v2` / `openhands` | `harbor` / `HarborAdapter` | patch / official Pro replay |
| `terminalbench_2_1` / `openhands` | `harbor` / `HarborAdapter` | live state / task verifier |

新的 run 在创建时冻结 execution binding、driver/profile、task snapshot、subject/evaluator policy 和可冻结的依赖身份。resume 先读取冻结快照再选择 adapter；旧 run 仍按原配置派生兼容 binding，不改旧 hash。

调度、task×repetition、seed、attempt、retry 和 serving assignment 仍由 DyCon runner 唯一持有。adapter 只负责 run context、环境、agent transport、subject/evaluation facts 和 cleanup receipt，不另建任务选择或隐藏 retry。

## Harbor 入口

Harbor SDK 只在隔离 worker 中加载。静态配置、catalog 和旧 evidence 读取不依赖 Harbor 安装。可用入口包括：

```bash
.venvs/runner/bin/dycon harbor runtime-prepare ...
.venvs/runner/bin/dycon harbor runtime-verify PUBLICATION.json
.venvs/runner/bin/dycon harbor schema-probe --runtime ... --task-path ...
.venvs/runner/bin/dycon harbor import SNAPSHOT --source ... --benchmark swebench_pro_v2 ...
.venvs/runner/bin/dycon harbor images-prepare SNAPSHOT.json
.venvs/runner/bin/dycon harbor profile PROFILE_ID --dependency-snapshot ...
.venvs/runner/bin/dycon harbor preflight RUN_ID
.venvs/runner/bin/dycon harbor controls RUN_ID --control-id ID --agent oracle|nop|gold-replay|empty-replay
```

Harbor worker 固定源码 commit `5eb713ff9500daced4e143b98765a7c732744a1b`（0.25.0），通过 JSON-RPC stdio 托管固定 OpenHands ACP driver。每个 physical attempt 绑定 run/sample/attempt ownership、NAS image lease、worker process、Trial ID 和 cleanup receipt；禁止隐式 pull/build。Pro agent phase 使用专用 Unix model channel，并对公网、外部 DNS、宿主 SSH 和代理做拒绝探针。

rootless Docker 无法严格执行 CPU CFS quota 时，配置必须显式使用 `unenforced_with_receipt`；不能把该偏差报告成官方严格配额。内存、超时、镜像 digest 和 ownership 仍需保留可审计 receipt。

## 验证结果与边界

合并后主工程完成 compileall；backend/Harbor/WebArena/rollout 相关回归为 `72 passed`。隔离工程阶段曾完成 `79 passed` 的 WebArena/Harbor 相关验证；官方 evaluator fixture 仍可能因专用 evaluator venv 的依赖闭包差异返回 `infra_error`，该状态必须保留为 unknown，不能改写为 benchmark 失败。

GPU 4–5 canary `gpu45-adapter-canary-20261031-r3` 使用已验证 EasySteer 解释器，以 TP=2 启动 Qwen3.5-9B；实际完成模型服务启动、OpenHands ACP 请求、环境创建、canonical trajectory、subject 绑定和 cleanup。该次 agent 以 `reasoning_loop` 结束，不能作为任务解题成功；原始请求、轨迹、subject 和 cleanup 证据已保留。

当前仍未宣称：Harbor Pro/Terminal 完整真实模型 baseline 已通过，或 WebArena 真实站点/browser E2E 已完成。Harbor transport、rootless CPU quota 和 WebArena site/image pool 的限制应在验收报告中单独列出。

## 维护规则

- 新 benchmark 必须在 `core/backends.py`、`runtime/backends/registry.py`、catalog schema 和对应测试中显式注册；禁止把未知 benchmark 默认归入 SWE。
- 新 adapter 不得复制 runner、OpenHands decision/capture 或 artifact owner；优先复用现有 bridge、mini loop、evaluator 和 evidence service。
- 运行产物必须通过 `core.cli` / `dycon` 登记；临时探针放在项目 `temp/` 对应任务子目录，不能混入长期代码目录。
- 修改后至少运行 backend registry、Harbor contract、WebArena contract/harness 和相关 rollout 测试，并记录实际命令与结果。
