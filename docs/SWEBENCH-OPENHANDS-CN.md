# SWE-bench + OpenHands

DyCon 通过显式 `(benchmark, agent)` binding 选择运行 adapter。基础 SWE-bench 的 OpenHands 组合使用 `native_swe_openhands`，与 SWE/mini 的 `native_swe` 和 Pro/Terminal 的 Harbor binding 分开。

## Profile

使用固定的、项目内的 SWE OpenHands profile；profile 不复用 SkillsBench 的 BenchFlow 字段：

```bash
.venvs/runner/bin/dycon swebench openhands-profile PROFILE_ID \
  --bridge-address 127.0.0.1 \
  --dependency-snapshot runtime/dependency-snapshots/SNAPSHOT
```

profile 冻结 ACP driver、非 root `agent`、`/testbed` workspace、网络策略、依赖快照和 patch 大小上限。依赖快照必须由项目拥有并通过 hash 校验。

## Run

在根配置中设置 `benchmark.backend=swebench`、`agent.backend=openhands` 和 profile 引用，然后继续使用统一入口：

```bash
.venvs/runner/bin/dycon config-check
.venvs/runner/bin/dycon catalog-check
.venvs/runner/bin/dycon run create RUN_ID --operation rollout --task-id INSTANCE_ID
.venvs/runner/bin/dycon run start RUN_ID --detach
.venvs/runner/bin/dycon run wait RUN_ID
```

新 run 会冻结 execution binding、profile hash、依赖 hash、task revision、OpenHands protocol 和 patch subject。旧 run 的 snapshot 不会被改写。

## Preflight and qualification

```bash
.venvs/runner/bin/dycon swebench preflight RUN_ID
```

baseline 可以在 request-role qualification 前运行；capture 和 parallel 会要求 profile 的 `request_roles=openhands-1.22.1`。OpenHands steering 仍未支持。

## Patch and harness

agent 退出后，adapter 在 `/testbed` 内导出 binary-safe `git diff`，校验 base commit、路径边界、大小和空 patch 状态，写入 `patch-binding.json`，再登记 canonical trajectory。空 diff 为 `unsubmitted`，不能伪装成 Submitted。官方 SWE-bench 4.1.0 harness 继续使用既有 `harness prepare/run/status` 入口。

## Capture and parallel

OpenHands 请求、tool-call/result、native terminal receipt、ACP events 和 canonical trajectory 绑定到同一 attempt。capture drain/finalize 失败会保留 `CaptureError`/恢复证据。parallel 分支只生成候选，只有 winner 的 tool call 可以进入 SWE workspace；winner 选择和 branch evidence 由现有 OpenHands bridge/consensus 组件负责。

当前实现已完成 registry、profile、patch contract、adapter lifecycle seam 和共享组件回归；真实 task image、OpenHands ACP container transport、capture qualification 和并行 smoke 仍需在具备镜像、模型服务和依赖 publication 的环境中按计划逐阶段验收。
