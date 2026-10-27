# Capture 控制资源修订与指定失败任务重跑

正式入口为 runner 环境的 `dycon run retry-failed` 和 `dycon run revise-capture-control`；入口登记维护变更，执行使用原 run 的 resume。不要删除 result/attempt 文件或通过临时 launcher 绕过恢复机制。

## 1. 范围与停止条件

retry-failed 只匹配当前 result.json 的 exit_status，要求 disposition 为 failed/blocked，拒绝已 accepted 的结果。重复 --exit-status 取并集，重复 --task-id 限定 task 范围；没有错误文字筛选参数。不传 task ID 时选择全 run 的匹配样本，多 repetition 按 task×repetition 处理。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/dycon run cancel RUN_ID --reason 'explicitly authorized maintenance'
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon run status RUN_ID
```

已有 execution 必须有真实 exit 回执和 termination.cleanup_complete。wait 可能返回 TERM/KILL 的非零退出码，需核对 exit 和清理事实。进程身份仍匹配、运行/supervisor 锁被占用或清理未完成时拒绝维护。retry-failed 只要求目标 run 停止，其他 run 可以继续。revise-capture-control 仍复用项目级 capture producer 停止检查，可能需要协调停止其他 capture run；不要为满足检查擅自扩大进程终止范围。

## 2. 可选：修订已有 run 的 capture 控制资源

```bash
.venvs/runner/bin/dycon run revise-capture-control RUN_ID --revision-id capture-control-001 --control-seconds 120 --receipt-workers 16 --reason 'explicitly authorized capture control resource revision'
```

仅修改 capture.control_seconds、capture.receipt_workers。兼容默认值为30秒/4线程，线程范围1–128；120秒/16线程是显式设置示例。根 config.yaml 控制后续新 run；已有 run 必须单独修订，普通 resume 不接受 --set 覆盖。

修订前后 resolved-config.json、plan.json、config.yaml 位于 `runs/RUN_ID/config-revisions/REVISION_ID/{before,after}/`；receipt.json 记录 reason、previous_execution、字段值、前后 config/plan hash、受保护文件 hash。selection/tasks/work-items、模型、generation/seed、GPU/TP 和存储预算不因此变化。旧轨迹/capture 不被改写为新配置的证据，新 session 使用版本化 capture 配置引用。

## 3. 登记失败任务并恢复完整队列

```bash
.venvs/runner/bin/dycon run retry-failed RUN_ID --request-id retry-valueerror-001 --exit-status ValueError --reason 'explicitly authorized ValueError retry'
# 可选：经明确授权，重新生成具体 CaptureError/pending_recovery 样本。
.venvs/runner/bin/dycon run retry-failed RUN_ID --request-id retry-capture-001 --exit-status CaptureError --task-id INSTANCE_ID_1 --task-id INSTANCE_ID_2 --reason 'explicitly authorized pending-recovery task regeneration'
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

先核对返回的 sample_count/items；零匹配请求不会新建采样任务。ValueError 状态可能包括其他根因，若只重跑特定错误，应先核对当前结果并用 --task-id 限定。Submitted/capture partial 不会仅因 partial 被选中。

显式重跑样本优先，原有 Pending/Running/中断样本仍按完整冻结 work-items 队列恢复，不把 run 缩成仅重跑任务的子集。restart 可能创建新 physical attempt，不能保证从工具调用或生成中间精确续接。原始 capture 字节继续保留；CaptureError/blocked 要重新生成时须明确登记，不能只反复 resume。

## 4. 审计位置与幂等语义

```text
runs/RUN_ID/
  rollout/operator-retries/REQUEST_ID/
    request.json
    before/SAMPLE_KEY/result.json
    before/SAMPLE_KEY/ATTEMPT_ID.json
  events/operator-retry-REQUEST_ID.json
  rollout/tasks/INSTANCE_ID/repetitions/REP_ID/attempts/NEW_ATTEMPT_ID/
    attempt-context.json
    attempt-started.json
```

request.json 绑定 reason、selectors、previous_execution、config_hash、selection_sha256、匹配样本、旧 cursor/attempt hash、副本路径及 request_sha256。这些运行产物归属被 .gitignore 排除的 runs/，不提交到源码仓库。before 是审计保存副本，不表示已设置文件系统只读权限。

runner 仅跳过请求绑定的旧失败 outcome。旧 attempt、trajectory、patch、capture、agent outcome 保留；新 attempt 沿用 seed_round/effective_seed，attempt-started.json 标记 retry_kind=operator，不新增 repetition，不消耗基础设施重试计数，也不冒充 LimitsExceeded 换 seed 重试。新结果结算后更新当前 result cursor。

同 request ID、selectors 和 reason 在满足离线检查后幂等返回原请求；不同参数/原因拒绝。旧请求不持续授权新失败的主动重跑，下一轮需新 request ID。LimitsExceeded/基础设施自动重试仍按既有冻结策略执行。

## 5. 验收

- 用 run status 核对新 execution、完整 planned_samples、progress/phases 和原 selection/tasks/work-items。
- 请求登记不等于开始：检查目标样本的新 attempt-started.json、execution_attempt_id 和 retry_kind；排队样本等待 worker 空位。
- serving/launch.json 记录 control_seconds/receipt_workers；运行实例 GET /dycon/capture/engine/status 返回实际值和 execution_lanes。管理器通过 DYCON_CAPTURE_RECEIPT_WORKERS 向插件传值。
- 检查 generation token 增长、HTTP health、新 failure 和 capture 回执。增加超时/并发不证明持锁 I/O 等底层瓶颈已消除；Submitted 不等于 resolved 或 capture complete。

源码：core/cli.py、application/execution/failed_retry.py、application/execution/capture_budget_revision.py、rollout/runner.py、runtime/serving/manager.py、runtime/serving/vllm_plugin.py。回归测试：tests/test_capture_control_retry.py。
