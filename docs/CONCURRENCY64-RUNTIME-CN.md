# 64并发运行资源profile

此profile是有界、较宽松的运行预算，不保证NAS永久故障或操作系统故障下零失败。保持模型、GPU/TP、estimator、generation、seed、32分支、64 inflight和4个任务worker不变。

| 配置 | profile值 |
|---|---:|
| parallel.control_request_seconds | 600秒 |
| parallel.results_wait_seconds | 600秒 |
| parallel.result_workers | 16 |
| parallel.score_timeout_ms | 5000ms |
| parallel.decision_ttl_seconds | 64800秒 |
| capture.control_seconds | 600秒 |
| capture.receipt_workers | 64 |
| capture.control_workers | 16 |
| capture.flush_timeout_seconds | 1800秒 |
| capture.finalize_seconds | 900秒 |
| capture.finalize_request_seconds | 7200秒 |
| capture.admission_seconds | 21600秒 |
| capture.export.upload_max_elapsed_seconds | 3600秒 |
| capture.export.queue_size | 2048 |

正式维护流程：

```bash
dycon run cancel RUN_ID --reason 'explicit authorization'
dycon run wait RUN_ID
dycon run revise-concurrency RUN_ID --revision-id REVISION --reason 'explicit authorization'
dycon run retry-failed RUN_ID --request-id REQUEST --exit-status ParallelSelectionError --task-id CONFIRMED_TIMEOUT_TASK --reason 'explicit authorization'
dycon run resume RUN_ID --detach
```

必须核验实际exit和cleanup_complete，不能把cancel请求当成已退出。revise-concurrency复用既有预算修订的离线锁、before/after备份、hash绑定、输入保护和receipt，且只能变更声明的14个profile字段。根config.yaml供未来新run使用，已有run各自冻结快照必须单独修订。

结果查询使用独立result_lane，不与长期branch准入共用等待线程；不再在results接口同步执行整个drain。待评分的有效分支会在600秒等待窗口内继续查询，单次HTTP另有600秒上限；这不是覆盖所有网络/系统调用的严格统一wall-clock截止时间。不会据瞬时pending/missing立即判算法失败。评分特征先于批次逐点I/O提交；算法真实无共识和identity冲突仍保留fail-closed。

session发布由controller提供该session的native request和sample身份，避免全run扫描；扫描/hash/metadata清理不持有共享writer锁。旧session恢复保留兼容扫描路径，但扫描移到锁外。stop的manifest与journal I/O也移出controller锁。

采集失败文件按增量读取，诊断引用原始文件而不重复内嵌完整token payload。flush按session过滤失败；flush失败后仍独立尝试stop与prepare。单点staging在未确认ACK时保留，retained-handoff reservation不按退出进程的临时handoff自动回收，后续恢复需依据保留文件和receipt核验，不能伪造完成或盲目重发非幂等请求。

Capture容量等待超时返回AdmissionWaiting并重新排队，不产生未启动agent的终态WorkerFailure。重复人工重跑请求允许共享同一原失败attempt的不同备份路径，仍严格核对原outcome及source hash。

Submitted但capture partial不是failed；本次维护默认保留，不覆盖成功提交。失败任务筛选应核对直接error或该session的decision证据，不能因为其他session累计capture错误就重跑真实算法失败。

2026-10-27部署按用户要求未执行回归或人工压力测试，只做语法、配置和实际运行核验。运行实例可从serving/launch.json及GET /dycon/capture/engine/status核验池大小与控制预算。
