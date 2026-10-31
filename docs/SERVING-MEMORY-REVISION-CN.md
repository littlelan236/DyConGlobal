# 已有 run 的显存比例审计修订

在用户明确授权调整已有 rollout run 的显存比例时，使用正式入口：

```bash
.venvs/runner/bin/dycon run cancel RUN_ID --reason 'User authorized serving memory revision'
.venvs/runner/bin/dycon run wait RUN_ID
# 确认旧 execution 的 termination.json 中 cleanup_complete=true。
.venvs/runner/bin/dycon run revise-serving-memory RUN_ID \
  --revision-id UNIQUE_REVISION_ID --gpu-memory-utilization 0.35 \
  --reason 'User authorized GPU memory fraction change'
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

该入口复用既有资源修订的锁、停止/清理校验、前后快照、hash/plan绑定和receipt流程。
只允许修改 serving.gpu_memory_utilization；设备、TP、模型、generation、任务清单、seed、capture和历史结果保持。
显式设置kv_cache_memory_bytes的run拒绝此修订，避免比例被固定KV预算覆盖。
已有取消、失败或正常退出的execution均必须有真实exit和cleanup回执。修订不能热更新活动进程。

新旧配置保存在 runs/RUN_ID/config-revisions/REVISION_ID/{before,after}/，receipt记录授权原因和受保护文件hash。重复同一revision ID和值/原因幂等；不匹配则拒绝。
实际可用KV容量仍受启动时profiling影响。启动后核对新PID、CUDA_VISIBLE_DEVICES、launch参数、GPU KV cache size、health、生成token增长和抢占计数，不能仅凭CLI返回detached声称成功。
