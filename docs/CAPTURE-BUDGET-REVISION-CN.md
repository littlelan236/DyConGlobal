# 显式授权的离线 Capture 预算修订

默认仍按不可变run快照执行。仅在用户明确要求修改已有run的capture资源预算时，使用以下维护入口；普通resume不读取根配置覆盖旧run。

```bash
.venvs/runner/bin/dycon run cancel RUN_ID --reason 'Authorized capture budget revision'
.venvs/runner/bin/dycon run wait RUN_ID
.venvs/runner/bin/dycon run revise-capture-budget RUN_ID \
  --revision-id capture-128g-w95-20261027 \
  --max-bytes 137438953472 --high 0.95 --low 0.95 \
  --reason 'Explicit user authorization and scope'
.venvs/runner/bin/dycon run resume RUN_ID --detach
```

入口要求旧execution有真实退出及cleanup_complete回执，持有run、supervisor和active-supervisor锁，拒绝活动producer；只修改capture.export的spool_max_bytes、spool_high_watermark、spool_low_watermark。模型、生成、任务、分支、seed和其他参数不改。

run/config.yaml、resolved-config.json和plan.json的新版本配套更新hash。完整before/after文件及原因保存在runs/RUN_ID/config-revisions/REVISION_ID，旧execution和canonical结果不重写。初始config_source/hash继续表达原输入来源，后续资源差异由revision记录，不能声称原历史数据使用了新资源配置。

capture-config.json的旧本地和NAS字节保留；新配置写入capture-config-revisions/HASH.json，新session的capture_config_ref与capture_config_sha256绑定该版本。正常session发布会带上对应版本，避免覆盖旧NAS回执。

运行时只读核验可访问run/serving/launch.json中capture.control_url的GET /dycon/capture/storage。返回本controller实例的实际limit、reserve、水位、shared usage与物理目录；不是热修改接口。vLLM进程启动时从新版run export取得对应预算。

三个run使用同一个工程账本，128GiB是共享capture上限，并非每run独享128GiB。high=low=0.95取消滞回区，但超过121.6GiB仍有准入背压，128GiB仍是硬上限；独立的reserve_bytes物理空闲保护保持各run原值。
