# 本地 Capture Spool 迁盘

使用统一维护入口迁移实体目录，保持所有 run 的冻结逻辑路径不变：

```bash
.venvs/runner/bin/dycon capture relocate-spool \
  --destination /data1/wangtianyi/DyConGlobal/hidden/spool \
  --migration-id local-data1-20261027 --apply
```

必须先通过 run cancel/wait 停止所有共享 capture producer、serving 和 upload 进程。命令持有各 run 的运行锁和 active-supervisor 锁，拒绝发现的运行进程。未加 --apply 时只检查并返回计划，不复制/切换目录。

该命令逐文件核验 SHA-256、文件集合和符号链接目标；归档到 NAS 的历史 run 指针保持为原链接，不跟随复制。目标必须是本地文件系统，不得使用 NAS。切换时将原 hidden/spool 变为指向本地目标的目录链接，不能把单个 active run 变成归档链接。原配置、snapshot/hash、canonical trajectory 与最终 NAS publication 路径不变。

磁盘账本 key 和逻辑 path 保持不变；在同一个维护事务中，使用 DiskBudget 的账务原语将对应持久 reservation 的 device 从源盘改为目标盘。device totals 随之迁移，其他镜像/磁盘预约不动。源副本在目标字节和账本核验通过后才删除。

结果、manifest、copy 日志和阶段记录位于 state/capture-storage-relocations/MIGRATION_ID。复制或校验失败不会切换源目录；切换事务失败时回滚指针和数据库。若进程/主机在切换阶段崩溃，必须检查 progress、retired 源副本、目标和账本后再恢复，不能直接重启 producer。

完成后使用原 run 的正式 `dycon run resume RUN_ID --detach`。未完成轨迹可能按现有恢复逻辑重跑；历史已完成和失败证据不改写。迁盘不会重建已缺失的 capture 点，也不会消除与磁盘无关的控制超时。容量上限和水位仍来自原冻结配置，不因目标盘更大自动提高。
