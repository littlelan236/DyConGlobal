# 历史 Capture Spool 归档

在项目根目录使用统一入口：

```bash
.venvs/runner/bin/dycon capture archive RUN_ID --archive-id ARCHIVE_ID
```

该命令持有 run 锁，只处理该 run 的冻结配置所指定的本地 capture spool。
运行中的 producer/capture 进程会阻止归档。其他 run 的目录、reservation 和服务不受此任务操作。

处理顺序：

1. 冻结全部普通文件的 SHA-256、大小及文件集合；拒绝嵌套 symlink 或特殊文件。
2. 通过 rsync 复制并 fsync 至 NAS：`archives/capture-spool/RUN_ID/ARCHIVE_ID/spool/`。
3. 对 NAS 副本与原目录再次逐文件核验 SHA-256 和完整文件集合。
4. 将 `manifest.json` 和不可变 `receipt.json` 写入 NAS 并持久化。
5. 原路径切换为 NAS 历史引用，再删除本地重复字节。
6. 通过既有 DiskBudget 接口释放该 run 路径前缀下的 capture reservations。

原 absolute path 继续可读，因此历史 source proofs 保持可复核。CaptureSpool 拒绝在已归档或重定向的 run 目录中启动写入。
原始 canonical 轨迹、训练 dataset/split/checkpoint、NAS online tensors 和 run 级审计凭证保持原位。

Run 级归档记录位于：

```text
runs/RUN_ID/hidden/capture-archives/ARCHIVE_ID/
  selection.json
  manifest.json
  progress.json
  rsync.log
  result.json
```

相同 ID 可以恢复或验证复用；NAS 内容不符时拒绝删除本地证据。重新采集前需显式将该 run 的 spool 恢复为本地物理目录。

已复制完成但明确不要求数据 SHA 复核时，可显式追加 `--size-only-cleanup`。
此选项要求已有冻结 manifest，跳过复制及数据 SHA 复核，仅确认源/NAS 文件集合和每个文件大小一致，然后完成本地清理。
归档 receipt 明确记录 `file_set_and_size_only_per_user`；冻结 manifest 中原先记录的 SHA 不代表本次已完成 NAS SHA 复核。
