# ContainerCorrupted 轨迹追加清理（2026-10-17）

按照用户明确要求，追加删除此前保留的 4 条 ContainerCorrupted 轨迹及补丁文件，同步移除唯一 baseline/manifest.json 中的记录。旧工程数据保持不变。

当前共 1,104 条轨迹、258 个 task，全部用于 train / training。互斥结果分类：resolved 321、submitted 551、unsubmitted 232。其余 1,104 条保留记录与清理前完全一致，全部评测记录不变，Catalog 全量哈希校验通过，manifest revision 为 1237。

审计目录：baseline/provenance/container-corrupted-cleanup-20261017/。其中 manifest-before.json 为清理前元数据，removed-records.json 保存已删除记录与原因，receipt.json 保存校验回执，trajectory-index.csv 和 source-occurrences.csv 是清理后的当前索引。先前迁移和中断清理目录内的 CSV 是历史快照。
