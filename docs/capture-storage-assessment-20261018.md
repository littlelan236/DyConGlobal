# Capture 的 NAS 时间开销与存储容量评估

本次只读取已有产物、代码和系统配置并计算容量，不启动 rollout 或速度测试，
不修改采集存储精度。以下时间判断不等于本地/NAS 对照实测的性能百分比。

结论：NAS 容量足够当前 200-task 工作，直接写入也不一定成为带宽瓶颈；
但当前“逐 token 小文件 + NAS staging + 再压缩 + 重复完整 prompt 元数据”的组织方式，
会放大 CPU、NFS 元数据与同步等待。异步 writer 只能隐藏部分开销，不能保证无限解耦。
本地磁盘保留整批产物存在占满风险，适合做有界缓冲，而非唯一最终存储。

## 实际数据与测量边界

从 `capture-verified200-20261018-120720` 的完整 index/raw 配对中核实 283 个采集点：

| 项 | 实际值 |
|---|---:|
| checkpoint | Qwen3.5-9B |
| site 数 | 32，native sites 0..30 与 32 |
| hidden size | 4096 |
| 实际数组存储 dtype | float32；模型计算仍为 FP16 |
| 每点未压缩向量 | 32 × 4096 × 4 = 524,288 bytes = 512 KiB |
| 每点压缩 NPZ 均值 | 332,433 bytes，约 324.6 KiB |
| 每份 JSON sidecar 均值 | 16,496 bytes，约 16.1 KiB |
| 每点 NPZ + raw JSON + index JSON | 365,426 bytes，约 356.9 KiB |

`tensors/<sample_id>.npz` 与 `raw/<sample_id>.npz` 在当前 NAS 上经 samefile/inode
核实为硬链接，不能按目录表观大小重复计算为两份 hidden。未完成的单个写入以及
staging 临时文件不纳入上述 283 个完整采集点的均值。

这些产物来自短的错误启动请求，平均 prompt 只有约千 token 量级。张量大小可用于
容量参考，但这些失败 rollout 的请求速率不能作为正式 coding rollout 的吞吐依据。

## 当前 I/O 链路与时间风险

```text
原生 worker capture / TP assembly
 -> _publish 在后台线程将数组压缩，写 NAS staging NPZ
 -> HTTP 把 staging 文件名交给 controller
 -> controller 读 NPZ 并解压到 CPU 数组
 -> 放入异步导出队列
 -> 单 writer 再压缩并写 NAS raw NPZ
 -> 写 raw/index 两份 JSON，建立兼容硬链接和 journal
```

每个点有两次压缩、一次解压及至少两次 tensor 文件写入；staging 读取可能命中本机
page cache，所以不能机械地算为三倍网络流量。raw/index JSON 使用原子写、fsync、rename，
加上临时文件、hardlink、unlink 等，会产生许多 NFS 元数据操作。

NAS 路由实际经过 enp194s0f0，链路速率 10,000 Mbps，挂载为 NFSv4.1。
理论单向带宽上限约 1.25 GB/s；这不是应用实际可用带宽，也不表示 NAS 独占。
若每秒产生 50/100 个采集点，仅最终压缩向量的写入约为 16.6/33.2 MB/s。
因此在这种假设事件率下，链路字节带宽未必是首要约束；CPU 压缩、逐点处理和小文件延迟
可能先限制速度。长 prompt 元数据和 staging 重写会继续增加流量。

设每秒产出 R 个点，导出每点平均耗时 t，则单 writer 要维持稳态至少须满足 R × t < 1。
例如“每点 20 ms”是一个计算示例，不是本机测量值：writer 最多约处理 50 点/秒；
若输入 100 点/秒，256 点队列约 5.1 秒就满。队列满后当前实现可出现 export_backpressure
和缺失 hidden；不能因为使用异步线程就认为 NAS 开销不存在。

已有短请求的 session receipt 曾显示队列为空、无落盘失败，这只是低负载证据，
不能据此证明完整 12-worker 长 rollout 不受影响。NFS 使用 hard mount，NAS 长时间抖动
还可能使 I/O 等待很长；单纯在 Python HTTP 层设置 timeout 并不能中断所有底层文件 I/O。

## 长上下文的元数据放大

当前每个采集点 sidecar 都重复包含整份 prompt_token_ids，raw/index 又各存一次。
根据实际 sidecar 结构，使用平均五位数 token ID 的序列进行 JSON 大小计算：

| prompt token 数 | 单份 JSON | NPZ + 两份 JSON / 点 |
|---:|---:|---:|
| 当前短样本均值 | 约 16.1 KiB | 约 356.9 KiB |
| 8,000 | 约 93.6 KiB | 约 511.9 KiB |
| 16,000 | 约 179.6 KiB | 约 683.8 KiB |
| 32,000 | 约 351.4 KiB | 约 1,027.5 KiB |

真实 token ID 位数和其他字段长度会改变结果；表格用于估算量级。
32k 场景下，重复 JSON 可比压缩 hidden 本身更大。它不增加研究信息量：同一个 request
的所有采集点引用同一 token ledger 即可，不需要各自复制完整 prefix。

## 200 个任务的容量场景

下表只计算现有布局中的压缩向量和两份 sidecar，不重复统计硬链接，不包含轨迹正文、
request receipt、日志、后续 packed hidden 副本和临时缓冲。事件数是场景假设，不是预测。

| 每任务 raw 采集点数 | 总采集点数 | 短 prefix | 平均 prefix 约 8k | 平均 prefix 约 32k |
|---:|---:|---:|---:|---:|
| 500 | 100,000 | 34.0 GiB | 48.8 GiB | 98.0 GiB |
| 1,000 | 200,000 | 68.1 GiB | 97.6 GiB | 196.0 GiB |
| 5,000 | 1,000,000 | 340.3 GiB | 488.2 GiB | 979.9 GiB |

当前 native selector 是候选集合，还会捕获 `</think>` 后模型生成的工具调用 XML/命令
文本中的换行，最终 reasoning 区间过滤才排除它们。因此 raw 采集点数可以多于最终训练
索引中的有效点数。

历史 50 条随机抽样仅作量级参考：assistant 轮数中位 58，reasoning 换行连续段数中位 188，
90 分位换行段数为 1091，最大 4505。这不是精确 tokenizer 事件数，且历史 step 上限、
输出上限不同于当前 config，不能直接外推为这 200 个 Verified 任务的总点数。

评估时本地根分区可用约 590 GiB，NAS 可用约 42 TiB。200 个任务的数十至数百 GiB
对 NAS 容量可以承受；长上下文/密集采集接近 1 TiB 的场景会超过本地剩余空间。
此外本地仍需留给 Docker 镜像加载、容器工作层和已有任务，不能把 590 GiB 全分给 capture。

## 建议

1. 最终产物保存在 NAS。本地 NVMe 使用约 20–50 GiB 的有界 spool/staging，独立后台
   上传并校验成功后释放本地副本，避免每个点的 staging 写入/读回都经过 NAS。
   缓冲满必须有明确状态和准入策略；不能无限占盘或静默丢数据。
2. 优先让每个 request 的 prompt/output IDs 只保存一次，逐点索引仅保留 request_id、
   position、token_id、anchor、层/文件引用和校验值。保持精确对齐，不改变 hidden 精度。
3. 先减少重复压缩和重复元数据；随后如有必要，使用现有 pack 能力或标准容器格式，
   将小文件组织为按轨迹/固定大小分片，避免上百万个 NAS 小文件。
   不为本次评估另建数据格式或新调度器。
4. raw 与 packed 如果都保留，会真实增加一份张量容量；明确二者的保留策略，不能误当作
   当前 tensors/raw 的硬链接关系。无后续消费需求时，不默认生成额外 packed 副本。

按“每任务约 1,000 raw 点、平均 prefix 8k–32k”的假设，当前布局约需 98–196 GiB，
加上非 tensor 产物应预留数百 GiB。若不约束事件数且保留所有中间副本，建议为这一批
留出约 1 TiB 余量；这是一项容量预算，不是实测总量承诺。

这些是存储路径和产物布局建议，不改变已经确定的唯一 config/正式 CLI 修复原则，
也不恢复已经停用的速度测试。
