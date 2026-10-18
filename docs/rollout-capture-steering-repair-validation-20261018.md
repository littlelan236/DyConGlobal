# Rollout / capture / steering 修复验收

日期：2026-10-18，Asia/Shanghai。对应[修复计划](rollout-capture-steering-repair-plan-20261018.md)。
远程工程：`/home/wangtianyi/DyConGlobal`。

## 实现范围

- 普通、capture、steering、steering+capture 共用 `core.cli rollout`、任务选择、
  `ServingManager`、模型客户端、mini-swe-agent、容器和 NAS 镜像管理。
  正式 rollout 拒绝 `--set`；resume 使用原快照。动态采集对象不再写入已签名 config。
- 唯一 config 明确配置 Qwen3.5-9B、FP16、qwen3_xml/qwen3、GPU4–7、TP4、
  12 workers、max_num_seqs=12，每 GPU 显式 48 GiB KV cache。
  其他提示词、采样参数、agent 限制沿用原 config。
- 原生 prefill/APC/batching 保留，在线 anchor 为 reasoning 换行与 `</think>`。
  tokenizer inventory、anchor 子集、all/double_only、export 开关和 flush 间隔落实到执行路径。
- 每次 query/API attempt/native request 明确关联；成功响应和 FormatError 响应使用同一证据读取入口。
  移除逐 query 强制等待回执，在任务收尾进行一次有界 join，然后导入 canonical 轨迹并冻结 SHA。
- 完整 prompt/output IDs 每请求只保存在一个不可变 ledger；轨迹和采集点引用 ledger/坐标。
  reasoning 剩余量跨当前及所有后续模型响应求和，包括格式错误响应；agent step 与请求序号分别记录。
- 本地未压缩 handoff → 有界异步队列 → 一次 NPZ 压缩 → 标准未压缩 tar 分片
  → 后台 NAS `.part`/SHA 校验/原子发布。分片默认约 128 MiB，session 结束封存最后一片。
  NAS 使用分片事件索引，避免逐点写大 sidecar；旧 loose NPZ/内联 IDs 继续可读。
- 项目总 spool 预算 50 GiB，高/低水位为 80%/60%。SQLite 事务记录跨进程预约与总量，
  同时用于镜像加载空间检查；避免随事件数增长而反复重写完整 JSON 账本。
  SQLite 仅是运行时资源账本，不是第二份任务 manifest 或配置。
- 两个独立 uploader 槽位隔离 NFS hard-mount 阻塞；不无限派生替代进程。
  未确认 NAS 发布前保留本地副本；已确认分片及时释放，最终 metadata 在 session 发布后释放。
  `capture-recover RUN` 复用冻结快照和本地字节恢复，不重新请求模型。
- failure_policy 覆盖控制面、回执、队列、写入、上传和收尾；agent 终止结果与采集状态分开。
  capture 故障不转成服务端 completion 500 来触发重新采样；block 策略在保存响应后抛出 CaptureError。
- 发布产物可直接建立 `online-index RUN INDEX`，训练/评估使用共享 artifact reader，
  不默认额外复制 packed tensor。训练标签仍要求 resolved 且未 censored，分区取自唯一 baseline manifest。
- 旧错误启动的 22 条记录按四个精确 run ID 隔离，保留原文和审计证据；原 1,104 条记录不改写。
  本轮真实单任务验收另行隔离，不占用正式 200-task 的 unrolled 范围。

未运行速度测试；未改采集 storage dtype。实际计算为 FP16，实际存储 dtype 如实登记。

## 自动化测试

最后一轮执行结果：

| 环境 | 范围 | 结果 |
|---|---|---|
| runner | 全部 tests | 134 passed，7 skipped |
| analysis | hidden、estimators、enrichment、capture_repair | 32 passed，4 skipped |
| easysteer | online_capture、vllm_native_capture、capture_repair | 45 passed，2 skipped |

这些范围有交集，不把通过数相加。环境缺少的 torch/matplotlib/Starlette 或 mini-swe-agent
测试在具备对应依赖的环境补跑。修改范围的 Ruff check 通过。

新增回归覆盖：四模式公共参数一致及扩展去重、config 篡改/resume、anchor 子集和 tokenizer 策略、
32k prefix 去重、FormatError 证据、缺失回执策略、无 API 重采样、export on/off、
tar/loose 读取一致、NAS 失败保留/恢复、损坏 `.part` 与 ACK 丢失、事务回滚、
12 线程/多 run 字节上限、镜像共享空间预约、跨进程水位唤醒、按任务分区的直接分片训练读取。

## 四模式真实 GPU 验收

使用正式 CLI、相同公共 config、Qwen3.5-9B FP16 TP4 和真实 NAS 容器。
验收 run 前缀：`repair-accept-20261018-b`。所有模式均有真实 bash action/observation。

| 模式 | 结果 | 附加采集证据 |
|---|---|---|
| plain | 通过 | 原生 XML 解析和真实工具执行 |
| capture | 通过 | 自然请求 13 点；受控混批 38 点；NAS 发布/读回 |
| steering | 通过 | 非零 direct 向量，layer 15，scale=1 |
| combined | 通过 | 自然请求 12 点；受控混批 38 点；NAS 发布/读回 |

混批使用原生调度，实际事件同时记录到 decode/FULL 与 mixed/eager；未强迫混批进入 FULL。
向量作用另以 38 个相同 token prefix/位置核验：layer 15 的 hidden 差分与非零向量的
最低余弦为 0.999233，最大相对 L2 为 3.92%，向量 L2 范数为 6.38484。
这提供了实际作用的数值证据，不只检查 payload 非空。

原始凭据：`runs/repair-accept-20261018-b/report.json`、各 mode 的
`startup/receipt.json`、`steering-effect.json` 及 NAS 对应 run 目录。

## Online 与离线 FP16 vLLM replay

保持 checkpoint、TP4、32 个 site 和对应 steering 条件；从 NAS 发布的单份 ledger
取 native IDs，以共享 reader 读取在线分片。比较冷缓存、热缓存与重排批次。
固定门槛：cosine ≥ 0.999，relative L2 ≤ 0.05，未放宽。

| 数据 | 测点 | 条件 | 最低余弦 | 最大相对 L2 |
|---|---:|---|---:|---:|
| capture 自然工具请求及受控混批 | 51 | cold | 0.99991884 | 1.2785% |
| 同上 | 51 | warm | 0.99991337 | 1.3213% |
| 同上 | 51 | reordered | 0.99994236 | 1.0812% |
| 非零 steering+capture | 50 | cold | 0.99991068 | 1.3399% |
| 同上 | 50 | warm | 0.99991068 | 1.3399% |
| 同上 | 50 | reordered | 0.99990990 | 1.4110% |
| 真实 coding 轨迹首尾 newline/close | 4 | cold | 0.99998707 | 0.5098% |
| 同上 | 4 | warm | 0.99998744 | 0.5018% |
| 同上 | 4 | reordered | 0.99998726 | 0.5054% |

每个测点均比较全部 32 个 site（native `[0..30,32]`，含 final norm）。
真实 coding 测点位置为 702、721、23751、23752，验证了长上下文下的实际轨迹证据。
三组均通过；结果表明在这些测点和条件下在线采集与原生离线 replay 数值接近，不要求逐位一致。

逐事件、逐层、按 anchor/graph/batch 分组的原始误差在：
`runs/repair-accept-20261018-b/replay-{capture,combined,real-coding}/`。

## 真实单任务全链路

Run：`repair-e2e-20261018-c`，task：`astropy__astropy-12907`。

- 24 次持久化模型响应、24 条工具 observation、0 次 FormatError。
- agent 按原配置的上下文上限终止：ContextWindowExceededError；未提交补丁，不标记 resolved。
- capture_status=complete；251/251 个 reasoning 事件覆盖，missing/unexpected 均为空。
- ledger 引用在 canonical import 前补齐；最终 canonical SHA 与 finalization 一致。
- NAS 分片/发布凭据可读；直接分片 dataset index 为 251 点，读回 32×4096 hidden。
- 标签是观测轨迹剩余量；未评测成功的轨迹不提供成功剩余难度训练标签。

Canonical rollout：`r-30e16c45a50a174fb237c5ea8d3b5c6a`。该验收记录保留并隔离为诊断，
原 200-task scope 恢复为 200 个尚无有效正式 rollout 的任务。

## 正式运行恢复

正式 run：`capture-verified200-20261018-143317`。使用唯一 config 与官方 CLI，
由 `runtime.scripts.supervise_rollout` 等待子进程退出并写 exit receipt，无进度轮询器。
启动验证由 `runtime.scripts.wait_capture_startup` 等待文件/进程事件，要求真实工具 observation
以及预先完成的两种 anchor、全层有限值、exact token 与 NAS 发布读回。

14:35:55 启动凭据已返回 healthy：首个正式工具调用在 `astropy__astropy-7166`
真实容器中执行，returncode=0；启动采集 session 的 13 点、两种 anchor、32 个 site 与
NAS 读回均通过。实际 config hash 为
`ed3c69e8df9aad865a77aac88aaa7af87ed2fe381d7dcd94c23b776a0ce547de`。
凭据：`runs/capture-verified200-20261018-143317/startup-verified.json`。
确认后未再查询运行进度。

轨迹：`baseline/trajectories/<rollout_id>/`；最终采集：
`/VM/wangtianyi/DyConGlobal/hidden/online/<run_id>/`；临时缓冲：`hidden/spool/<run_id>/`；
标签收尾：`enrichment/outputs/<run_id>/`。范围证明、快照和启动凭据在该 run 的 `runs/` 目录。

200 条任务的完成/评测结果由运行产物记录；本验收不把“已正常启动”写成“200 条已完成”。
