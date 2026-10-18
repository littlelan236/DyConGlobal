# 421 tasks 正式 capture rollout 重启

本轮 run ID：`capture-verified421-20261018-195500`。实际启动时间以服务器
`runs/<run>/process.json` 为准（2026-10-18 20:11，UTC+8），run ID 仅为标识。

## 清理及任务范围

上一轮 `capture-repaired200-20261018-162700` 的 37 条 submitted 轨迹及对应
NAS capture 保留。submitted 表示已交 patch，尚未经 harness 评测，不代表 resolved。

删除上一轮 163 条 `ContextWindowExceededError` 轨迹及对应任务目录、canonical
trajectory/patch、163 个 capture session、5,295 个 request ledger、179 个 NAS hidden
分片及索引/发布回执；同时删除近期显式测试 run 的 23 条轨迹和测试运行目录。
不按时间模糊匹配删除；历史迁移数据、模型与镜像保持保留。

三个 manifest 已清理相关引用。旧 run 的 progress/config/selection 作为历史回执保留，
`cleanup.json` 解释哪些产物已删除。保留产物 1,493 项校验和复核通过；重建后的
enrichment coverage 为 4,817 / 4,817，missing/unexpected 均为空。

详细删除清单、manifest 清理前的元数据快照、校验结果及下一轮冻结任务清单位于
`state/cleanup/restart421-20261018/`。执行脚本为
`baseline/scripts/cleanup_capture_cohort.py`（默认为 dry run）。

新任务范围为旧 200 tasks 中未 submitted 的 163 tasks，加 NAS 中其余 258 tasks，
共 421 个唯一任务；与保留的 37 tasks 无交集。421 项 NAS OCI 描述图均已通过预检。

## 执行与产物

走 `launch_selection → supervise_rollout → core.cli rollout` 正式调用链，读取唯一
`config.yaml`，不使用速度测试启动方式。FP16 Qwen3.5-9B，GPU 4–7 / TP4，
rollout workers=12，max_num_seqs=12，GPU utilization=0.5，context=262144，
batch tokens=8192，max generation=81920，agent step limit=250，连续格式错误上限=10。

- 运行回执/任务原始输出：`runs/<run>/`；冻结配置：`resolved-config.json`。
- canonical trajectory/patch 与 task/run 身份：`baseline/manifest.json` 及其引用路径。
- 在线 hidden：`/VM/wangtianyi/DyConGlobal/hidden/online/<run>/`，按 session/request/token 对齐。
- 本地异步暂存：`hidden/spool/<run>/`，发布完成后回收张量暂存。
- enrichment：`enrichment/outputs/<run>/`；产品记录：enrichment/hidden 各自的 manifest。
- 正式日志：`state/supervisors/<run>/rollout.log`。

## 首条完成轨迹验收

后台事件脚本先等待正式启动门禁及真实工具观测，再执行
`hidden.evaluation.first_rollout`，通过 inotify/pidfd 触发并等待 replay 子进程退出。
按实际最早完成时间选轨迹，不跳过异常或 incomplete 样本。

检查 trajectory/patch/ledger/shard 校验和、run/task/session/request 身份、token 坐标、
reasoning newline / `</think>` 采集覆盖、全层张量有限性，并人工阅读 reasoning、行动、
工具反馈和最终 patch。FP16 vLLM 独立离线重放使用同模型、TP 与 sites，在 cold、warm、
reordered 三种路径逐点逐层比较 NAS 在线张量；阈值 cosine≥0.999、relative L2≤0.05。
重放的 max_model_len 取实际最长采集 prefix +1，避免为短样本按 256k 预留 KV。

回归测试：`tests/test_first_rollout.py tests/test_capture_repair.py tests/test_online_trace.py`，
33 passed、2 skipped。

## 首条轨迹实际验收结果

2026-10-18 20:34（UTC+8）检查时，正式进程仍在运行，9/421 tasks 已完成，均为
Submitted；其余尚未完成。GPU 4–7 每卡约占用 38.8 GiB，约余 34 GiB；prefix cache
命中率约 95%。此后不轮询全队列进度。

最早完成的是 `pytest-dev__pytest-5631`，rollout ID
`r-fe68ae281c8e7957846cad05ecbee852`，20:15:10–20:17:53，共 31 轮，正常提交。
已实际阅读 reasoning、行动、工具反馈与 patch：先复现 numpy array patch 收集时的
ValueError，再将 sentinel 的相等比较改为 identity 比较；原始复现用例修复后通过，
compat 测试 8 passed，integration 测试 15 passed / 4 skipped。整个 rollout 无格式错误。

这不是 harness 的 resolved 结论。模型也遇到缺少 hypothesis/mock、较广测试集中的
7 个失败，以及错误测试路径；部分 shell pipeline 掩盖 pytest 返回码。后加的某些
“回归测试”仅定义嵌套函数，不能独立证明收集逻辑正确；有效证据是较早直接复现的
数组 patch 用例。以上不妨碍判断调用、响应、修改、提交的 rollout 链路正常。

NAS 在线采集覆盖为 128/128：97 个 reasoning newline、31 个 `</think>`，32 sites，
missing=0、unexpected=0，张量全部有限，身份、token 坐标、ledger 与 shard 哈希通过。
其中 119 点来自 decode FULL graph，9 点来自 mixed eager。

FP16、同模型及 TP4 的 vLLM 独立离线重放于 20:31:37 完成。逐点逐 site 比较结果：

| 重放路径 | 最差 cosine | 最大相对 L2 | 验收 |
| --- | ---: | ---: | --- |
| cold | 0.99995160 | 0.9860% | 通过 |
| warm | 0.99996335 | 0.8580% | 通过 |
| reordered | 0.99997566 | 0.6992% | 通过 |

每种路径检查 128×32=4,096 个采集点/site 对。阈值为 cosine≥0.999、relative L2≤5%。
不要求逐位相同；本次最大相对 L2 小于 1%。离线执行资源受限为 2 seqs、batch tokens=512、
每卡 KV=2 GiB，max_model_len=16230，由实际最长待采集 prefix 决定；未中断正式 rollout。

原始 `audit.json/result.json` 的 `protocol_ok=false/overall_passed=false` 保留不改。
原因是旧检查只统计 `role=tool`，把最终提交的 `role=exit` 漏算为反馈：31 个命令实际
对应 30 个普通工具反馈 +1 个 Submitted exit。已逐一验证 30 个 tool_call_id 顺序一致，
最终命令为提交指令、exit submission 与保存 patch 一致，并单独写入
`hidden/reports/<run>/first-rollout/manual-review.json`，人工复核验收通过。
这项诊断误报不表示 capture 或数值测试失败。

原始审计、精确采集点、重放日志及三种路径报告均保存在
`hidden/reports/<run>/first-rollout/`，事件脚本退出码为 0。
