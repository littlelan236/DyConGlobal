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
33 passed、2 skipped。实际首条轨迹与 replay 结果待事件脚本完成后补充。
