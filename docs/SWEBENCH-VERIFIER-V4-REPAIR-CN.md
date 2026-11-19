# 145项 gold 复测与 v4 修改记录

本次修改对应《本轮评测阻塞原因与修改方案》。全量复测的新 run：`swebench-verifier-repair-20261117-gold145-v3`；7项独立 synthetic negative run：`swebench-verifier-repair-20261117-negative-v3`。运行是否启动以 startup-confirmation.json 为准，结果以 completion.json 为准。

## 已实现

- 审计策略 `swebench_verifier_audit_v4` 支持唯一失败块内的 pytest DID NOT WARN；非断言异常须有绑定当前 evidence_digest、source hashes、required test、异常类型和失败块摘要的 subject finding，以及完整 call/setup/teardown 回执。环境候选、超时、身份冲突、异常退出、skip、缺失测试继续拒绝有效分数。
- 正式 `EnvironmentPolicy.wrap_eval → swebench_receipts → harness adapter → audit collector` 采集每进程 pytest session、解释器、argv、祖先进程、collected nodeID、分阶段结果、异常与 exitstatus。随机 session binding 与 command/output SHA256 由宿主回执绑定。重复/嵌套 session 不猜测归属。tox 外层与 pytest 内层仍分开判断。
- 同一个 TestSpec 的相同原脚本只生成一次 wrapper，避免官方 harness 多次读取 property 时产生不同 token。
- 资源通过统一容器 factory 的只读挂载与 bootstrap 准备；宿主和容器内均核对 SHA256，拒绝路径逃逸与覆盖不同内容。
- pvlib-1026 的原始资源来自 upstream PR1026 head `f42bf51a27fb9d7fc553c6fd905ce483aec423c0`，2,025 bytes/1,620 NULs，SHA256 `0b2cc0bae5f1f41f05baec4388fb281f39bdbd439a1486873d9a9ad3ec7d4f69`。
- pvlib-718 synthetic patch 保留 gold 新增模块、导入和 API，只把返回参数 IL 改为 IL+1.0。reference gold、test patch、required集合不变。
- 4项非断言 synthetic no-op 的归因契约明确绑定冻结任务摘要和已审阅的实现/测试语义。新运行后仅当具体异常、栈片段、消息及完整阶段证据全部匹配时生成审阅 finding，由正式 `harness audit --labels` 写到新目录。其他提交不能使用该契约。
- 任务级 gold通过、任务级正负对照验收、类别视图分别记录；不再用类别 bool 隐藏个别通过。总分母仍为1276；流程 finished 与验收 complete/partial 分开。

## 验证与边界

已完成一次13条历史证据只读回放，source_changes=[]；旧无阶段回执负例仍待审。CPU回归、真实 shell/pytest 和 tox4.16/pytest8.3集成的具体回执在同任务 temp 目录。全量145项测试尚未据此宣称通过。

新环境 profile `swebench-verifier-environment-20261117-v3` SHA256：`ae4ba82c57c0a9b9ede861f5e4c58def68791c481a2eb87519b774afa65b931c`。它继承 v2 精确 wheel pins，新增1026资源。145项集合 SHA256：`58601ba5d2f4a1d7dd2956c6b4fd297eeaddb77a74e0ca28d2c061e21af2ff83`。

原计划中图形系统库、gettext、外部网络资源、镜像 ownership、参数ID和21项P2因果实验尚未因此完成；本次全量测试包含这些任务，保留实际未恢复结果，不能将一次全量启动等同于全部修复。未改宿主Docker/UID映射、NAS内容、历史run、sealed数据集或checkpoint。

## 后台执行

正式CLI每阶段阻塞运行，单次gold阶段上限18小时、negative阶段3小时，托管服务上限24小时。没有进度/status循环。gold结束后自动审计并写145项ledger，再执行7项negative及独立审阅；写清理与最终回执。启动确认后结束会话，不轮询。

远程任务目录：`/home/wangtianyi/DyConGlobal/temp/swebench-verifier-repair-20261117-v3`。正式评分原始文件不因外部finding回放而改写。
