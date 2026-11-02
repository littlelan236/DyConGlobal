# OpenHands FormatError / ParallelSelectionError 修复记录（2026-11-01）

未声明工具名、非法 JSON 工具参数由 OpenHandsProtocolError 表示，并携带可恢复的 FormatError 消息。bridge 返回 owner-bound 的 FormatError、code=format_error、retryable=true；SDK transport 转换为 FunctionCallValidationError，由 Agent 追加纠错消息并继续下一轮，不增加 native retry。

只有全部并行分支均有匹配的 native reasoning-loop receipt，且 session_id、decision_id、branch_id、api_attempt_id、native_request_id 一致时，才生成 all_parallel_branches_reasoning_loop。格式错误与 loop 混合时保留 FormatError 恢复。

隔离工作树的最终格式/loop 回归为 153 项通过。Qwen9B GPU0 自然 qualification run `skillsbench-sdk-v4-qwen9b-natural-qualify-20261101` 有 56 个请求、47 个 tool/result pair；格式错误后继续到 step 50，最终 AgentCompleted + accepted，cleanup 完成。qualification receipt：`0933324fc6440313262df10378bdd6521eae8330882326837da42ee52eb565a0`。SDK publication manifest receipt：`85ba1c820f8b08d4a5ec1b8cabfe257a218cbe35b6207bf5f31092af6efa307c`。

capture canary 因共享本地磁盘 reserve 超限在启动阶段失败，已通过统一 CLI 确认 cleanup；不作为 capture 验收通过。原目标 run 未恢复、未改写，GPU0–1 其他活动 run 未停止；7 条历史 SDK timeout 的具体网络阶段仍为 unknown。


主目录合并记录：publication 已独立复制到 runtime/dependency-snapshots/openhands-sdk-format-v4-20261101，4843 个文件通过 dependencies-verify；主目录 profile 使用 qualified-main-20261101 并绑定主目录 publication。

## 主目录复制完成

本次不新增验证 run。主目录 SDK 日志入口、清理前本地日志收集、ACP timeout 投影、dependency builder 和测试均已补齐；主目录相关回归 153 项通过，替代此前 90 passed / 8 failed 的不完整合并状态。

publication 共 4843 个文件，已通过正式 dependencies-verify，并与隔离源逐文件对照：不共享源 inode，manifest 原字节及 receipt 不变。副本位于 runtime/dependency-snapshots/openhands-sdk-format-v4-20261101。

主目录使用 profile runtime/agent-profiles/openhands-sdk-v4-qualified-main-20261101/manifest.json；其依赖路径绑定主目录 publication。迁移时仅允许两端完整校验通过且 manifest 完全一致，通过正式 skillsbench qualify 重用既有自然资格 run 的 56 个请求、47 个 tool/result pair，没有创建新的运行。原始 qualification 文件与 receipt 保留。

主目录 profile SHA256：aae6c2950c5732abfe5d4e310feec65921cccc6f1aca2a62977fe3da7b18cdc1；正式迁移 qualification receipt：1a0a63abda7abd41de5ba2557d70272dc40fbe20be609e150e1495573a733fb1。

未修改根 config.yaml、旧冻结 run/profile/publication 或恢复原目标 run；原 11 个受保护文件 hash 未变化。capture canary 仍保留磁盘 reserve 失败记录，不宣称 capture 验收通过；7 条历史 timeout 阶段仍为 unknown。
