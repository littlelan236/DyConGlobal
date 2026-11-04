# SkillsBench 零分 verifier 三态判定

正式 SkillsBench rollout 在写入 evaluation report、导入 rollout 和结算 attempt 之前，
只审计原本可 accepted 且 `raw_reward.reward == 0` 的候选。无需新增启动参数。

- `VerifierError`：明确的结构化未完成状态，或 verifier 日志中依赖安装/测试启动最终失败且没有恢复证据。结算为 `exit_status=VerifierError`、`disposition=failed`、`resolved=null`、`success_policy=verifier_incomplete`，保留原始 reward。
- `ValidZero`：CTRF 或 pytest 汇总证明执行了测试，或任务专属契约提供完成标记。断言失败、测试错误和零分仍保留正常结果。
- `Review`：证据不足，例如缺失 CTRF、未知契约、不完整日志、仅见下载失败/重试、安装恢复但没有测试结果。保留原状态和 reward，并明确记录待核查结论。

每个候选在自己的 `diagnostics/verifier-zero-audit.json` 中记录结论、原因、冻结任务契约、所用 verifier 文件及 SHA256；evaluation-report 和 attempt 结果也包含 `verifier_zero_audit`。
审计只读本 attempt 的 `evaluation-raw.json`、`evaluation-report.json` 和 `benchflow/**/verifier/` 下的 CTRF、`test-stdout.txt`、`test-stderr.txt`，不读 `trainer/verifiers.jsonl` 或其他 repetition。

CTRF 契约只从 run selection 中绑定的任务清单核验：脚本内容必须匹配冻结清单中的 SHA256，且包含明确的 CTRF 输出命令。当前 checkout 与冻结版本不同、脚本不明确或不使用 CTRF 时，均不推断报告一定存在。`classify_zero_reward_trajectory` 的 `ctrf_expected=False` 和 `task_completion_patterns` 可供具有明确非 CTRF 契约的调用方使用；没有可靠完成标记时返回 Review。

单个 `Failed to fetch` / `Failed to download` / `Could not fetch URL` 不足以改状态。自动故障要求明确安装器终止（例如 uv 的 `× Failed to download` 加 `Caused by`）、最终 pip 解析错误或 pytest 启动失败。后续安装恢复会阻止自动改状态；stdout/stderr 无统一时序时保守返回 Review。测试执行证据优先于依赖错误文本，但结构化未完成状态优先。

本次变更仅影响加载新代码后执行并结算的 attempt，不改写历史 run。已经运行的 worker 是否采用新实现取决于其进程加载时机；不应为更新代码修改冻结 run 快照。
