# SkillsBench 零分 verifier 三态判定

正式 SkillsBench rollout 在写入 evaluation report、导入 rollout 和结算 attempt 之前，
只审计原本可 accepted 且 `raw_reward.reward == 0` 的候选。无需新增启动参数。

- `VerifierError`：明确的结构化未完成状态，或 verifier 日志中依赖安装/测试启动最终失败且没有恢复证据。结算为 `exit_status=VerifierError`、`disposition=failed`、`resolved=null`、`success_policy=verifier_incomplete`，保留原始 reward。
- `ValidZero`：CTRF 或 pytest 汇总证明执行了测试，或任务专属契约提供完成标记。断言失败、测试错误和零分仍保留正常结果。
- `Review`：证据不足，例如缺失 CTRF、未知契约、不完整日志、仅见下载失败/重试、安装恢复但没有测试结果。保留原状态和 reward，并明确记录待核查结论。

每个候选在自己的 `diagnostics/verifier-zero-audit.json` 中记录结论、原因、冻结任务契约、所用 verifier 文件及 SHA256；evaluation-report 和 attempt 结果也包含 `verifier_zero_audit`。
审计只读本 attempt 的 `evaluation-raw.json`、`evaluation-report.json` 和 `benchflow/**/verifier/` 下的 CTRF、`test-stdout.txt`、`test-stderr.txt`，不读 `trainer/verifiers.jsonl` 或其他 repetition。

CTRF 契约只从 run selection 中绑定的任务清单核验：脚本内容必须匹配冻结清单中的 SHA256，且包含明确的 CTRF 输出命令。当前 checkout 与冻结版本不同、脚本不明确或不使用 CTRF 时，均不推断报告一定存在。`classify_zero_reward_trajectory` 的 `ctrf_expected=False` 和 `task_completion_patterns` 可供具有明确非 CTRF 契约的调用方使用；没有可靠完成标记时返回 Review。

### Accepted reward below-one audit

The read-only `dycon skillsbench audit-rewards RUN_ID --output DIR` command audits every current projected sample with `disposition=accepted` and a finite numeric `raw_reward.reward < 1`. It includes partial rewards and negative values (the latter are Review / `invalid_reward_range`); missing, boolean, string, NaN and infinite rewards are counted as data anomalies, never coerced to zero. Current attempts and retry supersession come from the same projection used by run status.

The command requires a frozen SkillsBench run and a new output directory. It does not read another run, mutate attempts or status, launch a verifier, or change online settlement. It writes `summary.json`, `samples.jsonl`, `samples.csv`, and `report.md`. Policy `skillsbench_accepted_reward_audit_v1` classifies completed test evidence as `ValidOutcome`, proven terminal verifier/dependency failures as `VerifierError`, and unresolved or conflicting evidence as `Review`. The terminal summary stays compact. Reports record source hashes and mark progress or attempt changes during the scan as incomplete.

Exit status is 0 when the audit report is produced, including reports with Review findings; 1 for an invalid run, core loading or output failure; 2 for CLI argument errors; and 3 for an incomplete source read/change. An active run is only observed at scan time and is not represented as its future terminal state.

```bash
.venvs/runner/bin/dycon skillsbench audit-rewards RUN_ID --output runs/RUN_ID/analysis/accepted-reward-audit-ID
```

单个 `Failed to fetch` / `Failed to download` / `Could not fetch URL` 不足以改状态。自动故障要求明确安装器终止（例如 uv 的 `× Failed to download` 加 `Caused by`）、最终 pip 解析错误或 pytest 启动失败。后续安装恢复会阻止自动改状态；stdout/stderr 无统一时序时保守返回 Review。测试执行证据优先于依赖错误文本，但结构化未完成状态优先。

本次变更仅影响加载新代码后执行并结算的 attempt，不改写历史 run。已经运行的 worker 是否采用新实现取决于其进程加载时机；不应为更新代码修改冻结 run 快照。

## Accepted reward audit CLI

DyConGlobal provides a read-only run-level audit for accepted SkillsBench samples with rewards below one:

```bash
.venvs/runner/bin/dycon skillsbench audit-rewards RUN_ID --output OUTPUT_DIR
```

The command requires an explicitly named frozen SkillsBench run and reuses the current sample/attempt projection used by run status, including retry supersession. It scans all current samples where `disposition=accepted` and finite numeric `raw_reward.reward < 1`. Rewards 0, partial rewards, and negative values are candidates; missing, null, boolean, string, NaN, infinity, and rewards equal to or above one are not silently converted or included.

Reports are written only below the new `--output` directory:

- `summary.json`: scope, source identity hashes, counts, task/rule summaries, and completeness;
- `samples.jsonl`: detailed per-candidate evidence and classification;
- `samples.csv`: tabular candidate records;
- `report.md`: readable summary and unresolved Review list.

The policy version is `skillsbench_accepted_reward_audit_v1`. The three classifications are `VerifierError` (unrecovered verifier failure proven by evidence), `ValidOutcome` (completed evaluation evidence explains the low reward), and `Review` (evidence or contract ownership is insufficient). The command is read-only: it does not rerun models or verifiers and does not rewrite attempts, evaluation receipts, status, capture, or configuration. An existing output directory is rejected.
