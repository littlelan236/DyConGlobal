# tactRefractor 轨迹迁移记录（2026-10-17）

> Historical migration snapshot: 96 interrupted checkpoints were subsequently removed. The current catalog contains 1,108 trajectories. See [cleanup report](interrupted-cleanup-20261017.md).

迁移目标：`/home/wangtianyi/DyConGlobal`。范围是用户提供的 `rolloutReport.md` 逐 run 表中列出的 run 和 canonical baseline，采用用户确认的当前全部 attempt，包含 baseline、steered、smoke、成功和失败轨迹。旧工程只读，新工程保存独立副本。

## 结果

1,927 个原始文件按 SHA-256 去重后得到 1,204 条轨迹，覆盖 258 个 task。全部 task 的 `partition=train`，全部轨迹的 `purposes=["training"]`。

| 互斥结果分类 | 数量 |
|---|---:|
| resolved | 321 |
| submitted（尚未确认成功） | 551 |
| unsubmitted | 332 |

已提交总数为 872，包括已成功的 321 条。已完成评测 619 条，其中成功 321、未解决 298。评测基础设施错误或缺失补丁 311 条；没有可绑定评测 274 条。共 585 条保留 `resolved=null`。MissingPatch 和 EvaluationFailed 不转换成 `resolved=false`。

| 来源模型 | 轨迹数 |
|---|---:|
| Qwen3.5-9B | 1054 |
| Qwen3.5-27B | 27 |
| Qwen3.6-27B | 61 |
| Qwen3.6-35B-A3B | 62 |

## 唯一 manifest 和审计文件

唯一任务、分区和轨迹目录仍为 `baseline/manifest.json`，没有新增 splits 配置，没有设置 baseline 评测选择。rollout 通过 task_key 引用 task.instance_id，保存 run_id、attempt_id、submission_status、evaluation_status、resolved、选定评测 ID、轨迹和补丁哈希。

`provenance.source_occurrences` 保存全部复用副本的原始 run、attempt、path 和是否为旧工程当前选中的 attempt。旧 steering、generation、模型和运行配置身份保存在同一迁移目录的证据快照中。baseline、steered 都用于训练，但保留各自来源条件。

迁移快照目录：`baseline/provenance/tact-report-20261017/`。

- `source-rolloutReport.md`：原始清单副本，保留原文件字节。
- `plan.json`：固定导入计划、范围差异、源哈希和评测绑定，只作为审计证据。
- `trajectory-index.csv`：每条去重轨迹一行，含 task_id、run_id、attempt_id、partition、purpose、status、独立提交/评测字段及文件路径。
- `source-occurrences.csv`：每个原始文件一行，保存 1,927 个来源与 1,204 条轨迹的对应关系。
- `legacy-config-identities.json`：各 run 的研究配置和模型身份快照。
- `evidence/`：冻结的 harness summary、predictions 和实际被评测补丁。
- `receipt.json`：数量、校验结果及导出文件哈希。

CSV 是迁移时从唯一 manifest 导出的只读快照，不参与运行配置。后续更新评测以 manifest 为准。原始轨迹、补丁和正式评测证据分别存于 `baseline/trajectories/`、`baseline/patches/`、`baseline/evaluations/`，沿用现有 .gitignore，不加入代码版本库。

## 与原报告的差异

原报告统计 1,926 个文件、1,203 条去重轨迹；当前多出 seed42 gated 的 `astropy__astropy-13579/attempts/attempt-001`。部分报告时仍在运行的轨迹也已结束。

原报告成功数为 266；当前为 321，增加的 55 条来自后来完成的 seed42 cap（25）和 gated（30）评测。

只有同一 run 当前选中的 attempt 与实际评测补丁 SHA-256 相符时才绑定结果；canonical baseline 同时校验旧 manifest 中的轨迹 SHA-256。相同原始轨迹的复用副本共享一个样本；不会把相同补丁的成功结果扩散给其他 attempt。完成评测结果互相冲突时拒绝迁移；基础设施失败不会覆盖已有完成评测。

旧轨迹没有可验证的模型权重指纹，因此明确使用 `legacy-unverified-*`，并标记 `checkpoint_verified=false`，保留模型名称和旧配置身份。后续提取 hidden 前仍需核验并显式绑定实际 checkpoint。此次没有选择或训练剩余难度模型，没有生成 latent。

## 复现与验证

迁移脚本：`baseline/scripts/migrate_tact.py`，复用现有 rollout/evaluation 导入器和 Catalog。

```bash
cd /home/wangtianyi/DyConGlobal
.venvs/runner/bin/python -m baseline.scripts.migrate_tact --root . prepare \
  --source /home/wangtianyi/tactRefractor --report /path/to/rolloutReport.md \
  --migration-id NEW_UNIQUE_MIGRATION_ID
.venvs/runner/bin/python -m baseline.scripts.migrate_tact --root . apply \
  --plan baseline/provenance/NEW_UNIQUE_MIGRATION_ID/plan.json
.venvs/runner/bin/dycon catalog-check
```

现有迁移可重新 apply，完成后校验回执并返回，无重复导入。prepare 不修改 manifest，apply 在写入前核对全部源轨迹和冻结评测证据哈希。

验证结果：Ruff 全项目检查通过；runner 测试 27 passed、2 skipped，1 条上游 LiteLLM 弃用警告。新增的 5 项测试覆盖同补丁不同 attempt 的标签隔离、按内容去重及重复执行、源轨迹/评测补丁篡改拒绝、冲突评测拒绝、显式清单范围。两个跳过项是已有 Torch/Matplotlib 可选依赖测试。

最终 Catalog 对全部轨迹、补丁和评测报告的哈希校验通过；独立检查确认 258 个 task 全部 train、1,204 个唯一轨迹哈希、1,927 个来源映射及原始轨迹均非软链接。重复 apply 后 catalog_revision 保持 1235。

计划 SHA-256：`ae0e4bd2d1baeb23c58191050828540b3a5283d881d6e90bf10730832321a421`。
