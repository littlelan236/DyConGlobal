# 同 task 多次 rollout 实施记录（2026-10-24）

项目：`/home/wangtianyi/DyConGlobal`
主机：`10.249.187.9`

## 已完成

- `run create --repeat-count R` 已加入统一 CLI。
- schema-2 active run 使用 `rollout/work-items.json` 保存 task × repetition 工作项。
- runner 使用平坦队列和 bounded futures；一个 repetition 完成后立即补位。
- repetition、sample seed、基础设施 attempt 和 canonical rollout 身份分离。
- 新增一次性迁移命令：`runs-migrate-to-repetition`、`migration-verify`。
- 当前保存的 runs 已迁移：5 个 active schema-2 run（`repeat_count=1`），1 个 artifact-only archive。
- 原始目录归档于 `archive/runs-schema1/`；`migration/CUTOVER.json` 已生成。

## 迁移结果

迁移清单：`migration/runs-to-repetition-20261024.json`
active run：

- `migration-closeonly331-20261021`
- `nonverified600-baseline-close-gpu47-w24-s16-20261022`
- `verified80-parallel-baseline-20261021`
- `verified80-parallel-reasoning-log1p-20261021`
- `verified80-parallel-step-linear-20261021`

无 plan 的 `capture-closeonly325-gpu47-w18-s16-bt2048-20261019-184246` 已作为 artifact-only 归档，未伪造成可执行 run。

## 验证

- 重复采样、run plan、runner、evidence、harness 子集：`79 passed`。
- parallel 专项：`46 passed, 1 skipped`。
- 完整 runner pytest：`567 passed, 29 skipped, 1 failed`。
- 唯一失败是已有根配置引用的 predictor manifest 缺失：
  `runtime/parallel-artifacts/ridge-event-reasoning-best-20261019/manifest.json`。

该历史 artifact 缺失未被伪造或补写；GPU/capture 真实大规模重复 rollout 仍需专项资源验收。

## ??????????????

- Harness ? `invalid_patch`?`infra_error` ???? `network_error` ??????????????????????? `ConnectionError`??? attempt ???
- Managed rollout ????? GPU ? MIG UUID?MIG ?? run ????? TP=1??? GPU TP=4 ????????????MIG ???????????? serving launch receipt?
- ???????????`587 passed, 29 skipped`?MIG/serving ?????`69 passed, 2 skipped`?
