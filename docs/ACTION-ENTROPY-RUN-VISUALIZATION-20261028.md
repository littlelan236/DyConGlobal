# verified80-parallel-reasoning-log1p-20261021：run 级独立指标图

日期：2026-10-28（Asia/Shanghai）。已完成远程正式分析、本地图片收集和逐字节校验。

## 结果与统计范围

- 来源 run：`verified80-parallel-reasoning-log1p-20261021`，状态 complete。
- 分析 ID：`verified80-reasoning-log1p-run-metrics-separate-20261028`。
- 全部默认 eligible/final canonical 轨迹：80 条；80 条都有 parallel step，合计 4,109 个 parallel step。未额外筛选 Submitted 或 harness resolved。
- 图形单位为 run：每条轨迹按 `assistant_index/(L-1)` 归一化到 0–100%，先线性插值到 101 点，再在 run 内按轨迹等权平均；不外推、不跨显式缺失值插值。
- 10 个指标分别成图，每图 PNG/SVG 两种格式，共 **20 个 run 级图片文件**。既有整数直方图是另附产物，不计入这 20 个文件。

## 必须区分的历史策略

该 run 创建于 2026-10-21，采用 **4 分支 estimator argmin**，冻结配置没有 consensus retained 集合或 entropy retention rule；不能视为后来实现的 `parallel_enthophy`，也不是 32 分支共识 run。

因此本次使用 `--branch-scope all`：

- H 是全部 inventory 内可识别 action 的 Shannon 熵，不是后来的动态决策池 H。
- `branches` 是生成分支数，所有 4,109 个 step 均为 4。
- `valid_actions` 是可识别 action 的有效分支数，不是动态 eligible 池大小。
- 动态 top-k 保留数量 **不可用**；未伪造 k、未将最终执行一个代表 action 当成动态保留数、未模拟 A/B/C 反事实规则。相应没有生成虚假的动态 k 数值图。

## 查看独立图片

所有本地图片在：`results/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/`。

| 独立指标图 | PNG | SVG |
|---|---|---|
| Shannon 熵 H（nats） | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/entropy.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/entropy.svg) |
| 归一化熵 H/ln(K) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/normalized_entropy.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/normalized_entropy.svg) |
| 有效 action 数 exp(H) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/effective_action_count.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/effective_action_count.svg) |
| 实际 winner 支持率 q | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_support.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_support.svg) |
| Winner 不确定性 1−q | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_uncertainty.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_uncertainty.svg) |
| Winner 惊奇度 −ln(q) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_surprisal.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_surprisal.svg) |
| Action 种类数 K | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/action_types.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/action_types.svg) |
| 实际 winner 支持分支数 | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_count.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/winner_count.svg) |
| 有效 action 分支数 | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/valid_actions.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/valid_actions.svg) |
| 生成分支数 | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/branches.png) | [查看](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/per-run/verified80-parallel-reasoning-log1p-20261021/branches.svg) |

## 首、中、末进度点

| 进度 | 平均 H（nats） | 平均生成分支数 | 平均有效 action 分支数 | 各指标贡献轨迹数 |
|---|---:|---:|---:|---:|
| 0% | 0.8688769133 | 4 | 4 | 80 |
| 50% | 0.9936173587 | 4 | 3.9875 | 80 |
| 100% | 0.2988350277 | 4 | 3.95 | 80 |

轨迹末端平均熵下降是当前观测；本次未据此声称因果解释或动态阈值效果。均值允许小数，尤其有效分支数均值不是某个实际 step 的整数预算。

## 正式统一入口

此次实际执行：

```bash
cd /home/wangtianyi/DyConGlobal
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg \
  .venvs/analysis/bin/dycon action-branch-stats \
  verified80-reasoning-log1p-run-metrics-separate-20261028 \
  --source-run verified80-parallel-reasoning-log1p-20261021 \
  --branch-scope all --line-mode per-run --plot-layout separate \
  --progress-points 101 --format png --format svg

.venvs/analysis/bin/dycon action-branch-stats \
  verified80-reasoning-log1p-run-metrics-separate-20261028 --check
```

同 ID 已存在，重复生成必须使用新 ID。默认一次生成全部可用指标；重复 `--line-metric` 可以只选需要的指标，如 entropy、valid_actions、branches。

远程分析根目录：

```text
/home/wangtianyi/DyConGlobal/action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/
```

关键审计文件：[图片清单](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/plots/plot-index.json)、[平均曲线 CSV](../action/outputs/verified80-reasoning-log1p-run-metrics-separate-20261028/curves/per-run.csv)、[验证记录](../temp/action_entropy_run_visualization_20261028/VERIFICATION.json)。本地只收集展示与数值文件子集，完整源码/来源证据仍保存在远程正式分析中；不能把本地展示副本冒充完整可验证分析目录。

## 实施及验证

- 在现有 `action/evaluation/branch_curves.py`、`action/analyzers/branches.py` 和 `core/cli.py` 中增加可选 `--plot-layout separate`、批量图片索引与完整覆盖校验，没有另造绘图 launcher 或 trajectory reader。
- `branches` 曲线能力由冻结 `branch_count_curves` 显式启用；历史分析按旧列集合重算，默认 panels 布局不变。
- 新增测试覆盖单图单指标、所有绘图模式、分支数显式启用、argmin 不伪造 retained 数、批量 CLI、PNG/SVG 和删图后重签 receipt 仍被拒绝。
- 相关 CPU 测试：**114 passed**；正式实际分析 `--check`：verified=true、80 trajectories、4109 steps、current_implementation_matches_snapshot=true。
- 另一个历史平均分析 `consensus32-all-trajectory-average-20261027`（86 条轨迹、2,703 steps）也通过 `--check`；正常代码升级导致其 implementation snapshot 匹配提示为 false，不影响旧产物完整性。
- Ruff 检查仍有 11 项既有 I001 import 排序问题；对四份文件逐一使用修改前字节作相同检查，基线与当前数量及类型一致，本次没有新增 lint 项。未扩大修改范围整理无关 import。
- 本地收集的 30 个图片/数值/元数据文件逐项匹配远程 receipt SHA；已目视检查主熵图与生成分支数图。
- 未启动模型、GPU 或 rollout；未修改源 run 的冻结配置、轨迹、策略或历史分析产物。修改均通过基线/发布 SHA 防并发覆盖，并在远程任务 temp 目录备份。
- 完整动态 H/k/N 指标和 `--line-metric-set entropy-retention` 预设仍属于先前计划，未因本次历史 argmin run 的绘图而宣称完成。
