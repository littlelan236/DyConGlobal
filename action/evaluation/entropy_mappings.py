"""Offline entropy-retention comparisons on frozen historical decision pools.

Uses the production retention operator and the shared equal-trajectory aggregator.
Never changes the historical winner or executes a hypothetical trajectory.
"""

import base64
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from action.evaluation.branch_curves import build_curve_tables
from core.io import hash_json, owned_path, read_json, write_json
from rollout.consensus import consensus_options, select_consensus, validate_consensus_evidence
from rollout.entropy import RETENTION_RULES, entropy_retention

LEGACY_VERSION = "historical_pool_entropy_mapping_v1"
CANONICAL_REFERENCE_VERSION = "historical_pool_entropy_mapping_v2"
PREVIOUS_VERSION = "historical_pool_entropy_mapping_v4"
VERSION = "historical_pool_entropy_mapping_v5"
LEGACY_METRICS = (
    ("pool_entropy", "决策候选池 Shannon 熵 H", "nats"),
    ("keep_count", "映射得到的保留分支数量 k", "branches"),
    ("pool_count", "决策候选池分支数量 N", "branches"),
    ("cutoff_score", "映射 top-k 最后一个分支的 estimator score", "score"),
)
METRICS = (
    ("pool_entropy", "决策候选池 Shannon 熵 H", "nats"),
    ("keep_count", "映射得到的保留分支数量 k", "branches"),
    ("winner_support_before", "estimator 筛选前 winner 支持率", "ratio"),
    ("winner_support_after", "动态阈值筛选后 winner 支持率", "ratio"),
    ("winner_changed_rate", "estimator 筛选后 winner 改变率", "ratio"),
    ("cutoff_score", "映射 top-k 最后一个分支的 estimator score", "score"),
)
COMPARISON_METRIC = ("winner_changed_vs_fixed_ratio", "熵映射 winner 相对固定比例 winner 的改变率", "ratio")
MAPPING_RULES = RETENTION_RULES + ("fixed_ratio_v1",)
FIXED_RATIO_DEFAULT = 0.5
RULE_LABELS = {
    "effective_count_v1": ("A", "exp(H)"),
    "shannon_exp_keep_v1": ("B", "N − (N−m) exp(−H)"),
    "baseline_floor_v1": ("C", "N [1−(1−r₀) exp(−H)]"),
    "normalized_entropy_v1": ("E", "m + (N?m) H / ln(N)"),
    "effective_count_linear_v1": ("F", "m + (N?m)(exp(H)?1)/(N?1)"),
    "effective_count_quadratic_v1": ("G", "m + (N?m)[(exp(H)?1)/(N?1)]^2"),
    "effective_count_type_linear_v1": ("H", "m + (N?m)(exp(H)?1)/(K?1)"),
    "shannon_saturating_a25_v1": ("I", "m + (N?m)[1−exp(−2.5H)]"),
    "effective_count_sigmoid_v1": ("J", "m + (N?m) σ(1.5[exp(H)−√N])"),
    "effective_count_sigmoid_4_v1": ("K", "m + (N?m) σ(4[exp(H)−√N])"),
    "effective_count_tilt_a1_v1": ("L", "exp(H + H(H−1))"),
    "effective_count_scale_a08_v1": ("M", "exp(H)[1+0.8H²]"),
    "effective_count_sigmoid_c7_v1": ("N", "m + (N−m) σ(4[exp(H)−7])"),
    "entropy_logistic_c18_a8_v1": ("O", "m + (N−m) sigmoid(8[H−1.8])"),
    "effective_width_logistic_c6_b2_v1": ("P", "m + (N−m) sigmoid(2[e^H−6])"),
    "effective_width_hill_c6_p8_v1": ("Q", "m + (N−m) E⁸/(E⁸+6⁸)"),
    "effective_width_hill_c7_p4_v1": ("R", "m + (N−m) E⁴/(E⁴+7⁴)"),
    "effective_width_power_p2_v1": ("S", "m + (N−m)[(E−1)/(N−1)]²"),
    "fixed_ratio_v1": ("D", "ceil(0.5 × N)"),
}


def metric_specs_for_version(version):
    if version not in {VERSION, PREVIOUS_VERSION, LEGACY_VERSION, CANONICAL_REFERENCE_VERSION}:
        raise ValueError("Unknown entropy mapping version: " + str(version))
    return LEGACY_METRICS if version == LEGACY_VERSION else METRICS + (COMPARISON_METRIC,) if version == VERSION else METRICS


def mapping_version(analysis):
    version = analysis["entropy_mapping_comparison"]["version"]
    metric_specs_for_version(version)
    return version


def decision_mapping_input(decision, record, config, *, version=VERSION):
    """Recover the final scored/action-comparable pool, not inventory or top-k."""
    from action.analyzers.branches import _action

    parallel = config.get("parallel", {})
    if parallel.get("mode") not in {"parallel_consensus", "parallel_enthophy"}:
        raise ValueError("Entropy mapping comparison requires a consensus source run")
    if decision.get("selection_mode") != parallel["mode"]:
        raise ValueError("Mapping source decision mode differs from frozen run")
    options = consensus_options(parallel)
    validate_consensus_evidence(decision, options)
    pool, excluded = [], []
    for branch in decision["candidates"]:
        bid = branch["branch_id"]
        status = branch.get("status")
        if not isinstance(status, str):
            raise ValueError("Mapping comparison requires recorded candidate statuses")
        if status != "eligible":
            excluded.append({"branch_id": bid, "reason": status})
            continue
        score = branch.get("score")
        if type(score) not in (float, int) or not math.isfinite(score):
            raise ValueError("Mapping eligible candidate has no finite estimator score")
        action, reason = _action(branch)
        if reason == "raw_response_and_action_unavailable":
            raise ValueError("Mapping input lacks unfiltered candidate action evidence")
        if action is None:
            excluded.append({"branch_id": bid, "reason": "action_identity_unavailable"})
            continue
        if record["actions"].get(str(bid)) != action:
            raise ValueError("Mapping action differs from frozen decision record")
        pool.append({"branch_id": bid, "score": score, "action": action})
    pool.sort(key=lambda branch: (branch["score"], branch["branch_id"]))
    metric_specs_for_version(version)
    return {"version": version, "pool_stage": "post_loop_escape_pre_estimator_top_k",
            "source_mode": parallel["mode"],
            "source_retention_rule": options.get("retention_rule"),
            "minimum": options["min_retained_branches"], "retention_ratio": options["retention_ratio"],
            "ranked_candidates": pool, "excluded": excluded,
            "actual_retained_branch_ids": record["retained_branch_ids"]}


def _mapping_budget(counts, minimum, rule, retention_ratio):
    if rule != "fixed_ratio_v1":
        return entropy_retention(counts, minimum, rule, retention_ratio)
    n = sum(counts)
    keep = 0 if n < minimum else min(n, max(minimum, math.ceil(FIXED_RATIO_DEFAULT * n)))
    return {"definition": "fixed_score_ratio", "retention_rule": rule, "eligible_count": n,
            "action_types": len(counts), "entropy": None, "min_retained_branches": minimum,
            "raw_keep": FIXED_RATIO_DEFAULT * n, "keep_count": keep,
            "retention_ratio": FIXED_RATIO_DEFAULT, "retention_ratio_used": True,
            "effective_retention_ratio": keep / n if n else None,
            "status": "available" if n >= minimum else "insufficient_branches" if n else "unavailable"}


def replay_mapping_winner(pool, source, rule, *, keep_all=False):
    """Reuse production action voting and all score/mean/branch tie breakers."""
    candidates = []
    for row in pool:
        response = {"choices": [{"message": {"tool_calls": [
            {"function": action} for action in json.loads(row["action"])]}}]}
        candidates.append(SimpleNamespace(branch_id=row["branch_id"], message={},
            response=SimpleNamespace(model_dump=lambda mode="json", value=response: value)))
    if keep_all:
        ratio, minimum, retention = 1.0, source["minimum"], None
    elif rule == "fixed_ratio_v1":
        keep = max(source["minimum"], math.ceil(FIXED_RATIO_DEFAULT * len(pool)))
        ratio, minimum, retention = keep / len(pool), keep, None
    else:
        ratio, minimum, retention = source["retention_ratio"], source["minimum"], rule
    winner, metadata = select_consensus(candidates, {row["branch_id"]: row["score"] for row in pool},
        ratio, minimum, retention_rule=retention)
    return metadata.get("consensus_action") if winner is not None else None


def mapping_rows(records, trajectories, *, version=VERSION):
    """Run all production mappings per step before any interpolation or average."""
    lengths = {(t["source_run_id"], t["rollout_id"]): t["assistant_steps"] for t in trajectories}
    result = []
    for record in records:
        source = record["entropy_mapping_input"]
        if source["version"] != version:
            raise ValueError("Unknown entropy mapping input version")
        metric_specs_for_version(version)
        pool = source["ranked_candidates"]
        if pool != sorted(pool, key=lambda item: (item["score"], item["branch_id"])):
            raise ValueError("Mapping pool ranking differs from score/branch ordering")
        counts = Counter(item["action"] for item in pool)
        actual_winner = record["actions"].get(str(record["selected_branch_id"]))
        reference_winner = (replay_mapping_winner(pool, source, None, keep_all=True)
                            if version in {VERSION, PREVIOUS_VERSION} else actual_winner)
        before_support = (sum(item["action"] == reference_winner for item in pool) / len(pool)
                          if pool and reference_winner is not None else None)
        fixed_budget = _mapping_budget(list(counts.values()), source["minimum"], "fixed_ratio_v1", FIXED_RATIO_DEFAULT)
        fixed_winner = (replay_mapping_winner(pool, source, "fixed_ratio_v1")
                        if fixed_budget["keep_count"] else None)
        length = lengths[(record["source_run_id"], record["rollout_id"])]
        for rule in MAPPING_RULES:
            budget = _mapping_budget(list(counts.values()), source["minimum"], rule, source["retention_ratio"])
            kept = pool[:budget["keep_count"]]
            kept_counts = Counter(item["action"] for item in kept)
            if kept_counts and version != LEGACY_VERSION:
                mapped_winner = replay_mapping_winner(pool, source, rule)
                after_support = (kept_counts.get(reference_winner, 0) / len(kept)
                                 if reference_winner is not None else None)
                winner_changed = (float(mapped_winner != reference_winner)
                                  if mapped_winner is not None and reference_winner is not None else None)
            else:
                mapped_winner, after_support, winner_changed = None, None, None
            row = {**{key: record[key] for key in (
                "source_run_id", "rollout_id", "instance_id", "decision_id", "step_id", "assistant_index")},
                "normalized_progress": record["assistant_index"] / (length - 1) if length > 1 else 0.0,
                "retention_rule": rule, "interpretation": "counterfactual_on_observed_pool",
                "pool_entropy": budget["entropy"], "pool_count": budget["eligible_count"],
                **({"winner_support_before": before_support, "winner_support_after": after_support,
                    "winner_changed_rate": winner_changed, "actual_winner_action": actual_winner,
                    "mapped_winner_action": mapped_winner} if version != LEGACY_VERSION else {}),
                "keep_count": budget["keep_count"], "raw_keep": budget["raw_keep"],
                "cutoff_score": kept[-1]["score"] if kept else None,
                "cutoff_branch_id": kept[-1]["branch_id"] if kept else None,
                "retained_branch_ids": [branch["branch_id"] for branch in kept],
                "actual_retained_count": len(source["actual_retained_branch_ids"]),
                "minimum": source["minimum"], "retention_ratio": source["retention_ratio"],
                "status": budget["status"]}
            if version == VERSION:
                row["fixed_ratio_winner_action"] = fixed_winner
                row["fixed_ratio_reference"] = FIXED_RATIO_DEFAULT
                row["winner_changed_vs_fixed_ratio"] = (float(mapped_winner != fixed_winner)
                    if mapped_winner is not None and fixed_winner is not None else None)
            result.append(row)
            if version in {VERSION, PREVIOUS_VERSION}:
                row["reference_winner_action"] = reference_winner
                row["winner_reference"] = "unfiltered_pool_consensus_action"
    return result


def mapping_curve_tables(rows, trajectories, points, *, version=VERSION):
    raw, normalized = [], []
    columns = [(key, key, "counterfactual") for key, _, _ in metric_specs_for_version(version)]
    for rule in MAPPING_RULES:
        rule_rows = [row for row in rows if row["retention_rule"] == rule]
        r, n = build_curve_tables(rule_rows, trajectories, points, metric_specs=columns)
        raw.extend({**row, "retention_rule": rule} for row in r)
        normalized.extend({**row, "retention_rule": rule} for row in n)
    return raw, normalized


def mapping_plot_index(trajectories, analysis):
    version = mapping_version(analysis)
    entries = []
    for run in sorted({t["source_run_id"] for t in trajectories}):
        for rule in MAPPING_RULES:
            for metric, _, _ in metric_specs_for_version(version):
                if rule == "fixed_ratio_v1" and metric == "winner_changed_vs_fixed_ratio":
                    continue
                entries.append({"source_run_id": run, "retention_rule": rule, "metric": metric,
                    "interpretation": "counterfactual_on_observed_pool",
                    "files": [f"plots/entropy-mappings/{run}/{rule}/{metric}.{ext}"
                              for ext in analysis["formats"]]})
    return {"version": version, "plot_layout": "one_metric_and_rule_per_figure", "entries": entries,
            "figure_count": len(entries), "file_count": sum(len(entry["files"]) for entry in entries)}


def render_mapping_plots(stage, curves, trajectories, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from action.evaluation.branch_histograms import plot_style

    index = mapping_plot_index(trajectories, analysis)
    labels = {key: (label, unit) for key, label, unit in metric_specs_for_version(mapping_version(analysis))}
    colors = dict(zip(MAPPING_RULES, ("#2563eb", "#0d9488", "#d97706", "#9333ea", "#7c3aed", "#db2777", "#be123c", "#0891b2", "#4f46e5", "#16a34a", "#ca8a04", "#0f766e", "#be185d", "#4338ca", "#7e22ce", "#0369a1", "#b91c1c", "#a16207", "#166534")))
    with plt.rc_context(plot_style()):
        for entry in index["entries"]:
            rule, run, metric = (entry[key] for key in ("retention_rule", "source_run_id", "metric"))
            points = [row for row in curves if row["retention_rule"] == rule
                      and row["source_run_id"] == run and row["metric"] == metric]
            points.sort(key=lambda row: row["x"])
            fig, ax = plt.subplots(figsize=(11.5, 6), dpi=analysis["dpi"])
            fig.subplots_adjust(left=0.09, right=0.97, bottom=0.25, top=0.72)
            xs = [row["x"] for row in points]
            ys = [float("nan") if row["mean"] is None else row["mean"] for row in points]
            ax.plot(xs, ys, color=colors[rule], linewidth=2.2, label="历史候选池上的轨迹等权均值")
            lower = [float("nan") if row["mean"] is None or row["standard_error"] is None
                     else row["mean"] - row["standard_error"] for row in points]
            upper = [float("nan") if row["mean"] is None or row["standard_error"] is None
                     else row["mean"] + row["standard_error"] for row in points]
            ax.fill_between(xs, lower, upper, color=colors[rule], alpha=0.13, label="±1 标准误")
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("归一化轨迹进度（0–100%）")
            ax.set_ylabel(labels[metric][1])
            if metric.startswith("winner_"):
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
            letter, formula = RULE_LABELS[rule]
            fig.text(0.09, 0.94, f"{letter} · {labels[metric][0]}", fontsize=18, weight="bold", va="top")
            fig.text(0.09, 0.865, f"{rule}   |   raw_keep = {formula}", fontsize=11)
            fig.text(0.09, 0.815, run, fontsize=8)
            fig.text(0.09, 0.10, "离线映射对照：同一历史候选池；不是该规则重新运行后的轨迹或实际决策。", fontsize=10)
            fig.text(0.09, 0.065, "逐 step 计算 H 和整数 k 后，按轨迹插值、等权平均；均值允许小数，不外推、不补零。", fontsize=9)
            valid = [row["n_trajectories"] for row in points]
            fig.text(0.09, 0.03, f"逐点贡献轨迹数：{min(valid, default=0)}–{max(valid, default=0)}；完整分母见 CSV。", fontsize=9)
            for filename in entry["files"]:
                path = stage / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=analysis["dpi"])
            plt.close(fig)
    return index


def gallery_index(stage, mapping_index):
    """One gallery entry per logical figure, including original analysis plots."""
    entries, used = [], set()
    for entry in mapping_index["entries"]:
        stem = str(Path(entry["files"][0]).with_suffix(""))
        used.add(stem)
        entries.append({**entry, "section": entry["retention_rule"]})
    figures = {}
    for path in sorted((stage / "plots").rglob("*")):
        if path.suffix not in {".png", ".svg"}:
            continue
        relative = path.relative_to(stage).as_posix()
        figures.setdefault(str(Path(relative).with_suffix("")), []).append(relative)
    for stem, files in figures.items():
        if stem not in used:
            entries.append({"section": "observed", "metric": Path(stem).name, "files": files})
    return {"version": mapping_index["version"], "entries": entries, "figure_count": len(entries),
            "file_count": sum(len(entry["files"]) for entry in entries)}


def report_html(stage, analysis, trajectories, rows, curves, gallery):
    version = mapping_version(analysis)
    escape = html.escape
    runs = sorted({t["source_run_id"] for t in trajectories})
    valid = sum(t["parallel_steps"] > 0 for t in trajectories)
    terminations = dict(Counter(t.get("termination_reason", "unknown") for t in trajectories))
    params = sorted({(row["minimum"], row["retention_ratio"]) for row in rows})
    nav = ''.join(f'<a href="#{rule}">{RULE_LABELS[rule][0]} · {escape(rule)}</a>' for rule in MAPPING_RULES)
    sections = []
    for section in (*MAPPING_RULES, "observed"):
        if section == "observed":
            heading = "历史实际观测与原有统计图"
            description = ("以下图片来自原始分支统计，包括 all/retained 熵、实际保留分支数和整数直方图。"
                           if version == LEGACY_VERSION else
                           "以下图片来自原始分支统计，包括 all/retained 熵、实际 winner 支持和整数直方图。")
        else:
            letter, formula = RULE_LABELS[section]
            heading = f"{letter} · {section}"
            description = (f"raw_keep = {formula}；N≥m 时 k=min(N,max(m,stable_ceil(raw_keep)))，否则 k=0。" +
                           ("H 和 N 在三种映射间相同，单独展示便于查阅。" if version == LEGACY_VERSION else
                            "winner 支持率以历史 canonical winner action 为基准，改变率表示映射后的共识 winner 与其不同。"))
        cards = []
        for entry in gallery["entries"]:
            if entry["section"] != section:
                continue
            filename = next((name for name in entry["files"] if name.endswith('.png')), entry["files"][0])
            mime = "image/png" if filename.endswith('.png') else "image/svg+xml"
            encoded = base64.b64encode((stage / filename).read_bytes()).decode("ascii")
            title = entry["metric"]
            cards.append(f'<figure><figcaption>{escape(title)}</figcaption><img loading="lazy" alt="{escape(title)}" src="data:{mime};base64,{encoded}"><p class="path">{escape(filename)}</p></figure>')
        sections.append(f'<section id="{section}"><h2>{escape(heading)}</h2><p>{escape(description)}</p><div class="grid">{"".join(cards)}</div></section>')
    checks = []
    for run in runs:
        for rule in MAPPING_RULES:
            for metric, _, _ in metric_specs_for_version(version):
                for x in (0.0, 0.5, 1.0):
                    match = next((r for r in curves if r["source_run_id"] == run and r["retention_rule"] == rule
                                  and r["metric"] == metric and math.isclose(r["x"], x)), None)
                    if match:
                        mean = "N/A" if match["mean"] is None else f'{match["mean"]:.6f}'
                        checks.append(f'<tr><td>{escape(rule)}</td><td>{escape(metric)}</td><td>{x:.0%}</td><td>{mean}</td><td>{match["n_trajectories"]}</td></tr>')
    plot_description = ("H、k、N 与 score cutoff 各自单独成图；同一映射独立分组。" if version == LEGACY_VERSION else
                        "H、k、筛选前/后 winner 支持率、winner 改变率与 score cutoff 各自单独成图；不再绘制决策候选池分支数量。")
    if version in {VERSION, PREVIOUS_VERSION}:
        for index, section in enumerate(sections):
            sections[index] = section.replace(
                "winner 支持率以历史 canonical winner action 为基准，改变率表示映射后的共识 winner 与其不同。",
                "winner 基准为筛选前候选池的共识 action；筛选前后支持率均统计该 action。改变率比较筛选前后共识 action，按正式投票及平票规则重放。")
        plot_description += "基准 winner 为筛选前候选池的共识 action；支持率为该 action 票数/对应集合分支数；改变率逐 step 为 0/1，再沿进度按轨迹等权平均。"
    if version == VERSION:
        plot_description += "新增 winner_changed_vs_fixed_ratio：每个 step 直接比较熵映射 winner 与固定比例 0.5 winner 的 action 身份；不是两条改变率曲线相减。任一 winner 缺失时该点缺失。"
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>80-task consensus · 熵与动态阈值映射</title><style>
body{{margin:0;background:#f2f5fa;color:#17243b;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1440px;margin:auto;padding:36px}}header,section{{background:white;border:1px solid #dde5ef;border-radius:16px;padding:26px;margin-bottom:24px}}h1{{font-size:30px;margin:0 0 12px}}h2{{font-size:22px}}.badge{{background:#fff2ce;padding:12px 16px;border-left:4px solid #d97706;border-radius:6px}}.meta,.path{{overflow-wrap:anywhere;color:#5b6b82}}nav{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}nav a{{background:#e8efff;color:#184ca5;border-radius:8px;padding:9px 12px;text-decoration:none}}.stats{{display:flex;flex-wrap:wrap;gap:24px;margin:20px 0}}.stats strong{{font-size:25px;display:block}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:18px}}figure{{margin:0;border:1px solid #dce5f0;border-radius:10px;overflow:hidden}}figcaption{{padding:12px 16px;background:#f4f7fc;font-weight:600}}img{{display:block;width:100%;height:auto}}.path{{font-size:11px;padding:0 14px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;border-bottom:1px solid #e2e8f0;padding:7px}}code{{overflow-wrap:anywhere}}details{{margin-top:20px}}@media print{{main{{padding:0}}section{{break-inside:avoid}}}}
</style></head><body><main><header><h1>80-task consensus · 熵与动态阈值映射</h1>
<p class="meta">分析：{escape(analysis["analysis_id"])}<br>来源：{escape(', '.join(runs))}<br>生成时间：{escape(analysis["created_at"])}</p>
<div class="badge">离线反事实映射：A/B/C 在同一组历史决策候选池上计算。没有重新运行 agent，没有改变历史 winner，也不代表三种规则各自实际运行后的轨迹或成功率。</div>
<div class="stats"><div><strong>{len(trajectories)}</strong>已选择轨迹</div><div><strong>{valid}</strong>有 parallel step 的轨迹</div><div><strong>{len(rows)//len(MAPPING_RULES):,}</strong>canonical parallel steps</div><div><strong>{gallery["figure_count"]}</strong>独立图片</div><div><strong>{gallery["file_count"]}</strong>图片文件</div></div>
<p>选择范围：以本分析冻结的 selection 和 termination_reason 筛选为准。终态计数：{escape(json.dumps(terminations,ensure_ascii=False))}。</p>
<p>参数 (m, r₀)：{escape(str(params))}，读取来源 run 的冻结共识配置。每 step 先用正式 <code>rollout.entropy.entropy_retention</code> 计算 H 和整数 k，再逐轨迹归一化到 0–100%，线性插值至 {analysis["progress_points"]} 点并按轨迹等权平均；无 parallel step 不补零，不外推，不跨显式缺失值插值。均值 k 允许小数。</p>
<p>候选池：已通过评分、最终状态为 eligible 且 action 可规范化的分支；如存在 loop escape，使用其最终定案后的池。cutoff score 为按 (score, branch_id) 排序后第 k 个分支的分数，分数越低越优；分数相同仍精确保留 k 条。</p>
<p>{plot_description}阴影表示轨迹间均值的 ±1 标准误。全部图片嵌入本 HTML，可离线单文件查看。</p>
<nav>{nav}<a href="#observed">历史实际观测</a><a href="#audit">数据与核对</a></nav></header>
{''.join(sections)}
<section id="audit"><h2>数据与核对</h2><p>逐 step 映射：<code>entropy-mappings/steps.jsonl</code> / <code>entropy-mappings/steps.csv</code><br>run 平均：<code>entropy-mappings/per-run.csv</code><br>原始 step 聚合：<code>entropy-mappings/raw-steps.csv</code><br>图清单：<code>entropy-mappings/gallery-index.json</code><br>来源与完整性：<code>selection.json</code>、<code>decisions.jsonl</code>、<code>branch-receipt.json</code>。</p>
<details><summary>展开首、中、末进度点数值及贡献轨迹数</summary><table><thead><tr><th>映射</th><th>量</th><th>进度</th><th>均值</th><th>贡献轨迹</th></tr></thead><tbody>{''.join(checks)}</tbody></table></details></section></main></body></html>'''


def write_mapping_artifacts(stage, records, trajectories, analysis):
    from action.analyzers.branches import write_csv

    stage = Path(stage)
    directory = stage / "entropy-mappings"
    directory.mkdir()
    rows = mapping_rows(records, trajectories)
    raw, normalized = mapping_curve_tables(rows, trajectories, analysis["progress_points"])
    with (directory / "steps.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(directory / "steps.csv", rows)
    write_csv(directory / "raw-steps.csv", raw)
    write_csv(directory / "per-run.csv", normalized)
    index = render_mapping_plots(stage, normalized, trajectories, analysis)
    write_json(directory / "plot-index.json", index)
    gallery = gallery_index(stage, index)
    write_json(directory / "gallery-index.json", gallery)
    (stage / "index.html").write_text(report_html(stage, analysis, trajectories, rows, normalized, gallery), encoding="utf-8")


def _check_csv(path, entries):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        stored = list(csv.DictReader(stream))
    expected = [{key: "" if value is None else json.dumps(value, ensure_ascii=False)
                 if isinstance(value, (list, dict)) else str(value) for key, value in row.items()}
                for row in entries]
    if stored != expected:
        raise ValueError("Entropy mapping CSV differs from recomputation: " + path.name)


def check_mapping_artifacts(out, root, records, trajectories, analysis, selection, receipt):
    """Rebuild inputs from immutable source decisions, then all tables and gallery."""
    from action.datasets.trajectories import read_messages

    version = mapping_version(analysis)
    configs = {run["run_id"]: read_json(out / "source-config" / (run["run_id"] + ".json"))
               for run in selection["source_runs"]}
    messages = {}
    for record in records:
        proof = record["proof"]
        if proof["kind"] == "full":
            decision = read_json(proof["path"])
        else:
            key = (record["source_run_id"], record["rollout_id"])
            if key not in messages:
                source = next(s for s in selection["sources"] if (s["source_run_id"], s["rollout_id"]) == key)
                messages[key] = read_messages(root, source)
            decision = messages[key][record["message_index"]]["extra"]["dycon_parallel"]
        expected = decision_mapping_input(decision, record, configs[record["source_run_id"]], version=version)
        if expected != record.get("entropy_mapping_input"):
            raise ValueError("Entropy mapping input differs from original source decision")
    directory = out / "entropy-mappings"
    rows = mapping_rows(records, trajectories, version=version)
    stored = [json.loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    if stored != rows:
        raise ValueError("Entropy mapping step values differ from recomputation")
    raw, normalized = mapping_curve_tables(rows, trajectories, analysis["progress_points"], version=version)
    for filename, entries in (("steps.csv", rows), ("raw-steps.csv", raw), ("per-run.csv", normalized)):
        _check_csv(directory / filename, entries)
    index = mapping_plot_index(trajectories, analysis)
    gallery = gallery_index(out, index)
    if read_json(directory / "plot-index.json") != index or read_json(directory / "gallery-index.json") != gallery:
        raise ValueError("Entropy mapping gallery coverage differs from frozen selection")
    for entry in gallery["entries"]:
        for filename in entry["files"]:
            if filename not in receipt["files"] or not owned_path(out, filename).is_file():
                raise ValueError("Missing entropy gallery figure: " + filename)
    expected_html = report_html(out, analysis, trajectories, rows, normalized, gallery)
    if (out / "index.html").read_text(encoding="utf-8") != expected_html:
        raise ValueError("Entropy mapping HTML differs from frozen figure/data content")
    if hash_json(analysis["entropy_mapping_comparison"]) != hash_json({
            "version": version, "rules": list(MAPPING_RULES), "interpretation": "counterfactual_on_observed_pool"}):
        raise ValueError("Entropy mapping comparison settings differ")
