"""Auditable equal-trajectory raw-step and normalized-progress action curves."""

import bisect
import math
import statistics
from collections import defaultdict
from pathlib import Path

from core.io import identifier, write_json

METRICS = (
    ("entropy", "Shannon 熵 H（nats）"),
    ("normalized_entropy", "归一化熵 H / ln(K)"),
    ("effective_action_count", "有效 action 数 / Perplexity：exp(H)"),
    ("winner_support", "实际 winner 支持率 q"),
    ("winner_uncertainty", "Winner 不确定性 U = 1-q"),
    ("winner_surprisal", "Winner 惊奇度 I = -ln(q)"),
    ("action_types", "Action 种类数 K"),
    ("winner_count", "实际 winner 支持分支数"),
    ("valid_actions", "有效 action 分支数 N"),
    ("delta_entropy", "筛选影响 ΔH（retained − all）"),
    ("delta_normalized_entropy", "筛选影响 Δ归一化熵"),
    ("delta_effective_action_count", "筛选影响 ΔE"),
    ("delta_winner_support", "筛选影响 Δq"),
    ("delta_winner_uncertainty", "筛选影响 ΔU"),
    ("delta_winner_surprisal", "筛选影响 ΔI"),
    ("branches", "分支数量（all：生成分支；retained：已记录筛选保留分支）"),
)

TRAJECTORY_METRICS = ("entropy", "normalized_entropy", "action_types",
                      "effective_action_count", "winner_uncertainty")


def resolve_line_metrics(rows, mode, requested=(), *, include_branch_counts=False):
    """Select chart panels without changing computed statistics."""
    if isinstance(requested, str):
        requested = [requested]
    available = {metric for row in rows for _, metric, _ in
                 metric_columns(row, include_branch_counts=include_branch_counts)}
    selected = list(dict.fromkeys(requested)) if requested else (
        list(TRAJECTORY_METRICS) if mode == "per-trajectory" else [m for m, _ in METRICS if m in available])
    missing = set(selected) - available
    if missing:
        raise ValueError("Line metrics unavailable for selected branch scope: " + ", ".join(sorted(missing)))
    return selected


def build_trajectory_curves(rows, *, include_branch_counts=False):
    """Preserve every observation's run/rollout identity and original step value."""
    result, seen = [], set()
    for row in sorted(rows, key=lambda r: (r["source_run_id"], r["rollout_id"], r["assistant_index"])):
        key = (row["source_run_id"], row["rollout_id"], row["assistant_index"])
        if key in seen:
            raise ValueError("Duplicate trajectory assistant step in per-trajectory curves")
        seen.add(key)
        for column, metric, scope in metric_columns(row, include_branch_counts=include_branch_counts):
            result.append({"source_run_id": row["source_run_id"], "rollout_id": row["rollout_id"],
                           "instance_id": row.get("instance_id"), "metric": metric, "scope": scope,
                           "axis": "assistant_step", "x": row["assistant_index"] + 1,
                           "value": row[column]})
    return result


def trajectory_plot_index(trajectories, analysis):
    """List one figure per canonical trajectory, including explicit no-data figures."""
    entries = []
    for source in sorted(trajectories, key=lambda t: (t["source_run_id"], t["rollout_id"])):
        run, rid = identifier(source["source_run_id"]), identifier(source["rollout_id"])
        entries.append({"source_run_id": run, "rollout_id": rid, "instance_id": source["instance_id"],
                        "trajectory_sha256": source["trajectory_sha256"],
                        "parallel_steps": source["parallel_steps"],
                        "status": "plotted" if source["parallel_steps"] else "no_parallel_steps",
                        "metrics": analysis["line_metrics"], "axis": "assistant_step",
                        "files": ([f"plots/per-trajectory/{run}/{rid}/{metric}.{ext}"
                                   for metric in analysis["line_metrics"] for ext in analysis["formats"]]
                                  if analysis.get("plot_layout") == "separate" else
                                  [f"plots/per-trajectory/{run}/{rid}.{ext}" for ext in analysis["formats"]])})
    if len({(e["source_run_id"], e["rollout_id"]) for e in entries}) != len(entries):
        raise ValueError("Duplicate trajectory plot identity")
    return {"schema_version": 1, "unit": "source_run_id + rollout_id; never grouped by task",
            "trajectory_count": len(entries), "entries": entries}


def separate_plot_index(trajectories, analysis):
    """Freeze the complete metric-by-unit file set for one batch invocation."""
    runs = sorted({t["source_run_id"] for t in trajectories})
    mode = analysis["line_mode"]
    units = []
    if mode == "per-trajectory":
        for source in sorted(trajectories, key=lambda t: (t["source_run_id"], t["rollout_id"])):
            run, rid = identifier(source["source_run_id"]), identifier(source["rollout_id"])
            units.append(("per-trajectory", [run], rid,
                          "plotted" if source["parallel_steps"] else "no_parallel_steps",
                          f"plots/per-trajectory/{run}/{rid}"))
    if mode in {"per-run", "both"}:
        for run in runs:
            status = "plotted" if any(t["parallel_steps"] for t in trajectories
                                     if t["source_run_id"] == run) else "no_parallel_steps"
            units.append(("per-run", [run], None, status, f"plots/per-run/{identifier(run)}"))
    if mode in {"all-runs", "both"}:
        units.append(("all-runs", runs, None, "plotted", "plots/all-runs-normalized"))
    entries = [{"mode": unit, "source_run_ids": selected_runs, "rollout_id": rid,
                "metric": metric, "status": status,
                "axis": "assistant_step" if unit == "per-trajectory" else "normalized_progress",
                "files": [f"{prefix}/{identifier(metric)}.{ext}" for ext in analysis["formats"]]}
               for unit, selected_runs, rid, status, prefix in units for metric in analysis["line_metrics"]]
    return {"schema_version": 1, "plot_layout": "one_metric_per_figure", "entries": entries,
            "figure_count": len(entries), "file_count": sum(len(e["files"]) for e in entries)}


def metric_columns(row, *, include_branch_counts=False):
    columns = []
    for metric, _ in METRICS:
        if metric == "branches" and not include_branch_counts:
            continue
        if metric.startswith("delta_"):
            if metric in row:
                columns.append((metric, metric, "delta"))
        else:
            for scope in ("all", "retained"):
                key = f"{metric}_{scope}"
                if key in row:
                    columns.append((key, metric, scope))
    return columns


def _finite(value):
    return value is not None and math.isfinite(value)


def _aggregate(run, metric, scope, axis, x, values, total):
    count = len(values)
    deviation = statistics.stdev(values) if count > 1 else None
    return {"source_run_id": run, "metric": metric, "scope": scope, "axis": axis, "x": x,
            "mean": statistics.mean(values) if count else None,
            "n_trajectories": count, "total_trajectories": total,
            "sample_std": deviation, "standard_error": deviation / math.sqrt(count) if deviation is not None else None}


def interpolate_at(points, x):
    """Linear interpolation only inside adjacent valid observations, no extrapolation."""
    positions = [p[0] for p in points]
    i = bisect.bisect_left(positions, x)
    if i < len(points) and math.isclose(positions[i], x, rel_tol=0, abs_tol=1e-12):
        return points[i][1] if _finite(points[i][1]) else None
    if i > 0 and math.isclose(positions[i - 1], x, rel_tol=0, abs_tol=1e-12):
        return points[i - 1][1] if _finite(points[i - 1][1]) else None
    if i == 0 or i == len(points):
        return None
    x0, y0 = points[i - 1]
    x1, y1 = points[i]
    if not _finite(y0) or not _finite(y1):
        return None
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def build_curve_tables(rows, trajectories, progress_points=101, *, include_branch_counts=False,
                       metric_specs=None):
    if type(progress_points) is not int or not 2 <= progress_points <= 1001:
        raise ValueError("progress-points must be an integer in [2, 1001]")
    if not rows:
        return [], []
    columns = list(metric_specs) if metric_specs is not None else (list(dict.fromkeys(column for row in rows for column in
                                 metric_columns(row, include_branch_counts=True)))
               if include_branch_counts else metric_columns(rows[0]))
    totals = defaultdict(int)
    metadata = {}
    for t in trajectories:
        key = (t["source_run_id"], t["rollout_id"])
        if key in metadata:
            raise ValueError("Duplicate trajectory metadata")
        metadata[key] = t
        totals[key[0]] += 1
    groups = defaultdict(list)
    for row in rows:
        key = (row["source_run_id"], row["rollout_id"])
        if key not in metadata:
            raise ValueError("Curve row has no trajectory metadata")
        groups[key].append(row)
    for key, values in groups.items():
        values.sort(key=lambda r: r["assistant_index"])
        if len({r["assistant_index"] for r in values}) != len(values):
            raise ValueError("Duplicate trajectory assistant step in curves")
        length = metadata[key]["assistant_steps"]
        for r in values:
            expected = r["assistant_index"] / (length - 1) if length > 1 else 0.0
            if not 0 <= r["assistant_index"] < length or not math.isclose(r["normalized_progress"], expected):
                raise ValueError("Progress does not match canonical trajectory length")
    raw = []
    normalized = []
    for run in sorted(totals):
        run_groups = [g for (source, _), g in groups.items() if source == run]
        raw_indices = sorted({r["assistant_index"] + 1 for g in run_groups for r in g})
        for key, metric, scope in columns:
            raw_values = defaultdict(list)
            for group in run_groups:
                for row in group:
                    if _finite(row.get(key)):
                        raw_values[row["assistant_index"] + 1].append(row[key])
            for x in raw_indices:
                raw.append(_aggregate(run, metric, scope, "assistant_step", x, raw_values[x], totals[run]))
            points_by_trajectory = [[(r["normalized_progress"], r.get(key)) for r in group] for group in run_groups]
            for index in range(progress_points):
                x = index / (progress_points - 1)
                values = [interpolate_at(points, x) for points in points_by_trajectory]
                values = [v for v in values if _finite(v)]
                normalized.append(_aggregate(run, metric, scope, "normalized_progress", x, values, totals[run]))
    return raw, normalized


def render_line_plots(directory, raw_curves, normalized_curves, analysis,
                      trajectory_curves=(), trajectories=()):
    from action.evaluation.branch_histograms import plot_style

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    mode = analysis["line_mode"]
    if mode == "none":
        return
    directory = Path(directory)
    runs = sorted({r["source_run_id"] for r in raw_curves} | {t["source_run_id"] for t in trajectories})
    available = {r["metric"] for r in raw_curves}
    selected_metrics = analysis.get("line_metrics") or (
        TRAJECTORY_METRICS if mode == "per-trajectory" else [m for m, _ in METRICS if m in available])
    labels = dict(METRICS)
    panels = [(m, labels[m]) for m in selected_metrics if m in available]
    if not panels:
        raise ValueError("No line metrics available for plotting")

    def render(table, filename, selected_runs, normalized, trajectory=None, one_metric=None):
        selected_panels = [p for p in panels if one_metric is None or p[0] == one_metric]
        count = len(selected_panels)
        ncols = min(3, count)
        nrows = math.ceil(count / ncols)
        width = 10 if ncols == 1 else 12 if ncols == 2 else 16
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, 3.2 * nrows + 2.2), dpi=analysis["dpi"], squeeze=False)
        top = 0.75 if trajectory is not None and nrows == 1 else 0.80 if nrows <= 2 else 0.86
        fig.subplots_adjust(left=0.065, right=0.98, top=top,
                            bottom=0.17 if nrows <= 2 else 0.10, wspace=0.25, hspace=0.50)
        for ax, (metric, label) in zip(axes.flat, selected_panels):
            subset = [r for r in table if r["metric"] == metric and r["source_run_id"] in selected_runs]
            for ri, run in enumerate(selected_runs):
                for scope in ("all", "retained", "delta"):
                    points = sorted((r for r in subset if r["source_run_id"] == run and r["scope"] == scope),
                                    key=lambda r: r["x"])
                    if not points:
                        continue
                    color = plt.get_cmap("tab20")((2 * ri + (scope == "retained")) % 20)
                    scope_name = {"all": "全部分支", "retained": "筛选后", "delta": "筛选差值"}[scope]
                    name = f"R{runs.index(run) + 1} · {scope_name}" if len(selected_runs) > 1 else scope_name
                    ys = [r.get("value", r.get("mean")) for r in points]
                    ys = [y if y is not None else float("nan") for y in ys]
                    ax.plot([r["x"] for r in points], ys, label=name, color=color,
                            linestyle="--" if scope == "retained" else "-", linewidth=1.6,
                            marker="." if len(points) <= 2 else None)
            if metric.startswith("delta_"):
                ax.axhline(0, linewidth=0.65, color="#777777", alpha=0.5)
            if metric in {"normalized_entropy", "winner_support", "winner_uncertainty"}:
                ax.set_ylim(-0.03, 1.03)
            ax.set_title(label, fontsize=11, loc="left")
            ax.set_xlabel("归一化轨迹进度" if normalized else "Canonical assistant step（从 1 开始）", fontsize=9)
            if normalized:
                ax.set_xlim(0, 1)
                ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric == selected_panels[0][0] or metric == "delta_entropy":
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=8, frameon=False)
        for ax in list(axes.flat)[count:]:
            ax.set_visible(False)
        title = ("单轨迹统计曲线：原始 step" if trajectory is not None else
                 "单 run 统计曲线：轨迹长度归一化" if len(selected_runs) == 1 else
                 "跨 run 统计曲线：轨迹长度归一化")
        fig.text(0.065, 0.98, title, fontsize=19, weight="bold", color="#182B45", va="top")
        names = "\n".join(f"R{runs.index(run) + 1}: {run}" for run in selected_runs)
        if trajectory is not None:
            names += "\nRollout: " + trajectory["rollout_id"] + "\nInstance: " + trajectory["instance_id"]
        fig.text(0.065, 0.925, names, fontsize=8, color="#4e5968", va="top")
        note = ("本图仅含这一条 rollout 的原始 step；不跨轨迹平均，不做长度归一化。"
                if trajectory is not None else
                "每条轨迹先线性插值至统一 0–100% 网格，再在每个 run 内按轨迹等权平均；不外推，不跨缺失值插值。")
        fig.text(0.065, 0.045, note, fontsize=9, color="#596474")
        footer = ("原始指标值见 curves/per-trajectory.csv；缺失值不补零。H / ln(K) 衡量均匀度，exp(H) 为 Perplexity。"
                  if trajectory is not None else
                  "q 和 action 分布以有效 action 分支数 N 为分母；H / ln(K) 衡量均匀度。贡献轨迹数、标准差与标准误见 curves/*.csv。")
        fig.text(0.065, 0.025, footer,
                 fontsize=9, color="#596474")
        filename.parent.mkdir(parents=True, exist_ok=True)
        if trajectory is not None and not trajectory["parallel_steps"]:
            for ax in axes.flat:
                if ax.get_visible():
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "该轨迹无 parallel decision，指标不可用", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, color="#777777")
        for extension in analysis["formats"]:
            fig.savefig(str(filename) + "." + extension, dpi=analysis["dpi"])
        plt.close(fig)

    with plt.rc_context(plot_style()):
        if analysis.get("plot_layout") == "separate":
            index = separate_plot_index(trajectories, analysis)
            sources = {(t["source_run_id"], t["rollout_id"]): t for t in trajectories}
            for entry in index["entries"]:
                source = None
                table = normalized_curves
                if entry["mode"] == "per-trajectory":
                    key = (entry["source_run_ids"][0], entry["rollout_id"])
                    source = sources[key]
                    table = [r for r in trajectory_curves
                             if (r["source_run_id"], r["rollout_id"]) == key]
                filename = directory.parent / entry["files"][0].rsplit(".", 1)[0]
                render(table, filename, entry["source_run_ids"], source is None,
                       source, one_metric=entry["metric"])
            write_json(directory / "plot-index.json", index)
            if mode == "per-trajectory":
                write_json(directory / "trajectory-index.json", trajectory_plot_index(trajectories, analysis))
            return
        if mode == "per-trajectory":
            grouped = defaultdict(list)
            for row in trajectory_curves:
                grouped[(row["source_run_id"], row["rollout_id"])].append(row)
            index = trajectory_plot_index(trajectories, analysis)
            for source in trajectories:
                run, rid = source["source_run_id"], source["rollout_id"]
                render(grouped.get((run, rid), []),
                       directory / "per-trajectory" / identifier(run) / identifier(rid), [run], False, source)
            write_json(directory / "trajectory-index.json", index)
            return
        if mode in {"per-run", "both"}:
            for run in runs:
                render(normalized_curves, directory / "per-run" / run, [run], True)
        if mode in {"all-runs", "both"}:
            render(normalized_curves, directory / "all-runs-normalized", runs, True)
