"""Trajectory plots are generated only from explicit prediction tables."""

from pathlib import Path


def plot_trajectory(rows, output, *, title="remaining workload", ylabel="remaining workload",
                    comparison=False, xlim=None, ylim=None, logylim=None, note="", truth_label="True remaining"):
    """Plot explicit predictions and truth; None creates gaps, never a zero label.

    comparison=True adds native log1p scores and log1p truth in a second panel.
    It also renders an explicit unavailable panel when no qualified rows exist.
    Existing evaluation callers retain their single-panel behavior.
    """
    if not rows:
        raise ValueError("No trajectory rows")
    import math
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    def value(row, key):
        v = row.get(key)
        if v is None:
            return float('nan')
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError('Curve values must be finite numbers or None')
        return v

    x = [r['step_id'] for r in rows]
    if any(type(v) is not int or v < 0 for v in x) or x != sorted(set(x)):
        raise ValueError('Curve step IDs must be unique sorted nonnegative integers')
    # Materialize missing integer steps so matplotlib cannot bridge absent samples.
    indexed = {r['step_id']: r for r in rows}
    x = list(range(x[0], x[-1] + 1))
    rows = [indexed.get(step, {'step_id': step}) for step in x]
    true = [value(r, 'target') for r in rows]
    pred = [value(r, 'prediction') for r in rows]
    fig = Figure(figsize=(8, 6.5 if comparison else 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2 if comparison else 1, 1, squeeze=False).ravel()
    def panel(ax, y, p, label):
        ax.plot(x, y, color='#15803d', linewidth=1.7, label=truth_label, marker='.', markersize=3)
        ax.plot(x, p, color='#2563eb', linewidth=1.2, label='Predicted remaining', marker='.', markersize=3)
        ax.set(xlabel='Canonical assistant step (0-based)', ylabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.18)
        if xlim is not None:
            ax.set_xlim(xlim)
        if not any(math.isfinite(v) for v in y + p):
            ax.text(.5, .5, 'No qualified native hidden / labels', transform=ax.transAxes,
                    ha='center', va='center', color='#9a3412')
        ax.legend(fontsize=8)
    panel(axes[0], true, pred, ylabel + ' (raw units)')
    axes[0].set_title(title, fontsize=10)
    if ylim is not None:
        low, high = ylim
        pad = max(high-low, 1) * .05
        axes[0].set_ylim(low-pad, high+pad)
    if comparison:
        truth_log = [math.log1p(v) if math.isfinite(v) and v >= 0 else float('nan') for v in true]
        pred_log = [value(r, 'prediction_log1p') for r in rows]
        panel(axes[1], truth_log, pred_log, 'log1p(' + ylabel + ')')
        if logylim is not None:
            low, high = logylim
            pad = max(high-low, 1) * .05
            axes[1].set_ylim(low-pad, high+pad)
    if note:
        fig.text(.08, .015, note, fontsize=8, color='#9a3412')
    fig.tight_layout(rect=(0, .035 if note else 0, 1, 1))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)


def plot_parallel_scores(steps, branches, output, *, target, title, subtitle):
    """Plot native stored scores; missing selected values break the line."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(13, 5.4))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    orange, blue = "#d97706", "#2563eb"
    valid = [r for r in branches if r["score"] is not None and r["status"] == "eligible"]
    other = [r for r in branches if r["score"] is not None and r["status"] != "eligible"]
    if valid:
        ax.scatter([r["step_id"] for r in valid], [r["score"] for r in valid],
                   color=orange, alpha=.48, s=27, linewidths=0, label="All eligible branches", zorder=2)
    if other:
        ax.scatter([r["step_id"] for r in other], [r["score"] for r in other],
                   color=orange, alpha=.8, marker="x", s=34, label="Scored ineligible / unscored-valid branches", zorder=3)
    ax.plot([r["step_id"] for r in steps],
            [r["selected_score"] if r["selected_score"] is not None else float("nan") for r in steps],
            color=blue, linewidth=1.6, marker="o", markersize=3.4,
            label="Selected branch on actual trajectory", zorder=4)
    missing = [r for r in branches if r["score"] is None]
    if missing:
        ax.scatter([r["step_id"] for r in missing], [.025] * len(missing),
                   transform=ax.get_xaxis_transform(), color=orange, marker="|", s=65,
                   label="Missing branch score (axis marker)", zorder=5)
    gaps = [r for r in steps if r["selected_score"] is None]
    if gaps:
        ax.scatter([r["step_id"] for r in gaps], [.065] * len(gaps),
                   transform=ax.get_xaxis_transform(), color=blue, marker="x", s=32,
                   label="Selected score unavailable (line gap)", zorder=5)
    if target.endswith("_log1p"):
        ylabel = "Remaining workload estimate (log1p; native score)"
    elif target.endswith("steps_linear"):
        ylabel = "Remaining steps estimate (native score)"
    else:
        ylabel = "Remaining workload estimate (native score)"
    ax.set(xlabel="Actual agent step (0-based)", ylabel=ylabel,
           title=f"{title}\n{target}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(steps[0]["step_id"] - .6, steps[-1]["step_id"] + .6)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8, framealpha=.94)
    fig.text(.07, .035, subtitle, fontsize=8, color="#475569")
    fig.text(.07, .012, "Stored predictions; branch spread is not a ground-truth error bar. No smoothing or score clipping.",
             fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"trajectory.{extension}", dpi=180)
