"""Trajectory plots are generated only from explicit prediction tables."""

from pathlib import Path


def plot_trajectory(rows, output, *, title="remaining workload"):
    if not rows:
        raise ValueError("No trajectory rows")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the analysis extra") from exc
    x = [r["step_id"] for r in rows]
    true = [r["target"] for r in rows]
    pred = [r["prediction"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, true, label="true")
    ax.plot(x, pred, label="predicted")
    ax.set(xlabel="agent step", ylabel="remaining workload", title=title)
    ax.legend()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


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
