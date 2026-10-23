"""Render frozen descriptive action matrices without reading trajectories or models."""

from importlib.metadata import version
from pathlib import Path

from core.io import sha256, write_json

PLOT_NAMES = ("progress-share", "transition-probability")


def render_plots(output, summary, *, analysis_id, labels_sha256, summary_sha256,
                 names=PLOT_NAMES, aggregation="micro", formats=("png",), dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names, formats = tuple(names), tuple(formats)
    if not names or len(names) != len(set(names)) or set(names) - set(PLOT_NAMES):
        raise ValueError("Unknown or repeated action plot")
    if aggregation not in {"micro", "macro"} or type(dpi) is not int or not 50 <= dpi <= 600:
        raise ValueError("Invalid plot aggregation or DPI (50-600)")
    if not formats or len(formats) != len(set(formats)) or set(formats) - {"png", "svg"}:
        raise ValueError("Plot formats must be unique png/svg values")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files, specs = {}, {}
    style = {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "dycon-action-v2"}
    with plt.rc_context(style):
        for name in names:
            phase = name == "progress-share"
            values = summary["phase"][aggregation] if phase else summary["transition"]["probabilities"]
            data = np.ma.masked_invalid(np.asarray(values, dtype=float))
            labels = summary["phase" if phase else "transition"]["labels"]
            xlabels = summary["phase"]["bins"] if phase else labels
            title = f"Action share by trajectory progress ({aggregation})" if phase else "Next-action probability given current action"
            figsize = (9, 5.5) if phase else (9, 7)
            fig, ax = plt.subplots(figsize=figsize)
            try:
                cmap = plt.get_cmap("viridis").with_extremes(bad="#E2E3E5")
                plot = ax.imshow(data, vmin=0, vmax=1, cmap=cmap, aspect="auto")
                ax.set_xticks(range(len(xlabels)), xlabels, rotation=0 if phase else 35, ha="center" if phase else "right")
                ax.set_yticks(range(len(labels)), labels)
                ax.set_xlabel("Relative assistant-step progress" if phase else "Next action")
                ax.set_ylabel("Action category" if phase else "Current action")
                ax.set_title(title)
                for i, line in enumerate(values):
                    for j, value in enumerate(line):
                        ax.text(j, i, "N/A" if value is None else f"{value:.0%}", ha="center", va="center",
                                color="#222222" if value is None or value > .6 else "white", fontsize=8)
                bar = fig.colorbar(plot, ax=ax, pad=.02)
                bar.set_label("Fraction (0-1)")
                fig.tight_layout()
                for fmt in formats:
                    filename = f"{name}.{fmt}"
                    fig.savefig(output / filename, dpi=dpi, metadata={"Date": None} if fmt == "svg" else None)
                    files[filename] = sha256(output / filename)
            finally:
                plt.close(fig)
            specs[name] = {"figsize": list(figsize), "xlabels": xlabels, "ylabels": labels,
                           "matrix_schema": summary["phase" if phase else "transition"]["schema"],
                           "aggregation": aggregation if phase else "row-conditional", "empty": "masked/N/A"}
    manifest = {"schema_version": 1, "analysis_id": analysis_id,
                "input": {"labels_sha256": labels_sha256, "summary_sha256": summary_sha256},
                "settings": {"names": list(names), "aggregation": aggregation, "formats": list(formats),
                             "dpi": dpi, "colormap": "viridis", "vmin": 0, "vmax": 1, "backend": "Agg",
                             "font": "DejaVu Sans"},
                "packages": {name: version(name) for name in ("matplotlib", "numpy")}, "plots": specs, "files": files}
    write_json(output / "plots.json", manifest)
    return manifest
