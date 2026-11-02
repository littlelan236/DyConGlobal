"""LDA panels and manifold scatters consume frozen coordinates, never fit models."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

FORMATS = {"png", "svg", "pdf"}


def display_selection(n, seed=0, maximum=None):
    if maximum is not None and maximum < 1:
        raise ValueError("Display point cap must be positive")
    rng = np.random.default_rng(seed)
    indices = (
        np.arange(n) if maximum is None or maximum >= n else np.sort(rng.choice(n, maximum, replace=False))
    )
    displayed = np.zeros(n, dtype=bool)
    displayed[indices] = True
    jitter = np.random.default_rng(seed).uniform(-0.12, 0.12, n)
    return indices, displayed, jitter


def render_site(
    output,
    site,
    coordinates,
    rows,
    *,
    formats=("png", "svg"),
    displayed=None,
    jitter=None,
    facet=None,
    method="lda",
):
    if method != "lda":
        return render_embedding(
            output, site, coordinates, rows, method=method, formats=formats, displayed=displayed, facet=facet
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not set(formats) <= FORMATS or not formats:
        raise ValueError("Formats must be png, svg or pdf")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x = np.asarray(coordinates).reshape(-1)
    labels = np.array([row["y"] for row in rows])
    displayed = np.ones(len(rows), dtype=bool) if displayed is None else displayed
    jitter = np.zeros(len(rows)) if jitter is None else jitter
    bins = np.histogram_bin_edges(x, bins=40)
    groups = [("all samples", np.ones(len(rows), dtype=bool))]
    if facet is not None:
        if facet not in {"source_run_id", "retained"}:
            raise ValueError("Unsupported plot facet")
        keys = sorted({str(row[facet]) for row in rows})
        groups += [(f"{facet}={key}", np.array([str(row[facet]) == key for row in rows])) for key in keys]
    paths = []
    for plot_type in ("distribution", "strip"):
        fig, axes = plt.subplots(len(groups), 1, figsize=(9, max(3.4, 3.2 * len(groups))), squeeze=False)
        for ax, (group_name, group) in zip(axes[:, 0], groups):
            for y, name, color in ((0, "negative", "#4472c4"), (1, "positive", "#c00000")):
                mask = group & (labels == y)
                if plot_type == "distribution":
                    ax.hist(
                        x[mask],
                        bins=bins,
                        alpha=0.55,
                        color=color,
                        label=f"{name}: n={int(mask.sum())}; fit prior=0.5",
                    )
                else:
                    mask &= displayed
                    ax.scatter(
                        x[mask],
                        y + jitter[mask],
                        s=6,
                        alpha=0.38,
                        color=color,
                        label=f"{name}: shown={int(mask.sum())}",
                    )
            ax.set_xlabel("LD1 (one supervised discriminant dimension)")
            ax.set_title(f"Site {site} | {group_name}")
            if plot_type == "distribution":
                ax.set_ylabel("Sample count (all coordinates)")
            else:
                ax.set_yticks([0, 1], ["negative", "positive"])
                ax.set_ylabel("Class + display jitter\n(not a latent dimension)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        for fmt in formats:
            path = output / f"site-{site}-{plot_type}.{fmt}"
            fig.savefig(path, dpi=160)
            paths.append(path)
        plt.close(fig)
    return paths


def render_embedding(output, site, coordinates, rows, *, method, formats, displayed=None, facet=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if method not in {"tsne", "umap"} or not formats or not set(formats) <= FORMATS:
        raise ValueError("Embedding plot requires tsne/umap and png, svg or pdf formats")
    if facet not in {None, "source_run_id", "retained"}:
        raise ValueError("Unsupported plot facet")
    xy = np.asarray(coordinates)
    if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
        raise ValueError("Embedding scatter requires finite 2-D coordinates aligned with metadata")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = np.array([row["y"] for row in rows])
    displayed = np.ones(len(rows), dtype=bool) if displayed is None else displayed
    groups = [("all samples", np.ones(len(rows), dtype=bool))]
    if facet is not None:
        for key in sorted({str(row[facet]) for row in rows}):
            groups.append((f"{facet}={key}", np.array([str(row[facet]) == key for row in rows])))
    title = "t-SNE" if method == "tsne" else "UMAP"
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, max(4.5, 4.2 * len(groups))), squeeze=False)
    for ax, (name, group) in zip(axes[:, 0], groups):
        for y, label, color in ((0, "negative", "#4472c4"), (1, "positive", "#c00000")):
            total = group & (labels == y)
            mask = total & displayed
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=7,
                alpha=0.4,
                color=color,
                linewidths=0,
                rasterized=True,
                label=f"{label}: shown={int(mask.sum())}, total={int(total.sum())}",
            )
        ax.set(
            title=f"Site {site} | {title} | {name} (labels only color points)",
            xlabel=f"{title} 1 (arbitrary embedding coordinate)",
            ylabel=f"{title} 2 (arbitrary embedding coordinate)",
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths = []
    for fmt in formats:
        path = output / f"site-{site}-{method}-scatter.{fmt}"
        fig.savefig(path, dpi=160)
        paths.append(path)
    plt.close(fig)
    return paths


def write_report(directory, analysis, summary, plots):
    directory = Path(directory)
    warning = (
        "Exploratory supervised visualization. Same-data separation is not evidence of generalization, "
        "action correctness or steering efficacy. Related branches are not independent generalization samples. "
        "Each site has its own coordinate system; LD1 is not a steering payload."
    )
    method = analysis["method"]
    title = "Consensus branch LDA"
    explanation = (
        "Binary LDA has one discriminant axis. Strip vertical coordinates encode class plus display jitter."
    )
    if method != "lda":
        title = "Consensus branch " + ("t-SNE" if method == "tsne" else "UMAP")
        warning = (
            "Exploratory unsupervised full-cohort visualization; labels only color points and never enter fitting. "
            "Apparent separation is not evidence of generalization, action correctness or steering efficacy. "
            "Related branches are not independent samples. Axes, orientation and distances are not calibrated "
            "across methods, sites or seeds. No out-of-sample reducer is exported."
        )
        explanation = (
            "Both axes are embedding coordinates, without display jitter. All frozen samples enter fitting "
            "and coordinate export; a display cap affects plots only. transform/*.npz contains only an audit "
            "descriptor and optional PCA arrays, not a reusable t-SNE/UMAP model. Redraw never refits."
        )
    lines = [
        f"# {title}",
        "",
        warning,
        "",
        f"Fit scope: {analysis['fit_scope']}. Samples: {analysis['sample_count']}.",
        explanation,
        "",
        "| Site | Status | Negative | Positive |",
        "|---|---|---:|---:|",
    ]
    for site, stats in summary["sites"].items():
        counts = stats.get("class_counts", [0, 0])
        lines.append(f"| {site} | {stats['status']} | {counts[0]} | {counts[1]} |")
    lines += ["", "[Complete sample coordinates](points.csv)", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    images = []
    for path in plots:
        relative = Path(path).relative_to(directory).as_posix()
        if Path(path).suffix in {".png", ".svg"}:
            images.append(
                f'<img style="max-width:100%" src="{html.escape(relative)}" alt="{html.escape(Path(path).stem)}">'
            )
        else:
            images.append(f'<p><a href="{html.escape(relative)}">{html.escape(Path(path).name)}</a></p>')
    markup = (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f'<main style="max-width:1100px;margin:auto;font-family:sans-serif"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(warning)}</p><p>Fit scope: {html.escape(analysis['fit_scope'])}; "
        f'samples: {analysis["sample_count"]}</p><p>{html.escape(explanation)}</p><p><a href="points.csv">Complete coordinates</a> · '
        '<a href="summary.json">Summary</a> · <a href="report.md">Report</a></p>'
        + "".join(images)
        + "</main>\n"
    )
    (directory / "index.html").write_text(markup, encoding="utf-8")
