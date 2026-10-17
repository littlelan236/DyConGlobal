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
