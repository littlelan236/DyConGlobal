"""Plot token-aligned remaining-work labels and optional event predictions."""

from collections import defaultdict
from pathlib import Path

from core.io import hash_json, read_json, write_json


def visualize_token_positions(index_dir, output, predictions=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_json(Path(index_dir) / "index.json")
    rows = data["samples"]
    prediction = {} if predictions is None else {row["sample_id"]: row["prediction"] for row in read_json(predictions)}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["rollout_id"]].append(row)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    products = []
    for rollout_id, events in grouped.items():
        events.sort(key=lambda row: (row["step_id"], row["absolute_position"]))
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        x = list(range(len(events)))
        axes[0].plot(x, [row["remaining_reasoning_tokens_after"] for row in events], "o-", label="Observed remaining reasoning tokens")
        selected = [(i, prediction[row["sample_id"]]) for i, row in enumerate(events) if row["sample_id"] in prediction]
        if selected:
            axes[0].plot([item[0] for item in selected], [item[1] for item in selected], "x--", label="Prediction")
        axes[0].legend()
        axes[1].plot(x, [row["remaining_steps"] for row in events], "o-", label="Remaining steps")
        axes[1].set_xlabel("Exact token-aligned event order")
        axes[1].legend()
        fig.suptitle(rollout_id)
        fig.tight_layout()
        name = hash_json(rollout_id)[:24] + ".png"
        fig.savefig(out / name)
        plt.close(fig)
        products.append({"rollout_id": rollout_id, "path": name, "events": len(events)})
    write_json(out / "index.json", {"prediction_supplied": predictions is not None, "plots": products})
    return out
