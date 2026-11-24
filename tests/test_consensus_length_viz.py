import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
import pytest

from core.io import artifact_dir
from difficulty.evaluation.consensus_length_viz import (
    build_plot_data, heatmap_matrix, plot_consensus_length, rank_normalized, weighted_ecdf,
)


def group(task, rollout, decision, lengths=(10, 20), scores=(1, 2), status="comparable"):
    binding = {"task_key": task, "rollout_id": rollout, "session_id": "session", "decision_id": decision}
    best = min(range(len(scores)), key=lambda i: (scores[i], i))
    step = {**binding, "status": status, "winner_count": str(len(lengths)),
            "token_excess": str(lengths[best] - min(lengths))}
    members = [{**binding, "branch_id": str(i), "completion_tokens": str(x), "score": str(y)}
               for i, (x, y) in enumerate(zip(lengths, scores))]
    return step, members


def test_average_ties_and_constant_ranks():
    assert rank_normalized([]) == []
    assert rank_normalized([10]) == [0]
    assert rank_normalized([30, 10, 20]) == [1, 0, .5]
    assert rank_normalized([10, 10, 30]) == [.25, .25, 1]
    assert rank_normalized([10, 10, 10]) == [.5, .5, .5]


def test_task_hierarchy_weights_and_zero_mass():
    groups = [group("a", "r1", "d1"), group("a", "r1", "d2"), group("a", "r2", "d3"),
              group("b", "r3", "d1", scores=(2, 1)),  # Same decision ID in another trajectory.
              group("c", "r4", "d4", lengths=(10,), scores=(1,), status="singleton"),
              group("c", "r4", "d5", status="missing_length")]
    data = build_plot_data([s for s, _ in groups], [b for _, rows in groups for b in rows])
    weights = defaultdict(float)
    for p in data["rank_points"]:
        weights[p["task_key"]] += p["weight"]
    assert weights == {"a": .5, "b": .5}
    assert data["steps"] == 4 and data["branches"] == 8
    assert weighted_ecdf(data["excess"]) == [{"token_excess": 0, "cdf": .5}, {"token_excess": 10, "cdf": 1}]
    matrix = heatmap_matrix(data["rank_points"])
    assert matrix.sum() == pytest.approx(1)
    assert matrix[0, 0] == pytest.approx(.25)
    assert matrix[-1, 0] == pytest.approx(.25)


def test_complete_group_cannot_drop_a_missing_member():
    step, rows = group("a", "r", "d")
    with pytest.raises(ValueError, match="inventory"):
        build_plot_data([step], rows[:1])
    rows[1]["score"] = ""
    with pytest.raises(ValueError, match="Missing numeric score"):
        build_plot_data([step], rows)
    step["token_excess"] = "2"
    rows[1]["score"] = "2"
    with pytest.raises(ValueError, match="Token excess differs"):
        build_plot_data([step], rows)


def test_render_receipt_rejects_corruption_and_preserves_sources(tmp_path):
    cfg = {"project_root": str(tmp_path)}
    directory = artifact_dir(cfg, "difficulty", "sample", "evaluations")
    directory.mkdir(parents=True)
    step, members = group("a", "r", "d")
    summary = {"run_id": "verified80-example", "coverage": {"step_status": {"comparable": 1}}}
    (directory / "summary.json").write_text(json.dumps(summary))
    for name, rows in (("steps", [step]), ("branches", members)):
        with (directory / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    (directory / "receipt.json").write_text(json.dumps({"state": "complete", "files": files}))
    result = plot_consensus_length(cfg, "plot", evaluation_ids=["sample"])
    output = tmp_path / result["output"]
    receipt = json.loads((output / "receipt.json").read_text())
    for name, digest in receipt["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert "组内排名热力图" in (output / "index.html").read_text()
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 2
    for name, digest in files.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        plot_consensus_length(cfg, "plot", evaluation_ids=["sample"])
    with pytest.raises(ValueError, match="unique"):
        plot_consensus_length(cfg, "duplicate", evaluation_ids=["sample", "sample"])
    with (directory / "branches.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        plot_consensus_length(cfg, "corrupt", evaluation_ids=["sample"])
    assert not artifact_dir(cfg, "difficulty", "corrupt", "evaluations").exists()
