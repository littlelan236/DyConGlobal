"""Exact action counts, historical retained sets, provenance and CLI coverage."""

import copy
import json
import math
from pathlib import Path

import pytest

from action.analyzers.branches import (
    analyze_branches, check_branch_analysis, decision_record, metric_summary, step_statistics,
)
from action.evaluation.branch_curves import build_curve_tables, interpolate_at
from core.action_identity import canonical_action
from core.io import read_json, sha256, write_json
from core.io import hash_json
from test_action_visualization import source


def message(command):
    return {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": command}}}]}


def decision(commands=("cat x", "cat x", "pytest", "pytest"), retained=(1, 2), selected=1):
    branches = [{"branch_id": i, "response": {"choices": [{"message": message(cmd)}]},
                 "retained": i in retained} for i, cmd in enumerate(commands)]
    return {"decision_id": "decision-0", "session_id": "session-0", "step_id": 0,
            "branch_count": len(commands), "selected_branch_id": selected, "candidates": branches,
            "consensus": {"retained_branch_ids": list(retained), "retained_count": len(retained)}}


BINDING = {"source_run_id": "run", "rollout_id": "r", "instance_id": "task", "cohort": None,
           "assistant_index": 0, "message_index": 0}


def record(d, scope="both", msg=None, kind="full"):
    return decision_record(d, d, msg or message("cat x"), scope, BINDING, {"kind": kind})


def test_actual_winner_is_not_reselected_all_branch_mode():
    d = decision(("cat x", "pytest", "pytest", "pytest"), retained=(0, 1), selected=0)
    r = step_statistics(record(d), "both")
    assert r["action_types_all"] == r["action_types_retained"] == 2
    assert r["winner_count_all"] == r["winner_count_retained"] == 1
    assert r["group_sizes_all"] == [3, 1]


def test_no_action_not_an_extra_type_and_zero_is_a_real_bin():
    d = decision()
    d["candidates"][2]["response"] = {"choices": [{"message": {}}]}
    r = step_statistics(record(d), "both")
    assert r["unavailable_all"] == 1 and r["unavailable_retained"] == 1
    assert r["action_types_all"] == 2 and r["action_types_retained"] == 1
    d["consensus"]["retained_branch_ids"] = []
    d["consensus"]["retained_count"] = 0
    for b in d["candidates"]:
        b["retained"] = False
    assert step_statistics(record(d), "both")["action_types_retained"] == 0


def test_missing_compact_evidence_rejected_and_retained_only_supported():
    d = decision()
    for b in d["candidates"]:
        if b["retained"]:
            b["action"] = canonical_action(b["response"]["choices"][0]["message"])[1]
        del b["response"]
    with pytest.raises(ValueError, match="Incomplete action evidence"):
        record(d, kind="embedded")
    r = step_statistics(record(d, "retained", kind="embedded"), "retained")
    assert r["action_types_retained"] == 2 and "action_types_all" not in r


def test_non_consensus_requires_all_scope():
    d = decision()
    del d["consensus"]
    for b in d["candidates"]:
        del b["retained"]
    with pytest.raises(ValueError, match="No recorded"):
        record(d)
    assert step_statistics(record(d, "all"), "all")["winner_count_all"] == 2


def test_reject_duplicates_vote_mismatch_and_wrong_winner():
    d = decision()
    d["candidates"][3]["branch_id"] = 2
    with pytest.raises(ValueError, match="branch IDs"):
        record(d)
    d = decision()
    with pytest.raises(ValueError, match="canonical assistant"):
        record(d, msg=message("wrong"))
    d["consensus"]["votes"] = [{"action": canonical_action(message("cat x"))[1], "count": 2}]
    with pytest.raises(ValueError, match="votes differ"):
        record(d)


def test_canonical_identity_whitespace_and_order():
    assert canonical_action(message("cat x")) != canonical_action(message("cat  x"))
    a = message("cat x")
    b = message("pytest")
    ab = {"tool_calls": a["tool_calls"] + b["tool_calls"]}
    ba = {"tool_calls": b["tool_calls"] + a["tool_calls"]}
    assert canonical_action(ab) != canonical_action(ba)


def fixture_run(cfg, run, rid="same-id", iid=None):
    root = Path(cfg["project_root"])
    row = source(cfg, run, rid=rid, iid=iid, commands=("cat x", "pytest"))
    trajectory = root / row["trajectory_path"]
    content = read_json(trajectory)
    for index, msg in enumerate(content["messages"]):
        cmd = "cat x" if index == 0 else "pytest"
        d = decision((cmd, cmd, "other", "other"))
        d["decision_id"] = f"decision-{index}"
        session = f"session-{run}" if rid == "same-id" else f"session-{run}-{rid}"
        d["session_id"] = session
        d["step_id"] = index
        compact = copy.deepcopy(d)
        for b in compact["candidates"]:
            del b["response"]
        msg["extra"] = {"dycon_parallel": compact}
        write_json(root / f"hidden/spool/{run}/sessions/{session}/parallel/decisions/decision-{index}.json", d)
    write_json(trajectory, content)
    ep = root / f"runs/{run}/evidence.json"
    evidence = read_json(ep)
    evidence["rollouts"][rid]["trajectory_sha256"] = sha256(trajectory)
    write_json(ep, evidence)
    snapshot = copy.deepcopy(cfg)
    snapshot["project_root"] = str(root)
    snapshot["selection"] = {}
    snapshot["selection_sha256"] = hash_json(snapshot["selection"])
    snapshot.pop("config_hash", None)
    snapshot["config_hash"] = hash_json({k: v for k, v in snapshot.items()
                                          if k not in {"config_hash", "selection", "selection_sha256"}})
    snapshot["capture"]["export"]["spool_root"] = "hidden/spool"
    write_json(root / f"runs/{run}/resolved-config.json", snapshot)
    write_json(root / f"runs/{run}/status.json", {"stage": "complete"})


def test_cross_run_cli_frozen_outputs_and_tamper(cfg, capsys):
    from core.cli import main
    fixture_run(cfg, "one")
    fixture_run(cfg, "two")
    root = Path(cfg["project_root"])
    listing = root / "bindings.json"
    write_json(listing, [{"source_run_id": run, "rollout_id": "same-id", "cohort": run} for run in ("one", "two")])
    main(["--config", cfg["config_source"], "action-branch-stats", "branches", "--source-run", "one",
          "--source-run", "two", "--rollout-list", str(listing), "--format", "png"])
    capsys.readouterr()
    out = root / "action/outputs/branches"
    summary = read_json(out / "summary.json")
    assert summary["steps"] == 4 and summary["trajectories"] == 2
    assert set(summary["by_source_run"]) == {"one", "two"}
    assert len(list((out / "plots").glob("*.png"))) == 5
    assert len(list((out / "plots/per-run").glob("*.png"))) == 2
    assert (out / "plots/all-runs-normalized.png").is_file()
    import csv
    with (out / "curves/per-run.csv").open(encoding="utf-8-sig", newline="") as stream:
        curve_rows = list(csv.DictReader(stream))
    assert all(r["axis"] == "normalized_progress" for r in curve_rows)
    assert {float(r["x"]) for r in curve_rows} == {i / 100 for i in range(101)}
    assert (out / "curves/raw-steps.csv").is_file()
    assert summary["metrics"]["normalized_entropy_all"]["mean"] == pytest.approx(1)
    assert summary["filter_effects"]["entropy"]["mean"] == pytest.approx(0)
    assert check_branch_analysis(cfg, "branches")["verified"]
    before = sha256(out / "branch-receipt.json")
    main(["--config", cfg["config_source"], "action-branch-stats", "branches", "--check"])
    assert sha256(out / "branch-receipt.json") == before
    with pytest.raises(FileExistsError):
        analyze_branches(cfg, "branches", run_id="one")
    with (out / "steps.csv").open("a") as f:
        f.write("tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        check_branch_analysis(cfg, "branches")


def test_source_change_or_missing_full_record_fails(cfg):
    fixture_run(cfg, "one")
    root = Path(cfg["project_root"])
    path = root / "hidden/spool/one/sessions/session-one/parallel/decisions/decision-0.json"
    d = read_json(path)
    d["selected_branch_id"] = 3
    write_json(path, d)
    with pytest.raises(ValueError, match="Embedded/full decision mismatch"):
        analyze_branches(cfg, "bad", run_id="one", rollout_ids=["same-id"])
    assert not (root / "action/outputs/bad").exists()
    path.unlink()
    with pytest.raises(ValueError, match="Incomplete action evidence"):
        analyze_branches(cfg, "missing", run_id="one", rollout_ids=["same-id"])


def test_all_entropy_formulas_and_filter_effects():
    d = decision(tuple(["cat x"] * 12 + ["b"] * 5 + ["c"] * 3),
                 retained=tuple(list(range(8)) + [12, 13]), selected=0)
    r = step_statistics(record(d), "both")
    h = -(0.6 * math.log(0.6) + 0.25 * math.log(0.25) + 0.15 * math.log(0.15))
    kept_h = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    assert r["entropy_all"] == pytest.approx(h)
    assert r["normalized_entropy_all"] == pytest.approx(h / math.log(3))
    assert r["effective_action_count_all"] == pytest.approx(math.exp(h))
    assert r["winner_support_all"] == pytest.approx(0.6)
    assert r["winner_uncertainty_all"] == pytest.approx(0.4)
    assert r["winner_surprisal_all"] == pytest.approx(-math.log(0.6))
    assert r["winner_surprisal_status_all"] == "finite"
    assert r["delta_entropy"] == pytest.approx(kept_h - h)
    assert r["delta_effective_action_count"] == pytest.approx(math.exp(kept_h) - math.exp(h))
    assert r["delta_winner_support"] == pytest.approx(0.2)
    assert sum(metric_summary([r], "action_types_all")["histogram"].values()) == 1
    assert metric_summary([r], "entropy_all")["histogram"] == {}


def test_consensus_uniform_empty_and_zero_winner_support():
    r = step_statistics(record(decision(("cat x",) * 4)), "both")
    assert r["entropy_all"] == r["normalized_entropy_all"] == 0
    assert r["effective_action_count_all"] == r["winner_support_all"] == 1
    assert r["winner_uncertainty_all"] == r["winner_surprisal_all"] == 0
    d = decision(("cat x", "b", "c", "d"), retained=(1, 2), selected=0)
    r = step_statistics(record(d), "both")
    assert r["entropy_all"] == pytest.approx(math.log(4))
    assert r["normalized_entropy_all"] == pytest.approx(1)
    assert r["effective_action_count_all"] == pytest.approx(4)
    assert r["winner_support_retained"] == 0
    assert r["winner_uncertainty_retained"] == 1
    assert r["winner_surprisal_retained"] is None
    assert r["winner_surprisal_status_retained"] == "positive_infinity"
    assert metric_summary([r], "winner_surprisal_retained")["positive_infinity_steps"] == 1
    r = step_statistics(record(decision(retained=())), "both")
    for m in ("entropy", "normalized_entropy", "effective_action_count", "winner_support"):
        assert r[f"{m}_retained"] is None


def test_length_normalization_equal_trajectory_weight_and_endpoints():
    trajectories = [{"source_run_id": "run", "rollout_id": rid, "assistant_steps": len(values)}
                    for rid, values in (("short", [0, 0]), ("long", [0, 10, 10, 10, 0]))]
    rows = [{"source_run_id": "run", "rollout_id": rid, "assistant_index": i,
             "normalized_progress": i / (len(values) - 1), "entropy_all": value}
            for rid, values in (("short", [0, 0]), ("long", [0, 10, 10, 10, 0]))
            for i, value in enumerate(values)]
    raw, normalized = build_curve_tables(rows, trajectories, 5)
    assert [p["x"] for p in normalized] == [0, 0.25, 0.5, 0.75, 1]
    assert [p["mean"] for p in normalized] == [0, 5, 5, 5, 0]
    assert all(p["n_trajectories"] == 2 for p in normalized)
    assert normalized[2]["sample_std"] == pytest.approx(math.sqrt(50))
    assert normalized[2]["standard_error"] == pytest.approx(5)
    assert raw[-1]["n_trajectories"] == 1


def test_no_extrapolation_missing_gaps_singleton_and_gapped_parallel_steps():
    assert interpolate_at([(0, 1), (0.5, None), (1, 9)], 0.25) is None
    assert interpolate_at([(0.25, 2), (0.75, 6)], 0) is None
    assert interpolate_at([(0.25, 2), (0.75, 6)], 0.5) == 4
    assert interpolate_at([(0, 5)], 0) == 5
    assert interpolate_at([(0, 5)], 0.1) is None
    trajectory = [{"source_run_id": "run", "rollout_id": "r", "assistant_steps": 5}]
    rows = [{"source_run_id": "run", "rollout_id": "r", "assistant_index": i,
             "normalized_progress": i / 4, "entropy_retained": float(i)} for i in (1, 3)]
    raw, normalized = build_curve_tables(rows, trajectory, 5)
    assert [p["x"] for p in raw] == [2, 4]
    assert [p["n_trajectories"] for p in normalized] == [0, 1, 1, 1, 0]
    assert normalized[2]["mean"] == 2
    rows[0]["normalized_progress"] = 0
    with pytest.raises(ValueError, match="trajectory length"):
        build_curve_tables(rows, trajectory, 5)


@pytest.mark.parametrize("mode", ["both", "per-trajectory"])
def test_run_axes_normalized_and_individual_trajectory_axes_raw(tmp_path, monkeypatch, mode):
    from matplotlib.figure import Figure
    from action.evaluation.branch_curves import build_trajectory_curves, render_line_plots
    trajectories = [{"source_run_id": "run", "rollout_id": "r", "assistant_steps": 3,
                     "instance_id": "same-task", "trajectory_sha256": "fixture-sha", "parallel_steps": 3}]
    rows = [{"source_run_id": "run", "rollout_id": "r", "assistant_index": i,
             "normalized_progress": i / 2, "entropy_all": float(i)} for i in range(3)]
    raw, normalized = build_curve_tables(rows, trajectories, 5)
    captured = []
    def capture(self, filename, **kwargs):
        ax = self.axes[0]
        captured.append({"filename": str(filename), "x": list(ax.lines[0].get_xdata()),
                         "limits": ax.get_xlim(), "xlabel": ax.get_xlabel(),
                         "ticks": [t.get_text() for t in ax.get_xticklabels()]})
    monkeypatch.setattr(Figure, "savefig", capture)
    render_line_plots(tmp_path, raw, normalized, {"line_mode": mode, "dpi": 60, "formats": ["png"],
                                               "line_metrics": ["entropy"]}, build_trajectory_curves(rows), trajectories)
    assert len(captured) == (2 if mode == "both" else 1)
    for plot in captured:
        if mode == "per-trajectory":
            assert plot["x"] == [1, 2, 3]
            assert "Canonical assistant step" in plot["xlabel"]
        else:
            assert plot["x"] == [0, 0.25, 0.5, 0.75, 1]
            assert plot["limits"] == (0, 1)
            assert any("%" in tick for tick in plot["ticks"])


def test_per_trajectory_cli_keeps_same_task_repetitions_separate(cfg, capsys):
    import csv
    from core.cli import main
    fixture_run(cfg, "one")
    fixture_run(cfg, "one", rid="repeat-2", iid="one-task")
    root = Path(cfg["project_root"])
    main(["--config", cfg["config_source"], "action-branch-stats", "individual", "--source-run", "one",
          "--rollout-id", "same-id", "--rollout-id", "repeat-2", "--line-mode", "per-trajectory", "--format", "png"])
    capsys.readouterr()
    out = root / "action/outputs/individual"
    analysis = read_json(out / "analysis.json")
    assert "task_ids" not in analysis
    assert analysis["curve_schema_version"] == 4
    assert {r["instance_id"] for r in read_json(out / "selection.json")["sources"]} == {"one-task"}
    assert len(list((out / "plots/per-trajectory/one").glob("*.png"))) == 2
    assert not (out / "plots/all-runs-normalized.png").exists()
    assert not (out / "curves/all-runs-normalized.csv").exists()
    with (out / "curves/per-trajectory.csv").open(encoding="utf-8-sig", newline="") as stream:
        curves = list(csv.DictReader(stream))
    assert {r["axis"] for r in curves} == {"assistant_step"}
    assert {r["rollout_id"] for r in curves} == {"same-id", "repeat-2"}
    assert all("value" in r and "mean" not in r for r in curves)
    assert {int(r["x"]) for r in curves} == {1, 2}
    assert check_branch_analysis(cfg, "individual")["verified"]
    index = read_json(out / "plots/trajectory-index.json")
    assert index["trajectory_count"] == 2
    assert all(e["metrics"] == ["entropy", "normalized_entropy", "action_types", "effective_action_count", "winner_uncertainty"]
               for e in index["entries"])
    main(["--config", cfg["config_source"], "action-branch-stats", "one-trajectory", "--source-run", "one",
          "--rollout-id", "repeat-2", "--line-mode", "per-trajectory", "--line-metric", "entropy", "--format", "png"])
    single = root / "action/outputs/one-trajectory"
    assert len(list((single / "plots/per-trajectory/one").glob("*.png"))) == 1
    assert read_json(single / "analysis.json")["line_metrics"] == ["entropy"]
    assert check_branch_analysis(cfg, "one-trajectory")["trajectories"] == 1
    with pytest.raises(ValueError, match="Line metrics unavailable"):
        analyze_branches(cfg, "bad-metric", run_id="one", rollout_ids=["same-id"], scope="all",
                         line_mode="per-trajectory", line_metrics=["delta_entropy"])
    with pytest.raises(SystemExit):
        main(["--config", cfg["config_source"], "action-branch-stats", "no-task-option", "--source-run", "one",
              "--task-id", "one-task"])


def test_individual_curve_values_are_never_averaged_by_task():
    from action.evaluation.branch_curves import build_trajectory_curves
    rows = [{"source_run_id": run, "rollout_id": rid, "instance_id": "same-task", "assistant_index": 0,
             "entropy_all": value} for run, rid, value in (("a", "r", 0.2), ("a", "repeat", 3.0), ("b", "r", 1.0))]
    curves = build_trajectory_curves(rows)
    assert {(r["source_run_id"], r["rollout_id"]): r["value"] for r in curves} == {
        ("a", "r"): 0.2, ("a", "repeat"): 3.0, ("b", "r"): 1.0}
    assert all(r["x"] == 1 for r in curves)
    with pytest.raises(ValueError, match="Duplicate trajectory"):
        build_trajectory_curves(rows + [rows[0]])


def test_plot_index_disambiguates_runs_and_marks_empty_trajectories():
    from action.evaluation.branch_curves import trajectory_plot_index
    sources = [{"source_run_id": run, "rollout_id": "same-rid", "instance_id": "same-task",
                "trajectory_sha256": "test-sha", "parallel_steps": steps} for run, steps in (("one", 2), ("two", 0))]
    index = trajectory_plot_index(sources, {"line_metrics": ["entropy"], "formats": ["png"]})
    assert index["entries"][0]["files"] != index["entries"][1]["files"]
    assert index["entries"][1]["status"] == "no_parallel_steps"


@pytest.mark.parametrize("mode,count", [("per-run", 2), ("all-runs", 2), ("both", 4), ("per-trajectory", 2)])
def test_separate_layout_has_one_axis_and_one_metric_per_file(tmp_path, monkeypatch, mode, count):
    from matplotlib.figure import Figure

    from action.evaluation.branch_curves import build_trajectory_curves, render_line_plots
    trajectories = [{"source_run_id": "run", "rollout_id": "r", "assistant_steps": 3,
                     "instance_id": "task", "trajectory_sha256": "fixture-sha", "parallel_steps": 3}]
    rows = [{"source_run_id": "run", "rollout_id": "r", "assistant_index": i,
             "normalized_progress": i / 2, "entropy_all": float(i), "branches_all": 4}
            for i in range(3)]
    raw, normalized = build_curve_tables(rows, trajectories, 5, include_branch_counts=True)
    captured = []

    def capture(self, filename, **kwargs):
        assert len(self.axes) == 1
        values = list(self.axes[0].lines[0].get_ydata())
        captured.append((str(filename), values))

    monkeypatch.setattr(Figure, "savefig", capture)
    render_line_plots(tmp_path, raw, normalized,
                      {"line_mode": mode, "plot_layout": "separate", "dpi": 60,
                       "formats": ["png"], "line_metrics": ["entropy", "branches"]},
                      build_trajectory_curves(rows, include_branch_counts=True), trajectories)
    assert len(captured) == count
    for filename, values in captured:
        if filename.endswith("/branches.png"):
            assert set(values) == {4}
        else:
            assert filename.endswith("/entropy.png")
            assert values[0] == 0 and values[-1] == 2
    index = read_json(tmp_path / "plot-index.json")
    assert index["figure_count"] == index["file_count"] == count


def test_branch_counts_are_explicitly_enabled_and_historical_curve_columns_stay_stable():
    from action.evaluation.branch_curves import metric_columns, resolve_line_metrics
    row = {"branches_all": 4, "branches_retained": 2, "entropy_all": 0.5}
    assert not any(metric == "branches" for _, metric, _ in metric_columns(row))
    assert ("branches_all", "branches", "all") in metric_columns(row, include_branch_counts=True)
    assert resolve_line_metrics([{"entropy_all": 0.5}, row], "per-run", ["branches"],
                                include_branch_counts=True) == ["branches"]


def test_separate_batch_cli_checks_all_files_and_does_not_invent_argmin_retention(cfg, capsys):
    from core.cli import main
    fixture_run(cfg, "one")
    root = Path(cfg["project_root"])
    # A historical argmin decision has no consensus retained set.
    trajectory = root / "runs/one/rollout/trajectories/same-id/raw.traj.json"
    content = read_json(trajectory)
    for msg in content["messages"]:
        embedded = msg["extra"]["dycon_parallel"]
        embedded.pop("consensus", None)
        for branch in embedded["candidates"]:
            branch.pop("retained", None)
        full_path = root / (f"hidden/spool/one/sessions/{embedded['session_id']}/parallel/decisions/"
                            f"{embedded['decision_id']}.json")
        full = read_json(full_path)
        full.pop("consensus", None)
        for branch in full["candidates"]:
            branch.pop("retained", None)
        write_json(full_path, full)
    write_json(trajectory, content)
    ep = root / "runs/one/evidence.json"
    evidence = read_json(ep)
    evidence["rollouts"]["same-id"]["trajectory_sha256"] = sha256(trajectory)
    write_json(ep, evidence)
    main(["--config", cfg["config_source"], "action-branch-stats", "separate", "--source-run", "one",
          "--rollout-id", "same-id",
          "--branch-scope", "all", "--line-mode", "per-run", "--plot-layout", "separate",
          "--line-metric", "entropy", "--line-metric", "branches", "--format", "png", "--format", "svg"])
    capsys.readouterr()
    out = root / "action/outputs/separate"
    index = read_json(out / "plots/plot-index.json")
    assert index["figure_count"] == 2 and index["file_count"] == 4
    assert all((out / f).is_file() for e in index["entries"] for f in e["files"])
    rows = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    assert all(r["branches_all"] == 4 and "branches_retained" not in r and "decision_keep_count" not in r
               for r in rows)
    assert check_branch_analysis(cfg, "separate")["verified"]
    # Even re-signing a receipt must not hide omission of an expected image.
    missing = index["entries"][-1]["files"][-1]
    (out / missing).unlink()
    receipt = read_json(out / "branch-receipt.json")
    receipt["files"].pop(missing)
    write_json(out / "branch-receipt.json", receipt)
    with pytest.raises(ValueError, match="Missing per-metric figure"):
        check_branch_analysis(cfg, "separate")
