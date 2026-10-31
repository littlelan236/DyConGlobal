"""Historical pool reconstruction, production mappings, and complete HTML output."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_action_branches import BINDING, fixture_run, message

from action.analyzers.branches import analyze_branches, check_branch_analysis, decision_record
from action.evaluation.entropy_mappings import (
    VERSION,
    decision_mapping_input,
    mapping_curve_tables,
    mapping_rows,
    replay_mapping_winner,
)
from core.action_identity import canonical_action
from core.io import hash_json, read_json, sha256, write_json
from rollout.consensus import select_consensus
from rollout.entropy import RETENTION_RULES


def real_decision(commands=("x", "x", "y", "y"), scores=(1, 1, 2, 3)):
    candidates = [SimpleNamespace(branch_id=i, message={}, response=SimpleNamespace(
        model_dump=lambda mode="json", cmd=cmd: {"choices": [{"message": message(cmd)}]}))
        for i, cmd in enumerate(commands)]
    options = {"retention_ratio": 0.5, "min_retained_branches": 2,
               "rule": "exact_action", "tie_policy": "estimator_score"}
    score_map = dict(enumerate(scores))
    winner, meta = select_consensus(candidates, score_map)
    by_id = {item["branch_id"]: item for item in meta["branch_metadata"]}
    branches = [{"branch_id": i, "status": "eligible", "reason": None, "score": scores[i],
                 "response": candidate.response.model_dump(), **by_id[i]} for i, candidate in enumerate(candidates)]
    return {"decision_id": "decision-0", "session_id": "session-0", "step_id": 0,
            "branch_count": len(branches), "candidates": branches,
            "selected_branch_id": winner.branch_id,
            "selection_mode": "parallel_consensus", "consensus_config": options, "consensus": meta}


def frozen_input():
    decision = real_decision()
    record = decision_record(decision, decision, message("x"), "both", BINDING, {"kind": "full"})
    config = {"parallel": {"mode": "parallel_consensus", "consensus": decision["consensus_config"]}}
    record["entropy_mapping_input"] = decision_mapping_input(decision, record, config)
    return record, decision, config


def test_production_mapping_budget_ranks_and_counterfactual_label():
    record, _, _ = frozen_input()
    trajectories = [{"source_run_id": "run", "rollout_id": "r", "assistant_steps": 1}]
    rows = mapping_rows([record], trajectories)
    assert [row["retention_rule"] for row in rows] == list(RETENTION_RULES)
    assert [row["keep_count"] for row in rows] == [2, 3, 3]
    assert [row["cutoff_branch_id"] for row in rows] == [1, 2, 2]
    assert rows[0]["retained_branch_ids"] == [0, 1]
    assert all(row["pool_entropy"] == pytest.approx(math.log(2)) for row in rows)
    assert all(row["interpretation"] == "counterfactual_on_observed_pool" for row in rows)
    assert all(row["actual_retained_count"] == 2 for row in rows)
    assert all(row["winner_support_before"] is not None for row in rows)
    assert all(row["winner_support_after"] is not None for row in rows)
    assert all(row["winner_changed_rate"] in (0.0, 1.0) for row in rows)


def test_mapping_input_validates_original_config_and_unfiltered_action():
    record, decision, config = frozen_input()
    assert len(record["entropy_mapping_input"]["ranked_candidates"]) == 4
    bad_config = copy.deepcopy(config)
    bad_config["parallel"]["consensus"]["retention_ratio"] = 0.8
    with pytest.raises(ValueError, match="differs from frozen run"):
        decision_mapping_input(decision, record, bad_config)
    decision["candidates"][3].pop("response")
    with pytest.raises(ValueError, match="lacks unfiltered candidate action"):
        decision_mapping_input(decision, record, config)


def test_empty_and_insufficient_pools_and_no_average_then_map():
    record, _, _ = frozen_input()
    trajectories = [{"source_run_id": "run", "rollout_id": "r", "assistant_steps": 3}]
    # Utility-level boundary records, not invented successful canonical decisions.
    for pool, expected_h in (([], None), ([{"branch_id": 0, "score": 1,
                                          "action": canonical_action(message("x"))[1]}], 0.0)):
        edge = copy.deepcopy(record)
        edge["entropy_mapping_input"]["ranked_candidates"] = pool
        rows = mapping_rows([edge], trajectories)
        assert all(row["keep_count"] == 0 and row["cutoff_score"] is None for row in rows)
        assert all(row["pool_entropy"] == expected_h for row in rows)
    last = copy.deepcopy(record)
    last["assistant_index"] = 2
    for candidate in last["entropy_mapping_input"]["ranked_candidates"]:
        candidate["action"] = canonical_action(message("one"))[1]
    rows = mapping_rows([record, last], trajectories)
    _, curves = mapping_curve_tables(rows, trajectories, 3)
    midpoint = next(row for row in curves if row["retention_rule"] == "shannon_exp_keep_v1"
                    and row["metric"] == "keep_count" and row["x"] == 0.5)
    assert midpoint["mean"] == 2.5
    assert midpoint["n_trajectories"] == 1


def mapping_fixture_run(cfg):
    fixture_run(cfg, "one")
    root = Path(cfg["project_root"])
    snapshot_path = root / "runs/one/resolved-config.json"
    config = read_json(snapshot_path)
    config.setdefault("parallel", {})["mode"] = "parallel_consensus"
    config["parallel"]["consensus"] = real_decision()["consensus_config"]
    config["config_hash"] = hash_json({key: value for key, value in config.items()
                                     if key not in {"config_hash", "selection", "selection_sha256"}})
    write_json(snapshot_path, config)
    evidence_path = root / "runs/one/evidence.json"
    evidence = read_json(evidence_path)
    trajectory_path = root / evidence["rollouts"]["same-id"]["trajectory_path"]
    trajectory = read_json(trajectory_path)
    for index, msg in enumerate(trajectory["messages"]):
        cmd = "cat x" if index == 0 else "pytest"
        decision = real_decision((cmd, cmd, "other", "other"))
        decision.update(decision_id=f"decision-{index}", session_id="session-one", step_id=index)
        compact = copy.deepcopy(decision)
        for branch in compact["candidates"]:
            del branch["response"]
        msg["extra"] = {"dycon_parallel": compact}
        write_json(root / f"hidden/spool/one/sessions/session-one/parallel/decisions/decision-{index}.json", decision)
    write_json(trajectory_path, trajectory)
    evidence["rollouts"]["same-id"]["trajectory_sha256"] = sha256(trajectory_path)
    write_json(evidence_path, evidence)


def test_cli_batch_html_all_figures_and_recompute_tamper(cfg, capsys):
    from core.cli import main

    mapping_fixture_run(cfg)
    main(["--config", cfg["config_source"], "action-branch-stats", "mapping-test", "--source-run", "one",
          "--rollout-id", "same-id",
          "--line-metric", "entropy", "--format", "png", "--format", "svg", "--dpi", "60"])
    output = json.loads(capsys.readouterr().out)
    assert output["html_report"].endswith("index.html")
    root = Path(cfg["project_root"])
    out = root / "action/outputs/mapping-test"
    analysis = read_json(out / "analysis.json")
    assert analysis["line_mode"] == "per-run"
    assert analysis["plot_layout"] == "separate"
    assert analysis["entropy_mapping_comparison"]["rules"] == list(RETENTION_RULES)
    gallery = read_json(out / "entropy-mappings/gallery-index.json")
    mapping_index = read_json(out / "entropy-mappings/plot-index.json")
    assert mapping_index["figure_count"] == 18 and mapping_index["file_count"] == 36
    assert gallery["figure_count"] == 23 and gallery["file_count"] == 46
    page = (out / "index.html").read_text(encoding="utf-8")
    assert page.count('<img ') == gallery["figure_count"]
    assert 'src="http' not in page and 'src="data:image/png;base64,' in page
    assert "没有重新运行 agent" in page
    assert check_branch_analysis(cfg, "mapping-test")["verified"]
    path = out / "entropy-mappings/steps.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["keep_count"] += 1
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding="utf-8")
    receipt_path = out / "branch-receipt.json"
    receipt = read_json(receipt_path)
    receipt["files"]["entropy-mappings/steps.jsonl"] = sha256(path)
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="step values differ"):
        check_branch_analysis(cfg, "mapping-test")


def test_report_rejects_conflicting_explicit_plot_mode(cfg):
    mapping_fixture_run(cfg)
    with pytest.raises(ValueError, match="per-run.*separate"):
        analyze_branches(cfg, "invalid-mode", run_id="one", rollout_ids=["same-id"],
                         line_mode="per-trajectory", entropy_mapping_comparison=True)


def test_default_report_can_be_disabled_and_check_rejects_override(cfg, capsys):
    from core.cli import main

    mapping_fixture_run(cfg)
    base = ["--config", cfg["config_source"], "action-branch-stats", "mapping-off"]
    main(base + ["--source-run", "one", "--rollout-id", "same-id",
                 "--no-entropy-mapping-comparison", "--line-mode", "none", "--format", "png"])
    output = json.loads(capsys.readouterr().out)
    out = Path(output["output"])
    assert not (out / "index.html").exists()
    assert "entropy_mapping_comparison" not in read_json(out / "analysis.json")
    assert check_branch_analysis(cfg, "mapping-off")["verified"]
    with pytest.raises(SystemExit):
        main(base + ["--check", "--no-entropy-mapping-comparison"])


def test_rank_and_version_tampering_rejected():
    record, _, _ = frozen_input()
    record["entropy_mapping_input"]["ranked_candidates"].reverse()
    trajectories = [{"source_run_id": "run", "rollout_id": "r", "assistant_steps": 1}]
    with pytest.raises(ValueError, match="ranking differs"):
        mapping_rows([record], trajectories)
    record["entropy_mapping_input"]["version"] = VERSION + "tamper"
    with pytest.raises(ValueError, match="Unknown"):
        mapping_rows([record], trajectories)


def test_mapping_winner_reuses_mean_score_tiebreak():
    x, y, z = [canonical_action(message(command))[1] for command in ("x", "y", "z")]
    pool = [{"branch_id": bid, "score": score, "action": action}
            for bid, score, action in ((0, 0, x), (1, 0, y), (2, 0.2, z), (3, 1, y), (4, 10, x))]
    source = {"minimum": 2, "retention_ratio": 0.5}
    assert replay_mapping_winner(pool, source, "baseline_floor_v1") == y


def test_support_and_change_reference_unfiltered_pool_winner():
    x, y = [canonical_action(message(command))[1] for command in ("x", "y")]
    record, _, _ = frozen_input()
    record["entropy_mapping_input"]["ranked_candidates"] = [
        {"branch_id": i, "score": i, "action": action} for i, action in enumerate((x, x, y, y, y))]
    trajectories = [{"source_run_id": "run", "rollout_id": "r", "assistant_steps": 1}]
    row = mapping_rows([record], trajectories)[0]
    assert row["reference_winner_action"] == y
    assert row["winner_support_before"] == pytest.approx(3 / 5)
    assert row["winner_support_after"] == 0
    assert row["mapped_winner_action"] == x
    assert row["winner_changed_rate"] == 1.0
