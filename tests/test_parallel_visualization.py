"""Canonical selection, retry alignment, missing scores and real headless plots."""

import csv
from pathlib import Path

import pytest

from conftest import task

from application.execution.evidence import RunEvidence
from core.io import hash_json, read_json, sha256, write_json
from difficulty.evaluation.parallel import collect_parallel_scores, visualize_parallel


@pytest.fixture
def published(tmp_path, monkeypatch):
    cfg = {"project_root": str(tmp_path)}
    frozen = {**cfg, "paths": {"nas_root": str(tmp_path / "nas")},
              "capture": {"export": {"root": "hidden/online"}},
              "parallel": {"enabled": True, "estimator_sha256": "predictor"}, "environment": {"cwd": "/testbed"}}
    frozen["config_hash"] = hash_json(frozen)
    frozen.update(selection={}, selection_sha256=hash_json({}))
    write_json(tmp_path / "runs/run/resolved-config.json", frozen)
    trajectory = tmp_path / "runs/run/rollout/trajectories/r-one/raw.traj.json"
    write_json(trajectory, {"messages": [{"role": "system"}, {"role": "user"},
               {"role": "assistant"}, {"role": "tool"}, {"role": "assistant"},
               {"role": "tool"}, {"role": "assistant"}]})
    definition = task("repo__task-1")
    key = hash_json({k: definition[k] for k in ("benchmark", "revision", "instance_id")})
    patch = tmp_path / "runs/run/rollout/patches/r-one/patch.diff"
    patch.parent.mkdir(parents=True)
    patch.write_text("patch")
    rollout = {"task_key": key, "trajectory_path": str(trajectory.relative_to(tmp_path)),
               "trajectory_sha256": sha256(trajectory), "run_id": "run", "attempt_id": "attempt-000",
               "model_fingerprint": "model", "resolved": None, "submission_status": "submitted",
               "termination_reason": "Submitted", "provenance": {}, "evaluation_status": "not_run",
               "selected_evaluation_id": None, "patch_path": str(patch.relative_to(tmp_path)), "patch_sha256": sha256(patch)}
    catalog = {"schema_version": 1, "run_id": "run", "rollouts": {"r-one": rollout},
               "tasks": {key: definition}, "evaluations": {}}
    catalog["revision"] = 1
    def read_fixture(self):
        self.validate(catalog)
        write_json(self.path, catalog)
        return catalog
    monkeypatch.setattr(RunEvidence, "read", read_fixture)
    root = tmp_path / "nas/hidden/online/run"
    session = root / "sessions/session-one"
    final = {"rollout_id": "r-one", "trajectory_sha256": sha256(trajectory),
             "model_fingerprint": "model", "accepted_native_request_ids": ["format", "s0b2", "s1", "s2b1"],
             "responses": {
                 "format": {"agent_step_index": 0, "message_index": 1, "is_action_response": False},
                 "s0b2": {"agent_step_index": 0, "message_index": 2, "is_action_response": True},
                 "s1": {"agent_step_index": 1, "message_index": 4, "is_action_response": True},
                 "s2b1": {"agent_step_index": 2, "message_index": 6, "is_action_response": True}}}
    write_json(session / "finalization.json", final)

    def decision(did, step, winner, scores, natives, statuses=None):
        return {"decision_id": did, "session_id": session.name, "step_id": step, "branch_count": len(scores),
                "selected_branch_id": winner, "estimator_sha256": "predictor", "target": "reasoning_log1p",
                "endpoint": "submission", "fallback_reason": None,
                "candidates": [{"branch_id": b, "score": score, "native_request_id": native,
                                "status": statuses[b] if statuses else "eligible", "reason": None}
                               for b, (score, native) in enumerate(zip(scores, natives, strict=True))]}

    decisions = [decision("z-step0", 0, 2, [5., 1., .5, 4.], ["s0b0", "s0b1", "s0b2", "s0b3"]),
                 decision("a-step2", 2, 1, [None, None], ["s2b0", "s2b1"], ["ineligible", "unscored_valid"]),
                 decision("format", 0, 0, [8.], ["format"], ["ineligible"]),
                 decision("terminal", 3, None, [None], [None], ["ineligible"])]

    def seal():
        index = []
        for d in decisions:
            write_json(session / "parallel/decisions" / (d["decision_id"] + ".json"), d)
            for c in d["candidates"]:
                index.append({"decision_id": d["decision_id"], "branch_id": c["branch_id"],
                              "selected": c["branch_id"] == d["selected_branch_id"],
                              "native_request_id": c["native_request_id"], "score": c["score"],
                              "rollout_id": "r-one", "trajectory_sha256": sha256(trajectory)})
        write_json(session / "parallel/branch-index.json", index)
        write_json(session / "parallel/branch-coverage.json", {})
        write_json(session / "parallel/branch-receipt.json", {
            "rollout_id": "r-one", "trajectory_sha256": sha256(trajectory),
            "files": {str(p.relative_to(root)): sha256(p) for p in (session / "parallel").rglob("*.json")
                      if p.name != "branch-receipt.json"}})
    seal()
    return cfg, catalog, session, decisions, seal


def test_actual_path_joins_and_gaps(published):
    cfg, _, _, _, _ = published
    data = collect_parallel_scores(cfg, "run", "r-one")
    assert [r["step_id"] for r in data["steps"]] == [0, 1, 2]
    assert [r["selected_score"] for r in data["steps"]] == [.5, None, None]
    assert len(data["branches"]) == 6  # Every actual branch, no format retry or terminal pseudo-step.
    assert [r["selected_branch_id"] for r in data["steps"]] == [2, None, 1]
    assert data["steps"][0]["score_span"] == 4.5
    assert {r["reason"] for r in data["excluded_decisions"]} == {
        "no_winner", "format_error_response_not_an_action"}


def test_workspace_task_without_repo_keeps_canonical_binding(published,monkeypatch):
    cfg,catalog,_,_,_=published
    from action.datasets.trajectories import select_trajectories
    task=next(iter(catalog['tasks'].values()))
    task.pop('repo',None)
    monkeypatch.setattr(RunEvidence,'validate',lambda *args:None)
    selected=select_trajectories(cfg,'run',['r-one'])
    assert selected['sources'][0]['repo'] is None
    assert selected['sources'][0]['rollout_id']=='r-one'


def test_preserve_actual_winner_instead_of_reselecting_minimum(published):
    cfg, _, _, ds, seal = published
    ds[0]["candidates"][2]["score"] = 3.
    ds[0]["fallback_reason"] = "fixture_fallback"
    seal()
    data = collect_parallel_scores(cfg, "run", "r-one")
    assert data["steps"][0]["selected_score"] == 3.  # Minimum is 1, but it did not continue.


def test_parallel_proof_preserves_nonzero_winner_and_rejects_alias(published):
    from hidden.artifacts import read_parallel_proof

    cfg, catalog, session, decisions, seal = published
    root = Path(cfg['project_root']) / 'nas/hidden/online/run'
    trajectory_hash = catalog['rollouts']['r-one']['trajectory_sha256']
    proof = read_parallel_proof(root, session.name, rollout_id='r-one', trajectory_sha256=trajectory_hash)
    winner = next(row for row in proof['samples'] if row['decision_id'] == 'z-step0' and row['selected'])
    assert winner['branch_id'] == 2 and winner['native_request_id'] == 's0b2'
    decisions[0]['candidates'][0]['native_request_id'] = 's0b2'
    seal()
    with pytest.raises(ValueError, match='multiple candidates'):
        read_parallel_proof(root, session.name, rollout_id='r-one', trajectory_sha256=trajectory_hash)


def test_parallel_proof_rejects_unreceipted_branch_index(published):
    from hidden.artifacts import read_parallel_proof

    cfg, catalog, session, _, _ = published
    root = Path(cfg['project_root']) / 'nas/hidden/online/run'
    receipt_path = session / 'parallel/branch-receipt.json'
    receipt = read_json(receipt_path)
    receipt['files'].pop((session / 'parallel/branch-index.json').relative_to(root).as_posix())
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match='omits branch-index'):
        read_parallel_proof(root, session.name, rollout_id='r-one',
                            trajectory_sha256=catalog['rollouts']['r-one']['trajectory_sha256'])


def test_headless_files_and_receipt(published):
    cfg, _, _, _, _ = published
    result = visualize_parallel(cfg, "run", "plot-one", task_id="repo__task-1")
    out = Path(result["output"])
    assert (out / "trajectory.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (out / "trajectory.svg").read_text()
    assert result["missing_branch_scores"] == 2
    assert result["steps_without_parallel_decision"] == 1
    assert len(list(csv.DictReader((out / "branches.csv").open()))) == 6
    for name, digest in read_json(out / "receipt.json")["files"].items():
        assert sha256(out / name) == digest
    with pytest.raises(FileExistsError):
        visualize_parallel(cfg, "run", "plot-one", rollout_id="r-one")


def test_empty_score_zoom_is_not_fabricated(published):
    cfg, _, _, _, _ = published
    result = visualize_parallel(cfg, "run", "plot-gap", rollout_id="r-one", step_start=1, step_end=1)
    assert result["step_range"] == [1, 1]
    assert result["selected_scores"] == 0
    assert result["branch_candidates"] == 0


def test_retries_require_explicit_attempt(published):
    cfg, catalog, _, _, _ = published
    catalog["rollouts"]["r-two"] = {**catalog["rollouts"]["r-one"], "attempt_id": "attempt-001"}
    with pytest.raises(ValueError, match="specify --attempt-id"):
        visualize_parallel(cfg, "run", "ambiguous", task_id="repo__task-1")
    assert not (Path(cfg["project_root"]) / "difficulty/evaluations/ambiguous").exists()
    assert visualize_parallel(cfg, "run", "explicit", task_id="repo__task-1", attempt_id="attempt-000")


@pytest.mark.parametrize("fault", ["canonical", "receipt", "binding", "predictor", "duplicate", "wrong_run"])
def test_reject_wrong_or_changed_evidence(published, fault):
    cfg, catalog, session, ds, seal = published
    if fault == "canonical":
        (Path(cfg["project_root"]) / catalog["rollouts"]["r-one"]["trajectory_path"]).write_text('{}')
    elif fault == "receipt":
        p = session / "parallel/decisions/z-step0.json"
        p.write_text(p.read_text() + ' ')
    elif fault == "binding":
        p = session / "finalization.json"
        final = read_json(p)
        final["responses"]["s0b2"]["message_index"] = 4
        write_json(p, final)
    elif fault == "predictor":
        ds[0]["estimator_sha256"] = "different"
        seal()
    elif fault == "duplicate":
        ds[0]["candidates"][0]["branch_id"] = 2
        seal()
    elif fault == "wrong_run":
        catalog["rollouts"]["r-one"]["run_id"] = "other"
    with pytest.raises(ValueError):
        collect_parallel_scores(cfg, "run", "r-one")


@pytest.mark.parametrize("argv", [
    ["run", "--task-id", "task"],
    ["run", "plot", "--list"],
    ["run", "plot", "--rollout-id", "r-one", "--attempt-id", "attempt-000"],
])
def test_cli_invalid_scope_rejected_before_config(argv):
    from core.cli import main

    with pytest.raises(SystemExit):
        main(["difficulty-plot-parallel", *argv])
