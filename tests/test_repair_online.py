import json
import pytest


def test_repair_cli_is_registered():
    from core.cli import main

    # argparse reaches the repair subcommand without importing serving workers.
    try:
        main(["capture", "repair", "repair-run", "--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_repair_report_is_a_separate_namespace(tmp_path):
    from hidden.repair_online import _public

    audit = {"rollout_id": "r", "events": [(None, {})], "bindings": [],
             "session": {"session_id": "s"}, "complete_candidate": False}
    assert _public(audit) == {"rollout_id": "r", "complete_candidate": False}


def test_retry_preserves_original_finalization(tmp_path):
    from hidden.repair_online import _reuse_finalization
    from core.io import write_json, sha256
    roots = [tmp_path / "spool", tmp_path / "nas"]
    prior = {"run_id": "run", "trajectory_sha256": "abc", "responses": {"native": {}},
             "repair_id": "old", "repaired_at": "old-time"}
    coverage = {"expected": 1, "observed": 1, "missing": []}
    for root in roots:
        write_json(root / "sessions/s/finalization.json", prior)
        write_json(root / "sessions/s/coverage.json", coverage)
    hashes = [sha256(root / "sessions/s/finalization.json") for root in roots]
    assert _reuse_finalization(roots, "s", {**prior, "repair_id": "new", "repaired_at": "new-time"}, coverage) == prior
    assert hashes == [sha256(root / "sessions/s/finalization.json") for root in roots]


def test_retry_rejects_changed_evidence(tmp_path):
    from hidden.repair_online import _reuse_finalization, EvidenceConflict
    from core.io import write_json
    write_json(tmp_path / "sessions/s/finalization.json", {"trajectory_sha256": "original"})
    with pytest.raises(EvidenceConflict, match="evidence conflicts"):
        _reuse_finalization([tmp_path], "s", {"trajectory_sha256": "changed"}, {})


def test_repair_cli_dispatches_repair(monkeypatch, tmp_path, capsys):
    import application.execution.run_plan as plans
    import hidden.repair_online as repair
    import core.cli as cli
    monkeypatch.setattr(plans, "load_plan", lambda *args: {"config": {"project_root": str(tmp_path)}})
    monkeypatch.setattr(cli, "Catalog", lambda *args: None)
    calls = []
    def fake(cfg, run_id, **kwargs):
        calls.append((run_id, kwargs))
        return {"dispatched": "repair"}
    monkeypatch.setattr(repair, "repair_online", fake)
    cli.main(["capture", "repair", "repair-run", "--apply", "--repair-id", "retry"])
    assert calls == [("repair-run", {"rollout_ids": [], "session_ids": [], "apply": True, "repair_id": "retry", "pointwise": False})]
    assert '"dispatched": "repair"' in capsys.readouterr().out
