from pathlib import Path
import pytest
from benchmark_evaluation.selection import prepare_run
from core.config import load_snapshot
from tests.conftest import import_fixture


def test_prepare_run_freezes_explicit_rollouts(cfg, monkeypatch):
    rid = import_fixture(cfg)
    monkeypatch.setattr("benchmark_evaluation.selection.require_model", lambda cfg: None)
    run = prepare_run({**cfg, "source_run_id": "source"}, "evaluation", rollout_ids=[rid])
    selected = load_snapshot(cfg["project_root"], "evaluation")["selection"]
    assert run == Path(cfg["project_root"]) / "runs/evaluation"
    assert selected["source_run_id"] == "source"
    assert selected["rollout_ids"] == [rid]
    assert len(selected["tasks"]) == 1


def test_prepare_run_rejects_unknown_explicit_rollout(cfg, monkeypatch):
    import_fixture(cfg)
    monkeypatch.setattr("benchmark_evaluation.selection.require_model", lambda cfg: None)
    with pytest.raises(ValueError, match="Unknown rollout IDs"):
        prepare_run({**cfg, "source_run_id": "source"}, "evaluation", rollout_ids=["missing"])


def test_prepare_requires_a_source_run(cfg, monkeypatch):
    monkeypatch.setattr("benchmark_evaluation.selection.require_model", lambda cfg: None)
    with pytest.raises(ValueError, match="source run"):
        prepare_run(cfg, "evaluation", rollout_ids=["some-id"])
