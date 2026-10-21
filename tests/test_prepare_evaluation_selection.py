import pytest

from benchmark_evaluation.selection import prepare_run


def test_prepare_run_freezes_explicit_rollouts(monkeypatch):
    class Catalog:
        def __init__(self, root): pass
        def read(self):
            return {
                "catalog_revision": "rev",
                "rollouts": {
                    "r1": {"task_key": "t1", "purposes": ["baseline"], "patch_path": "p"},
                    "r2": {"task_key": "t2", "purposes": ["baseline"], "patch_path": "q"},
                },
                "tasks": {
                    "t1": {"instance_id": "i1", "partition": "eval"},
                    "t2": {"instance_id": "i2", "partition": "eval"},
                },
            }
    captured = {}
    monkeypatch.setattr("benchmark_evaluation.selection.require_model", lambda cfg: None)
    monkeypatch.setattr("benchmark_evaluation.selection.Catalog", Catalog)
    monkeypatch.setattr("benchmark_evaluation.selection.snapshot", lambda cfg, run_id, selection: captured.update(selection) or "ok")
    assert prepare_run({"project_root": "."}, "run", partition="eval", purposes=("baseline",), rollout_ids=("r2",)) == "ok"
    assert captured["rollout_ids"] == ["r2"]
    assert set(captured["tasks"]) == {"t2"}


def test_prepare_run_rejects_unknown_explicit_rollout(monkeypatch):
    class Catalog:
        def __init__(self, root): pass
        def read(self): return {"rollouts": {}, "tasks": {}, "catalog_revision": "rev"}
    monkeypatch.setattr("benchmark_evaluation.selection.require_model", lambda cfg: None)
    monkeypatch.setattr("benchmark_evaluation.selection.Catalog", Catalog)
    with pytest.raises(ValueError, match="Unknown rollout IDs"):
        prepare_run({"project_root": "."}, "run", rollout_ids=("missing",))
