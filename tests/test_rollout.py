from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

pytest.importorskip("docker")
pytest.importorskip("minisweagent")

from conftest import task

from baseline.catalog import Catalog
from core.deadline import Deadline
from core.io import write_json


def test_rollout_registers_canonical_artifacts_and_resumes(cfg, monkeypatch):
    import core.model_identity
    from baseline.rollout import agent, model
    from baseline.rollout.runner import run_manifest
    from runtime.containers import rollout_adapter
    from runtime.serving import manager

    catalog = Catalog(cfg["project_root"])
    catalog.register_tasks([task(f"t-{i}") for i in range(4)])
    barrier = Barrier(4, timeout=5)

    class FakeAgent:
        def __init__(self, model, env, **kwargs):
            self.env = env

        def run(self, problem):
            barrier.wait()
            return {"exit_status": "Submitted", "submission": "patch"}

        def save(self, path, extra):
            write_json(path, {**extra, "messages": [{"role": "assistant", "reasoning_content": "done"}]})

    class Env:
        def __init__(self, *args):
            pass

        def close(self):
            pass

    class Service:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(core.model_identity, "verify_model", lambda cfg: None)
    monkeypatch.setattr(agent, "DurableAgent", FakeAgent)
    monkeypatch.setattr(model, "make_model", lambda *args: None)
    monkeypatch.setattr(rollout_adapter, "RolloutEnvironment", Env)
    monkeypatch.setattr(manager, "ServingManager", Service)
    factory = SimpleNamespace(
        deadline=Deadline(), lock_images=lambda rows: None, cleanup_owned=lambda run: None
    )
    result = run_manifest(cfg, "run", factory=factory)
    assert len(result) == 4
    assert catalog.verify()["rollouts"] == 4
    for row in catalog.read()["rollouts"].values():
        assert row["trajectory_path"].startswith("baseline/trajectories/")
        assert row["submission_status"] == "submitted"
        assert row["resolved"] is None
    assert list((Path(cfg["project_root"]) / "runs/run").rglob("raw.traj.json")) == []
    resumed = run_manifest(cfg, "run", resume=True, factory=factory)
    assert resumed == result


def test_excluded_tasks_not_selected(cfg):
    catalog = Catalog(cfg["project_root"])
    catalog.register_tasks([task("excluded", "excluded")])
    assert catalog.select_tasks()["task_keys"] == []
