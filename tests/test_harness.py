from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("docker")
from conftest import import_fixture

from baseline.catalog import Catalog
from baseline.evaluation import harness
from core.deadline import Deadline


def test_harness_registers_unresolved_and_infra_separately(cfg, monkeypatch):
    rid = import_fixture(cfg)
    selected = Catalog(cfg["project_root"]).select()
    cfg["selection"] = selected
    cleaned = []
    official = SimpleNamespace(
        make_test_spec=lambda row, **kw: row,
        run_instance=lambda *args: {"completed": True, "resolved": False},
        dycon_restore=lambda: cleaned.append("restore"),
    )
    monkeypatch.setattr(harness, "install_harness_adapter", lambda *args: official)
    factory = SimpleNamespace(
        lock_images=lambda x: None,
        client=None,
        deadline=Deadline(),
        cleanup_owned=lambda x: cleaned.append("cleanup"),
    )
    result = harness.evaluate(cfg, Path(cfg["project_root"]) / "runs/run", "eval", factory=factory)
    assert result["outcomes"][rid]["resolved"] is False
    assert Catalog(cfg["project_root"]).read()["rollouts"][rid]["evaluation_status"] == "completed"
    assert cleaned == ["cleanup", "restore"]


def test_harness_failure_has_null_success(cfg, monkeypatch):
    rid = import_fixture(cfg)
    cfg["selection"] = Catalog(cfg["project_root"]).select()
    official = SimpleNamespace(
        make_test_spec=lambda row, **kw: row,
        run_instance=lambda *args: {"completed": False, "resolved": False},
        dycon_restore=lambda: None,
    )
    monkeypatch.setattr(harness, "install_harness_adapter", lambda *args: official)
    factory = SimpleNamespace(
        lock_images=lambda x: None, client=None, deadline=Deadline(), cleanup_owned=lambda x: None
    )
    result = harness.evaluate(cfg, Path(cfg["project_root"]) / "runs/run", "bad", factory=factory)
    assert result["outcomes"][rid]["resolved"] is None
    assert Catalog(cfg["project_root"]).read()["rollouts"][rid]["evaluation_status"] == "infra_error"
