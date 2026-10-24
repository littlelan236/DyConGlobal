from application.execution.evidence import RunEvidence
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("docker")
from conftest import import_fixture

from catalog.store import Catalog
from benchmark_evaluation import harness
from core.deadline import Deadline


def test_harness_registers_unresolved_and_infra_separately(cfg, monkeypatch):
    rid = import_fixture(cfg)
    selected = RunEvidence(cfg["project_root"], "source").select()
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
    assert RunEvidence(cfg["project_root"], "run").read()["rollouts"][rid]["evaluation_status"] == "completed"
    assert cleaned == ["cleanup", "restore"]


def test_harness_failure_has_null_success(cfg, monkeypatch):
    rid = import_fixture(cfg)
    cfg["selection"] = RunEvidence(cfg["project_root"], "source").select()
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
    assert RunEvidence(cfg["project_root"], "run").read()["rollouts"][rid]["evaluation_status"] == "infra_error"


def test_harness_retries_network_error_and_records_attempts(cfg, monkeypatch):
    rid = import_fixture(cfg)
    cfg["selection"] = RunEvidence(cfg["project_root"], "source").select()
    calls = []
    def run_instance(*args):
        calls.append(1)
        if len(calls) <= 2:
            raise ConnectionError("reset")
        return {"completed": True, "resolved": True}
    official = SimpleNamespace(
        make_test_spec=lambda row, **kw: row,
        run_instance=run_instance,
        dycon_restore=lambda: None,
    )
    monkeypatch.setattr(harness, "install_harness_adapter", lambda *args: official)
    monkeypatch.setattr(harness, "NETWORK_RETRY_DELAY_SECONDS", 0)
    factory = SimpleNamespace(lock_images=lambda x: None, client=None, deadline=Deadline(), cleanup_owned=lambda x: None)
    result = harness.evaluate(cfg, Path(cfg["project_root"]) / "runs/run", "retry", factory=factory)
    outcome = result["outcomes"][rid]
    assert outcome["status"] == "completed" and outcome["resolved"] is True
    assert len(outcome["attempts"]) == 3


def test_harness_marks_garbage_patch_as_failed_resolution(cfg, monkeypatch):
    rid = import_fixture(cfg)
    cfg["selection"] = RunEvidence(cfg["project_root"], "source").select()
    official = SimpleNamespace(
        make_test_spec=lambda row, **kw: row,
        run_instance=lambda *args: {"completed": False, "resolved": False},
        dycon_restore=lambda: None,
    )
    monkeypatch.setattr(harness, "install_harness_adapter", lambda *args: official)
    monkeypatch.setattr(harness, "_evaluation_evidence", lambda *args: "Patch Apply Failed: Only garbage was found in the patch input")
    factory = SimpleNamespace(lock_images=lambda x: None, client=None, deadline=Deadline(), cleanup_owned=lambda x: None)
    result = harness.evaluate(cfg, Path(cfg["project_root"]) / "runs/run", "garbage", factory=factory)
    outcome = result["outcomes"][rid]
    assert outcome["status"] == "completed" and outcome["resolved"] is False
    assert outcome["failure_type"] == "invalid_patch"
    assert RunEvidence(cfg["project_root"], "run").read()["rollouts"][rid]["evaluation_status"] == "completed"


def test_harness_exhausted_network_error_uses_connection_error_status(cfg, monkeypatch):
    rid = import_fixture(cfg)
    cfg["selection"] = RunEvidence(cfg["project_root"], "source").select()
    official = SimpleNamespace(
        make_test_spec=lambda row, **kw: row,
        run_instance=lambda *args: (_ for _ in ()).throw(ConnectionError("reset")),
        dycon_restore=lambda: None,
    )
    monkeypatch.setattr(harness, "install_harness_adapter", lambda *args: official)
    monkeypatch.setattr(harness, "NETWORK_RETRY_DELAY_SECONDS", 0)
    factory = SimpleNamespace(lock_images=lambda x: None, client=None, deadline=Deadline(), cleanup_owned=lambda x: None)
    result = harness.evaluate(cfg, Path(cfg["project_root"]) / "runs/run", "exhausted", factory=factory)
    outcome = result["outcomes"][rid]
    assert outcome["status"] == "ConnectionError" and outcome["resolved"] is None
    assert outcome["failure_type"] == "network_error"
    assert len(outcome["attempts"]) == 3
    assert RunEvidence(cfg["project_root"], "run").read()["rollouts"][rid]["evaluation_status"] == "ConnectionError"
