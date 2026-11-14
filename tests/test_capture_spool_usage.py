import json
import sqlite3

import pytest

from core.disk_budget import DiskBudget
from core.io import write_json
from hidden.spool_usage import spool_usage


def configuration(tmp_path):
    return {"project_root": str(tmp_path), "capture": {"export": {
        "spool_root": "spool", "spool_max_bytes": 1000,
        "spool_high_watermark": 0.8, "spool_low_watermark": 0.6}}}


def test_empty_query_creates_nothing(tmp_path):
    result = spool_usage(configuration(tmp_path))
    assert result["ledger_source"] == "absent"
    assert result["reserved_bytes"] == 0
    assert result["remaining_bytes"] == 1000
    assert result["local_usage"]["bytes"] == 0
    assert result["local_usage"]["complete"]
    assert list(tmp_path.iterdir()) == []


def test_project_quota_run_filter_and_no_reclamation(tmp_path):
    cfg = configuration(tmp_path)
    base = tmp_path / "spool"
    budget = DiskBudget(tmp_path, base, 1000)
    budget.acquire("capture:" + str(base / "one/data"), 300)
    budget.acquire("capture:" + str(base / "two/data"), 400)
    budget.acquire("queue:pending", 150)
    budget.acquire("image:transient", 99, kind="image")
    with sqlite3.connect(budget.state) as db:
        db.execute("UPDATE reservations SET pid=99999999")
    before = budget.allocations()
    (base / "one").mkdir()
    (base / "one/data").write_bytes(b"abc")
    result = spool_usage(cfg, "one")
    assert result["reserved_bytes"] == 300
    assert result["project_reserved_bytes"] == 850
    assert result["unattributed_reserved_bytes"] == 150
    assert result["remaining_bytes"] == 150
    assert result["usage_ratio"] == 0.85
    assert result["at_or_above_high_watermark"]
    assert result["local_usage"]["bytes"] == 3
    assert budget.allocations() == before
    cfg["capture"]["export"]["spool_max_bytes"] = 800
    result = spool_usage(cfg, ledger_only=True)
    assert result["remaining_bytes"] == 0
    assert result["over_limit_bytes"] == 50
    assert result["local_usage"] is None


def test_relocation_follows_base_but_skips_archive_and_file_links(tmp_path):
    cfg = configuration(tmp_path)
    physical = tmp_path / "physical"
    physical.mkdir()
    (tmp_path / "spool").symlink_to(physical, target_is_directory=True)
    (physical / "run").mkdir()
    (physical / "run/local").write_bytes(b"local")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "large").write_bytes(b"a" * 100)
    (physical / "archived").symlink_to(archive, target_is_directory=True)
    (physical / "run/link").symlink_to(archive / "large")
    result = spool_usage(cfg)
    assert result["locations"][0]["physical_spool_root"] == str(physical)
    assert result["local_usage"]["bytes"] == 5
    assert result["local_usage"]["skipped_symlinks"] == 2
    archived = spool_usage(cfg, "archived")
    assert archived["local_usage"]["bytes"] == 0
    assert archived["local_usage"]["skipped_symlinks"] == 1


def test_legacy_and_other_spool_roots(tmp_path):
    cfg = configuration(tmp_path)
    other = tmp_path / "old-spool"
    (other / "old").mkdir(parents=True)
    (other / "old/data").write_bytes(b"four")
    rows = {"capture:" + str(other / "old/data"): {
        "bytes": 40, "kind": "capture", "path": str(other), "device": 1, "pid": 2}}
    write_json(tmp_path / "state/disk-budget.json", rows)
    result = spool_usage(cfg)
    assert result["ledger_source"] == "legacy_json"
    assert result["reserved_bytes"] == 40
    assert result["local_usage"]["bytes"] == 4
    assert len(result["locations"]) == 2
    assert not (tmp_path / "state/disk-budget.sqlite3").exists()


def test_scan_timeout_and_bad_parameters(tmp_path):
    cfg = configuration(tmp_path)
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool/file").write_bytes(b"data")
    result = spool_usage(cfg, scan_seconds=1e-12)
    assert not result["local_usage"]["complete"]
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            spool_usage(cfg, scan_seconds=value)
    with pytest.raises(ValueError):
        spool_usage(cfg, "../escape")


def test_cli_global_and_frozen_run(tmp_path, monkeypatch, capsys):
    from core.cli import main
    import application.execution.run_plan as plans
    import core.cli as cli
    cfg = configuration(tmp_path)
    monkeypatch.setattr(cli, "resolve", lambda *args: cfg)
    main(["capture", "spool-usage", "--ledger-only"])
    assert json.loads(capsys.readouterr().out)["configuration_source"] == "root_config"
    frozen = configuration(tmp_path)
    frozen["capture"]["export"]["spool_max_bytes"] = 2000
    def load(project, run):
        assert run == "one"
        return {"config": frozen}
    monkeypatch.setattr(plans, "load_plan", load)
    main(["--config", str(tmp_path / "config.yaml"), "capture", "spool-usage", "one"])
    result = json.loads(capsys.readouterr().out)
    assert result["configuration_source"] == "run_snapshot"
    assert result["limit_bytes"] == 2000
    for args in (["--set", "capture.enabled=true"], ["--run-id", "one"]):
        with pytest.raises(SystemExit) as exc:
            main([*args, "capture", "spool-usage"])
        assert exc.value.code == 2
