"""Receipt replacement must be reversible and cannot silently change selection."""

import json

import pytest

from core.io import sha256
from runtime.agents.pddl_regrading import _transaction


def test_exact_byte_backups_and_replacement(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    a, b = run / "a.json", run / "b.json"
    a.write_bytes(b'{"reward":0}\r\n')
    before = a.read_bytes()
    rows = _transaction(run, run / "regrades/id", {a: b'{"reward":1}\n', b: b"new"})
    assert a.read_bytes() == b'{"reward":1}\n' and b.read_bytes() == b"new"
    assert __import__("pathlib").Path(rows[0]["backup"]).read_bytes() == before
    assert all(sha256(r["path"]) == r["after_sha256"] for r in rows)


def test_failure_rolls_back_modified_and_new_files(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    a, b = run / "a.json", run / "b.json"
    a.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="injected"):
        _transaction(run, run / "regrades/id", {a: b"changed", b: b"new"}, fail_after=2)
    assert a.read_bytes() == b"original" and not b.exists()
    assert json.loads((run / "regrades/id/replacement-state.json").read_text())["phase"] == "rolled_back"


def test_refuses_file_outside_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    external = tmp_path / "outside.json"
    external.write_bytes(b"original")
    with pytest.raises(ValueError, match="escaped"):
        _transaction(run, run / "regrades/id", {external: b"changed"})
    assert external.read_bytes() == b"original"


def test_existing_backup_cannot_be_replaced(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    a = run / "a"
    a.write_bytes(b"old")
    _transaction(run, run / "regrades/id", {a: b"new"})
    with pytest.raises(FileExistsError):
        _transaction(run, run / "regrades/id", {a: b"newer"})
    assert a.read_bytes() == b"new"


def test_cli_requires_explicit_replacement_and_uses_frozen_run(monkeypatch, tmp_path, capsys):
    import core.cli
    import runtime.agents.pddl_regrading

    calls = []
    monkeypatch.setattr(core.cli, "resolve", lambda *a, **k: pytest.fail("mutable config loaded"))
    monkeypatch.setattr(runtime.agents.pddl_regrading, "regrade", lambda *a, **k:
                        calls.append((a, k)) or {"valid_executions": 14, "sample_count": 14})
    core.cli.main(["--config", str(tmp_path / "config.yaml"), "skillsbench", "regrade-pddl", "run", "eval",
        "--snapshot", "snapshot.json", "--samples", "samples.json", "--replace-original-with-backup"])
    assert len(calls) == 1 and calls[0][1]["replace"] is True
    assert '"valid_executions": 14' in capsys.readouterr().out
