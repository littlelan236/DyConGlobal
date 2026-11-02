"""Code identity uses the available host Git without rewriting past receipts."""
import shutil
import subprocess

import pytest

from application.execution.run_plan import _git_identity


@pytest.mark.parametrize("executable", ["/usr/bin/git", "C:/Program Files/Git/cmd/git.exe"])
def test_code_identity_uses_available_git(monkeypatch, tmp_path, executable):
    monkeypatch.setattr(shutil, "which", lambda name: executable if name == "git" else None)
    calls = []

    def output(argv, text):
        calls.append(argv)
        assert text is True
        return "a" * 40 + "\n" if argv[-2:] == ["rev-parse", "HEAD"] else " M file.py\n"

    monkeypatch.setattr(subprocess, "check_output", output)
    result = _git_identity(tmp_path)
    assert result["head"] == "a" * 40 and result["dirty"] is True
    assert len(result["dirty_sha256"]) == 64
    assert len(calls) == 2 and all(argv[0] == executable for argv in calls)


def test_code_identity_keeps_unavailability_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def missing(*args, **kwargs):
        raise FileNotFoundError("Git is unavailable")

    monkeypatch.setattr(subprocess, "check_output", missing)
    assert _git_identity(tmp_path) == {"unavailable": "Git is unavailable"}
