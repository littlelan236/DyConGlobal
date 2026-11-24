import io
import json
import subprocess

import pytest

from runtime.containers.policy import PROBE_CODE, parse_probe


def test_probe_matches_distribution_names_without_ignoring_bad_versions(monkeypatch, capsys):
    packages = [
        {"name": "zope.interface", "version": "7.2"},
        {"name": "My._-Distribution", "version": "2.0"},
    ]
    monkeypatch.setattr(subprocess, "check_output", lambda args: json.dumps(packages).encode())
    constraints = "zope-interface==7.2\nmy-distribution==2.0\nzope_interface==7.1\nmissing==1\n"
    with pytest.raises(SystemExit) as result:
        exec(PROBE_CODE, {"open": lambda path: io.StringIO(constraints)})

    assert result.value.code == 3
    checks = parse_probe(capsys.readouterr().out)["constraints"]
    assert [(check["actual"], check["satisfied"]) for check in checks] == [
        ("7.2", True),
        ("2.0", True),
        ("7.2", False),
        (None, False),
    ]
