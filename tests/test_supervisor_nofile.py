import sys
from types import SimpleNamespace

import pytest


def test_supervisor_nofile_limit_is_applied(monkeypatch):
    from application.execution import supervisor

    calls = []
    limits = [1024, 1048576]
    def getrlimit(resource):
        return tuple(limits)
    def setrlimit(resource, new_limits):
        calls.append((resource, new_limits))
        limits[:] = new_limits
    fake = SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=getrlimit,
        setrlimit=setrlimit,
    )
    monkeypatch.setitem(sys.modules, "resource", fake)
    result = supervisor._apply_open_file_limit({"execution": {"supervisor_max_open_files": 262144}})

    assert calls == [(7, (262144, 1048576))]
    assert result["configured_max_open_files"] == 262144
    assert result["soft_limit"] == 262144


def test_supervisor_nofile_limit_cannot_exceed_hard_limit(monkeypatch):
    from application.execution import supervisor

    fake = SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda resource: (1024, 65536),
        setrlimit=lambda resource, limits: None,
    )
    monkeypatch.setitem(sys.modules, "resource", fake)
    with pytest.raises(RuntimeError, match="exceeds hard RLIMIT_NOFILE"):
        supervisor._apply_open_file_limit({"execution": {"supervisor_max_open_files": 262144}})
