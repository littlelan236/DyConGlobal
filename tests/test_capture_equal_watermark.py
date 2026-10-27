from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.config import CaptureExport
from core.disk_budget import DiskBudget


def test_equal_watermarks_are_valid():
    value = CaptureExport(spool_max_bytes=128 * 2**30,
                          spool_high_watermark=.95, spool_low_watermark=.95)
    assert value.spool_low_watermark == value.spool_high_watermark == .95


def test_inverted_watermarks_still_fail():
    with pytest.raises(ValidationError, match='low <= high'):
        CaptureExport(spool_high_watermark=.94, spool_low_watermark=.95)


def test_equal_threshold_does_not_wait_at_boundary(tmp_path, monkeypatch):
    @contextmanager
    def changes(path):
        def wait(seconds):
            raise AssertionError('Exact threshold must not wait')
        yield wait
    monkeypatch.setattr('core.locking.directory_changes', changes)
    budget = SimpleNamespace(limit=100, state=tmp_path / 'ledger', usage=lambda: 95)
    DiskBudget.admit(budget, .95, .95, 1)


def test_equal_threshold_releases_without_old_low_water_hysteresis(tmp_path, monkeypatch):
    levels = iter([96, 96, 94])
    calls = []
    @contextmanager
    def changes(path):
        yield lambda seconds: calls.append(seconds)
    monkeypatch.setattr('core.locking.directory_changes', changes)
    budget = SimpleNamespace(limit=100, state=tmp_path / 'ledger', usage=lambda: next(levels))
    DiskBudget.admit(budget, .95, .95, 1)
    assert len(calls) == 1
