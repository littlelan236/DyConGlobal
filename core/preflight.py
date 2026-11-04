"""Frozen SkillsBench startup policy and explicit, timed check receipts."""
import logging
import time
from contextlib import contextmanager

from core.backends import identity
from core.io import now

SKIP_REASON = "execution.preflight.long_checks=false"


def long_checks_enabled(cfg):
    if identity(cfg) != ("skillsbench", "openhands"):
        return True
    return cfg.get("execution", {}).get("preflight", {}).get("long_checks", True)


@contextmanager
def timed_check(receipts, name, *, enabled=True, reason=SKIP_REASON):
    started = time.monotonic()
    row = receipts.setdefault(name, {"status": "not_run", "seconds": 0.0, "calls": 0})
    row.update(status="running" if enabled else "skipped", started_at=now())
    if not enabled:
        row["reason"] = reason
    try:
        yield
    except BaseException as exc:
        row.update(status="failed", error=str(exc))
        raise
    else:
        row["status"] = "verified" if enabled else "skipped"
    finally:
        row["seconds"] += time.monotonic() - started
        row["calls"] += 1
        row["finished_at"] = now()
        logging.getLogger(__name__).info("startup preflight %s: %s (%.3fs)",
                                        name, row["status"], row["seconds"])
