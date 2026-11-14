"""Read-only capture quota snapshot and optional local file inventory."""

import os
import math
import shutil
import sqlite3
import stat
import time
from contextlib import closing
from pathlib import Path

from core.io import identifier, now, read_json
from runtime.serving.spool import local_filesystem


def _reservations(project):
    state = project / "state/disk-budget.sqlite3"
    if state.exists():
        # Do not construct DiskBudget: its constructor initializes and reclaims.
        with closing(sqlite3.connect(state.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            rows = db.execute("""
                WITH captures AS (SELECT path,
                       CASE WHEN substr(key, 1, length(path) + 9) = 'capture:' || path || '/'
                            THEN substr(key, length(path) + 10) ELSE '' END AS relative,
                       bytes
                FROM reservations WHERE kind = 'capture')
                SELECT path, CASE WHEN instr(relative, '/') > 0
                                  THEN substr(relative, 1, instr(relative, '/') - 1) || '/'
                                  ELSE '' END AS run_prefix, SUM(bytes)
                FROM captures GROUP BY path, run_prefix
            """
            ).fetchall()
        source = "sqlite"
    else:
        legacy = project / "state/disk-budget.json"
        rows = []
        for key, value in (read_json(legacy) if legacy.exists() else {}).items():
            if value["kind"] == "capture":
                prefix = "capture:" + value["path"] + "/"
                rows.append((value["path"], key[len(prefix):] if key.startswith(prefix) else "", value["bytes"]))
        source = "legacy_json" if legacy.exists() else "absent"
    groups = {}
    for path, relative, amount in rows:
        run_id = relative.split("/", 1)[0] if "/" in relative else None
        key = (path, run_id)
        groups[key] = groups.get(key, 0) + amount
    return source, groups


def _disk(path):
    while not path.exists():
        if path.is_symlink():
            raise ValueError("Broken spool symlink: " + str(path))
        path = path.parent
    local_filesystem(path)
    usage = shutil.disk_usage(path)
    return {"device": path.stat().st_dev, "total_bytes": usage.total,
            "used_bytes": usage.used, "free_bytes": usage.free}


def _scan(paths, deadline):
    result = {"bytes": 0, "allocated_bytes": 0, "files": 0, "skipped_symlinks": 0,
              "complete": True, "errors": []}

    def error(exc):
        result["complete"] = False
        if len(result["errors"]) < 5:
            result["errors"].append(str(exc))

    # Resolve the spool base (relocation), but never follow child/run symlinks (NAS archives).
    for path, is_base in paths:
        if path.is_symlink() and not is_base:
            result["skipped_symlinks"] += 1
            continue
        if not path.exists():
            continue
        for base, directories, files in os.walk(path, followlinks=False, onerror=error):
            for name in [*directories, *files]:
                if time.monotonic() >= deadline:
                    result["complete"] = False
                    result["errors"].append("Scan timeout; byte counts are lower bounds")
                    return result
                try:
                    info = (Path(base) / name).lstat()
                    if stat.S_ISLNK(info.st_mode):
                        result["skipped_symlinks"] += 1
                    elif stat.S_ISREG(info.st_mode):
                        result["bytes"] += info.st_size
                        result["allocated_bytes"] += info.st_blocks * 512
                        result["files"] += 1
                except OSError as exc:
                    error(exc)
            directories[:] = [name for name in directories if not (Path(base) / name).is_symlink()]
    return result


def spool_usage(cfg, run_id=None, *, ledger_only=False, scan_seconds=10):
    if scan_seconds <= 0 or not math.isfinite(scan_seconds):
        raise ValueError("scan_seconds must be finite and positive")
    project = Path(cfg["project_root"]).resolve()
    if run_id is not None:
        run_id = identifier(run_id)
    options = cfg["capture"]["export"]
    configured = Path(os.path.abspath(project / options["spool_root"]))
    source, groups = _reservations(project)
    total = sum(groups.values())
    selected = sum(amount for (_, name), amount in groups.items() if name == run_id) if run_id else total
    limit = options["spool_max_bytes"]
    high, low = options["spool_high_watermark"], options["spool_low_watermark"]
    roots = sorted({configured, *(Path(path) for path, _ in groups)}, key=str)
    locations, scans = [], []
    # Deduplicate physical bases after a local storage relocation.
    seen = set()
    for root in roots:
        physical = root.resolve()
        if physical in seen:
            continue
        seen.add(physical)
        filesystem = _disk(root)
        target = root / run_id if run_id else root
        locations.append({"spool_root": str(root), "physical_spool_root": str(physical),
                          "filesystem": filesystem})
        scans.append((target, run_id is None))
    return {"schema_version": 1, "observed_at": now(), "run_id": run_id,
            "configuration_source": "run_snapshot" if run_id else "root_config",
            "ledger_source": source, "reserved_bytes": selected,
            "project_reserved_bytes": total,
            "unattributed_reserved_bytes": sum(v for (_, name), v in groups.items() if name is None),
            "limit_bytes": limit, "remaining_bytes": max(0, limit - total),
            "over_limit_bytes": max(0, total - limit), "usage_ratio": total / limit,
            "high_watermark_bytes": limit * high, "low_watermark_bytes": limit * low,
            "at_or_above_high_watermark": total >= limit * high,
            "locations": locations,
            "local_usage": None if ledger_only else _scan(scans, time.monotonic() + scan_seconds)}
