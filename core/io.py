"""Atomic artifact operations (adapted from tactRefractor artifacts.py)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".write-", delete=False
        ) as f:
            tmp = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", value):
        raise ValueError(f"Invalid artifact identifier: {value!r}")
    return value


def owned_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Artifact path escapes project: {relative}")
    return path


def artifact_dir(cfg, module, artifact_id, category="outputs"):
    if module not in {"enrichment", "hidden", "difficulty", "action", "runtime"}:
        raise ValueError(module)
    if module == "difficulty":
        return owned_path(cfg["project_root"], f"difficulty/{category}/{identifier(artifact_id)}")
    if cfg.get("run_id") and module in {"enrichment", "hidden", "difficulty", "action"}:
        base = f"runs/{identifier(cfg['run_id'])}/{module}"
        if module == "enrichment":
            rel = "offline" if category == "outputs" else category
            return owned_path(cfg["project_root"], f"{base}/{rel}")
        return owned_path(cfg["project_root"], f"{base}/{category}/{identifier(artifact_id)}")
    return owned_path(cfg["project_root"], f"{module}/{category}/{identifier(artifact_id)}")


def update_status(path, stage, **values):
    write_json(path, {"stage": stage, "updated_at": now(), **values})
