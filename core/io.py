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


def artifact_roots(cfg, kind):
    """Return configured physical roots, preserving legacy project-root behavior."""
    paths = cfg.get("paths", {})
    roots = paths.get("artifact_scan_roots") or [cfg.get("project_root", ".")]
    preferred = paths.get("artifact_roots", {}).get(kind)
    values = [preferred] if preferred else []
    values.extend(roots)
    result = []
    for value in values:
        raw = Path(value).expanduser()
        if not raw.is_absolute() or raw == Path("/") or ".." in raw.parts:
            raise ValueError(f"Artifact root must be an absolute non-root path: {value!r}")
        path = raw.resolve()
        if path not in result:
            result.append(path)
    return result


def run_path(cfg_or_root, run_id):
    """Resolve a run through its frozen physical root or configured scan roots."""
    run_id = identifier(run_id)
    if isinstance(cfg_or_root, dict):
        frozen = cfg_or_root.get("run_storage_root")
        if frozen:
            return Path(frozen).resolve() / "runs" / run_id
        roots = artifact_roots(cfg_or_root, "run")
    else:
        roots = [Path(cfg_or_root).resolve()]
        config_path = roots[0] / "config.yaml"
        if config_path.exists():
            try:
                import yaml
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                paths = raw.get("paths", {})
                configured = list(paths.get("artifact_scan_roots", []))
                preferred = paths.get("artifact_roots", {}).get("run")
                if preferred:
                    configured.insert(0, preferred)
                roots = [Path(value).expanduser().resolve() for value in [*configured, *roots]]
            except (OSError, ValueError, yaml.YAMLError):
                pass
        roots = list(dict.fromkeys(roots))
    matches = [root / "runs" / run_id for root in roots if (root / "runs" / run_id).exists()]
    if len(matches) > 1:
        identities = set()
        for match in matches:
            plan = match / "plan.json"
            snapshot = match / "resolved-config.json"
            identities.add((sha256(plan) if plan.exists() else None,
                            sha256(snapshot) if snapshot.exists() else None))
        if len(identities) > 1:
            raise ValueError(f"Conflicting run copies found across storage roots: {run_id}")
    if matches:
        return matches[0]
    return roots[0] / "runs" / run_id


def cache_roots(cfg):
    return [root / "difficulty" / "cache" for root in artifact_roots(cfg, "difficulty_cache")]


def owned_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Artifact path escapes project: {relative}")
    return path


def artifact_dir(cfg, module, artifact_id, category="outputs"):
    if module not in {"enrichment", "hidden", "difficulty", "action", "runtime"}:
        raise ValueError(module)
    if module in {"difficulty", "action"}:
        return owned_path(cfg["project_root"], f"{module}/{category}/{identifier(artifact_id)}")
    if cfg.get("run_id") and module in {"enrichment", "hidden", "difficulty", "action"}:
        base = f"runs/{identifier(cfg['run_id'])}/{module}"
        if module == "enrichment":
            rel = "offline" if category == "outputs" else category
            return owned_path(cfg["project_root"], f"{base}/{rel}")
        return owned_path(cfg["project_root"], f"{base}/{category}/{identifier(artifact_id)}")
    return owned_path(cfg["project_root"], f"{module}/{category}/{identifier(artifact_id)}")


def update_status(path, stage, **values):
    write_json(path, {"stage": stage, "updated_at": now(), **values})
