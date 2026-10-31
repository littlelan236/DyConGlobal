"""Workspace-to-patch export and validation for native SWE-bench attempts."""
from __future__ import annotations

import re
from pathlib import Path

from core.io import write_json

_PATH = re.compile(r"^(?:a|b)/(.*)$")


def _sha256_bytes(value: bytes):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def validate_diff(diff: str, *, max_bytes=16 * 1024 * 1024):
    if not isinstance(diff, str):
        raise ValueError("git diff output must be text")
    raw = diff.encode("utf-8")
    if len(raw) > max_bytes:
        raise ValueError("workspace patch exceeds configured size limit")
    paths = []
    for line in diff.splitlines():
        if line.startswith(("+++ ", "--- ")):
            value = line[4:].split("\t", 1)[0]
            if value == "/dev/null":
                continue
            match = _PATH.match(value)
            if not match or match.group(1).startswith("/") or ".." in Path(match.group(1)).parts:
                raise ValueError("workspace patch contains an unsafe path")
            paths.append(match.group(1))
    if not diff.strip():
        return {"empty": True, "paths": [], "sha256": _sha256_bytes(raw), "size": 0}
    if not paths:
        raise ValueError("workspace patch has no file headers")
    return {"empty": False, "paths": sorted(set(paths)), "sha256": _sha256_bytes(raw), "size": len(raw)}


def publish_patch(output, diff: str, *, binding: dict, base_commit: str | None,
                  workspace="/testbed", max_bytes=16 * 1024 * 1024):
    facts = validate_diff(diff, max_bytes=max_bytes)
    patch = Path(output) / "patch.diff"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(diff, encoding="utf-8")
    receipt = {"schema_version": 1, "binding": binding, "base_commit": base_commit,
               "workspace": workspace, "sha256": facts["sha256"], "size": facts["size"],
               "empty": facts["empty"], "paths": facts["paths"],
               "status": "unsubmitted" if facts["empty"] else "submitted"}
    write_json(Path(output) / "patch-binding.json", receipt)
    return patch, receipt
