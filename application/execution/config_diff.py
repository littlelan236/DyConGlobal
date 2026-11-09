"""Read-only comparison of complete frozen run configurations."""
from __future__ import annotations

import json
from pathlib import Path

from application.execution.run_plan import load_plan
from core.io import read_json, sha256


def _pointer(parts):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def diff_config_values(left, right):
    """Return every leaf-level difference, preserving missing vs explicit null."""
    rows = []

    def visit(left_present, left_value, right_present, right_value, parts):
        if left_present and right_present and isinstance(left_value, dict) and isinstance(right_value, dict):
            keys = sorted(set(left_value) | set(right_value), key=str)
            if not keys and left_value == right_value:
                return
            for key in keys:
                visit(key in left_value, left_value.get(key),
                      key in right_value, right_value.get(key), [*parts, key])
            return
        if left_present != right_present:
            present_value = left_value if left_present else right_value
            if isinstance(present_value, dict) and present_value:
                for key in sorted(present_value, key=str):
                    visit(key in left_value if left_present else False,
                          left_value.get(key) if left_present else None,
                          key in right_value if right_present else False,
                          right_value.get(key) if right_present else None,
                          [*parts, key])
                return
        if left_present and right_present and left_value == right_value:
            return
        path = _pointer(parts) if parts else "/"
        kind = ("added" if right_present else "removed") if left_present != right_present else "changed"
        rows.append({"path": path, "kind": kind,
                     "left_present": left_present, "left_value": left_value if left_present else None,
                     "right_present": right_present, "right_value": right_value if right_present else None})

    visit(True, left, True, right, [])
    return sorted(rows, key=lambda row: row["path"])


def _leaf_paths(value):
    if isinstance(value, dict) and value:
        for child in value.values():
            yield from _leaf_paths(child)
    else:
        yield value


def compare_run_configs(project_root, left_run_id, right_run_id):
    """Validate and compare two immutable resolved-config snapshots."""
    root = Path(project_root).resolve()
    loaded = [load_plan(root, run_id) for run_id in (left_run_id, right_run_id)]
    configs, metadata = [], []
    for run_id, item in zip((left_run_id, right_run_id), loaded):
        path = item["run"] / "resolved-config.json"
        config = read_json(path)
        configs.append(config)
        metadata.append({
            "run_id": run_id,
            "resolved_config_path": str(path),
            "resolved_config_sha256": sha256(path),
            "config_hash": config.get("config_hash"),
            "field_count": sum(1 for _ in _leaf_paths(config)),
        })
    differences = diff_config_values(*configs)
    return {"schema_version": 1, "left": metadata[0], "right": metadata[1],
            "difference_count": len(differences), "differences": differences}


def _markdown_cell(present, value):
    if not present:
        return "<missing>"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).replace("|", "\\|")


def format_config_diff_markdown(report):
    left, right = report["left"], report["right"]
    lines = [
        "# Frozen run configuration diff",
        "",
        f"- Left: `{left['run_id']}` (`{left['config_hash']}`)",
        f"- Right: `{right['run_id']}` (`{right['config_hash']}`)",
        f"- Differences: {report['difference_count']}",
        "",
        "| JSON Pointer | Change | Left value | Right value |",
        "|---|---|---|---|",
    ]
    for row in report["differences"]:
        path = row["path"].replace("|", "\\|")
        lines.append("| " + " | ".join((
            path, row["kind"],
            _markdown_cell(row["left_present"], row["left_value"]),
            _markdown_cell(row["right_present"], row["right_value"]),
        )) + " |")
    return "\n".join(lines)
