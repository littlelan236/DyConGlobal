import json
from pathlib import Path

import pytest

from application.execution.config_diff import compare_run_configs, diff_config_values
from application.execution.run_plan import create_run
from catalog.store import Catalog
from core.cli import main
from core.io import sha256
from tests.conftest import task


def make_source(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("config-diff-task")])
    return create_run(cfg, "config-diff-left", operation="rollout", task_ids=["config-diff-task"])


def test_diff_config_values_reports_added_removed_changed_and_null():
    rows = diff_config_values(
        {"explicit_null": None, "removed": 1, "nested": {"value": 1}},
        {"explicit_null": None, "added": None, "nested": {"value": 2}},
    )
    by_path = {row["path"]: row for row in rows}
    assert set(by_path) == {"/added", "/nested/value", "/removed"}
    assert by_path["/added"] == {
        "path": "/added", "kind": "added", "left_present": False,
        "left_value": None, "right_present": True, "right_value": None,
    }
    assert by_path["/removed"]["kind"] == "removed"
    assert by_path["/nested/value"]["left_value"] == 1
    assert by_path["/nested/value"]["right_value"] == 2


def test_compare_run_configs_reads_frozen_snapshots(cfg):
    left = make_source(cfg)
    right = create_run(cfg, "config-diff-right", operation="rollout", from_run=left.name,
                       clone_overrides=["serving.max_num_seqs=2"])
    report = compare_run_configs(cfg["project_root"], left.name, right.name)
    rows = {row["path"]: row for row in report["differences"]}
    assert report["difference_count"] == len(report["differences"])
    assert rows["/serving/max_num_seqs"]["left_value"] != rows["/serving/max_num_seqs"]["right_value"]
    assert report["left"]["resolved_config_sha256"] == sha256(left / "resolved-config.json")
    assert report["right"]["resolved_config_sha256"] == sha256(right / "resolved-config.json")


def test_cli_diff_config_json_markdown_and_output_no_overwrite(cfg, capsys):
    left = make_source(cfg)
    right = create_run(cfg, "config-diff-cli-right", operation="rollout", from_run=left.name,
                       clone_overrides=["serving.max_num_seqs=2"])
    output = Path(cfg["project_root"]) / "config-diff.md"
    args = ["--config", cfg["config_source"], "run", "diff-config", left.name, right.name,
            "--output", str(output)]
    main(args)
    rendered = capsys.readouterr().out
    assert "| /serving/max_num_seqs | changed |" in rendered
    assert output.read_text(encoding="utf-8") == rendered
    digest = sha256(output)
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert sha256(output) == digest

    main(["--config", cfg["config_source"], "run", "diff-config", left.name, right.name,
          "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert report["difference_count"] == len(report["differences"])
    assert any(row["path"] == "/serving/max_num_seqs" for row in report["differences"])


def test_cli_diff_config_uses_snapshots_not_mutable_root_yaml(cfg, capsys):
    left = make_source(cfg)
    right = create_run(cfg, "config-diff-broken-root-right", operation="rollout", from_run=left.name,
                       clone_overrides=["serving.max_num_seqs=2"])
    Path(cfg["config_source"]).write_text("invalid: [", encoding="utf-8")
    main(["--config", cfg["config_source"], "run", "diff-config", left.name, right.name,
          "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert any(row["path"] == "/serving/max_num_seqs" for row in report["differences"])
