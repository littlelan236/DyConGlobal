import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from core.io import read_json, write_json
from RL.artifacts import commit_receipt, verify_receipt
from RL.config import attempt_directory, load
from RL.service import run

ROOT = Path(__file__).parents[1]


def test_cli_help_and_light_imports():
    command = """
import sys
from core.cli import main
try:
    main(['lora-train', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules
assert 'peft' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", command], cwd=ROOT, check=True, timeout=30, capture_output=True)


def test_existing_attempt_and_bad_source_fail_without_loading_model(tmp_path):
    values = yaml.safe_load((ROOT / "RL/experiments/lora-smoke.yaml").read_text())
    values["runtime"]["output_root"] = str(tmp_path)
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(values))
    config, _ = load(source, ROOT)
    directory = attempt_directory(config)
    directory.mkdir(parents=True)
    write_json(directory / "attempt.json", {"status": "failed"})
    with pytest.raises(FileExistsError, match="already exists"):
        run({"project_root": str(ROOT)}, source)
    values["run"]["attempt_id"] = "second"
    values["run"]["start"] = dict(mode="resume", checkpoint=str(directory))
    source.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="complete checkpoint"):
        run({"project_root": str(ROOT)}, source)
    assert json.loads((directory / "attempt.json").read_text()) == {"status": "failed"}


def test_receipt_binds_all_files_and_rejects_symlinks(tmp_path):
    write_json(tmp_path / "state.json", {"status": "ok"})
    commit_receipt(tmp_path, "lora-test-v1")
    verify_receipt(tmp_path)
    (tmp_path / "extra").write_text("unregistered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_receipt(tmp_path)
    (tmp_path / "extra").unlink()
    (tmp_path / "alias").symlink_to(tmp_path / "state.json")
    with pytest.raises(ValueError, match="Symlinks"):
        verify_receipt(tmp_path)


def test_runner_does_not_need_torch():
    runner = ROOT / ".venvs/runner/bin/python"
    if not runner.exists():
        pytest.skip("Remote runner environment is not present")
    command = """
import importlib.util
from core.cli import main
assert importlib.util.find_spec('torch') is None
try:
    main(['lora-train', '--help'])
except SystemExit as exc:
    assert exc.code == 0
"""
    subprocess.run([str(runner), "-c", command], cwd=ROOT, env=os.environ.copy(),
                   check=True, timeout=30, capture_output=True)


def test_initialization_failure_retains_failure_state_without_receipt(tmp_path):
    values = yaml.safe_load((ROOT / "RL/experiments/lora-smoke.yaml").read_text())
    values["runtime"]["output_root"] = str(tmp_path)
    values["model"]["path"] = str(tmp_path / "missing-model")
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(values))
    with pytest.raises(OSError):
        run({"project_root": str(ROOT)}, source)
    directory = tmp_path / "RL/checkpoints" / values["run"]["attempt_id"]
    assert read_json(directory / "attempt.json")["status"] == "failed"
    assert (directory / "traceback.txt").is_file()
    assert not (directory / "receipt.json").exists()
