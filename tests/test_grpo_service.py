"""Public service lifecycle using an explicitly tiny model in tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from core.io import hash_json, read_json, write_json
from RL.artifacts import verify
from RL.config import attempt_directory
from RL.service import run, _train_grpo
from tests.lora_helpers import LORA, model_and_targets


def test_public_service_publishes_reconstructible_grpo_receipt(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    raw = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    raw["run"]["attempt_id"] = "public-grpo-tiny"
    raw["run"]["mode"] = "smoke"
    raw["runtime"]["output_root"] = str(tmp_path / "artifacts")
    raw["lora"] = LORA.model_dump()
    file = tmp_path / "config.yaml"; file.write_text(yaml.safe_dump(raw))
    base = {"files": {"tiny_fixture": True}}
    base["fingerprint"] = hash_json(base["files"])
    monkeypatch.setattr("RL.service._preflight", lambda *args: base)
    def tiny_training(config, root, directory, base, parent, attempt):
        model, targets = model_and_targets()
        write_json(directory / "environment.json", {"tiny_fixture": True})
        write_json(directory / "trainable-parameters.json", {"targets": targets, "target_count": len(targets),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        return _train_grpo(config, root, directory, base, parent, attempt, model, targets,
                           optimizer_settings=config.training)
    monkeypatch.setattr("RL.service._train", tiny_training)
    result = run({"project_root": str(root)}, file)
    assert result["status"] == "completed"
    verified = verify({"project_root": str(root)}, "lora-attempt", result["attempt_id"],
                      storage_root=raw["runtime"]["output_root"])
    assert verified["verified"]
    path = Path(result["path"])
    report = read_json(path / "grpo-report.json")
    assert report["state"]["consumed_groups"] == 5
    assert report["updates"][-1]["group_count"] == 1
    assert len(report["updates"][-1]["ratio_quantiles"]) == 5
    with pytest.raises(FileExistsError):
        run({"project_root": str(root)}, file)


def test_dry_run_imports_no_torch_and_creates_no_attempt(tmp_path):
    root = Path(__file__).parents[1]
    raw = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    raw["runtime"]["output_root"] = str(tmp_path)
    file = tmp_path / "dry.yaml"; file.write_text(yaml.safe_dump(raw))
    script = '''
import importlib.abc, sys
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == "torch" or fullname.startswith("torch."):
   raise AssertionError("dry-run imported torch")
sys.meta_path.insert(0, NoTorch())
from RL.service import run
import RL.service as service
service.memory_plan=lambda c: {}
service.base_identity=lambda *args: {"fingerprint":"test"}
r=run({"project_root":sys.argv[1]},sys.argv[2])
assert r["status"]=="validated" and r["dataset"]["group_count"]==5
assert "torch" not in sys.modules and not Path(r["output"]).exists()
print("dry-run validated: no torch, no model load, no attempt")
'''
    output = subprocess.run([sys.executable, "-c", script, str(root), str(file)],
                            cwd=root, check=True, capture_output=True, text=True, timeout=30)
    assert "no torch" in output.stdout
