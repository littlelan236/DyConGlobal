"""Opt-in real Qwen public-CLI data-parallel lifecycle acceptance."""
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest
import torch
import yaml

pytestmark = pytest.mark.skipif(os.environ.get("DYCON_GRPO_PARALLEL_GPU") != "1",
                               reason="requires explicitly selected idle physical GPUs")


def test_real_qwen_public_cli_parallel_and_exact_resume(tmp_path):
    from core.io import read_json, write_json
    from tests.test_grpo_lifecycle import assert_tree_close
    root = Path(__file__).parents[1]
    evidence = Path(os.environ.get("DYCON_GRPO_EVIDENCE", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    selected = [int(value) for value in os.environ["DYCON_GRPO_GPUS"].split(",")]
    assert len(selected) in {2, 4}
    raw = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    raw["run"]["mode"] = "smoke"
    raw["runtime"].update(gpu=None, gpus=selected, output_root=str(evidence / "artifacts"),
                          distributed_timeout_seconds=180)
    raw["preflight"] = {"check_resources": True, "verify_checkpoint": True}
    raw["grpo"].update(group_batch_size=4, prefix_reuse={"mode": "boundary_grad"})
    raw["training"].update(max_updates=2, checkpoint_every_updates=1)
    raw["training"]["memory"].update(profile_first_update=False, attention_policy="flash_only")
    label = "parallel-" + uuid.uuid4().hex[:10]
    timings = {}
    def training(name, config):
        config = copy.deepcopy(config)
        config["run"]["attempt_id"] = label + "-" + name
        file = evidence / (name + ".yaml")
        file.write_text(yaml.safe_dump(config))
        started = time.perf_counter()
        log = evidence / (name + ".log")
        with log.open("w") as output:
            result = subprocess.run([str(root / ".venvs/rl/bin/dycon"), "lora-train",
                                     "--training-config", str(file)], cwd=root, stdout=output,
                                     stderr=subprocess.STDOUT, timeout=420)
        assert result.returncode == 0, log.read_text()[-10000:]
        timings[name] = time.perf_counter() - started
        path = evidence / "artifacts/RL/checkpoints" / config["run"]["attempt_id"]
        with (evidence / (name + "-verify.log")).open("w") as output:
            result = subprocess.run([str(root / ".venvs/runner/bin/dycon"), "artifact", "verify",
                                     config["run"]["attempt_id"], "--scope", "lora-attempt",
                                     "--storage-root", config["runtime"]["output_root"]], cwd=root,
                                    stdout=output, stderr=subprocess.STDOUT, timeout=90)
        assert result.returncode == 0, (evidence / (name + "-verify.log")).read_text()[-7000:]
        print(name, "passed", round(timings[name], 2), flush=True)
        return path
    serial = copy.deepcopy(raw)
    serial["runtime"].update(gpu=selected[0], gpus=None)
    reused = os.environ.get("DYCON_GRPO_SERIAL_BASELINE")
    baseline = Path(reused) if reused else training("serial", serial)
    if reused:
        from RL.artifacts import verify_receipt
        from RL.config import Config, schedule_contract
        verify_receipt(baseline, "lora-attempt-v1")
        saved = read_json(baseline / "resolved-config.json")
        assert schedule_contract(Config.model_validate(saved)) == schedule_contract(Config.model_validate(serial))
    full = training("full", raw)
    paused_config = copy.deepcopy(raw)
    paused_config["validation"]["stop_after_update"] = 1
    paused = training("paused", paused_config)
    resumed_config = copy.deepcopy(raw)
    resumed_config["run"]["start"] = {"mode": "resume", "checkpoint": str(paused / "checkpoints/step-000001")}
    resumed = training("resumed", resumed_config)
    assert read_json(full / "final-adapter.json") == read_json(resumed / "final-adapter.json")
    def payload(path):
        return torch.load(path / "checkpoints/step-000002/training-state.pt", map_location="cpu", weights_only=True)
    a, b = payload(full), payload(resumed)
    for key in ["optimizer", "scheduler", "rng", "rank_rng"]:
        assert_tree_close(a[key], b[key])
    from safetensors.torch import load_file
    before = load_file(str(baseline / "adapter/adapter_model.safetensors"))
    actual = load_file(str(full / "adapter/adapter_model.safetensors"))
    # BF16 gradient accumulation has a different reduction order across ranks;
    # use a fixed absolute tolerance for the resulting adapter tensors.
    for key in before:
        torch.testing.assert_close(actual[key], before[key], atol=2e-4, rtol=2e-3, msg=key)
    report = read_json(full / "grpo-report.json")
    assert report["state"]["consumed_groups"] == 5
    assert report["updates"][-1]["parallel"]["ranks"][1]["group_count"] == 0
    write_json(evidence / "acceptance.json", {"gpus": selected, "model": raw["model"]["path"],
        "serial": str(baseline), "full": str(full), "paused": str(paused), "resumed": str(resumed),
        "state": report["state"], "timings": timings, "exact_resume": True,
        "serial_adapter_close": True, "real_long_context_capacity_verified": False})
