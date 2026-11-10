"""Opt-in real single-GPU acceptance, using only the supported CLI in fresh processes.

Run: DYCON_LORA_GPU_TEST=1 .venvs/rl/bin/python -m pytest -q -s tests/test_lora_qwen35_integration.py
All outputs/configs/logs stay in the task temp directory or the configured /data3 attempt root.
"""

import copy
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from core.io import read_json, sha256, write_json

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("DYCON_LORA_GPU_TEST") != "1",
                                reason="Real 9B acceptance must be explicitly enabled")


def test_real_qwen35_fresh_resume_reload_warm_start_and_capacity():
    # The orchestrator imports no CUDA stack; only one child ever owns the model.
    import numpy as np
    from safetensors import safe_open

    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = ROOT / "temp/lora-foundation-20261110" / ("acceptance-" + stamp)
    work.mkdir(parents=True, exist_ok=False)
    template = yaml.safe_load((ROOT / "RL/experiments/lora-smoke.yaml").read_text())
    output_root = Path(template["runtime"]["output_root"])
    cli = str(ROOT / ".venvs/rl/bin/dycon")
    results = {}
    tolerances = {"adapter_atol": 2e-5, "adapter_rtol": 2e-3, "logits_atol": 0.125,
                  "logits_rtol": 0.01, "loss_atol": 0.05, "loss_rtol": 0.01}
    write_json(work / "acceptance-spec.json", {"tolerances": tolerances, "gpu": template["runtime"]["gpu"],
                                               "sequence_lengths": [512, 1024], "max_updates": 4})

    def execute(name, *, start="fresh", checkpoint=None, stop=None, updates=4, length=512, mode="smoke"):
        config = copy.deepcopy(template)
        attempt_id = f"lora-accept-{stamp}-{name}"
        config["run"].update(attempt_id=attempt_id, mode=mode, start=dict(mode=start, checkpoint=checkpoint))
        config["training"]["max_updates"] = updates
        config["validation"].update(stop_after_update=stop, max_sequence_length=length)
        path = work / (name + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        log = work / (name + ".log")
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_VERBOSITY"] = "warning"
        print(f"ACCEPTANCE start {name}: {attempt_id}", flush=True)
        with log.open("w") as handle:
            child = subprocess.run([cli, "lora-train", "--training-config", str(path)], cwd=ROOT,
                                   env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=1200)
        if child.returncode:
            tail = log.read_text(errors="replace")[-10000:]
            pytest.fail(f"{name} CLI failed with {child.returncode}:\n{tail}")
        directory = output_root / "RL/checkpoints" / attempt_id
        if mode == "dry-run":
            assert not directory.exists()
            print("ACCEPTANCE dry-run passed without attempt output", flush=True)
            return None
        with (work / (name + "-verify.log")).open("w") as handle:
            checked = subprocess.run([cli, "artifact", "verify", attempt_id, "--scope", "lora-attempt",
                                      "--storage-root", str(output_root)], cwd=ROOT, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT, timeout=120)
        assert checked.returncode == 0, (work / (name + "-verify.log")).read_text()
        validation = read_json(directory / "validation.json")
        attempt = read_json(directory / "attempt.json")
        assert validation["tolerances"] == tolerances
        assert validation["frozen_base_unchanged"] and validation["adapter_changed"]
        assert {row["category"] for row in validation["coverage"]} >= {
            "mlp", "full_attention", "linear_attention"}
        assert all(row["nonzero_B_gradient"] and row["adapter_changed"]
                   for row in validation["coverage"] if row["executed"])
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        assert len(metrics) == (stop or updates) - attempt["initial_state"]["global_step"]
        results[name] = {"path": str(directory), "state": validation["state"], "loading": validation["loading"],
                         "target_count": len(validation["coverage"]),
                         "executed_target_count": sum(row["executed"] for row in validation["coverage"]),
                         "peak_allocated_bytes": max(m["peak_allocated_bytes"] for m in metrics),
                         "peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in metrics),
                         "step_seconds": [m["seconds"] for m in metrics],
                         "losses": [m["loss"] for m in metrics],
                         "receipt_sha256": sha256(directory / "receipt.json")}
        write_json(work / "progress.json", results)
        print(f"ACCEPTANCE passed {name}: step={validation['state']['global_step']} "
              f"peak={results[name]['peak_allocated_bytes']/2**30:.2f}GiB", flush=True)
        return directory

    execute("dry", mode="dry-run")
    baseline = execute("baseline")
    paused = execute("paused", stop=2)
    assert read_json(baseline / "initial-adapter.json") == read_json(paused / "initial-adapter.json")
    with safe_open(baseline / "checkpoints/step-000002/adapter/adapter_model.safetensors", framework="numpy") as left, \
            safe_open(paused / "checkpoints/step-000002/adapter/adapter_model.safetensors", framework="numpy") as right:
        for key in left.keys():
            np.testing.assert_allclose(left.get_tensor(key), right.get_tensor(key),
                                       atol=tolerances["adapter_atol"], rtol=tolerances["adapter_rtol"])
    resumed = execute("resumed", start="resume", checkpoint=str(paused / "checkpoints/step-000002"))
    # Predeclared bf16 adapter tolerances; report the measured maximum difference.
    max_difference = 0.0
    with safe_open(baseline / "adapter/adapter_model.safetensors", framework="numpy") as left, \
            safe_open(resumed / "adapter/adapter_model.safetensors", framework="numpy") as right:
        assert left.keys() == right.keys()
        for key in left.keys():
            a, b = left.get_tensor(key), right.get_tensor(key)
            np.testing.assert_allclose(a, b, atol=tolerances["adapter_atol"], rtol=tolerances["adapter_rtol"])
            max_difference = max(max_difference, float(np.max(np.abs(a - b))))
    base_metrics = [json.loads(line) for line in (baseline / "metrics.jsonl").read_text().splitlines()]
    resume_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert read_json(baseline / "validation.json")["state"] == read_json(resumed / "validation.json")["state"]
    for a, b in zip(base_metrics[2:], resume_metrics, strict=True):
        assert a["consumed_microbatches"] == b["consumed_microbatches"]
        assert a["consumed_tokens"] == b["consumed_tokens"]
        assert a["learning_rate"] == b["learning_rate"]
        np.testing.assert_allclose(a["loss"], b["loss"], atol=tolerances["loss_atol"], rtol=tolerances["loss_rtol"])
    warm = execute("warm", start="warm_start", checkpoint=str(baseline / "adapter"), updates=1)
    assert read_json(warm / "initial-adapter.json") == read_json(baseline / "final-adapter.json")
    warm_validation = read_json(warm / "validation.json")
    assert warm_validation["initial_optimizer_entries"] == 0
    assert all(value == 0 for value in warm_validation["initial_state"].values())
    # Independent process reload was performed before the warm-start update.
    import torch
    torch.testing.assert_close(torch.load(warm / "initial-probe.pt", weights_only=True),
                               torch.load(baseline / "final-probe.pt", weights_only=True),
                               atol=tolerances["logits_atol"], rtol=tolerances["logits_rtol"])
    execute("capacity-1k", updates=1, length=1024)
    used = subprocess.check_output(["nvidia-smi", "-i", str(template["runtime"]["gpu"]),
                                     "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    text=True, timeout=15).strip()
    assert int(used) <= 512, f"Test GPU still occupied after child exit: {used} MiB"
    report = {"status": "passed", "tolerances": tolerances, "results": results,
              "resume_adapter_max_abs_difference": max_difference,
              "gpu_memory_after_MiB": int(used), "new_process_reload_verified": True,
              "warm_start_reset_verified": True}
    write_json(work / "report.json", report)
    write_json(ROOT / "temp/lora-foundation-20261110/latest-acceptance.json", {"report": str(work / "report.json")})
    print(f"ACCEPTANCE REPORT {work / 'report.json'}", flush=True)
