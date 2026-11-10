"""Opt-in real Qwen3.5 validation through the formal training CLI.

DYCON_LORA_LONG_GPU_TEST=1 .venvs/rl/bin/python -m pytest -q -s tests/test_lora_long_context_gpu.py
No temporary training launcher; test cases freeze YAML and inspect registered artifacts.
"""

import copy
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from core.io import read_json, write_json

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("DYCON_LORA_LONG_GPU_TEST") != "1",
                                reason="Real long-context GPU validation is opt-in")
TOLERANCES = dict(adapter_atol=2e-5, adapter_rtol=2e-3, logits_atol=0.125,
                  logits_rtol=0.01, loss_atol=0.05, loss_rtol=0.01)


class Cases:
    def __init__(self, name):
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.work = ROOT / "temp/lora-long-context-20261110" / (name + "-" + self.stamp)
        self.work.mkdir(parents=True, exist_ok=False)
        self.template = yaml.safe_load((ROOT / "RL/experiments/lora-smoke.yaml").read_text())
        self.output = Path(self.template["runtime"]["output_root"]) / "RL/checkpoints"
        self.results = {}
        write_json(self.work / "spec.json", dict(tolerances=TOLERANCES, safety_margin_bytes=15 * 2**30,
                                                 minimum_capacity_updates=2, synthetic_only=True))

    def execute(self, name, memory, *, length=2048, policy="layer", updates=2,
                stop=None, start="fresh", checkpoint=None):
        config = copy.deepcopy(self.template)
        config["run"].update(attempt_id=f"lora-long-{self.stamp}-{name}",
                              start=dict(mode=start, checkpoint=str(checkpoint) if checkpoint else None))
        config["training"].update(max_updates=updates, gradient_accumulation_steps=1,
                                   checkpoint_policy=policy, memory=memory)
        config["validation"].update(max_sequence_length=length, stop_after_update=stop)
        path = self.work / (name + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        cli = str(ROOT / ".venvs/rl/bin/dycon")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        print(f"LONG start {name}: length={length} policy={policy}", flush=True)
        with (self.work / (name + ".log")).open("w") as log:
            child = subprocess.run([cli, "lora-train", "--training-config", str(path)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if child.returncode:
            pytest.fail((self.work / (name + ".log")).read_text(errors="replace")[-9000:])
        directory = self.output / config["run"]["attempt_id"]
        with (self.work / (name + "-verify.log")).open("w") as log:
            check = subprocess.run([cli, "artifact", "verify", config["run"]["attempt_id"],
                                    "--scope", "lora-attempt", "--storage-root", str(self.output.parents[1])],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        assert check.returncode == 0, (self.work / (name + "-verify.log")).read_text()
        validation = read_json(directory / "validation.json")
        assert validation["frozen_base_unchanged"] and validation["adapter_changed"]
        assert len(validation["coverage"]) == 248
        assert all(row["executed"] and row["nonzero_B_gradient"] and row["adapter_changed"]
                   for row in validation["coverage"])
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        result = {"path": str(directory), "length": length, "policy": policy, "memory": memory,
                  "metrics": metrics, "peak_allocated_bytes": max(m["peak_allocated_bytes"] for m in metrics),
                  "peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in metrics),
                  "median_update_seconds": statistics.median(m["seconds"] for m in metrics),
                  "unprofiled_update_seconds": [m["seconds"] for m in metrics[1:]],
                  "first_loss": metrics[0]["loss"], "state": validation["state"]}
        self.results[name] = result
        write_json(self.work / "progress.json", self.results)
        print(f"LONG passed {name}: allocated={result['peak_allocated_bytes']/2**30:.2f}GiB "
              f"reserved={result['peak_reserved_bytes']/2**30:.2f}GiB "
              f"seconds={result['median_update_seconds']:.2f} loss={result['first_loss']:.6f}", flush=True)
        return directory, result


def test_real_memory_ablations_and_checkpoint_policies():
    cases = Cases("ablations")
    configurations = [
        ("baseline", {}, "layer"),
        ("ce", dict(loss_chunk_tokens=512), "layer"),
        ("attention", dict(loss_chunk_tokens=512, exact_length=True, attention_policy="flash_only"), "layer"),
        ("mlp", dict(loss_chunk_tokens=512, mlp_chunk_tokens=512, exact_length=True, attention_policy="flash_only"), "layer"),
        ("offload", dict(loss_chunk_tokens=512, mlp_chunk_tokens=512, exact_length=True,
                         attention_policy="flash_only", activation_offload="checkpoint_boundary"), "layer"),
        ("mlp-selective", dict(loss_chunk_tokens=512, mlp_chunk_tokens=512, exact_length=True,
                               attention_policy="flash_only"), "selective_mlp"),
        ("layer-group", dict(loss_chunk_tokens=512, mlp_chunk_tokens=512, exact_length=True,
                             attention_policy="flash_only"), "layer_group"),
    ]
    for name, memory, policy in configurations:
        cases.execute(name, dict(memory, profile_first_update=True), policy=policy)
    for name in ["ce", "attention", "mlp", "offload", "mlp-selective", "layer-group"]:
        # Cross-kernel comparisons are reported separately; CE is compared with auto,
        # MLP/offload/checkpoint schedules with the same Flash reference.
        reference = cases.results["baseline" if name == "ce" else "attention"]
        actual = cases.results[name]
        assert abs(actual["first_loss"] - reference["first_loss"]) <= (
            TOLERANCES["loss_atol"] + TOLERANCES["loss_rtol"] * abs(reference["first_loss"]))
    write_json(cases.work / "report.json", dict(status="passed", results=cases.results))
    write_json(ROOT / "temp/lora-long-context-20261110/latest-ablations.json",
               dict(report=str(cases.work / "report.json")))


def test_real_progressive_capacity_resume_and_independent_adapter_reload():
    import torch
    cases = Cases("capacity")
    # Fixed before running: layer checkpoint plus chunked CE/MLP, no CPU offload.
    memory = dict(loss_chunk_tokens=512, mlp_chunk_tokens=512, exact_length=True,
                  attention_policy="flash_only", profile_first_update=False)
    lengths = [2048, 4096, 8192, 16384, 24576, 32768]
    if os.environ.get("DYCON_LORA_EXTENDED_CAPACITY") == "1":
        lengths += [65536, 131072]
    write_json(cases.work / "length-spec.json", dict(lengths=lengths, memory=memory))
    completed = []
    for length in lengths:
        name = f"{length}"
        paused, result = cases.execute(name, memory, length=length, updates=3, stop=2)
        checkpoint = paused / "checkpoints/step-000002"
        resumed, resumed_result = cases.execute(name + "-resume", memory, length=length, updates=3,
                                                start="resume", checkpoint=checkpoint)
        warm, _ = cases.execute(name + "-reload", memory, length=length, updates=1,
                                 start="warm_start", checkpoint=resumed / "adapter")
        assert read_json(warm / "initial-adapter.json") == read_json(resumed / "final-adapter.json")
        torch.testing.assert_close(torch.load(warm / "initial-probe.pt", weights_only=True),
                                   torch.load(resumed / "final-probe.pt", weights_only=True),
                                   atol=TOLERANCES["logits_atol"], rtol=TOLERANCES["logits_rtol"])
        assert resumed_result["state"]["global_step"] == 3
        completed.append(length)
        peak = max(result["peak_reserved_bytes"], resumed_result["peak_reserved_bytes"])
        total = result["metrics"][0]["total_bytes"]
        if total - peak < 15 * 2**30:
            print(f"LONG safety stop after {length}: reserved margin below 15GiB", flush=True)
            break
    assert 24576 in completed, f"First target 24K was not accepted; passed lengths: {completed}"
    write_json(cases.work / "report.json", dict(status="passed", lengths=completed, results=cases.results))
    write_json(ROOT / "temp/lora-long-context-20261110/latest-capacity.json",
               dict(report=str(cases.work / "report.json")))
