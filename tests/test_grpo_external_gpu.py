"""Opt-in real Qwen3.5-9B validation through the public CLI, never a launcher."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest
import torch
import yaml

from core.io import read_json, sha256

pytestmark = pytest.mark.skipif(os.environ.get("DYCON_GRPO_EXTERNAL_GPU") != "1",
    reason="requires explicit real Qwen3.5-9B/GPU acceptance opt-in")


def test_public_cli_external_anchor_two_updates_pause_resume(tmp_path, monkeypatch):
    from RL.config import Config
    from tests import test_grpo_external_reference as fixtures
    from tests.test_grpo_lifecycle import assert_tree_close

    root = Path(__file__).parents[1]
    evidence = Path(os.environ.get("DYCON_GRPO_EVIDENCE", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    storage = Path(os.environ.get("DYCON_GRPO_ARTIFACT_ROOT", str(evidence / "artifacts")))
    prefix = "grpo-binding-" + uuid.uuid4().hex[:10]
    config = yaml.safe_load((root / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    config["run"]["mode"] = "smoke"
    config["runtime"].update(output_root=str(storage), gpu=int(os.environ.get("DYCON_GRPO_GPU", "0")))
    config["model"]["path"] = os.environ.get("DYCON_GRPO_MODEL", config["model"]["path"])
    timings = {}

    def cli(label, value):
        value = copy.deepcopy(value); value["run"]["attempt_id"] = prefix + "-" + label
        filename = evidence / (label + ".yaml"); filename.write_text(yaml.safe_dump(value))
        started = time.perf_counter()
        with (evidence / (label + ".log")).open("w") as log:
            result = subprocess.run([str(root / ".venvs/rl/bin/dycon"), "lora-train",
                "--training-config", str(filename)], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                timeout=300, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        assert result.returncode == 0, (evidence / (label + ".log")).read_text()[-6000:]
        timings[label] = time.perf_counter() - started
        path = storage / "RL/checkpoints" / value["run"]["attempt_id"]
        with (evidence / (label + "-verify.log")).open("w") as log:
            verified = subprocess.run([str(root / ".venvs/rl/bin/dycon"), "artifact", "verify",
                value["run"]["attempt_id"], "--scope", "lora-attempt", "--storage-root", str(storage)],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        assert verified.returncode == 0, (evidence / (label + "-verify.log")).read_text()[-4000:]
        return path

    baseline = cli("model-forward", config)
    base = read_json(baseline / "model-identity.json")
    monkeypatch.setattr(fixtures, "BASE", base)
    from RL.grpo.reference_artifact import ReferenceReader
    from RL.grpo.prepared import resolve
    reference = ReferenceReader(resolve(read_json(baseline / "grpo-contract.json")["grpo"]["reference_artifact"], "rl-reference"))
    vectors = {gid:reference[gid] for gid in reference.prepared.ids}
    source_root = evidence / "source"
    source_id = prefix + "-raw-fixture"
    cfg = Config.model_validate(config)
    def scoring_identity(manifest, rows, path):
        from core.io import hash_json
        scoring = manifest["scoring_identity"]
        scoring["collector_dtype"] = "bfloat16"
        scoring["runtime"] = {"backend": "transformers", "logprobs_mode": "raw_logprobs",
                              "evidence_attempt": str(baseline)}
        scoring["implementation"] = read_json(baseline / "environment.json")["source"]["fingerprint"]
        for row in rows: row["scoring_identity_sha256"] = hash_json(scoring)
    source = fixtures.make_source(source_root, cfg, vectors=vectors, name=source_id, mutate=scoring_identity)
    config["grpo"]["reference_cache"] = {"source": "probability_artifact",
        "artifact": {"scope": "rl-replay", "root": str(source_root), "id": source_id},
        "dtype_compatibility": "same_dtype", "initial_diagnostic_branches": 2}
    external = cli("external", config)
    paused_config = copy.deepcopy(config); paused_config["validation"]["stop_after_update"] = 1
    drift_retries = []
    offline_source = source.with_name(source.name + "-offline-during-resume")
    for retry in range(1):
        suffix = "" if retry == 0 else "-retry" + str(retry)
        if offline_source.exists(): offline_source.rename(source)
        paused = cli("paused" + suffix, paused_config)
        # Remove the entire source from its configured location, retaining evidence.
        source.rename(offline_source)
        resumed_config = copy.deepcopy(config)
        resumed_config["run"]["start"] = {"mode": "resume", "checkpoint": str(paused / "checkpoints/step-000001")}
        try:
            resumed = cli("resumed" + suffix, resumed_config)
            break
        except AssertionError:
            # Source drift is no longer a compatibility gate or a reason to retry.
            raise

    expected = read_json(baseline / "final-adapter.json")
    assert read_json(external / "final-adapter.json") == expected
    assert read_json(resumed / "final-adapter.json") == expected
    for path in (baseline, external, resumed):
        state = read_json(path / "attempt.json")["final_state"]
        assert state["global_step"] == 2 and state["group_cursor"] == 5
    first = torch.load(baseline / "checkpoints/step-000002/training-state.pt", weights_only=True)
    last = torch.load(resumed / "checkpoints/step-000002/training-state.pt", weights_only=True)
    for field in ("optimizer", "scheduler", "rng"):
        assert_tree_close(first[field], last[field])
    metric_keys = ("loss", "policy_loss", "k3_penalty_mean", "clip_fraction", "ratio_mean",
        "ratio_min", "ratio_max", "ratio_quantiles", "positive_logprob_change", "negative_logprob_change",
        "gradient_norm", "learning_rate", "group_cursor", "consumed_groups", "consumed_branches",
        "consumed_tokens", "group_count", "branch_count", "token_count", "adapter_gradient_coverage")
    def metrics_equal(a, b):
        if isinstance(a, dict): return a.keys() == b.keys() and all(metrics_equal(a[k], b[k]) for k in a)
        if isinstance(a, list): return len(a) == len(b) and all(metrics_equal(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
        return a == b
    baseline_updates = read_json(baseline / "grpo-report.json")["updates"]
    for path in (external, resumed):
        updates = read_json(path / "grpo-report.json")["updates"]
        assert len(updates) == len(baseline_updates)
        for a, b in zip(baseline_updates, updates):
            for key in metric_keys: assert metrics_equal(a[key], b[key]), (path, key)
    anchor_hash = read_json(paused / "grpo-contract.json")["grpo"]["reference_logprob_cache_sha256"]
    assert read_json(resumed / "grpo-contract.json")["grpo"]["reference_logprob_cache_sha256"] == anchor_hash
    summary = {"scope": "synthetic Qwen3.5-9B same-dtype external-cache acceptance",
        "attempts": {"baseline": str(baseline), "external": str(external), "paused": str(paused), "resumed": str(resumed)},
        "adapter_equal": True, "optimizer_scheduler_rng_equal": True,
        "objective_gradient_metrics_equal": True,
        "objective_metric_tolerance": {"atol": 1e-7, "rtol": 1e-7},
        "source_unavailable_during_resume": True, "anchor_sha256": anchor_hash,
        "source_drift_retries": drift_retries,
        "cli_seconds": timings, "reference_preparation": {
            "baseline": read_json(baseline / "grpo/reference-preparation.json"),
            "external": read_json(external / "grpo/reference-preparation.json")},
        "initial_current_diagnostic": read_json(external / "grpo/initial-current-diagnostic.json"),
        "first_update": {"baseline": read_json(baseline / "grpo-report.json")["updates"][0],
                         "external": read_json(external / "grpo-report.json")["updates"][0]}}
    (evidence / "gpu-acceptance.json").write_text(json.dumps(summary, indent=2))
