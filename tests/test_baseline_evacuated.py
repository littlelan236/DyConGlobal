from application.execution.evidence import RunEvidence
"""The production packages must work with an empty baseline directory."""
import ast
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", [
    "rollout.runner", "rollout.agent", "rollout.model", "rollout.parallel", "rollout.artifacts",
    "benchmark_evaluation.harness", "benchmark_evaluation.metadata", "benchmark_evaluation.selection",
    "catalog.benchmark", "benchmark_evaluation.import_evidence", "application.execution.harness",
    "runtime.serving.startup", "runtime.serving.parallel_validation",
])
def test_relocated_module_imports_without_baseline(module):
    importlib.import_module(module)


def test_production_source_has_no_baseline_imports():
    root = Path(__file__).resolve().parents[1]
    packages = ["core", "application", "catalog", "rollout", "benchmark_evaluation", "runtime",
                "hidden", "enrichment", "difficulty", "action", "diagnostics"]
    for package in packages:
        for source in (root / package).rglob("*.py"):
            if any(p in source.parts for p in ["datasets", "outputs", "checkpoints", "cache", "reports"]):
                continue
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                assert not any(n == "baseline" or n.startswith("baseline.") for n in names), source


def test_capture_uses_canonical_identity_after_path_relocation(cfg):
    from conftest import import_fixture
    from catalog.store import Catalog
    from application.execution.evidence import verified_trajectory

    rid = import_fixture(cfg)
    row = RunEvidence(cfg["project_root"], "source").read()["rollouts"][rid]
    proof = {"producer_run_id": "source", "rollout_id": rid, "trajectory_sha256": row["trajectory_sha256"],
             "trajectory_path": "/retired/baseline/trajectory.json"}
    actual = verified_trajectory(cfg["project_root"], proof)
    assert actual == Path(cfg["project_root"]) / row["trajectory_path"]
    proof["trajectory_sha256"] = "changed"
    with pytest.raises(ValueError, match="hashes disagree"):
        verified_trajectory(cfg["project_root"], proof)


def test_startup_probe_is_hash_bound_without_catalog_registration(tmp_path):
    from application.execution.evidence import verified_trajectory
    from core.io import sha256, write_json
    path = tmp_path / "runs/diagnostic/startup/trajectory.json"
    write_json(path, {"messages": []})
    proof = {"rollout_id": "startup-diagnostic", "trajectory_path": str(path),
             "trajectory_sha256": sha256(path)}
    assert verified_trajectory(tmp_path, proof) == path
    path.write_text("changed")
    with pytest.raises(ValueError, match="diagnostic trajectory changed"):
        verified_trajectory(tmp_path, proof)
