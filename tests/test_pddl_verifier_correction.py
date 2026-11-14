"""Regression tests run in the pinned PDDL validator environment."""

import copy
import importlib.util
from pathlib import Path

import pytest

from catalog.skillsbench import package_manifest, verify_package
from catalog.skillsbench_verifier import MEMBER, PATCH, publish_variant
from core.io import hash_json, read_json, sha256, write_json


@pytest.fixture
def verifier():
    pytest.importorskip("unified_planning")
    spec = importlib.util.spec_from_file_location("corrected_airport", PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text,expected", [
    ("drive(truck1, depot1, market1)", "(drive truck1 depot1 market1)\n"),
    ("(drive truck1 depot1 market1)", "(drive truck1 depot1 market1)\n"),
    ("finish()", "(finish)\n"),
    (" ; note\n\nfinish() ; cost = 1", "(finish)\n"),
    ("", ""),
])
def test_documented_and_standard_syntax(verifier, text, expected):
    assert verifier.normalize_plan_text(text) == expected


@pytest.mark.parametrize("text", ["a(x) b(y)", "a(x,)", "a((x))", "a(x); b(y)\nbad",
    "__import__('os').system('id')", "(a x) trailing", "1: (a x) [2]", "a(x y)", "(a x,y)"])
def test_malformed_lines_are_rejected(verifier, text):
    with pytest.raises(ValueError):
        verifier.normalize_plan_text(text)


def test_real_airport_semantics_and_negative_controls(verifier, tmp_path):
    project = Path(__file__).parents[1]
    environment = project / "benchmarks/skillsbench/tasks/pddl-airport-planning/environment"
    # Real historical plan text, fixture supplied under this task's temp directory.
    for suffix in ("01", "02"):
        original = (project / "temp/pddl-verifier-fix-20261110" / f"task{suffix}.txt").read_text()
        plan = tmp_path / f"task{suffix}.txt"
        domain, problem = (environment / "airport" / f"domain{suffix}.pddl",
                           environment / "airport" / f"task{suffix}.pddl")
        for text in (original, verifier.normalize_plan_text(original)):
            plan.write_text(text)
            assert verifier.evaluate_plan(domain, problem, plan)["status"] == "VALID"
        # Truncating a real plan leaves its goal unsatisfied.
        plan.write_text("\n".join(original.splitlines()[:-1]))
        assert verifier.evaluate_plan(domain, problem, plan)["status"] == "INVALID"
        # Reversing a real plan makes its first action inapplicable.
        plan.write_text("\n".join(reversed(original.splitlines())))
        assert verifier.evaluate_plan(domain, problem, plan)["status"] == "INVALID"
        plan.write_text("")
        assert verifier.validate_plan(domain, problem, plan) is False
        plan.write_text("unknown_action(unknown_object)")
        assert verifier.evaluate_plan(domain, problem, plan)["valid"] is False


def source(tmp_path):
    project = Path(__file__).parents[1]
    upstream = project / "benchmarks/skillsbench/tasks/pddl-airport-planning"
    task_root = tmp_path / "benchmarks/skillsbench/tasks/pddl-airport-planning"
    (task_root / "verifier").mkdir(parents=True)
    (task_root / MEMBER).write_bytes((upstream / MEMBER).read_bytes())
    (task_root / "task.md").write_bytes((upstream / "task.md").read_bytes())
    files = package_manifest(task_root)
    task = {"benchmark": "skillsbench", "revision": "a" * 40,
        "instance_id": "pddl-airport-planning", "partition": "train", "instance": {
            "instance_id": "pddl-airport-planning", "task_path": str(task_root),
            "files": files, "package_sha256": hash_json(files), "images": [], "metadata": {}}}
    manifest = tmp_path / "source.json"
    write_json(manifest, {"benchmark": "skillsbench", "tasks": {"key": task}})
    return manifest, task


def test_publication_preserves_source_and_rejects_tampering(tmp_path):
    manifest, original = source(tmp_path)
    before = sha256(manifest)
    result = publish_variant({"project_root": str(tmp_path)}, "fixed", manifest)
    corrected = read_json(tmp_path / result["snapshot_ref"])["tasks"]["key"]
    assert sha256(manifest) == before
    verify_package(original)
    verify_package(corrected)
    with pytest.raises(FileExistsError):
        publish_variant({"project_root": str(tmp_path)}, "fixed", manifest)
    path = Path(corrected["instance"]["task_path"]) / MEMBER
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError):
        verify_package(corrected)


def test_unapproved_metadata_and_source_edits_fail(tmp_path):
    manifest, original = source(tmp_path)
    result = publish_variant({"project_root": str(tmp_path)}, "fixed", manifest)
    corrected = read_json(tmp_path / result["snapshot_ref"])["tasks"]["key"]
    tampered = copy.deepcopy(corrected)
    tampered["instance"]["metadata"]["new"] = "unapproved"
    with pytest.raises(ValueError):
        verify_package(tampered)
    path = Path(original["instance"]["task_path"]) / MEMBER
    path.write_text(path.read_text() + "\n# changed upstream\n")
    with pytest.raises(ValueError):
        publish_variant({"project_root": str(tmp_path)}, "refused", manifest)
    assert not (tmp_path / "catalog/task-variants/refused").exists()


def test_correction_is_authorized_by_the_existing_package_resolver(tmp_path):
    from catalog.skillsbench_index import task_package_path
    manifest, _ = source(tmp_path)
    result = publish_variant({"project_root": str(tmp_path)}, "fixed", manifest)
    corrected = read_json(tmp_path / result["snapshot_ref"])["tasks"]["key"]
    cfg = {"project_root": str(tmp_path), "skillsbench_images": {"source_root": "benchmarks/skillsbench"}}
    assert task_package_path(cfg, corrected) == Path(corrected["instance"]["task_path"])


def test_semantic_cli_uses_frozen_run_without_root_config(monkeypatch, tmp_path, capsys):
    import core.cli
    import runtime.agents.pddl_revalidation
    monkeypatch.setattr(core.cli, "resolve", lambda *a, **k: pytest.fail("mutable config loaded"))
    calls = []
    monkeypatch.setattr(runtime.agents.pddl_revalidation, "revalidate", lambda *a, **k:
                        calls.append((a, k)) or {"complete": True})
    core.cli.main(["--config", str(tmp_path / "config.yaml"), "skillsbench", "revalidate-pddl", "run",
                   "--snapshot", "snapshot.json", "--samples", "samples.json", "--output", "analysis/new",
                   "--validator-python", ".venvs/validator/bin/python"])
    assert len(calls) == 1 and calls[0][0][1] == "run"
    assert '"complete": true' in capsys.readouterr().out
