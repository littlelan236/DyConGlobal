import shutil
from pathlib import Path

import pytest
from conftest import task

from baseline.catalog import Catalog
from baseline.scripts.migrate_tact import apply, prepare, report_runs
from core.io import read_json, sha256, write_json


@pytest.fixture
def legacy(tmp_path):
    src = tmp_path / "legacy"
    run = "legacy-eval-run"
    instances = src / "instances.json"
    write_json(instances, [task("t-1")["instance"], task("t-2")["instance"]])
    config = {"model": {"path": "/models/tiny"}, "benchmark": {"instances": str(instances)}}
    write_json(src / f"runs/{run}/resolved-config.json", config)
    write_json(src / "baseline/config/resolved-config.json", config)
    paths = []
    for iid, attempt, patch, status in [
        ("t-1", "a0", "same-patch", "Submitted"),
        ("t-1", "a1", "same-patch", "Submitted"),
        ("t-2", "a0", "", "LimitsExceeded"),
    ]:
        path = src / f"runs/{run}/tasks/{iid}/attempts/{attempt}/{iid}.traj.json"
        write_json(
            path,
            {
                "instance_id": iid,
                "info": {"exit_status": status, "submission": patch},
                "messages": [{"role": "assistant", "content": attempt}],
            },
        )
        paths.append(path)
        write_json(src / f"runs/{run}/tasks/{iid}/result.json", {"attempt_id": attempt})
    canonical = src / "baseline/traj/t-1.traj.json"
    canonical.parent.mkdir(parents=True)
    shutil.copyfile(paths[1], canonical)
    write_json(
        src / "baseline/manifest.json",
        {
            "trajectories": [
                {"instance_id": "t-1", "sha256": sha256(canonical), "source": str(paths[1].relative_to(src))}
            ]
        },
    )
    predictions = [
        {"instance_id": "t-1", "model_patch": "same-patch"},
        {"instance_id": "t-2", "model_patch": ""},
    ]
    summary = {
        "gold": False,
        "outcomes": {
            "t-1": {"resolved": True, "completed": True},
            "t-2": {"resolved": False, "completed": False, "outcome": "MissingPatch"},
        },
    }
    for folder in [src / "baseline/harness", src / f"runs/{run}/evaluation/candidate"]:
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "summary.json", summary)
    report = src / "report.md"
    report.write_text(
        f"| run | all | selected | Submitted | resolved |\n| canonical-baseline | 1 | 1 | 1 | 1 |\n| {run} | 3 | 2 | 1 | 1 |\n"
    )
    return src, report, paths


def test_migration_deduplicates_without_spreading_same_patch_labels(cfg, legacy):
    src, report, paths = legacy
    root = Path(cfg["project_root"])
    plan = prepare(root, src, report, "migration")
    result = apply(root, plan)
    assert result["unique_trajectories"] == 3 and result["total_source_files"] == 4
    assert result["status_counts"] == {"resolved": 1, "submitted": 1, "unsubmitted": 1}
    data = Catalog(root).read()
    assert all(t["partition"] == "train" for t in data["tasks"].values())
    rows = {r["trajectory_sha256"]: r for r in data["rollouts"].values()}
    assert rows[sha256(paths[0])]["resolved"] is None
    assert rows[sha256(paths[0])]["evaluation_status"] == "not_run"
    success = rows[sha256(paths[1])]
    assert success["resolved"] is True and success["run_id"] == "legacy-eval-run"
    assert len(success["provenance"]["source_occurrences"]) == 2
    assert rows[sha256(paths[2])]["resolved"] is None
    assert rows[sha256(paths[2])]["evaluation_status"] == "infra_error"
    assert all(r["purposes"] == ["training"] for r in rows.values())
    assert apply(root, plan) == result
    assert Catalog(root).read()["catalog_revision"] == data["catalog_revision"]


def test_migration_rejects_changed_trajectory_before_catalog_write(cfg, legacy):
    src, report, paths = legacy
    root = Path(cfg["project_root"])
    plan = prepare(root, src, report, "changed")
    paths[0].write_text("{}")
    with pytest.raises(ValueError, match="Source trajectory changed"):
        apply(root, plan)
    assert Catalog(root).read()["rollouts"] == {}


def test_migration_rejects_modified_evaluated_patch(cfg, legacy):
    src, report, _ = legacy
    root = Path(cfg["project_root"])
    plan = prepare(root, src, report, "changed-patch")
    evidence = read_json(plan)["evaluations"][0]
    patch = root / next(iter(evidence["bindings"].values()))["evaluated_patch"]
    patch.write_text("different")
    with pytest.raises(ValueError, match="differs from frozen predictions"):
        apply(root, plan)
    assert Catalog(root).read()["rollouts"] == {}


def test_migration_rejects_conflicting_completed_evidence(cfg, legacy):
    src, report, _ = legacy
    f = src / "runs/legacy-eval-run/evaluation/candidate/summary.json"
    summary = read_json(f)
    summary["outcomes"]["t-1"]["resolved"] = False
    write_json(f, summary)
    with pytest.raises(ValueError, match="Conflicting completed results"):
        prepare(cfg["project_root"], src, report, "conflict")
    assert Catalog(cfg["project_root"]).read()["rollouts"] == {}


def test_report_requires_explicit_run_scope(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("| unknown | 4 | 2 | 1 | 1 |\n")
    with pytest.raises(ValueError, match="per-run table"):
        report_runs(report)
