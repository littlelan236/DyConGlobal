from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import import_fixture, task

from baseline.catalog import Catalog
from core.config import load_snapshot, resolve, snapshot


def test_empty_and_duplicate_config(cfg):
    assert Catalog(cfg["project_root"]).verify()["rollouts"] == 0
    p = Path(cfg["project_root"]) / "config.yaml"
    p.write_text(p.read_text() + "\nschema_version: 1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        resolve(p)


def test_import_reuse_and_patch_integrity(cfg):
    rid = import_fixture(cfg)
    assert import_fixture(cfg) == rid
    catalog = Catalog(cfg["project_root"])
    assert catalog.verify()["rollouts"] == 1
    row = catalog.read()["rollouts"][rid]
    (Path(cfg["project_root"]) / row["patch_path"]).write_text("changed")
    with pytest.raises(ValueError, match="Changed patch"):
        catalog.verify()


def test_partition_conflict_is_atomic(cfg):
    catalog = Catalog(cfg["project_root"])
    catalog.register_tasks([task()])
    revision = catalog.read()["catalog_revision"]
    with pytest.raises(ValueError):
        catalog.register_tasks([task(partition="test")])
    assert catalog.read()["catalog_revision"] == revision


def test_evaluation_outcome_binding(cfg):
    rid = import_fixture(cfg, resolved=False)
    catalog = Catalog(cfg["project_root"])
    with pytest.raises(ValueError, match="disagrees"):
        with catalog.transaction() as data:
            data["rollouts"][rid]["resolved"] = True


def test_concurrent_catalog_updates(cfg):
    catalog = Catalog(cfg["project_root"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: catalog.register_tasks([task(f"task-{i}")]), range(12)))
    assert len(catalog.read()["tasks"]) == 12


def test_frozen_snapshot_integrity(cfg):
    root = Path(cfg["project_root"])
    snapshot(cfg, "run", {"tasks": {}})
    assert load_snapshot(root, "run")["config_hash"] == cfg["config_hash"]
    from core.io import read_json, write_json

    p = root / "runs/run/resolved-config.json"
    value = read_json(p)
    value["generation"]["seed"] = 12
    write_json(p, value)
    with pytest.raises(ValueError, match="modified"):
        load_snapshot(root, "run")


def test_imported_harness_evidence_must_match_patch(cfg):
    from baseline.importers.evaluation import import_evaluation
    from core.io import sha256, write_json

    rid = import_fixture(cfg)
    root = Path(cfg["project_root"])
    report = root / "old-report.json"
    write_json(report, {"t-1": {"resolved": True}})
    patch = root / "evaluated.diff"
    patch.write_text("patch")
    spec = {
        "evaluation_id": "legacy",
        "report": str(report),
        "report_sha256": sha256(report),
        "bindings": {rid: {"evaluated_patch": str(patch), "outcome_path": ["t-1"]}},
    }
    import_evaluation(root, spec)
    assert Catalog(root).read()["rollouts"][rid]["resolved"] is True
    spec["evaluation_id"] = "mismatch"
    patch.write_text("different")
    with pytest.raises(ValueError, match="different patch"):
        import_evaluation(root, spec)
