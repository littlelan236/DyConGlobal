from pathlib import Path

import pytest
from conftest import import_fixture, task

from baseline.catalog import Catalog
from baseline.importers.legacy import import_rollout
from baseline.scripts.clean_interrupted import apply, classify
from core.io import hash_json, read_json, write_json


@pytest.mark.parametrize(
    "termination",
    [
        "LimitsExceeded",
        "ContextWindowExceededError",
        "Submitted",
        "RepeatedFormatError",
        "ContainerCorrupted",
    ],
)
def test_explicit_terminal_evidence_prevents_removal(termination):
    assert classify(termination, ["Interrupted"], ["stopped_by_user"], orphaned=True) is None


def test_stale_sidecar_cannot_override_attempt_limit():
    assert classify("", ["LimitsExceeded"], ["paused"], orphaned=True) is None
    assert classify("", [None], ["rollout"], orphaned=False) is None


def test_cleanup_removes_payloads_and_preserves_terminal_rollout(cfg):
    root = Path(cfg["project_root"])
    keep = import_fixture(cfg, resolved=True)
    raw = root / "interrupted.traj.json"
    write_json(raw, {"instance_id": "t-2", "info": {"exit_status": ""}, "messages": []})
    remove = import_rollout(
        root,
        {
            "source": str(raw),
            "task": task("t-2"),
            "run_id": "old",
            "attempt_id": "a0",
            "purposes": ["training"],
            "model_fingerprint": "fixture-model",
        },
    )
    catalog = Catalog(root)
    before = catalog.read()
    removed_row = before["rollouts"][remove]
    out = root / "baseline/provenance/cleanup"
    plan = out / "plan.json"
    write_json(
        plan,
        {
            "cleanup_id": "cleanup",
            "catalog_hash": hash_json(before),
            "remove": [{"rollout_id": remove}],
            "reason_counts": {"explicit_interrupted_attempt": 1},
        },
    )
    result = apply(root, plan)
    assert result["removed"] == 1 and result["remaining"] == 1
    assert catalog.read()["rollouts"][keep] == before["rollouts"][keep]
    assert not (root / removed_row["trajectory_path"]).exists()
    assert not (root / removed_row["patch_path"]).exists()
    assert raw.exists()
    assert apply(root, plan) == result
    assert read_json(out / "receipt.json")["verified"]["verified"]


def test_forged_cleanup_cannot_delete_submitted_rollout(cfg):
    root = Path(cfg["project_root"])
    keep = import_fixture(cfg)
    before = Catalog(root).read()
    plan = root / "baseline/provenance/unsafe/plan.json"
    write_json(
        plan,
        {
            "cleanup_id": "unsafe",
            "catalog_hash": hash_json(before),
            "remove": [{"rollout_id": keep}],
            "reason_counts": {},
        },
    )
    with pytest.raises(ValueError, match="Only unfinalized"):
        apply(root, plan)
    assert Catalog(root).read() == before
    assert (root / before["rollouts"][keep]["trajectory_path"]).exists()
