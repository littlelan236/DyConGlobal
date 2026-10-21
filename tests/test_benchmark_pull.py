"""Selection and registry transfer contracts; no live network/Docker required."""

import hashlib
import json

import pytest

from catalog.benchmark import _balanced_selection, _task_record
from runtime.images.registry import persist_bytes


def test_selection_excludes_verified_existing_missing_and_unsolved():
    rows = []
    results = {str(i): {"resolved": [], "no_generation": [], "no_logs": []} for i in range(6)}
    for repo in range(3):
        for bucket, solved in [("hard", 1), ("medium", 3), ("easy", 5)]:
            for index in range(4):
                iid = f"org__repo{repo}-{bucket}{index}"
                rows.append({"instance_id": iid, "repo": f"org/repo{repo}", "patch": "diff",
                             "test_patch": "test diff", "FAIL_TO_PASS": ["a"], "PASS_TO_PASS": ["b"]})
                for i in range(solved):
                    results[str(i)]["resolved"].append(iid)
    blocked = rows[0]["instance_id"]
    existing = rows[1]["instance_id"]
    missing = rows[2]["instance_id"]
    results["5"]["no_logs"].append(missing)
    chosen, summary = _balanced_selection(rows, [existing], results, 9, [blocked])
    assert summary["bucket_counts"] == {"easy": 3, "medium": 3, "hard": 3}
    assert not {blocked, existing, missing} & set(summary["instance_ids"])
    again, _ = _balanced_selection(list(reversed(rows)), [existing], results, 9, [blocked])
    assert [x["row"]["instance_id"] for x in chosen] == [x["row"]["instance_id"] for x in again]


def test_task_row_retains_official_difficulty_and_normalizes_test_lists():
    item = {"row": {"instance_id": "org__repo-1", "repo": "org/repo", "base_commit": "abc",
                    "image": "org/image:latest", "FAIL_TO_PASS": ["a"], "PASS_TO_PASS": ["b"],
                    "difficulty": ""}, "bucket": "hard", "resolution_score": 1,
            "resolution_evidence": {"test": True}}
    task = _task_record(item, {"manifest_sha256": "abc", "image_id": "sha256:0"}, "cohort", "rev", "0"*64)
    assert task["instance"]["difficulty"] == ""
    assert task["instance"]["dycon_selection"]["difficulty_bucket"] == "hard"
    assert json.loads(task["instance"]["FAIL_TO_PASS"]) == ["a"]


def test_registry_metadata_rejects_sha_and_corrupt_cache(tmp_path):
    raw = b'{"schemaVersion":2}'
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="SHA mismatch"):
        persist_bytes(tmp_path, raw, "sha256:" + "0" * 64)
    assert persist_bytes(tmp_path, raw, digest) == digest
    (tmp_path / digest[7:]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        persist_bytes(tmp_path, raw, digest)
