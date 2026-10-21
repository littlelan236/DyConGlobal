from application.execution.evidence import RunEvidence
"""Action categories, trajectory-only workflow and immutable evidence contracts."""

import json
from pathlib import Path

import pytest
from conftest import import_fixture

from action.analyzers.taxonomy import Classifier, shell_literal
from action.analyzers.trajectories import check_analysis, classify_trajectories
from catalog.store import Catalog
from core.io import read_json, sha256, write_json


@pytest.mark.parametrize(
    "command,label",
    [
        ("grep -R foo src", "INSPECT"),
        ("git diff file.py", "INSPECT"),
        ("git diff -- file.py > patch.txt", "DELIVER"),
        ("cat > /tmp/test.py <<'PY'\nprint(1)\nPY\npython /tmp/test.py", "EDIT"),
        ("pip install pytest && pytest -q", "ENV"),
        ("rm /tmp/x && pytest -q", "RUN"),
        ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt", "DELIVER"),
        ("echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' > x", "EDIT"),
        ("rm -f /tmp/x", "CLEAN"),
        ("python -c \"open('x.py','w').write('x')\"", "EDIT"),
        ("python -m pytest x.py 2>&1 | head -5", "RUN"),
        ("unknown_tool foo", "OTHER"),
        ("find . -delete", "OTHER"),
        ("curl -X DELETE https://example.invalid", "OTHER"),
    ],
)
def test_categories(command, label):
    assert Classifier().classify(command)["label"] == label


def test_shell_quoting_and_advisory_helper_source():
    assert shell_literal("'a\\nb'") == "a\\nb"
    assert shell_literal('"a\\nb"') == "a\\nb"
    assert shell_literal('"a\\"b"') == 'a"b'
    c = Classifier()
    c.classify("cat > /tmp/edit.py <<'PY'\nopen('x.py', 'w').write('x')\nPY")
    assert c.classify("python /tmp/edit.py")["label"] == "EDIT"
    other = Classifier().classify("python /tmp/edit.py")
    assert "script_source_unavailable" in other["uncertainties"]


def _source(cfg):
    rid = import_fixture(cfg)
    root = Path(cfg["project_root"])
    row = RunEvidence(root, "source").read()["rollouts"][rid]
    path = root / row["trajectory_path"]
    data = read_json(path)
    messages = [m for m in data["messages"] if m["role"] == "assistant"]
    for m, cmd in zip(messages, ["cat source.py", "python -m pytest tests"]):
        m["tool_calls"] = [{"function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]
        # Must not hydrate unavailable token ledgers for action classification.
        m["extra"] = {"response": {"ledger_ref": "not-present", "ledger_root": "not-present"}}
    write_json(path, data)
    with RunEvidence(root, "source").transaction() as manifest:
        manifest["rollouts"][rid]["trajectory_sha256"] = sha256(path)
    return rid, path


def test_direct_cli_no_enrichment_immutable_and_tamper_detection(cfg, capsys):
    from core.cli import main

    rid, path = _source(cfg)
    root = Path(cfg["project_root"])
    manifest_before = sha256(root / "catalog/manifest.json")
    main(["--config", cfg["config_source"], "action-classify", "report", "--source-run", "source", "--rollout-id", rid])
    assert not (root / "enrichment").exists()
    assert sha256(root / "catalog/manifest.json") == manifest_before
    out = root / "action/outputs/report"
    rows = [json.loads(s) for s in (out / "labels.jsonl").read_text().splitlines()]
    assert [r["label"] for r in rows] == ["INSPECT", "RUN"]
    assert [r["phase_bin"] for r in rows] == [0, 4]
    assert check_analysis(cfg, "report")["verified"]
    with pytest.raises(FileExistsError):
        classify_trajectories({**cfg, "source_run_id": "source"}, "report", [rid])
    (out / "labels.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        check_analysis(cfg, "report")


def test_selection_sha_malformed_unsupported_and_no_partial_publish(cfg):
    rid, path = _source(cfg)
    root = Path(cfg["project_root"])
    listing = root / "scope.json"
    write_json(listing, [{"rollout_id": rid, "trajectory_sha256": "wrong"}])
    with pytest.raises(ValueError, match="SHA mismatch"):
        classify_trajectories({**cfg, "source_run_id": "source"}, "bad-binding", rollout_list=listing)
    assert not (root / "action/outputs/bad-binding").exists()
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        classify_trajectories({**cfg, "source_run_id": "source"}, "bad-bytes", [rid])
    assert not (root / "action/outputs/bad-bytes").exists()


def test_unsupported_tool_shapes_are_explicit():
    from action.analyzers.taxonomy import classify_action_evidence

    for calls in [
        [],
        [{"function": {"name": "bash", "arguments": "bad json"}}],
        [{"function": {"name": "other"}}],
    ]:
        result = classify_action_evidence({"tool_calls": calls}, Classifier())
        assert result["label"] == "OTHER"
        assert "unsupported_tool_shape" in result["uncertainties"]
