import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_evaluation.swebench_audit import collect
from benchmark_evaluation.swebench_receipts import get_test_command
from benchmark_evaluation.swebench_verifier import classify
from runtime.containers.factory import ContainerFactory
from test_swebench_audit import fixture


@pytest.mark.parametrize("suffix", [b"\r\n", b"\r", b"\x1b[31mred\x1b[0m\r\n", b"\xff\r\n"])
def test_raw_bytes_digest_and_real_tamper(tmp_path, monkeypatch, suffix):
    output, logs, env, task, outcome, rid, eid, _ = fixture(tmp_path, monkeypatch)
    path = logs / "test_output.txt"
    raw = path.read_bytes() + suffix
    path.write_bytes(raw)
    receipt = {"instance_id": task["instance_id"], "evaluation_task_id": eid,
               "output_sha256": hashlib.sha256(raw).hexdigest(), "returncode": 0}
    (env / "eval-execution.json").write_text(json.dumps(receipt))
    document = collect(output, eid, rid, task, outcome, gold=True)
    assert not document["collection_errors"]
    assert document["execution"]["output_verified"] is True
    path.write_bytes(raw + b"x")
    document = collect(output, eid, rid, task, outcome, gold=True)
    assert "Execution output digest mismatch" in document["collection_errors"]
    assert classify(document)["effective_resolved"] is None


def test_sympy_assignments_preserve_command_identity():
    command = "PYTHONWARNINGS='ignore::DeprecationWarning' bin/test sympy/core/tests/test_x.py"
    script = ": '>>>>> Start Test Output'\n" + command + "\n: '>>>>> End Test Output'\n"
    details = get_test_command(script)
    assert details["runner"] == "sympy"
    assert details["command"] == command


@pytest.mark.parametrize("member_name,kind", [("root/a.json", "regular"), ("root/a.json", "symlink"), ("root/sub/a.json", "regular")])
def test_archive_collection_rejects_links_and_nested_files(tmp_path, member_name, kind):
    data = b'{"schema_version":1}'
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        member = tarfile.TarInfo(member_name)
        member.size = len(data)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            member.size = 0
        tar.addfile(member, io.BytesIO(data))
    container = SimpleNamespace(raw=SimpleNamespace(get_archive=lambda _: ([archive.getvalue()], {})),
                                context=SimpleNamespace(output_dir=tmp_path))
    if kind == "regular" and member_name == "root/a.json":
        result = ContainerFactory.collect_eval_receipts(None, container, {"token": "a" * 32})
        assert result[0]["sha256"] == hashlib.sha256(data).hexdigest()
    else:
        with pytest.raises(ValueError):
            ContainerFactory.collect_eval_receipts(None, container, {"token": "a" * 32})
