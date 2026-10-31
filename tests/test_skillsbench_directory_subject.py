import asyncio
import shlex
from pathlib import Path
from types import SimpleNamespace

from core.io import read_json, sha256, write_json
from application.execution.subject import export_workspace, verify_subject


class FakeEnvironment:
    def __init__(self):
        self.archives = {}

    async def exec(self, command, **kwargs):
        if command.startswith("test ! -L"):
            kind = "directory" if "test -d" in command else "file"
            return SimpleNamespace(return_code=0, stdout=kind + "\n755\n")
        if command.startswith("tar -czf"):
            parts = shlex.split(command)
            self.archives[parts[2]] = b"archive-bytes"
            return SimpleNamespace(return_code=0, stdout="")
        if command.startswith("rm -f"):
            self.archives.pop(shlex.split(command)[-1], None)
            return SimpleNamespace(return_code=0, stdout="")
        raise AssertionError(command)

    async def download_file(self, source, destination):
        Path(destination).write_bytes(self.archives[source])


def test_workspace_subject_can_bind_and_verify_directory_archive(tmp_path):
    env = FakeEnvironment()
    subject = asyncio.run(export_workspace(env, ["/workspace/project"], tmp_path / "subject",
        {"run_id": "r"}, "a" * 64, {"image_id": "sha256:" + "b" * 64}))
    data = read_json(subject)
    assert data["complete"] is True
    assert data["outputs"][0]["kind"] == "directory_archive"
    assert data["outputs"][0]["source_path"] == "/workspace/project"
    trajectory = tmp_path / "trajectory.json"
    write_json(trajectory, {"messages": []})
    row = {"evaluation_subject": {"kind": "workspace"}, "subject_path": str(subject),
           "subject_sha256": sha256(subject), "trajectory_path": str(trajectory),
           "trajectory_sha256": sha256(trajectory)}
    assert verify_subject(tmp_path, row)
