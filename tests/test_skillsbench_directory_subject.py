import asyncio
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from application.execution.subject import export_workspace, verify_subject
from core.io import read_json, sha256, write_json


def test_organize_submission_resolves_complete_common_parent_before_export(tmp_path):
    from runtime.agents.skillsbench import resolve_submission_outputs

    names = ['LLM', 'trapped_ion_and_qc', 'black_hole', 'DNA', 'music_history']
    root = tmp_path / 'sandbox-root'
    # A partial group at /root must not steal the complete official location.
    (root / 'LLM').mkdir(parents=True)
    for name in names:
        (root / 'papers' / name).mkdir(parents=True)
        (root / 'papers' / name / 'paper.pdf').write_bytes(b'paper')

    class Environment(FakeEnvironment):
        async def exec(self, command, **kwargs):
            if command.startswith('python3 -c '):
                script = shlex.split(command)[2].replace("Path('/root/papers')", f'Path({str(root / "papers")!r})')
                script = script.replace("Path('/root')", f'Path({str(root)!r})')
                result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                        text=True, timeout=5)
                return SimpleNamespace(return_code=result.returncode,
                    stdout=result.stdout.replace(str(root), '/root'), stderr=result.stderr)
            return await super().exec(command, **kwargs)

    env = Environment()
    declared = ['/root/' + n for n in names]
    outputs, digest = asyncio.run(resolve_submission_outputs(
        env, 'organize-messy-files', declared, '/root', tmp_path, {'attempt_id': 'a'}))
    assert outputs == ['/root/papers/' + n for n in names]
    evidence = tmp_path / 'diagnostics/submission-output-resolution.json'
    assert digest == sha256(evidence)
    assert read_json(evidence)['discovered_root'] == '/root/papers'
    subject = asyncio.run(export_workspace(env, outputs, tmp_path / 'subject', {}, 'a' * 64, {}))
    assert read_json(subject)['complete']
    assert len(read_json(subject)['outputs']) == 5


def test_organize_missing_group_keeps_missing_contract_and_diagnostic(tmp_path):
    import json

    from runtime.agents.skillsbench import resolve_submission_outputs

    class Environment:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout=json.dumps({'root': None, 'observed': []}))

    declared = ['/root/' + n for n in ['LLM', 'trapped_ion_and_qc', 'black_hole', 'DNA', 'music_history']]
    outputs, _ = asyncio.run(resolve_submission_outputs(
        Environment(), 'organize-messy-files', declared, '/root', tmp_path, {}))
    assert outputs == declared
    assert read_json(tmp_path / 'diagnostics/submission-output-resolution.json')['status'] == 'no_complete_group'


def test_submission_resolution_preserves_explicit_profile_paths(tmp_path):
    from runtime.agents.skillsbench import resolve_submission_outputs

    class Environment:
        async def exec(self, *args, **kwargs):
            raise AssertionError('Explicit output declarations require no discovery')

    declared = ['/root/papers']
    assert asyncio.run(resolve_submission_outputs(
        Environment(), 'organize-messy-files', declared, '/root', tmp_path, {})) == (declared, None)


def test_submission_discovery_timeout_does_not_block_official_verification(tmp_path):
    from runtime.agents.skillsbench import resolve_submission_outputs

    class Environment:
        async def exec(self, *args, **kwargs):
            raise TimeoutError('injected bounded discovery timeout')

    declared = ['/root/' + n for n in ['LLM', 'trapped_ion_and_qc', 'black_hole', 'DNA', 'music_history']]
    outputs, _ = asyncio.run(resolve_submission_outputs(
        Environment(), 'organize-messy-files', declared, '/root', tmp_path, {}))
    assert outputs == declared
    receipt = read_json(tmp_path / 'diagnostics/submission-output-resolution.json')
    assert receipt['status'] == 'discovery_failed'
    assert receipt['error_type'] == 'TimeoutError'


class FakeEnvironment:
    def __init__(self):
        self.archives = {}

    async def exec(self, command, **kwargs):
        if command.startswith("if test -L"):
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
    assert data["output_inventory"] == [{"required_path": "/workspace/project", "status": "exported",
        "output_index": 0, "artifact_path": data["outputs"][0]["path"]}]
    trajectory = tmp_path / "trajectory.json"
    write_json(trajectory, {"messages": []})
    row = {"evaluation_subject": {"kind": "workspace"}, "subject_path": str(subject),
           "subject_sha256": sha256(subject), "trajectory_path": str(trajectory),
           "trajectory_sha256": sha256(trajectory)}
    assert verify_subject(tmp_path, row)


def test_workspace_subject_distinguishes_missing_and_rejected_outputs(tmp_path):
    class IncompleteEnvironment(FakeEnvironment):
        async def exec(self, command, **kwargs):
            if "/workspace/missing" in command:
                return SimpleNamespace(return_code=1, stdout="missing\n", stderr="")
            if "/workspace/rejected.txt" in command:
                return SimpleNamespace(return_code=0, stdout="file\n644\n", stderr="")
            return await super().exec(command, **kwargs)

        async def download_file(self, source, destination):
            if source == "/workspace/rejected.txt":
                raise OSError("injected export failure")
            await super().download_file(source, destination)

    env = IncompleteEnvironment()
    subject = asyncio.run(export_workspace(env,
        ["/workspace/missing", "/workspace/rejected.txt", "/workspace/../unsafe"],
        tmp_path / "subject", {"run_id": "r"}, "a" * 64, {}))
    data = read_json(subject)
    assert data["complete"] is False
    assert data["missing_outputs"] == ["/workspace/missing"]
    assert data["rejected_outputs"] == ["/workspace/rejected.txt", "/workspace/../unsafe"]
    assert [row["status"] for row in data["output_inventory"]] == ["missing", "rejected", "rejected"]


def test_verifier_command_evidence_has_safe_phase_and_target_host(tmp_path):
    import json
    from types import SimpleNamespace

    from runtime.agents.skillsbench import _run_verifier_with_evidence

    class Sandbox:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout="ok", stderr="token=do-not-save")

    class Verifier:
        def __init__(self):
            self._sandbox = Sandbox()
            self._rollout_paths = SimpleNamespace(test_stdout_path=tmp_path / "test-stdout.txt")

        async def verify(self):
            await self._sandbox.exec("curl https://user:pass@example.test/pkg.whl", timeout_sec=8)
            return SimpleNamespace(rewards={"reward": 1})

    owner = SimpleNamespace(output=tmp_path, ownership={"attempt_id": "a"})
    result = asyncio.run(_run_verifier_with_evidence(Verifier(), owner))
    assert result.rewards == {"reward": 1}
    receipt = json.loads((tmp_path / "diagnostics/verifier-execution.json").read_text())
    command = receipt["commands"][0]
    assert command["phase"] == "download"
    assert command["target_hosts"] == ["example.test"]
    assert "user:pass" not in command["command_summary"]
    assert "do-not-save" not in json.dumps(command)


def test_missing_bootstrap_tools_do_not_become_a_valid_zero_reward(tmp_path):
    import pytest

    from runtime.agents.skillsbench import _run_verifier_with_evidence

    log = tmp_path / 'test-stdout.txt'
    log.write_text('/verifier/test.sh: line 11: /root/.local/bin/env: No such file or directory\n'
                   '/verifier/test.sh: line 17: uvx: command not found\n')

    class Verifier:
        _rollout_paths = SimpleNamespace(test_stdout_path=log)

        async def verify(self):
            return SimpleNamespace(rewards={'reward': 0})

    owner = SimpleNamespace(output=tmp_path, ownership={'attempt_id': 'a'})
    with pytest.raises(RuntimeError, match='bootstrap failed'):
        asyncio.run(_run_verifier_with_evidence(Verifier(), owner))
    receipt = read_json(tmp_path / 'diagnostics/verifier-execution.json')
    assert receipt['phase'] == 'failed'
    assert receipt['attempts'][-1]['failure_reason'] == 'verifier_bootstrap_tools_missing'
