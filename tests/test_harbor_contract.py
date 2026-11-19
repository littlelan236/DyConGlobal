import copy
import math
import re
import socket
from types import SimpleNamespace

import pytest

from application.execution.evidence import RunEvidence
from application.execution.subject import verify_subject
from benchmark_evaluation.harbor import trial_outcome
from catalog.harbor import REVISIONS
from core.backends import capabilities
from core.deadline import Deadline
from core.io import hash_json, sha256, write_json
from rollout.artifacts import import_rollout
from runtime.agents.model_socket import UnixModelChannel
from runtime.backends.harbor.adapter import worker_request
from runtime.backends.harbor.factory import HarborFactory
from runtime.backends.registry import resolve


def harbor_task(benchmark="terminalbench_2_1"):
    return {"benchmark": benchmark, "revision": REVISIONS[benchmark], "instance_id": "canary", "partition": "train",
            "instance": {"instance_id": "canary", "task_path": "/test-only/package", "package_sha256": hash_json([]),
                "files": [], "instruction": "Do the task", "images": {"agent": "example/task:pinned"},
                "metadata": {"schema_version": "1.1" if benchmark == "terminalbench_2_1" else "1.4",
                             "environment": {"cpus": 2}, "agent": {"network_mode": "no-network", "timeout_sec": 900}}}}


@pytest.mark.parametrize("benchmark,kind,mode", [
    ("swebench_pro_v2", "patch", "replay"), ("terminalbench_2_1", "live_state", "live")])
def test_new_benchmark_bindings_never_fall_into_native_swe(benchmark, kind, mode):
    cfg = {"benchmark": {"backend": benchmark}, "agent": {"backend": "openhands"}}
    binding = resolve(cfg, {"task": harbor_task(benchmark)})
    assert (binding.backend_family, binding.subject_kind, binding.evaluation_mode) == ("harbor", kind, mode)
    assert capabilities(cfg)["action_analysis"] is False
    with pytest.raises(ValueError, match="combination"):
        resolve({**cfg, "agent": {"backend": "minisweagent"}})


@pytest.mark.parametrize("rewards,exception,expected", [
    ({"reward": 0}, None, False), ({"reward": 1}, None, True),
    ({"reward": 0.5}, None, False), ({}, None, None), (None, None, None),
    ({"reward": float("nan")}, None, None), ({"reward": float("inf")}, None, None),
    ({"reward": "1"}, None, None), ({"reward": True}, None, None),
    ({"reward": -1}, None, None), ({"reward": 1}, {"exception_type": "VerifierError"}, None),
])
def test_terminal_reward_failure_and_unknown_remain_distinct(rewards, exception, expected):
    result = {"verifier_result": {"rewards": rewards}, "exception_info": exception}
    before = copy.deepcopy(result)
    outcome = trial_outcome(result, task=harbor_task(), attempt_id="a")
    assert outcome["resolved"] is expected
    if rewards and any(isinstance(v, float) and not math.isfinite(v) for v in rewards.values()):
        assert "invalid_numeric" in outcome["raw_reward"]["reward"]
    else:
        assert outcome["raw_reward"] == rewards
    assert result.keys() == before.keys()
    if expected is False:
        assert outcome["status"] == "completed"


def test_timeout_does_not_erase_a_reward_valid_under_upstream_policy():
    outcome = trial_outcome({"verifier_result": {"rewards": {"reward": 1}},
        "exception_info": {"exception_type": "AgentTimeoutError"}, "reward_valid": True},
        task=harbor_task(), attempt_id="a")
    assert outcome["resolved"] is True
    assert outcome["exception"]["exception_type"] == "AgentTimeoutError"


def test_identical_attempt_ids_from_different_samples_are_cleaned_separately():
    seen = []
    factory = HarborFactory.__new__(HarborFactory)
    factory.run_id = "run"
    factory.owner_id = "factory"
    factory.client = SimpleNamespace(
        containers=SimpleNamespace(list=lambda **kw: seen.append(kw["filters"]["label"]) or []),
        networks=SimpleNamespace(list=lambda **kw: []))
    factory.cleanup_attempt("rep-000-attempt-000", "sample-one")
    factory.cleanup_attempt("rep-000-attempt-000", "sample-two")
    assert "dycon.sample=sample-one" in seen[0]
    assert "dycon.sample=sample-two" in seen[2]


def test_cleanup_accepts_disappearing_container_but_reports_residual_network(tmp_path):
    from docker.errors import APIError, NotFound
    from core.io import read_json

    def removed_elsewhere(**_):
        raise NotFound("gone", response=SimpleNamespace(status_code=404))

    def active_endpoints():
        raise APIError("still in use", response=SimpleNamespace(status_code=409))

    container = SimpleNamespace(id="container", remove=removed_elsewhere)
    network = SimpleNamespace(id="network", remove=active_endpoints)
    containers = iter([[container], []])
    factory = HarborFactory.__new__(HarborFactory)
    factory.run_id, factory.owner_id = "run", "factory"
    factory.client = SimpleNamespace(
        containers=SimpleNamespace(list=lambda **kw: next(containers)),
        networks=SimpleNamespace(list=lambda **kw: [network]))
    with pytest.raises(RuntimeError, match="owned resource cleanup failed"):
        factory.cleanup_attempt("physical", "sample", tmp_path, trial_name="trial")
    receipt = read_json(tmp_path / "cleanup.json")
    assert not receipt["complete"]
    assert receipt["residual"] == []
    assert receipt["networks_residual"] == ["network"]
    assert receipt["errors"] == [{"kind": "network", "id": "network", "status_code": 409}]


def test_worker_trial_identity_is_bounded_and_scoped_to_resource_owner(cfg, tmp_path):
    cfg.update(execution_context={"profile": {"dependency_snapshot": "/offline"}}, harbor={"cpu_quota": "strict"})
    factory = SimpleNamespace(cpu_supported=True, owner_id="owner", images={})
    binding = {"sample_key": "sample-one", "attempt_id": "rep-000-attempt-000"}

    def request(run="RUN/" * 100, sample=binding, operation="execute"):
        row = worker_request(cfg, run, factory, harbor_task(), sample, tmp_path, operation)
        assert row["trial_name"] == row["environment"]["trial_name"]
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", row["trial_name"])
        return row["trial_name"]

    original = request()
    assert request() == original
    others = {request(run="other"), request(sample={**binding, "sample_key": "sample-two"}),
              request(sample={**binding, "attempt_id": "rep-000-attempt-001"}), request(operation="replay")}
    factory.owner_id = "another-owner"
    others.add(request())
    assert len(others) == 5 and original not in others


def test_worker_request_preserves_task_timeout_and_explicit_cpu_deviation(cfg, tmp_path):
    cfg.update(execution_context={"profile": {"dependency_snapshot": "/offline"}},
               harbor={"cpu_quota": "strict"})
    factory = SimpleNamespace(cpu_supported=False, owner_id="owner", images={})
    binding = {"sample_key": "sample", "attempt_id": "physical"}
    with pytest.raises(ValueError, match="CPU quota"):
        worker_request(cfg, "run", factory, harbor_task(), binding, tmp_path)
    cfg["harbor"]["cpu_quota"] = "unenforced_with_receipt"
    request = worker_request(cfg, "run", factory, harbor_task(), binding, tmp_path)
    assert request["driver"]["agent_seconds"] == 840
    assert request["environment"]["resources"]["applied_cpus"] is None
    assert request["environment"]["resources"]["memory_enforcement"] == "limit"
    assert not request["disable_verifier"]


def test_terminal_task_python_sources_are_bound_to_worker_request(cfg, tmp_path):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    manifest = {"schema_version": 1, "publication_id": "test-wheels"}
    manifest["manifest_sha256"] = hash_json(manifest)
    write_json(wheelhouse / "manifest.json", manifest)
    cfg.update(execution_context={"profile": {"dependency_snapshot": "/offline"}},
        harbor={"cpu_quota": "strict", "task_python_index": "https://mirrors.aliyun.com/pypi/simple/",
                "task_apt_mirror": "https://mirrors.aliyun.com/ubuntu/"},
        environment={**cfg["environment"], "wheelhouse": str(wheelhouse)})
    factory = SimpleNamespace(cpu_supported=True, owner_id="owner", images={})
    binding = {"sample_key": "sample", "attempt_id": "physical"}
    request = worker_request(cfg, "run", factory, harbor_task(), binding, tmp_path)
    environment = request["environment"]
    assert environment["task_python_index"] == "https://mirrors.aliyun.com/pypi/simple/"
    assert environment["task_apt_mirror"] == "https://mirrors.aliyun.com/ubuntu/"
    assert environment["task_wheelhouse"] == str(wheelhouse)
    assert environment["task_wheelhouse_manifest_sha256"] == sha256(wheelhouse / "manifest.json")


def test_task_dependency_overlay_uses_mirror_and_read_only_wheels(tmp_path):
    from runtime.backends.harbor.task_dependencies import task_dependency_overlay

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    manifest = wheelhouse / "manifest.json"
    manifest.write_text("{}")
    dycon = {"task_python_index": "https://mirrors.aliyun.com/pypi/simple/",
             "task_wheelhouse": str(wheelhouse),
             "task_wheelhouse_manifest_sha256": sha256(manifest)}
    mounts, environment = task_dependency_overlay(dycon, "public")
    assert mounts == [{"type": "bind", "source": str(wheelhouse.resolve()),
                       "target": "/opt/dycon-task-wheels", "read_only": True}]
    assert environment == {
        "PIP_FIND_LINKS": "/opt/dycon-task-wheels",
        "UV_FIND_LINKS": "/opt/dycon-task-wheels",
        "PIP_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/",
        "UV_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/",
        "UV_DEFAULT_INDEX": "https://mirrors.aliyun.com/pypi/simple/",
    }
    _, offline_environment = task_dependency_overlay(dycon, "none")
    assert "UV_DEFAULT_INDEX" not in offline_environment
    assert offline_environment["UV_FIND_LINKS"] == "/opt/dycon-task-wheels"


def test_task_dependency_overlay_rejects_changed_wheelhouse(tmp_path):
    from runtime.backends.harbor.task_dependencies import task_dependency_overlay

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    (wheelhouse / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        task_dependency_overlay({"task_wheelhouse": str(wheelhouse),
                                 "task_wheelhouse_manifest_sha256": "wrong"}, "public")


def test_apt_mirror_command_preserves_update_policy_and_only_warms_when_needed():
    from runtime.backends.harbor.task_dependencies import apt_mirror_setup_command

    mirror = "https://mirrors.aliyun.com/ubuntu/"
    rewritten = apt_mirror_setup_command(mirror, warmup=False)
    assert "archive|security" in rewritten
    assert "Signed-By" not in rewritten
    assert "apt-get update" not in rewritten
    warmed = apt_mirror_setup_command(mirror, warmup=True)
    assert "apt-get update -o Acquire::Retries=2" in warmed


def test_apt_mirror_warmup_detects_install_before_update(tmp_path):
    from runtime.backends.harbor.task_dependencies import apt_mirror_warmup_required

    task_path = tmp_path / "task"
    script = task_path / "solution/solve.sh"
    script.parent.mkdir(parents=True)
    script.write_text("apt-get update\napt-get install -y make\n")
    assert not apt_mirror_warmup_required({"instance": {"task_path": str(task_path)}})
    script.write_text("apt-get install -y make\napt-get update\n")
    assert apt_mirror_warmup_required({"instance": {"task_path": str(task_path)}})


def test_task_python_index_requires_https_without_credentials():
    from core.config import Harbor

    with pytest.raises(ValueError, match="HTTPS URL"):
        Harbor(runtime_ref="runtime.json", image_store="images-v1",
               task_python_index="http://mirrors.aliyun.com/pypi/simple/")
    with pytest.raises(ValueError, match="HTTPS URL"):
        Harbor(runtime_ref="runtime.json", image_store="images-v1",
               task_python_index="https://user:secret@mirror.example/simple/")
    with pytest.raises(ValueError, match="HTTPS URL"):
        Harbor(runtime_ref="runtime.json", image_store="images-v1",
               task_apt_mirror="http://mirrors.aliyun.com/ubuntu/")


def live_subject(root, *, wrong_sample=False):
    task = harbor_task()
    binding = {"run_id": "terminal", "task_key": hash_json({k: task[k] for k in ("benchmark", "revision", "instance_id")}),
               "sample_key": "wrong" if wrong_sample else "sample", "attempt_id": "physical"}
    directory = root / "proof"
    write_json(directory / "verifier.json", {"rewards": {"reward": 0}, "trial_id": "actual-trial"})
    write_json(directory / "subject.json", {"schema_version": 1, "kind": "live_state", "binding": binding,
        "trial_id": "actual-trial", "task_package_sha256": task["instance"]["package_sha256"], "re_evaluation": False,
        "outputs": [{"path": "verifier.json", "sha256": sha256(directory / "verifier.json")} ]})
    write_json(root / "canonical.json", {"instance_id": task["instance_id"], "messages": [],
                                         "info": {"exit_status": "AgentCompleted"}})
    return {"task": task, "source": str(root / "canonical.json"), "run_id": "terminal", "attempt_id": "physical",
            "model_fingerprint": "model", "subject": str(directory / "subject.json"),
            "subject_sha256": sha256(directory / "subject.json"), "sample": {"sample_key": "sample"}}


def test_terminal_import_requires_live_state_and_detects_mismatched_attempt(tmp_path):
    spec = live_subject(tmp_path)
    patch = {k: v for k, v in spec.items() if k not in {"subject", "subject_sha256"}}
    with pytest.raises(ValueError, match="live trial proof"):
        import_rollout(tmp_path, patch)
    bad = live_subject(tmp_path, wrong_sample=True)
    with pytest.raises(ValueError, match="another attempt"):
        import_rollout(tmp_path, bad)


def test_terminal_live_proof_is_auditable_but_not_a_replay_snapshot(tmp_path):
    spec = live_subject(tmp_path)
    rid = import_rollout(tmp_path, spec)
    evidence = RunEvidence(tmp_path, "terminal")
    row = evidence.read()["rollouts"][rid]
    assert row["evaluation_subject"] == {"kind": "live_state", "schema_version": 1, "re_evaluation": False}
    assert "patch_path" not in row
    assert evidence.verify()["verified"]
    path = tmp_path / row["subject_path"]
    (path.parent / "verifier.json").write_text("tampered")
    with pytest.raises(ValueError, match="Changed subject output"):
        verify_subject(tmp_path, row)


def test_unix_model_channel_reaches_only_authenticated_bridge_without_generation(cfg, tmp_path):
    from rollout.adapters.openhands_bridge import OpenHandsBridge
    profile = {"bridge_listen": "127.0.0.1", "bridge_advertise": "127.0.0.1", "request_roles": "unqualified"}
    with OpenHandsBridge(cfg, {"attempt_id": "a"}, profile, tmp_path, Deadline(10)) as bridge:
        path = tmp_path / "channel/model.sock"
        with UnixModelChannel(path, "127.0.0.1", bridge.server.server_port):
            for token, expected in (("wrong", b"401"), (bridge.secret, b"200")):
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(path))
                    client.sendall(("GET /v1/dycon-health HTTP/1.0\r\nHost: model\r\nAuthorization: Bearer " +
                                    token + "\r\nContent-Length: 0\r\n\r\n").encode())
                    assert expected in client.recv(4096).split(b"\r\n")[0]
            assert bridge.records == []
        assert not path.exists()


@pytest.mark.parametrize("warnings,expected", [(None, True), ([], True),
    (["No cpu cfs quota support"], False)])
def test_factory_accepts_null_docker_warnings(cfg, monkeypatch, warnings, expected):
    import docker
    import runtime.images.nas
    client = SimpleNamespace(info=lambda: {"Warnings": warnings})
    monkeypatch.setattr(docker, "DockerClient", lambda **kw: client)
    monkeypatch.setattr(runtime.images.nas, "ImageLeaseManager", lambda *args: None)
    factory = HarborFactory(cfg, "warnings-test")
    assert factory.cpu_supported is expected


@pytest.mark.parametrize("controllers,expected", [("memory pids", False), ("cpu memory pids", True)])
def test_cpu_quota_checks_rootless_user_delegation(monkeypatch, controllers, expected):
    from pathlib import Path
    from runtime.backends.harbor.factory import cpu_quota_supported
    monkeypatch.setattr(Path, "is_file", lambda p: True)
    monkeypatch.setattr(Path, "read_text", lambda p: controllers)
    info = {"Warnings": None, "CpuCfsQuota": True, "CpuCfsPeriod": True,
            "CgroupVersion": "2", "SecurityOptions": ["name=rootless"]}
    assert cpu_quota_supported(info, "unix:///run/user/1023/docker.sock") is expected
    assert cpu_quota_supported(info, "tcp://remote:2375") is True


def test_cpu_quota_respects_explicit_daemon_capability():
    from runtime.backends.harbor.factory import cpu_quota_supported
    assert not cpu_quota_supported({"CpuCfsQuota": False}, "unix:///socket")
