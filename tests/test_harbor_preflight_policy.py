"""Harbor long-check policy preserves shallow identity and attempt validation."""
import contextlib
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from catalog.harbor import verify_package
from catalog.skillsbench import package_manifest
from core.io import hash_json, read_json, sha256, write_json
from core.preflight import SKIP_REASON, long_checks_enabled
from runtime.backends.harbor import adapter as harbor
from runtime.backends.harbor.dependencies import HARBOR_REVISION, HARBOR_VERSION, verify_runtime
from runtime.backends.harbor.factory import HarborFactory
from tests.test_harbor_contract import harbor_task


@pytest.mark.parametrize("benchmark", ["skillsbench", "terminalbench_2_1", "swebench_pro_v2"])
def test_switch_default_and_explicit_policy(benchmark):
    cfg = {"benchmark": {"backend": benchmark}, "agent": {"backend": "openhands"}}
    assert long_checks_enabled(cfg)
    cfg["execution"] = {"preflight": {"long_checks": False}}
    assert not long_checks_enabled(cfg)


def test_other_backend_is_unchanged():
    assert long_checks_enabled({"execution": {"preflight": {"long_checks": False}}})


def package(tmp_path):
    task = harbor_task()
    root = tmp_path / "task"
    root.mkdir()
    (root / "task.toml").write_text("AAAA")
    task["instance"].update(task_path=str(root), files=package_manifest(root))
    task["instance"]["package_sha256"] = hash_json(task["instance"]["files"])
    return task, root


def test_fast_package_does_not_hash_bytes_but_attempt_default_does(tmp_path):
    task, root = package(tmp_path)
    (root / "task.toml").write_text("BBBB")
    verify_package(task, deep=False)
    with pytest.raises(ValueError, match="package changed"):
        verify_package(task)


@pytest.mark.parametrize("change", ["missing", "mode", "extra", "manifest", "symlink"])
def test_fast_package_keeps_path_manifest_and_mode_checks(tmp_path, change):
    task, root = package(tmp_path)
    f = root / "task.toml"
    if change == "missing":
        f.unlink()
    elif change == "mode":
        f.chmod(0o700)
    elif change == "extra":
        (root / "extra").write_text("new")
    elif change == "manifest":
        task["instance"]["package_sha256"] = "0" * 64
    else:
        f.unlink()
        f.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        verify_package(task, deep=False)


def runtime_publication(root):
    root.mkdir()
    member = root / "wheel.whl"
    member.write_bytes(b"AAAA")
    python = root / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    python.chmod(0o755)
    manifest = {"schema_version": 1, "revision": HARBOR_REVISION, "version": HARBOR_VERSION,
                "files": [{"path": "wheel.whl", "size": 4, "sha256": sha256(member)}],
                "python": {"executable_sha256": sha256(python)}}
    manifest["receipt_sha256"] = hash_json(manifest)
    write_json(root / "manifest.json", manifest)
    return member, python


def test_worker_runtime_fast_and_deep_content_checks(tmp_path):
    root = tmp_path / "runtime"
    member, python = runtime_publication(root)
    member.write_bytes(b"BBBB")
    verify_runtime(root, deep=False)
    with pytest.raises(ValueError, match="dependency changed"):
        verify_runtime(root)
    member.write_bytes(b"AAAA")
    python.write_bytes(b"PYTHON")
    verify_runtime(root, deep=False)
    with pytest.raises(ValueError, match="interpreter changed"):
        verify_runtime(root)


@pytest.mark.parametrize("change", ["missing", "size", "manifest", "python", "escape"])
def test_worker_runtime_fast_keeps_structure_checks(tmp_path, change):
    root = tmp_path / "runtime"
    member, python = runtime_publication(root)
    if change == "missing":
        member.unlink()
    elif change == "size":
        member.write_bytes(b"A")
    elif change == "manifest":
        d = read_json(root / "manifest.json")
        d["version"] = "wrong"
        write_json(root / "manifest.json", d)
    elif change == "python":
        python.unlink()
    else:
        member.unlink()
        member.symlink_to(python)
    with pytest.raises(ValueError):
        verify_runtime(root, deep=False)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("benchmark", ["terminalbench_2_1", "swebench_pro_v2"])
def test_adapter_routes_all_checks_and_records_skip(cfg, monkeypatch, deep, benchmark):
    from runtime.agents import dependencies
    from runtime.backends.harbor import dependencies as worker
    from catalog import harbor as catalog
    from runtime.backends.registry import adapter_for
    from core import model_identity

    cfg = copy.deepcopy(cfg)
    cfg["benchmark"]["backend"] = benchmark
    cfg["agent"]["backend"] = "openhands"
    cfg["execution"]["preflight"] = {"long_checks": deep}
    cfg["harbor"] = {"runtime_ref": "worker/manifest.json", "cpu_quota": "ignore"}
    cfg["execution_context"] = {"profile_sha256": "profile", "profile": {"dependency_snapshot": "deps"}}
    task = harbor_task(benchmark)
    calls = []
    monkeypatch.setattr(catalog, "execution_context", lambda c, ts: cfg["execution_context"])
    monkeypatch.setattr(catalog, "verify_package", lambda t, **kw: calls.append(("task", kw["deep"])))
    monkeypatch.setattr(dependencies, "verify_snapshot", lambda p, **kw: calls.append(("deps", kw["deep"])))
    monkeypatch.setattr(worker, "verify_runtime", lambda p, **kw: calls.append(("worker", kw["deep"])))
    monkeypatch.setattr(model_identity, "verify_model", lambda c, **kw: calls.append(("model", kw["deep"])))
    cfg["model"]["verify_identity"] = True
    factory = SimpleNamespace(cpu_supported=True,
        lock_images=lambda ts, **kw: calls.append(("images", kw["deep"])))

    def probe(context, task, binding, output, *args):
        calls.append(("probe", True))
        write_json(output / "worker-response.json", {})
        write_json(output / "network-probe.json", {"verified": True})
        return {"result": {}}, SimpleNamespace(records=[])

    monkeypatch.setattr(harbor, "run_worker_bridge", probe)
    adapter = adapter_for(cfg, {"task": task})
    context = SimpleNamespace(cfg=cfg, run_id="switch", factory=factory)
    result = adapter.preflight(context, {"key": task})
    assert result.passed, result.blockers
    assert calls[:5] == [(name, deep) for name in ("model", "task", "deps", "worker", "images")]
    assert ("probe", True) in calls if deep else not any(k == "probe" for k, _ in calls)
    receipt = read_json(adapter.run_path(context) / "validation/harbor-preflight.json")
    assert receipt["long_checks"] is deep
    for key in ("model_sha", "task_package_sha", "dependency_snapshot_sha", "worker_runtime_sha",
                "oci_deep_verify", "environment_audit"):
        check = receipt["checks"][key]
        assert check["status"] == ("verified" if deep else "skipped")
        if not deep:
            assert check["reason"] == SKIP_REASON
    assert receipt["checks"]["execution_context"]["status"] == "verified"
    # Fast startup never constructs even an environment probe request.
    assert len(receipt["tasks"]) == (1 if deep else 0)


@pytest.mark.parametrize("deep", [True, False])
def test_factory_routes_depth_even_for_existing_image(cfg, monkeypatch, deep):
    import runtime.images.nas as nas
    calls = []
    class Store:
        image_id = "sha256:image"
        def __init__(self, *args):
            pass
        def verify(self, **kw):
            calls.append(("verify", kw["deep"]))
        def identity(self):
            return {}
    @contextlib.contextmanager
    def lease(store, **kw):
        calls.append(("lease", kw["deep"]))
        yield store.image_id
    monkeypatch.setattr(nas, "NasImageStore", Store)
    factory = HarborFactory.__new__(HarborFactory)
    factory.started = False
    factory.cfg = {"execution_context": {"image_references": {"image": {}}}}
    factory.leases = SimpleNamespace(lease=lease)
    factory.stack = contextlib.ExitStack()
    factory.images = {}
    factory.client = SimpleNamespace(api=SimpleNamespace(tag=lambda *args: None),
        images=SimpleNamespace(get=lambda tag: SimpleNamespace(id="sha256:image")))
    try:
        factory.lock_images([{"images": {"agent": "image"}}], deep=deep)
        assert calls == [("verify", deep), ("lease", deep)]
    finally:
        factory.stack.close()


@pytest.mark.parametrize("failure", ["identity", "cpu"])
def test_fast_mode_still_blocks_identity_or_cpu_mismatch(cfg, monkeypatch, failure):
    from catalog import harbor as catalog
    from runtime.backends.registry import adapter_for
    cfg = copy.deepcopy(cfg)
    cfg["benchmark"]["backend"] = "terminalbench_2_1"
    cfg["agent"]["backend"] = "openhands"
    cfg["model"]["verify_identity"] = False
    cfg["execution"]["preflight"] = {"long_checks": False}
    cfg["harbor"] = {"cpu_quota": "strict"}
    cfg["execution_context"] = {"profile": {}}
    task = harbor_task()
    monkeypatch.setattr(catalog, "execution_context",
                        lambda *args: {} if failure == "identity" else cfg["execution_context"])
    context = SimpleNamespace(cfg=cfg, run_id="blocked", factory=SimpleNamespace(cpu_supported=False))
    result = adapter_for(cfg, {"task": task}).preflight(context, {"task": task})
    assert not result.passed
    assert len(result.blockers) == 1
    assert ("frozen identity" if failure == "identity" else "CPU quota") in result.blockers[0]
    assert not result.receipt["tasks"]
