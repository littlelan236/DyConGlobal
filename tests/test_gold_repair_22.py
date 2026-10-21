import pytest
import requests

from benchmark_evaluation.metadata import make_cached_get
from core.io import hash_json, read_json, write_json


def response(code, content=b"numpy<2\n"):
    out = requests.Response()
    out.status_code = code
    out._content = content
    out.url = "https://example.test/"
    return out


def test_metadata_retries_reset_and_uses_exact_cached_content(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark_evaluation.metadata.time.sleep", lambda _: None)
    calls = []
    url = "https://raw.githubusercontent.com/owner/repo/revision/requirements.txt"

    def fetch(candidate, **kwargs):
        calls.append(candidate)
        if len(calls) < 3:
            raise requests.ConnectionError("Connection reset by peer")
        return response(200)

    get = make_cached_get(fetch, tmp_path, 1)
    assert get(url).content == b"numpy<2\n"
    assert len(calls) == 3
    assert "cdn.jsdelivr.net" in calls[1]
    assert get(url).content == b"numpy<2\n"
    assert len(calls) == 3
    assert not list(tmp_path.glob(".metadata-*"))


def test_real_404_reaches_official_path_fallback(tmp_path):
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return response(404)

    url = "https://raw.githubusercontent.com/owner/repo/revision/missing.txt"
    assert make_cached_get(fetch, tmp_path, 1)(url).status_code == 404
    assert len(calls) == 1
    assert not (tmp_path / (hash_json(url) + ".txt")).exists()


def test_exhausted_metadata_failure_never_fabricates_empty_requirements(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark_evaluation.metadata.time.sleep", lambda _: None)
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        raise requests.ConnectionError("reset")

    with pytest.raises(requests.ConnectionError):
        make_cached_get(fetch, tmp_path, 1)("https://raw.githubusercontent.com/o/r/c/requirements.txt")
    assert len(calls) == 6
    assert list(tmp_path.iterdir()) == []


def test_audit_uses_frozen_config_without_resolving_changed_yaml(tmp_path, monkeypatch):
    from runtime.scripts import gold_image_audit as audit

    inputs = tmp_path / "instances.json"
    write_json(inputs, [{"instance_id": "task", "nas_image": {"manifest_sha256": "hash"}}])
    cfg = {"project_root": str(tmp_path), "generation": {"future_setting": None}}
    monkeypatch.setattr(audit, "load_cfg", lambda *a: pytest.fail("Live config was reloaded"))

    class Store:
        def __init__(self, *args):
            self.image_id = "sha256:" + "1" * 64
            self.record = {"compressed_bytes": 1}

        def verify(self, **kwargs):
            return []

    monkeypatch.setattr("runtime.images.nas.NasImageStore", Store)

    class Prepared(Exception):
        pass

    def factory(effective):
        assert effective["generation"]["future_setting"] is None
        assert effective["evaluation_run_id"] == "frozen-test"
        raise Prepared

    monkeypatch.setattr(audit, "ContainerFactory", factory)
    with pytest.raises(Prepared):
        audit.audit(tmp_path, inputs, "frozen-test", 1, cfg=cfg)
    assert "evaluation_run_id" not in cfg
    assert (
        read_json(tmp_path / "state/gold-image-audit/frozen-test/audit-config.json")["generation"]
        == cfg["generation"]
    )


@pytest.mark.parametrize(
    "iid,repo",
    [
        ("pydicom__pydicom-1194", "pydicom/pydicom"),
        ("pydicom__pydicom-1555", "pydicom/pydicom"),
        ("pyvista__pyvista-3747", "pyvista/pyvista"),
        ("pyvista__pyvista-4808", "pyvista/pyvista"),
    ],
)
def test_missing_resource_task_network_is_shared_by_all_container_roles(cfg, iid, repo):
    from runtime.containers.factory import ContainerFactory

    factory = ContainerFactory(cfg, client=object())
    factory.lock_images = lambda rows: None
    factory.images[iid] = {"image_id": "sha256:" + "1" * 64}
    row = {"instance_id": iid, "repo": repo}
    assert factory.spec(row)["network_mode"] == "bridge"
    assert factory.policy_for(row).environment["network_fallback"] is False
